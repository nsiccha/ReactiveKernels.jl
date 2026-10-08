include("evidence.jl")

# Program generator: StructuralPlan → self-contained `@kernel` program.
#
# Emission order per program: layout transforms (unconstrained →
# constrained) → scalar + derived assignments in topo order → preprocessing
# recipes (data-only, folded by `bound=`) → linear predictors →
# per-response plate likelihoods over distribution-kernel endpoints → prior
# terms → canonical `prior`/`likelihood`/`log_jacobian`/`posterior` nodes.
# Recipes sit after assignments because design/offset blocks may reference
# derived-column locals; folding is input-driven, so position changes
# nothing for data-only recipes. Plates compile to allocation-free loops
# with shared work hoisted; constraining stays hand-rolled (no bijectors).
# The `@kernel` def is evaluated in the dedicated `PPLGeneratedModels` scope
# without binding a name there.


"""
    build_kernel(plan) -> (; spec, layout, program)

Validate, assign layout, emit, and evaluate a self-contained `@kernel`
program for `plan`. `spec` is the `KernelSpec` (callable after `prepare`
with `have=(:unconstrained, data…)`); `layout` is its
[`LayoutTable`](@ref) (R10 read API for the sampler side). `program` is the
generated program the graph evaluates: [`prepare_query`](@ref) accepts any
bound plan that generates the same program, with other row counts and data
values or an equivalent lowering in another gensym-created isolation module,
and refuses one that generates another.

Concurrency: `build_kernel` is concurrency-safe. Independent plans may be
built from concurrent tasks with no caller-side synchronization: a build
binds nothing in a shared module and takes no package lock. Construction
compiles little per-model code, so concurrent builds largely run in
parallel; Julia serializes only the compilation that remains (1.10: all
compilation; 1.12 and 1.13: type inference).

The returned spec closes over build-time
eval'd code. Construction resolves module bindings in a package-owned
latest-world scope, including just-evaluated kernels; subsequent execution
uses its own boundary: `prepare` it and call it through [`prepare_query`](@ref) /
[`prepare_sampler`](@ref) (which carry the `Base.invokelatest` world-age
barrier) or wrap those calls in `Base.invokelatest` yourself.
"""
build_kernel(plan::StructuralPlan) = Base.invokelatest(_build_kernel_latest, plan)

function _build_kernel_latest(plan::StructuralPlan)
    validate_plan(plan)
    isbound(plan) || throw(ContractValidationError(
        "[generator] build_kernel requires a bound plan (bind_data first)"))
    layout = assign_layout(plan)
    def = kernel_expr(plan, layout)
    spec = _eval_kernel_def(def)
    return (; spec, layout, program = _program_identity(def))
end

"""
    kernel_expr(plan, layout; name=:ppl_model) -> Expr

The `@kernel` definition expression (`Expr(:(=), signature, body)`).
Pure (no eval): the generator tests inspect and evaluate it.
"""
kernel_expr(plan::StructuralPlan, layout::LayoutTable; name::Symbol = :ppl_model) =
    Base.invokelatest(_kernel_expr_latest, plan, layout; name)

function _kernel_expr_latest(plan::StructuralPlan, layout::LayoutTable; name::Symbol = :ppl_model)
    validate_plan(plan)
    isbound(plan) || throw(ContractValidationError(
        "[generator] kernel_expr requires a bound plan (bind_data first)"))
    stmts = Expr[]
    # Level gathers (`z[g]` over a `levels(h)` axis) read level-code
    # vectors; collect them from every expression before emitting.
    gathers = Set{Tuple{Symbol,Symbol,Int}}()
    assigns = _assignment_statements(plan; gathers)
    free = _density_selection(plan, n -> n ∉ plan.conditioned)
    priors = _prior_statements(free, layout; gathers, context = plan)
    transforms = Expr[]
    for e in layout.entries
        append!(transforms, transform_statements(e))
    end
    coefs = _coef_reassembly_statements(plan, layout)
    conditioned = _conditioned_value_statements(plan)
    values = Expr[assigns..., preprocessing_recipes(plan)...,
        _scan_reconstruction_statements(plan, layout)...,
        _affine_coefficient_statements(plan, layout)...,
        _predictor_statements(plan)...]
    likelihoods = _likelihood_statements(plan, layout; gathers,
        upstream = Expr[transforms..., coefs..., conditioned..., values...])
    append!(stmts, transforms)
    append!(stmts, coefs)
    append!(stmts, conditioned)
    append!(stmts, _array_level_index_statements(plan, gathers))
    append!(stmts, values)
    append!(stmts, likelihoods)
    append!(stmts, priors)
    push!(stmts, _log_jacobian_statement(plan, layout))
    push!(stmts, :(posterior::Float64 = prior + likelihood + log_jacobian))
    push!(stmts, :(return posterior))
    sig = Expr(:call, name, :(unconstrained::Vector{Float64}),
        (_data_arg(colname, col) for (colname, col) in _ordered_columns(plan))...)
    return Expr(:(=), sig, Expr(:block, stmts...))
end

# A generated definition with its private names (`gensym`s, which differ
# between lowerings) renamed in order of first appearance. References owned
# by a gensym-created lowering module get the same treatment: the module is an
# isolation namespace, not part of the emitted program. Stable named modules
# remain part of the identity. Two bindings generate the same program exactly
# when these are equal.
function _program_identity(def)
    names = Dict{Symbol,Symbol}()
    modules = IdDict{Module,Symbol}()
    rename(x::Symbol) = Base.isgensym(x) ?
        get!(() -> Symbol("##", length(names) + 1), names, x) : x
    rename(x::GlobalRef) = Base.isgensym(nameof(x.mod)) ?
        Expr(:., get!(() -> Symbol("_rkppl_generated_module_", length(modules) + 1),
            modules, x.mod), QuoteNode(x.name)) : x
    rename(x::Expr) = Expr(x.head, Any[rename(a) for a in x.args]...)
    rename(x) = x
    return rename(def)
end

# Conditioning selects the same declaration-density graph as a prior. Its
# constrained value is a read-only input, with no layout entry or Jacobian.
function _density_selection(plan, selected)
    return _with(plan;
        parameters = filter(p -> selected(p.name), plan.parameters),
        array_parameters = filter(p -> selected(p.name), plan.array_parameters),
        vector_parameters = filter(p -> selected(p.name), plan.vector_parameters),
        plate_parameters = filter(p -> selected(p.name), plan.plate_parameters))
end

function _conditioned_value_statements(plan)
    stmts = Expr[]
    for name in sort!(collect(plan.conditioned))
        push!(stmts, :($name = $(_conditioned_input(name))))
        for p in plan.array_parameters
            p.name === name || continue
            nd = length(_array_dims(plan, p))
            if p.family === :lkj_cholesky_stack
                # Only the intrinsic factor width expands; a stack's data axis
                # stays a vector read and reduction in the existing LKJ graph.
                K = _array_dims(plan, p)[1]
                for i in 2:K
                    rhs = nd == 3 ? :($name[$i, $i, :]) : :($name[$i, $i])
                    push!(stmts, :($(_rl_name(name, i, i)) = $rhs))
                end
            elseif nd > 1 && !_is_structured_array(p)
                push!(stmts, :($(_array_flat_name(name, nd)) = vec($name)))
            end
        end
    end
    return stmts
end

# Pack references to constrained parameter values for the affine matmul.
# Only authored terms are expanded; array sizes remain array operations.
function _affine_coefficient_statements(plan::StructuralPlan, layout::LayoutTable)
    stmts = Expr[]
    for p in plan.predictors
        _parameter_terms(p) || continue
        shape = design_shape(p, plan.columns; levelmaps = plan.levelmaps,
            matrices = plan.matrices)
        chunks = Any[]
        legacy_offset = 0
        for (t, b) in zip(p.terms, shape.blocks)
            b.width == 0 && continue
            if _parameter_term(t)
                value = t.options.parameter
                value = t.options.sign == 1 ? value : :(-$value)
                # All concatenation inputs are vectors. Mixing a traced
                # scalar with a vector takes Julia's scalar-fill cat path.
                push!(chunks, t.kind in (FactorTerm, MatrixTerm) ?
                    :(Float64.($value)) : :([$value]))
            else
                coef = block_name(p.name)
                slice = Expr(:ref, coef,
                    Expr(:call, :(:), legacy_offset + 1,
                        legacy_offset + b.width))
                push!(chunks, :(Float64.($slice)))
                legacy_offset += b.width
            end
        end
        terms = [t for (t, b) in zip(p.terms, shape.blocks) if b.width > 0]
        if length(chunks) == 1 && any(t -> _parameter_term(t) &&
                t.kind in (FactorTerm, MatrixTerm), terms)
            t = only(t for t in terms if _parameter_term(t))
            value = t.options.parameter
            rhs = t.options.sign == 1 ? value : :(-$value)
        else
            rhs = _affine_coordinate_view(p, shape, layout)
            rhs === nothing && (rhs = :(vcat($(chunks...))))
        end
        push!(stmts, :($(_affine_block_name(p)) = $rhs))
    end
    return stmts
end

# A contiguous run of identity-transformed declarations needs no packing
# allocation. This is an optimization of parameter reads, not their layout.
function _affine_coordinate_view(pred, shape, layout)
    reads = Tuple{Symbol,Int}[]
    for (t, b) in zip(pred.terms, shape.blocks)
        b.width == 0 && continue
        _parameter_term(t) && t.options.sign == 1 || return nothing
        push!(reads, (t.options.parameter, b.width))
    end
    return _parameter_coordinate_view(reads, layout)
end

function _parameter_coordinate_view(reads, layout)
    start = nothing
    stop = nothing
    for (name, width) in reads
        i = findfirst(e -> e.name === name, layout.entries)
        i === nothing && return nothing
        e = layout.entries[i]
        e.transform === :identity && e.size == width || return nothing
        stop === nothing || e.offset == stop + 1 || return nothing
        start === nothing && (start = e.offset)
        stop = e.offset + e.size - 1
    end
    start === nothing && return nothing
    return :(view(unconstrained, $start:$stop))
end

# Reassemble split coefficient blocks: predictors whose layout holds
# several `:coefficient` entries (identity/interval runs from uniform
# priors) rejoin into the block name by offset order. Single-entry
# predictors emit nothing (their entry already binds the block name).
# Every segment is materialized with `Float64.(…)`: a `vcat` mixing a
# `SubArray` and a `Vector` lowers through a Union-typed path the
# native Enzyme reverse pass rejects (same rule as the leveled vector
# edges in `_vector_transform_statements`).
function _coef_reassembly_statements(plan::StructuralPlan, layout::LayoutTable)
    stmts = Expr[]
    for pred in plan.predictors
        segs = [e for e in layout.entries
            if e.kind === :coefficient && e.predictor === pred.name]
        length(segs) <= 1 && continue
        sort!(segs; by = e -> e.offset)
        mats = [:(Float64.($(s.name))) for s in segs]
        push!(stmts, :($(block_name(pred.name))::AbstractVector{Float64} =
            vcat($(mats...))))
    end
    return stmts
end

_ordered_columns(plan::StructuralPlan) =
    sort!(collect(plan.columns); by = first)

# The element type and rank come from the column's type parameters, so this
# compiles no generic `eltype`/`ndims` call that other packages' array methods
# could invalidate.
_data_arg(name::Symbol, ::AbstractVector{T}) where {T} = Expr(:(::), name, Vector{T})
_data_arg(name::Symbol, ::AbstractMatrix{T}) where {T} = Expr(:(::), name, Matrix{T})
_data_arg(name::Symbol, v::Number) = Expr(:(::), name, typeof(v))
_data_arg(name::Symbol, ::AbstractArray{T,N}) where {T,N} = Expr(:(::), name, Array{T,N})

# Dedicated eval scope for generated models. The `using` lines resolve via
# this package's own Project (by file location), so generated code loads in
# ANY consumer session with no LOAD_PATH dependence. Builds bind nothing
# here (see `_eval_kernel_def`).
module PPLGeneratedModels
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, bernoulli, poisson, cauchy, exponential, gamma, lognormal,
    beta, inverse_gamma, binomial, negative_binomial2, beta_binomial,
    negative_binomial, weibull,
    uniform, laplace, logistic,
    student_t, zero_inflated_poisson, zero_inflated_binomial,
    normal_id_glm, bernoulli_logit_glm, poisson_log_glm,
    rk_inverse_gaussian_tail, rk_von_mises_cdf, rk_logprob_tail,
    rk_von_mises_periodic_cdf, rk_ordinal_stopping_logpdf, rk_ordinal_stopping_tail
using SpecialFunctions: besseli, besselix, erfc, loggamma
using ReactiveKernelsDistributionKernels.DistributionKernelSources: logbeta
# Selective (explicit imports win over any re-export chain, so no `using`
# ambiguity if ReactiveKernels ever exports these too): the only Statistics
# names in the assignment allowlist.
using Statistics: mean, std, var
using LinearAlgebra: dot
using LogExpFunctions: log1pexp, logaddexp
# Plain `logistic` here is the Logistic distribution kernel object. Keep the
# inverse link as ordinary stable math, including its differentiated graph:
# differentiating a positive-tail exp(x)/(1 + exp(x)) produces Inf/Inf.
import LogExpFunctions: cexpexp as _ppl_cexpexp
_ppl_logistic(x) = exp(-log1pexp(-x))
_ppl_normcdf(x) = 0.5 * erfc(-x / sqrt(2))
# Bijector objects the generated program splices (constrained-parameter
# transforms); imported from the enclosing module so the emitted
# `positive_bijector()` / `unit_bijector()` calls resolve.
import ..positive_bijector, ..unit_bijector
# In-model grouping encoder (`_ppl_gidx_<group>` nodes call it with the
# raw column + literal declared levels).
import .._declared_codes
# Stopping-ratio stage-lane tables (data-only recipes over the bound
# response; `preprocessing.jl`).
import .._ordinal_stage_obs, .._ordinal_stage_idx, .._ordinal_effects_matrix
import .._broadcast_gather
# Multivariate slice priors (`mv_slices.jl`): orientations, per-slice
# arguments, simplex / ordered slice transforms and the slice densities.
import .._SliceRows, .._SliceCols, .._SliceWhole, .._PerSlice
import .._mvnormal_cholesky_slices_logpdf, .._mvnormal_slices_logpdf
import .._dirichlet_slices_logpdf, .._ordered_normal_slices_logpdf
import .._mvnormal_cholesky_slices_pointwise, .._mvnormal_slices_pointwise
import .._dirichlet_slices_pointwise, .._ordered_normal_slices_pointwise
import .._simplex_slices_constrain, .._simplex_slices_logjac
import .._ordered_slices_constrain, .._ordered_slices_logjac
end

# Evaluate the generated `@kernel` definition and return its spec. Nothing is
# bound in `PPLGeneratedModels`: concurrent builds share no package state (no
# lock, no counter) and a built spec is retained only by its caller. RK
# evaluates the recipe closures separately so construction compiles no
# per-model code (`ReactiveKernels._kernel_eval_definition`).
_eval_kernel_def(def::Expr) =
    ReactiveKernels._kernel_eval_definition(PPLGeneratedModels, def)

# Scalar + derived assignments in topo order (params already constrained
# above, so every scalar name resolves; derived columns resolve as locals
# for the recipes below). Unannotated: Int temporaries (e.g. `length`)
# must not meet a Float64 assertion.
function _assignment_statements(plan::StructuralPlan;
        gathers::Set{Tuple{Symbol,Symbol,Int}} = Set{Tuple{Symbol,Symbol,Int}}())
    by_name = Dict{Symbol,Any}(a.name => a for a in plan.assignments)
    for d in plan.derived
        by_name[d.name] = d
    end
    # Data definitions needed at bind (including reduced scales) were
    # evaluated once and arrive as data arguments; the kernel never recomputes them.
    computed = _bound_module_data_names(plan)
    dataonly = Set{Symbol}(keys(plan.columns))
    stmts = Expr[]
    for name in topological_order(plan)
        (haskey(by_name, name) && name ∉ computed) || continue
        ex = _guard_missing_observation_argument(name, by_name[name].expr, plan, plan.columns)
        ex === by_name[name].expr || !_guarded_argument_is_bound(ex, plan.columns) ||
            (ex = _concrete_guarded_argument(ex))
        ex = _value_math_rewrite(_array_gather_rewrite(ex, plan, gathers))
        if _expr_value_symbols(ex) ⊆ dataonly
            push!(dataonly, name)
        else
            ex = _split_data_calls!(stmts, name, ex, dataonly)
        end
        push!(stmts, :($(name) = $(ex)))
    end
    return stmts
end

# The generated scope's `logistic` names a distribution kernel. Values
# use the inverse-logit function, including inside reductions.
_value_math_rewrite(ex) = ex
function _value_math_rewrite(ex::Expr)
    args = Any[_value_math_rewrite(a) for a in ex.args]
    if ex.head in (:call, :.) && !isempty(args) && args[1] === :logistic
        args[1] = :_ppl_logistic
    end
    return Expr(ex.head, args...)
end

# Functions as values: a data-only module call nested in a parameter-
# dependent statement — a data-only definition inlined into the call that
# consumes it (`reads = f(g(s), b)`), or a gathered `(b .* g(gx))[oi]` —
# becomes its own statement. Preparation's `bound=` folding then
# evaluates it once, outside the differentiated program, rather than on
# every evaluation inside it. Only value compositions are entered: an
# operand of a lazy branch, a loop or a generator keeps its own
# evaluation (snag `interpolated-err-41772e14`).
function _split_data_calls!(stmts::Vector{Expr}, name::Symbol, ex,
        dataonly::Set{Symbol})
    count = 0
    function walk(node)
        node isa Expr || return node
        _is_plate_column_expr(node) && return node
        if node.head in (:call, :ref, :.) && _contains_module_call(node) &&
                _expr_value_symbols(node) ⊆ dataonly
            count += 1
            tmp = Symbol(:_ppl_data_, name, :_, count)
            push!(stmts, :($(tmp) = $(node)))
            return tmp
        end
        if node.head === :call
            return Expr(:call, node.args[1], map(walk, node.args[2:end])...)
        elseif node.head === :. && length(node.args) == 2 &&
                Meta.isexpr(node.args[2], :tuple)
            return Expr(:., node.args[1],
                Expr(:tuple, map(walk, node.args[2].args)...))
        elseif node.head === :kw && length(node.args) == 2
            return Expr(:kw, node.args[1], walk(node.args[2]))
        elseif node.head in (:ref, :parameters, :tuple, :vect)
            return Expr(node.head, map(walk, node.args)...)
        end
        return node
    end
    return walk(ex)
end

_lp_name(pred::PredictorSpec) = Symbol(:_ppl_lp_, pred.name)

function _predictor_statements(plan::StructuralPlan)
    stmts = Expr[]
    for pred in plan.predictors
        shape = design_shape(pred, plan.columns; levelmaps = plan.levelmaps,
            matrices = plan.matrices)
        lp = _lp_name(pred)
        terms = Any[]
        if _broadcast_affine(plan, pred)
            append!(terms, _affine_block_terms(plan, shape; broadcast = true))
        elseif shape.width > 0
            push!(terms, :($(design_name(pred.name)) * $(_affine_block_name(pred))))
        end
        if any(b -> b.kind === OffsetTerm, shape.blocks)
            push!(terms, offset_name(pred.name))
        end
        # A latent term contributes the per-cell latent VECTOR directly
        # (identity design): `lp = theta` on its own, or added to fixed-effect
        # design/offset terms for a random-intercept-plus-covariates predictor.
        for b in shape.blocks
            b.kind === LatentTerm && push!(terms, b.column)
        end
        # A scan summand contributes its state's direct scaled expression
        # (`state .* coef`, SB's `ar` latent path with its free beta),
        # resolved from the TERMS like any summand.
        for t in pred.terms
            t.kind === ScanSummandTerm &&
                push!(terms, _scan_summand_expr(plan, pred, t))
        end
        # A composed term evaluates its combination tree in-graph:
        # sub-predictors resolve to their LP nodes (emitted above —
        # contract orders subs first), scalars to their constrained
        # locals (params constrained above, assignments emitted above).
        # Plain broadcast math: ordinary reverse mode on every backend.
        rows = false
        for t in pred.terms
            t.kind === ComposedTerm || continue
            ex = _composed_expr(plan, pred, t)
            rows |= _composed_needs_rows(plan, pred, t, ex)
            push!(terms, ex)
        end
        # Degenerate (e.g. single-level-factor-only) predictors carry a scalar
        # zero LP, which broadcasts everywhere a vector LP would.
        rhs = isempty(terms) ? :(0.0) : foldl((a, b) -> :($a .+ $b), terms)
        # Full-length vector plans read one location entry per row. A value
        # of unknown shape gets those rows once, as Julia broadcasting
        # against the rows would, rather than a ones vector per summand.
        rows && (rhs = Expr(:call, GlobalRef(@__MODULE__, :_ppl_rows), rhs,
            _located_rows_source(plan, pred)))
        push!(stmts, :($lp = $rhs))
    end
    return stmts
end

# Scalar coefficient-coordinate read (`coef[k]`, the `coordinate_read` shape
# over a coefficient block rather than the packed vector).
_coef_coord(coef::Symbol, k::Int) = :($coef[$k])

# Broadcast affine blocks against their coefficient coordinates while
# retaining the authored axes. Factor and matrix blocks use their own
# coefficient slices in design order.
function _affine_block_terms(plan::StructuralPlan, shape::DesignShape;
        broadcast::Bool = false)
    pred = only(p for p in plan.predictors if p.name === shape.predictor)
    coef = _affine_block_name(pred)
    terms = Any[]
    k = 1
    for b in shape.blocks
        if b.kind === InterceptTerm
            push!(terms, broadcast ? _coef_coord(coef, k) :
                Expr(:call, :.*, Expr(:call, :ones, _predictor_rows_source(plan, shape.predictor)),
                    _coef_coord(coef, k)))
            k += 1
        elseif b.kind === ContinuousTerm
            push!(terms, :($(b.column) .* $(_coef_coord(coef, k))))
            k += 1
        elseif b.kind === FactorTerm
            w = b.width
            push!(terms, :($(_contrast_expr(b)) *
                $(:(view($coef, $k:$(k + w - 1))))))
            k += w
        elseif b.kind === MatrixTerm
            w = b.width
            push!(terms, :($(_matrix_block_expr(b, _predictor_rows_source(plan, shape.predictor))) *
                $(:(view($coef, $k:$(k + w - 1))))))
            k += w
        end
    end
    return terms
end


function _scan_summand_expr(plan::StructuralPlan, pred::PredictorSpec, t::TermSpec)
    o = t.options
    any(s -> o.scan_id in s.states, plan.scans) || throw(ContractValidationError(
        "[generator] scan summand in predictor $(pred.name) addresses " *
        "unknown scan :$(o.scan_id)"))
    o.coef === nothing && return o.scan_id
    o.coef in _union_names(plan) || throw(ContractValidationError(
        "[generator] scan summand coef :$(o.coef) is not a scalar name"))
    return Expr(:call, :.*, o.scan_id, o.coef)
end

const _COMPOSED_MAP_EMIT = Dict{Symbol,Symbol}(:exp => :exp,
    :logistic => :_ppl_logistic, :normcdf => :_ppl_normcdf,
    :cexpexp => :_ppl_cexpexp)

_composed_map_emit(f::Symbol) = get(_COMPOSED_MAP_EMIT, f, f)
_composed_map_emit(f) = f

"""Rewrite a composed tree to in-graph nodes (contract validated it)."""
function _composed_rewrite(node, subs::Vector{Symbol}, plan::StructuralPlan,
        pred::Symbol)
    if _is_plate_column_expr(node)
        original = node
        node = _guard_missing_observation_argument(pred, node, plan, plan.columns)
        aliases = Dict{Symbol,Symbol}(s => _lp_name(_predictor(plan, s)) for s in subs)
        inputs = [_hsubst(a, aliases) for a in node.args[1].args[2:end]]
        value = Expr(:do, Expr(:call, :plate, inputs...), node.args[2])
        return node !== original && _guarded_argument_is_bound(node, plan.columns) ?
            _concrete_guarded_argument(value) : value
    end
    if node isa Symbol
        node in subs || return node
        i = findfirst(p -> p.name === node, plan.predictors)
        i === nothing && throw(ContractValidationError(
            "[generator] composed term in predictor $pred addresses " *
            "unknown sub-predictor $node"))
        return _lp_name(plan.predictors[i])
    end
    # Dotted map `f.(x, ...)`: `Expr(:., f, Expr(:tuple, x, ...))`, the
    # map renamed to its generated-module math binding.
    node.head === :. && return Expr(:., _composed_map_emit(node.args[1]),
        Expr(:tuple, (_composed_rewrite(a, subs, plan, pred)
            for a in node.args[2].args)...))
    return Expr(node.head, node.args[1],
        (_composed_rewrite(a, subs, plan, pred) for a in node.args[2:end])...)
end

_composed_expr(plan::StructuralPlan, pred::PredictorSpec, t::TermSpec) =
    _composed_rewrite(t.options.tree, t.options.subs, plan, pred.name)

# A composition reading no sub-predictor and no column has no row axis of
# its own. Scalar shape is retained when observation operands need
# broadcasting; ordinary full-length vector plans give it the location's rows.
_composed_needs_rows(plan::StructuralPlan, pred::PredictorSpec, t::TermSpec, ex) =
    !_is_plate_column_expr(ex) && isempty(t.options.subs) &&
        isempty(t.columns) && !_broadcast_affine(plan, pred)

# Rows of the responses a predictor locates — their own observation axis,
# which is not `n_obs` when responses observe different rows.
function _located_rows(plan::StructuralPlan, pred::PredictorSpec)
    rows = unique!([_response_rows(plan, r) for r in plan.responses
        if _response_uses_predictor(r, pred.name)])
    length(rows) == 1 || throw(ContractValidationError("[generator] " *
        "predictor $(pred.name) locates responses with rows $rows " *
        "(one row count per scalar-valued predictor)"))
    return only(rows)
end

# Row counts as graph source. A row count the lowering resolved from the
# bound data is emitted as `_observation_rows(anchors...)` over those same
# bound values, never as the build's number, so a built graph evaluates any
# binding of its program (snag rkppl-opaque-loc-37de7b80). The anchors
# mirror the row resolvers (`_located_rows`, `_response_rows`,
# `_value_rows`), which keep validating and measuring; `_rows_source`
# checks that the call reproduces their count on this binding. A count no
# bound value carries (no data anchor) stays a literal.
function _rows_source(n::Int, anchors::Vector)
    isempty(anchors) && return n
    values = Any[last(a) for a in anchors]
    _observation_rows_compatible(values) && _observation_rows(values...) == n ||
        return n
    return Expr(:call, GlobalRef(@__MODULE__, :_observation_rows),
        Any[first(a) for a in anchors]...)
end

_located_rows_source(plan::StructuralPlan, pred::PredictorSpec) =
    _rows_source(_located_rows(plan, pred), _response_rows_anchors(plan,
        first(r for r in plan.responses if _response_uses_predictor(r, pred.name))))

_response_rows_source(plan::StructuralPlan, r::LikelihoodSpec) =
    _rows_source(_response_rows(plan, r), _response_rows_anchors(plan, r))

_predictor_rows_source(plan::StructuralPlan, name::Symbol) =
    _rows_source(_predictor_rows(plan, name), _value_rows_anchors(plan, name))

# The bound values `_value_rows` measures: the value itself when bound, else
# the per-observation columns it reads, else the rows of the responses that
# read it (of every response, when none does).
function _value_rows_anchors(plan::StructuralPlan, name::Symbol)
    haskey(plan.columns, name) && return Any[_column_anchor(plan, name)]
    reads = _response_reads(plan, name, _row_columns(plan))
    isempty(reads) || return Any[_column_anchor(plan, c) for c in sort!(collect(reads))]
    names = Set{Symbol}([name])
    users = [r for r in plan.responses if name in _response_reads(plan, r, names)]
    isempty(users) && (users = plan.responses)
    isempty(users) && return Any[]
    return _response_rows_anchors(plan, first(users))
end

_column_anchor(plan::StructuralPlan, c::Symbol) = c => plan.columns[c]

# A ranged response's selection, as its likelihood statements index it.
_selection_anchor(plan::StructuralPlan, r::LikelihoodSpec) =
    Expr(:ref, r.response, _range_index_expr(r), r.range.args[3:end]...) =>
        _selected_response_column(plan, r)

_range_index_expr(r::LikelihoodSpec) = r.range.args[2] === :(:) ?
    Expr(:call, :eachindex, r.response) : r.range.args[2]

# The bound values whose broadcast rows `_response_rows` measures: the
# response (or its selection) and the per-observation columns its
# broadcast domain reads; structured responses keep their row contract.
function _response_rows_anchors(plan::StructuralPlan, r::LikelihoodSpec)
    if !_uses_structured_observation_axes(plan) && haskey(plan.columns, r.response) &&
            _observation_axes(plan) !== nothing
        modelvals, managed = _axis_exempt_columns(plan)
        perobs = Set{Symbol}(k for (k, v) in plan.columns
            if k ∉ modelvals && k ∉ managed && v isa AbstractArray)
        head = r.range isa Expr ? _selection_anchor(plan, r) :
            _column_anchor(plan, r.response)
        # Indexed operands are validated to the selection's own axis.
        r.range isa Expr && r.response in plan.indexed_observations && return Any[head]
        reads = _response_reads(plan, r, perobs)
        values = r.range isa Expr ?
            _response_reads(plan, _with(r; range = nothing), perobs) : reads
        return Any[head; [_column_anchor(plan, c)
            for c in sort!(collect(setdiff(reads, (r.response,)))) if c in values]]
    end
    r.range isa Expr && return Any[_selection_anchor(plan, r)]
    r.mi_jobs === nothing && haskey(plan.columns, r.response) &&
        return Any[_column_anchor(plan, r.response)]
    reads = _response_reads(plan, r, _row_columns(plan))
    return Any[_column_anchor(plan, c) for c in sort!(collect(reads))]
end



_lik_name(label::Symbol) = Symbol(:_ppl_lik_, label)

function _likelihood_statements(plan::StructuralPlan, layout; gathers,
        upstream::Vector{Expr} = Expr[])
    stmts = Expr[]
    terms = Any[]
    points = Pair{Symbol,Any}[]
    for r in plan.responses
        # A plate loop covers its whole response, so a cell-broadcast
        # response observes every entry of each index's array.
        rs = _cell_broadcast_response(plan, r) ?
            _cell_broadcast_stmts(r, plan,
                _response_likelihood_stmts(_with(r; range = nothing), plan), upstream) :
            _response_likelihood_stmts(r, plan)
        append!(stmts, rs)
        push!(terms, _lik_name(r.label))
        pw = _pw_name(r.label)
        if r.family === OrdinalFam && r.ordinal_structure === :stopping && r.evidence.kind === :none &&
                r.n_levels > 1 && _response_rows(plan, r) > 0 && _has_observed_response(plan, r)
            # Stopping-ratio emits one lane per visited stage. Gather the
            # prefix sum at each observation's last stage, then difference.
            ends = Symbol(pw, :_ends)
            cumulative = Symbol(pw, :_cumulative)
            values = Symbol(pw, :_observations)
            observed = r.range isa Expr ? Symbol(:_ppl_range_response_, r.label) :
                r.mi_jobs !== nothing ? Symbol(:_ppl_mi_response_, r.label) : r.response
            push!(stmts, :($ends = cumsum(min.($observed, $(r.n_levels-1)))),
                :($cumulative = cumsum($pw)[$ends]),
                :($values = $cumulative .- vcat(0.0, $cumulative)[1:end-1]))
            pw = values
        end
        push!(points, r.response => pw)
    end
    for p in plan.external_observations
        ps = _external_density_statements(p)
        append!(stmts, _cell_broadcast_observation(plan, p.name) ?
            _cell_broadcast_stmts(p, plan, ps, upstream) : ps)
        push!(terms, Symbol(:_ppl_prior_, p.name))
        push!(points, p.name => Symbol(:_ppl_pw_prior_, p.name))
    end
    observed = _density_selection(plan, n -> n in plan.conditioned)
    _parameter_prior_statements!(stmts, terms, observed, layout;
        prefix = :_ppl_condition_, group_scalars = false)
    for p in observed.parameters
        push!(points, p.name => Symbol(
            p.family === :external && p.args.broadcast ? :_ppl_pw_prior_ : :_ppl_prior_, p.name))
    end
    for p in observed.vector_parameters
        _vector_parameter_prior_stmts!(stmts, terms, p)
        value = p.family === :vector_normal ?
            (p.size == 0 ? :(zeros(0)) : Symbol(:_ppl_pw_prior_, p.name)) :
            Symbol(:_ppl_prior_, p.name)
        push!(points, p.name => value)
    end
    for p in observed.plate_parameters
        if p.family === :external
            append!(stmts, _external_density_statements(p))
            push!(terms, Symbol(:_ppl_prior_, p.name))
        else
            _vector_prior_stmts!(stmts, terms, p.name, p.family, p.args,
                p.support_override; conditioned = true, rows = _plate_rows(plan, p))
        end
        push!(points, p.name => Symbol(:_ppl_pw_prior_, p.name))
    end
    _array_prior_stmts!(stmts, terms, observed, gathers; context = plan, pointwise = points)
    joint = foldl((a, b) -> :($a + $b), terms; init = :(0.0))
    push!(stmts, :(likelihood::Float64 = $joint))
    names = first.(points)
    length(unique(names)) == length(names) || throw(ContractValidationError(
        "[query] pointwise observation names must be unique, got $names"))
    values = isempty(points) ? :(NamedTuple()) :
        Expr(:tuple, (Expr(:(=), name, value) for (name, value) in points)...)
    push!(stmts, Expr(:(=), :pointwise, values))
    return stmts
end

# A dotted `@plate` cell (`y[i] .~ D.(v[i], s)`) broadcasts over its own
# iteration's values, as Julia does. When the bound response holds one array
# per index, the response's ordinary statements run once per index inside an
# outer RK plate, giving RK's nested group and observation plates
# (`reactivekernels-use` §4e1). Values the cell reads per index, and values
# computed from them, supply that index's array or number; every other value
# is shared with all indices (`Ref`). The pointwise result keeps the
# response's shape: one array of observation densities per index.
function _cell_broadcast_stmts(r::LikelihoodSpec, plan::StructuralPlan,
        stmts::Vector{Expr}, upstream::Vector{Expr})
    # No index: the ordinary empty-domain statements already sum to zero.
    _response_rows(plan, r) == 0 && return stmts
    # The response's predictor nodes lie on its observation axis, here the
    # indices.
    return _cell_broadcast_group(plan, r.response, _lik_name(r.label), _pw_name(r.label),
        stmts, upstream, [_lp_name(p) for p in plan.predictors],
        "response $(r.label) observes one array per index, and its $(r.family) statements")
end

# A caller-owned sampling RHS (`y[i] .~ LogDensity.(f, loc[i], s)`) in the
# same cell: its law, including a visible `KernelSpec` density, runs on each
# index's entries exactly as the flat observation plate runs on the entries
# of a numeric response.
function _cell_broadcast_stmts(p::SampledParameter, plan::StructuralPlan,
        stmts::Vector{Expr}, upstream::Vector{Expr})
    isempty(plan.columns[p.name]) && return stmts
    return _cell_broadcast_group(plan, p.name, Symbol(:_ppl_prior_, p.name),
        Symbol(:_ppl_pw_prior_, p.name), stmts, upstream, Symbol[],
        "observation $(p.name) holds one array per index, and its sampling-RHS statements")
end

function _cell_broadcast_group(plan::StructuralPlan, response::Symbol, node::Symbol,
        pw::Symbol, stmts::Vector{Expr}, upstream::Vector{Expr},
        axisnodes::Vector{Symbol}, what::String)
    k = findall(st -> Meta.isexpr(st, :(=), 2) && st.args[1] == :($node::Float64), stmts)
    (length(k) == 1 && stmts[only(k)].args[2] == :(sum($pw))) ||
        throw(ContractValidationError("[generator] $what have no summed " *
            "pointwise plate to run per index"))
    body = Expr[st for (i, st) in enumerate(stmts) if i != only(k)]
    defined = Set{Symbol}(something.(_assigned_name.(body), :_))
    known = Set{Symbol}(keys(plan.columns))
    for st in upstream
        name = _assigned_name(st)
        name === nothing || push!(known, name)
    end
    reads = Set{Symbol}()
    foreach(st -> _statement_value_reads!(reads,
        Meta.isexpr(st, :(=), 2) ? st.args[2] : st), body)
    inputs = sort!(collect(intersect(setdiff!(reads, defined), known)))
    # Per-index values: the response, the cell's indexed reads, nodes on the
    # response's observation axis and every upstream definition computed
    # from them.
    pergroup = Set{Symbol}([response; plan.cell_broadcasts[response]; axisnodes])
    for st in upstream
        name = _assigned_name(st)
        name === nothing && continue
        used = _statement_value_reads!(Set{Symbol}(), st.args[2])
        isempty(intersect(used, pergroup)) || push!(pergroup, name)
    end
    aliases = Dict{Symbol,Symbol}(v => Symbol(:_ppl_group_, v) for v in inputs)
    group = Symbol(pw, :_group)
    aliases[pw] = group
    cell = Expr[_hsubst(st, aliases) for st in body]
    _cell_value_types!(cell, Set{Symbol}(aliases[v] for v in inputs if v in pergroup))
    # Densities are Float64; declaring the observation cell's result also
    # types an empty index's densities (RK's empty-domain result evidence).
    for st in cell
        Meta.isexpr(st, :(=), 2) && st.args[1] === group &&
            Meta.isexpr(st.args[2], :do) && _declare_cell_result!(st.args[2].args[2])
    end
    outer = Any[v in pergroup ? v : :(Ref($v)) for v in inputs]
    # The pointwise query reads each index's densities and the likelihood
    # its per-index totals; RK evaluates only the plate a query selects. The
    # declared total type also seeds RK's summed plate when the operands'
    # element types are known only at run time (a composed child's output).
    groupplate(result...) = Expr(:do, Expr(:call, :plate, outer...),
        Expr(:(->), Expr(:tuple, (aliases[v] for v in inputs)...),
            Expr(:block, LineNumberNode(0, :generator), deepcopy(cell)..., result...)))
    totals = Symbol(pw, :_totals)
    return Expr[
        :($pw = $(groupplate(Expr(:call, GlobalRef(Base, :identity), group)))),
        :($totals = $(groupplate(:(_ppl_group_total::Float64 = sum($group)),
            :_ppl_group_total))),
        :($node::Float64 = sum($totals))]
end

# The ordinary statements declare array types for values on the response's
# observation axis (`_ppl_sc_y::AbstractVector = …`). In a group cell such a
# value holds one index's value: an array, or a number when the cell reads a
# per-group scalar (`dose[i] * sigma`). Those statements keep their values
# and drop the axis declaration; scalar declarations and statements reading
# only shared values are unchanged.
function _cell_value_types!(cell::Vector{Expr}, pervalue::Set{Symbol})
    for st in cell
        Meta.isexpr(st, :(=), 2) || continue
        reads = _statement_value_reads!(Set{Symbol}(), st.args[2])
        isempty(intersect(reads, pervalue)) && continue
        lhs = st.args[1]
        if Meta.isexpr(lhs, :(::), 2) && _array_type_annotation(lhs.args[2])
            st.args[1] = lhs = lhs.args[1]
        end
        name = _assigned_name(st)
        name === nothing || push!(pervalue, name)
    end
    return cell
end

_array_type_annotation(T) = false
_array_type_annotation(T::Type) = T <: AbstractArray
_array_type_annotation(T::Symbol) =
    T in (:AbstractArray, :AbstractVector, :AbstractMatrix, :Array, :Vector, :Matrix)
_array_type_annotation(T::Expr) =
    Meta.isexpr(T, :curly) && _array_type_annotation(T.args[1])

function _declare_cell_result!(lambda::Expr)
    body = lambda.args[2]
    k = findlast(a -> !(a isa LineNumberNode), body.args)
    body.args[k] = Expr(:(=), :(_ppl_density::Float64), body.args[k])
    push!(body.args, :_ppl_density)
    return lambda
end

_assigned_name(st) = nothing
function _assigned_name(st::Expr)
    Meta.isexpr(st, :(=), 2) || return nothing
    lhs = st.args[1]
    Meta.isexpr(lhs, :(::), 2) && (lhs = lhs.args[1])
    return lhs isa Symbol ? lhs : nothing
end

# Value names a generated statement reads: call heads, keyword names and
# plate cell bodies (which read only their own arguments) are skipped.
_statement_value_reads!(out, ex) = out
_statement_value_reads!(out, ex::Symbol) = push!(out, ex)
function _statement_value_reads!(out, ex::Expr)
    if ex.head === :do
        _statement_value_reads!(out, ex.args[1])
    elseif ex.head === :call && !isempty(ex.args)
        foreach(a -> _statement_value_reads!(out, a), ex.args[2:end])
    elseif ex.head === :. && length(ex.args) == 2
        ex.args[2] isa QuoteNode && _statement_value_reads!(out, ex.args[1])
        ex.args[2] isa Expr && foreach(a -> _statement_value_reads!(out, a),
            ex.args[2].args)
    elseif ex.head === :kw
        _statement_value_reads!(out, ex.args[2])
    elseif ex.head === :(->)
        nothing
    else
        foreach(a -> _statement_value_reads!(out, a), ex.args)
    end
    return out
end

# Packed observations keep their own axis. Every other observation input
# uses the same retained gather plate, including bounds, weights and trials.
_mi_row_value(x::Number, i) = x
@traceable _mi_row_value(x::AbstractVector, i) = x[i]

_ppl_range_values(x::Number, indices) = x
_ppl_range_values(x::AbstractArray, indices) = x[indices]

# `ones(n) .* x` without the ones operand: a number fills the rows, and a
# vector already holding one Float64 per row is that location value itself.
# Other values (traced, integer, singleton or mismatched) keep Julia's
# broadcast against the rows, including its dimension errors.
@inline _ppl_rows(x::Float64, n::Int) = fill(x, n)
@inline _ppl_rows(x::Vector{Float64}, n::Int) = length(x) == n ? x : ones(n) .* x
@inline _ppl_rows(x, n::Int) = ones(n) .* x

# A selected plate column already holds one value per selected response cell
# (`_selected_plate_indices`); only whole operands are gathered at the
# authored indices. Positions and indices differ for a literal `a:b`.
function _selected_cell_value(plan::StructuralPlan, r::LikelihoodSpec, value::Symbol)
    i = findfirst(p -> _lp_name(p) === value, plan.predictors)
    i === nothing && return false
    p = plan.predictors[i]
    p.link === IdentityLink && length(p.terms) == 1 &&
        p.terms[1].kind === OffsetTerm || return false
    d = findfirst(d -> d.name === only(p.terms[1].columns), plan.derived)
    return d !== nothing && _selected_plate_indices(plan.derived[d].expr) == r.range.args[2]
end

function _ranged_response_stmts(r, plan, stmts)
    idx = Symbol(:_ppl_range_indices_, r.label)
    selected = Symbol(:_ppl_range_response_, r.label)
    pre = Expr[:($idx = collect($(_range_index_expr(r))))]
    if _response_rows(plan, r) == 0 && r.response in plan.indexed_observations
        push!(pre, :($selected = zeros(0)))
    else
        push!(pre, Expr(:(=), selected, Expr(:ref, r.response, idx, r.range.args[3:end]...)))
    end
    aliases = Dict{Symbol,Symbol}(r.response => selected)
    if r.response in plan.indexed_observations
        rows = Set{Symbol}()
        # Select ordinary row values before family-specific conversions or
        # ordinal stage expansion. Simplexes and covariance factors stay whole.
        if r.family ∉ (CategoricalFam, MultinomialFam)
            push!(rows, _is_bare_param_location(r, plan) ? r.predictor :
                _location_node(r, plan))
            foreach(p -> push!(rows, _lp_name(_predictor(plan, p))), r.extra_predictors)
        end
        for slot in (r.scale, r.nu, r.zi, r.discrimination, r.weights, r.trials,
                r.evidence.lower, r.evidence.upper, r.threshold_columns...,
                r.count_columns..., r.extra_responses...)
            if slot isa ScalePredictorRef
                push!(rows, _lp_name(_predictor(plan, slot.predictor)))
            elseif slot isa Symbol
                pred = findfirst(p -> p.name === slot, plan.predictors)
                push!(rows, pred === nothing ? slot : _lp_name(plan.predictors[pred]))
            end
        end
        for (i, value) in enumerate(sort!(collect(rows)))
            value === r.response && continue
            _selected_cell_value(plan, r, value) && continue
            alias = Symbol(:_ppl_range_operand_, r.label, :_, i)
            getter = GlobalRef(@__MODULE__, :_ppl_range_values)
            push!(pre, :($alias = $getter($value, $idx)))
            aliases[value] = alias
        end
    end
    stmts = Expr[_hsubst(st, aliases) for st in stmts]
    return Expr[pre..., stmts...]
end

# Pure scalar math keeps the CDF inside the selected censoring arm; endpoint
# expansion currently cannot splice the normal CDF's multi-recipe graph there.
_mixture_normal_cdf(x, mu, sigma) = 0.5 * erfc((mu - x) / (sqrt(2.0) * sigma))

function _response_likelihood_stmts(r::LikelihoodSpec, plan::StructuralPlan)
    # No cell is evaluated on an empty bound observation domain. Its sum is
    # the additive identity, independently of the response family.
    if _response_rows(plan, r) == 0 || !_has_observed_response(plan, r)
        obs = _observation_axes(plan)
        shape = obs !== nothing && hasproperty(obs, :domains) ?
            length.(obs.domains[r.label]) :
            r.range isa Expr ? size(_selected_response_column(plan, r)) : size(plan.columns[r.response])
        return Expr[
        :($(_pw_name(r.label)) = zeros($shape)),
        :($(_lik_name(r.label))::Float64 = 0.0)]
    end
    stmts = _response_likelihood_stmts_full(r, plan)
    if r.range isa Expr
        return _mask_response_stmts(r, plan, _ranged_response_stmts(r, plan, stmts))
    end
    stmts = _mask_response_stmts(r, plan, stmts)
    r.mi_jobs === nothing && return stmts
    if r.family === OrdinalFam && r.ordinal_structure === :stopping
        return _mi_stopping_response_stmts(r, plan, stmts)
    end
    _is_glm_family(r.family) && r.evidence.kind === :none && return stmts # fused design selects Jobs before eta
    pre = Expr[]
    jobs = r.mi_jobs
    packed = nothing
    if r.range !== nothing
        packed = Symbol(:_ppl_mi_packed_, r.label)
        selected = Symbol(:_ppl_mi_rows_, r.label)
        # Data-only selection runs at preparation; it never emits one body
        # per selected row. Jobs address the full axis, y the packed axis.
        rng = r.range
        push!(pre, :($packed = findall(j -> j in $rng, $jobs)))
        push!(pre, :($selected = $jobs[$packed]))
        jobs = selected
    end
    obsidx = findfirst(st -> st.head === :(=) && st.args[1] === _pw_name(r.label), stmts)
    obsidx === nothing && throw(ContractValidationError(
        "[generator] mi response $(r.label) has no observation plate"))
    let st = stmts[obsidx]
        call = st.args[2].args[1]
        call.args[1] === :plate || throw(ContractValidationError(
            "[generator] mi response $(r.label) needs an observation plate"))
        for i in 2:length(call.args)
            ref = call.args[i]
            # Ref inputs are whole model values (e.g. a simplex), never rows.
            ref isa Symbol || continue
            ispacked = i == 2 || ref in _mi_packed_columns(r)
            ispacked && packed === nothing && continue
            rows = ispacked ? packed : jobs
            call.args[i] = _mi_gather_node!(pre, rows, ref, r.label)
        end
    end
    return Expr[stmts[1:obsidx-1]..., pre..., stmts[obsidx:end]...]
end

# Presence is bound data. RK's ordinary plate branch lowering partitions the
# lanes before density evaluation, so Missing/union values never enter RK or AD.
# Full response axes and all model declarations remain intact.
function _mask_response_stmts(r, plan, stmts)
    mask = _observed_mask_name(r.response)
    haskey(plan.columns, mask) || return stmts
    pre = Expr[]
    if r.range isa Expr
        selected = Symbol(:_ppl_present_range_, r.label)
        idx = Symbol(:_ppl_range_indices_, r.label)
        push!(pre, Expr(:(=), selected, Expr(:ref, mask, idx, r.range.args[3:end]...)))
        mask = selected
    end
    if r.family === OrdinalFam && r.ordinal_structure === :stopping &&
            r.evidence.kind === :none && r.n_levels > 1
        stages = Symbol(:_ppl_present_stages_, r.label)
        obs = _stage_lane(r.label, :obs)
        push!(pre, :($stages = $mask[$obs]))
        mask = stages
    end
    pw = _pw_name(r.label)
    i = findfirst(st -> st.head === :(=) && st.args[1] === pw, stmts)
    i === nothing && throw(ContractValidationError("[generator] response $(r.label) has no pointwise observation"))
    st = deepcopy(stmts[i])
    rhs = st.args[2]
    if Meta.isexpr(rhs, :do) && rhs.args[1].args[1] === :plate
        inputs, lambda = rhs.args[1], rhs.args[2]
        present = _dovar(length(inputs.args))
        push!(inputs.args, mask)
        push!(lambda.args[1].args, present)
        lambda.args[2] = Expr(:block, LineNumberNode(0, :generator),
            Expr(:if, present, lambda.args[2], 0.0))
        return Expr[stmts[1:i-1]..., pre..., st, stmts[i+1:end]...]
    end
    throw(ContractValidationError(
        "[generator] response $(r.label) needs an observation plate for presence guards"))
end

# Stopping-ratio inputs must be packed before stage expansion. Selecting
# its final plate would mistake stage lanes for full observation rows.
function _mi_stopping_response_stmts(r, plan, stmts)
    pre = Expr[]
    jobs = r.mi_jobs
    observed = Symbol(:_ppl_mi_response_, r.label)
    if r.range === nothing
        push!(pre, :($observed = $(r.response)))
    else
        packed = Symbol(:_ppl_mi_packed_, r.label)
        selected = Symbol(:_ppl_mi_rows_, r.label)
        push!(pre, :($packed = findall(j -> j in $(r.range), $jobs)),
            :($selected = $jobs[$packed]),
            :($observed = $(r.response)[$packed]))
        jobs = selected
    end
    aliases = Dict{Symbol,Symbol}(r.response => observed)
    rows = Set{Symbol}([_location_node(r, plan)])
    for ref in (r.weights, r.discrimination, r.evidence.lower, r.evidence.upper,
            r.threshold_columns...)
        if ref isa ScalePredictorRef
            push!(rows, _lp_name(_predictor(plan, ref.predictor)))
        elseif ref isa Symbol
            pred = findfirst(p -> p.name === ref, plan.predictors)
            push!(rows, pred === nothing ? ref : _lp_name(plan.predictors[pred]))
        end
    end
    for ref in sort!(collect(rows))
        aliases[ref] = _mi_gather_node!(pre, jobs, ref, r.label)
    end
    if r.threshold_effects !== nothing
        matrix = Symbol(:_ppl_mi_effects_, r.label)
        push!(pre, :($matrix = $(r.threshold_effects)[$jobs, :]))
        aliases[r.threshold_effects] = matrix
    end
    return Expr[pre..., (_hsubst(st, aliases) for st in stmts)...]
end

# One plate likelihood per response (pointwise plate + scalar sum node).
# Triples 2 and 3 (Bernoulli-logit) lower identically; the triple only
# selects the form. Branches are explicit per family; the else is a
# fail-closed guard for enum members without an emitter (never silent).
function _response_likelihood_stmts_full(r::LikelihoodSpec, plan::StructuralPlan)
    node = _lik_name(r.label)
    pw = _pw_name(r.label)
    if r.family === GaussianFam
        return _gaussian_plate_stmts(r, plan, node, pw)
    elseif r.family === StudentTFam
        return _student_plate_stmts(r, plan, node, pw)
    elseif r.family === LogNormalFam
        return _lognormal_plate_stmts(r, plan, node, pw)
    elseif r.family === BernoulliLogitFam
        # Base GLM case (no evidence, no weights, no literal range): fused
        # whole-vector reduction. Ranged responses stay on the plate path
        # (the cover rule makes them whole-column today, but the fused sum
        # must never silently outgrow a future partial range). A response
        # holding one array per index runs its pointwise plate per index.
        r.evidence.kind === :none && r.weights === nothing &&
            r.range === nothing && r.mi_jobs === nothing && r.link === LogitLink &&
            !haskey(plan.columns, _observed_mask_name(r.response)) &&
            !_is_bare_param_location(r, plan) &&
            !_cell_broadcast_response(plan, r) &&
            _wholevec_location(r, plan) &&
            return _bernoulli_wholevec_stmts(r, plan, node)
        return _bernoulli_plate_stmts(r, plan, node, pw)
    elseif r.family === PoissonLogFam
        # Base GLM case (no evidence, no weights, no literal range): fused
        # whole-vector reduction (faster native + Reactant; the per-cell
        # plate handles evidence/weights/ranges).
        r.evidence.kind === :none && r.weights === nothing &&
            r.range === nothing && r.mi_jobs === nothing && r.link === LogLink &&
            !haskey(plan.columns, _observed_mask_name(r.response)) &&
            !_is_bare_param_location(r, plan) &&
            !_cell_broadcast_response(plan, r) &&
            _wholevec_location(r, plan) &&
            return _poisson_wholevec_stmts(r, plan, node)
        return _poisson_plate_stmts(r, plan, node, pw)
    elseif r.family === HurdlePoissonFam
        return _hurdle_plate_stmts(r, plan, node, pw)
    elseif r.family === ZeroInflatedPoissonFam
        return _zip_plate_stmts(r, plan, node, pw)
    elseif r.family === InverseGaussianFam
        return _ig_plate_stmts(r, plan, node, pw)
    elseif r.family === ExponentialLogFam
        return _exponential_plate_stmts(r, plan, node, pw)
    elseif r.family === VonMisesFam
        return _vonmises_plate_stmts(r, plan, node, pw)
    elseif r.family === BinomialLogitFam
        return _binomial_plate_stmts(r, plan, node, pw)
    elseif r.family === BinomialProbFam
        return _binomial_prob_plate_stmts(r, plan, node, pw)
    elseif r.family === ZeroInflatedBinomialFam
        return _zib_plate_stmts(r, plan, node, pw)
    elseif r.family === NegativeBinomial2Fam
        return _nb2_plate_stmts(r, plan, node, pw)
    elseif r.family === NegativeBinomialFam
        return _nb1_plate_stmts(r, plan, node, pw)
    elseif r.family === WeibullFam || r.family === WeibullValueFam
        return _weibull_plate_stmts(r, plan, node, pw)
    elseif r.family === GammaLogFam || r.family === GammaValueFam
        return _gamma_plate_stmts(r, plan, node, pw)
    elseif r.family === BernoulliProbitFam
        return _bernoulli_probit_plate_stmts(r, plan, node, pw)
    elseif r.family === BernoulliCloglogFam
        return _bernoulli_cloglog_plate_stmts(r, plan, node, pw)
    elseif r.family === BinomialProbitFam
        return _binomial_probit_plate_stmts(r, plan, node, pw)
    elseif r.family === BinomialCloglogFam
        return _binomial_cloglog_plate_stmts(r, plan, node, pw)
    elseif r.family === BetaLogitFam
        return _beta_plate_stmts(r, plan, node, pw)
    elseif r.family === BetaShapeFam
        return _beta_shape_plate_stmts(r, plan, node, pw)
    elseif r.family === BetaBinomial2Fam
        return _betabinomial2_plate_stmts(r, plan, node, pw)
    elseif r.family === CategoricalLogitFam
        return _categorical_plate_stmts(r, plan, node, pw)
    elseif r.family === OrderedLogisticFam || r.family === OrdinalFam
        return _ordinal_plate_stmts(r, plan, node, pw)
    elseif r.family === MultinomialFam
        return _multinomial_plate_stmts(r, plan, node, pw)
    elseif r.family === CategoricalFam
        return _categorical_plain_plate_stmts(r, plan, node, pw)
    elseif r.family === MvNormalCholeskyFam
        return _mvn_cholesky_plate_stmts(r, plan, node, pw)
    elseif r.family === MixtureFam
        return _mixture_plate_stmts(r, plan, node, pw)
    elseif _is_glm_family(r.family)
        return _glm_object_stmts(r, plan, node, pw)
    else
        throw(ContractValidationError(
            "[generator] response family $(r.family) has no emitter"))
    end
end

# Finite-mixture likelihood (SB `MixtureModel` mirror): K same-family
# components over dedicated slots lower to ONE plate; each row contributes
# a stable logaddexp fold over `logw_k + lpdf_k` (linear in K).
# Predictor locations ride their LP nodes
# (link-space, inverted per the component link like the single-family
# builders); sampled params thread scalar (broadcast) and literals inline
# (both constrained-scale, no inversion). K=1 uses the general form
# (exact: the single component contribution).
function _mixture_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan,
        node::Symbol, pw::Symbol)
    f = r.mixture_family
    K = length(r.mixture_locs)
    y = r.response
    inputs = Any[y]
    yv = _dovar(1)
    pre = Expr[]
    # Bernoulli widths: validated Bool-or-0/1-Int; the endpoint takes
    # Bool lanes (bind-folded `_ppl_yb_` recipe, never an in-cell
    # comparison — `_bernoulli_yplate!`).
    yref = yv
    if f === BernoulliLogitFam
        yin, yref = _bernoulli_yplate!(pre, plan, y, r.label, yv)
        inputs[1] = yin
    end
    # Equal trial arguments share one input; distinct arguments thread per component.
    nref = nothing
    if f === BinomialLogitFam
        inputs[1] = _count_yplate!(pre, plan, y, r.label)
        if r.trials !== nothing
            nref = _thread_ref!(inputs, r.trials, true)
        end
    end
    # Weights: literals fold at codegen; a simplex parameter binds one
    # log-vector hoisted out of the plate, threaded by `Ref` (the
    # multinomial-plate precedent — plate cells cannot capture body
   # locals) — K logs, not n×K.
    w = r.mixture_weights
    logw_lit = w isa Vector ? log.(w) : nothing
    lwv = nothing
    logw_comp = nothing
    if w isa Symbol
        logp = _logp_name(r.label)
        push!(pre, :($logp::AbstractVector{Float64} = log.($w)))
        push!(inputs, :(Ref($logp)))
        lwv = _dovar(length(inputs))
    elseif w isa MixtureComplementWeights
        # Complement pair (K == 2 by contract): thread the constrained
        # scalar once; `log1p(-p)` is the stable `log(1-p)` arm.
        pv = _thread_ref!(inputs, w.param)
        first, second = :(log($pv)), :(log1p(-$pv))
        logw_comp = w.param_first ? (first, second) : (second, first)
    end
    terms = Expr[]
    distributions = Any[]
    logweights = Any[]
    parameter_guards = Any[]
    for k in 1:K
        klab = Symbol(r.label, :_mix, k)
        locref, is_lp = _mixture_loc_ref(r, plan, k)
        nk = isempty(r.mixture_trials) ? nref :
            _thread_ref!(inputs, r.mixture_trials[k], true)
        normal_params = Any[]
        lpdf = _mixture_component_lpdf(f, r, plan, pre, inputs, k, klab,
            locref, is_lp, yv, yref, nk; normal_params)
        if lpdf.head === :if
            push!(parameter_guards, lpdf.args[1])
            # One lazy domain guard encloses the entire mixture, including
            # its evidence corrections. Each component is evaluated only in
            # that valid arm, so repeating its guard inside the arm adds no
            # semantics and needlessly nests the differentiated branches.
            lpdf = lpdf.args[2]
        end
        logw_k = logw_lit === nothing ?
            (logw_comp === nothing ? :($lwv[$k]) : logw_comp[k]) :
            logw_lit[k]
        push!(terms, :($logw_k + $lpdf))
        push!(distributions, _evidence_endpoint(lpdf))
        push!(logweights, logw_k)
    end
    cell = foldl((a, b) -> :(logaddexp($a, $b)), terms)
    function tail(b, upper)
        pieces = map(eachindex(distributions)) do k
            dist = distributions[k]
            t = if f === BernoulliLogitFam
                upper ? :($b < 0 ? 1.0 : ($b >= 1 ? 0.0 : $dist.cdf(true) - $dist.cdf(false))) :
                    :($b < 0 ? 0.0 : ($b >= 1 ? 1.0 : $dist.cdf(false)))
            else
                upper ? :($dist.ccdf($b)) : :($dist.cdf($b))
            end
            :(exp($(logweights[k])) * $t)
        end
        foldl((a, b) -> :($a + $b), pieces)
    end
    cell = _wrap_evidence!(r, plan, pre, inputs, cell;
        cdf=b -> tail(b, false), ccdf=b -> tail(b, true))
    if !isempty(parameter_guards)
        valid = foldl((a, b) -> :($a && $b), parameter_guards)
        shared = r.weights === nothing && r.range === nothing && r.mi_jobs === nothing ?
            _shared_mixture_guard(valid, inputs, plan, pre) : nothing
        if shared !== nothing
            # Compute model-level support once; the observation cell still
            # takes a lazy branch, retaining its own native/compiled loop.
            support = Symbol(:_ppl_mix_valid_, r.label)
            push!(pre, :($support::Bool = $shared))
            valid = _thread_ref!(inputs, support)
        end
        cell = :($valid ? $cell : -Inf)
    end
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

function _shared_mixture_guard(valid, inputs, plan, pre)
    refs = _expr_value_symbols(valid)
    replacements = Dict{Symbol,Symbol}()
    for (i, input) in enumerate(inputs)
        ref = _dovar(i)
        ref in refs || continue
        input isa Symbol || return nothing
        scalar = any(p -> p.name === input, plan.parameters) ||
            get(plan.columns, input, nothing) isa Number ||
            any(p -> _lp_name(p) === input &&
                _predictor_value_type(plan, p) === :Number, plan.predictors) ||
            any(pre) do st
                st.head === :(=) && st.args[1] isa Expr &&
                    st.args[1].head === :(::) && st.args[1].args == Any[input, :Number]
            end
        scalar || return nothing
        replacements[ref] = input
    end
    length(replacements) == length(refs) || return nothing
    return _subst_syms(valid, replacements)
end

# One mixture location slot → `(ref, is_lp)`: a predictor yields its LP
# node (link-space); a sampled parameter yields its name (threaded scalar
# at the use site); a literal yields its Float64 (inlined). Threading
# happens at the use site via `_thread_ref!`, never here: pre-statements
# need layout-legal refs (nodes, params, literals), not plate do-vars.
function _mixture_loc_ref(r::LikelihoodSpec, plan::StructuralPlan, k::Int)
    loc = r.mixture_locs[k]
    loc isa Real && return Float64(loc), false
    if any(p -> p.name === loc, plan.predictors)
        return _lp_name(_predictor(plan, loc)), true
    end
    return loc, false
end

# One mixture component's scalar log-density: the single-family endpoint
# spelling over that component's slots (per-component precompute labels).
function _mixture_component_lpdf(f::LikelihoodFamily, r::LikelihoodSpec,
        plan::StructuralPlan, pre::Vector{Expr}, inputs::Vector{Any}, k::Int,
        klab::Symbol, locref, is_lp::Bool, yv::Symbol, yref, nref;
        normal_params::Vector{Any} = Any[])
    if f === BernoulliLogitFam && is_lp && r.link === LogitLink
        eta = _thread_ref!(inputs, locref)
        return :(bernoulli(; logit = $eta).logpdf($yref))
    elseif f === PoissonLogFam && is_lp && r.link === LogLink
        eta = _thread_ref!(inputs, locref)
        return :(poisson(; log_rate = $eta).logpdf($yv))
    elseif f === BinomialLogitFam && is_lp && r.link === LogitLink
        eta = _thread_ref!(inputs, locref)
        return :(binomial(; n = $nref, logit = $eta).logpdf($yv))
    end
    value = locref
    if is_lp && r.link !== IdentityLink
        value = _mu_name(klab)
        typ = _predictor_value_type(plan, _predictor(plan, r.mixture_locs[k]))
        push!(pre, :($value::$typ = $(_inverse_link_expr(r.link, locref))))
    end
    locv = _thread_ref!(inputs, value)
    if f === GaussianFam
        sarg = _scale_use_plate_arg(r, plan, pre, r.mixture_scales[k], klab)
        sref = _thread_ref!(inputs, sarg)
        append!(normal_params, (locv, sref))
        return :($sref > 0 ? normal($locv, $sref).logpdf($yv) : -Inf)
    elseif f === BernoulliLogitFam
        return :((($locv >= 0) & ($locv <= 1)) ? bernoulli($locv).logpdf($yref) : -Inf)
    elseif f === PoissonLogFam
        return :($locv >= 0 ? poisson($locv).logpdf($yv) : -Inf)
    elseif f === BinomialLogitFam
        return :((($locv >= 0) & ($locv <= 1)) ? binomial($nref, $locv).logpdf($yv) : -Inf)
    elseif f === NegativeBinomial2Fam
        sarg = _scale_use_plate_arg(r, plan, pre, r.mixture_scales[k], klab)
        phiref = _thread_ref!(inputs, sarg)
        return :($phiref > 0 && $locv >= 0 ?
            negative_binomial2($locv, $phiref).logpdf($yv) : -Inf)
    elseif f === GammaLogFam
        sarg = _scale_use_plate_arg(r, plan, pre, r.mixture_scales[k], klab)
        aref = _thread_ref!(inputs, sarg)
        return :($aref > 0 && $locv > 0 ? gamma($aref, $aref / $locv).logpdf($yv) : -Inf)
    elseif f === BetaLogitFam
        sarg = _scale_use_plate_arg(r, plan, pre, r.mixture_scales[k], klab)
        kap = sarg isa Symbol ? sarg : Float64(sarg)
        a = _shape_a_name(klab)
        b = _shape_b_name(klab)
        push!(pre, :($a = $value .* $kap), :($b = (1 .- $value) .* $kap))
        avv = _thread_ref!(inputs, a)
        bvv = _thread_ref!(inputs, b)
        return :($avv > 0 && $bvv > 0 ? beta($avv, $bvv).logpdf($yv) : -Inf)
    end
    throw(ContractValidationError("[generator] mixture over $f has no cell emitter"))
end

# A GLM-object response: one fused constructed-endpoint application
# over the whole column (no plate — the object owns eta). The
# intercept-free design matrix gains its ones column from the bound
# row count (data-only, folds at prepare) and the split coefficients
# rejoin as `beta_full = [alpha; beta]` (the validated P2 spelling).
function _glm_object_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol,
        pw::Symbol)
    if r.evidence.kind !== :none || haskey(plan.columns, _observed_mask_name(r.response))
        lp = Symbol(:_ppl_glm_eta_, r.label)
        pre = Expr[:($lp = $(r.glm_alpha) .+ $(r.predictor) * $(r.glm_beta))]
        inputs = Any[r.response, lp]
        yv, etav = _dovar(1), _dovar(2)
        base = if r.family === NormalIDGLMFam
            sv = _thread_ref!(inputs, r.scale)
            :(normal($etav, $sv).logpdf($yv))
        elseif r.family === BernoulliLogitGLMFam
            :(bernoulli(; logit=$etav).logpdf($yv == 1))
        else
            :(poisson(; log_rate=$etav).logpdf($yv))
        end
        cell = _wrap_evidence!(r, plan, pre, inputs, base)
        return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
    end
    obj = r.family === NormalIDGLMFam ? :normal_id_glm :
        r.family === BernoulliLogitGLMFam ? :bernoulli_logit_glm :
        :poisson_log_glm
    y, X = r.response, r.predictor
    yf = _yfloat_name(r.label)
    xaug = Symbol(:_ppl_glm_X_, r.label)
    bfull = Symbol(:_ppl_glm_b_, r.label)
    yconv = r.family === NormalIDGLMFam ? :Float64 : :Int
    s = r.scale isa Symbol ? r.scale : :(Float64($(r.scale)))
    call = r.family === NormalIDGLMFam ?
        :($obj($xaug, $bfull, $s).pointwise($yf)) :
        :($obj($xaug, $bfull).pointwise($yf))
    design = r.mi_jobs === nothing ? X : Expr(:ref, X, r.mi_jobs, :(:))
    n = r.mi_jobs === nothing ? _response_rows_source(plan, r) : Expr(:call, :length, r.mi_jobs)
    return Expr[
        :($yf = $yconv.($y)),
        :($xaug = hcat(ones($n), $design)),
        :($bfull = [$(r.glm_alpha); $(r.glm_beta)]),
        :($pw = $call),
        :($node::Float64 = sum($pw)),
    ]
end

_predictor(plan::StructuralPlan, name::Symbol) =
    only(p for p in plan.predictors if p.name === name)

# A scan-state or per-cell (plate) latent vector read directly as the
# location: `r.predictor` names that vector, not a PredictorSpec.
_is_latent_location(r::LikelihoodSpec, plan::StructuralPlan) =
    any(s -> r.predictor in s.states, plan.scans) || _is_plate_param(plan, r.predictor)

# The per-observation location node feeding a response's likelihood plate: a
# latent vector fed directly (its own name — the layout view), or a linear
# predictor's `_ppl_lp_<name>` node otherwise.
function _location_node(r::LikelihoodSpec, plan::StructuralPlan)
    _is_latent_location(r, plan) && return r.predictor
    return _lp_name(_predictor(plan, r.predictor))
end

# The value shape of a response's location (see `_predictor_value_type`).
_location_value_type(r::LikelihoodSpec, plan::StructuralPlan) =
    _is_latent_location(r, plan) ? :AbstractVector :
        _predictor_value_type(plan, _predictor(plan, r.predictor))

# Entry count of a latent vector: a scan trajectory or a plate latent.
function _latent_rows(plan::StructuralPlan, name::Symbol)
    for s in plan.scans
        name in s.states && return _scan_length(plan, s)
    end
    i = findfirst(p -> p.name === name, plan.plate_parameters)
    return i === nothing ? nothing : _plate_rows(plan, plan.plate_parameters[i])
end

# The fused whole-vector likelihoods (`dot(y, η)`, `sum(exp, η)`) need one
# location entry per response entry. A latent vector keeps its own length and
# broadcasts as a Julia array does (one entry stretches over every row); the
# plate path evaluates that, so fusion requires equal lengths.
function _wholevec_location(r::LikelihoodSpec, plan::StructuralPlan)
    y = get(plan.columns, r.response, nothing)
    rows = y isa AbstractVector ? length(y) : nothing
    _is_latent_location(r, plan) && return _latent_rows(plan, r.predictor) == rows
    pred = _predictor(plan, r.predictor)
    _broadcast_affine(plan, pred) && return false
    return all(pred.terms) do t
        t.kind === LatentTerm || return true
        n = _latent_rows(plan, only(t.columns))
        return n === nothing || n == rows
    end
end

# A bare sampled-parameter location (constrained-scale, no link inversion):
# `r.predictor` names a scalar parameter, not a PredictorSpec.
_is_bare_param_location(r::LikelihoodSpec, plan::StructuralPlan) =
    any(p -> p.name === r.predictor, plan.parameters)

# The predictor emits its raw Julia value; only this parameter use owns
# the inverse. Bare sampled locations already carry constrained values.
function _response_value_node!(pre::Vector{Expr}, r::LikelihoodSpec,
        plan::StructuralPlan)
    _is_bare_param_location(r, plan) && return r.predictor
    lp = _location_node(r, plan)
    r.link === IdentityLink && return lp
    value = _mu_name(r.label)
    rhs = _inverse_link_expr(r.link, lp)
    typ = _location_value_type(r, plan)
    push!(pre, :($value::$typ = $rhs))
    return value
end

_dovar(i::Int) = Symbol(:_ppl_c, i)
_pw_name(label::Symbol) = Symbol(:_ppl_pw_, label)

# `pointwise = plate(inputs...) do dovars...; cell; end` + scalar sum node.
# A plate must be a whole recipe RHS (never nested under `sum`), and the
# do-block body carries a LineNumberNode or the cell types as Any. Response
# `y` and predictor `lp` are always inputs 1-2 (`_ppl_c1/_ppl_c2`).
function _plate_sum_stmts(pointwise::Symbol, node::Symbol, inputs::Vector{Any},
        cell::Union{Expr,Vector{Expr}})
    dovars = [_dovar(i) for i in eachindex(inputs)]
    body = Expr(:block, LineNumberNode(0, :generator),
        (cell isa Expr ? (cell,) : cell)...)
    lambda = Expr(:(->), Expr(:tuple, dovars...), body)
    doex = Expr(:do, Expr(:call, :plate, inputs...), lambda)
    return Expr[:($pointwise = $doex), :($node::Float64 = sum($pointwise))]
end

# Case-A `mi()` gather naming, per response (`_ppl_mi_<label>_<ref>`):
# twin responses sharing one predictor gather through distinct nodes.
_mi_gather_name(label::Symbol, ref::Symbol) = Symbol(:_ppl_mi_, label, :_, ref)

# Gather a computed full-length node by `Jobs` (an lp/rate/shape/scale
# node the emitter created), returning the short node. Scalars broadcast.
# The gather is its own short plate over `Jobs` with the source `Ref`'d
# (the ordinal `c[yv]` per-lane-gather precedent): a caller-level fancy
# `node[Jobs]` does not trace under Reactant (`TracedRArray[Vector{Int}]`
# shape-inference failure), while per-lane scalar gathers do.
function _mi_gather_node!(pre::Vector{Expr}, jobs::Symbol, node::Symbol,
        label::Symbol)
    g = _mi_gather_name(label, node)
    jv, rf = _dovar(1), _dovar(2)
    getter = GlobalRef(@__MODULE__, :_mi_row_value)
    body = Expr(:block, LineNumberNode(0, :generator), :($getter($rf, $jv)))
    lambda = Expr(:(->), Expr(:tuple, jv, rf), body)
    doex = Expr(:do, Expr(:call, :plate, jobs, :(Ref($node))), lambda)
    push!(pre, :($g = $doex))
    return g
end

# Thread a Symbol ref as a plate input (returning its do-var); Real
# literals inline (`as_int` for Poisson bounds — validated integer-valued).
function _thread_ref!(inputs::Vector{Any}, ref, as_int::Bool = false)
    if ref isa Symbol
        push!(inputs, ref)
        return _dovar(length(inputs))
    end
    return as_int ? Int(ref) : Float64(ref)
end

# A weighted cell. A cell that is a lazy branch keeps the branch as its own
# cell statement (`_ppl_arm = c ? a : b`) with the weight applied after it:
# only a top-level branch is visible to plate lowering, which splits the
# lanes when the condition reads bound data only.
function _weighted_cell(wv, cell::Expr)
    cell.head === :if || return :($wv * $cell)
    return Expr[:(_ppl_arm::Float64 = $cell), :($wv * _ppl_arm)]
end

function _gaussian_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    pre = Expr[]
    lp = _response_value_node!(pre, r, plan)
    sarg = _scale_plate_arg(r, plan, pre)
    inputs = Any[y, lp]
    yv, lpv = _dovar(1), _dovar(2)
    sref = _thread_ref!(inputs, sarg)
    base = :(normal($lpv, $sref).logpdf($yv))
    cell = base
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    cell = :($sref > 0 ? $cell : -Inf)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

_gaussian_cell(kind::Symbol, base::Expr, yv::Symbol, lb, ub, lpv::Symbol, sref) =
    _evidence_cell(Val(kind), base, yv, lb, ub,
        b -> :(normal($lpv, $sref).cdf($b)),
        b -> :(normal($lpv, $sref).ccdf($b)), false)

# Student-t plate: the Gaussian shape with a df argument — validation
# guarantees `nu` (a sampled name, a literal, or a predictor-fed
# per-observation nu) and sigma; evidence arms mirror the Gaussian
# clamp law over the `student_t` cdf (`_student_cell`), plus optional
# weights. A predictor-fed nu binds its own `_ppl_sc_<label>_nu` node
# (the `_nu`-suffixed label cannot collide with any `<lhs>_resp` scale
# node), so scale and nu predictors coexist.
function _student_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    pre = Expr[]
    lp = _response_value_node!(pre, r, plan)
    sarg = _scale_plate_arg(r, plan, pre)
    nuarg = _scale_use_plate_arg(r, plan, pre, r.nu, Symbol(r.label, :_nu))
    inputs = Any[y, lp]
    yv, lpv = _dovar(1), _dovar(2)
    sref = _thread_ref!(inputs, sarg)
    nuv = _thread_ref!(inputs, nuarg)
    base = :(student_t($nuv, $lpv, $sref).logpdf($yv))
    cell = base
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    cell = :($sref > 0 && $nuv > 0 ? $cell : -Inf)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# LogNormal likelihood (Stan `lognormal_lpdf` mirror): the cell calls
# the prior-proven `lognormal` distribution endpoint
# (`LOGNORMAL_KERNEL_SOURCE`, Distributions.jl `LogNormal(mu, sigma)`
# order) — no closed-form respelling, so SB parity is by shared
# operation order. Mu reads its argument value; sigma threads scalar
# or per observation via `_scale_plate_arg`, including a modeled sigma.
# The endpoint's lazy `y > 0` guard returns
# -Inf off support; `y` is bound data so preparation splits the plate
# per taken arm and no backend receives the branch. The shared evidence
# algebra wraps its density and tails before optional weights.
function _lognormal_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    pre = Expr[]
    lp = _response_value_node!(pre, r, plan)
    sarg = _scale_plate_arg(r, plan, pre)
    inputs = Any[y, lp]
    yv, lpv = _dovar(1), _dovar(2)
    sref = _thread_ref!(inputs, sarg)
    cell = :(lognormal($lpv, $sref).logpdf($yv))
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    cell = :($sref > 0 ? $cell : -Inf)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Bernoulli Bool normalization (snag `bernoulli-int-la-78487520`): a
# non-Bool validated 0/1 response column reads through a NAMED
# `_ppl_yb_<label>` recipe instead of an in-cell `yv != 0` comparison.
# The recipe is data-only so `bound=` folds it exactly once (the
# `_ppl_yf_` precedent — zero per-eval alloc, verified); the plate then
# reads Bool lanes and the endpoint's lazy branch never sees a computed
# comparison, which misdifferentiates under native Enzyme in some plate
# shapes (observed: -10.10 vs true -4.97, NaN on perturbation; the same
# math inline or over Bool lanes is exact). Dense `Vector{Bool}`:
# broadcast comparison yields a BitVector and BitArrays are
# overlay-hostile (the kernel-plate twin precedent). Bool columns skip
# the recipe. Returns the plate input symbol and the cell lane ref.
_ybool_name(label::Symbol) = Symbol(:_ppl_yb_, label)

# A column whose element type is exactly Bool. Type tests rather than an
# `eltype` call keep this check concretely inferred.
_is_bool_column(col) = col isa Bool || col isa AbstractArray{Bool}

# Count endpoints take integer values. Keep the caller's Bool column available
# to every other reader and form a separate zero/one integer value for the
# density. This data-only recipe is shared by native and compiled execution.
function _count_yplate!(pre::Vector{Expr}, plan::StructuralPlan,
        y::Symbol, label::Symbol)
    _is_bool_column(plan.columns[y]) || return y
    yi = Symbol(:_ppl_yi_, label)
    push!(pre, :($yi = Int.($y)))
    return yi
end

function _bernoulli_yplate!(pre::Vector{Expr}, plan::StructuralPlan,
        y::Symbol, label::Symbol, yv::Symbol)
    col = plan.columns[y]
    _is_bool_column(col) && return y, yv
    yb = _ybool_name(label)
    push!(pre, :($yb = Array{Bool}($y .!= 0)))
    return yb, yv
end

# Student-t evidence arms: the Gaussian clamp law over the `student_t`
# cdf (continuous bounds, non-strict censored arms).
_student_cell(kind::Symbol, base::Expr, yv::Symbol, lb, ub, nuv, lpv::Symbol, sref) =
    _evidence_cell(Val(kind), base, yv, lb, ub,
        b -> :(student_t($nuv, $lpv, $sref).cdf($b)),
        b -> :(student_t($nuv, $lpv, $sref).ccdf($b)), false)

function _bernoulli_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    yv = _dovar(1)
    pre = Expr[]
    yin, yref = _bernoulli_yplate!(pre, plan, y, r.label, yv)
    inputs = Any[yin]
    if _is_bare_param_location(r, plan) || r.link !== LogitLink
        p = _response_value_node!(pre, r, plan)
        pv = _thread_ref!(inputs, p)
        cell = :((($pv >= 0) & ($pv <= 1)) ? bernoulli($pv).logpdf($yref) : -Inf)
    else
        lp = _location_node(r, plan)
        push!(inputs, lp)
        etav = _dovar(2)
        cell = :(bernoulli(; logit = $etav).logpdf($yref))
    end
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Fused whole-vector Poisson-log likelihood (base case: no evidence, no
# weights). Value-identical to `Σ poisson(; log_rate=ηᵢ).logpdf(yᵢ)` for the
# contract's nonnegative-integer `y`: `Σ yᵢ·ηᵢ − Σ exp(ηᵢ) − C`, with
# `C = Σ loggamma(yᵢ+1)` a NAMED data-only recipe over the bound response
# (`_ppl_lfact_<label>`), never on the gradient tape. `bound=` folds it to a
# constant, and a binding with other counts computes its own (it is not a
# build-time literal). `_ppl_yf_<label> = Float64.(y)` is a NAMED
# recipe so `bound=` folds it to a constant Float vector (no per-eval alloc)
# AND gives the fused `dot` a Float operand (Reactant `dot_general` type match).
# `sum(exp, η)` reduces without materialising the intermediate. Gradient stays
# ordinary Enzyme/Reactant AD — no analytic adjoint.
_yfloat_name(label::Symbol) = Symbol(:_ppl_yf_, label)

# Fused whole-vector Bernoulli-logit likelihood (base case). Logit-form log-mass
# `Σ yᵢ·ηᵢ − Σ log1pexp(ηᵢ)` — no data-only normalizer, a plain fused reduction.
# Value-identical to `Σ bernoulli(; logit=ηᵢ).logpdf(yᵢ)`; same folded-`_ppl_yf`
# and `sum(f, x)` treatment as the Poisson form. Gradient stays ordinary AD.
function _bernoulli_wholevec_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    yf = _yfloat_name(r.label)
    return Expr[
        _bernoulli_plate_stmts(r, plan, node, _pw_name(r.label))[1:end-1]...,
        :($yf = Float64.($y)),
        :($node::Float64 = dot($yf, $lp) - sum(log1pexp, $lp)),
    ]
end

function _poisson_wholevec_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    yf = _yfloat_name(r.label)
    cterm = Symbol(:_ppl_lfact_, r.label)
    return Expr[
        _poisson_plate_stmts(r, plan, node, _pw_name(r.label))[1:end-1]...,
        :($yf = Float64.($y)),
        :($cterm = sum(loggamma.($yf .+ 1.0))),
        :($node::Float64 = dot($yf, $lp) - sum(exp, $lp) - $cterm),
    ]
end

function _poisson_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    pre = Expr[]
    y = r.response
    pre = Expr[]
    inputs = Any[y]
    yv = _dovar(1)
    if _is_bare_param_location(r, plan) || r.link !== LogLink
        rate = _response_value_node!(pre, r, plan)
        ratev = _thread_ref!(inputs, rate)
        base = :(poisson($ratev).logpdf($yv))
        cell = :($ratev >= 0 ? $base : -Inf)
    else
        lp = _location_node(r, plan)
        push!(inputs, lp)
        etav = _dovar(2)
        base = :(poisson(; log_rate = $etav).logpdf($yv))
        cell = base
    end
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# ZIP plate: the Poisson-plate shape with a zero-inflation argument —
# validation guarantees `zi` (a sampled name, a literal, or a
# predictor-fed zi submodel). The shared evidence algebra wraps the
# `zero_inflated_poisson` endpoint before optional weights. A zi
# predictor binds its constrained vector once (`_ppl_sc_`, the
# scale-predictor precedent — linked per use at the contract gate) and the
# plate iterates it per cell.
function _zip_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    pre = Expr[]
    ziarg = _scale_use_plate_arg(r, plan, pre, r.zi, r.label)
    loc = r.link === LogLink ? lp : _response_value_node!(pre, r, plan)
    inputs = Any[y, loc]
    yv, etav = _dovar(1), _dovar(2)
    ziref = _thread_ref!(inputs, ziarg)
    # All-keyword: the object constructor cannot mix positional and named
    # owner bindings (matches the `:observed/:log_rate/:zi` HAVE ports).
    cell = if r.link === LogLink
        :((($ziref >= 0) & ($ziref <= 1)) ?
            zero_inflated_poisson(; log_rate = $etav, zi = $ziref).logpdf($yv) : -Inf)
    else
        :($etav >= 0 && (($ziref >= 0) & ($ziref <= 1)) ?
            zero_inflated_poisson($etav, $ziref).logpdf($yv) : -Inf)
    end
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Lower-side cdf argument for the inclusive discrete cdf: the mass below
# lb is F(lb - 1), so TRUNCATED low arms shift their lower argument by one
# (interval cells stay unshifted — the response is the open lower
# endpoint). The shared evidence algebra distinguishes real-valued
# inclusive and exclusive endpoints with floor/ceil.

_poisson_cell(kind::Symbol, base::Expr, yv::Symbol, lb, ub, etav::Symbol) =
    _evidence_cell(Val(kind), base, yv, lb, ub,
        b -> :(poisson(; log_rate = $etav).cdf($b)),
        b -> :(poisson(; log_rate = $etav).ccdf($b)), true)

# Hurdle-Poisson likelihood (SB `hurdle_poisson` mirror): a zero part
# plus a zero-truncated Poisson positive part. Per cell:
# `y == 0 ? log(p0) : log1p(-p0) + poisson_logpdf - log(1 - e^-λ)`.
# The Poisson factor reuses the `poisson(; log_rate)` endpoint HAVE
# (the Poisson-plate precedent). The truncation correction is the
# closed form `log(-expm1(-λ))` (Poisson cdf(0) is exactly e^-λ; SB
# subtracts `poisson_lccdf(0 | λ)` — the same quantity) — NOT the
# `.cdf(0)` endpoint, whose `gamma_inc` has no Reactant tracing rule
# (`MethodError` on compile; the truncated/censored Poisson cells
# carry the same gap). The `y == 0` select is lazy, so the positive
# count's density and truncation correction remain inactive at zero.
# p_zero threads scalar or via the `_ppl_sc_` node
# (the scale-predictor precedent — a hurdle p_zero predictor is
# linked per use at the contract gate). The shared evidence algebra supplies
# the hurdle tails before optional weights. No whole-vector fusion yet: both parts carry
# per-cell parameter-dependent work (the truncation correction varies
# with λ even for scalar p_zero) — a perf-lane follow-up, not this
# slice.
function _hurdle_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    pre = Expr[]
    sarg = _scale_plate_arg(r, plan, pre)
    loc = r.link === LogLink ? lp : _response_value_node!(pre, r, plan)
    inputs = Any[y, loc]
    yv, etav = _dovar(1), _dovar(2)
    p0v = _thread_ref!(inputs, sarg)
    rate = r.link === LogLink ? :(exp($etav)) : etav
    base = r.link === LogLink ? :(poisson(; log_rate = $etav).logpdf($yv)) :
        :(poisson($etav).logpdf($yv))
    trunc = :(log(-expm1(-$rate)))
    cell = :($yv == 0 ? log($p0v) : log1p(-$p0v) + $base - $trunc)
    cell = :($rate > 0 && (($p0v >= 0) & ($p0v <= 1)) ? $cell : -Inf)
    distribution = r.link === LogLink ? :(poisson(; log_rate = $etav)) :
        :(poisson($etav))
    cell = _wrap_evidence!(r, plan, pre, inputs, cell;
        cdf = b -> :($b < 0 ? 0.0 : ($b == 0 ? $p0v :
            $p0v + (1 - $p0v) * ($distribution.cdf($b) - exp(-$rate)) / (-expm1(-$rate)))),
        ccdf = b -> :($b < 0 ? 1.0 : (1 - $p0v) * $distribution.ccdf($b) / (-expm1(-$rate))))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Inverse-Gaussian / Wald likelihood (SB `brm_inverse_gaussian_lpdf`
# mirror): the closed form spelled operation-for-operation —
# `(log(λ) - (log2π + 3*log(y)) - λ*(y-μ)²/(μ²*y))/2` — with the SB
# `log2π` literal `1.8378770664093456` (`Float64(Distributions.log2π)`
# exactly). The mean precomputes outside the cell (`_ppl_mu_`, the NB2
# precedent — a computed `exp` constructor arg miscompiles the Enzyme
# pullback); lambda threads via `_scale_plate_arg` (scalar
# sampled/literal/assignment/column, or a log-link predictor — the
# VonMises-kappa precedent, linked per use at the contract gate).
# The `y > 0` guard is lazy `?:` (the DK gamma precedent, not eager
# `ifelse`); `y` is bound data so preparation splits the plate per
# taken arm and no backend receives the branch. μ/λ positivity is
# guarded outside the evidence wrapper, including raw live values.
# Evidence uses
# the shared wrapper algebra and stable scaled-erfc tails before optional
# weights. No whole-vector fusion yet
# (the `3*log(y)` normalizer is data-only and would hoist) — a
# perf-lane follow-up, not this slice.
function _ig_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    mu = _mu_name(r.label)
    typ = _location_value_type(r, plan)
    pre = Expr[:($mu::$typ = $(_inverse_link_expr(r.link, lp)))]
    sarg = _scale_plate_arg(r, plan, pre)
    inputs = Any[y, mu]
    yv, muv = _dovar(1), _dovar(2)
    lamv = _thread_ref!(inputs, sarg)
    base = :((log($lamv) - (1.8378770664093456 + 3.0 * log($yv)) -
        $lamv * ($yv - $muv) * ($yv - $muv) / ($muv * $muv * $yv)) / 2.0)
    cell = :($yv > 0 ? $base : -Inf)
    cell = _wrap_evidence!(r, plan, pre, inputs, cell;
        cdf = b -> :(rk_inverse_gaussian_tail($muv, $lamv, $b, false)),
        ccdf = b -> :(rk_inverse_gaussian_tail($muv, $lamv, $b, true)))
    cell = :($lamv > 0 && $muv > 0 ? $cell : -Inf)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Exponential likelihood (SB `exponential_lpdf` mirror): the existing
# `exponential` distribution-kernel endpoint (prior-proven, Enzyme-covered),
# spliced per cell with the log-link mean (the Gamma-plate precedent).
# The mean precomputes outside the cell (`_ppl_mu_`, the NB2 precedent —
# a computed `exp` constructor arg miscompiles the Enzyme pullback).
# The `y >= 0` guard lives in the endpoint (`ifelse`, DK-owned); `y` is
# bound data so preparation splits the plate per taken arm and no
# backend receives the branch. μ positivity is guarded outside evidence,
# including raw live values. The family takes no scale
# (Poisson-shaped; SB's rate is `1 ./ mu`, carried inside the kernel's
# `-log(μ) - y/μ` spelling). Evidence uses the shared wrapper algebra
# before optional weights.
# No whole-vector fusion yet — a perf-lane follow-up, not this slice.
function _exponential_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    mu = _mu_name(r.label)
    typ = _location_value_type(r, plan)
    pre = Expr[:($mu::$typ = $(_inverse_link_expr(r.link, lp)))]
    inputs = Any[y, mu]
    yv, muv = _dovar(1), _dovar(2)
    cell = :($muv > 0 ? exponential($muv).logpdf($yv) : -Inf)
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Von-Mises likelihood (SB `brm_von_mises_lpdf` mirror): the branch
# structure spelled exactly — `kappa <= 0 → -Inf` outermost, then the
# support guard (exact: `y` outside inclusive `[mu - pi, mu + pi]`;
# circular: `y` outside half-open `[lo, hi)`), then the Stan-native
# in-support value `-log2π - log(I0(κ)) + κ*cos(y - mu)` with the SB
# `log2π` literal `1.8378770664093456` and the SB pi literal
# `3.141592653589793`. The circular mu wrap is the SB fmod spelling
# `lo + rem(rem(mu - lo, w) + w, w)`. The identity link needs no mu
# precompute (the lp node IS mu); kappa threads via `_scale_plate_arg`
# (scalar sampled/literal/assignment/column, or a log-link predictor —
# the hurdle precedent, linked per use at the contract gate). All guards are
# lazy `?:` (never eager `ifelse`); bound-data conditions (circular
# support, literal/column kappa) split the plate per taken arm at
# prepare, while live conditions (sampled/predictor kappa, exact
# moving support) keep their authored branch to the backend. Weights
# multiply the cell (the NB2 precedent). Evidence uses the shared wrapper algebra. No whole-vector fusion yet —
# a perf-lane follow-up, not this slice.
function _vonmises_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    pre = Expr[]
    lp = _response_value_node!(pre, r, plan)
    sarg = _scale_plate_arg(r, plan, pre)
    inputs = Any[y, lp]
    yv, muv = _dovar(1), _dovar(2)
    kapv = _thread_ref!(inputs, sarg)
    if r.interval === nothing
        base = :(-1.8378770664093456 - log(besseli(0, $kapv)) +
            $kapv * cos($yv - $muv))
        inner = :($yv > $muv + 3.141592653589793 ? -Inf : $base)
        sup = :($yv < $muv - 3.141592653589793 ? -Inf : $inner)
    else
        lo = _thread_ref!(inputs, r.interval[1])
        hi = _thread_ref!(inputs, r.interval[2])
        w = :($hi - $lo)
        wmu = :($lo + rem(rem($muv - $lo, $w) + $w, $w))
        base = :(-1.8378770664093456 - log(besseli(0, $kapv)) +
            $kapv * cos($yv - $wmu))
        inner = :($yv >= $hi ? -Inf : $base)
        support = :($yv < $lo ? -Inf : $inner)
        valid = :(isfinite($lo) && isfinite($hi) && $lo < $hi &&
            abs($w - $(2 * Float64(pi))) <= $(8eps(Float64) * 2 * Float64(pi)))
        sup = :($valid ? $support : -Inf)
    end
    cell = :($kapv <= 0 ? -Inf : $sup)
    ecdf = if r.interval === nothing
        b -> :(rk_von_mises_cdf($muv, $kapv, $b))
    else
        lo, hi = r.interval
        b -> :(rk_von_mises_periodic_cdf($muv, $kapv, $b, $lo, $hi))
    end
    cell = _wrap_evidence!(r, plan, pre, inputs, cell; cdf=ecdf, ccdf=b -> :(1.0 - $(ecdf(b))))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

function _binomial_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    pre = Expr[]
    y = r.response
    pre = Expr[]
    yin = _count_yplate!(pre, plan, y, r.label)
    inputs = Any[yin]
    yv = _dovar(1)
    if _is_bare_param_location(r, plan) || r.link !== LogitLink
        nref = _thread_ref!(inputs, r.trials, true)
        p = _response_value_node!(pre, r, plan)
        pv = _thread_ref!(inputs, p)
        cell = :((($pv >= 0) & ($pv <= 1)) ? binomial($nref, $pv).logpdf($yv) : -Inf)
    else
        lp = _location_node(r, plan)
        push!(inputs, lp)
        etav = _dovar(2)
        nref = _thread_ref!(inputs, r.trials, true)
        # All-keyword: the object constructor cannot mix positional and named
        # owner bindings (matches the `:observed/:n/:logit` HAVE ports).
        cell = :(binomial(; n = $nref, logit = $etav).logpdf($yv))
    end
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Prob-space Binomial (SB `binomial(n, theta)` with a Beta prior — the Rate
# family): the location is the Beta parameter name itself (no LP node —
# the Categorical `r.predictor`-as-symbol precedent), threaded scalar
# through the plate against the positional prob-space kernel.
function _binomial_prob_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    pre = Expr[]
    y = r.response
    pre = Expr[]
    yin = _count_yplate!(pre, plan, y, r.label)
    inputs = Any[yin]
    yv = _dovar(1)
    nref = _thread_ref!(inputs, r.trials, true)
    pref = _thread_ref!(inputs, any(p -> p.name === r.predictor, plan.predictors) ?
        _lp_name(_predictor(plan, r.predictor)) : r.predictor)
    cell = :(binomial($nref, $pref).logpdf($yv))
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Zero-inflated-Binomial plate (the ZIB head): prob-space scalar p (the
# BinomialProb precedent) plus the zi slot (the ZIP precedent). The
# kernel endpoint takes all three positionally; weights multiply the
# cell (the NB2 precedent).
function _zib_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    pre = Expr[]
    y = r.response
    pre = Expr[]
    p = _response_value_node!(pre, r, plan)
    ziarg = _scale_use_plate_arg(r, plan, pre, r.zi, Symbol(r.label, :_zi))
    yin = _count_yplate!(pre, plan, y, r.label)
    inputs = Any[yin]
    yv = _dovar(1)
    nref = _thread_ref!(inputs, r.trials, true)
    pref = _thread_ref!(inputs, p)
    ziref = _thread_ref!(inputs, ziarg)
    cell = :((($pref >= 0) & ($pref <= 1)) && (($ziref >= 0) & ($ziref <= 1)) ?
        zero_inflated_binomial($nref, $pref, $ziref).logpdf($yv) : -Inf)
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Mean/rate vectors are precomputed statements (like `_ppl_lp_*`): plate
# cells take plain do-vars — a computed `exp` constructor arg miscompiles
# the Enzyme pullback (NB2 eta-gradient, found by test).
_mu_name(label::Symbol) = Symbol(:_ppl_mu_, label)
_rate_name(label::Symbol) = Symbol(:_ppl_rate_, label)

function _nb2_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    mu = _mu_name(r.label)
    typ = _location_value_type(r, plan)
    pre = Expr[:($mu::$typ = $(_inverse_link_expr(r.link, lp)))]
    sarg = _scale_plate_arg(r, plan, pre)
    inputs = Any[y, mu]
    yv, muv = _dovar(1), _dovar(2)
    phiref = _thread_ref!(inputs, sarg)
    cell = :(negative_binomial2($muv, $phiref).logpdf($yv))
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    cell = :($phiref > 0 && $muv >= 0 ? $cell : -Inf)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# NB1 likelihood (SB `neg_binomial` mirror): the successes-shape r
# precomputes outside the cell (`_ppl_r_`, the NB2 `_ppl_mu_`
# precedent — a computed `exp` constructor arg miscompiles the Enzyme
# pullback); the success probability p threads scalar, per-obs, or
# predictor-fed (linked per use, the hurdle precedent) via
# `_scale_plate_arg`. Weights multiply the cell (the
# NB2 precedent). No whole-vector fusion yet — a perf-lane follow-up,
# not this slice.
_nb1_r_name(label::Symbol) = Symbol(:_ppl_r_, label)

function _nb1_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    rr = _nb1_r_name(r.label)
    typ = _location_value_type(r, plan)
    pre = Expr[:($rr::$typ = $(_inverse_link_expr(r.link, lp)))]
    sarg = _scale_plate_arg(r, plan, pre)
    inputs = Any[y, rr]
    yv, rv = _dovar(1), _dovar(2)
    pref = _thread_ref!(inputs, sarg)
    cell = :(negative_binomial($rv, $pref).logpdf($yv))
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    cell = :($rv > 0 && (($pref >= 0) & ($pref <= 1)) ? $cell : -Inf)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Weibull likelihood (SB `weibull` mirror): the scale's `log_theta` HAVE
# route takes the link predictor directly (the Poisson/ZIP precedent —
# no `exp` precompute, no `log(exp())` round trip); the shape k threads
# scalar or per observation via `_scale_plate_arg`, including modeled
# shapes under their own links. Weights multiply the
# cell (the NB2 precedent). No whole-vector fusion yet — a perf-lane
# follow-up, not this slice.
function _weibull_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    pre = Expr[]
    sarg = _scale_plate_arg(r, plan, pre)
    theta = r.link === LogLink ? lp : _response_value_node!(pre, r, plan)
    inputs = Any[y, theta]
    yv, etav = _dovar(1), _dovar(2)
    kref = _thread_ref!(inputs, sarg)
    # All-keyword: the object constructor cannot mix positional and named
    # owner bindings (the ZIP precedent).
    cell = if r.link === LogLink
        :($kref > 0 ? weibull(; k = $kref, log_theta = $etav).logpdf($yv) : -Inf)
    else
        :($kref > 0 && $etav > 0 ? weibull($kref, $etav).logpdf($yv) : -Inf)
    end
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

function _gamma_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    pre = Expr[]
    mu = _response_value_node!(pre, r, plan)
    sarg = _scale_plate_arg(r, plan, pre)
    inputs = Any[r.response, mu]
    yv, muv = _dovar(1), _dovar(2)
    aref = _thread_ref!(inputs, sarg)
    rate = r.family === GammaValueFam ? :(1.0 / $muv) : :($aref / $muv)
    cell = :($aref > 0 && $muv > 0 ?
        gamma($aref, $rate).logpdf($yv) : -Inf)
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    if r.weights !== nothing
        cell = _weighted_cell(_thread_ref!(inputs, r.weights), cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]

end

# The scale argument a likelihood plate threads per cell: scalar scales
# (parameter, assignment, literal, raw data column) pass through untouched;
# a predictor-fed scale binds its constrained vector once
# (`_ppl_sc_<label>`, the `_ppl_mu_`/`_ppl_rate_` precompute precedent —
# the link inverts here, never inside the cell) and the plate iterates
# the node. Link inversion reuses the shared ordinary inverse-link
# mathematics used by generated arguments.
_sc_name(label::Symbol) = Symbol(:_ppl_sc_, label)

function _scale_plate_arg(r::LikelihoodSpec, plan::StructuralPlan, pre::Vector{Expr})
    return _scale_use_plate_arg(r, plan, pre, r.scale, r.label)
end

# One scale-slot use over a likelihood plate: scalar scales pass through
# untouched; a predictor-fed scale binds its constrained vector once
# (`_ppl_sc_<label>`) and the plate iterates the node. Mixture
# components pass their own use + per-component label.
function _scale_use_plate_arg(r::LikelihoodSpec, plan::StructuralPlan,
        pre::Vector{Expr}, s, label::Symbol)
    s isa ScalePredictorRef || return s
    pred = _predictor(plan, s.predictor)
    lp = _lp_name(pred)
    sc = _sc_name(label)
    rhs = _inverse_link_expr(s.link, lp)
    # The annotation is load-bearing for AD, not decoration: it proves the
    # plate input `:axis` statically, so `prepare` lowers the straight-line
    # plate form. Unannotated (metadata-`Any`) vector inputs lower with the
    # runtime `_authored_plate_is_axis` / `_plate_dependency_changed` guards,
    # whose form defeats Enzyme's static-activity analysis on some endpoint
    # bodies (NB2, found by test: silently wrong gradients). The eltype-free
    # array annotation retains matrix/tensor axes and integer offset LPs;
    # a parameter-only composed expression retains scalar metadata.
    typ = _predictor_value_type(plan, pred)
    push!(pre, :($sc::$typ = $rhs))
    return sc
end

# Linked values retain the raw predictor's broadcast shape. The explicit
# scalar/array type also lets plate AD determine activity without a runtime
# shape guard. Ordinary vector designs keep their existing annotation.
function _predictor_value_type(plan::StructuralPlan, pred::PredictorSpec)
    _broadcast_affine(plan, pred) || return :AbstractVector
    scalar = all(pred.terms) do t
        t.kind === InterceptTerm && return true
        t.kind in (ContinuousTerm, OffsetTerm) &&
            return all(c -> get(plan.columns, c, nothing) isa Number, t.columns)
        t.kind === ComposedTerm || return false
        isempty(t.columns) || return false
        return all(q -> _predictor_value_type(plan, _predictor(plan, q)) === :Number,
            t.options.subs)
    end
    return scalar ? :Number : :AbstractArray
end

# Inverse links are ordinary pure arithmetic in every slot. The same
# expression feeds primal, native AD and compiled AD.
_inverse_link_expr(link::LinkFunction, value) =
    _inverse_link_expr(Val(link), value)
_inverse_link_expr(::Val{IdentityLink}, value) = value
_inverse_link_expr(::Val{LogLink}, value) = :(exp.($value))
_inverse_link_expr(::Val{LogitLink}, value) = :(_ppl_logistic.($value))
_inverse_link_expr(::Val{ProbitLink}, value) = :(0.5 .* erfc.(-$value ./ sqrt(2)))
_inverse_link_expr(::Val{CloglogLink}, value) = :(-expm1.(-exp.($value)))

_prob_name(label::Symbol) = Symbol(:_ppl_p_, label)
_logitp_name(label::Symbol) = Symbol(:_ppl_logitp_, label)
_shape_a_name(label::Symbol) = Symbol(:_ppl_a_, label)
_shape_b_name(label::Symbol) = Symbol(:_ppl_b_, label)

# Bernoulli probit: Phi precompute as pure Base arithmetic (Gamma-pre
# pattern). Phi is 0.5*erfc(-z/sqrt(2)), exactly the `standard_normal.cdf`
# formula — inlined rather than broadcast through the endpoint object
# because Enzyme cannot differentiate the object-broadcast (runtime
# activity on the const kernel object). Positional-p cell (primary form).
function _bernoulli_probit_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    p = _prob_name(r.label)
    pre = Expr[:($p = 0.5 .* erfc.(-$lp ./ sqrt(2)))]
    yv, pv = _dovar(1), _dovar(2)
    yin, yref = _bernoulli_yplate!(pre, plan, y, r.label, yv)
    inputs = Any[yin, p]
    cell = :(bernoulli($pv).logpdf($yref))
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Bernoulli cloglog: pure-arithmetic p precompute, positional-p cell.
function _bernoulli_cloglog_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    p = _prob_name(r.label)
    pre = Expr[:($p = 1 .- exp.(-exp.($lp)))]
    yv, pv = _dovar(1), _dovar(2)
    yin, yref = _bernoulli_yplate!(pre, plan, y, r.label, yv)
    inputs = Any[yin, p]
    cell = :(bernoulli($pv).logpdf($yref))
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Binomial probit/cloglog: precompute p, then logit(p), and reuse the
# proven logit route — the binomial kernel's p port is unverified, while
# the (:n, :logit) route is what slice-1 Binomial emits.
function _binomial_probit_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    pre = Expr[]
    y = r.response
    lp = _location_node(r, plan)
    p = _prob_name(r.label)
    logitp = _logitp_name(r.label)
    pre = Expr[]
    yin = _count_yplate!(pre, plan, y, r.label)
    inputs = Any[yin, logitp]
    yv, lpv = _dovar(1), _dovar(2)
    nref = _thread_ref!(inputs, r.trials, true)
    cell = :(binomial(; n = $nref, logit = $lpv).logpdf($yv))
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., :($p = 0.5 .* erfc.(-$lp ./ sqrt(2))),
        :($logitp = log.($p) .- log1p.(-$p)),
        _plate_sum_stmts(pw, node, inputs, cell)...]
end

function _binomial_cloglog_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    pre = Expr[]
    y = r.response
    lp = _location_node(r, plan)
    p = _prob_name(r.label)
    logitp = _logitp_name(r.label)
    pre = Expr[]
    yin = _count_yplate!(pre, plan, y, r.label)
    inputs = Any[yin, logitp]
    yv, lpv = _dovar(1), _dovar(2)
    nref = _thread_ref!(inputs, r.trials, true)
    cell = :(binomial(; n = $nref, logit = $lpv).logpdf($yv))
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., :($p = 1 .- exp.(-exp.($lp))),
        :($logitp = log.($p) .- log1p.(-$p)),
        _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Beta mean-concentration: mu/a/b precomputes (Gamma-pre pattern), kappa
# by name (Symbol), inlined (literal), or bound once (`_ppl_sc_<label>`
# via `_scale_plate_arg` for a log-link predictor — the vector folds
# into the a/b precomputes through broadcast); cell needs only (y, a,
# b), so kappa is never a plate input. Positional beta cell (primary
# form).
function _beta_shape_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan,
        node::Symbol, pw::Symbol)
    pre = Expr[]
    inputs = Any[r.response, _location_node(r, plan)]
    y, alpha = _dovar(1), _dovar(2)
    beta_arg = _thread_ref!(inputs, _scale_plate_arg(r, plan, pre))
    valid = :(isfinite($alpha) && $alpha > 0 &&
        isfinite($beta_arg) && $beta_arg > 0)
    # The ordinary Beta density stays inside the domain branch. The
    # existing owned logbeta callable carries its generated derivative
    # rule; endpoint expansion cannot currently bind that global here.
    density = :(($alpha - 1) * log($y) +
        ($beta_arg - 1) * log1p(-$y) - logbeta($alpha, $beta_arg))
    density = _wrap_evidence!(r, plan, pre, inputs, density;
        cdf = b -> :(beta($alpha, $beta_arg).cdf($b)),
        ccdf = b -> :(beta($alpha, $beta_arg).ccdf($b)))
    cell = :($valid ? $density : -Inf)
    if r.weights !== nothing
        weight = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(weight, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

function _beta_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    mu = _mu_name(r.label)
    a = _shape_a_name(r.label)
    b = _shape_b_name(r.label)
    typ = _location_value_type(r, plan)
    pre = Expr[:($mu::$typ = $(_inverse_link_expr(r.link, lp)))]
    sarg = _scale_plate_arg(r, plan, pre)
    k = sarg isa Symbol ? sarg : Float64(sarg)
    push!(pre, :($a = $mu .* $k), :($b = (1 .- $mu) .* $k))
    inputs = Any[y, a, b]
    yv, avv, bvv = _dovar(1), _dovar(2), _dovar(3)
    cell = :(beta($avv, $bvv).logpdf($yv))
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    cell = :($avv > 0 && $bvv > 0 ? $cell : -Inf)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# BetaBinomial2 mean-precision: mu/a/b precomputes (the Beta-plate
# pattern) plus trials threading (the Binomial-plate pattern); the cell
# is the Stan-native `beta_binomial(n, alpha, beta)` endpoint. phi is
# never a plate input: a Symbol threads through the precomputes (scalar
# parameter or per-observation column, both broadcast), a literal
# inlines, and a predictor-fed phi binds its constrained vector once
# (`_ppl_sc_`, the NB2 precedent) and broadcasts through `a`/`b`.
function _betabinomial2_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    pre = Expr[]
    sarg = _scale_plate_arg(r, plan, pre)
    k = sarg isa Symbol ? sarg : Float64(sarg)
    mu = _mu_name(r.label)
    a = _shape_a_name(r.label)
    b = _shape_b_name(r.label)
    yin = _count_yplate!(pre, plan, y, r.label)
    inputs = Any[yin, a, b]
    yv, avv, bvv = _dovar(1), _dovar(2), _dovar(3)
    nref = _thread_ref!(inputs, r.trials, true)
    cell = :(beta_binomial($nref, $avv, $bvv).logpdf($yv))
    cell = _wrap_evidence!(r, plan, pre, inputs, cell)
    cell = :($avv > 0 && $bvv > 0 ? $cell : -Inf)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv, cell)
    end
    typ = _location_value_type(r, plan)
    return Expr[pre...,
        :($mu::$typ = $(_inverse_link_expr(r.link, lp))),
        :($a = $mu .* $k),
        :($b = (1 .- $mu) .* $k),
        _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Reference-coded multi-logit categorical (SB `CategoricalLogit`): K−1
# linear predictors supply the non-reference logits; class 1 is the
# implicit zero reference. The cell is the `categorical_logit_ref` math
# in scalar form (per-row logit vectors would need matrix assembly):
# the observed term selects by `y` (ifelse chain) and the normalizer is
# a max-shifted log-sum-exp over (0, etas...) — linear-size in K (a
# nested logaddexp chain would double nodes per level; the max chain is
# re-embedded per term, so K² worst case — K is small).
function _categorical_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    pre = Expr[]
    preds = [r.predictor; r.extra_predictors...]
    lps = [_lp_name(_predictor(plan, q)) for q in preds]
    inputs = Any[r.response, lps...]
    yv = _dovar(1)
    etas = [_dovar(i) for i in 2:length(inputs)]
    K = length(preds) + 1
    obs = :(0.0)
    for j in K:-1:2
        obs = :(ifelse($yv == $j, $(etas[j-1]), $obs))
    end
    m = :(0.0)
    for e in etas
        m = :(max($m, $e))
    end
    sumexp = :(exp(0.0 - $m))
    for e in etas
        sumexp = :($sumexp + exp($e - $m))
    end
    cell = :($obs - ($m + log($sumexp)))
    function categorical_tail(b, upper)
        terms = Any[upper ? :($b < 1 ? exp(-$m) : 0.0) : :($b >= 1 ? exp(-$m) : 0.0)]
        for j in 2:K
            t = upper ? :($b < $j ? exp($(etas[j-1]) - $m) : 0.0) :
                :($b >= $j ? exp($(etas[j-1]) - $m) : 0.0)
            push!(terms,t)
        end
        numerator=foldl((a,b)->:($a+$b),terms)
        :($numerator / $sumexp)
    end
    cell = _wrap_evidence!(r,plan,pre,inputs,cell;
        cdf=b->categorical_tail(b,false),ccdf=b->categorical_tail(b,true))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv,cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw,node,inputs,cell)...]
end

# Stable log-logistic (Stan `log_inv_logit`): `z ≥ 0` takes
# `-log1p(exp(-z))`, `z < 0` takes `z - log1p(exp(z))` — no overflow
# either tail. Base ops only (transparent to the reverse pass).
_log_inv_logit(z) = :(ifelse($z >= 0.0, -log1p(exp(-$z)), $z - log1p(exp($z))))

# Ordinal link log-CDF / log-CCDF over a scalar z (SB `brm_ordinal_logcdf`
# / `brm_ordinal_logccdf`). Probit uses the slice-1 erfc treatment
# (`0.5*erfc(∓z/√2)` under a log — accurate in moderate ranges; extreme
# tails round, the accepted slice-1 probit caveat).
function _ordinal_logF(link::LinkFunction, z)
    link === LogitLink && return _log_inv_logit(z)
    link === ProbitLink && return :(log(0.5 * erfc(-$z / sqrt(2))))
    return :(log(-expm1(-exp($z))))
end
function _ordinal_logCC(link::LinkFunction, z)
    link === LogitLink && return _log_inv_logit(:(-$z))
    link === ProbitLink && return :(log(0.5 * erfc($z / sqrt(2))))
    return :(-exp($z))
end

# Stable log-difference of log-probs (a ≥ b): `a + log1p(-exp(b - a))`.
_log_diff_exp(a, b) = :($a + log1p(-exp($b - $a)))

# Modeled-scale precompute name (response `label`).
_disc_name(label::Symbol) = Symbol(:_ppl_disc_, label)

# Ordinal per-observation reference as a plate do-var: `nothing` inlines
# `absent`, a literal inlines, and a Symbol (data column / precompute)
# threads — directly for the per-observation cumulative plate
# (`rows === nothing`), or gathered onto the stopping-ratio stage lanes
# (`<lane> = ref[rows]`, emitted into `prests`).
function _ordinal_lane_ref!(inputs::Vector{Any}, prests::Vector{Expr}, ref,
        rows, lane::Symbol; absent = 1.0)
    ref === nothing && return absent
    ref isa Symbol || return Float64(ref)
    rows === nothing && return _thread_ref!(inputs, ref)
    push!(prests, :($lane = _broadcast_gather($ref, $rows)))
    return _thread_ref!(inputs, lane)
end

# Ordinal latent scale source: absent (`nothing` — the 3-positional form),
# a literal, a data column, or a log-link predictor's `exp` precompute
# (structural positivity — the Poisson `exp.(lp)` precedent).
function _ordinal_scale_source!(prests::Vector{Expr}, r::LikelihoodSpec,
        plan::StructuralPlan)
    d = r.discrimination
    if d isa ScalePredictorRef
        return _scale_use_plate_arg(r, plan, prests, d, Symbol(r.label, :_disc))
    end
    if d isa Symbol && any(p -> p.name === d, plan.predictors)
        return _disc_pre!(prests, r, plan, d)
    end
    return d
end

# Modeled-scale column: `exp` over the scale predictor's lp node (the
# predictor statements run before the likelihood, so the node exists;
# validation proved the link is LogLink). Explicit dotted form, evaluated
# once and threaded like the stage-effect columns.
function _disc_pre!(prests::Vector{Expr}, r::LikelihoodSpec,
        plan::StructuralPlan, sname::Symbol)
    pred = _predictor(plan, sname)
    lp = _lp_name(pred)
    name = _disc_name(r.label)
    typ = _predictor_value_type(plan, pred)
    push!(prests, :($name::$typ = $(_inverse_link_expr(pred.link, lp))))
    return name
end

# Ordered response plate (OrderedLogistic + Ordinal; OrderedLogistic is
# cumulative-logit with d = 1 and no threshold effects). The thresholds
# thread as ONE shared vector (`Ref(t)`) that each cell gathers by its own
# level, so nothing in the emitted program — statements or cell — grows
# with the level count K. K=1 lowers to a zero cell (SB's
# zero-information likelihood).
_ordinal_cutpoints_valid(c) = all(c[2:end] .> c[1:end-1])

function _ordinal_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    K = r.n_levels
    K === nothing && throw(ContractValidationError(
        "[generator] ordered response $(r.label) has unresolved n_levels " *
        "(bind_data infers it)"))
    structure = r.family === OrderedLogisticFam ? :cumulative : r.ordinal_structure
    structure === :cumulative ||
        return _ordinal_stopping_stmts(r, plan, node, pw, K)
    lp = _location_node(r, plan)
    inputs = Any[r.response, lp]
    yv, etav = _dovar(1), _dovar(2)
    prests = Expr[]
    cutref = nothing
    validref = nothing
    dref = _ordinal_lane_ref!(inputs, prests,
        _ordinal_scale_source!(prests, r, plan), nothing, :_)
    cell = if K == 1
        # SB's zero-information likelihood; the cell stays a real Expr
        # over the (integer) response do-var (`:(0.0)` would quote to a
        # bare Float64, which the plate builder does not take).
        :(0.0 * $yv)
    else
        push!(inputs, :(Ref($(r.thresholds))))
        cutref = _dovar(length(inputs))
        cumulative = _ordinal_cumulative_cell(r.link, K, yv, etav, dref, cutref)
        # A response reads the authored vector; it does not impose an
        # Ordered transform or replace its prior. Guard before logarithms,
        # including when only boundary categories are observed.
        parameter = only(p for p in plan.vector_parameters if p.name === r.thresholds)
        if _is_ordered_parameter(parameter.family) && parameter.name ∉ plan.conditioned
            cumulative
        else
            valid = Symbol(:_ppl_cutpoints_valid_,r.label)
            check = Expr(:call,GlobalRef(@__MODULE__,:_ordinal_cutpoints_valid),r.thresholds)
            push!(prests,:($valid = $check))
            push!(inputs,valid)
            validref = _dovar(length(inputs))
            cumulative
        end
    end
    if r.discrimination !== nothing
        cell = :(isfinite($dref) && $dref > 0 ? $cell : -Inf)
    end
    c = cutref
    function ordinal_tail(b, upper)
        K == 1 && return upper ? :($b < 1 ? 1.0 : 0.0) : :($b < 1 ? 0.0 : 1.0)
        logtail=upper ? _ordinal_logCC(r.link,:($dref * ($c[$b]-$etav))) :
            _ordinal_logF(r.link,:($dref * ($c[$b]-$etav)))
        :($b < 1 ? $(upper ? 1.0 : 0.0) : ($b >= $K ? $(upper ? 0.0 : 1.0) : exp($logtail)))
    end
    cell=_wrap_evidence!(r,plan,prests,inputs,cell;
        cdf=b->ordinal_tail(b,false),ccdf=b->ordinal_tail(b,true))
    validref === nothing || (cell = :($validref ? $cell : -Inf))
    if r.weights !== nothing
        cell = _weighted_cell(_thread_ref!(inputs, r.weights), cell)
    end
    return Expr[prests..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Cumulative cell over the shared threshold vector `c`: the first level
# takes logF at `c[1]`, the last logCC at `c[K-1]`, an interior level `y`
# the stable log-difference of logF at `c[y]` and `c[y-1]` (thresholds
# are ordered, so hi ≥ lo). Authored as lazy branches on the observed
# level: only the observation's own arm runs, so every gather is in bounds
# by construction; the branch condition reads bound data only, so plate
# lowering splits the lanes by arm before any backend sees a branch.
function _ordinal_cumulative_cell(link::LinkFunction, K::Int, yv::Symbol,
        etav::Symbol, dref, c::Symbol)
    first = _ordinal_logF(link, :($dref * ($c[1] - $etav)))
    last = _ordinal_logCC(link, :($dref * ($c[$(K - 1)] - $etav)))
    K == 2 && return :($yv == 1 ? $first : $last)
    hi = _ordinal_logF(link, :($dref * ($c[$yv] - $etav)))
    lo = _ordinal_logF(link, :($dref * ($c[$yv - 1] - $etav)))
    return :($yv == 1 ? $first :
        ($yv == $K ? $last : $(_log_diff_exp(hi, lo))))
end

# Stage-lane names for response `label`.
_stage_lane(label::Symbol, what::Symbol) = Symbol(:_ppl_srl_, what, :_, label)

# Stopping-ratio plate over the stage lanes: every lane column is gathered
# once by the (bound) lane tables — the observation's predictor/scale/weight,
# the stage's threshold `t[stage]`, and with per_threshold effects the
# stage's coefficient pack entries (stage-major: stage j occupies
# `(j-1)*p+1 .. j*p`) — so the cell itself reads only scalars. The cell survives (logCC) or stops
# (logF) by a lazy branch on the bound stage/level pair, which plate
# lowering splits by arm. K=1 runs zero stages (SB's zero-information
# likelihood) and lowers to a zero cell over the observations.
function _ordinal_stopping_stmts(r::LikelihoodSpec, plan::StructuralPlan,
        node::Symbol, pw::Symbol, K::Int)
    r.evidence.kind === :none || return _ordinal_stopping_evidence_stmts(r, plan, node, pw, K)
    y = r.response
    if K == 1
        return _plate_sum_stmts(pw, node, Any[y], :(0.0 * $(_dovar(1))))
    end
    lp = _location_node(r, plan)
    obs, stage = _stage_lane(r.label, :obs), _stage_lane(r.label, :stage)
    prests = Expr[:($obs = _ordinal_stage_obs($y, $K)),
        :($stage = _ordinal_stage_idx($y, $K))]
    level = _stage_lane(r.label, :y)
    eta = _stage_lane(r.label, :eta)
    thr = _stage_lane(r.label, :t)
    push!(prests, :($level = $y[$obs]), :($eta = _broadcast_gather($lp, $obs)),
        :($thr = $(r.thresholds)[$stage]))
    inputs = Any[stage, level, eta, thr]
    sv, yv, etav, tv = _dovar(1), _dovar(2), _dovar(3), _dovar(4)
    dref = _ordinal_lane_ref!(inputs, prests,
        _ordinal_scale_source!(prests, r, plan), obs, _stage_lane(r.label, :d))
    z = if r.threshold_effects !== nothing
        matrix = _stage_lane(r.label, :effects_matrix)
        eff = _stage_lane(r.label, :eff)
        push!(prests,
            :($matrix = _ordinal_effects_matrix($(r.threshold_effects), $y, $K)),
            :($eff = $matrix[$obs .+ ($stage .- 1) .* length($y)]))
        push!(inputs, eff)
        :($dref * ($tv - $etav - $(_dovar(length(inputs)))))
    elseif isempty(r.threshold_columns)
        :($dref * ($tv - $etav))
    else
        eff = _stage_lane(r.label, :eff)
        p = length(r.threshold_columns)
        terms = Any[:($col[$obs] .* $(r.threshold_coefs)[($stage .- 1) .* $p .+ $ci])
            for (ci, col) in enumerate(r.threshold_columns)]
        push!(prests, :($eff = $(foldl((a, b) -> :($a .+ $b), terms))))
        push!(inputs, eff)
        :($dref * ($tv - $etav - $(_dovar(length(inputs)))))
    end
    cell = :($sv < $yv ? $(_ordinal_logCC(r.link, z)) : $(_ordinal_logF(r.link, z)))
    if r.discrimination !== nothing
        cell = :(isfinite($dref) && $dref > 0 ? $cell : -Inf)
    end
    if r.weights !== nothing
        cell = _weighted_cell(_ordinal_lane_ref!(inputs, prests, r.weights,
            obs, _stage_lane(r.label, :w)), cell)
    end
    return Expr[prests..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

function _ordinal_stopping_evidence_stmts(r, plan, node, pw, K)
    # One plate lane per observation; the helper retains the stage loop.
    # Evidence describes the whole ordinal law, rather than individual stops.
    K == 1 && return _ordinal_single_evidence_stmts(r, plan, node, pw)
    pre = Expr[]
    lp = _location_node(r, plan)
    row = _stage_lane(r.label, :evidence_row)
    push!(pre, :($row = collect(eachindex($(r.response)))))
    inputs = Any[r.response, lp, :(Ref($(r.thresholds))), row]
    yv, eta, t, iv = (_dovar(i) for i in 1:4)
    d = _ordinal_lane_ref!(inputs, pre, _ordinal_scale_source!(pre, r, plan), nothing, :_)
    effects = nothing
    if r.threshold_effects !== nothing
        matrix = _stage_lane(r.label, :effects_matrix)
        push!(pre, :($matrix = _ordinal_effects_matrix($(r.threshold_effects), $(r.response), $K)))
        push!(inputs, :(Ref($matrix)))
        effects = _dovar(length(inputs))
    elseif !isempty(r.threshold_columns)
        matrix = _stage_lane(r.label, :effects_matrix)
        p = length(r.threshold_columns)
        push!(pre, :($matrix = hcat($(r.threshold_columns...)) * reshape($(r.threshold_coefs), $p, $(K-1))))
        push!(inputs, :(Ref($matrix)))
        effects = _dovar(length(inputs))
    end
    link = r.link === LogitLink ? :(Val(:logit)) :
        r.link === ProbitLink ? :(Val(:probit)) : :(Val(:cloglog))
    cell = :(rk_ordinal_stopping_logpdf($t, $eta, $d, $effects, $iv, ReactiveKernels._tensorized_trunc(Int, $yv), $link))
    cell = _wrap_evidence!(r, plan, pre, inputs, cell;
        cdf=b -> :(rk_ordinal_stopping_tail($t, $eta, $d, $effects, $iv, $b, $link, false)),
        ccdf=b -> :(rk_ordinal_stopping_tail($t, $eta, $d, $effects, $iv, $b, $link, true)))
    if r.discrimination isa ScalePredictorRef
        cell = :(isfinite($d) && $d > 0 ? $cell : -Inf)
    end
    r.weights === nothing || (cell = _weighted_cell(_thread_ref!(inputs, r.weights), cell))
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

function _ordinal_single_evidence_stmts(r, plan, node, pw)
    pre, inputs = Expr[], Any[r.response]
    cell = _wrap_evidence!(r, plan, pre, inputs, :(0.0 * $(_dovar(1)));
        cdf=b -> :($b < 1 ? 0.0 : 1.0), ccdf=b -> :($b < 1 ? 1.0 : 0.0))
    r.weights === nothing || (cell = _weighted_cell(_thread_ref!(inputs, r.weights), cell))
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Shared-simplex multinomial (SB `brm_multinomial` vector[K] method): the
# count matrix crosses as K raw columns — program structure (the count
# columns the model names), not a data-inferred size — and the level
# probabilities thread as one shared vector the cell reads per column.
# The cell is Stan's `multinomial_lpmf` in scalar form —
# `lgamma(N+1) − Σ lgamma(c+1) + Σ c*log(p)` — with the `0*log(0) = 0`
# convention guarded per term (Stan treats a zero count at a zero
# probability as 0, not NaN). A literal N folds its `lgamma(N+1)`
# host-side (exact same value, computed once).
function _multinomial_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    counts = [r.response; r.count_columns...]
    K = length(counts)
    inputs = Any[counts...]
    cvs = [_dovar(i) for i in 1:K]
    nref = _thread_ref!(inputs, r.trials, true)
    push!(inputs, :(Ref($(r.predictor))))
    pv = _dovar(length(inputs))
    lfact = r.trials isa Int ? loggamma(r.trials + 1.0) : :(loggamma($nref + 1.0))
    cell = :($lfact)
    stmts = Expr[]
    for (i, cv) in enumerate(cvs)
        term = Symbol(:_ppl_multinomial_term_, r.label, :_, i)
        # Bound zero counts take the constant arm during plate preparation.
        # Their logarithm and its derivative must never be evaluated.
        push!(stmts, :($term = if $cv == 0
            0.0
        else
            $cv * log($pv[$i])
        end))
        cell = :($cell - loggamma($cv + 1.0) + $term)
    end
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    push!(stmts, cell)
    return _plate_sum_stmts(pw, node, inputs, stmts)
end

# Plain categorical over shared-simplex probabilities (Stan
# `categorical_lpmf`): the level log-probabilities `log.(p)` are one
# vector statement, and each cell gathers its observed level's entry
# (`logp[y]`) — no per-level work in the cell. K=1 lowers to
# `log(1.0) = 0` uniformly.
_logp_name(label::Symbol) = Symbol(:_ppl_logp_, label)

function _categorical_plain_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    r.n_levels === nothing && throw(ContractValidationError(
        "[generator] categorical response $(r.label) has unresolved n_levels " *
        "(bind_data infers it)"))
    pre=Expr[]
    logp = _logp_name(r.label)
    inputs = Any[r.response, :(Ref($logp))]
    yv, lv = _dovar(1), _dovar(2)
    cell = :($lv[$yv])
    cell = _wrap_evidence!(r,plan,pre,inputs,cell;
        cdf=b -> :(rk_logprob_tail($lv,$b,false)),
        ccdf=b -> :(rk_logprob_tail($lv,$b,true)))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = _weighted_cell(wv,cell)
    end
    return Expr[pre..., :($logp::AbstractVector{Float64} = log.($(r.predictor))),
        _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Joint correlated-outcomes likelihood (SB's per-row
# `multi_normal_cholesky(mean_row, L)` with
# `L = diag_pre_multiply(scales, L_corr)`): one plate over the K outcome
# columns + K mean LPs sums the per-row density. The row cell is scalar
# forward substitution over threaded do-vars (the multinomial-cell shape —
# no matrix, no triangular solve in the cell, so the tensorized plate
# traces; a core-`mvnormal` splice was probed and fails Reactant primal
# inside the plate — the in-cell `\` hits scalar indexing in Reactant's
# `generic_trimatdiv!`, a primal gap distinct from the §7f gradient gap).
# The L entries materialize as `_ppl_mvn_Le_` scalars (row-scaled layout
# temps) and thread as shared scalar plate inputs.
function _mvn_cholesky_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan,
        node::Symbol, pw::Symbol)
    outcomes = [r.response; r.extra_responses...]
    preds = [r.predictor; r.extra_predictors...]
    K = length(outcomes)
    ap = _joint_factor_arrays(r, plan)
    if ap === nothing
        si = findfirst(p -> p.name === r.factor_scales, plan.vector_parameters)
        ci = findfirst(p -> p.name === r.factor_corr, plan.vector_parameters)
        (si === nothing || ci === nothing) && throw(ContractValidationError(
            "[generator] joint response $(r.label) factor pieces unresolved " *
            "(validate_plan links them)"))
        sc, cr = plan.vector_parameters[si], plan.vector_parameters[ci]
        (sc.size == K && cr.size == K) || throw(ContractValidationError(
            "[generator] joint response $(r.label) factor sizes disagree " *
            "with the $K outcomes (validate_plan checks this)"))
    else
        sc, cr = ap
    end
    stmts = Expr[]
    # L[i,j] = scales[i] * L_corr[i,j] (j ≤ i), one scalar per lower entry.
    for i in 1:K, j in 1:i
        scale = ap === nothing ? _vector_elt_name(sc.name, i) : :($(sc.name)[$i])
        corr = ap === nothing ? _rl_name(cr.name, i, j) : :($(cr.name)[$i, $j])
        push!(stmts, :($(_mvn_L_entry(r.label, i, j))::Float64 = $scale * $corr))
    end
    lps = [_lp_name(_predictor(plan, q)) for q in preds]
    inputs = Any[outcomes...; lps...]
    yvs = [_dovar(i) for i in 1:K]
    mvs = [_dovar(K + i) for i in 1:K]
    Ld = Dict{Tuple{Int,Int},Symbol}()
    for i in 1:K, j in 1:i
        Ld[(i, j)] = _thread_ref!(inputs, _mvn_L_entry(r.label, i, j))
    end
    cell = _mvn_row_cell(K, yvs, mvs, Ld)
    append!(stmts, _plate_sum_stmts(pw, node, inputs, cell))
    return stmts
end

# In-graph L-entry name for a joint response (`_ppl_mvn_Le_<label>_<i>_<j>`,
# j ≤ i). All `_ppl_`-hygienic.
_mvn_L_entry(label::Symbol, i::Int, j::Int) =
    Symbol(:_ppl_mvn_Le_, label, :_, i, :_, j)

# One joint row's log-density as a single scalar expression: residuals,
# forward substitution (`z[i] = (d[i] − Σ L[i,j]·z[j]) / L[i,i]`, inlined —
# K is small), quadratic form, and the row constant. Pointwise-pure: each
# lane evaluates its row's full `multi_normal_cholesky` log-density.
function _mvn_row_cell(K::Int, yvs::Vector{Symbol}, mvs::Vector{Symbol},
        Ld::Dict{Tuple{Int,Int},Symbol})
    ds = [:( $(yvs[i]) - $(mvs[i]) ) for i in 1:K]
    zs = Any[]
    for i in 1:K
        num = ds[i]
        for j in 1:i-1
            num = :( $num - $(Ld[(i, j)]) * $(zs[j]) )
        end
        push!(zs, :( ($num) / $(Ld[(i, i)]) ))
    end
    quad = foldl((a, z) -> :( $a + $z * $z ), zs; init = :(0.0))
    logdet = foldl((a, i) -> :( $a + log($(Ld[(i, i)])) ), 1:K; init = :(0.0))
    row_const = -0.5 * K * log(2 * pi)
    return :( $row_const - $logdet - 0.5 * $quad )
end

# One homogeneous coefficient plate over `coefaccess` (the block symbol or
# a static-range slice) with per-lane (location, scale[, nu]) vectors.
# A lane is a `Vector{Float64}` (literal lane, bound here) or a `Symbol`
# (shared hyperparameter, aliased here and threaded as a plate scalar —
# RK auto-threads an unthreaded signature scalar read by the cell — so
# the centered shape keeps one plate, O(1) in levels).
# The cell is the table-driven endpoint splice; only the 2-vs-3 arg
# plate shape differs (StudentT threads a nu vector; nu is always a
# literal — validation pins it).
# Bind one plate-prior lane: a literal lane vector, or a shared-hyper
# alias (the hyper name reads its constrained value — transforms and
# assignments all precede the prior statements in the generated body).
_bind_plate_prior_arg!(stmts::Vector{Expr}, name::Symbol,
    v::Vector{Float64}) = push!(stmts, :($name = Float64[$(v...)]))
_bind_plate_prior_arg!(stmts::Vector{Expr}, name::Symbol, v::Symbol) =
    push!(stmts, :($name = $v))
# A per-element lane mixing literals and names (`Normal.(0, [s1, 2.0])`):
# one in-graph vector over the constrained names.
_bind_plate_prior_arg!(stmts::Vector{Expr}, name::Symbol, v::Vector{Any}) =
    push!(stmts, :($name = [$(v...)]))

# Collapse one prior-arg position across a block's rows to its plate
# lane: all-literal rows give the literal lane vector; rows sharing one
# hyper name give the name; per-element names (or names mixed with
# literals) give one element per row.
function _plate_prior_lane(rows::Vector{PopulationPrior}, field::Symbol,
        what::String)
    vals = map(r -> getfield(r, field), rows)
    all(v -> v isa Real, vals) &&
        return Float64[v for v in vals]
    all(v -> v isa Symbol, vals) && allequal(vals) &&
        return vals[1]
    return Any[v isa Symbol ? v : Float64(v) for v in vals]
end

function _append_coef_plate!(stmts::Vector{Expr}, coefaccess::Any,
        node::Symbol, pw::Symbol, mut::Symbol, sdt::Symbol, nut::Symbol,
        loc::Union{Vector{Float64},Symbol,Vector{Any}},
        sca::Union{Vector{Float64},Symbol,Vector{Any}},
        nus::Vector{Float64}, fam::Symbol)
    fam === :flat && throw(ContractValidationError(
        "[generator] internal: flat coefficients contribute no plate"))
    _bind_plate_prior_arg!(stmts, mut, loc)
    _bind_plate_prior_arg!(stmts, sdt, sca)
    if fam === :student_t
        push!(stmts, :($nut = Float64[$(nus...)]))
        cv, nuv, mv, sv = _dovar(1), _dovar(2), _dovar(3), _dovar(4)
        cell = _family_logpdf_expr(fam, Any[nuv, mv, sv], cv)
        append!(stmts, _plate_sum_stmts(pw, node,
            Any[coefaccess, nut, mut, sdt], cell))
    else
        cv, mv, sv = _dovar(1), _dovar(2), _dovar(3)
        cell = _family_logpdf_expr(fam, Any[mv, sv], cv)
        append!(stmts, _plate_sum_stmts(pw, node,
            Any[coefaccess, mut, sdt], cell))
    end
    return nothing
end

# Endpoint args for one population row: a literal, or a parameter /
# assignment name read as its constrained local (transforms and
# assignments precede the prior statements). StudentT carries nu.
_prior_endpoint(v::Symbol) = v
_prior_endpoint(v::Real) = Float64(v)
_population_endpoint_args(pr::PopulationPrior) = pr.family === :student_t ?
    Any[Float64(pr.nu), _prior_endpoint(pr.location),
        _prior_endpoint(pr.scale)] :
    Any[_prior_endpoint(pr.location), _prior_endpoint(pr.scale)]

# Per-addressee emission for a MIXED predictor (the homogeneous fast path
# in `_prior_statements` keeps today's exact plate): width-1 blocks become
# scalar density nodes over literal coefficient reads (the heterogeneous
# sd-prior margins precedent); wide Factor/MatrixTerm blocks — one family
# per block, validated — become one plate each over a static-range slice
# (data-derived widths stay in loops); flat addressees contribute nothing.
function _mixed_prior_stmts!(stmts::Vector{Expr}, terms::Vector{Any},
        pred::PredictorSpec, shape::DesignShape,
        priors::Vector{PopulationPrior})
    by_addressee = Dict{Symbol,PopulationPrior}()
    for pr in priors
        pr.predictor === pred.name || continue
        by_addressee[pr.addressee] = pr
    end
    coef = block_name(pred.name)
    offset = 0
    for b in shape.blocks
        b.width == 0 && continue
        if b.kind === FactorTerm || b.kind === MatrixTerm
            elrows = b.kind === FactorTerm ?
                fill(by_addressee[b.addressee], b.width) :
                [by_addressee[e === nothing ? :Intercept : e]
                    for e in b.elements]
            fam = elrows[1].family
            fam === :flat || _mixed_wide_block_stmts!(stmts, terms,
                pred.name, b, coef, offset, elrows, fam)
        else
            b.width == 1 || throw(ContractValidationError(
                "[generator] internal: mixed-prior scalar block " *
                "$(b.addressee) of $(pred.name) has width $(b.width)"))
            pr = by_addressee[b.addressee]
            # Flat scalars contribute no node but still occupy their
            # design position — the offset accumulates below for every
            # block (a `continue` here would misalign every later
            # read).
            if pr.family !== :flat
                node = Symbol(:_ppl_prior_, pred.name, :_, b.addressee)
                expr = _family_logpdf_expr(pr.family,
                    _population_endpoint_args(pr),
                    Expr(:ref, coef, offset + 1))
                push!(stmts, :($node::Float64 = $expr))
                push!(terms, node)
            end
        end
        offset += b.width
    end
    return nothing
end

# One wide homogeneous block of a mixed predictor: a plate over the
# block's static-range slice with the block family's cell.
function _mixed_wide_block_stmts!(stmts::Vector{Expr}, terms::Vector{Any},
        pname::Symbol, b::DesignBlock, coef::Symbol, offset::Int,
        elrows::Vector{PopulationPrior}, fam::Symbol)
    tag = b.kind === FactorTerm ? b.addressee : b.column
    node = Symbol(:_ppl_prior_, pname, :_, tag)
    pw = Symbol(:_ppl_pw_prior_, pname, :_, tag)
    mut = Symbol(:_ppl_prmu_, pname, :_, tag)
    sdt = Symbol(:_ppl_prsd_, pname, :_, tag)
    nut = Symbol(:_ppl_prnu_, pname, :_, tag)
    slice = Expr(:ref, coef,
        Expr(:call, :(:), offset + 1, offset + b.width))
    what = "$(b.kind) block $tag of $pname"
    loc = _plate_prior_lane(elrows, :location, what)
    sca = _plate_prior_lane(elrows, :scale, what)
    nus = [Float64(r.nu) for r in elrows]
    _append_coef_plate!(stmts, slice, node, pw, mut, sdt, nut, loc, sca,
        nus, fam)
    push!(terms, node)
    return nothing
end

# Group compatible literal scalar priors as an evaluation optimization.
# Declarations remain the owners of names, transforms and prior arguments.
function _parameter_prior_groups(plan)
    groups = Vector{SampledParameter}[]
    for family in (:normal, :cauchy, :laplace, :logistic, :student_t)
        ps = SampledParameter[p for p in plan.parameters
            if p.family === family && p.support_override === nothing &&
                all(v -> v isa Real, values(p.args))]
        length(ps) > 1 && push!(groups, ps)
    end
    return groups
end

function _parameter_prior_input(ps, plan, layout)
    names = [p.name for p in ps]
    for pred in plan.predictors
        shape = design_shape(pred, plan.columns; levelmaps = plan.levelmaps,
            matrices = plan.matrices)
        terms = [t for (t, b) in zip(pred.terms, shape.blocks) if b.width > 0]
        all(t -> _parameter_term(t) && t.options.sign == 1, terms) || continue
        [t.options.parameter for t in terms] == names || continue
        all(b -> b.width <= 1, shape.blocks) || continue
        return _affine_block_name(pred)
    end
    view = _parameter_coordinate_view([(n, 1) for n in names], layout)
    return view === nothing ? :([$(names...)]) : view
end

function _parameter_prior_statements!(stmts, terms, plan, layout;
        prefix = :_ppl_prior_, group_scalars = true)
    grouped = Set{Symbol}()
    for ps in (group_scalars ? _parameter_prior_groups(plan) : Vector{SampledParameter}[])
        family = first(ps).family
        _, li, si, ni = _COEF_SHAPES[family]
        args = [collect(values(p.args)) for p in ps]
        loc = Float64[a[li] for a in args]
        sca = Float64[a[si] for a in args]
        nus = ni == 0 ? fill(NaN, length(ps)) : Float64[a[ni] for a in args]
        node = Symbol(prefix, :parameters_, family)
        _append_coef_plate!(stmts, _parameter_prior_input(ps, plan, layout),
            node, Symbol(node, :_pw), Symbol(node, :_mu), Symbol(node, :_sd),
            Symbol(node, :_nu), loc, sca, nus, family)
        push!(terms, node)
        union!(grouped, (p.name for p in ps))
    end
    for p in plan.parameters
        p.name in grouped && continue
        node = Symbol(:_ppl_prior_, p.name)
        if p.family === :external
            append!(stmts, _external_density_statements(p))
            push!(terms, node)
            continue
        end
        prior = _sampled_prior_expr(p; pre = stmts, conditioned = p.name in plan.conditioned)
        push!(stmts, :($node::Float64 = $prior))
        push!(terms, node)
    end
    return nothing
end

function _prior_statements(plan::StructuralPlan, layout::LayoutTable;
        gathers::Set{Tuple{Symbol,Symbol,Int}} = Set{Tuple{Symbol,Symbol,Int}}(), context = plan)
    stmts = Expr[]
    terms = Any[]
    for pred in plan.predictors
        pred = _legacy_predictor(pred)
        shape = design_shape(pred, plan.columns; levelmaps = plan.levelmaps,
            matrices = plan.matrices)
        shape.width == 0 && continue
        node = Symbol(:_ppl_prior_, pred.name)
        pw = Symbol(:_ppl_pw_prior_, pred.name)
        mut = Symbol(:_ppl_prmu_, pred.name)
        sdt = Symbol(:_ppl_prsd_, pred.name)
        specs = coefficient_prior_specs(shape, plan.population_priors)
        fams = map(s -> s.family, specs)
        alllit = all(s -> s.location isa Real && s.scale isa Real, specs)
        if all(==(fams[1]), fams) && fams[1] !== :flat && alllit
            # Homogeneous fast path: today's exact plate, family cell.
            nut = Symbol(:_ppl_prnu_, pred.name)
            loc = [Float64(s.location) for s in specs]
            sca = [Float64(s.scale) for s in specs]
            nus = [Float64(s.nu) for s in specs]
            _append_coef_plate!(stmts, block_name(pred.name), node, pw,
                mut, sdt, nut, loc, sca, nus, fams[1])
            push!(terms, node)
        elseif all(==(:flat), fams)
            push!(stmts, :($node::Float64 = 0.0))
            push!(terms, node)
        else
            _mixed_prior_stmts!(stmts, terms, pred, shape,
                plan.population_priors)
        end
    end
    _parameter_prior_statements!(stmts, terms, plan, layout)
    # GLM-object coefficient vectors: the same plate-prior shape as a
    # population-prior coefficient block, driven by the response matrix
    # columns (priors addressed by response label — validation pins
    # full coverage).
    for r in plan.responses
        _is_glm_family(r.family) || continue
        _is_array_param(plan, r.glm_beta) && continue
        m = _find_matrix(plan, r.predictor)
        cols = Symbol[c for c in m.columns if c !== nothing]
        loc = Float64[]
        sca = Float64[]
        for c in cols
            i = findfirst(p -> p.predictor === r.label && p.addressee === c,
                plan.population_priors)
            pr = plan.population_priors[i]
            (pr.location isa Real && pr.scale isa Real) ||
                throw(ContractValidationError(
                    "[generator] internal: GLM beta prior for " *
                    "$(r.label).$c carries a hyperparameter " *
                    "(hierarchical GLM priors are not in slice 1)"))
            push!(loc, Float64(pr.location))
            push!(sca, Float64(pr.scale))
        end
        node = Symbol(:_ppl_prior_, r.label)
        pw = Symbol(:_ppl_pw_prior_, r.label)
        mut = Symbol(:_ppl_prmu_, r.label)
        sdt = Symbol(:_ppl_prsd_, r.label)
        push!(stmts, :($mut = Float64[$(loc...)]))
        push!(stmts, :($sdt = Float64[$(sca...)]))
        cv, mv, sv = _dovar(1), _dovar(2), _dovar(3)
        cell = :(normal($mv, $sv).logpdf($cv))
        append!(stmts, _plate_sum_stmts(pw, node,
            Any[r.glm_beta, mut, sdt], cell))
        push!(terms, node)
    end
    # Vector latents (cutpoints/thresholds/simplexes/coefficient packs,
    # joint-factor scales/Cholesky): one plate-sum or broadcast per vector
    # (bound plans carry concrete sizes). Empty packs contribute 0.0.
    for p in plan.vector_parameters
        _vector_parameter_prior_stmts!(stmts, terms, p)
    end
    # Per-cell latent (plate) parameters: one plate over the block, summing the
    # shared-prior log-density across cells (the same plate-sum shape as a
    # population-prior coefficient block, generalized to any standard family).
    # Every value the cell reads is threaded as a plate PORT (the latent vector
    # plus each scalar prior arg) — captured free names are rejected by the
    # `@kernel` plate expander, exactly as the Gaussian-likelihood scale is
    # threaded.
    for p in plan.plate_parameters
        if p.family === :external
            append!(stmts, _external_density_statements(p))
            push!(terms, Symbol(:_ppl_prior_, p.name))
        else
            _vector_prior_stmts!(stmts, terms, p.name, p.family, p.args,
                p.support_override; rows = _plate_rows(plan, p),
                float64_variate = p.name ∉ plan.conditioned)
        end
    end
    # Sequential-recurrence (scan) latents: setup + recurrence density.
    scanstmts, scannodes = _scan_prior_statements(plan, layout)
    append!(stmts, scanstmts)
    append!(terms, scannodes)
    # Declared array parameters (`arrays.jl`).
    _array_prior_stmts!(stmts, terms, plan, gathers; context)
    joint = foldl((a, b) -> :($a + $b), terms; init = :(0.0))
    push!(stmts, :(prior::Float64 = $joint))
    return stmts
end

# Shared LKJ-prior node body (Stan `lkj_corr_cholesky_lpdf` op order,
# preserved verbatim from the host `lkj_corr_cholesky_logpdf`: constant
# literal first, then per-diagonal terms in row order — `(K-i)*log(L[i,i])`
# at `eta == 1.0`, else `a*log + b*log` with emission-time coefficients).
# Reads the named `_ppl_rl_` diagonal scalars — no matrix materializes, and
# the scalar-only form keeps the native Enzyme reverse pass on the same
# straight-line shape as every other prior. K=1 is the `0.0` literal
# (Stan's K=1 LKJ term is ±0.0 — no diagonal, zero constant).
# `nstack = S` is the stratified form: the diagonal edges are S-vectors
# (one entry per stratum), so the constant counts S times and each `log`
# term sums its vector.
function _lkj_prior_terms(L::Symbol, K::Int, eta::Float64;
        nstack::Union{Nothing,Int} = nothing, pointwise = false)
    K == 1 && return pointwise ? :(zeros($nstack)) : :(0.0)
    c = lkj_logconst(K, eta)
    terms = Any[pointwise ? :(fill($c, $nstack)) : nstack === nothing ? c : nstack * c]
    lg(i) = nstack === nothing ? :(log($(_rl_name(L, i, i)))) :
        pointwise ? :(log.($(_rl_name(L, i, i)))) : :(sum(log.($(_rl_name(L, i, i)))))
    for i in 2:K
        push!(terms, _lkj_prior_diagonal(lg(i), K, i, eta))
    end
    return foldl((a, c) -> pointwise ? :($a .+ $c) : :($a + $c), terms)
end

# Live eta uses the differentiable normalizer, including at eta == 1.
# Only the intrinsic factor width expands; stack/observation lengths are
# reductions over arrays. loggamma's adapters come from its owned math graph.
function _lkj_prior_terms(L::Symbol, K::Int, eta::Symbol;
        nstack::Union{Nothing,Int} = nothing)
    lgamma = GlobalRef(DistributionKernelSources, :loggamma)
    constant = :($(K - 1) * $lgamma($eta + $(0.5 * (K - 1))))
    for k in 1:(K - 1)
        constant = :($constant - $(0.5 * k * log(pi)) -
            $lgamma($eta + $(0.5 * (K - 1 - k))))
    end
    rhs = nstack === nothing ? constant : :($nstack * $constant)
    for i in 2:K
        diagonal = _rl_name(L, i, i)
        term = nstack === nothing ? :(log($diagonal)) : :(sum(log.($diagonal)))
        rhs = :($rhs + ($(K - i) + 2 * $eta - 2) * $term)
    end
    return :(if isfinite($eta) && $eta > 0
        $rhs
    else
        -Inf
    end)
end

function _lkj_prior_diagonal(ld, K, i, eta)
    coefficient = i isa Int ? K - i : :($K - $i)
    return eta == 1.0 ? :($coefficient * $ld) :
        :($coefficient * $ld + $(2 * eta - 2) * $ld)
end

# One plate over a latent vector parameter,
# summing the shared-prior log-density across cells. Every value the cell
# reads is threaded as a plate PORT (the vector plus each scalar prior
# arg) — captured free names are rejected by the `@kernel` plate
# expander, exactly as the Gaussian-likelihood scale is threaded.
function _vector_prior_stmts!(stmts::Vector{Expr}, terms::Vector{Any},
        name::Symbol, family::Symbol, args::NamedTuple,
        support::SupportOverride; conditioned = false, rows = nothing,
        float64_variate::Bool = false)
    node = Symbol(:_ppl_prior_, name)
    pw = Symbol(:_ppl_pw_prior_, name)
    if family === :flat
        # An improper density contributes zero; evaluating a posterior does
        # not request a draw from the unnormalizable prior.
        push!(stmts, :($pw = zero.($name)), :($node::Float64 = 0.0))
        push!(terms, node)
        return nothing
    end
    if rows === 0
        # An empty authored loop evaluates no prior arguments or cell body.
        # Keep its pointwise identity so conditioned declarations retain shape.
        push!(stmts, :($pw = zeros(0)), :($node::Float64 = 0.0))
        push!(terms, node)
        return nothing
    end
    inputs = Any[name]
    tv = _dovar(1)
    argvals = Any[_thread_ref!(inputs, v) for v in values(args)]
    cell = _family_logpdf_expr(family, argvals, tv; float64_variate)
    thread(x) = x isa Expr ? Expr(x.head,
        (i == 1 && x.head === :call ? a : thread(a) for (i, a) in enumerate(x.args))...) :
        _thread_ref!(inputs, x)
    threaded_support = support isa Tuple ?
        (support[1], map(thread, support[2:end])...) : support
    pre = Expr[]
    corr = _support_correction(family, threaded_support, argvals; pre, stem = name)
    corr === nothing || (cell = :($cell + $corr))
    if conditioned
        cell = _conditioned_support_guard(tv, threaded_support,
            isempty(pre) ? cell : Expr(:block, pre..., cell))
        empty!(pre)
    end
    append!(stmts, _plate_sum_stmts(pw, node, inputs, Expr[pre..., cell]))
    push!(terms, node)
    return nothing
end

# Sampled/prior family → distribution-kernel endpoint object. One row per
# family — a future family adds a row, never a branch.
const _PRIOR_ENDPOINTS = Dict{Symbol,Symbol}(
    :normal => :normal, :cauchy => :cauchy,
    :exponential => :exponential, :gamma => :gamma,
    :lognormal => :lognormal, :beta => :beta,
    :inverse_gamma => :inverse_gamma, :student_t => :student_t,
    :laplace => :laplace, :logistic => :logistic, :uniform => :uniform,
    :weibull => :weibull,
)

# Shared `<endpoint>(remapped args…).logpdf(x)` splice for a variate
# expression `x` (a scalar parameter name, a plate do-var, a scan
# setup/recurrence read, or a coefficient element read). Args are literals
# (inlined) or parameter/assignment/threaded refs (Distributions.jl
# semantics). Shared by scalar priors, per-cell plate priors, population
# priors, and the scan density. Gamma takes rate, so the contract's scale
# inverts; Flat() contributes zero. Explicit support is normalized below.
# Distribution kernels use Float64 ports. Promote at that boundary rather
# than redeclaring the source name: bound data and ordinary Julia helpers
# must retain their original numeric types. The same boundary applies to
# density and truncation endpoints, including threaded plate arguments.
# Multiplication by a floating unit preserves the value (and signed zero)
# through ordinary Julia numeric promotion and backend arithmetic. A numeric
# literal takes the same promotion (and Gamma's reciprocal) at emission. A
# variate the caller knows is a Float64 parameter (`float64_variate`) is read
# as is.
_float64_port(v) = :(1.0 * $v)
_float64_port(v::Union{Bool,Base.BitInteger,Base.IEEEFloat}) = 1.0 * v
_gamma_rate(v) = :(1 / $v)
_gamma_rate(v::Union{Bool,Base.BitInteger,Base.IEEEFloat}) = 1 / v

function _prior_endpoint_expr(family::Symbol, a, method::Symbol, x;
        float64_variate::Bool = false)
    ep = get(_PRIOR_ENDPOINTS, family, nothing)
    ep === nothing && throw(ContractValidationError(
        "[generator] prior family $family has no endpoint object"))
    args = family === :gamma ? (a[1], _gamma_rate(a[2])) : Tuple(a)
    args = map(_float64_port, args)
    return Expr(:call, Expr(:., Expr(:call, ep, args...), QuoteNode(method)),
        float64_variate ? x : _float64_port(x))
end

function _family_logpdf_expr(family::Symbol, a, x; float64_variate::Bool = false)
    family === :flat && return :(0.0)
    family === :binomial && return :((isfinite($x) && floor($x) == $x &&
        isfinite($(a[2])) && 0 <= $(a[2]) && $(a[2]) <= 1) ?
        binomial(Int($(a[1])), $(_float64_port(a[2]))).logpdf(Int($x)) : -Inf)
    return _prior_endpoint_expr(family, a, :logpdf, x; float64_variate)
end

# Normalize the base density over the declared support. Symmetric halves
# at literal zero add log(2); general bounds use the owned CDF endpoints.
function _support_correction(family::Symbol, ov::SupportOverride, argvals;
        pre::Vector{Expr} = Expr[], stem::Symbol = :prior)
    ov === nothing && return nothing
    ov isa Tuple && ov[1] === :restricted && return nothing
    ov isa Tuple && ov[1] === :restricted_half && return :(log(2))
    if ov isa Tuple && ov[1] === :truncated
        lo, hi = ov[2], ov[3]
        cdf(x) = _prior_endpoint_expr(family, argvals, :cdf, x)
        ccdf(x) = _prior_endpoint_expr(family, argvals, :ccdf, x)
        lo == -Inf && hi == Inf && return nothing
        lo == -Inf && return :(-log($(cdf(hi))))
        hi == Inf && return :(-log($(ccdf(lo))))
        # Use the upper tails when both endpoints are in that tail. The
        # branch is lazy, so an unused difference is never differentiated.
        fl, fh, sl, sh = (Symbol(:_ppl_tail_, stem, suffix)
            for suffix in (:_fl, :_fh, :_sl, :_sh))
        append!(pre, Expr[:($fl::Float64 = $(cdf(lo))),
            :($fh::Float64 = $(cdf(hi))), :($sl::Float64 = $(ccdf(lo))),
            :($sh::Float64 = $(ccdf(hi)))])
        # The CDF values are defined on both sides. Keep log itself lazy:
        # a rounded-to-zero inactive difference must never be logged or AD'd.
        return :($fl > 0.5 ? -log($sl - $sh) : -log($fh - $fl))
    end
    if ov isa Tuple
        if ov[1] === :lower
            # `truncated(LogNormal(m, s), lo, Inf)`: `-log P(X > lo)` =
            # `-log Φ((m - log(lo)) / s)`; nothing at `lo == 0`. A data
            # name `lo` is a bound kernel argument.
            lo = ov[2]
            lo isa Real && lo == 0 && return nothing
            m, s = argvals[1], argvals[2]
            tail = _prior_endpoint_expr(:normal, Any[:(-$m), s], :cdf, :(-log($lo)))
            return :(-log($tail))
        end
        if ov[1] === :upper
            family === :flat && return nothing
            return :(-log($(_prior_endpoint_expr(family, argvals, :cdf, ov[2]))))
        end
        ov[1] === :interval || throw(ContractValidationError(
            "[generator] tuple support override must be (:interval, lo, hi), " *
            "or (:upper, hi), got $ov"))
        family === :flat && return nothing  # improper: Jacobian only
        lo, hi = ov[2], ov[3]
        mu, s = argvals[1], argvals[2]
        upper = _prior_endpoint_expr(:normal, Any[mu, s], :cdf, hi)
        lower = _prior_endpoint_expr(:normal, Any[mu, s], :cdf, lo)
        return :(-log($upper - $lower))
    end
    ov === :positive || throw(ContractValidationError(
        "[generator] support override must be :positive, got $ov"))
    return :(log(2))  # :positive half
end

# One normalized scalar prior body, shared by parameter and hyper-prior slots.
function _sampled_prior_expr(p::SampledParameter; pre::Vector{Expr} = Expr[], conditioned = false)
    argvals = Any[v for v in values(p.args)]
    base = _family_logpdf_expr(p.family, argvals, p.name; float64_variate = !conditioned)
    local_pre = conditioned ? Expr[] : pre
    corr = _support_correction(p.family, p.support_override, argvals; pre = local_pre, stem = p.name)
    rhs = corr === nothing ? base : :($base + $corr)
    return conditioned ? _conditioned_support_guard(p.name, p.support_override,
        isempty(local_pre) ? rhs : Expr(:block, local_pre..., rhs)) : rhs
end

# Sampling transforms enforce support; observations use a lazy density guard.
function _untyped_branch_locals(ex::Expr)
    # Branch locals are ordinary Julia assignments. Their annotations would
    # convert traced numbers to Float64 during compiled execution; the enclosing
    # kernel node already supplies the result type. Endpoint calls such as
    # `normal(m, s).cdf(x)` stay source: ReactiveKernels lowers an object
    # endpoint under a branch arm inside that arm.
    if ex.head === :(=) && first(ex.args) isa Expr && first(ex.args).head === :(::)
        return Expr(:(=), first(first(ex.args).args),
            _untyped_branch_locals(ex.args[2]))
    end
    return Expr(ex.head, map(_untyped_branch_locals, ex.args)...)
end
_untyped_branch_locals(ex) = ex

function _conditioned_support_guard(value, support, density)
    support === nothing && return density
    valid = if support === :positive
        :($value >= 0)
    elseif support[1] === :lower
        :($value >= $(support[2]))
    elseif support[1] === :upper
        :($value <= $(support[2]))
    else
        :($(support[2]) <= $value && $value <= $(support[3]))
    end
    density = _untyped_branch_locals(density)
    return :(if $valid
        $density
    else
        -Inf
    end)
end

# Vector-latent prior node `_ppl_prior_<name>`: elementwise Normal for
# threshold/coefficient packs as one plate-sum over the vector (Stan
# `ordered`/`vector` semantics — no factorial normalizer, matching
# `_BRMThresholdPrior`), Dirichlet for simplexes as one broadcast (Stan
# `dirichlet_lpdf`: the log-multivariate-Beta normalizer folds host-side —
# data-only — plus Σ (α−1)·log(s); a symmetric concentration inlines one
# scalar, an asymmetric one its literal α−1 vector), and — for the
# structural joint factor, whose size is the joint outcome count —
# elementwise Exponential over the scale scalars (a literal scale inlines;
# a sampled hyperparameter resolves as a body local) and the shared LKJ
# node over the Cholesky scalars. None of the leveled (data-sized) forms
# grows with the vector length.
function _vector_parameter_prior_stmts!(stmts::Vector{Expr}, terms::Vector{Any},
        p::VectorParameter)
    m = p.size
    m === nothing && throw(ContractValidationError(
        "[generator] vector parameter $(p.name) has unresolved size " *
        "(bind_data infers it)"))
    node = Symbol(:_ppl_prior_, p.name)
    if haskey(_VECTOR_ELEMENT_FAMILIES, p.family)
        if m == 0
            push!(stmts, :($node::Float64 = 0.0))
            push!(terms, node)
            return nothing
        end
        _vector_prior_stmts!(stmts, terms, p.name,
            _VECTOR_ELEMENT_FAMILIES[p.family], p.args, nothing)
        return nothing
    end
    rhs = if p.family === :simplex_dirichlet
        alpha = p.args.arg1
        arg = alpha isa AbstractVector ? :(Float64[$(alpha...)]) : alpha
        :(_dirichlet_slices_logpdf(_SliceWhole(), $(p.name), $arg))
    elseif p.family === :cholesky_corr_lkj
        _lkj_prior_terms(p.name, m, Float64(p.args.arg1))
    elseif p.family === :positive_exponential
        th = p.args.arg1
        theta = th isa Symbol ? th : Float64(th)
        foldl((x, y) -> :($x + $y),
            Any[:(exponential($theta).logpdf($(_vector_elt_name(p.name, i))))
                for i in 1:m]; init = :(0.0))
    else
        throw(ContractValidationError(
            "[generator] vector parameter $(p.name) family $(p.family) " *
            "has no prior form"))
    end
    push!(stmts, :($node::Float64 = $rhs))
    push!(terms, node)
    return nothing
end

# --- Sequential-recurrence (scan) density (slice 1: CENTERED) ---

# Replace each `state[loopvar-j]` lag read with its aligned-slice do-var.
function _subst_scan_lags(ex, state::Symbol, loopvar::Symbol, dovar::Dict{Int,Symbol})
    ex isa Expr || return ex
    if ex.head === :ref && length(ex.args) == 2 && ex.args[1] === state
        idx = ex.args[2]
        if idx isa Expr && idx.head === :call && length(idx.args) == 3 &&
           idx.args[1] === :- && idx.args[2] === loopvar && idx.args[3] isa Int
            return dovar[idx.args[3]]
        end
    end
    return Expr(ex.head,
        (_subst_scan_lags(a, state, loopvar, dovar) for a in ex.args)...)
end

# Value symbols in a (lag-substituted) recurrence arg that must be threaded as
# plate inputs: captured scalars (params/assignments). Excludes call heads and
# the already-substituted lag do-vars (`_ppl_c…`).
function _scan_cell_caps!(caps::Vector{Symbol}, ex)
    if ex isa Symbol
        (startswith(string(ex), "_ppl_c") || ex in caps) || push!(caps, ex)
        return nothing
    end
    ex isa Expr || return nothing
    args = ex.head === :call ? ex.args[2:end] : ex.args
    for a in args
        _scan_cell_caps!(caps, a)
    end
    return nothing
end

# Replace symbols per `map` (leaves call heads alone — they never appear in the
# capture map).
function _subst_syms(ex, map::Dict{Symbol,Symbol})
    ex isa Symbol && return get(map, ex, ex)
    ex isa Expr || return ex
    return Expr(ex.head, (_subst_syms(a, map) for a in ex.args)...)
end

# --- Non-centered scan reconstruction (tuple carry) ---

_scan_where(s::ScanSpec) = "[generator] scan $(join(s.states, ", "))"

# The innovation steps of an emittable non-centered scan, in body order; a
# shape the emitter does not build (`_scan_shape_gap`) throws
# `ContractValidationError` naming the gap.
function _scan_innovations(s::ScanSpec)
    gap = _scan_shape_gap(s)
    gap === nothing || throw(ContractValidationError("[generator] " * gap))
    return ScanStep[st for st in s.step if st.kind === :sample]
end

# Positions in the latent slice: sampled seeds first (setup order), then each
# innovation's `n = T - m` per-step block (body order). Returns
# (seedpos::Dict{(state, index) => position}, innovation ranges).
function _scan_latent_positions(s::ScanSpec, innov, T::Int)
    seedpos = Dict{Tuple{Symbol,Int},Int}()
    pos = 0
    for f in s.setup
        f.kind === :sample || continue
        pos += 1
        seedpos[(f.target, f.index)] = pos
    end
    n = T - (s.lo - 1)
    ranges = UnitRange{Int}[(pos + (j - 1) * n + 1):(pos + j * n)
        for j in eachindex(innov)]
    return seedpos, ranges
end

# In-graph name of a seed's value (`_ppl_scan_init_<state>_<k>`).
_scan_init_name(a::Symbol, k::Int) = Symbol(:_ppl_scan_init_, a, :_, k)

# Translate a seed expression: an earlier seed `a[j]` becomes its value name;
# every other leaf must be a scalar parameter or definition.
function _scan_translate_seed(ex, s::ScanSpec, scalars)
    if ex isa Symbol
        ex in scalars || throw(ContractValidationError(
            "$(_scan_where(s)): seed leaf `$(ex)` is not a scalar parameter " *
            "or definition"))
        return ex
    end
    ex isa Expr || return ex
    if ex.head === :ref && length(ex.args) == 2 && ex.args[1] in s.states &&
            ex.args[2] isa Int
        return _scan_init_name(ex.args[1], ex.args[2])
    end
    args = ex.head === :call ? ex.args[2:end] : ex.args
    newargs = Any[_scan_translate_seed(a, s, scalars) for a in args]
    return ex.head === :call ?
        Expr(:call, ex.args[1], newargs...) : Expr(ex.head, newargs...)
end

# The carry window: each carried array read at backward lags keeps its last
# `L` values as `<state>_lag1 … <state>_lagL` fields of a NamedTuple carry.
_scan_lag_field(a::Symbol, k::Int) = Symbol(a, :_lag, k)

function _scan_windows(s::ScanSpec)
    lags = _scan_lags(s.step, s.states, s.loopvar)
    return [(a, isempty(lags[a]) ? 0 : max(s.maxlag, maximum(lags[a]))) for a in s.states]
end

# Translate a step expression into the reconstruction's do-block: a lag read
# `a[t - k]` becomes its carry field, a current read `a[t]` the value written
# earlier this step, a local its in-step binding, and every other leaf must be
# a scalar parameter or definition (threaded as a `Ref` and collected into
# `refs`). The contract never inspects step expressions, so these leaves are
# the only screen for hand-built plans — everything else fails closed.
function _scan_translate_step(ex, s::ScanSpec, env, refs::Set{Symbol}, scalars)
    where = _scan_where(s)
    if ex isa Symbol
        haskey(env.locals, ex) && return env.locals[ex]
        ex in s.states && throw(ContractValidationError(
            "$where: bare read of the carried array `$(ex)` — read the " *
            "backward lag `$(ex)[$(s.loopvar) - 1]`"))
        ex in scalars || throw(ContractValidationError(
            "$where: step leaf `$(ex)` is not a scalar parameter or " *
            "definition or supplied data value"))
        push!(refs, ex)
        return ex
    end
    ex isa Expr || return ex
    if ex.head === :ref && length(ex.args) == 2 && ex.args[1] in s.states
        a, idx = ex.args[1], ex.args[2]
        if idx === s.loopvar
            haskey(env.current, a) || throw(ContractValidationError(
                "$where: `$(a)[$(s.loopvar)]` is read before the step that " *
                "writes it"))
            return env.current[a]
        end
        k = _scan_lag_of(idx, a, s.loopvar)
        return Expr(:., :_ppl_carry, QuoteNode(_scan_lag_field(a, k)))
    end
    args = ex.head === :call ? ex.args[2:end] : ex.args
    newargs = Any[_scan_translate_step(a, s, env, refs, scalars) for a in args]
    return ex.head === :call ?
        Expr(:call, ex.args[1], newargs...) : Expr(ex.head, newargs...)
end

# The do-block body of one reconstruction: the steps in order, the next carry
# window, and `(next, <output>)`. Returns (body statements, sorted refs).
function _scan_step_block(s::ScanSpec, innov, output::Union{Symbol,Nothing}, scalars)
    refs = Set{Symbol}()
    env = (; locals = Dict{Symbol,Any}(s.loopvar => :_ppl_scan_index),
        current = Dict{Symbol,Symbol}())
    samples = Dict(st.target => Symbol(:_ppl_e, j) for (j, st) in enumerate(innov))
    body, logps = Expr[], Symbol[]
    for st in s.step
        rhs = if st.kind === :sample
            value = samples[st.target]
            if output === nothing
                args = [_scan_translate_step(a, s, env, refs, scalars) for a in st.args]
                lp = Symbol(:_ppl_scan_lp_, st.target)
                push!(body, :($lp::Float64 = $(_family_logpdf_expr(st.family, args, value))))
                push!(logps, lp)
            end
            value
        else
            _scan_translate_step(st.expr, s, env, refs, scalars)
        end
        if st.indexed
            nm = Symbol(:_ppl_scan_n_, st.target)
            push!(body, :($nm = $rhs))
            env.current[st.target] = nm
        else
            nm = Symbol(:_ppl_scan_l_, st.target)
            push!(body, :($nm = $rhs))
            env.locals[st.target] = nm
        end
    end
    fields = Expr[]
    for (a, L) in _scan_windows(s), k in 1:L
        val = k == 1 ? env.current[a] :
            Expr(:., :_ppl_carry, QuoteNode(_scan_lag_field(a, k - 1)))
        push!(fields, Expr(:(=), _scan_lag_field(a, k), val))
    end
    push!(body, :(_ppl_next = $(Expr(:tuple, fields...))))
    result = output === nothing ? foldl((a, b) -> :($a + $b), logps; init = 0.0) : env.current[output]
    push!(body, :((_ppl_next, $result)))
    return body, sort!(collect(refs))
end

function _scan_fold(s, innov, ranges, zname, T, output, scalars)
    m = s.lo - 1
    init = Expr[Expr(:(=), _scan_lag_field(a, k), _scan_init_name(a, m - k + 1))
        for (a, L) in _scan_windows(s) for k in 1:L]
    seqs = Any[:($(s.lo):$T), [:(view($zname, $(first(r)):$(last(r)))) for r in ranges]...]
    elems = Symbol[:_ppl_scan_index, [Symbol(:_ppl_e, j) for j in eachindex(innov)]...]
    body, refs = _scan_step_block(s, innov, output, scalars)
    lambda = Expr(:->, Expr(:tuple, :_ppl_carry, elems..., refs...), Expr(:block, body...))
    kw = Expr(:parameters, Expr(:kw, :init, Expr(:tuple, init...)))
    call = Expr(:call, :scan, kw, seqs..., (:(Ref($r)) for r in refs)...)
    return Expr(:do, call, lambda)
end

# Reconstruction statements for every non-centered scan, in plan order. The
# seeds bind first (`_ppl_scan_init_<a>_<k>`: a sampled seed reads its slice
# coordinate, a deterministic one evaluates its expression). Each carried
# array then folds its own RK-core `scan(...)` over the innovation blocks:
#   `rest = scan(view(z, r₁), …, Ref(params)…; init = (a_lag1 = …, …)) do
#        _ppl_carry, _ppl_e1, …, params…   # the steps, in order
#        (next, <a's new value>) end`
#   `a = vcat(<a's seeds>…, rest)`
# The carry is the tuple of every carried array's lag window, seeded from the
# last `L` seeds. One fold per carried array keeps every per-step output a
# scalar (the RK scan output contract under Reactant); the planner prunes a
# carried array nothing reads. Runs before predictors/likelihood (both may
# read a state); the latent prior stays in `_scan_prior_statements`.
function _scan_reconstruction_statements(plan::StructuralPlan,
        layout::LayoutTable)
    stmts = Expr[]
    scalars = union(_union_names(plan), collect(keys(plan.columns)))
    for raw in plan.scans
        s = _resolve_scan(plan, raw)
        _is_noncentered_scan(s) || continue
        innov = _scan_innovations(s)
        T = _scan_length(plan, s)
        zname = _scan_innovation_name(s)
        (_scan_latent_size(s, T) == 0 ||
            any(e -> e.kind === :scan && e.name === zname, layout.entries)) ||
            throw(ContractValidationError("$(_scan_where(s)): layout has no " *
                "latent slice :$zname"))
        seedpos, ranges = _scan_latent_positions(s, innov, T)
        for f in s.setup
            nm = _scan_init_name(f.target, f.index)
            val = f.kind === :sample ? :($zname[$(seedpos[(f.target, f.index)])]) :
                _scan_translate_seed(f.expr, s, scalars)
            push!(stmts, :($nm::Float64 = $val))
        end
        m = s.lo - 1
        for a in s.states
            if T == m
                seeds = [_scan_init_name(a, k) for k in 1:m]
                push!(stmts, :($a = [$(seeds...)]))
                continue
            end
            rest = Symbol(:_ppl_scan_rest_, a)
            push!(stmts, :($rest = $(_scan_fold(s, innov, ranges, zname, T, a, scalars))))
            seeds = [_scan_init_name(a, k) for k in 1:m]
            push!(stmts, :($a = vcat([$(seeds...)], $rest)))
        end
    end
    return stmts
end

# Seeds contribute their stated density once. The same ordered recurrence
# supplies each per-step sample's conditional density before binding its new
# value, so innovation scales may read carried values, locals, data and time.
function _scan_noncentered_prior!(stmts::Vector{Expr}, nodes::Vector{Symbol},
        s::ScanSpec, plan::StructuralPlan)
    s = _resolve_scan(plan, s)
    innov = _scan_innovations(s)
    T = _scan_length(plan, s)
    scalars = union(_union_names(plan), collect(keys(plan.columns)))
    zname = _scan_innovation_name(s)
    seedpos, ranges = _scan_latent_positions(s, innov, T)
    head = first(s.states)
    terms = Any[]
    for f in s.setup
        f.kind === :sample || continue
        lp = Symbol(:_ppl_scan_seedlp_, f.target, :_, f.index)
        args = Any[_scan_translate_seed(a, s, scalars) for a in f.args]
        push!(stmts, :($lp::Float64 = $(_family_logpdf_expr(f.family, args,
            :($zname[$(seedpos[(f.target, f.index)])])))))
        push!(terms, lp)
    end
    if T >= s.lo && !isempty(innov)
        pointwise = Symbol(:_ppl_scan_density_, head)
        node = Symbol(:_ppl_scan_stepslp_, head)
        push!(stmts, :($pointwise = $(_scan_fold(s, innov, ranges, zname, T, nothing, scalars))))
        push!(stmts, :($node::Float64 = sum($pointwise)))
        push!(terms, node)
    end
    total = Symbol(:_ppl_scan_, head)
    push!(stmts, :($total::Float64 = $(foldl((a, b) -> :($a + $b), terms; init = 0.0))))
    push!(nodes, total)
    return nothing
end

_scan_reads_carried(ex, s::ScanSpec) = ex isa Expr &&
    ((ex.head === :ref && !isempty(ex.args) && ex.args[1] in s.states) ||
     any(a -> _scan_reads_carried(a, s), ex.args))

# Prior-density statements for every scan plus the total-node names to add to
# the prior sum. Centered: the recurrence body is exactly one indexed `~` of
# the carried state (`state[t] ~ dist`); the density is the seed term(s) plus
# a plate over aligned lagged slices (the recurrence factorizes given the
# state). Non-centered: the iid-innovation prior over the `_ppl_scan_z_`
# slice (the plate-vector prior shape), with the state reconstructed in
# `_scan_reconstruction_statements`.
function _scan_prior_statements(plan::StructuralPlan, layout::LayoutTable)
    stmts = Expr[]
    nodes = Symbol[]
    for raw in plan.scans
        sc = _resolve_scan(plan, raw)
        if _is_noncentered_scan(sc)
            _scan_noncentered_prior!(stmts, nodes, sc, plan)
            continue
        end
        gap = _scan_shape_gap(sc)
        gap === nothing || throw(ContractValidationError("[generator] " * gap))
        state = only(sc.states)
        step = sc.step[1]
        entry = only(e for e in layout.entries
                     if e.kind === :scan && e.name === state)
        T = entry.size
        scalars = union(_union_names(plan), collect(keys(plan.columns)))
        m = length(sc.setup)
        terms = Any[]
        for (k, f) in enumerate(sc.setup)
            push!(stmts, :($(_scan_init_name(state, k)) = $state[$k]))
            seed = Symbol(:_ppl_scan_seed_, state, :_, k)
            push!(stmts, :($seed::Float64 =
                $(_family_logpdf_expr(f.family, [_scan_translate_seed(a, sc, scalars) for a in f.args], :($(state)[$k])))))
            push!(terms, seed)
        end
        lags = sort!(collect(_scan_lags([step], [state], sc.loopvar)[state]))
        inputs = Any[:(view($(state), $(m + 1):$T))]     # hcur → do-var _ppl_c1
        dovar = Dict{Int,Symbol}()
        for (i, j) in enumerate(lags)
            push!(inputs, :(view($(state), $(m + 1 - j):$(T - j))))
            dovar[j] = _dovar(i + 1)
        end
        # Substitute the lag reads, then thread the recurrence's captured scalars
        # (params/assignments) as explicit plate inputs — RK requires a plate
        # cell's distribution args to be caller ports, not lexical captures.
        lagargs = [_subst_scan_lags(a, state, sc.loopvar, dovar) for a in step.args]
        push!(inputs, :($(sc.lo):$T))
        lagargs = [_subst_syms(a, Dict(sc.loopvar => _dovar(length(inputs)))) for a in lagargs]
        caps = Symbol[]
        for a in lagargs
            _scan_cell_caps!(caps, a)
        end
        sc.loopvar in caps && throw(ContractValidationError(
            "[generator] scan $(state): the recurrence uses the loop index " *
            "`$(sc.loopvar)` directly — not supported in slice 1"))
        capmap = Dict{Symbol,Symbol}()
        for c in caps
            c in scalars || throw(ContractValidationError("$(_scan_where(sc)): unknown captured value $c"))
            push!(inputs, :(Ref($c)))
            capmap[c] = _dovar(length(inputs))
        end
        cellargs = [_subst_syms(a, capmap) for a in lagargs]
        cell = _family_logpdf_expr(step.family, cellargs, _dovar(1))
        recnode = Symbol(:_ppl_scan_rec_, state)
        recpw = Symbol(:_ppl_scan_pw_, state)
        append!(stmts, _plate_sum_stmts(recpw, recnode, inputs, cell))
        push!(terms, recnode)
        total = Symbol(:_ppl_scan_, state)
        push!(stmts, :($total::Float64 = $(foldl((a, b) -> :($a + $b), terms))))
        push!(nodes, total)
    end
    return stmts, nodes
end

function _log_jacobian_statement(plan::StructuralPlan, layout::LayoutTable)
    terms = Any[]
    for e in layout.entries
        t = jacobian_term(e)
        t === nothing || push!(terms, t)
    end
    jac = isempty(terms) ? :(0.0) : foldl((a, b) -> :($a + $b), terms)
    return :(log_jacobian::Float64 = $jac)
end
