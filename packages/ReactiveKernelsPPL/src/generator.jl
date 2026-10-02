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
# (counter-suffixed binding per build).

"""
    build_kernel(plan) -> (; spec, layout)

Validate, assign layout, emit, and evaluate a self-contained `@kernel`
program for `plan`. `spec` is the `KernelSpec` (callable after `prepare`
with `have=(:unconstrained, data…)`); `layout` is its
[`LayoutTable`](@ref) (R10 read API for the sampler side).

Thread safety: concurrent `build_kernel` calls over independent plans are
supported — the counter-suffixed `PPLGeneratedModels` binding is assigned
under a package-owned lock, so every build gets a distinct binding with no
caller-side synchronization. The returned spec closes over build-time
eval'd code: `prepare` it and call it through [`prepare_query`](@ref) /
[`prepare_sampler`](@ref) (which carry the `Base.invokelatest` world-age
barrier) or wrap those calls in `Base.invokelatest` yourself.
"""
function build_kernel(plan::StructuralPlan)
    validate_plan(plan)
    isbound(plan) || throw(ContractValidationError(
        "[generator] build_kernel requires a bound plan (bind_data first)"))
    layout = assign_layout(plan)
    def = kernel_expr(plan, layout)
    spec = _eval_kernel_def(def)
    return (; spec, layout)
end

"""
    kernel_expr(plan, layout; name=:ppl_model) -> Expr

The `@kernel` definition expression (`Expr(:(=), signature, body)`).
Pure (no eval): the generator tests inspect and evaluate it.
"""
function kernel_expr(plan::StructuralPlan, layout::LayoutTable; name::Symbol = :ppl_model)
    validate_plan(plan)
    isbound(plan) || throw(ContractValidationError(
        "[generator] kernel_expr requires a bound plan (bind_data first)"))
    stmts = Expr[]
    # Level gathers (`z[g]` over a `levels(h)` axis) read level-code
    # vectors; collect them from every expression before emitting.
    gathers = Set{Tuple{Symbol,Symbol}}()
    assigns = _assignment_statements(plan; gathers)
    priors = _prior_statements(plan, layout; gathers)
    _each_layout_unit(plan, layout) do unit
        append!(stmts, unit isa LayoutEntry ? transform_statements(unit) :
            _stratified_transform_statements(unit, layout))
    end
    append!(stmts, _coef_reassembly_statements(plan, layout))
    append!(stmts, _array_value_statements(plan))
    append!(stmts, _array_level_index_statements(plan, gathers))
    append!(stmts, assigns)
    append!(stmts, preprocessing_recipes(plan))
    append!(stmts, _varying_statements(plan))
    append!(stmts, _hsgp_basis_statements(plan))
    append!(stmts, _scan_reconstruction_statements(plan, layout))
    append!(stmts, _dar_reconstruction_statements(plan, layout))
    append!(stmts, _horseshoe_coef_statements(plan))
    append!(stmts, _affine_coefficient_statements(plan, layout))
    append!(stmts, _predictor_statements(plan))
    append!(stmts, _event_lp_statements(plan))
    append!(stmts, _likelihood_statements(plan))
    append!(stmts, priors)
    push!(stmts, _log_jacobian_statement(plan, layout))
    push!(stmts, :(posterior::Float64 = prior + likelihood + log_jacobian))
    push!(stmts, :(return posterior))
    sig = Expr(:call, name, :(unconstrained::Vector{Float64}),
        (_data_arg(colname, col) for (colname, col) in _ordered_columns(plan))...)
    return Expr(:(=), sig, Expr(:block, stmts...))
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
        if length(chunks) == 1 && any(t -> _parameter_term(t) &&
                t.kind in (FactorTerm, MatrixTerm), p.terms)
            t = only(t for t in p.terms if _parameter_term(t))
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

_data_arg(name::Symbol, col::AbstractVector) =
    Expr(:(::), name, Vector{eltype(col)})
_data_arg(name::Symbol, col::AbstractMatrix) =
    Expr(:(::), name, Matrix{eltype(col)})
_data_arg(name::Symbol, v::Number) = Expr(:(::), name, typeof(v))
_data_arg(name::Symbol, col::AbstractArray) =
    Expr(:(::), name, Array{eltype(col),ndims(col)})

# Dedicated eval scope for generated models. The `using` lines resolve via
# this package's own Project (by file location), so generated code loads in
# ANY consumer session with no LOAD_PATH dependence. One counter-suffixed
# binding per build; slice-1 scale makes interning harmless.
module PPLGeneratedModels
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, bernoulli, poisson, cauchy, exponential, gamma, lognormal,
    beta, inverse_gamma, binomial, negative_binomial2, beta_binomial,
    negative_binomial, weibull,
    uniform, laplace, logistic,
    student_t, zero_inflated_poisson, zero_inflated_binomial,
    gp_exp_quad_cov, gp_periodic_cov, gp_chol_latent,
    normal_id_glm, bernoulli_logit_glm, poisson_log_glm
using SpecialFunctions: besseli, besselix, erfc, loggamma
# Selective (explicit imports win over any re-export chain, so no `using`
# ambiguity if ReactiveKernels ever exports these too): the only Statistics
# names in the assignment allowlist.
using Statistics: mean, std, var
using LinearAlgebra: dot
using LogExpFunctions: log1pexp, logaddexp
# The inverse-logit MATH function under a distinct name: plain `logistic`
# here is the Logistic distribution kernel object (imported above).
import LogExpFunctions: logistic as _ppl_logistic
# Bijector objects the generated program splices (constrained-parameter
# transforms); imported from the enclosing module so the emitted
# `positive_bijector()` / `unit_bijector()` calls resolve.
import ..positive_bijector, ..unit_bijector
# In-model grouping encoder (`_ppl_gidx_<group>` nodes call it with the
# raw column + literal declared levels).
import .._declared_codes
# Stopping-ratio stage-lane tables (data-only recipes over the bound
# response; `preprocessing.jl`).
import .._ordinal_stage_obs, .._ordinal_stage_idx
# Grouped-kernel cell vocabulary: the subject-batched runners (one call
# per cell assignment over the bound op columns + `op_ends`, per-subject
# args marked `SubjectScalar` / `SubjectSlice`) and the cells they run.
# Every `import` here binds when this file loads — names defined by
# LATER includes (`tgi_segmented_nadir` and the per-element TGI
# likelihood cells the joint plates call) cannot register here; they
# import after their file loads (see the bottom of
# `ReactiveKernelsPPL.jl`).
import ..linear_pk_read_locs, ..linear_pk_read_locs_auc
import ..linear_pk_read_locs_over_subjects,
    ..linear_pk_read_locs_auc_over_subjects, ..SubjectScalar, ..SubjectSlice
import .._centered_correlated_logpdf
# Multivariate slice priors (`mv_slices.jl`): orientations, per-slice
# arguments, simplex / ordered slice transforms and the slice densities.
import .._SliceRows, .._SliceCols, .._SliceWhole, .._PerSlice
import .._mvnormal_cholesky_slices_logpdf, .._mvnormal_slices_logpdf
import .._dirichlet_slices_logpdf, .._ordered_normal_slices_logpdf
import .._simplex_slices_constrain, .._simplex_slices_logjac
import .._ordered_slices_constrain, .._ordered_slices_logjac
# Event-LP provider (one call over the flat event axis — the flat
# `log_F` local the batched cell runner slices per subject).
import ..linear_pk_event_log_f
end

const _MODEL_COUNTER = Ref(0)

# Package-owned binding lock: the counter increment plus the two
# `PPLGeneratedModels` evals are one critical section, so concurrent
# `build_kernel` calls always land on distinct bindings (an unsynchronized
# `Ref` increment drops updates under contention and two builds would
# silently share one binding — the second model's def wins and the first
# task reads back the wrong kernel).
const _MODEL_EVAL_LOCK = ReentrantLock()

function _eval_kernel_def(def::Expr)
    lock(_MODEL_EVAL_LOCK) do
        _MODEL_COUNTER[] += 1
        name = Symbol(:ppl_model_, _MODEL_COUNTER[])
        sig = def.args[1]
        renamed = Expr(:(=), Expr(:call, name, sig.args[2:end]...), def.args[2])
        call = Expr(:macrocall, Symbol("@kernel"), LineNumberNode(1, :generator),
            renamed)
        Core.eval(PPLGeneratedModels, call)
        return Core.eval(PPLGeneratedModels, name)
    end
end

# Scalar + derived assignments in topo order (params already constrained
# above, so every scalar name resolves; derived columns resolve as locals
# for the recipes below). Unannotated: Int temporaries (e.g. `length`)
# must not meet a Float64 assertion.
function _assignment_statements(plan::StructuralPlan;
        gathers::Set{Tuple{Symbol,Symbol}} = Set{Tuple{Symbol,Symbol}}())
    by_name = Dict{Symbol,Any}(a.name => a for a in plan.assignments)
    for d in plan.derived
        by_name[d.name] = d
    end
    # Data definitions calling module functions were evaluated once at
    # bind and arrive as data arguments; the kernel never recomputes them.
    computed = _bound_module_data_names(plan)
    dataonly = Set{Symbol}(keys(plan.columns))
    stmts = Expr[]
    for name in topological_order(plan)
        (haskey(by_name, name) && name ∉ computed) || continue
        ex = _array_gather_rewrite(by_name[name].expr, plan, gathers)
        if _expr_value_symbols(ex) ⊆ dataonly
            push!(dataonly, name)
        else
            ex = _split_data_calls!(stmts, name, ex, dataonly)
        end
        push!(stmts, :($(name) = $(ex)))
    end
    return stmts
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
        if any(b -> b.kind === MonotonicTerm, shape.blocks)
            append!(terms, _mo_block_terms(plan, shape))
        elseif shape.width > 0
            push!(terms, :($(design_name(pred.name)) * $(_affine_block_name(pred))))
        end
        if any(b -> b.kind === OffsetTerm, shape.blocks)
            push!(terms, offset_name(pred.name))
        end
        # A monotonic summand (mo1) contributes its contrast directly —
        # beta-free, the offset-arm shape with a parameter-derived column.
        for t in pred.terms
            t.kind === MonotonicSummandTerm &&
                push!(terms, monotonic_name(t.options.increments))
        end
        # A latent term contributes the per-cell latent VECTOR directly
        # (identity design): `lp = theta` on its own, or added to fixed-effect
        # design/offset terms for a random-intercept-plus-covariates predictor.
        for b in shape.blocks
            b.kind === LatentTerm && push!(terms, b.column)
        end
        # A spline summand contributes its basis's direct summand expression
        # (SB's `X*b + Z*(sd*z)` shape over materialized basis columns and
        # SplineVector layout blocks).
        for b in shape.blocks
            b.kind === SplineSummandTerm &&
                push!(terms, _spline_summand_expr(plan, b.column))
        end
        # An HSGP summand contributes its basis's direct `PHI * w`
        # expression (SB `_sb_hsgp`'s `PHI * (sqrt_spd .* beta_raw)`,
        # evaluated in-graph by `_hsgp_basis_statements`).
        for b in shape.blocks
            b.kind === HSGPSummandTerm &&
                push!(terms, _hsgp_summand_expr(plan, b.column))
        end
        # A varying effect contributes its draws block's direct `r`
        # expression (SB's `r_<target>_<suffix>` summand), resolved from
        # the TERMS — the draws label does not fit a design block.
        for t in pred.terms
            t.kind === VaryingEffectTerm &&
                push!(terms, _varying_effect_expr(plan, pred, t))
        end
        # A scan summand contributes its state's direct scaled expression
        # (`state .* coef`, SB's `ar` latent path with its free beta),
        # resolved from the TERMS like any summand.
        for t in pred.terms
            t.kind === ScanSummandTerm &&
                push!(terms, _scan_summand_expr(plan, pred, t))
        end
        # A dar summand contributes its trajectory state directly (bare,
        # beta-free — SB's `dar` zero-started path; the formula intercept
        # is the initial level), resolved from the TERMS like a scan.
        for t in pred.terms
            t.kind === DarSummandTerm &&
                push!(terms, _dar_summand_expr(plan, pred, t))
        end
        # A composed term evaluates its combination tree in-graph:
        # sub-predictors resolve to their LP nodes (emitted above —
        # contract orders subs first), scalars to their constrained
        # locals (params constrained above, assignments emitted above).
        # Plain broadcast math: ordinary reverse mode on every backend.
        for t in pred.terms
            t.kind === ComposedTerm &&
                push!(terms, _composed_expr(plan, pred, t))
        end
        # Degenerate (e.g. single-level-factor-only) predictors carry a scalar
        # zero LP, which broadcasts everywhere a vector LP would.
        rhs = isempty(terms) ? :(0.0) : foldl((a, b) -> :($a + $b), terms)
        push!(stmts, :($lp = $rhs))
    end
    return stmts
end

# One event-LP provider call (SB `log_F ~ 0 + op_log_dose +
# hsgp(op_log_dose; k)`): the flat op-ordered `log_F` local over the
# bound event-axis column + the frozen bind fit + the traced
# hyperparameters — the same `linear_pk_event_log_f` the host-side
# oracle path calls, so spec and graph agree by construction. Runs
# with the predictors (it IS an LP node); the grouped expansion
# slices the flat local per subject.
function _event_lp_statements(plan::StructuralPlan)
    stmts = Expr[]
    for el in plan.event_lps
        el.fit === nothing && throw(ContractValidationError(
            "[generator] event-LP `$(el.name)`: fit not filled at bind " *
            "(bind_data fits one (mu, L) over the event axis)"))
        mu, L = el.fit
        names = _event_lp_names(el)
        axis = _sched_col_name(el.schedule, :op_log_dose)
        push!(stmts, :($(el.name) = linear_pk_event_log_f($axis,
            $(names.slope), $(names.rho), $(names.sigma), $(names.beta),
            $(Float64(mu)), $(Float64(L)), $(el.k))))
    end
    return stmts
end

# Scalar coefficient-coordinate read (`sum(view(coef, k:k))`, the
# `coordinate_read` shape over a coefficient block rather than the packed
# vector).
_coef_coord(coef::Symbol, k::Int) = :(sum(view($coef, $k:$k)))

# Per-block LP terms for a predictor with `mo` columns. The fused
# `design * coef` matvec cannot cover a monotonic block — its contrast
# column is parameter-derived, and `hcat` cannot mix data with symbolic
# columns under the Enzyme reverse pass — so each coefficient-carrying
# block splices against its own coefficient coordinates (positions follow
# design order, the layout block's own order): intercept/continuous/
# monotonic blocks scale one column by one coordinate, factor and matrix
# blocks keep the data-matrix × coefficient-slice matvec. Predictors
# without `mo` keep the fused form above, untouched.
function _mo_block_terms(plan::StructuralPlan, shape::DesignShape)
    pred = only(p for p in plan.predictors if p.name === shape.predictor)
    coef = _affine_block_name(pred)
    terms = Any[]
    k = 1
    for b in shape.blocks
        if b.kind === InterceptTerm
            push!(terms, Expr(:call, :.*, Expr(:call, :ones, _predictor_rows(plan,shape.predictor)),
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
            push!(terms, :($(_matrix_block_expr(b, _predictor_rows(plan,shape.predictor))) *
                $(:(view($coef, $k:$(k + w - 1))))))
            k += w
        elseif b.kind === MonotonicTerm
            push!(terms,
                :($(monotonic_name(b.column)) .* $(_coef_coord(coef, k))))
            k += 1
        end
    end
    return terms
end

# One basis's direct summand as a scaled-column sum (SB `_sb_s_generic` /
# `_sb_t2_generic`): fixed blocks `X[j] .* b[j]`, pen blocks
# `Z[j] .* (sd[k] * r[j])`, all joined with `.+`. Reads the BOUND basis's
# materialized columns (bind asserted widths) and the `_spline_block_roles`
# vector names (the contract's single source — no re-derivation here).
function _spline_summand_expr(plan::StructuralPlan, id::Symbol)
    i = findfirst(b -> b.id === id, plan.spline_bases)
    i === nothing && throw(ContractValidationError(
        "[generator] spline summand addresses unknown basis :$id"))
    sb = plan.spline_bases[i]
    byblock = Dict{Symbol,SplineBasisBlock}(b.name => b for b in sb.blocks)
    roles, sd = _spline_block_roles(sb.id, sb.kind, sb.k)
    parts = Any[]
    for (block, coef, sdidx) in roles
        haskey(byblock, block) || throw(ContractValidationError(
            "[generator] spline :$id basis is missing block :$block"))
        cols = byblock[block].columns
        isempty(cols) && throw(ContractValidationError(
            "[generator] spline :$id block :$block has no materialized " *
            "columns (bind_data fills these)"))
        for (j, c) in enumerate(cols)
            cel = Expr(:call, :.*, c, Expr(:ref, coef, j))
            if sdidx !== nothing
                scaled = Expr(:call, :*, Expr(:ref, sd, sdidx),
                    Expr(:ref, coef, j))
                cel = Expr(:call, :.*, c, scaled)
            end
            push!(parts, cel)
        end
    end
    return foldl((a, b) -> :($a .+ $b), parts)
end

# `sqrt(2π)` verbatim from SB `brm_hsgp_sqrt_spd` (the spectral scale).
const _HSGP_SQRT2PI = 2.5066282746310002

# In-graph HSGP node names for one basis (all `_ppl_`-hygienic): per-axis
# trig columns, tensor-product columns, the `hcat` basis matrix, the
# spectral scale, per-basis `sqrt_spd` scalars, their `vect`, the
# spectral weights, and the predictor summand.
_hsgp_ax_name(id::Symbol, j::Int, k::Int) = Symbol(:_ppl_hsgp_, id, :_ax, j, :_k, k)
_hsgp_phi_name(id::Symbol, b::Int) = Symbol(:_ppl_hsgp_, id, :_phi_, b)
_hsgp_cos_name(id::Symbol, j::Int) = Symbol(:_ppl_hsgp_, id, :_cos_, j)
_hsgp_sin_name(id::Symbol, j::Int) = Symbol(:_ppl_hsgp_, id, :_sin_, j)
_hsgp_a_name(id::Symbol) = Symbol(:_ppl_hsgp_, id, :_a)
_hsgp_PHI_name(id::Symbol) = Symbol(:_ppl_hsgp_, id, :_PHI)
_hsgp_sscale_name(id::Symbol) = Symbol(:_ppl_hsgp_, id, :_sscale)
_hsgp_s_name(id::Symbol, b::Int) = Symbol(:_ppl_hsgp_, id, :_s_, b)
_hsgp_S_name(id::Symbol) = Symbol(:_ppl_hsgp_, id, :_S)
_hsgp_w_name(id::Symbol) = Symbol(:_ppl_hsgp_, id, :_w)
_hsgp_sum_name(id::Symbol) = Symbol(:_ppl_hsgp_, id)

# One basis's in-graph evaluation (SB `_brm_apply_hsgp` /
# `brm_hsgp_sqrt_spd` / `_sb_hsgp`, SB op order throughout): per-axis 1D
# trig columns from the frozen bind fits (`(mu, L)` literals), their
# tensor-product columns in `CartesianIndices(K)` order, the `hcat` basis
# matrix, unrolled `sqrt_spd` scalars over the sampled `(rho, sigma)`,
# and the spec-literal matmul summand `PHI * (S .* beta)`. The basis
# columns are data-only (bound-folded, the `design_recipe` precedent);
# the spectral weights stay symbolic. Runs before the predictors (the
# summand node is the LP splice); the priors stay in `_prior_statements`
# (order-free).
function _hsgp_basis_statements(plan::StructuralPlan)
    stmts = Expr[]
    for hb in plan.hsgp_bases
        if hb.cov === :periodic
            isempty(hb.fits) || throw(ContractValidationError(
                "[generator] hsgp :$(hb.id): periodic carries no fits " *
                "(no domain to fit)"))
            append!(stmts, _hsgp_periodic_stmts(hb))
        else
            length(hb.fits) == length(hb.axes) || throw(ContractValidationError(
                "[generator] hsgp :$(hb.id): fits not filled at bind " *
                "(bind_data fills one (mu, L) per axis)"))
            append!(stmts, hb.by === nothing ? _hsgp_basis_stmts(hb) :
                _hsgp_grouped_stmts(hb))
        end
    end
    return stmts
end

# One grouped basis (SB `_sb_hsgp_by` / `brm_hsgp_by_hyper_S`), one
# isotropic axis: the shared basis matrix `PHI` (n x M, frozen fits), the
# per-group length scales / marginal scales (a G-vector from the
# hyper-predictor `exp.(beta0 .+ sd .* z)` — length scales floored per
# group at the validity floor, SB `fmax(rho_g, rho_lower)` — or the
# shared scalar), the per-group spectral weights `SPD` (G x M, SB
# `brm_hsgp_sqrt_spd` per group), the per-group standardized weights
# `W = reshape(beta_raw, G, M)`, and the row-wise summand
# `sum(PHI .* (OH * (SPD .* W)); dims = 2)` over the data-only one-hot
# group matrix `OH` (n x G) — SB `rows_dot_product(S, beta[group_idx, :])`
# with its row mask, matmul-only so it traces through Reactant.
function _hsgp_grouped_stmts(hb::HSGPBasis)
    id = hb.id
    stmts = Expr[]
    axis = only(hb.axes)
    mu, L = Float64.(only(hb.fits))
    inv_sqrt_L = 1.0 / sqrt(L)
    K = only(hb.K)
    cols = Symbol[]
    for k in 1:K
        lam_sqrt = sqrt((k * pi / (2.0 * L))^2)
        col = _hsgp_ax_name(id, 1, k)
        push!(stmts, :($col =
            $inv_sqrt_L .* sin.($lam_sqrt .* ($axis .- $mu .+ $L))))
        push!(cols, col)
    end
    PHI = _hsgp_PHI_name(id)
    push!(stmts, :($PHI = hcat($(cols...))))
    names = _hsgp_names(hb)
    levels = hb.by.levels
    G = length(levels)
    gidx = Symbol(:_ppl_hsgp_, id, :_gidx)
    lvlvec = Expr(:vect, (_level_literal(lv) for lv in levels)...)
    push!(stmts, :($gidx = _declared_codes($(hb.by.column), $lvlvec)))
    OH = Symbol(:_ppl_hsgp_, id, :_OH)
    push!(stmts, :($OH = hcat($((:(Float64.($gidx .== $g)) for g in 1:G)...))))
    floor = only(_hsgp_floors(hb.K, hb.fits, hb.iso))
    function hyper_vec(h, floor)
        eta = h.intercept ? :($(h.beta0) .+ $(h.sd) .* $(h.z)) :
            :($(h.sd) .* $(h.z))
        v = :(exp.($eta))
        return floor > 0 ? :(max.($v, $floor)) : v
    end
    rho = names.rho_hyper === nothing ? only(names.rhos) :
        hyper_vec(names.rho_hyper, hb.rho_prior isa HSGPHyperLP ? floor : 0.0)
    sigma = names.sigma_hyper === nothing ? names.sigma :
        hyper_vec(names.sigma_hyper, 0.0)
    rv = Symbol(:_ppl_hsgp_, id, :_rho)
    sv = Symbol(:_ppl_hsgp_, id, :_sigma)
    push!(stmts, :($rv = $rho))
    push!(stmts, :($sv = $sigma))
    lamrow = Expr(:hcat, ((k * pi / (2.0 * L))^2 for k in 1:K)...)
    SPD = _hsgp_S_name(id)
    push!(stmts, :($SPD = ($sv .* sqrt.($rv .* $_HSGP_SQRT2PI)) .*
        exp.(-0.25 .* ($rv .* $rv) .* $lamrow)))
    W = _hsgp_w_name(id)
    push!(stmts, :($W = reshape($(names.beta), $G, $K)))
    push!(stmts, :($(_hsgp_sum_name(id)) =
        vec(sum($PHI .* ($OH * ($SPD .* $W)); dims = 2))))
    return stmts
end

function _hsgp_basis_stmts(hb::HSGPBasis)
    id = hb.id
    d = length(hb.axes)
    stmts = Expr[]
    # Per-axis 1D columns: `PHI[i,k] = inv_sqrt_L * sin(lam_sqrt[k] *
    # (x[i] - mu + L))` (SB `_brm_apply_hsgp`, element order verbatim).
    # `lam[k]` is SB's `lambda` literal, `lam_sqrt[k]` its `sqrt`.
    for (j, axis) in enumerate(hb.axes)
        mu, L = hb.fits[j]
        mu, L = Float64(mu), Float64(L)
        inv_sqrt_L = 1.0 / sqrt(L)
        for k in 1:hb.K[j]
            lam_sqrt = sqrt((k * pi / (2.0 * L))^2)
            col = _hsgp_ax_name(id, j, k)
            push!(stmts, :($col =
                $inv_sqrt_L .* sin.($lam_sqrt .* ($axis .- $mu .+ $L))))
        end
    end
    # Tensor-product columns in `CartesianIndices(K)` order (SB's
    # `enumerate(CartesianIndices(K))`): one axis reuses its column.
    midcs = collect(CartesianIndices(Tuple(hb.K)))
    phis = Symbol[]
    for (b, I) in enumerate(midcs)
        if d == 1
            push!(phis, _hsgp_ax_name(id, 1, I[1]))
        else
            phi = _hsgp_phi_name(id, b)
            cols = [_hsgp_ax_name(id, j, I[j]) for j in 1:d]
            push!(stmts, :($phi = $(foldl((a, c) -> :($a .* $c), cols))))
            push!(phis, phi)
        end
    end
    push!(stmts, :($(_hsgp_PHI_name(id)) = hcat($(phis...))))
    # Spectral weights (SB `brm_hsgp_sqrt_spd`): `scale = sigma *
    # prod(sqrt(rho_j * sqrt(2π)))`, `s[b] = scale * exp(-0.25 *
    # sum(rho_j^2 * omega2[b,j]))` — left-assoc folds, SB order. Iso
    # shares one rho across axes; `omega2` is the frozen `lambda`
    # literal above.
    names = _hsgp_names(hb)
    rhos = hb.iso ? fill(names.rhos[1], d) : names.rhos
    factors = Any[names.sigma]
    for j in 1:d
        push!(factors, :(sqrt($(rhos[j]) * $_HSGP_SQRT2PI)))
    end
    sscale = _hsgp_sscale_name(id)
    push!(stmts, :($sscale::Float64 = $(foldl((a, c) -> :($a * $c), factors))))
    snames = Symbol[]
    for (b, I) in enumerate(midcs)
        terms = Any[]
        for j in 1:d
            lam = (I[j] * pi / (2.0 * hb.fits[j][2]))^2
            push!(terms, :($(rhos[j]) * $(rhos[j]) * $lam))
        end
        expsum = foldl((a, c) -> :($a + $c), terms)
        s = _hsgp_s_name(id, b)
        push!(stmts, :($s::Float64 = $sscale * exp(-0.25 * $expsum)))
        push!(snames, s)
    end
    S = _hsgp_S_name(id)
    push!(stmts, :($S = $(Expr(:vect, snames...))))
    w = _hsgp_w_name(id)
    push!(stmts, :($w = $S .* $(names.beta)))
    push!(stmts, :($(_hsgp_sum_name(id)) = $(_hsgp_PHI_name(id)) * $w))
    return stmts
end

# One periodic basis's in-graph evaluation (SB
# `_brm_apply_hsgp_periodic` / `brm_hsgp_periodic_sqrt_spd` /
# `_sb_hsgp_periodic`): `k` harmonics of the fundamental angular
# frequency `w0 = 2π/period` as `2k` cosine/sine columns (cosines
# first, then sines — SB element order), unrolled `sqrt_spd` scalars
# over the sampled `(rho, sigma)`, and the spec-literal matmul
# summand `PHI * (S .* beta)`. The basis columns are data-only
# (bound-folded); the spectral weights stay symbolic.
#
# Spectral weights (SB `brm_hsgp_periodic_sqrt_spd`): with `a =
# 1/(rho*rho)`, `q_j = sigma*sqrt(2*exp(-a)*I_j(a))`. Stan evaluates
# this in log space through `log_modified_bessel_first_kind`;
# SpecialFunctions offers no log-Bessel, so the emission uses the
# exponentially scaled `besselix` (`I_j(a) = besselix(j,a)*exp(a)`,
# exact algebra): `q_b = exp(log(sigma) + 0.5*(log(2) +
# log(besselix(h_b, a))))`. Never overflows (the direct
# `sqrt(2*exp(-a)*besseli(j,a))` form throws AMOS for large `a`),
# Enzyme-clean (probed vs findiff). The harmonic index `h_b` is the
# frozen SB `harmonics` literal (`[1..k, 1..k]`).
function _hsgp_periodic_stmts(hb::HSGPBasis)
    id = hb.id
    x = only(hb.axes)
    k = only(hb.K)
    period = Float64(hb.period)
    stmts = Expr[]
    # Trig columns: `PHI[i,j] = cos(w0*j*x[i])`,
    # `PHI[i,k+j] = sin(w0*j*x[i])` (SB
    # `_brm_apply_hsgp_periodic`, element order verbatim). The
    # `w0*j` literal folds SB's `(w0*j)*x[i]` left-assoc product.
    w0 = 2.0 * pi / period
    coscols = Symbol[]
    sincols = Symbol[]
    for j in 1:k
        wj = w0 * j
        cc = _hsgp_cos_name(id, j)
        sc = _hsgp_sin_name(id, j)
        push!(stmts, :($cc = cos.($wj .* $x)))
        push!(stmts, :($sc = sin.($wj .* $x)))
        push!(coscols, cc)
        push!(sincols, sc)
    end
    push!(stmts, :($(_hsgp_PHI_name(id)) = hcat($(coscols...), $(sincols...))))
    # Spectral weights, one scalar per basis column over the frozen
    # harmonic index (SB `brm_hsgp_periodic_sqrt_spd` in scaled-log
    # space — see above).
    names = _hsgp_names(hb)
    rho = only(names.rhos)
    a = _hsgp_a_name(id)
    push!(stmts, :($a::Float64 = 1.0 / ($rho * $rho)))
    snames = Symbol[]
    for (b, h) in enumerate(vcat(1:k, 1:k))
        s = _hsgp_s_name(id, b)
        push!(stmts, :($s::Float64 = exp(log($(names.sigma)) +
            0.5 * (0.6931471805599453 + log(besselix($h, $a))))))
        push!(snames, s)
    end
    S = _hsgp_S_name(id)
    push!(stmts, :($S = $(Expr(:vect, snames...))))
    w = _hsgp_w_name(id)
    push!(stmts, :($w = $S .* $(names.beta)))
    push!(stmts, :($(_hsgp_sum_name(id)) = $(_hsgp_PHI_name(id)) * $w))
    return stmts
end

# One HSGP summand's direct expression: the basis's precomputed
# `_hsgp_basis_statements` node (resolved from the design block's basis
# id; the lookup below is loud defense in depth).
function _hsgp_summand_expr(plan::StructuralPlan, id::Symbol)
    any(hb -> hb.id === id, plan.hsgp_bases) || throw(ContractValidationError(
        "[generator] hsgp summand addresses unknown basis :$id"))
    return _hsgp_sum_name(id)
end

# One scan summand's direct expression (`state .* coef`, explicit dotted
# form): the in-graph recurrence state scaled by its sampled scalar
# coefficient, or the bare state for a beta-free summand (`coef ===
# nothing`). Both names resolve from the term's options (validated up
# front; the lookups below are loud defense in depth).
function _scan_summand_expr(plan::StructuralPlan, pred::PredictorSpec, t::TermSpec)
    o = t.options
    any(s -> o.scan_id in s.states, plan.scans) || throw(ContractValidationError(
        "[generator] scan summand in predictor $(pred.name) addresses " *
        "unknown scan :$(o.scan_id)"))
    o.coef === nothing && return o.scan_id
    any(p -> p.name === o.coef, plan.parameters) || throw(ContractValidationError(
        "[generator] scan summand coef :$(o.coef) is not a sampled parameter"))
    return Expr(:call, :.*, o.scan_id, o.coef)
end

# One dar summand's direct expression (the bare trajectory state): the
# in-graph `scan(...)` reconstruction is bound to the state's name by
# `_dar_reconstruction_statements`, so the LP splices the name itself.
# Validated up front; the lookup below is loud defense in depth.
function _dar_summand_expr(plan::StructuralPlan, pred::PredictorSpec, t::TermSpec)
    o = t.options
    any(s -> s.state === o.dar_id, plan.dar_paths) || throw(ContractValidationError(
        "[generator] dar summand in predictor $(pred.name) addresses " *
        "unknown dar :$(o.dar_id)"))
    return o.dar_id
end

# Composed elementwise maps → their generated-module bindings (`logistic`
# is the distribution object there; the math function is `_ppl_logistic`).
# Every other map keeps its head: a built-in math name resolves in the
# generated module, a module function is its `GlobalRef`.
const _COMPOSED_MAP_EMIT = Dict{Symbol,Symbol}(:exp => :exp,
    :logistic => :_ppl_logistic)

_composed_map_emit(f::Symbol) = get(_COMPOSED_MAP_EMIT, f, f)
_composed_map_emit(f) = f

"""Rewrite a composed tree to in-graph nodes (contract validated it)."""
function _composed_rewrite(node, subs::Vector{Symbol}, plan::StructuralPlan,
        pred::Symbol)
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

function _composed_expr(plan::StructuralPlan, pred::PredictorSpec, t::TermSpec)
    o = t.options
    ex = _composed_rewrite(o.tree, o.subs, plan, pred.name)
    # A composition over scalars only (a value location, `y .~
    # Normal.(mu, s)`) has no rows of its own: broadcast it over the rows
    # of the response it locates like an intercept (the `mo` intercept
    # shape), so every family emitter reads an ordinary LP vector.
    isempty(o.subs) && isempty(t.columns) || return ex
    return Expr(:call, :.*, Expr(:call, :ones, _located_rows(plan, pred)),
        ex)
end

# Rows of the responses a predictor locates — their own observation axis,
# which is not `n_obs` when responses observe different rows. Subject- or
# dose-level predictors keep their kernel rows.
function _located_rows(plan::StructuralPlan, pred::PredictorSpec)
    _predictor_level(plan, pred.name) === :obs ||
        return _predictor_rows(plan, pred.name)
    rows = unique!([_response_rows(plan, r) for r in plan.responses
        if _response_uses_predictor(r, pred.name)])
    length(rows) == 1 || throw(ContractValidationError("[generator] " *
        "predictor $(pred.name) locates responses with rows $rows " *
        "(one row count per scalar-valued predictor)"))
    return only(rows)
end

# Group-index encoder nodes, one per grouped column (`_ppl_gidx_<group>`):
# an in-model `_declared_codes` call over the draws' DECLARED levels
# (bind-known — filled or emitter-provided — so the order agrees with
# validation by construction; never sorted). Data-only, hence
# bound-folded; strings are native-only, exactly like factor contrasts.
# K=1 and correlated draws on the same group share one encoder
# (per-group dedup; same-group levels agreement is validated).
function _varying_statements(plan::StructuralPlan)
    stmts = Expr[]
    groups = Symbol[]
    for d in plan.varying_draws
        if d.mm !== nothing
            append!(stmts, _mm_preamble_stmts(d))
            continue
        end
        if !(d.group in groups)
            push!(groups, d.group)
            d.levels === nothing && throw(ContractValidationError(
                "[generator] internal: draws $(d.label) has no declared " *
                "levels (validate_plan proves this)"))
            lvlvec = Expr(:vect,
                (_level_literal(lv) for lv in d.levels)...)
            push!(stmts, Expr(:(=), Symbol(:_ppl_gidx_, d.group),
                Expr(:call, :_declared_codes, d.group, lvlvec)))
        end
        if d.strata !== nothing
            st = d.strata::VaryingStrata
            st.levels === nothing && throw(ContractValidationError(
                "[generator] internal: draws $(d.label) has no declared " *
                "strata levels (validate_plan proves this)"))
            slvlvec = Expr(:vect,
                (_level_literal(lv) for lv in st.levels)...)
            push!(stmts, Expr(:(=), _sidx_name(d),
                Expr(:call, :_declared_codes, st.by, slvlvec)))
        end
    end
    return stmts
end

# One membership slot's group-index encoder name (`_ppl_gidx_<suffix>_m<m>`).
_mm_gidx_name(d::VaryingDraws, m::Int) =
    Symbol(:_ppl_gidx_, d.suffix, :_m, m)

# Normalized-weight vector name (`_ppl_mmw_<suffix>_m<m>`) and the shared
# per-row weight-total name (`_ppl_mmwtot_<suffix>`).
_mm_w_name(d::VaryingDraws, m::Int) = Symbol(:_ppl_mmw_, d.suffix, :_m, m)
_mm_wtot_name(d::VaryingDraws) = Symbol(:_ppl_mmwtot_, d.suffix)

# Per-observation stratum-code vector name (`_ppl_sidx_<suffix>`).
_sidx_name(d::VaryingDraws) = Symbol(:_ppl_sidx_, d.suffix)

# Stratified draws in-graph: the layout keeps SB's per-stratum entries
# (all `L_<s>_s<k>`, then all `tau_<s>_s<k>` — names, coordinates and
# parity unchanged), but the stratum count S comes from data
# (`bind_data` fills the levels), so the graph reads them STACKED: one
# vine edge chain whose edges are S-vectors (`_ppl_rl_<sL>_<i>_<j>`,
# `L[1,1]` the scalar `1.0`) and one `tau` plate over the contiguous
# K×S column-major block. Statement count is independent of S
# (core constraint 1).
_strata_L_name(d::VaryingDraws) = Symbol(:_ppl_sL_, d.suffix)
_strata_tau_name(d::VaryingDraws) = Symbol(:_ppl_stau_, d.suffix)

# Iteration unit of the per-entry edges and Jacobian: every layout entry
# is its own unit, except a stratified draws block's per-stratum members,
# which form ONE unit (the draws block, visited at its first member).
function _each_layout_unit(f, plan::StructuralPlan, layout::LayoutTable)
    members = Dict{Symbol,VaryingDraws}()
    for d in plan.varying_draws
        d.strata === nothing && continue
        for k in 1:_strata_nlevels(d)
            Lk, tauk = _varying_strata_names(d, k)
            members[Lk] = d
            members[tauk] = d
        end
    end
    seen = Set{Symbol}()
    for e in layout.entries
        d = get(members, e.name, nothing)
        if d === nothing
            f(e)
        elseif !(d.label in seen)
            push!(seen, d.label)
            f(d)
        end
    end
    return nothing
end

# The stacked blocks' geometry, checked against the layout: S strata,
# K margins, P = K(K-1)/2 partials per stratum, and the offsets of the
# contiguous per-stratum `L` and `tau` runs the stacked reads stride over.
function _strata_geometry(d::VaryingDraws, layout::LayoutTable)
    S = _strata_nlevels(d)
    K = length(d.margins)
    P = K * (K - 1) ÷ 2
    byname = Dict(e.name => e for e in layout.entries)
    member(n) = haskey(byname, n) ? byname[n] :
        throw(ContractValidationError("[generator] internal: stratified " *
            "draws $(d.label) has no layout entry $n"))
    L1, tau1 = _varying_strata_names(d, 1)
    offL, offT = member(L1).offset, member(tau1).offset
    for k in 1:S
        Lk, tauk = _varying_strata_names(d, k)
        eL, eT = member(Lk), member(tauk)
        eL.offset == offL + (k - 1) * P && eL.size == P &&
            eT.offset == offT + (k - 1) * K && eT.size == K &&
            eT.transform === :exp || throw(ContractValidationError(
            "[generator] internal: stratified draws $(d.label) layout is " *
            "not the contiguous per-stratum L/tau runs the stacked edges read"))
    end
    return (; S, K, P, offL, offT)
end

# Partial p of every stratum's vine (stride P through the `L` run).
function _strata_partial_read(g, p::Int)
    lo = g.offL + p - 1
    return :(view(unconstrained, $lo:$(g.P):$(lo + (g.S - 1) * g.P)))
end

# The stacked `tau` block as one `:exp` plate entry (K×S column-major).
_strata_tau_entry(d::VaryingDraws, g) =
    LayoutEntry(:varying, nothing, _strata_tau_name(d),
        [_strata_tau_name(d)], g.offT, g.S * g.K, :exp)

function _stratified_transform_statements(d::VaryingDraws,
        layout::LayoutTable)
    g = _strata_geometry(d, layout)
    stmts = _lkj_vine_statements(_strata_L_name(d), g.K,
        p -> _strata_partial_read(g, p); stacked = true)
    append!(stmts, transform_statements(_strata_tau_entry(d, g)))
    return stmts
end

function _stratified_logjac_terms(d::VaryingDraws, layout::LayoutTable)
    g = _strata_geometry(d, layout)
    terms = Any[]
    lj = _lkj_vine_logjac(_strata_L_name(d), g.K; stacked = true)
    lj === nothing || push!(terms, lj)
    push!(terms, jacobian_term(_strata_tau_entry(d, g)))
    return terms
end

# One mm draws block's data preamble (SB `_brm_prepare_mm`, in-graph):
# per-slot encoders against the SHARED union levels (per-suffix names —
# never shared with plain encoders, whose numbering differs), plus —
# supplied weights with normalize — the per-row total (slot order, SB
# `sum` order) and the normalized per-slot weight vectors. Raw
# (`normalize=false`) weights read their bound columns directly (no
# preamble); default weights are the `inv(M)` scalar in the effect.
function _mm_preamble_stmts(d::VaryingDraws)
    mm = d.mm::VaryingMultiMembership
    M = length(mm.groups)
    d.levels === nothing && throw(ContractValidationError(
        "[generator] internal: draws $(d.label) has no declared " *
        "levels (validate_plan proves this)"))
    lvlvec = Expr(:vect,
        (_level_literal(lv) for lv in d.levels)...)
    stmts = Expr[]
    for m in 1:M
        push!(stmts, Expr(:(=), _mm_gidx_name(d, m),
            Expr(:call, :_declared_codes, mm.groups[m], lvlvec)))
    end
    if mm.weights !== nothing && mm.normalize
        tot = foldl((a, c) -> :($a .+ $c), mm.weights)
        push!(stmts, Expr(:(=), _mm_wtot_name(d), tot))
        for m in 1:M
            push!(stmts, Expr(:(=), _mm_w_name(d, m),
                :($(mm.weights[m]) ./ $(_mm_wtot_name(d)))))
        end
    end
    return stmts
end

# One mm slot's weight factor as an rvalue: the `inv(M)` literal for
# default weights (SB fills `inv(M)` unnormalized), the normalized
# preamble vector, or the raw bound column.
function _mm_weight_expr(d::VaryingDraws, m::Int)
    mm = d.mm::VaryingMultiMembership
    mm.weights === nothing && return inv(Float64(length(mm.groups)))
    mm.normalize && return _mm_w_name(d, m)
    return mm.weights[m]
end

# Term-to-draws join by label, plus the (draws, target) slice (unique
# by validation). The term carries the draws label; the slice carries
# the explicit column range — the generator never re-derives ranges.
function _slice_draws(plan::StructuralPlan, pred::PredictorSpec,
        t::TermSpec)
    i = findfirst(d -> d.label === t.options.draws, plan.varying_draws)
    i === nothing && throw(ContractValidationError(
        "[generator] effect term addresses unknown draws " *
        "($(t.options.draws))"))
    d = plan.varying_draws[i]
    si = findfirst(s -> s.draws === d.label && s.target === pred.name,
        plan.varying_slices)
    si === nothing && throw(ContractValidationError(
        "[generator] internal: effect term of draws $(d.label) in " *
        "predictor $(pred.name) has no slice (validate_plan proves this)"))
    return d, plan.varying_slices[si]
end

# One varying margin's Z as an rvalue: bare columns stay bare (raw
# ports and derived locals alike); dummies compare against the
# bind-known level value. `:ones` never reaches emission (an intercept
# needs no Z multiply).
function _varying_z_expr(z::VaryingZRecipe)
    z.kind === :column && return z.column
    z.kind === :dummy &&
        return Expr(:call, :.==, z.column, _level_literal(z.level))
    throw(ContractValidationError(
        "[generator] internal: ones-Z reached effect emission"))
end

# One varying draws block's direct `r` summand (no `b` node, the draws
# stay implicit): the K² implicit-draws arm (this slice's columns
# only, every K); mm draws take the weighted-gather arm (same geometry,
# per-slot gathers); centered draws read their sampled `b_flat`.
function _varying_effect_expr(plan::StructuralPlan, pred::PredictorSpec,
        t::TermSpec)
    d, s = _slice_draws(plan, pred, t)
    d.mm !== nothing && return _varying_mm_effect_expr(plan, d, s)
    d.kind === :centered_correlated && return _varying_centered_effect_expr(d,s)
    return _varying_corr_effect_expr(plan, d, s)
end

function _varying_centered_effect_expr(d::VaryingDraws,s::VaryingSlice)
    b = _varying_corr_names(d)[3]
    K = length(d.margins)
    gidx = Symbol(:_ppl_gidx_,d.group)
    parts = Any[]
    for j in s.columns
        idx = :($j .+ ($gidx .- 1) .* $K)
        effect = Expr(:ref,b,idx)
        z = d.margins[j].z
        push!(parts,z.kind === :ones ? effect : :($(_varying_z_expr(z)) .* $effect))
    end
    return foldl((a,c)->:($a .+ $c),parts)
end

# One mm draws block's direct `r` summand (SB `multi_membership_*`
# math, in-graph): per slot `m`, the plain-geometry margin expr at
# that slot's encoder, weighted by that slot's factor, summed in slot
# order (SB's per-observation `rv[i] += w*b` association). Every slot
# reuses the margin expr below with the shared tau/L/z.
function _varying_mm_effect_expr(plan::StructuralPlan, d::VaryingDraws,
        s::VaryingSlice)
    mm = d.mm::VaryingMultiMembership
    M = length(mm.groups)
    L, tau, _ = _varying_corr_names(d)
    parts = Any[]
    for m in 1:M
        gidx = _mm_gidx_name(d, m)
        w = _mm_weight_expr(d, m)
        inner = _corr_margin_expr(d, s, tau, L, gidx)
        push!(parts, :($w .* $inner))
    end
    return foldl((a, c) -> :($a .+ $c), parts)
end

# One correlated draws block's direct `r` summand for one slice
# (SB `rows_dot_product(Z, b[idx,cols])` with the draws implicit — the
# no-`b`-node precedent); stratified draws take the per-stratum
# indicator arm below instead.
function _varying_corr_effect_expr(plan::StructuralPlan, d::VaryingDraws,
        s::VaryingSlice)
    d.strata !== nothing && return _varying_strata_effect_expr(plan, d, s)
    L, tau, _ = _varying_corr_names(d)
    return _corr_margin_expr(d, s, tau, L, Symbol(:_ppl_gidx_, d.group))
end

# One stratified draws block's direct `r` summand (SB
# `ranef_correlated_by` math, in-graph): each observation's margin
# coefficient `tau[j]*L[j,q]` is GATHERED from its own stratum's frame
# by the per-observation stratum code — `tau` from the stacked K×S
# column-major vector (the `z_flat` gather idiom), `L[j,q]` from the
# stacked S-vector edge (the `xi[gidx]` idiom; `L[1,1]` is the scalar
# `1.0`). One expression whatever the stratum count S, which `bind_data`
# fills from data: per-stratum arms would duplicate the body S times
# (core constraint 1). Each gathered product is the same two-operand
# multiply the per-stratum frame performs, so values are unchanged.
function _varying_strata_effect_expr(plan::StructuralPlan, d::VaryingDraws,
        s::VaryingSlice)
    K = length(d.margins)
    sidx = _sidx_name(d)
    stau = _strata_tau_name(d)
    sL = _strata_L_name(d)
    coef(j, q) = begin
        Ljq = (j == 1 && q == 1) ? _rl_name(sL, 1, 1) :
            Expr(:ref, _rl_name(sL, j, q), sidx)
        :($(Expr(:ref, stau, :($j .+ ($sidx .- 1) .* $K))) .* $Ljq)
    end
    return _corr_margin_expr(d, s, coef, Symbol(:_ppl_gidx_, d.group))
end

# Per slice margin j, `Z_j .* sum_s (tau[j]*L[j,s]) .*
# z_flat[s + (gidx-1)*K]` over `s in 1:j` (L lower-triangular — the
# `s > j` terms are structural zeros, never emitted). `:ones` Z drops
# the factor (multiply by 1). K, the slice range, and the `s` bound
# are all static; tau reads are scalar refs (the coefficient-block
# precedent) and L reads the named `_ppl_rl_` scalars from the layout
# edges. Shared by the plain, mm (per-slot `gidx`), and stratified
# (gathered `coef`) arms.
_corr_margin_expr(d::VaryingDraws, s::VaryingSlice, tau::Symbol,
        L::Symbol, gidx::Symbol) =
    _corr_margin_expr(d, s,
        (j, q) -> :($(Expr(:ref, tau, j)) * $(_rl_name(L, j, q))), gidx)

# `coef(j, q)` is the margin coefficient expression `tau[j]*L[j,q]`.
function _corr_margin_expr(d::VaryingDraws, s::VaryingSlice, coef,
        gidx::Symbol)
    cols = s.columns
    K = length(d.margins)
    z = _varying_corr_names(d)[3]
    parts = Any[]
    for j in cols
        m = d.margins[j]
        inner = Any[]
        for q in 1:j
            A = coef(j, q)
            idx = :($q .+ ($gidx .- 1) .* $K)
            push!(inner, :($A .* $(Expr(:ref, z, idx))))
        end
        sj = foldl((a, c) -> :($a .+ $c), inner)
        if m.z.kind === :ones
            push!(parts, sj)
        else
            push!(parts, :($(_varying_z_expr(m.z)) .* $sj))
        end
    end
    return foldl((a, c) -> :($a .+ $c), parts)
end

_lik_name(label::Symbol) = Symbol(:_ppl_lik_, label)

function _likelihood_statements(plan::StructuralPlan)
    stmts = Expr[]
    terms = Any[]
    for r in plan.responses
        append!(stmts, _response_likelihood_stmts(r, plan))
        push!(terms, _lik_name(r.label))
    end
    for kp in plan.kernel_plates
        kstmts, kterm = _kernel_plate_likelihood(kp, plan)
        append!(stmts, kstmts)
        push!(terms, kterm)
    end
    joint = foldl((a, b) -> :($a + $b), terms; init = :(0.0))
    push!(stmts, :(likelihood::Float64 = $joint))
    return stmts
end

# Panel-kernel likelihood (flat codegen): panel-v1 cell bodies are
# elementwise, so the subject map dissolves into flat vector ops over the
# `n_sub*T` block (numerically identical to a per-subject loop for the
# admitted subset): slice params rewrite to flat column refs (scalar
# slices to their bind-time T-block expansions), cell assignments emit as
# flat statements, and the single Gaussian obs lowers as a flat plate
# reusing the plate-sum machinery. The collected name aliases its flat
# value (future generated quantities read it).
function _panel_kernel_likelihood(kp::KernelPlate, plan::StructuralPlan)
    isbound(plan) ||
        throw(ContractValidationError("[generator] kernel plates lower " *
              "from a bound plan (bind_data first)"))
    kp.subjects isa Int ||
        throw(ContractValidationError("[generator] kernel plate " *
              "`$(kp.result)` subjects unresolved (bind_data with dims first)"))
    flatmap = _kernel_flatmap(kp)
    stmts = Expr[]
    for (nm, ex) in _canonicalize_kernel_assignments(kp)
        push!(stmts, :($nm = $(_rewrite_kernel_refs(ex, flatmap))))
    end
    obs = only(kp.obs)
    obs.family in _KERNEL_SCALAR_FAMS ||
        throw(ContractValidationError("[generator] kernel plate " *
              "`$(kp.result)` obs family $(obs.family) has no panel " *
              "emitter (admitted: Normal, Bernoulli, Poisson, " *
              "NegativeBinomial2, Gamma, Beta, StudentT)"))
    klabel = Symbol(:kernel_, kp.result)
    rcol = _kernel_obs_rcol(kp, obs, flatmap[obs.response], plan)
    append!(stmts, _kernel_scalar_obs_stmts(obs, rcol,
        flatmap, plan, klabel, _pw_name(klabel), _lik_name(klabel)))
    collected = haskey(flatmap, kp.collected) ? flatmap[kp.collected] : kp.collected
    collected === kp.result ||
        push!(stmts, :($(kp.result) = $collected))
    return stmts, _lik_name(klabel)
end

# One scalar response-space in-cell observation → its plate statements
# (panel and grouped share the emitter — the grouped Gaussian arm was
# the panel arm's near-duplicate). Cell spellings are the
# mixture/plate endpoint precedents over the obs node's response-space
# (location, scale, params). `rcol` is the response plate input (flat
# for panel, whole-column for grouped); `label` scopes precomputes
# (Gamma rate); slice-param args ride `flatmap`; cell locals, model
# scalars, and literals pass through (`_thread_ref!`: symbols ride as
# inputs — the `_ppl_lp_` precedent — literals inline).
function _kernel_scalar_obs_stmts(obs::KernelObs, rcol::Symbol,
        flatmap::Dict{Symbol,Symbol}, plan::Union{StructuralPlan,Nothing},
        label::Symbol, pw::Symbol, node::Symbol)
    fam = obs.family
    loc = _kernel_obs_ref(obs.location, flatmap)
    scale = obs.scale === nothing ? nothing :
        _kernel_obs_ref(obs.scale, flatmap)
    params = map(p -> _kernel_obs_ref(p, flatmap), obs.params)
    inputs = Any[rcol]
    rv = _dovar(1)
    pre = Expr[]
    cell = if fam === GaussianFam
        locv = _thread_ref!(inputs, loc)
        sref = _thread_ref!(inputs, scale)
        :(normal($locv, $sref).logpdf($rv))
    elseif fam === BernoulliLogitFam
        pv = _thread_ref!(inputs, loc)
        :(bernoulli($pv).logpdf($rv))
    elseif fam === PoissonLogFam
        muv = _thread_ref!(inputs, loc)
        :(poisson($muv).logpdf($rv))
    elseif fam === NegativeBinomial2Fam
        muv = _thread_ref!(inputs, loc)
        phiref = _thread_ref!(inputs, scale)
        :(negative_binomial2($muv, $phiref).logpdf($rv))
    elseif fam === GammaLogFam
        # Surface is Distributions-SCALE `Gamma(alpha, scale)`; the
        # kernel takes rate — invert at the boundary (the Gamma-plate
        # precedent; literals fold, symbols precompute once outside
        # the plate over layout-legal refs).
        aref = _thread_ref!(inputs, loc)
        ratev = if scale isa Symbol
            rate = _rate_name(label)
            push!(pre, :($rate = 1 ./ $scale))
            _thread_ref!(inputs, rate)
        else
            1.0 / Float64(scale)
        end
        :(gamma($aref, $ratev).logpdf($rv))
    elseif fam === BetaLogitFam
        avv = _thread_ref!(inputs, loc)
        bvv = _thread_ref!(inputs, scale)
        :(beta($avv, $bvv).logpdf($rv))
    elseif fam === StudentTFam
        nuv = _thread_ref!(inputs, loc)
        muv = _thread_ref!(inputs, scale)
        sigv = _thread_ref!(inputs, params[1])
        :(student_t($nuv, $muv, $sigv).logpdf($rv))
    else
        throw(ContractValidationError(
            "[generator] in-cell obs family $fam has no scalar emitter"))
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# In-cell obs response column: Bernoulli plates read Bool lanes — a
# non-Bool flat redirects to its bind-materialized Bool twin (exact
# for 0/1; the `!=` comparison misdifferentiates under native Enzyme —
# snag `bernoulli-int-la-78487520`). All other families read `rcol`.
function _kernel_obs_rcol(kp::KernelPlate, obs::KernelObs, rcol::Symbol,
        plan::Union{StructuralPlan,Nothing})
    obs.family === BernoulliLogitFam || return rcol
    plan === nothing && throw(ContractValidationError(
        "[generator] Bernoulli in-cell obs needs the bound plan " *
        "(response eltype check — internal: pass plan)"))
    haskey(plan.columns, rcol) || throw(ContractValidationError(
        "[generator] Bernoulli in-cell response `$rcol` is not bound"))
    eltype(plan.columns[rcol]) === Bool && return rcol
    twin = _kbool_name(kp.result, obs.response)
    haskey(plan.columns, twin) || throw(ContractValidationError(
        "[generator] Bernoulli in-cell Bool twin `$twin` missing " *
        "(bind_data materializes it for non-Bool responses)"))
    return twin
end

# One grouped cell assignment — always ONE emitted statement, whatever
# the subject count (the generated program's statement count is O(1) in
# the data; only the layout, the bound columns and runtime loop trip
# counts scale with it): CELL_FN calls emit the subject-batched runner
# over the bound `op_ends` + op columns with marked per-subject args;
# segmented-nadir calls emit `tgi_segmented_nadir` over the bound ends
# column; gathers rewrite `v[sched.map]` to `vflat[mapcol]`; slice
# do-params rewrite to their bound columns (kernel ports are column
# names — the panel flatmap precedent); everything else emits verbatim
# (bind proved shapes).
function _grouped_cell_assignment(nm::Symbol, ex, kp::KernelPlate,
        sched::PKScheduleSpec, lps::Dict{Symbol,Symbol},
        columns::Dict{Symbol,ColumnData}, flatmap::Dict{Symbol,Symbol})
    if ex isa Expr && ex.head === :call && !isempty(ex.args) &&
            ex.args[1] isa Symbol && ex.args[1] in CELL_FNS
        return _expand_grouped_cell_call(nm, ex, sched, lps, flatmap)
    end
    if ex isa Expr && ex.head === :call && !isempty(ex.args) &&
            ex.args[1] isa Symbol && ex.args[1] in SEGMENT_CELL_FNS
        return _expand_segmented_nadir_call(nm, ex, columns, flatmap)
    end
    return Expr[:($nm = $(_rewrite_grouped_gather(ex, sched, lps, flatmap)))]
end

# Segmented nadir: the surface call emits verbatim over the bound ends
# column — `tgi_segmented_nadir` (tgi.jl) runs the per-segment running
# minimum as a plain eltype-generic loop (native and Enzyme) or one
# `_rectangular_fold` over all rows with host-built reset flags
# (Reactant — a retained traced loop, never a trace-time unroll).
# Empty segments are the function's own concern (it returns an empty
# block for them).
function _expand_segmented_nadir_call(nm::Symbol, ex::Expr,
        columns::Dict{Symbol,ColumnData}, flatmap::Dict{Symbol,Symbol})
    # The change vector may be a bare response slice (shapes admit
    # slices — `(:obs, len)`); slice do-params ride their columns.
    change = _kernel_obs_ref(ex.args[2], flatmap)
    endscol = ex.args[3]
    haskey(columns, endscol) || throw(ContractValidationError(
        "[generator] segmented nadir ends column `$endscol` is not bound"))
    return Expr[:($nm = $(ex.args[1])($change, $endscol))]
end

function _rewrite_grouped_gather(ex, sched::PKScheduleSpec,
        lps::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}(),
        flatmap::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}())
    ex isa Symbol && return get(flatmap, ex, ex)
    ex isa Expr || return ex
    if ex.head === :ref && length(ex.args) == 2
        vec, idx = ex.args[1], ex.args[2]
        # LP cell params in gather-source position rewrite to their LP
        # vectors (structure proved bare LPs reach generation ONLY as
        # gather sources — every other bare use fails closed there —
        # so any surviving bare LP elsewhere passes through to a loud
        # UndefVar instead of a silent wrong vector). The LP check
        # leads: LP params and slice do-params are disjoint, so order
        # never matters, but the LP rule must not depend on it.
        if vec isa Symbol && haskey(lps, vec)
            vec = lps[vec]
        else
            vec = _rewrite_grouped_gather(vec, sched, lps, flatmap)
        end
        if idx isa Expr && idx.head === :.
            m = idx.args[2].value
            return Expr(:ref, vec, _sched_col_name(sched.name, m))
        end
        # Plain-symbol (or computed) flat indices — TGI/QT prep maps and
        # other bind-materialized integer columns — pass through untouched.
        return Expr(:ref, vec,
            _rewrite_grouped_gather(idx, sched, lps, flatmap))
    end
    return Expr(ex.head,
        (_rewrite_grouped_gather(a, sched, lps, flatmap)
            for a in ex.args)...)
end

# The subject-batched cell statement: `<fn>_over_subjects(<sched>_op_ends,
# <sched>_<opfield>..., args...)` (pkcells.jl), each extra arg marked by
# its per-subject access — LP cell params `SubjectScalar(<lp vector>)`
# (entry `s`), flat event-frame vectors `SubjectSlice(v)` (the subject's
# op range), everything else verbatim.  The runner slices the op columns
# at runtime from the bound `op_ends`, so the statement is the same for
# one subject or ten thousand.
_over_subjects_name(fn::Symbol) = Symbol(fn, :_over_subjects)

function _expand_grouped_cell_call(nm::Symbol, ex::Expr,
        sched::LinearPKScheduleSpec, lps::Dict{Symbol,Symbol}, flatmap)
    fn = ex.args[1]
    callargs = ex.args[2:end]
    opfields = CELL_FN_OP_FIELDS[fn]
    sliced = get(CELL_FN_SLICED_ARGS, fn, Symbol[])
    args = Any[_sched_col_name(sched.name, :op_ends)]
    for field in opfields
        push!(args, _sched_col_name(sched.name, field))
    end
    for a in callargs[2:end]
        if a isa Symbol && haskey(lps, a)
            push!(args, :(SubjectScalar($(lps[a]))))
        elseif a isa Symbol && a in sliced
            push!(args, :(SubjectSlice($a)))
        else
            push!(args, a)
        end
    end
    return Expr[:($nm = $(Expr(:call, _over_subjects_name(fn), args...)))]
end

# Grouped-kernel likelihood: the panel flat map cannot express sequential
# recurrences, so each cell assignment emits ONE subject-batched call —
# `linear_pk_read_locs*_over_subjects` over the bound op columns +
# `op_ends` with per-subject LP vectors (`SubjectScalar`) and flat
# event-frame vectors (`SubjectSlice`) — whose runtime loop runs the
# per-subject event recurrence and concatenates the reads flat; schedule-
# map gathers move reads to obs space, and each in-cell observation
# lowers as a Gaussian plate reusing the plate-sum machinery. Cell calls
# always rewrite to the batched spelling (never emit verbatim — a
# verbatim schedule handle has no runtime binding); slice do-params
# rewrite to their bound columns (kernel ports are column names — the
# panel flatmap precedent); all other assignments emit verbatim under
# their surface names. The collected name aliases its flat value (future
# generated quantities + the sibling likelihood-node slice read it).
#
# Slice params to columns (grouped `_kernel_flatmap`: grouped slices
# are all `:response` kind over whole columns — no T-blocks — so the
# map is do-param → column).
_grouped_flatmap(kp::KernelPlate) =
    Dict{Symbol,Symbol}(p => c for (c, p, _) in kp.slices)

function _grouped_kernel_likelihood(kp::KernelPlate, plan::StructuralPlan)
    isbound(plan) ||
        throw(ContractValidationError("[generator] kernel plates lower " *
              "from a bound plan (bind_data first)"))
    kp.subjects isa Int ||
        throw(ContractValidationError("[generator] kernel plate " *
              "`$(kp.result)` subjects unresolved (bind_data with dims first)"))
    sched = only(kp.schedules)
    # The batched cell runner reads the subject ranges from this bound
    # column at runtime; it must be a kernel port (bind materializes it).
    haskey(plan.columns, _sched_col_name(sched.name, _sched_ends_field(sched))) ||
        throw(ContractValidationError("[generator] schedule " *
              "`$(sched.name)` has no bound subject-ends column"))
    lps = Dict{Symbol,Symbol}(c => _lp_name(_predictor(plan, p))
        for (p, c) in kp.lp_args)
    flatmap = _grouped_flatmap(kp)
    stmts = Expr[]
    for (nm, ex) in kp.assignments
        append!(stmts, _grouped_cell_assignment(nm, ex, kp, sched, lps,
            plan.columns, flatmap))
    end
    klabel = Symbol(:kernel_, kp.result)
    oterms = Any[]
    for (oi, obs) in enumerate(kp.obs)
        rcol = only(c for (c, p, _) in kp.slices if p === obs.response)
        olabel = Symbol(klabel, :_o, oi)
        ostmts, oterm =
            _grouped_obs_likelihood_stmts(kp, obs, rcol, olabel, plan)
        append!(stmts, ostmts)
        push!(oterms, oterm)
    end
    joint = foldl((a, b) -> :($a + $b), oterms; init = :(0.0))
    push!(stmts, :($(_lik_name(klabel))::Float64 = $joint))
    collected = haskey(flatmap, kp.collected) ? flatmap[kp.collected] :
        kp.collected
    collected === kp.result ||
        push!(stmts, :($(kp.result) = $collected))
    return stmts, _lik_name(klabel)
end

# One grouped in-cell observation → `(stmts, term)`: scalar obs
# (PK-QT-TGI continuous alike) ride the shared scalar emitter; the
# joint families route to their builders with the obs node's
# `(response, location, scale, params)` mapped to builder kwargs (the
# surface arity table + contract family checks proved the shapes, so
# the positional map below is total). QT Gaussian obs route through
# the SHARED path (the KernelObs node cannot carry the QT builder's
# separate weight — the surface spells `qt_sd = qt_scale .*
# qt_weight` pre-assignments instead; the QT builder stays the golden
# shape spec the emitter output is pinned to). `plan` threads the
# bound columns for the Bernoulli eltype check (direct unit calls
# over non-Bernoulli obs pass `nothing`).
function _grouped_obs_likelihood_stmts(kp::KernelPlate, obs::KernelObs,
        rcol::Symbol, olabel::Symbol, plan::Union{StructuralPlan,Nothing} = nothing)
    flatmap = _grouped_flatmap(kp)
    if obs.family in _KERNEL_SCALAR_FAMS
        pw, node = _pw_name(olabel), _lik_name(olabel)
        rcol2 = _kernel_obs_rcol(kp, obs, rcol, plan)
        return _kernel_scalar_obs_stmts(obs, rcol2, flatmap, plan, olabel,
            pw, node), node
    end
    # Obs location/scale/params naming slice do-params ride their bound
    # columns (kernel ports are column names — the panel
    # `_kernel_obs_ref` precedent); cell locals, model scalars, and
    # literals pass through.
    loc = _kernel_obs_ref(obs.location, flatmap)
    scale = _kernel_obs_ref(obs.scale, flatmap)
    params = map(p -> _kernel_obs_ref(p, flatmap), obs.params)
    if obs.family === CensoredAddpropnormalFam
        # `pk_obs_statement` spelling: `(location, scale = add, prop,
        # lloq)` — all names (the QT builder threads; literals spell
        # a pre-assignment).
        for (nm, ref) in ((:location, loc), (:scale, scale),
                (:params, params[1]), (:params, params[2]))
            ref isa Symbol ||
                throw(ContractValidationError("[generator] kernel plate " *
                      "`$(kp.result)` censored obs $nm `$ref` must be a " *
                      "cell/model name (literals do not lower — spell " *
                      "a pre-assignment)"))
        end
        return _qt_joint_pk_likelihood_stmts(; response = rcol,
            location = loc, add = scale, prop = params[1],
            lloq = params[2], label = olabel)
    elseif obs.family === TgiCategoryFam
        return tgi_category_stmts(; response = rcol, r = loc,
            ref = scale, c_cr = params[1], c_pr = params[2],
            c_pd = params[3], sigma = params[4], eps = params[5],
            label = olabel)
    elseif obs.family === TgiResponseFam
        return tgi_response_stmts(; response = rcol, r = loc,
            ref = scale, c_pr = params[1], c_pd = params[2],
            sigma = params[3], eps = params[4], label = olabel)
    elseif obs.family === TgiCensoredFam
        return tgi_censored_stmts(; response = rcol, mu = loc,
            sigma = scale, lloq = params[1], label = olabel)
    end
    throw(ContractValidationError("[generator] kernel plate " *
          "`$(kp.result)` obs family $(obs.family) has no in-cell " *
          "emitter (admitted: Normal, Bernoulli, Poisson, " *
          "NegativeBinomial2, Gamma, Beta, StudentT, " *
          "CensoredAddpropnormal, TgiCategory, TgiResponse, TgiCensored)"))
end

function _kernel_plate_likelihood(kp::KernelPlate, plan::StructuralPlan)
    _is_grouped_kernel(kp) && return _grouped_kernel_likelihood(kp, plan)
    return _panel_kernel_likelihood(kp, plan)
end

# Slice params to flat refs: vector slices ride their flat T-blocked
# column; scalar slices ride the bind-time T-block expansion (or the raw
# column in all-scalar models, where no expansion exists).
function _kernel_flatmap(kp::KernelPlate)
    flatmap = Dict{Symbol,Symbol}()
    for (col, param, kind) in kp.slices
        kind in (:vector, :scalar) ||
            throw(ContractValidationError("[generator] kernel plate " *
                  "`$(kp.result)` slice `$param` kind unresolved " *
                  "(bind_data first)"))
        flatmap[param] =
            (kind === :vector || kp.timepoints === nothing) ? col :
            _kexp_name(kp.result, col)
    end
    return flatmap
end

# Obs location/scale through the flatmap (slice params only); cell
# locals, globals, and literals pass to `_thread_ref!` unchanged.
_kernel_obs_ref(ref, flatmap::Dict{Symbol,Symbol}) =
    ref isa Symbol && haskey(flatmap, ref) ? flatmap[ref] : ref

function _rewrite_kernel_refs(ex, flatmap::Dict{Symbol,Symbol})
    ex isa Symbol && return get(flatmap, ex, ex)
    ex isa Expr || return ex
    return Expr(ex.head, (_rewrite_kernel_refs(a, flatmap) for a in ex.args)...)
end

# One plate likelihood per response (pointwise plate + scalar sum node).
# Triples 2 and 3 (Bernoulli-logit) lower identically; the triple only
# selects the form. Branches are explicit per family; the else is a
# fail-closed guard for enum members without an emitter (never silent).
function _response_likelihood_stmts(r::LikelihoodSpec, plan::StructuralPlan)
    node = _lik_name(r.label)
    pw = _pw_name(r.label)
    if r.mi_jobs !== nothing && !(r.family === GaussianFam ||
            r.family === GammaLogFam || r.family === BetaLogitFam)
        throw(ContractValidationError(
            "[generator] mi() response $(r.label) family $(r.family) " *
            "has no mi emitter (v1: Gaussian/Gamma/Beta)"))
    end
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
        # must never silently outgrow a future partial range).
        r.evidence.kind === :none && r.weights === nothing &&
            r.range === nothing && !_is_bare_param_location(r, plan) &&
            return _bernoulli_wholevec_stmts(r, plan, node)
        return _bernoulli_plate_stmts(r, plan, node, pw)
    elseif r.family === PoissonLogFam
        # Base GLM case (no evidence, no weights, no literal range): fused
        # whole-vector reduction (faster native + Reactant; the per-cell
        # plate handles evidence/weights/ranges).
        r.evidence.kind === :none && r.weights === nothing &&
            r.range === nothing && !_is_bare_param_location(r, plan) &&
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
    elseif r.family === WeibullFam
        return _weibull_plate_stmts(r, plan, node, pw)
    elseif r.family === GammaLogFam
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
# the max-shifted log-sum-exp over `logw_k + lpdf_k` (categorical-plate
# precedent: linear in K). Predictor locations ride their LP nodes
# (link-space, inverted per the component link like the single-family
# builders); sampled params thread scalar (broadcast) and literals inline
# (both constrained-scale, no inversion). K=1 uses the general form
# (exact: m=t1, log(exp(0))=0).
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
    # Binomial trials are shared across components: one threaded use.
    nref = nothing
    if f === BinomialLogitFam
        nref = _thread_ref!(inputs, r.trials, true)
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
    for k in 1:K
        klab = Symbol(r.label, :_mix, k)
        locref, is_lp = _mixture_loc_ref(r, plan, k)
        lpdf = _mixture_component_lpdf(f, r, plan, pre, inputs, k, klab,
            locref, is_lp, yv, yref, nref)
        logw_k = logw_lit === nothing ?
            (logw_comp === nothing ? :($lwv[$k]) : logw_comp[k]) :
            logw_lit[k]
        push!(terms, :($logw_k + $lpdf))
    end
    m = terms[1]
    for t in terms[2:end]
        m = :(max($m, $t))
    end
    sumexp = :(exp($(terms[1]) - $m))
    for t in terms[2:end]
        sumexp = :($sumexp + exp($t - $m))
    end
    cell = :($m + log($sumexp))
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
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
        klab::Symbol, locref, is_lp::Bool, yv::Symbol, yref, nref)
    if f === GaussianFam
        sarg = _scale_use_plate_arg(r, plan, pre, r.mixture_scales[k], klab)
        locv = _thread_ref!(inputs, locref)
        sref = _thread_ref!(inputs, sarg)
        return :(normal($locv, $sref).logpdf($yv))
    elseif f === BernoulliLogitFam
        if is_lp
            etav = _thread_ref!(inputs, locref)
            return :(bernoulli(; logit = $etav).logpdf($yref))
        end
        p = _thread_ref!(inputs, locref)
        return :(bernoulli($p).logpdf($yref))
    elseif f === PoissonLogFam
        if is_lp
            etav = _thread_ref!(inputs, locref)
            return :(poisson(; log_rate = $etav).logpdf($yv))
        end
        rate = _thread_ref!(inputs, locref)
        return :(poisson($rate).logpdf($yv))
    elseif f === BinomialLogitFam
        if is_lp
            etav = _thread_ref!(inputs, locref)
            return :(binomial(; n = $nref, logit = $etav).logpdf($yv))
        end
        p = _thread_ref!(inputs, locref)
        return :(binomial($nref, $p).logpdf($yv))
    elseif f === NegativeBinomial2Fam
        sarg = _scale_use_plate_arg(r, plan, pre, r.mixture_scales[k], klab)
        muv = if is_lp
            mu = _mu_name(klab)
            push!(pre, :($mu = exp.($locref)))
            _thread_ref!(inputs, mu)
        else
            _thread_ref!(inputs, locref)
        end
        phiref = _thread_ref!(inputs, sarg)
        return :(negative_binomial2($muv, $phiref).logpdf($yv))
    elseif f === GammaLogFam
        sarg = _scale_use_plate_arg(r, plan, pre, r.mixture_scales[k], klab)
        # Surface is Distributions-SCALE `Gamma(alpha, mu/alpha)`; the
        # kernel takes rate, so the boundary inverts (the Gamma-plate
        # precedent).
        av = sarg isa Symbol ? sarg : Float64(sarg)
        rate = _rate_name(klab)
        if is_lp
            push!(pre, :($rate = $av ./ exp.($locref)))
        else
            push!(pre, :($rate = $av ./ $locref))
        end
        ratev = _thread_ref!(inputs, rate)
        aref = _thread_ref!(inputs, sarg)
        return :(gamma($aref, $ratev).logpdf($yv))
    elseif f === BetaLogitFam
        sarg = _scale_use_plate_arg(r, plan, pre, r.mixture_scales[k], klab)
        kap = sarg isa Symbol ? sarg : Float64(sarg)
        mu = _mu_name(klab)
        muhandle = locref
        if is_lp
            push!(pre, :($mu = 1 ./ (1 .+ exp.(-$locref))))
            muhandle = mu
        end
        a = _shape_a_name(klab)
        b = _shape_b_name(klab)
        push!(pre, :($a = $muhandle .* $kap))
        push!(pre, :($b = (1 .- $muhandle) .* $kap))
        avv = _thread_ref!(inputs, a)
        bvv = _thread_ref!(inputs, b)
        return :(beta($avv, $bvv).logpdf($yv))
    end
    throw(ContractValidationError(
        "[generator] mixture over $f has no cell emitter"))
end

# A GLM-object response: one fused constructed-endpoint application
# over the whole column (no plate — the object owns eta). The
# intercept-free design matrix gains its ones column from the bound
# row count (data-only, folds at prepare) and the split coefficients
# rejoin as `beta_full = [alpha; beta]` (the validated P2 spelling).
function _glm_object_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol,
        pw::Symbol)
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
    return Expr[
        :($yf = $yconv.($y)),
        :($xaug = hcat(ones($(plan.n_obs)), $X)),
        :($bfull = [$(r.glm_alpha); $(r.glm_beta)]),
        :($pw = $call),
        :($node::Float64 = sum($pw)),
    ]
end

_predictor(plan::StructuralPlan, name::Symbol) =
    only(p for p in plan.predictors if p.name === name)

# The per-observation location node feeding a response's likelihood plate: a
# scan-state or per-cell (plate) latent vector fed directly (its own name —
# the layout view), or a linear predictor's `_ppl_lp_<name>` node otherwise.
function _location_node(r::LikelihoodSpec, plan::StructuralPlan)
    any(s -> r.predictor in s.states, plan.scans) && return r.predictor
    _is_plate_param(plan, r.predictor) && return r.predictor
    return _lp_name(_predictor(plan, r.predictor))
end

# A bare sampled-parameter location (constrained-scale, no link inversion):
# `r.predictor` names a scalar parameter, not a PredictorSpec.
_is_bare_param_location(r::LikelihoodSpec, plan::StructuralPlan) =
    any(p -> p.name === r.predictor, plan.parameters)

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
# node the emitter created — always a vector), returning the short node.
# The gather is its own short plate over `Jobs` with the source `Ref`'d
# (the ordinal `c[yv]` per-lane-gather precedent): a caller-level fancy
# `node[Jobs]` does not trace under Reactant (`TracedRArray[Vector{Int}]`
# shape-inference failure), while per-lane scalar gathers do.
function _mi_gather_node!(pre::Vector{Expr}, jobs::Symbol, node::Symbol,
        label::Symbol)
    g = _mi_gather_name(label, node)
    jv, rf = _dovar(1), _dovar(2)
    body = Expr(:block, LineNumberNode(0, :generator), :($rf[$jv]))
    lambda = Expr(:(->), Expr(:tuple, jv, rf), body)
    doex = Expr(:do, Expr(:call, :plate, jobs, :(Ref($node))), lambda)
    push!(pre, :($g = $doex))
    return g
end

# Gather a scale-like ref under `mi()`: scalar parameter/assignment names
# broadcast untouched, columns gather through a short plate, Real
# literals pass through for `_thread_ref!` to inline; anything else fails
# closed (gathering a scalar would index nonsense, an unknown name would
# thread garbage).
function _mi_gather_ref!(pre::Vector{Expr}, jobs::Symbol, ref,
        plan::StructuralPlan, label::Symbol)
    ref isa Real && return ref
    ref isa Symbol || throw(ContractValidationError(
        "[generator] mi() response $label gathers Symbol/Real refs only " *
        "(got $(repr(ref)))"))
    ref in _union_names(plan) && return ref
    haskey(plan.columns, ref) || throw(ContractValidationError(
        "[generator] mi() response $label cannot gather unknown name $ref"))
    return _mi_gather_node!(pre, jobs, ref, label)
end

# Gather a resolved scale arg under `mi()`: a predictor-fed scale already
# resolved to its `_ppl_sc_` node (always a full-length vector — gather
# it as a node); every other scale shape routes through `_mi_gather_ref!`.
function _mi_gather_scale!(pre::Vector{Expr}, jobs::Symbol, sarg, r::LikelihoodSpec,
        plan::StructuralPlan)
    r.scale isa ScalePredictorRef &&
        return _mi_gather_node!(pre, jobs, sarg, r.label)
    return _mi_gather_ref!(pre, jobs, sarg, plan, r.label)
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

# Thread Symbol bounds (do-vars), inline Real bounds; nothing stays nothing.
function _thread_bounds!(inputs::Vector{Any}, ev::ResponseEvidence, as_int::Bool)
    lb = ev.lower === nothing ? nothing : _thread_ref!(inputs, ev.lower, as_int)
    ub = ev.upper === nothing ? nothing : _thread_ref!(inputs, ev.upper, as_int)
    return (lb, ub)
end

function _gaussian_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    pre = Expr[]
    sarg = _scale_plate_arg(r, plan, pre)
    if r.mi_jobs !== nothing
        # Packed y_obs threads directly (it IS the short plate axis);
        # every other vector input gathers by Jobs.
        lp = _mi_gather_node!(pre, r.mi_jobs, lp, r.label)
        sarg = _mi_gather_scale!(pre, r.mi_jobs, sarg, r, plan)
    end
    inputs = Any[y, lp]
    yv, lpv = _dovar(1), _dovar(2)
    sref = _thread_ref!(inputs, sarg)
    lb, ub = _thread_bounds!(inputs, r.evidence, false)
    base = :(normal($lpv, $sref).logpdf($yv))
    cell = _gaussian_cell(r.evidence.kind, base, yv, lb, ub, lpv, sref)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

function _gaussian_cell(kind::Symbol, base::Expr, yv::Symbol, lb, ub, lpv::Symbol, sref)
    kind === :none && return base
    nccdf(b) = :(normal($lpv, $sref).cdf($b))
    if kind === :truncated
        corr = if lb === nothing && ub === nothing
            return base
        elseif lb === nothing
            :(log($(nccdf(ub))))
        elseif ub === nothing
            :(log(1.0 - $(nccdf(lb))))
        else
            :(log($(nccdf(ub)) - $(nccdf(lb))))
        end
        return :($base - $corr)
    elseif kind === :censored
        # Clamp law (Y = clamp(X)): at-bound rows are censored observations,
        # so the arms are non-strict (SB parity pair mod-weights).
        if lb === nothing && ub === nothing
            return base
        elseif lb === nothing
            return :(ifelse($yv >= $ub, log1p(-$(nccdf(ub))), $base))
        elseif ub === nothing
            return :(ifelse($yv <= $lb, log($(nccdf(lb))), $base))
        else
            return :(ifelse($yv <= $lb, log($(nccdf(lb))),
                ifelse($yv >= $ub, log1p(-$(nccdf(ub))), $base)))
        end
    else # :interval_censored
        return :(log($(nccdf(ub)) - $(nccdf(yv))))
    end
end

# Student-t plate: the Gaussian shape with a df argument — validation
# guarantees `nu` (a sampled name, a literal, or a predictor-fed
# per-observation nu) and sigma; evidence arms mirror the Gaussian
# clamp law over the `student_t` cdf (`_student_cell`), plus optional
# weights. A predictor-fed nu binds its own `_ppl_sc_<label>_nu` node
# (the `_nu`-suffixed label cannot collide with any `<lhs>_resp` scale
# node), so scale and nu predictors coexist.
function _student_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    pre = Expr[]
    sarg = _scale_plate_arg(r, plan, pre)
    nuarg = _scale_use_plate_arg(r, plan, pre, r.nu, Symbol(r.label, :_nu))
    inputs = Any[y, lp]
    yv, lpv = _dovar(1), _dovar(2)
    sref = _thread_ref!(inputs, sarg)
    nuv = _thread_ref!(inputs, nuarg)
    lb, ub = _thread_bounds!(inputs, r.evidence, false)
    base = :(student_t($nuv, $lpv, $sref).logpdf($yv))
    cell = _student_cell(r.evidence.kind, base, yv, lb, ub, nuv, lpv, sref)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# LogNormal likelihood (Stan `lognormal_lpdf` mirror): the cell calls
# the prior-proven `lognormal` distribution endpoint
# (`LOGNORMAL_KERNEL_SOURCE`, Distributions.jl `LogNormal(mu, sigma)`
# order) — no closed-form respelling, so SB parity is by shared
# operation order. Mu is the raw location node (identity link, the
# Student precedent); sigma threads scalar via `_scale_plate_arg`
# (predictor-fed sigma is deferred at the contract gate, the
# Beta-kappa precedent). The endpoint's lazy `y > 0` guard returns
# -Inf off support; `y` is bound data so preparation splits the plate
# per taken arm and no backend receives the branch. Evidence fails
# closed at the contract gate (Gaussian/Poisson only); weights
# multiply the cell (the NB2 precedent).
function _lognormal_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    pre = Expr[]
    sarg = _scale_plate_arg(r, plan, pre)
    inputs = Any[y, lp]
    yv, lpv = _dovar(1), _dovar(2)
    sref = _thread_ref!(inputs, sarg)
    cell = :(lognormal($lpv, $sref).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
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

function _bernoulli_yplate!(pre::Vector{Expr}, plan::StructuralPlan,
        y::Symbol, label::Symbol, yv::Symbol)
    col = plan.columns[y]
    eltype(col) === Bool && return y, yv
    yb = _ybool_name(label)
    push!(pre, :($yb = Vector{Bool}($y .!= 0)))
    return yb, yv
end

# Student-t evidence arms: the Gaussian clamp law over the `student_t`
# cdf (continuous bounds, non-strict censored arms).
function _student_cell(kind::Symbol, base::Expr, yv::Symbol, lb, ub,
        nuv, lpv::Symbol, sref)
    kind === :none && return base
    stcdf(b) = :(student_t($nuv, $lpv, $sref).cdf($b))
    if kind === :truncated
        corr = if lb === nothing && ub === nothing
            return base
        elseif lb === nothing
            :(log($(stcdf(ub))))
        elseif ub === nothing
            :(log(1.0 - $(stcdf(lb))))
        else
            :(log($(stcdf(ub)) - $(stcdf(lb))))
        end
        return :($base - $corr)
    elseif kind === :censored
        if lb === nothing && ub === nothing
            return base
        elseif lb === nothing
            return :(ifelse($yv >= $ub, log1p(-$(stcdf(ub))), $base))
        elseif ub === nothing
            return :(ifelse($yv <= $lb, log($(stcdf(lb))), $base))
        else
            return :(ifelse($yv <= $lb, log($(stcdf(lb))),
                ifelse($yv >= $ub, log1p(-$(stcdf(ub))), $base)))
        end
    else # :interval_censored
        return :(log($(stcdf(ub)) - $(stcdf(yv))))
    end
end

function _bernoulli_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    yv = _dovar(1)
    pre = Expr[]
    yin, yref = _bernoulli_yplate!(pre, plan, y, r.label, yv)
    inputs = Any[yin]
    if _is_bare_param_location(r, plan)
        pv = _thread_ref!(inputs, r.predictor)
        cell = :(bernoulli($pv).logpdf($yref))
    else
        lp = _lp_name(_predictor(plan, r.predictor))
        push!(inputs, lp)
        etav = _dovar(2)
        cell = :(bernoulli(; logit = $etav).logpdf($yref))
    end
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Fused whole-vector Poisson-log likelihood (base case: no evidence, no
# weights). Value-identical to `Σ poisson(; log_rate=ηᵢ).logpdf(yᵢ)` for the
# contract's nonnegative-integer `y`: `Σ yᵢ·ηᵢ − Σ exp(ηᵢ) − C`, with
# `C = Σ loggamma(yᵢ+1)` baked at generation from the bound response (data-only,
# never on the gradient tape). `_ppl_yf_<label> = Float64.(y)` is a NAMED
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
    lp = _lp_name(_predictor(plan, r.predictor))
    yf = _yfloat_name(r.label)
    return Expr[
        :($yf = Float64.($y)),
        :($node::Float64 = dot($yf, $lp) - sum(log1pexp, $lp)),
    ]
end

function _poisson_wholevec_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    ycol = plan.columns[y]
    cterm = sum(SpecialFunctions.loggamma(Float64(v) + 1.0) for v in ycol)
    yf = _yfloat_name(r.label)
    return Expr[
        :($yf = Float64.($y)),
        :($node::Float64 = dot($yf, $lp) - sum(exp, $lp) - $cterm),
    ]
end

function _poisson_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    inputs = Any[y]
    yv = _dovar(1)
    if _is_bare_param_location(r, plan)
        # Contract rejects evidence for bare locations, so no bounds thread.
        ratev = _thread_ref!(inputs, r.predictor)
        cell = :(poisson($ratev).logpdf($yv))
    else
        lp = _lp_name(_predictor(plan, r.predictor))
        push!(inputs, lp)
        etav = _dovar(2)
        lb, ub = _thread_bounds!(inputs, r.evidence, true)
        base = :(poisson(; log_rate = $etav).logpdf($yv))
        cell = _poisson_cell(r.evidence.kind, base, yv, lb, ub, etav)
    end
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return _plate_sum_stmts(pw, node, inputs, cell)
end

# ZIP plate: the Poisson-plate shape with a zero-inflation argument —
# validation guarantees `zi` (a sampled name, a literal, or a
# predictor-fed zi submodel) and fails evidence closed (the
# Gaussian/Poisson/StudentT-only gate), so the cell is the plain
# `zero_inflated_poisson` endpoint plus optional weights. A zi
# predictor binds its constrained vector once (`_ppl_sc_`, the
# scale-predictor precedent — logit-only at the contract gate) and the
# plate iterates it per cell.
function _zip_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    pre = Expr[]
    ziarg = _scale_use_plate_arg(r, plan, pre, r.zi, r.label)
    inputs = Any[y, lp]
    yv, etav = _dovar(1), _dovar(2)
    ziref = _thread_ref!(inputs, ziarg)
    # All-keyword: the object constructor cannot mix positional and named
    # owner bindings (matches the `:observed/:log_rate/:zi` HAVE ports).
    cell = :(zero_inflated_poisson(; log_rate = $etav, zi = $ziref).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Lower-side cdf argument for the inclusive discrete cdf: the mass below
# lb is F(lb - 1), so TRUNCATED low arms shift their lower argument by one
# (interval cells stay unshifted — the response is the open lower
# endpoint). CENSORED arms instead follow the clamp law (Y = clamp(X)):
# yv ≤ lb takes F(lb) unshifted (the at-bound mass includes F(lb)), while
# yv ≥ ub takes 1 - F(ub-1) (the shift moves to the upper arm — Y = ub
# means X ≥ ub). Int literals fold; do-vars convert via Int (cdf takes
# Int, which also hardens non-Int Integer columns). The kernel's
# `observed >= 0` guard maps -1 to 0.0, so no clamp is needed.
_poisson_below(b::Int) = b - 1
_poisson_below(b::Symbol) = :(Int($b) - 1)

function _poisson_cell(kind::Symbol, base::Expr, yv::Symbol, lb, ub, etav::Symbol)
    kind === :none && return base
    pcdf(b) = :(poisson(; log_rate = $etav).cdf($b))
    if kind === :truncated
        corr = if lb === nothing && ub === nothing
            return base
        elseif lb === nothing
            :(log($(pcdf(ub))))
        elseif ub === nothing
            :(log(1.0 - $(pcdf(_poisson_below(lb)))))
        else
            :(log($(pcdf(ub)) - $(pcdf(_poisson_below(lb)))))
        end
        return :($base - $corr)
    elseif kind === :censored
        if lb === nothing && ub === nothing
            return base
        elseif lb === nothing
            return :(ifelse($yv >= $ub, log1p(-$(pcdf(_poisson_below(ub)))), $base))
        elseif ub === nothing
            return :(ifelse($yv <= $lb, log($(pcdf(lb))), $base))
        else
            return :(ifelse($yv <= $lb, log($(pcdf(lb))),
                ifelse($yv >= $ub, log1p(-$(pcdf(_poisson_below(ub)))), $base)))
        end
    else # :interval_censored
        # Open below per the brm-use contract (`log(CDF(upper) -
        # CDF(response))` for `(response, upper]`): the response is the
        # EXCLUSIVE lower endpoint, so no `_poisson_below` shift (that
        # shift is for inclusive truncated bounds / clamp arms only).
        return :(log($(pcdf(ub)) - $(pcdf(yv))))
    end
end

# Hurdle-Poisson likelihood (SB `hurdle_poisson` mirror): a zero part
# plus a zero-truncated Poisson positive part. Per cell:
# `y == 0 ? log(p0) : log1p(-p0) + poisson_logpdf - log(1 - e^-λ)`.
# The Poisson factor reuses the `poisson(; log_rate)` endpoint HAVE
# (the Poisson-plate precedent). The truncation correction is the
# closed form `log(-expm1(-λ))` (Poisson cdf(0) is exactly e^-λ; SB
# subtracts `poisson_lccdf(0 | λ)` — the same quantity) — NOT the
# `.cdf(0)` endpoint, whose `gamma_inc` has no Reactant tracing rule
# (`MethodError` on compile; the truncated/censored Poisson cells
# carry the same gap). The `y == 0` select is an eager `ifelse` over
# count lanes (both sides already valid values — not the Bernoulli
# lazy-branch shape, whose in-cell comparison `_bernoulli_yplate!`
# now hoists); p_zero threads scalar or via the `_ppl_sc_` node
# (the scale-predictor precedent — a hurdle p_zero predictor is
# logit-only at the contract gate). Evidence fails closed at the
# contract gate (Gaussian/Poisson/StudentT only); weights multiply the cell
# (the NB2 precedent). No whole-vector fusion yet: both parts carry
# per-cell parameter-dependent work (the truncation correction varies
# with λ even for scalar p_zero) — a perf-lane follow-up, not this
# slice.
function _hurdle_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    pre = Expr[]
    sarg = _scale_plate_arg(r, plan, pre)
    inputs = Any[y, lp]
    yv, etav = _dovar(1), _dovar(2)
    p0v = _thread_ref!(inputs, sarg)
    base = :(poisson(; log_rate = $etav).logpdf($yv))
    trunc = :(log(-expm1(-exp($etav))))
    cell = :(ifelse($yv == 0, log($p0v), log1p(-$p0v) + $base - $trunc))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
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
# VonMises-kappa precedent, log-only at the contract gate).
# The `y > 0` guard is lazy `?:` (the DK gamma precedent, not eager
# `ifelse`); `y` is bound data so preparation splits the plate per
# taken arm and no backend receives the branch. μ/λ positivity is
# by-construction (`exp`, positive-constrained layout, contract-validated
# literals/columns), so no live-value guard enters the cell. Evidence
# fails closed at the contract gate (Gaussian/Poisson/StudentT only); weights
# multiply the cell (the NB2 precedent). No whole-vector fusion yet
# (the `3*log(y)` normalizer is data-only and would hoist) — a
# perf-lane follow-up, not this slice.
function _ig_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    mu = _mu_name(r.label)
    pre = Expr[:($mu = exp.($lp))]
    sarg = _scale_plate_arg(r, plan, pre)
    inputs = Any[y, mu]
    yv, muv = _dovar(1), _dovar(2)
    lamv = _thread_ref!(inputs, sarg)
    base = :((log($lamv) - (1.8378770664093456 + 3.0 * log($yv)) -
        $lamv * ($yv - $muv) * ($yv - $muv) / ($muv * $muv * $yv)) / 2.0)
    cell = :($yv > 0 ? $base : -Inf)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
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
# backend receives the branch. μ positivity is by-construction (`exp`),
# so no live-value guard enters the cell. The family takes no scale
# (Poisson-shaped; SB's rate is `1 ./ mu`, carried inside the kernel's
# `-log(μ) - y/μ` spelling). Evidence fails closed at the contract gate
# (Gaussian/Poisson only); weights multiply the cell (the NB2 precedent).
# No whole-vector fusion yet — a perf-lane follow-up, not this slice.
function _exponential_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    mu = _mu_name(r.label)
    pre = Expr[:($mu = exp.($lp))]
    inputs = Any[y, mu]
    yv, muv = _dovar(1), _dovar(2)
    cell = :(exponential($muv).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
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
# the hurdle precedent, log-only at the contract gate). All guards are
# lazy `?:` (never eager `ifelse`); bound-data conditions (circular
# support, literal/column kappa) split the plate per taken arm at
# prepare, while live conditions (sampled/predictor kappa, exact
# moving support) keep their authored branch to the backend. Weights
# multiply the cell (the NB2 precedent). Evidence fails closed at the
# contract gate (Gaussian/Poisson/StudentT only). No whole-vector fusion yet —
# a perf-lane follow-up, not this slice.
function _vonmises_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    pre = Expr[]
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
        lo, hi = r.interval
        w = hi - lo
        wmu = :($lo + rem(rem($muv - $lo, $w) + $w, $w))
        base = :(-1.8378770664093456 - log(besseli(0, $kapv)) +
            $kapv * cos($yv - $wmu))
        inner = :($yv >= $hi ? -Inf : $base)
        sup = :($yv < $lo ? -Inf : $inner)
    end
    cell = :($kapv <= 0 ? -Inf : $sup)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

function _binomial_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    inputs = Any[y]
    yv = _dovar(1)
    if _is_bare_param_location(r, plan)
        nref = _thread_ref!(inputs, r.trials, true)
        pv = _thread_ref!(inputs, r.predictor)
        cell = :(binomial($nref, $pv).logpdf($yv))
    else
        lp = _lp_name(_predictor(plan, r.predictor))
        push!(inputs, lp)
        etav = _dovar(2)
        nref = _thread_ref!(inputs, r.trials, true)
        # All-keyword: the object constructor cannot mix positional and named
        # owner bindings (matches the `:observed/:n/:logit` HAVE ports).
        cell = :(binomial(; n = $nref, logit = $etav).logpdf($yv))
    end
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return _plate_sum_stmts(pw, node, inputs, cell)
end

# Prob-space Binomial (SB `binomial(n, theta)` with a Beta prior — the Rate
# family): the location is the Beta parameter name itself (no LP node —
# the Categorical `r.predictor`-as-symbol precedent), threaded scalar
# through the plate against the positional prob-space kernel.
function _binomial_prob_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    inputs = Any[y]
    yv = _dovar(1)
    nref = _thread_ref!(inputs, r.trials, true)
    pref = _thread_ref!(inputs, r.predictor)
    cell = :(binomial($nref, $pref).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return _plate_sum_stmts(pw, node, inputs, cell)
end

# Zero-inflated-Binomial plate (the ZIB head): prob-space scalar p (the
# BinomialProb precedent) plus the zi slot (the ZIP precedent). The
# kernel endpoint takes all three positionally; weights multiply the
# cell (the NB2 precedent).
function _zib_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    inputs = Any[y]
    yv = _dovar(1)
    nref = _thread_ref!(inputs, r.trials, true)
    pref = _thread_ref!(inputs, r.predictor)
    ziref = _thread_ref!(inputs, r.zi)
    cell = :(zero_inflated_binomial($nref, $pref, $ziref).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return _plate_sum_stmts(pw, node, inputs, cell)
end

# Mean/rate vectors are precomputed statements (like `_ppl_lp_*`): plate
# cells take plain do-vars — a computed `exp` constructor arg miscompiles
# the Enzyme pullback (NB2 eta-gradient, found by test).
_mu_name(label::Symbol) = Symbol(:_ppl_mu_, label)
_rate_name(label::Symbol) = Symbol(:_ppl_rate_, label)

function _nb2_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    mu = _mu_name(r.label)
    pre = Expr[:($mu = exp.($lp))]
    sarg = _scale_plate_arg(r, plan, pre)
    inputs = Any[y, mu]
    yv, muv = _dovar(1), _dovar(2)
    phiref = _thread_ref!(inputs, sarg)
    cell = :(negative_binomial2($muv, $phiref).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# NB1 likelihood (SB `neg_binomial` mirror): the successes-shape r
# precomputes outside the cell (`_ppl_r_`, the NB2 `_ppl_mu_`
# precedent — a computed `exp` constructor arg miscompiles the Enzyme
# pullback); the success probability p threads scalar, per-obs, or
# predictor-fed (logit-only, the hurdle precedent) via
# `_scale_plate_arg`. Weights multiply the cell (the
# NB2 precedent). No whole-vector fusion yet — a perf-lane follow-up,
# not this slice.
_nb1_r_name(label::Symbol) = Symbol(:_ppl_r_, label)

function _nb1_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    rr = _nb1_r_name(r.label)
    pre = Expr[:($rr = exp.($lp))]
    sarg = _scale_plate_arg(r, plan, pre)
    inputs = Any[y, rr]
    yv, rv = _dovar(1), _dovar(2)
    pref = _thread_ref!(inputs, sarg)
    cell = :(negative_binomial($rv, $pref).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Weibull likelihood (SB `weibull` mirror): the scale's `log_theta` HAVE
# route takes the link predictor directly (the Poisson/ZIP precedent —
# no `exp` precompute, no `log(exp())` round trip); the shape k threads
# scalar or per-obs via `_scale_plate_arg` (predictor-fed k is deferred
# at the contract gate). Weights multiply the
# cell (the NB2 precedent). No whole-vector fusion yet — a perf-lane
# follow-up, not this slice.
function _weibull_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    pre = Expr[]
    sarg = _scale_plate_arg(r, plan, pre)
    inputs = Any[y, lp]
    yv, etav = _dovar(1), _dovar(2)
    kref = _thread_ref!(inputs, sarg)
    # All-keyword: the object constructor cannot mix positional and named
    # owner bindings (the ZIP precedent).
    cell = :(weibull(; k = $kref, log_theta = $etav).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

function _gamma_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    pre = Expr[]
    sarg = _scale_plate_arg(r, plan, pre)
    # Surface is Distributions-SCALE `Gamma(alpha, mu/alpha)`; the kernel
    # takes rate, so the boundary inverts (same as the sampled-gamma prior).
    av = sarg isa Symbol ? sarg : Float64(sarg)
    rate = _rate_name(r.label)
    push!(pre, :($rate = $av ./ exp.($lp)))
    if r.mi_jobs !== nothing
        rate = _mi_gather_node!(pre, r.mi_jobs, rate, r.label)
        sarg = _mi_gather_scale!(pre, r.mi_jobs, sarg, r, plan)
    end
    inputs = Any[y, rate]
    yv, ratev = _dovar(1), _dovar(2)
    aref = _thread_ref!(inputs, sarg)
    cell = :(gamma($aref, $ratev).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# The scale argument a likelihood plate threads per cell: scalar scales
# (parameter, assignment, literal, raw data column) pass through untouched;
# a predictor-fed scale binds its constrained vector once
# (`_ppl_sc_<label>`, the `_ppl_mu_`/`_ppl_rate_` precompute precedent —
# the link inverts here, never inside the cell) and the plate iterates
# the node. The logit inversion reuses the Beta plate's inlined
# `1 ./ (1 .+ exp.(-lp))` spelling (no new imports, Enzyme-safe).
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
    rhs = if s.link === IdentityLink
        lp
    elseif s.link === LogLink
        :(exp.($lp))
    elseif s.link === LogitLink
        :(1 ./ (1 .+ exp.(-$lp)))
    else
        throw(ContractValidationError(
            "[generator] scale predictor link $(s.link) has no inversion " *
            "(admitted: identity, log, logit)"))
    end
    # The annotation is load-bearing for AD, not decoration: it proves the
    # plate input `:axis` statically, so `prepare` lowers the straight-line
    # plate form. Unannotated (metadata-`Any`) vector inputs lower with the
    # runtime `_authored_plate_is_axis` / `_plate_dependency_changed` guards,
    # whose form defeats Enzyme's static-activity analysis on some endpoint
    # bodies (NB2, found by test: silently wrong gradients). `AbstractVector`
    # is eltype-free so integer offset-only LPs still match. The LP is
    # always a vector here: codegen entry points validate first (empty
    # predictors rejected) and gate HSGP (the only termless-at-emission
    # shape), so every scale predictor contributes a vector summand.
    push!(pre, :($sc::AbstractVector = $rhs))
    return sc
end

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
    lp = _lp_name(_predictor(plan, r.predictor))
    p = _prob_name(r.label)
    pre = Expr[:($p = 0.5 .* erfc.(-$lp ./ sqrt(2)))]
    yv, pv = _dovar(1), _dovar(2)
    yin, yref = _bernoulli_yplate!(pre, plan, y, r.label, yv)
    inputs = Any[yin, p]
    cell = :(bernoulli($pv).logpdf($yref))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Bernoulli cloglog: pure-arithmetic p precompute, positional-p cell.
function _bernoulli_cloglog_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    p = _prob_name(r.label)
    pre = Expr[:($p = 1 .- exp.(-exp.($lp)))]
    yv, pv = _dovar(1), _dovar(2)
    yin, yref = _bernoulli_yplate!(pre, plan, y, r.label, yv)
    inputs = Any[yin, p]
    cell = :(bernoulli($pv).logpdf($yref))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Binomial probit/cloglog: precompute p, then logit(p), and reuse the
# proven logit route — the binomial kernel's p port is unverified, while
# the (:n, :logit) route is what slice-1 Binomial emits.
function _binomial_probit_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    p = _prob_name(r.label)
    logitp = _logitp_name(r.label)
    inputs = Any[y, logitp]
    yv, lpv = _dovar(1), _dovar(2)
    nref = _thread_ref!(inputs, r.trials, true)
    cell = :(binomial(; n = $nref, logit = $lpv).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[:($p = 0.5 .* erfc.(-$lp ./ sqrt(2))),
        :($logitp = log.($p) .- log1p.(-$p)),
        _plate_sum_stmts(pw, node, inputs, cell)...]
end

function _binomial_cloglog_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    p = _prob_name(r.label)
    logitp = _logitp_name(r.label)
    inputs = Any[y, logitp]
    yv, lpv = _dovar(1), _dovar(2)
    nref = _thread_ref!(inputs, r.trials, true)
    cell = :(binomial(; n = $nref, logit = $lpv).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[:($p = 1 .- exp.(-exp.($lp))),
        :($logitp = log.($p) .- log1p.(-$p)),
        _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Beta mean-concentration: mu/a/b precomputes (Gamma-pre pattern), kappa
# by name (Symbol), inlined (literal), or bound once (`_ppl_sc_<label>`
# via `_scale_plate_arg` for a log-link predictor — the vector folds
# into the a/b precomputes through broadcast); cell needs only (y, a,
# b), so kappa is never a plate input. Positional beta cell (primary
# form).
function _beta_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    mu = _mu_name(r.label)
    a = _shape_a_name(r.label)
    b = _shape_b_name(r.label)
    pre = Expr[:($mu = 1 ./ (1 .+ exp.(-$lp)))]
    sarg = _scale_plate_arg(r, plan, pre)
    k = sarg isa Symbol ? sarg : Float64(sarg)
    push!(pre, :($a = $mu .* $k), :($b = (1 .- $mu) .* $k))
    if r.mi_jobs !== nothing
        # kappa never threads (it folds into the a/b precomputes), so
        # only the shape nodes gather.
        a = _mi_gather_node!(pre, r.mi_jobs, a, r.label)
        b = _mi_gather_node!(pre, r.mi_jobs, b, r.label)
    end
    inputs = Any[y, a, b]
    yv, avv, bvv = _dovar(1), _dovar(2), _dovar(3)
    cell = :(beta($avv, $bvv).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
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
    lp = _lp_name(_predictor(plan, r.predictor))
    pre = Expr[]
    sarg = _scale_plate_arg(r, plan, pre)
    k = sarg isa Symbol ? sarg : Float64(sarg)
    mu = _mu_name(r.label)
    a = _shape_a_name(r.label)
    b = _shape_b_name(r.label)
    inputs = Any[y, a, b]
    yv, avv, bvv = _dovar(1), _dovar(2), _dovar(3)
    nref = _thread_ref!(inputs, r.trials, true)
    cell = :(beta_binomial($nref, $avv, $bvv).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre...,
        :($mu = 1 ./ (1 .+ exp.(-$lp))),
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
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return _plate_sum_stmts(pw, node, inputs, cell)
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
    push!(prests, :($lane = $ref[$rows]))
    return _thread_ref!(inputs, lane)
end

# Ordinal latent scale source: absent (`nothing` — the 3-positional form),
# a literal, a data column, or a log-link predictor's `exp` precompute
# (structural positivity — the Poisson `exp.(lp)` precedent).
function _ordinal_scale_source!(prests::Vector{Expr}, r::LikelihoodSpec,
        plan::StructuralPlan)
    d = r.discrimination
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
    lp = _lp_name(_predictor(plan, sname))
    name = _disc_name(r.label)
    push!(prests, :($name = exp.($lp)))
    return name
end

# Ordered response plate (OrderedLogistic + Ordinal; OrderedLogistic is
# cumulative-logit with d = 1 and no threshold effects). The thresholds
# thread as ONE shared vector (`Ref(t)`) that each cell gathers by its own
# level, so nothing in the emitted program — statements or cell — grows
# with the level count K. K=1 lowers to a zero cell (SB's
# zero-information likelihood).
function _ordinal_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    K = r.n_levels
    K === nothing && throw(ContractValidationError(
        "[generator] ordered response $(r.label) has unresolved n_levels " *
        "(bind_data infers it)"))
    structure = r.family === OrderedLogisticFam ? :cumulative : r.ordinal_structure
    structure === :cumulative ||
        return _ordinal_stopping_stmts(r, plan, node, pw, K)
    lp = _lp_name(_predictor(plan, r.predictor))
    inputs = Any[r.response, lp]
    yv, etav = _dovar(1), _dovar(2)
    prests = Expr[]
    dref = _ordinal_lane_ref!(inputs, prests,
        _ordinal_scale_source!(prests, r, plan), nothing, :_)
    cell = if K == 1
        # SB's zero-information likelihood; the cell stays a real Expr
        # over the (integer) response do-var (`:(0.0)` would quote to a
        # bare Float64, which the plate builder does not take).
        :(0.0 * $yv)
    else
        push!(inputs, :(Ref($(r.thresholds))))
        _ordinal_cumulative_cell(r.link, K, yv, etav, dref, _dovar(length(inputs)))
    end
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
    y = r.response
    if K == 1
        return _plate_sum_stmts(pw, node, Any[y], :(0.0 * $(_dovar(1))))
    end
    lp = _lp_name(_predictor(plan, r.predictor))
    obs, stage = _stage_lane(r.label, :obs), _stage_lane(r.label, :stage)
    prests = Expr[:($obs = _ordinal_stage_obs($y, $K)),
        :($stage = _ordinal_stage_idx($y, $K))]
    level = _stage_lane(r.label, :y)
    eta = _stage_lane(r.label, :eta)
    thr = _stage_lane(r.label, :t)
    push!(prests, :($level = $y[$obs]), :($eta = $lp[$obs]),
        :($thr = $(r.thresholds)[$stage]))
    inputs = Any[stage, level, eta, thr]
    sv, yv, etav, tv = _dovar(1), _dovar(2), _dovar(3), _dovar(4)
    dref = _ordinal_lane_ref!(inputs, prests,
        _ordinal_scale_source!(prests, r, plan), obs, _stage_lane(r.label, :d))
    z = if isempty(r.threshold_columns)
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
    if r.weights !== nothing
        cell = _weighted_cell(_ordinal_lane_ref!(inputs, prests, r.weights,
            obs, _stage_lane(r.label, :w)), cell)
    end
    return Expr[prests..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Shared-simplex multinomial (SB `brm_multinomial` vector[K] method): the
# count matrix crosses as K raw columns — program structure (the count
# columns the model names), not a data-inferred size — and the level
# log-probabilities `log.(p)` thread as one shared vector the cell reads
# per column. The cell is Stan's `multinomial_lpmf` in scalar form —
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
    logp = _logp_name(r.label)
    push!(inputs, :(Ref($logp)))
    lv = _dovar(length(inputs))
    lfact = r.trials isa Int ? loggamma(r.trials + 1.0) : :(loggamma($nref + 1.0))
    cell = :($lfact)
    for (i, cv) in enumerate(cvs)
        cell = :($cell - loggamma($cv + 1.0) +
            ifelse($cv == 0, 0.0, $cv * $lv[$i]))
    end
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[:($logp::AbstractVector{Float64} = log.($(r.predictor))),
        _plate_sum_stmts(pw, node, inputs, cell)...]
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
    logp = _logp_name(r.label)
    inputs = Any[r.response, :(Ref($logp))]
    yv, lv = _dovar(1), _dovar(2)
    cell = :($lv[$yv])
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[:($logp::AbstractVector{Float64} = log.($(r.predictor))),
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
    si = findfirst(p -> p.name === r.factor_scales, plan.vector_parameters)
    ci = findfirst(p -> p.name === r.factor_corr, plan.vector_parameters)
    (si === nothing || ci === nothing) && throw(ContractValidationError(
        "[generator] joint response $(r.label) factor pieces unresolved " *
        "(validate_plan links them)"))
    sc, cr = plan.vector_parameters[si], plan.vector_parameters[ci]
    (sc.size == K && cr.size == K) || throw(ContractValidationError(
        "[generator] joint response $(r.label) factor sizes disagree " *
        "with the $K outcomes (validate_plan checks this)"))
    stmts = Expr[]
    # L[i,j] = scales[i] * L_corr[i,j] (j ≤ i), one scalar per lower entry.
    for i in 1:K, j in 1:i
        push!(stmts, :($(_mvn_L_entry(r.label, i, j))::Float64 =
            $(_vector_elt_name(sc.name, i)) * $(_rl_name(cr.name, i, j))))
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

# R2D2 prior bindings: the location vector stays a literal (SB
# `beta_loc`); the scale vector is ONE broadcast over the share simplex,
# `sqrt.(phi .* R2 .* tau^2 ./ varx)` (SB `brm_r2d2_scale`), whatever the
# number of design columns (factor levels included). `r2d2_column_scales`
# numbers the shares 1..S in design-column order, so the shared columns
# read `phi` in order; share-0 columns (intercept, explicit-Normal
# overrides) take literal fallbacks, placed by one constant-index gather
# over `[shared; fallbacks]` (the share map is static data — no
# data-dependent branching enters the graph). All radicands are positive
# by construction (simplex/logistic/exp transforms + validated varx). The
# consuming plate-sum shape is unchanged.
function _r2d2_prior_stmts(rp::R2D2Prior, shape::DesignShape,
        columns::AbstractDict{Symbol}, mut::Symbol, sdt::Symbol)
    share, fallback, loc, varx =
        r2d2_column_scales(shape, columns, rp.overrides)
    shared = findall(>(0), share)
    share[shared] == 1:length(shared) || throw(ContractValidationError(
        "[generator] R2D2 shares of $(rp.predictor) are not numbered in " *
        "design-column order (r2d2_column_scales assigns them so)"))
    t2 = rp.tau isa Symbol ? :($(rp.tau) * $(rp.tau)) : Float64(rp.tau)^2
    scales = :(sqrt.($(rp.phi) .* $(rp.r2) .* $t2 ./
        Float64[$(varx[shared]...)]))
    rhs = if length(shared) == length(share)
        scales
    else
        # Shared scales first, fallbacks after (`vcat(traced, host)` —
        # the order Reactant concatenates), gathered into column order.
        fb = findall(==(0), share)
        perm = zeros(Int, length(share))
        perm[shared] .= 1:length(shared)
        perm[fb] .= length(shared) .+ (1:length(fb))
        :(vcat($scales, Float64[$(fallback[fb]...)])[$(Expr(:vect, perm...))])
    end
    return Any[:($mut = Float64[$(loc...)]), :($sdt = $rhs)]
end

_r2d2_for(plan::StructuralPlan, pred::Symbol) = begin
    for rp in plan.r2d2_priors
        rp.predictor === pred && return rp
    end
    return nothing
end

# Horseshoe derived coefficient blocks: a predictor with any HorseshoePrior
# lays out no `:coefficient` block (layout skips it), so the block name
# binds here as a design-ordered vector — triple products on horseshoe
# addressees, Normal scalars elsewhere. Bound before the linear predictors,
# which read the name unchanged. Scalar-only by validation (width-1
# intercept/continuous blocks), so the literal has one entry per term —
# no data-derived unrolling.
function _horseshoe_coef_statements(plan::StructuralPlan)
    stmts = Expr[]
    for pred in plan.predictors
        hs = _horseshoe_for(plan, pred.name)
        isempty(hs) && continue
        shape = design_shape(pred, plan.columns; levelmaps = plan.levelmaps,
            matrices = plan.matrices)
        by_addr = Dict{Symbol,HorseshoePrior}(h.addressee => h for h in hs)
        coords = Any[]
        for b in shape.blocks
            b.width == 0 && continue
            (b.kind === InterceptTerm || b.kind === ContinuousTerm) ||
                throw(ContractValidationError(
                    "[generator] horseshoe over $(pred.name) meets a " *
                    "$(b.kind) block (validate_horseshoe restricts terms)"))
            b.width == 1 || throw(ContractValidationError(
                "[generator] horseshoe over $(pred.name) meets width " *
                "$(b.width) (scalar blocks only)"))
            addr = only(b.labels)
            h = get(by_addr, addr, nothing)
            if h === nothing
                push!(coords, horseshoe_normal_name(pred.name, addr))
            else
                raw = horseshoe_raw_name(pred.name, addr)
                lam = horseshoe_lambda_name(pred.name, addr)
                tau = horseshoe_tau_name(pred.name, addr)
                prod = :($raw * $lam * $tau)
                push!(coords, h.sign == 1 ? prod : :(-$prod))
            end
        end
        coef = block_name(pred.name)
        push!(stmts, :($coef = [$(coords...)]))
    end
    return stmts
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

function _parameter_prior_statements!(stmts, terms, plan, layout)
    grouped = Set{Symbol}()
    for ps in _parameter_prior_groups(plan)
        family = first(ps).family
        _, li, si, ni = _COEF_SHAPES[family]
        args = [collect(values(p.args)) for p in ps]
        loc = Float64[a[li] for a in args]
        sca = Float64[a[si] for a in args]
        nus = ni == 0 ? fill(NaN, length(ps)) : Float64[a[ni] for a in args]
        node = Symbol(:_ppl_prior_parameters_, family)
        _append_coef_plate!(stmts, _parameter_prior_input(ps, plan, layout),
            node, Symbol(node, :_pw), Symbol(node, :_mu), Symbol(node, :_sd),
            Symbol(node, :_nu), loc, sca, nus, family)
        push!(terms, node)
        union!(grouped, (p.name for p in ps))
    end
    for p in plan.parameters
        p.name in grouped && continue
        node = Symbol(:_ppl_prior_, p.name)
        push!(stmts, :($node::Float64 = $(_sampled_prior_expr(p))))
        push!(terms, node)
    end
    return nothing
end

function _prior_statements(plan::StructuralPlan, layout::LayoutTable;
        gathers::Set{Tuple{Symbol,Symbol}} = Set{Tuple{Symbol,Symbol}}())
    stmts = Expr[]
    terms = Any[]
    for pred in plan.predictors
        pred = _legacy_predictor(pred)
        shape = design_shape(pred, plan.columns; levelmaps = plan.levelmaps,
            matrices = plan.matrices)
        shape.width == 0 && continue
        # A horseshoe predictor carries no Normal plate prior: its
        # coordinates derive from triples/Normal scalars whose priors ride
        # the sampled-parameter loop below.
        isempty(_horseshoe_for(plan, pred.name)) || continue
        node = Symbol(:_ppl_prior_, pred.name)
        pw = Symbol(:_ppl_pw_prior_, pred.name)
        mut = Symbol(:_ppl_prmu_, pred.name)
        sdt = Symbol(:_ppl_prsd_, pred.name)
        rp = _r2d2_for(plan, pred.name)
        if rp === nothing
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
            continue
        end
        push!(stmts, _r2d2_prior_stmts(rp, shape, plan.columns, mut, sdt)...)
        coef = block_name(pred.name)
        cv, mv, sv = _dovar(1), _dovar(2), _dovar(3)
        cell = :(normal($mv, $sv).logpdf($cv))
        append!(stmts, _plate_sum_stmts(pw, node, Any[coef, mut, sdt], cell))
        push!(terms, node)
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
        _vector_prior_stmts!(stmts, terms, p.name, p.family, p.args,
            p.support_override)
    end
    # Spline coefficient vectors: the same plate-prior shape (broadcast the
    # shared prior over cells). `b_fixed` is flat — a 0.0 node, mirroring a
    # scalar flat parameter (never a plate: a vacuous cell would leave the
    # do-var unread).
    for v in plan.spline_vectors
        if v.family === :flat
            node = Symbol(:_ppl_prior_, v.name)
            push!(stmts, :($node::Float64 = 0.0))
            push!(terms, node)
            continue
        end
        _vector_prior_stmts!(stmts, terms, v.name, v.family, v.args,
            v.support_override)
    end
    # Varying draws: the LKJ node plus the `tau` prior plus the
    # `z_flat` plate (shared vector-prior helper), at every K (the 1x1
    # LKJ node is the `0.0` literal). `tau` emits WITHOUT the thin
    # layer's `+log(2)` half renormalizer: SB's `std_normal(; lower=0)`
    # is Stan lower-bound kernel semantics (exp Jacobian only, no
    # truncation normalizer). User-facing `HalfNormal` priors keep the
    # proper-half convention; draws-internal `tau` follows SB — under
    # every configured sd prior too (SB's generic path keeps the
    # positive bound with no truncation normalizer).
    for d in plan.varying_draws
        if d.strata !== nothing
            _stratified_prior_stmts!(stmts, terms, d, layout)
            continue
        end
        L, tau, z = _varying_corr_names(d)
        lnode = Symbol(:_ppl_prior_, L)
        push!(stmts, :($lnode::Float64 = $(_lkj_prior_expr(d))))
        push!(terms, lnode)
        _sd_prior_tau_stmts!(stmts, terms, d, tau)
        if d.kind === :centered_correlated
            K = length(d.margins)
            lower = Symbol(:_ppl_centered_L_,d.suffix)
            vals = Any[_rl_name(L,i,j) for i in 1:K for j in 1:i]
            push!(stmts,:($lower = Float64[$(vals...)]))
            node = Symbol(:_ppl_prior_,z)
            push!(stmts,:($node::Float64 = _centered_correlated_logpdf($z,$tau,$lower)))
            push!(terms,node)
        else
            _vector_prior_stmts!(stmts, terms, z, :normal,
                (arg1 = 0, arg2 = 1), nothing)
        end
    end
    # HSGP bases (SB `_sb_hsgp`/`_sb_hsgp_aniso`): the length scales and
    # marginal scale as scalar `lognormal(0, 1)` nodes plus the
    # standardized `beta_raw` plate (shared vector-prior helper). The
    # floored rhos emit WITHOUT a truncation normalizer: SB's
    # `lognormal(0,1; lower=rho_lower)` is Stan lower-bound kernel
    # semantics (offset-exp Jacobian only — the varying-`tau` precedent).
    # Stated hyper priors (`HyperPrior`) replace the defaults with the
    # same Stan-kernel semantics (plain `_lpdf`, no normalizer).
    for hb in plan.hsgp_bases
        names = _hsgp_names(hb)
        rfam, rargs = hb.rho_prior isa HyperPrior ?
            (hb.rho_prior.family, collect(Any, values(hb.rho_prior.args))) :
            (:lognormal, Any[0, 1])
        sfam, sargs = hb.sigma_prior isa HyperPrior ?
            (hb.sigma_prior.family,
                collect(Any, values(hb.sigma_prior.args))) :
            (:lognormal, Any[0, 1])
        # Per-group hyper-predictors (BRM defaults): intercept
        # `Normal(0, 1)`, sd `Normal(0, 1)` on the positive support
        # (Stan kernel — plain `_lpdf`), non-centered `z` standard normal.
        function hyper_priors!(h)
            if h.intercept
                bnode = Symbol(:_ppl_prior_, h.beta0)
                push!(stmts, :($bnode::Float64 =
                    $(_family_logpdf_expr(:normal, Any[0, 1], h.beta0))))
                push!(terms, bnode)
            end
            dnode = Symbol(:_ppl_prior_, h.sd)
            push!(stmts, :($dnode::Float64 =
                $(_family_logpdf_expr(:normal, Any[0, 1], h.sd))))
            push!(terms, dnode)
            _vector_prior_stmts!(stmts, terms, h.z, :normal,
                (arg1 = 0, arg2 = 1), nothing)
        end
        if names.rho_hyper !== nothing
            hyper_priors!(names.rho_hyper)
        else
            for rho in names.rhos
                node = Symbol(:_ppl_prior_, rho)
                cell = _family_logpdf_expr(rfam, rargs, rho)
                push!(stmts, :($node::Float64 = $cell))
                push!(terms, node)
            end
        end
        if names.sigma_hyper !== nothing
            hyper_priors!(names.sigma_hyper)
        else
            snode = Symbol(:_ppl_prior_, names.sigma)
            scell = _family_logpdf_expr(sfam, sargs, names.sigma)
            push!(stmts, :($snode::Float64 = $scell))
            push!(terms, snode)
        end
        _vector_prior_stmts!(stmts, terms, names.beta, :normal,
            (arg1 = 0, arg2 = 1), nothing)
    end
    # Event-LP providers (SB V2 term priors verbatim): the dose slope
    # `Normal(0, 0.6676)` (the specific `effect(log_F, op_log_dose)`
    # override — BRM specificity beats the wildcard), the length scale
    # `Uniform(floor, 2.0)` over the `:interval` support (constant
    # `-log(hi-lo)` — the support keeps it inside), the marginal scale
    # `Normal(0, 1)` WITHOUT a half normalizer (Stan lower-bound
    # kernel semantics — the varying-`tau` precedent), and the
    # standardized `beta_raw` plate (shared vector-prior helper).
    for el in plan.event_lps
        el.fit === nothing && throw(ContractValidationError(
            "[generator] event-LP `$(el.name)`: fit not filled at bind " *
            "(bind_data fits one (mu, L) over the event axis)"))
        names = _event_lp_names(el)
        floor = only(_hsgp_floors([el.k], [el.fit], true))
        slopesym = names.slope
        slnode = Symbol(:_ppl_prior_, slopesym)
        slcell = _family_logpdf_expr(:normal,
            Any[0.0, _EVENT_LP_SLOPE_PRIOR_SD], slopesym)
        push!(stmts, :($slnode::Float64 = $slcell))
        push!(terms, slnode)
        rhosym = names.rho
        rnode = Symbol(:_ppl_prior_, rhosym)
        rcell = :(uniform($floor, $(_EVENT_LP_RHO_PRIOR_HI)).logpdf($rhosym))
        push!(stmts, :($rnode::Float64 = $rcell))
        push!(terms, rnode)
        sigsym = names.sigma
        snode = Symbol(:_ppl_prior_, sigsym)
        scell = _family_logpdf_expr(:normal, Any[0, 1], sigsym)
        push!(stmts, :($snode::Float64 = $scell))
        push!(terms, snode)
        _vector_prior_stmts!(stmts, terms, names.beta, :normal,
            (arg1 = 0, arg2 = 1), nothing)
    end
    # Sequential-recurrence (scan) latents: setup + recurrence density.
    scanstmts, scannodes = _scan_prior_statements(plan, layout)
    append!(stmts, scanstmts)
    append!(terms, scannodes)
    # Differenced-AR(1) trajectories: the iid-innovation prior.
    darstmts, darnodes = _dar_prior_statements(plan, layout)
    append!(stmts, darstmts)
    append!(terms, darnodes)
    # Declared array parameters (`arrays.jl`).
    _array_prior_stmts!(stmts, terms, plan, gathers)
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
        nstack::Union{Nothing,Int} = nothing)
    K == 1 && return :(0.0)
    c = lkj_logconst(K, eta)
    terms = Any[nstack === nothing ? c : nstack * c]
    lg(i) = nstack === nothing ? :(log($(_rl_name(L, i, i)))) :
        :(sum(log.($(_rl_name(L, i, i)))))
    if eta == 1.0
        for i in 2:K
            push!(terms, :($(K - i) * $(lg(i))))
        end
    else
        bcoef = 2 * eta - 2
        for i in 2:K
            k = i - 2
            push!(terms, :($(K - 1 - k - 1) * $(lg(i)) + $bcoef * $(lg(i))))
        end
    end
    return foldl((a, c) -> :($a + $c), terms)
end

# One correlated draws block's LKJ prior node (names/sizes from the draws).
function _lkj_prior_expr(d::VaryingDraws)
    return _lkj_prior_terms(_varying_corr_names(d)[1], length(d.margins),
        d.lkj_eta)
end

# One stratified draws block's prior nodes, stacked across its S strata
# (S comes from data, so per-stratum nodes would replicate statements —
# core constraint 1): ONE LKJ node over the stacked diagonal S-vectors
# (S copies of the constant, then each diagonal term summed over the
# strata), ONE half-normal plate over the stacked K×S `tau` (validation
# proves no sd priors), then the shared `z_flat` plate. The strata are
# independent, so this is the SB `ranef_correlated_by` sum regrouped.
function _stratified_prior_stmts!(stmts::Vector{Expr}, terms::Vector{Any},
        d::VaryingDraws, layout::LayoutTable)
    g = _strata_geometry(d, layout)
    sL = _strata_L_name(d)
    lnode = Symbol(:_ppl_prior_, sL)
    push!(stmts, :($lnode::Float64 =
        $(_lkj_prior_terms(sL, g.K, d.lkj_eta; nstack = g.S))))
    push!(terms, lnode)
    _vector_prior_stmts!(stmts, terms, _strata_tau_name(d), :normal,
        (arg1 = 0, arg2 = 1), nothing)
    z = _varying_corr_names(d)[3]
    _vector_prior_stmts!(stmts, terms, z, :normal,
        (arg1 = 0, arg2 = 1), nothing)
    return nothing
end

# One margin's sd prior in the shared (family, args) prior shape (SB's
# generic-path mirror: `:std_normal` is Normal(0, 1), `:exponential`
# carries the contract's SCALE, `:normal` is Normal(0, σ)).
function _sd_prior_shape(p::VaryingSdPrior)
    p.family === :std_normal && return (:normal, (arg1 = 0.0, arg2 = 1.0))
    p.family === :exponential && return (:exponential, (arg1 = p.param,))
    p.family === :normal && return (:normal, (arg1 = 0.0, arg2 = p.param))
    p.family === :cauchy && return (:cauchy, (arg1 = 0.0, arg2 = p.param))
    throw(ContractValidationError("[generator] sd prior family " *
        "$(repr(p.family)) is not one of $(_SD_PRIOR_FAMILIES) " *
        "(validate_plan proves this)"))
end

# A draws block's per-margin `tau` prior shapes in margin order (empty
# `sd_priors` is all-`:std_normal`).
function _sd_prior_shapes(d::VaryingDraws)
    K = length(d.margins)
    isempty(d.sd_priors) &&
        return fill((:normal, (arg1 = 0.0, arg2 = 1.0)), K)
    return [_sd_prior_shape(p) for p in d.sd_priors]
end

# One correlated draws block's `tau` prior (SB's homogeneous /
# heterogeneous split): all-Normal(0, 1) keeps the historical plate
# emission bit-identical; a uniform configured prior stays one plate
# with the mapped family; mixed margins unroll to one scalar density
# per margin over `tau[k]` refs (the LKJ-sandwich precedent). Every
# path keeps `support = nothing` (Stan lower-bound kernel semantics —
# the layout's `:exp` Jacobian, no truncation renormalizer).
function _sd_prior_tau_stmts!(stmts::Vector{Expr}, terms::Vector{Any},
        d::VaryingDraws, tau::Symbol)
    shapes = _sd_prior_shapes(d)
    if all(s -> s == (:normal, (arg1 = 0.0, arg2 = 1.0)), shapes)
        _vector_prior_stmts!(stmts, terms, tau, :normal,
            (arg1 = 0, arg2 = 1), nothing)
        return nothing
    end
    if all(s -> s == shapes[1], shapes)
        fam, args = shapes[1]
        _vector_prior_stmts!(stmts, terms, tau, fam, args, nothing)
        return nothing
    end
    node = Symbol(:_ppl_prior_, tau)
    cells = Any[]
    for k in eachindex(shapes)
        fam, args = shapes[k]
        push!(cells, _family_logpdf_expr(fam, Any[values(args)...],
            Expr(:ref, tau, k)))
    end
    push!(stmts, :($node::Float64 = $(foldl((a, c) -> :($a + $c), cells))))
    push!(terms, node)
    return nothing
end

# One plate over a latent VECTOR (a plate parameter or a spline vector),
# summing the shared-prior log-density across cells. Every value the cell
# reads is threaded as a plate PORT (the vector plus each scalar prior
# arg) — captured free names are rejected by the `@kernel` plate
# expander, exactly as the Gaussian-likelihood scale is threaded.
function _vector_prior_stmts!(stmts::Vector{Expr}, terms::Vector{Any},
        name::Symbol, family::Symbol, args::NamedTuple,
        support::SupportOverride)
    node = Symbol(:_ppl_prior_, name)
    pw = Symbol(:_ppl_pw_prior_, name)
    inputs = Any[name]
    tv = _dovar(1)
    argvals = Any[_thread_ref!(inputs, v) for v in values(args)]
    cell = _family_logpdf_expr(family, argvals, tv)
    corr = _support_correction(family, support, argvals)
    corr === nothing || (cell = :($cell + $corr))
    append!(stmts, _plate_sum_stmts(pw, node, inputs, cell))
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
)

# Shared `<endpoint>(remapped args…).logpdf(x)` splice for a variate
# expression `x` (a scalar parameter name, a plate do-var, a scan
# setup/recurrence read, or a coefficient element read). Args are literals
# (inlined) or parameter/assignment/threaded refs (Distributions.jl
# semantics). Shared by scalar priors, per-cell plate priors, population
# priors, and the scan density. `gamma` takes rate, so the contract's
# scale inverts; `:flat` is the vacuous 0.0. Symmetric `:positive`
# halves (`+log(2)`) and the `:interval` correction live in
# `_support_correction` (the Stan-kernel `:positive_stan` /
# `:interval_stan` / `:upper` overrides add nothing there).
function _family_logpdf_expr(family::Symbol, a, x)
    family === :flat && return :(0.0)
    ep = get(_PRIOR_ENDPOINTS, family, nothing)
    ep === nothing && throw(ContractValidationError(
        "[generator] prior family $family has no endpoint object"))
    args = family === :gamma ? (a[1], :(1 / $(a[2]))) : Tuple(a)
    return :($ep($(args...)).logpdf($x))
end

# The additive support-override correction for a prior log-density (or `nothing`
# for no override): `:positive` (half-Normal/half-Cauchy) renormalizes by exactly
# +log(2) (symmetry at literal 0); `:positive_stan` (the Stan-kernel half)
# adds NOTHING — plain `_lpdf` plus the bare-`u` Jacobian;
# `(:interval, lo, hi)` (a truncated Normal) renormalizes by
# -log(cdf(hi) - cdf(lo)) at any location, where `argvals` are the family's
# (mu, s) argument expressions (literals/refs for a scalar prior, or per-cell
# do-vars for a plate prior — the CDF endpoints thread identically);
# `(:interval_stan, lo, hi)` adds NOTHING — Stan's two-sided-bound kernel
# is the plain normal_lpdf plus the bare-`u` Jacobian (the dar-beta
# precedent: SB truncation never renormalizes) — and neither does
# `:flat`, which is improper (Jacobian only, no renormalization,
# matching Stan lower/interval-bound kernel semantics);
# `(:upper, hi)` adds NOTHING — Stan's upper-bound kernel is the plain
# normal_lpdf plus the bare-`u` Jacobian (the varying-`tau`/`:floored`
# precedent: SB truncation never renormalizes).
function _support_correction(family::Symbol, ov::SupportOverride, argvals)
    ov === nothing && return nothing
    ov === :positive_stan && return nothing  # Stan kernel semantics
    if ov isa Tuple
        (ov[1] === :upper || ov[1] === :interval_stan) && return nothing  # Stan kernel semantics
        ov[1] === :interval || throw(ContractValidationError(
            "[generator] tuple support override must be (:interval, lo, hi), " *
            "(:interval_stan, lo, hi), or (:upper, hi), got $ov"))
        family === :flat && return nothing  # improper: Jacobian only
        lo, hi = ov[2], ov[3]
        mu, s = argvals[1], argvals[2]
        return :(-log(normal($mu, $s).cdf($hi) - normal($mu, $s).cdf($lo)))
    end
    ov === :positive || throw(ContractValidationError(
        "[generator] support override must be :positive or " *
        ":positive_stan, got $ov"))
    return :(log(2))  # :positive half
end

# Scalar prior log-density per family via distribution-kernel endpoints
# (Distributions.jl semantics). The support override adds the +log(2) half or
# the -log(cdf(hi)-cdf(lo)) truncated-interval renormalization (`_support_correction`;
# `:positive_stan`/`(:interval_stan, lo, hi)`/`(:upper, hi)` overrides add
# nothing — Stan kernel semantics).
function _sampled_prior_expr(p::SampledParameter)
    argvals = [v for v in values(p.args)]
    base = _family_logpdf_expr(p.family, argvals, p.name)
    corr = _support_correction(p.family, p.support_override, argvals)
    corr === nothing && return base
    return :($base + $corr)
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
    if p.family === :ordered_normal || p.family === :vector_normal
        if m == 0
            push!(stmts, :($node::Float64 = 0.0))
            push!(terms, node)
            return nothing
        end
        _vector_prior_stmts!(stmts, terms, p.name, :normal,
            (arg1 = Float64(p.args.arg1), arg2 = Float64(p.args.arg2)), nothing)
        return nothing
    end
    rhs = if p.family === :simplex_dirichlet
        alpha = Vector{Float64}(p.args.arg1)
        normalizer = loggamma(sum(alpha)) - sum(loggamma, alpha)
        am1 = alpha .- 1.0
        weights = all(==(am1[1]), am1) ? am1[1] : :(Float64[$(am1...)])
        :($normalizer + sum($weights .* log.($(p.name))))
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
    return ScanStep[st for st in s.step if st.kind === :sample && !st.indexed]
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
    ex.head === :ref && throw(ContractValidationError(
        "$(_scan_where(s)): indexed read `$(ex)` in a seed is not supported"))
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
    return [(a, isempty(lags[a]) ? 0 : maximum(lags[a])) for a in s.states]
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
        ex === s.loopvar && throw(ContractValidationError(
            "$where: a step uses the loop index `$(s.loopvar)` directly — not " *
            "supported yet"))
        ex in s.states && throw(ContractValidationError(
            "$where: bare read of the carried array `$(ex)` — read the " *
            "backward lag `$(ex)[$(s.loopvar) - 1]`"))
        ex in scalars || throw(ContractValidationError(
            "$where: step leaf `$(ex)` is not a scalar parameter or " *
            "definition (data-varying steps are planned)"))
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
    ex.head === :ref && throw(ContractValidationError(
        "$where: indexed read `$(ex.args[1])[...]` in a step — a step reads " *
        "carried arrays, locals and scalars"))
    args = ex.head === :call ? ex.args[2:end] : ex.args
    newargs = Any[_scan_translate_step(a, s, env, refs, scalars) for a in args]
    return ex.head === :call ?
        Expr(:call, ex.args[1], newargs...) : Expr(ex.head, newargs...)
end

# The do-block body of one reconstruction: the steps in order, the next carry
# window, and `(next, <output>)`. Returns (body statements, sorted refs).
function _scan_step_block(s::ScanSpec, innov, output::Symbol, scalars)
    refs = Set{Symbol}()
    env = (; locals = Dict{Symbol,Any}(), current = Dict{Symbol,Symbol}())
    for (j, st) in enumerate(innov)
        env.locals[st.target] = Symbol(:_ppl_e, j)
    end
    body = Expr[]
    for st in s.step
        st.kind === :sample && continue          # an innovation: bound above
        rhs = _scan_translate_step(st.expr, s, env, refs, scalars)
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
    for (a, L) in _scan_windows(s)
        for k in 1:L
            val = k == 1 ? env.current[a] :
                Expr(:., :_ppl_carry, QuoteNode(_scan_lag_field(a, k - 1)))
            push!(fields, Expr(:(=), _scan_lag_field(a, k), val))
        end
    end
    push!(body, :(_ppl_next = $(Expr(:tuple, fields...))))
    push!(body, :((_ppl_next, $(env.current[output]))))
    return body, sort!(collect(refs))
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
    scalars = _union_names(plan)
    for s in plan.scans
        _is_noncentered_scan(s) || continue
        innov = _scan_innovations(s)
        T = _scan_length(plan, s)
        zname = _scan_innovation_name(s)
        any(e -> e.kind === :scan && e.name === zname, layout.entries) ||
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
        init = Expr[]
        for (a, L) in _scan_windows(s), k in 1:L
            push!(init, Expr(:(=), _scan_lag_field(a, k),
                _scan_init_name(a, m - k + 1)))
        end
        seqs = Any[:(view($zname, $(first(r)):$(last(r)))) for r in ranges]
        elems = Symbol[Symbol(:_ppl_e, j) for j in eachindex(innov)]
        for a in s.states
            body, refs = _scan_step_block(s, innov, a, scalars)
            lambda = Expr(:->, Expr(:tuple, :_ppl_carry, elems..., refs...),
                Expr(:block, body...))
            kw = Expr(:parameters, Expr(:kw, :init, Expr(:tuple, init...)))
            call = Expr(:call, :scan, kw, seqs..., (:(Ref($r)) for r in refs)...)
            rest = Symbol(:_ppl_scan_rest_, a)
            push!(stmts, :($rest = $(Expr(:do, call, lambda))))
            seeds = [_scan_init_name(a, k) for k in 1:m]
            push!(stmts, :($a = vcat($(seeds...), $rest)))
        end
    end
    return stmts
end

# The non-centered latent prior: each sampled seed's density at its slice
# coordinate, and each innovation's density as a plate over its per-step
# block (the `_plate_sum_stmts` reduction the `PlateParameter` path uses;
# scalar distribution arguments thread as plate inputs), totalled under the
# scan-flavored `_ppl_scan_<first state>` node. Seed and innovation arguments
# read scalars (seed arguments may read earlier seeds); a per-step argument
# that reads a carried array, a local or the loop index is not supported yet.
function _scan_noncentered_prior!(stmts::Vector{Expr}, nodes::Vector{Symbol},
        s::ScanSpec, plan::StructuralPlan)
    innov = _scan_innovations(s)
    T = _scan_length(plan, s)
    scalars = _union_names(plan)
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
    for (j, st) in enumerate(innov)
        caps = Symbol[]
        for a in st.args
            _scan_cell_caps!(caps, a)
        end
        for c in caps
            c in scalars || throw(ContractValidationError(
                "$(_scan_where(s)): the innovation `$(st.target)` reads " *
                "`$(c)`, which is not a scalar parameter or definition " *
                "(per-step innovation scales are not supported yet)"))
        end
        for a in st.args
            _scan_reads_carried(a, s) && throw(ContractValidationError(
                "$(_scan_where(s)): the innovation `$(st.target)` reads a " *
                "carried array (per-step innovation scales are not supported " *
                "yet)"))
        end
        inputs = Any[:(view($zname, $(first(ranges[j])):$(last(ranges[j]))))]
        capmap = Dict{Symbol,Symbol}()
        for c in caps
            push!(inputs, c)
            capmap[c] = _dovar(length(inputs))
        end
        cell = _family_logpdf_expr(st.family,
            Any[_subst_syms(a, capmap) for a in st.args], _dovar(1))
        node = Symbol(:_ppl_scan_innov_, head, :_, j)
        pw = Symbol(:_ppl_scan_pw_, head, :_, j)
        append!(stmts, _plate_sum_stmts(pw, node, inputs, cell))
        push!(terms, node)
    end
    total = Symbol(:_ppl_scan_, head)
    push!(stmts, :($total::Float64 = $(foldl((a, b) -> :($a + $b), terms))))
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
    for sc in plan.scans
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
        m = length(sc.setup)
        terms = Any[]
        for (k, f) in enumerate(sc.setup)
            seed = Symbol(:_ppl_scan_seed_, state, :_, k)
            push!(stmts, :($seed::Float64 =
                $(_family_logpdf_expr(f.family, f.args, :($(state)[$k])))))
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
        caps = Symbol[]
        for a in lagargs
            _scan_cell_caps!(caps, a)
        end
        sc.loopvar in caps && throw(ContractValidationError(
            "[generator] scan $(state): the recurrence uses the loop index " *
            "`$(sc.loopvar)` directly — not supported in slice 1"))
        capmap = Dict{Symbol,Symbol}()
        for c in caps
            push!(inputs, c)
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

# --- Differenced-AR(1) trajectory reconstruction (dar slice) ---

# Reconstruction statements for every dar trajectory, in plan order — the
# shared RK-core `scan(...)` carry-fold with a `(x, d)` NamedTuple carry
# (level + AR(1) increment), zero-started exactly like SB's
# `differenced_ar1_path` (`x[1] = 0`, `d[0] = 0`):
#   `rest = scan(z, Ref(beta), Ref(sigma); init = (x = 0.0, d = 0.0)) do ... end`
#   `state = vcat(0.0, rest)`
# The per-step outputs are `x[2..T]`; the `vcat` heads the zero start.
# Runs before predictors/likelihood (the LP splices the state); the
# innovation prior stays in `_dar_prior_statements` (order-free).
function _dar_reconstruction_statements(plan::StructuralPlan,
        layout::LayoutTable)
    stmts = Expr[]
    for s in plan.dar_paths
        any(p -> p.name === s.beta, plan.parameters) || throw(
            ContractValidationError(
                "[generator] dar $(s.state): persistence :$(s.beta) is " *
                "not a sampled parameter"))
        any(p -> p.name === s.sigma, plan.parameters) || throw(
            ContractValidationError(
                "[generator] dar $(s.state): scale :$(s.sigma) is not a " *
                "sampled parameter"))
        zname = _dar_innovation_name(s)
        any(e -> e.kind === :scan && e.name === zname,
            layout.entries) || throw(ContractValidationError(
            "[generator] dar $(s.state): layout has no innovation slice " *
            ":$zname"))
        beta, sigma = s.beta, s.sigma
        lambda = Expr(:->,
            Expr(:tuple, :_ppl_carry, :_ppl_elem, beta, sigma),
            Expr(:block,
                :(_ppl_d = $beta * _ppl_carry.d + $sigma * _ppl_elem),
                :(_ppl_x = _ppl_carry.x + _ppl_d),
                :(_ppl_next = (x = _ppl_x, d = _ppl_d)),
                :((_ppl_next, _ppl_x))))
        kw = Expr(:parameters, Expr(:kw, :init, :((x = 0.0, d = 0.0))))
        call = Expr(:call, :scan, kw, zname, :(Ref($beta)), :(Ref($sigma)))
        rest = Symbol(:_ppl_dar_rest_, s.state)
        push!(stmts, :($rest = $(Expr(:do, call, lambda))))
        push!(stmts, :($(s.state) = vcat(0.0, $rest)))
    end
    return stmts
end

# The dar innovation prior: the iid `Normal(0, 1)` plate-vector prior
# shape over the `_ppl_dar_z_<state>` slice (one cell per innovation,
# the same `_plate_sum_stmts` reduction the non-centered-scan path
# uses), totalled under the dar-flavored `_ppl_dar_<state>` node.
function _dar_prior_statements(plan::StructuralPlan, layout::LayoutTable)
    stmts = Expr[]
    nodes = Symbol[]
    for s in plan.dar_paths
        zname = _dar_innovation_name(s)
        cell = _family_logpdf_expr(:normal, Any[0, 1], _dovar(1))
        node = Symbol(:_ppl_dar_, s.state)
        pw = Symbol(:_ppl_dar_pw_, s.state)
        append!(stmts, _plate_sum_stmts(pw, node, Any[zname], cell))
        push!(nodes, node)
    end
    return stmts, nodes
end

function _log_jacobian_statement(plan::StructuralPlan, layout::LayoutTable)
    terms = Any[]
    _each_layout_unit(plan, layout) do unit
        if unit isa LayoutEntry
            t = jacobian_term(unit)
            t === nothing || push!(terms, t)
        else
            append!(terms, _stratified_logjac_terms(unit, layout))
        end
    end
    jac = isempty(terms) ? :(0.0) : foldl((a, b) -> :($a + $b), terms)
    return :(log_jacobian::Float64 = $jac)
end
