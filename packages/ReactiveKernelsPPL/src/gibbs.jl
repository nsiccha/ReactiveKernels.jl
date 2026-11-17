# PPL conditional construction. WarmupHMC owns the continuous transition;
# this layer only derives model blocks and updates their named integer state.
"Prepared model conditionals returned by [`prepare_gibbs`](@ref)."
struct GibbsProblem{L,C,I,F}
    layout::L
    continuous::C
    integers::I
    joint::F
end

"""
    prepare_gibbs(built, plan, u0; discrete, backend=nothing, blocks=nothing)

Prepare exact finite integer conditionals and one continuous conditional over
all packed real coordinates. `blocks` is a tuple of tuples of integer names;
by default each scalar integer declaration is its own block. Group dependent
integers together when single-site updates cannot move between valid states.
Every free integer must occur exactly once. Names use the author namespaces,
for example `Symbol("child.k")` for a nested declaration.

Each conditional keeps the emitted density terms that transitively read its
block. Terms constant in that block are omitted; the resulting conditional
log density need not equal the full joint log density. Finite supports remain
runtime plate axes. A joint block enumerates their Cartesian product.

This initial driver supports explicit scalar integer declarations. Integer
arrays and models with already marginalized integers are not yet supported.
Continuous derivatives use the supplied ordinary AD backend. No backend is
needed for a model with zero continuous coordinates. Prepare one problem per
concurrent chain: its continuous query owns mutable derivative buffers.
"""
function prepare_gibbs(built, plan::StructuralPlan, u0::AbstractVector;
        discrete, backend=nothing, blocks=nothing, on_error=nothing)
    built = _binding_build(built, plan)
    _check_built_program(built, plan)
    isempty(plan.marginalized_integers) || throw(ArgumentError(
        "prepare_gibbs currently requires explicit integer parameters"))
    haskey(plan.frozen.values, :_ppl_discrete) && throw(ArgumentError(
        "prepare_gibbs needs mutable integer inputs, not a frozen integer query"))
    layout = built.layout
    u = Vector{Float64}(u0)
    length(u) == layout.total || throw(DimensionMismatch("Gibbs position has the wrong dimension"))
    _check_integer_parameters(layout, discrete)
    ps = _integer_parameters(plan)
    isempty(ps) && throw(ArgumentError("prepare_gibbs needs at least one integer parameter"))
    all(p -> p isa SampledParameter, ps) || throw(ArgumentError(
        "prepare_gibbs currently enumerates scalar integer declarations; integer arrays are not yet supported"))
    layout.total == 0 || backend !== nothing || throw(ArgumentError(
        "prepare_gibbs requires an AD backend for its continuous block"))
    paths = [get(layout.name_paths, p.name, (p.name,)) for p in ps]
    authored = Symbol.(_scope_path_string.(paths))
    groups = blocks === nothing ? Tuple((name,) for name in authored) : Tuple(blocks)
    all(g -> g isa Tuple && !isempty(g) && all(n -> n isa Symbol, g), groups) ||
        throw(ArgumentError("blocks must contain nonempty tuples of integer parameter names"))
    selected = [name for group in groups for name in group]
    length(selected) == length(authored) && Set(selected) == Set(authored) ||
        throw(ArgumentError("blocks must include each free integer exactly once: $(Tuple(authored))"))

    # Record individual prior/Jacobian events, so independent scalar priors
    # are not grouped into a single aggregate before selecting conditionals.
    sites = _DensitySite[]
    def = Base.invokelatest(_kernel_expr_latest, plan, layout; density_sites=sites)
    def = _elide_named_location_aliases(plan, def)
    def = _pointwise_arithmetic_plates(plan, def)
    def = _name_plate_cells(def, _location_value_aliases(plan, def))
    statements = Expr[s for s in def.args[2].args if s isa Expr && s.head !== :return]
    terms = Any[]
    for name in (:prior, :likelihood, :log_jacobian)
        st = only(s for s in statements if _assigned_name(s) === name)
        append!(terms, _gibbs_sum_terms(st.args[2]))
    end
    discrete = _gibbs_state(discrete)
    exemplar = (u, discrete)
    continuous = if layout.total == 0
        nothing
    else
        dependent = _gibbs_dependents(statements, (:unconstrained,))
        selected_terms = filter(t -> _gibbs_reads(t, dependent), terms)
        body = Expr(:block, statements...,
            :(_ppl_gibbs_density::Float64 = $(_sum_terms(selected_terms))),
            :(return _ppl_gibbs_density))
        spec = _eval_kernel_def(Expr(:(=), def.args[1], body))
        _register_packed_check!(spec, layout)
        kern = Base.invokelatest(_prepare_cached, plan, spec;
            have=_query_have(plan), want=:_ppl_gibbs_density,
            bound=_query_bound(plan), exemplars=exemplar, on_error)
        prep = Base.invokelatest(prepare_ad, kern, backend, exemplar...; active=:unconstrained)
        SamplerQuery(kern, prep, layout, deepcopy(discrete))
    end
    integers = map(groups) do group
        indices = [only(findall(==(name), authored)) for name in group]
        parameters = ps[indices]
        names = [p.name for p in parameters]
        dependent = _gibbs_dependents(statements, names)
        selected_terms = filter(t -> _gibbs_reads(t, dependent), terms)
        supports = [_integer_support_expr(p) for p in parameters]
        needed = Set{Symbol}()
        for ex in (selected_terms..., supports...)
            _statement_value_reads!(needed, ex)
        end
        _gibbs_ancestors!(needed, statements; stop=Set(names))
        outer, cell = Expr[], Expr[]
        for st in statements
            name = _assigned_name(st)
            name in names && continue # supplied by the support's current state
            name in needed || continue
            push!(name in dependent ? cell : outer, st)
        end
        prefix = Expr[:($(name) = _ppl_gibbs_state[$i]) for (i, name) in enumerate(names)]
        push!(outer, :(_ppl_gibbs_states = collect(Iterators.product($(supports...)))))
        body = Expr(:block, LineNumberNode(0, :generator), prefix..., cell...,
            _sum_terms(selected_terms))
        loop = Expr(:do, Expr(:call, :plate, :_ppl_gibbs_states),
            Expr(:->, Expr(:tuple, :_ppl_gibbs_state), body))
        push!(outer, :(_ppl_gibbs_logweights = $loop))
        push!(outer, :(_ppl_gibbs_conditional = (;
            states=_ppl_gibbs_states, logweights=_ppl_gibbs_logweights)))
        push!(outer, :(return _ppl_gibbs_conditional))
        spec = _eval_kernel_def(Expr(:(=), def.args[1], Expr(:block, outer...)))
        _register_packed_check!(spec, layout)
        kernel = Base.invokelatest(_prepare_cached, plan, spec;
            have=_query_have(plan), want=:_ppl_gibbs_conditional,
            bound=_query_bound(plan), exemplars=exemplar, on_error)
        (; names=group, paths=Tuple(paths[indices]), kernel)
    end
    joint = prepare_query(built, plan, :sampler; exemplars=exemplar, on_error)
    return GibbsProblem(layout, continuous, integers, joint)
end

# Only split the addition tree the PPL emitter uses for density aggregates.
# A distribution's own internals remain an indivisible density term.
_gibbs_sum_terms(x) = Meta.isexpr(x, :call) && x.args[1] === :+ ?
    reduce(vcat, _gibbs_sum_terms.(x.args[2:end]); init=Any[]) : Any[x]
_gibbs_reads(x, names) = !isempty(intersect(_statement_value_reads!(Set{Symbol}(), x), names))
function _gibbs_dependents(statements, roots)
    dependent = Set{Symbol}(roots)
    while true
        before = length(dependent)
        for st in statements
            name = _assigned_name(st)
            name === nothing || !_gibbs_reads(st.args[2], dependent) || push!(dependent, name)
        end
        length(dependent) == before && return dependent
    end
end
function _gibbs_ancestors!(needed, statements; stop=Set{Symbol}())
    while true
        before = length(needed)
        for st in statements
            name = _assigned_name(st)
            name in needed && name ∉ stop && _statement_value_reads!(needed, st.args[2])
        end
        length(needed) == before && return needed
    end
end

function _gibbs_set(state::NamedTuple, path::Tuple, value)
    key, rest = first(path), Base.tail(path)
    updated = isempty(rest) ? convert(typeof(state[key]), value) :
        _gibbs_set(state[key], rest, value)
    return merge(state, NamedTuple{(key,)}((updated,)))
end
_gibbs_state(state::NamedTuple) = map(_gibbs_state, state)
_gibbs_state(value::Integer) = Int(value)
_gibbs_state(value) = deepcopy(value)
function _gibbs_weights(logweights)
    isempty(logweights) && throw(ArgumentError("Gibbs block has empty support"))
    all(w -> isfinite(w) || w == -Inf, logweights) || throw(ArgumentError(
        "Gibbs conditional contains NaN or positive infinity"))
    top = maximum(logweights)
    isfinite(top) || throw(ArgumentError("Gibbs conditional has zero probability at every state"))
    weights = exp.(vec(logweights) .- top)
    return weights ./ sum(weights)
end

"""
    initialize_gibbs(rng, problem, u0; discrete, stepsize=nothing, metric=nothing,
        max_tree_depth=10, warmup=1000)

Initialize a chain for `prepare_gibbs`. Load `WarmupHMC`, `LogDensityProblems`
and `Random`.

Without `stepsize`, the continuous NUTS kernel is adapted during `warmup`
sweeps. Each sweep draws the integers exactly and then takes one transition of
WarmupHMC's continuously adapting kernel (`ContinuousNUTSState`, started by
Pathfinder from `u0`), which follows the continuous conditional as the
integers change. The chain starts at the last warm-up state, and its kernel
is then fixed: the adapted diagonal scale and step size. `state.warmup`
reports the warm-up sweeps, their divergences and gradient evaluations.

With `stepsize`, that fixed kernel is used from the start and no warm-up runs;
`metric` is then a scale L (covariance L*L'), defaulting to identity. A
`metric` without a `stepsize` is refused. Models without continuous
coordinates run no warm-up.

The state owns copies of the initial position, integer state and RNG, storing
integer parameters as `Int` so a narrow initial integer type does not truncate
the enumerated support. Integer
draws and NUTS use separate reproducible RNG streams. The full joint density
must be finite at the supplied start.
"""
function initialize_gibbs end

"""
    gibbs_step!(state)

Draw each finite integer block exactly in declaration/block order, then take
one WarmupHMC fixed-kernel NUTS transition on all continuous coordinates.
Return `(; state, divergent, stats)`. The state advances in place only after
the complete sweep succeeds. Each continuous transition reevaluates the
target under the new integers. No adaptation occurs in this operation.
"""
function gibbs_step! end

"""
    gibbs_draws(state, n; discard=0)

Advance a Gibbs state through `discard` sweeps, then save `n` sweeps. Returns
owned `positions` (coordinates × draws), `discrete` states, constrained
`parameters`, and per-draw `divergent` flags. Discarding sweeps does not adapt
the fixed NUTS kernel; adaptation happens only in `initialize_gibbs`.
Further steps cannot mutate saved draws.
"""
function gibbs_draws end
