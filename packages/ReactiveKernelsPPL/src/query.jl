# Sampler-query surface: canonical node registry + prepared queries over
# `build_kernel` output for the BRM-side LDP shim.
#
# Mirrors the `PPLWorkflow` pattern (registry + validated presets + thin
# `prepare` wrappers) but is self-contained: the thin layer must not depend on
# the examples package. The node set is the slice-1 subset of the PPLWorkflow
# vocabulary — the generator grows nodes as scope grows.

"""
    PPL_NODES

Canonical node names a generated `@kernel` program exposes, as a NamedTuple
mapping role → node `Symbol`. Slice-1 subset of the `PPLWorkflow.PPL_NODES`
vocabulary: every program built by [`build_kernel`](@ref) defines these nodes.
"""
const PPL_NODES = (
    likelihood = :likelihood,
    prior = :prior,
    log_jacobian = :log_jacobian,
    posterior = :posterior,
    pointwise = :pointwise,
    observations = :_ppl_observations,
)

"""
    WORKFLOW_WANTS

Preset `want` selections over generated programs, as a NamedTuple mapping
preset name → the `want` argument. Prefer [`workflow_wants`](@ref), which
validates the preset name.

`:pointwise` returns a `NamedTuple` keyed by observation name.
Elementwise observations retain their array shape; scalar and
joint-vector draws return scalars, and broadcast joint draws return one
density per slice. In-cell observations use their caller's data-column
name and flat data shape. Conditioned declarations contribute here as
observations. Summing every field gives `:likelihood`; prior-only models
return an empty `NamedTuple`.

`:observations` returns, under the same keys, the arguments each
observation's distribution receives: `(location = …, scale = …)` for
responses of the Normal, Student t and LogNormal families, where entry `i`
belongs to entry `i` of the response's `:pointwise` densities and a value
shared by every observation stays a scalar. Prior-only models return an empty
`NamedTuple`. For a response holding one array per `@plate` index, locations
and scales follow the nested `:pointwise` order: a per-entry value holds one
array per index, a per-index scalar holds one number per index, and a shared
scalar stays scalar. Preparing it for a program with any other observation
(another family, a sampling-RHS observation or a conditioned declaration)
throws a `ContractValidationError` naming them.
"""
const WORKFLOW_WANTS = (
    sampler = :posterior,
    likelihood = :likelihood,
    prior = :prior,
    log_jacobian = :log_jacobian,
    pointwise = :pointwise,
    observations = :_ppl_observations,
)

"""
    workflow_wants(preset::Symbol)

Return the `want` selection for a named workflow `preset` (see
[`WORKFLOW_WANTS`](@ref)). Throws `ArgumentError` naming the valid presets
when `preset` is unknown, so a typo fails loudly rather than planning the
wrong cut.
"""
function workflow_wants(preset::Symbol)
    haskey(WORKFLOW_WANTS, preset) || throw(ArgumentError(
        "unknown PPL query preset $(repr(preset)); choose one of $(keys(WORKFLOW_WANTS))"))
    WORKFLOW_WANTS[preset]
end

# The sampler boundary: packed unconstrained parameters plus the raw data
# columns, sorted by name (same order the generator uses for data args).
# Values the binding leaves absent are not inputs: a query reading one has
# nothing to read (`_absent_refusal`). Frozen definitions the graph computes
# are inputs too, at their own nodes: RK computes neither them nor what only
# they read (`_frozen_graph_names`).
_query_columns(plan::StructuralPlan) =
    sort!([k for (k, v) in plan.columns if _supplied(v)])

_query_have(plan::StructuralPlan) =
    (:unconstrained, _query_columns(plan)..., _frozen_graph_names(plan)...)

function _query_bound(plan::StructuralPlan)
    names = _query_columns(plan)
    return merge(NamedTuple{Tuple(names)}(Tuple(plan.columns[k] for k in names)),
        _frozen_graph_values(plan))
end

# One binding owns both the interpreter's named values and the core graph's
# prefix results. Graph copies for specialized queries share authored data
# definitions through their port names; anonymous values stay graph-local.
function _with_query_data(f, plan::StructuralPlan, spec, on_error)
    data = plan.data_values
    data === nothing && return f(nothing)
    lock(data.lock) do
        policies = get!(Dict{Any,Any}, data.graphs, spec.graph)
        cache = get!(policies, (on_error, ReactiveKernels._plan_mode())) do
            ReactiveKernels._DataValues(spec.graph)
        end
        exprs, _ = _definition_exprs(plan)
        empty!(cache.supplied)
        if on_error === nothing
            for (nm, value) in data.values
                haskey(exprs, nm) && haskey(spec.ports, nm) && _supplied(value) || continue
                cache.supplied[ReactiveKernels.canon_id(spec.graph, spec[nm].id)] = value
            end
        end
        result = f(cache)
        if on_error === nothing
            for nm in keys(exprs)
                haskey(spec.ports, nm) || continue
                key = ReactiveKernels.canon_id(spec.graph, spec[nm].id)
                haskey(cache.values, key) && (data.values[nm] = cache.values[key])
            end
        end
        result
    end
end

function _prepare_cached(plan::StructuralPlan, spec; on_error = nothing, kwargs...)
    _with_query_data(plan, spec, on_error) do cache
        prepare(spec; kwargs..., on_error, _data_cache = cache)
    end
end

_prepare_cached_partition(plan, spec; kwargs...) =
    task_local_storage(() -> _prepare_cached(plan, spec; kwargs...),
        ReactiveKernels._PLAN_MODE, true)

# Run a preparation of `want` from `have` over `plan`'s inputs. When it
# needs a value the binding leaves absent, RK's planner finds no producer for
# that input; planning again with the absent inputs available shows which
# ones the query reads, and the query is refused naming their observations.
function _absent_refusal(f, plan::StructuralPlan, spec, have, want)
    absent = _absent_inputs(plan)
    isempty(absent) && return f()
    try
        return f()
    catch e
        e isa ReactiveKernels.PlanningError || rethrow()
        selected = try
            Base.invokelatest(ReactiveKernels.plan, spec;
                have = (have..., absent...), want)
        catch
            rethrow(e)
        end
        # The planner collapses aliases (`k = <conditioned k>`): compare the
        # inputs' canonical values, not their port names.
        g = spec.graph
        canon(v) = ReactiveKernels.canon_id(g, v.id)
        reads = Set(canon(v) for r in selected.recipes for v in r.inputs)
        union!(reads, (canon(v) for v in selected.want))
        needed = filter(k -> haskey(spec.ports, k) && canon(spec.ports[k]) in reads, absent)
        isempty(needed) && rethrow(e)
        _refuse_absent(plan, unique!([_absent_source(plan.columns[k]) for k in needed]))
    end
end

# A cell query evaluates its observations' densities, which read them.
function _refuse_absent_cells(plan::StructuralPlan, names)
    absent = intersect(names, _absent_observations(plan))
    isempty(absent) || _refuse_absent(plan, absent)
    return nothing
end

function _refuse_absent(plan::StructuralPlan, names)
    quoted(ns) = join(("`$n`" for n in ns), ", ", " and ")
    them = quoted(names)
    it = length(names) == 1 ? "it" : "them"
    supply = unique!(Symbol[s for n in names for s in _absent_supply_names(plan, n)])
    what = supply == names ? "no value was supplied" :
        "no value was supplied for $(quoted(supply)), which " *
        (length(names) == 1 ? "it is" : "they are") * " computed from"
    throw(ContractValidationError("[query] this binding leaves $them absent " *
        "($what), and this query reads $it. Supply $(quoted(supply)), " *
        "or query values that do not read $it, such as a location by its " *
        "authored name or the `:observations` preset."))
end

# What a caller supplies to observe the absent `name`: its value, or, for a
# derived response (`_absent_derived_responses!`), the omitted data it reads.
function _absent_supply_names(plan::StructuralPlan, name::Symbol)
    name in _derived_response_names(plan) || return Symbol[name]
    order = _definition_read_orders(plan, _sorted_names(keys(plan.frozen.values)))[name]
    raw = Symbol[s for s in order if _is_absent_raw_source(plan, s, get(plan.columns, s, nothing))]
    return isempty(raw) ? Symbol[name] : raw
end

"""
    prepare_query(built, plan, preset::Symbol; on_error = nothing, exemplars = nothing)

Prepare the `build_kernel` output `built` for a named workflow `preset`
over the sampler boundary (`:unconstrained` + data columns, data hoisted
via `bound=`). Thin wrapper over `ReactiveKernels.prepare` with
`want = workflow_wants(preset)` and the supplied `on_error` policy. The
optional `exemplars` tuple is forwarded to RK's native type-evidence path.
The returned kernel maps an unconstrained float vector to the preset node's
value. `plan` may be any binding that generates the same program as `built`
(other row counts, including zero, and data values, or the same source lowered
in another gensym-created isolation module); one that generates another program
(other levels or parameter sizes, data element types, missing entries,
another broadcast shape) is refused with a
`ContractValidationError` naming the first difference, since the graph would
evaluate the wrong program for it. The preparation itself is
world-age safe (callable from compiled functions), but the RAW returned
kernel closes over build-time eval'd code: call it from top level, wrap
the call in `Base.invokelatest`, or use [`SamplerQuery`](@ref), whose
call paths carry the barrier.
"""
function prepare_query(built, plan::StructuralPlan, preset::Symbol;
        on_error = nothing, exemplars = nothing)
    return _prepare_wants(built, plan, workflow_wants(preset);
        on_error, exemplars)
end

# Prepare the built graph over the sampler boundary for one WANT node or a
# tuple of them (a tuple-WANT kernel returns a tuple in that order).
function _prepare_wants(built, plan::StructuralPlan, want;
        on_error = nothing, exemplars = nothing)
    isbound(plan) || throw(ContractValidationError(
        "[query] prepare_query requires a bound plan (bind_data first)"))
    _check_built_program(built, plan)
    # World-age barrier: `built.spec` holds closures eval'd at build time
    # (newer than any already-compiled caller), so partial evaluation must
    # run at the latest world. Same barrier guards every call below.
    have = _query_have(plan)
    return _absent_refusal(plan, built.spec, have, want) do
        Base.invokelatest(_prepare_cached, plan, built.spec; have,
            want, bound = _query_bound(plan), on_error, exemplars)
    end
end

# A built graph evaluates the program its build generated. A binding that
# generates the same program (other row counts and data values, or another
# gensym-created lowering namespace) evaluates on it exactly as on its own
# build. A binding that generates another program
# (other level labels or parameter sizes, data element types, missing
# entries, another broadcast shape) needs its own build;
# evaluating the old graph on it can be silently wrong, so it is refused.
# Values made outside `build_kernel` carry no `program` and are not checked.
# A value the binding leaves absent has the type its input has in the built
# program: the binding supplies no value to say otherwise.
function _check_built_program(built, plan::StructuralPlan)
    hasproperty(built, :program) || return nothing
    _checked_program(plan, built.program) && return nothing
    typed = _absent_typed_as(plan, built.program)
    program = _program_identity(kernel_expr(typed, assign_layout(typed)), typed)
    _same_program(program, built.program) || throw(ContractValidationError(
        "[query] this binding generates a " *
        "different program than the one this graph was built from, so it " *
        "needs its own `build_kernel(plan)`. First difference, " *
        _program_difference(built.program, program)))
    _record_checked_program!(plan, built.program)
    return nothing
end

# A binding generates its program from its own fields and bound values, which
# no step mutates (plans are immutable by contract), so a binding that
# generated a build's program once does again. Generating it costs the whole
# program, however small the query: about 45 s on a 224k-coordinate BRM
# program, paid by every preparation of one cell (snag
# `brm-s-nlme-cell-d70abd20`). So each binding is checked against each build
# once. The record sits in the binding's lowering cache (`_structural_cache`)
# and holds the binding's column table and the build's program weakly.
mutable struct _CheckedPrograms
    const columns::WeakRef
    programs::Vector{WeakRef}
end

_checked_program_key(plan::StructuralPlan) =
    (:checked_program, _structure_fields(plan), objectid(plan.columns), plan.n_obs)

function _checked_program(plan::StructuralPlan, program::Expr)
    cache = _structural_cache(plan)
    lock(cache.lock) do
        checked = get(cache.entries, _checked_program_key(plan), nothing)
        checked isa _CheckedPrograms && checked.columns.value === plan.columns &&
            any(ref -> ref.value === program, checked.programs)
    end
end

function _record_checked_program!(plan::StructuralPlan, program::Expr)
    cache = _structural_cache(plan)
    key = _checked_program_key(plan)
    lock(cache.lock) do
        checked = get(cache.entries, key, nothing)
        if !(checked isa _CheckedPrograms && checked.columns.value === plan.columns)
            length(cache.entries) >= _STRUCTURAL_CACHE_LIMIT && empty!(cache.entries)
            checked = cache.entries[key] = _CheckedPrograms(WeakRef(plan.columns), WeakRef[])
        end
        filter!(ref -> ref.value !== nothing, checked.programs)
        push!(checked.programs, WeakRef(program))
    end
    return nothing
end

function _absent_typed_as(plan::StructuralPlan, program::Expr)
    absent = _absent_inputs(plan)
    isempty(absent) && return plan
    types = Dict{Symbol,Any}()
    for a in program.args[1].args[2:end]
        Meta.isexpr(a, :(::), 2) && a.args[1] isa Symbol && (types[a.args[1]] = a.args[2])
    end
    columns = copy(plan.columns)
    for k in absent
        v, T = columns[k], get(types, k, nothing)
        T isa Type || continue
        if v isa _AbsentArray && T <: AbstractVector{<:AbstractArray} && ndims(v) == 1
            # The graph observes one array per index: one cell per index,
            # each of one entry (shapes are not part of the program).
            columns[k] = _absent_cells(v.name, eltype(T),
                fill(ntuple(_ -> 1, ndims(eltype(T))), length(v)))
        elseif v isa _AbsentCells && T <: AbstractVector{<:AbstractArray}
            columns[k] = _AbsentCells{eltype(T)}(v.name, v.cells)
        elseif v isa _AbsentArray && T <: AbstractArray && ndims(T) == ndims(v)
            columns[k] = _AbsentArray{eltype(T),ndims(v)}(v.name, v.dims)
        elseif v isa _AbsentNumber && T <: Real
            columns[k] = _AbsentNumber{T}(v.name)
        end
    end
    return _with(plan; columns)
end

function _program_difference(built::Expr, other::Expr)
    # Fresh `Expr` nodes for `remove_linenums!` to edit; the values they hold
    # are only printed. A long quoted array (a level axis's labels) prints as
    # its summary and ends.
    text(x) = x === nothing ? "(absent)" :
        sprint(print, x isa Expr ? Base.remove_linenums!(_elide_long_arrays(x)) :
            _elide_long_arrays(x))
    sig(def) = def.args[1]
    _same_program(sig(built), sig(other)) || return "in the data arguments:\n  built: " *
        text(sig(built)) * "\n  this:  " * text(sig(other))
    body(def) = Any[x for x in def.args[2].args if !(x isa LineNumberNode)]
    a, b = body(built), body(other)
    for i in 1:max(length(a), length(b))
        x, y = get(a, i, nothing), get(b, i, nothing)
        _same_program(x, y) && continue
        return "statement $i:\n  built: " * text(x) * "\n  this:  " * text(y)
    end
    return "in the program's layout"
end

# A copy of `x` whose quoted arrays of more than `limit` elements are
# replaced by a one-line summary with their first and last elements.
function _elide_long_arrays(x; limit = 20)
    x isa Expr && return Expr(x.head, Any[_elide_long_arrays(a; limit) for a in x.args]...)
    x isa QuoteNode && x.value isa AbstractArray && length(x.value) > limit || return x
    v = vec(x.value)
    shown(r) = join((string(e) for e in v[r]), ", ")
    return "<" * summary(x.value) * ": " * shown(1:6) * ", … (" *
        string(length(v) - 8) * " more), " * shown(length(v) - 1:length(v)) * ">"
end

"""
    SamplerQuery

Reusable sampler-space density + gradient over a built program: the prepared
`:posterior` value kernel plus its `prepare_ad` gradient preparation.
Construct with [`prepare_sampler`](@ref); not thread-safe (one per caller).
The call paths below carry a `Base.invokelatest` barrier.

The barrier copies nothing: a gradient call allocates only what the prepared
gradient itself allocates, and a value call adds one boxed `Float64`.
"""
mutable struct SamplerQuery{K,P,L}
    const kernel::K
    const ad::P
    const layout::L
    value::Float64
end
SamplerQuery(kernel, ad, layout) = SamplerQuery(kernel, ad, layout, NaN)

# Per-evaluation calls cross the world-age barrier with references only.
# `Base.invokelatest` passes each argument as a boxed reference, so a prepared
# kernel or AD operator stored inline in an immutable query was copied to the
# heap on every call (1.1-2.5 KB per sampler gradient of a small model). The
# query types are mutable: the query itself crosses uncopied. A value call
# returns its value through the barrier, which keeps it differentiable through
# the query (callers differentiate `q(u, i)`). A gradient call instead stores
# its value in the query's `value` field rather than returning a boxed
# `(value, gradient)` tuple; queries are one per caller, so the field is never
# shared. The helpers specialize on their function and argument count; an
# unspecialized vararg method called dynamically would allocate and dispatch
# again inside.
_store_value!(f::F, query, args::Vararg{Any,N}) where {F,N} =
    (query.value = f(query, args...); nothing)
function _latest_value(f::F, query, args::Vararg{Any,N}) where {F,N}
    Base.invokelatest(_store_value!, f, query, args...)
    return query.value
end

"""
    prepare_sampler(built, plan, u0::AbstractVector{<:Real}; backend, on_error = nothing, retain = (), warm = false, native_forward_batch = false, nonallocating = false) -> SamplerQuery

Prepare a reusable [`SamplerQuery`](@ref): the `:sampler`-cut value kernel
plus a `prepare_ad` gradient with `active = :unconstrained`. `u0` is a
length-consistent type/shape exemplar (checked against `layout.total`
before any preparation work); `backend` is any
`DifferentiationInterface.AbstractADType` (e.g. `AutoEnzyme` reverse mode).
`on_error` is forwarded to [`prepare_query`](@ref). The backend's package
must be loaded in the calling session (`using Enzyme` for `AutoEnzyme`);
the backend value alone does not load
DifferentiationInterface's backend extension, and preparation without it
fails loudly naming the missing `using`.

The sampler passes its converted `Vector{Float64}` exemplar to native
preparation by default. Its public calls convert other real vectors to that
type; the prepared native kernel uses the concrete type to infer plate inputs
without fixing the vector's length.

`retain` names further query presets (`:pointwise`, `:likelihood`, `:prior`,
`:log_jacobian`) whose values the gradient's own primal sweep returns through
[`sampler_value_gradient_and_retained!`](@ref), with no second primal pass.
It needs a backend that keeps that sweep's values (reverse-mode
`AutoEnzyme`; see `ReactiveKernels.ad_retains_primal_sweep`).

`warm = true` compiles the gradient during preparation (`prepare_ad`'s `warm`),
so [`sampler_value_and_gradient!`](@ref) with a `Vector{Float64}` point runs
compiled code from its first call.

The sampler preserves an explicitly requested AD mode. Native forward batching
is disabled by default. With `AutoEnzyme()` (no explicit mode), the opt-in
`native_forward_batch = true` may select one batched forward sweep for a pure
independent summed plate with one to four active coordinates. It differentiates
the unchanged generated kernel, including priors and transforms. Explicit
reverse mode, wider, effectful, scan, and retained-output programs keep the
ordinary reverse preparation. The opt-in forward preparation keeps a reverse
operator for a later input with a different active shape.

`nonallocating = true` prepares the density and its gradient over
`ReactiveKernels.prepare_nonallocating`, which needs `using MutatingFunctions`
in the calling session. Every operation with a destination-passing form then
writes into a buffer the query owns: an evaluation reuses the same memory
instead of allocating fresh observation-length intermediates (matrix-vector
products, plate outputs) on every call. Values and gradients equal the
ordinary sampler's. The density is usually faster, and loses less when another
allocating process shares the CPU core; the reverse gradient is faster on some
programs and slower on others, so measure both on the model you fit. The
query keeps the buffers, which is why a query serves one caller at a time.
Native forward batching does not apply,
and ReactiveKernels does not build retained outputs over a non-allocating
kernel yet, so `retain` fails there.
"""
function prepare_sampler(built, plan::StructuralPlan, u0::AbstractVector{<:Real};
        backend, on_error = nothing, retain::Tuple = (), warm::Bool = false,
        native_forward_batch::Bool = false, nonallocating::Bool = false)
    layout = built.layout::LayoutTable
    length(u0) == layout.total || throw(ContractValidationError(
        "[query] exemplar length $(length(u0)) ≠ layout total $(layout.total)"))
    u = Vector{Float64}(u0)
    exemplar = (u,)
    kern = ReactiveKernels._kernel_latest_prepared(
        prepare_query(built, plan, :sampler; on_error, exemplars = exemplar))
    gradient_kernel, wants = if isempty(retain)
        kern, ()
    else
        wants = _retained_wants(retain)
        ReactiveKernels._kernel_latest_prepared(_prepare_wants(built, plan,
            (workflow_wants(:sampler), wants...); on_error,
            exemplars = exemplar)), wants
    end
    if nonallocating
        kern = Base.invokelatest(ReactiveKernels.prepare_nonallocating, kern, u)
        gradient_kernel = isempty(wants) ? kern :
            Base.invokelatest(ReactiveKernels.prepare_nonallocating, gradient_kernel, u)
    end
    # The non-allocating `prepare_ad` method takes no forward-batching keyword.
    forward = nonallocating ? (;) : (; native_forward_batch = native_forward_batch &&
        isempty(wants) && ReactiveKernels._native_forward_batch_eligible(
            backend, gradient_kernel, u))
    prep = Base.invokelatest(prepare_ad, gradient_kernel, backend,
        u; active = :unconstrained, retain = wants, warm, forward...)
    return SamplerQuery(kern, prep, layout)
end

# The WANT nodes for retained presets. The sampler density is the gradient's
# own value, so it is not a retained output.
function _retained_wants(retain::Tuple)
    allunique(retain) || throw(ArgumentError(
        "[query] retain names the same preset more than once: $retain"))
    map(retain) do preset
        preset isa Symbol || throw(ArgumentError(
            "[query] retain entries are query preset names; got $(repr(preset))"))
        preset === :sampler && throw(ArgumentError(
            "[query] the sampler density is the gradient's value; retain " *
            "only other presets $(filter(!=(:sampler), keys(WORKFLOW_WANTS)))"))
        workflow_wants(preset)
    end
end

"""
    (q::SamplerQuery)(u) -> Float64

Posterior value at unconstrained `u`. Zero-copy for `Vector{Float64}` (the
HMC-loop case); any other real vector is converted. `u` must have
`q.layout.total` entries.
"""
# The generated kernel is scalar-valued, but invokelatest itself infers Any.
# Assert the public density contract after crossing the world-age barrier.
(q::SamplerQuery)(u::Vector{Float64}) =
    Base.invokelatest(_sampler_density, q, u)::Float64
(q::SamplerQuery)(u::AbstractVector{<:Real}) = q(Vector{Float64}(u))
_sampler_density(q::SamplerQuery, u) = q.kernel(u)

"""
    sampler_value_and_gradient!(q::SamplerQuery, g::AbstractVector, u)

Posterior value and gradient at unconstrained `u`, writing the gradient
into `g` in place. Returns the `(value, g)` pair. `g` must be a
valid gradient destination for `u`; `u` is converted to `Vector{Float64}`
unless it already is one.
"""
sampler_value_and_gradient!(q::SamplerQuery, g::AbstractVector, u::Vector{Float64}) =
    (_latest_value(_sampler_gradient!, q, g, u), g)
sampler_value_and_gradient!(q::SamplerQuery, g::AbstractVector, u::AbstractVector) =
    sampler_value_and_gradient!(q, g, Vector{Float64}(u))
_sampler_gradient!(q::SamplerQuery, g, u) = first(ad_value_and_gradient!(q.ad, g, u))

"""
    sampler_value_gradient_and_retained!(q::SamplerQuery, g::AbstractVector, u)

Posterior value and gradient at unconstrained `u` as
[`sampler_value_and_gradient!`](@ref) computes them, plus the presets named
by `prepare_sampler(...; retain)` evaluated by the same primal sweep. Returns
`(value, g, retained)`, where `retained` is a `NamedTuple` keyed by those
preset names (`retained.pointwise` has the shape `prepare_query(…, :pointwise)`
returns). Each call returns freshly computed retained values.
"""
function sampler_value_gradient_and_retained!(q::SamplerQuery, g::AbstractVector,
        u::AbstractVector)
    v = u isa Vector{Float64} ? u : Vector{Float64}(u)
    return Base.invokelatest(_sampler_retained!, q, g, v)
end
_sampler_retained!(q::SamplerQuery, g, u) = ad_value_gradient_and_retained!(q.ad, g, u)

"""
    QueryAD

A prepared ReactiveKernels derivative operator over one query of a built
program, plus the program's [`LayoutTable`](@ref). Construct one with
[`prepare_query_ad`](@ref). The ReactiveKernels derivative calls accept it in
place of the prepared operator, with the unconstrained vector as the one
argument: `ad_gradient(q, u)`, `ad_value_and_gradient!(q, g, u)`,
`ad_pullback(q, seed, u)`, `ad_pushforward(q, tangents, u)`,
`ad_hvp(q, tangents, u)`, `ad_gradient_and_hvp(q, tangents, u)`,
`ad_third(q, tangents, u)`, `ad_gradient_hvp_and_third(q, tangents, u)` and
their other value/in-place forms. Each call carries the `Base.invokelatest` barrier
the query's build-time code needs, so they are safe from compiled callers.
Not thread-safe (one per caller).
"""
mutable struct QueryAD{P,L}
    const prepared::P
    const layout::L
end

"""
    prepare_query_ad(prepare_operator, built, plan, preset, backend, exemplars...;
                     on_error = nothing) -> QueryAD

Prepare a ReactiveKernels derivative operator over the query
[`prepare_query`](@ref)`(built, plan, preset)`, differentiating with respect
to the whole unconstrained vector. `prepare_operator` is the ReactiveKernels
preparation to apply — `prepare_ad`, `prepare_ad_pullback`,
`prepare_ad_pushforward`, `prepare_ad_hvp` or `prepare_ad_third` — and
`exemplars` are that
function's arguments after its backend, ending with an unconstrained exemplar
`u0` of length `layout.total`:

```julia
second_order = SecondOrder(AutoEnzyme(; mode = Enzyme.Reverse),
                           AutoEnzyme(; mode = Enzyme.Forward))
q = prepare_query_ad(prepare_ad_hvp, built, plan, :sampler, second_order, (v,), u0)
gradient, (hv,) = ad_gradient_and_hvp(q, (v,), u)

j = prepare_query_ad(prepare_ad_pushforward, built, plan, :observations,
                     AutoEnzyme(; mode = Enzyme.Forward), (v,), u0)
observations, (jv,) = ad_value_and_pushforward(j, (v,), u)

third_order = SecondOrder(second_order, AutoEnzyme(; mode = Enzyme.Forward))
t = prepare_query_ad(prepare_ad_third, built, plan, :sampler, third_order, (v,), u0)
gradient, (hv,), (tv,) = ad_gradient_hvp_and_third(t, (v,), u)  # tv = ∇(vᵀ H v)
```

Gradients, Hessian-vector products and third-order derivatives need a scalar
preset (`:sampler`, `:likelihood`, `:prior`, `:log_jacobian`); pullbacks and
pushforwards also accept `:pointwise` and `:observations`. Directions with
disjoint supports over `coordinate_names(built.layout)` compress
block-structured Jacobians, Hessians and third-order contractions (see
`ReactiveKernels.prepare_ad_hvp` and `ReactiveKernels.prepare_ad_third`). The
backend's packages must be loaded in the calling session.
"""
function prepare_query_ad(prepare_operator, built, plan::StructuralPlan,
        preset::Symbol, backend, exemplars...; on_error = nothing)
    layout = built.layout::LayoutTable
    isempty(exemplars) && throw(ContractValidationError(
        "[query] prepare_query_ad needs an unconstrained exemplar u0 as its last argument"))
    u0 = last(exemplars)
    u0 isa AbstractVector{<:Real} && length(u0) == layout.total ||
        throw(ContractValidationError(
            "[query] the last exemplar must be an unconstrained vector of length " *
            "$(layout.total); got $(summary(u0))"))
    kernel = prepare_query(built, plan, preset; on_error)
    prepared = Base.invokelatest(prepare_operator, kernel, backend,
        Base.front(exemplars)..., Vector{Float64}(u0); active = :unconstrained)
    return QueryAD(prepared, layout)
end

for verb in (:ad_gradient, :ad_value_and_gradient, :ad_value_and_gradient!,
        :ad_pullback, :ad_value_and_pullback, :ad_value_and_pullback!,
        :ad_pushforward, :ad_value_and_pushforward,
        :ad_hvp, :ad_hvp!, :ad_gradient_and_hvp, :ad_gradient_and_hvp!,
        :ad_third, :ad_third!, :ad_gradient_hvp_and_third,
        :ad_gradient_hvp_and_third!)
    @eval ReactiveKernels.$verb(q::QueryAD, args...) =
        Base.invokelatest(_query_ad_call, ReactiveKernels.$verb, q, args...)
end
_query_ad_call(verb::F, q::QueryAD, args::Vararg{Any,N}) where {F,N} =
    verb(q.prepared, args...)

"""
    CellQuery

The log density of one cell of the plates that observe the selected
observations: for an `@plate for i in …` loop, the densities its iteration `i`
observes; for an elementwise observation (`y .~ …`), the density of entry `i`.
Construct with [`prepare_cell_query`](@ref) and call `q(u, i)`. The call paths
carry the `Base.invokelatest` barrier, as `SamplerQuery`'s do. Not thread-safe:
prepare one per thread.
"""
mutable struct CellQuery{K,L,C,I,S}
    const kernel::K
    const layout::L
    const observations::Tuple{Vararg{Symbol}}
    const prefix::C
    const specializations::I
    const shared::S
end
CellQuery(kernel, layout, observations, prefix, specializations) =
    CellQuery(kernel, layout, observations, prefix, specializations, nothing)
CellQuery(kernel, layout, observations, prefix) =
    CellQuery(kernel, layout, observations, prefix, nothing)
CellQuery(kernel, layout, observations) = CellQuery(kernel, layout, observations, nothing)

(q::CellQuery)(u::AbstractVector{Float64}, cell) =
    Base.invokelatest(_cell_density, q, u, cell)::Float64
(q::CellQuery)(u::AbstractVector{<:Real}, cell) = q(Vector{Float64}(u), cell)
function _cell_density(q::CellQuery, u, cell)
    if q.prefix === nothing
        _refresh_cell_shared!(q.shared, u)
        return _cell_density_uncached(q, u, cell)
    end
    length(u) == q.layout.total || throw(ContractValidationError(
        "[query] packed length $(length(u)) ≠ layout total $(q.layout.total)"))
    prefix = _query_prefix_kernel(q.prefix, u)
    cached = q.shared === nothing ? _refresh_partition_prefix!(q.prefix.cache, u, prefix) :
        _refresh_cell_shared!(q.shared, u, prefix)
    residual = _query_residual(q.prefix, (u, cell, cached...))
    residual(u, cell, cached...)
end
function _cell_density_uncached(q::CellQuery, u, cell)
    u isa Vector{Float64} && cell isa Int && return q.kernel(u, cell)
    q.specializations === nothing && return q.kernel(u, cell)
    # Preserve local reads through views and custom packed vectors.
    args = (u, cell)
    kernel = get!(q.specializations.kernels, typeof(args)) do
        Base.invokelatest(prepare, q.kernel.plan;
            exemplars = args, on_error = q.specializations.on_error)
    end
    kernel(u, cell)
end
ReactiveKernels.partition_locality(q::CellQuery) =
    ReactiveKernels.partition_locality(q.prefix === nothing ? q.kernel : q.prefix.residual)

"""
    prepare_cell_query(built, plan, observations; on_error = nothing, locality = nothing,
                       partitioned = (), u0 = nothing) -> CellQuery

Prepare the summed log density of ONE cell of the observations named in
`observations` (a name or a collection of names), evaluated at an
unconstrained point `u` and a position `i`: `q(u, i)`. Each observation's
pointwise values come from a plate (`:pointwise`'s array for that name); cell
`i` is that plate's entry `i`, so

    sum(q(u, i) for i in eachindex(pointwise.y)) == pointwise likelihood of y

`locality` says what preparation does with a model value the cell reads at its
rows but would compute whole (RK `plate_cell`): `:error` refuses the query
with a `ReactiveKernels.PartitionLocalityError` naming each such value and its
statement, `:warn` names each once per process, `:ignore` says nothing;
`nothing` reads the ReactiveKernels preference `cell_locality` (`"warn"`
unless set). `ReactiveKernels.partition_locality(q)` lists the executed
residual's values.

Optional `partitioned` identifies parameters that can change between local
evaluations, using the names or `parameter => level_labels` accepted by
[`prepare_partition_sampler`](@ref). Other coordinates are shared. The query
caches their intermediate values across cells and local changes, compares
shared coordinates by value (including in-place edits), and refreshes on a
shared change. Row-capable calculations remain in the residual. This cell
declaration identifies cache dependencies; it adds no per-key separability
claim because a cell query has no partition key. A partition query checks
separability when the same declaration is supplied.

Native preparation initially uses a packed `Vector{Float64}` and an `Int`
cell position. Query calls preserve other `AbstractVector{Float64}`
inputs and cache specializations for their argument types, including Cartesian
indices. Other element types are converted to `Float64`. Direct calls to `q.kernel` require
the initial exact types.
`u0` supplies a valid packed preparation point for the shared
prefix; it defaults to zeros. It is copied. Each query owns its cache and is not
thread-safe. `q.kernel` remains the original packed-coordinate kernel for
the footprint recipe below; through-prefix dependencies are not lost.
For cached evaluation call `q(u, i)`, and use `partition_locality(q)` to
inspect its residual.

For an `@plate for i in eachindex(y)` loop holding one array per index
(`y[i] .~ …`), entry `i` is the sum of iteration `i`'s densities. Several
observations of one loop share the position: `q(u, i)` adds their cells.

Only cell `i` of each plate runs, through the plate's own body (RK
`plate_cell`): an `@plate for` loop's location plate is composed into it, so
the cell's own observations cost what one iteration costs. Values the cell
reads whole, such as a predictor vector `log_k` indexed by `log_k[i]`, are
computed whole. Observations without a pointwise plate (a scalar `~`
observation, a joint multivariate response) are refused naming the
observation. `plan` must generate the program `built` was built from, as for
[`prepare_query`](@ref).

For the packed coordinates a cell reads, use RK's structural footprint API:

```julia
q = prepare_cell_query(built, plan, :y; locality = :error)
footprint = ReactiveKernels.partition_footprint_kernel(q.kernel)
read = Base.invokelatest(footprint, u, i).unconstrained
indices = read === :whole ? collect(eachindex(u)) : read
```

Prepare `q` and `footprint` once per selected observation (or observations
sharing a cell position), then vary `i`. This needs no AD preparation, cell
key column, or plan per cell. For a single call,
`ReactiveKernels.partition_footprint(q.kernel, u, i).unconstrained` prepares
and caches the footprint kernel. Positions are sorted, unique linear indices
into `u`; `:whole` means the whole packed vector. This is a structural read
set, including both arms of a retained lazy branch and reads multiplied by
zero, not a numerical nonzero-gradient test. To form a sparse Hessian pattern,
union the cliques of the individual cell supports (restricted to the desired
coordinates); a clique of their union introduces cross-cell pairs.

The set can be conservative: array captures read inside a retained conditional
can contribute their entire packed block, even with `locality = :error`.
That setting checks values requested at rows; it does not guarantee a minimal
footprint for every capture. Inspect the supports of the actual model before
assuming a fixed per-cell width.
"""
function prepare_cell_query(built, plan::StructuralPlan, observations;
        on_error = nothing, locality = nothing, partitioned = (), u0 = nothing)
    isbound(plan) || throw(ContractValidationError(
        "[query] prepare_cell_query requires a bound plan (bind_data first)"))
    _check_built_program(built, plan)
    names = observations isa Symbol ? (observations,) : Tuple(Symbol.(observations))
    isempty(names) && throw(ContractValidationError(
        "[query] prepare_cell_query needs at least one observation name"))
    _refuse_absent_cells(plan, names)
    spec, want = _cell_query_spec(built, plan, names; locality)
    have = (_query_have(plan)..., _CELL_PORT)
    rule = _query_partition_rule(plan, built.layout, partitioned, nothing)
    parts = _query_shared_parts(spec, plan, have, want, rule, (_CELL_PORT,), on_error)
    exemplar = _query_exemplar(built.layout, u0)
    prefix = if parts === nothing
        nothing
    else
        cache = _prepare_value_prefix(parts, built.layout, last(only(rule)), exemplar, on_error)
        residual = Base.invokelatest(prepare, parts.residual;
            exemplars = (exemplar, 1, cache.values...), on_error)
        _value_query_prefix(cache, residual, parts.residual,
            (exemplar, 1, cache.values...), on_error)
    end
    parts === nothing || ((spec, want) = _cell_query_spec(built, plan, names; locality = :ignore))
    kernel = _absent_refusal(plan, spec, have, want) do
        Base.invokelatest(_prepare_cached, plan, spec; have, want, bound = _query_bound(plan), on_error,
            exemplars = (exemplar, 1))
    end
    specializations = prefix === nothing ?
        (; kernels = Dict{DataType,Any}(), on_error) : nothing
    shared = isempty(rule) ? nothing : _cell_shared(parts, last(only(rule)), exemplar,
        prefix === nothing ? nothing : prefix.cache)
    return CellQuery(kernel, built.layout, names, prefix, specializations, shared)
end

const _CELL_PORT = :_ppl_cell

# Each observation's cell plate (its per-index totals when the loop observes
# one array per index), one `plate_cell` value per plate at the shared position
# port, and their sum when there are several. The plates are named in the
# program the build recorded, which `_check_built_program` proved `plan`
# generates; only a graph made outside `build_kernel` has the program
# generated again.
function _cell_query_spec(built, plan::StructuralPlan, names; locality = nothing)
    locality in (nothing, :error, :warn, :ignore) || throw(ContractValidationError(
        "[query] locality must be :error, :warn, :ignore or nothing; got $(repr(locality))"))
    def = hasproperty(built, :program) ? built.program : kernel_expr(plan, built.layout)
    body = def.args[2].args
    statement = findfirst(st -> Meta.isexpr(st, :(=), 2) && st.args[1] === :pointwise, body)
    pointwise = Dict{Symbol,Any}()
    if statement !== nothing && Meta.isexpr(body[statement].args[2], :tuple)
        for field in body[statement].args[2].args
            pointwise[field.args[1]] = field.args[2]
        end
    end
    assigned = Set{Symbol}()
    for st in body
        Meta.isexpr(st, :(=), 2) || continue
        lhs = st.args[1]
        Meta.isexpr(lhs, :(::), 2) && (lhs = lhs.args[1])
        lhs isa Symbol && push!(assigned, lhs)
    end
    spec = built.spec
    cells = Symbol[]
    for name in names
        haskey(pointwise, name) || throw(ContractValidationError(
            "[query] `$name` is not an observation of this model; its " *
            "observations are $(join(sort!(collect(keys(pointwise))), ", "))"))
        node = pointwise[name]
        node isa Symbol || throw(ContractValidationError(
            "[query] observation `$name` has no pointwise plate, so it has no cells"))
        totals = Symbol(node, :_totals)
        node = totals in assigned ? totals : node
        cell = Symbol(:_ppl_cell_, node)
        spec = try
            plate_cell(spec, node; index = _CELL_PORT, name = cell, locality)
        catch err
            err isa ArgumentError || rethrow()
            throw(ContractValidationError("[query] observation `$name` has no " *
                "pointwise plate, so it has no cells ($(err.msg))"))
        end
        push!(cells, cell)
    end
    length(cells) == 1 && return spec, only(cells)
    graph = kernel_graph(spec)
    total = value!(graph, :_ppl_cell_total, Float64)
    add!(graph, Tuple(spec[c] for c in cells) => total, +)
    return spec, total
end

"""
    CellSampler

A [`CellQuery`](@ref) with its `prepare_ad` gradient over the packed
unconstrained vector. Construct with [`prepare_cell_sampler`](@ref); use
[`cell_value_and_gradient!`](@ref). Not thread-safe: prepare one per thread.
"""
mutable struct CellSampler{K,P,L}
    const kernel::K
    const ad::P
    const layout::L
    const observations::Tuple{Vararg{Symbol}}
    value::Float64
end
CellSampler(kernel, ad, layout, observations) =
    CellSampler(kernel, ad, layout, observations, NaN)

"""
    prepare_cell_sampler(built, plan, observations, u0; backend, on_error = nothing, locality = nothing) -> CellSampler

[`prepare_cell_query`](@ref) plus a `prepare_ad` gradient with
`active = :unconstrained`; the position stays a constant (an `Int`). `u0` is a
length-consistent exemplar and `backend` a DifferentiationInterface backend,
as for [`prepare_sampler`](@ref). `locality` is passed to
[`prepare_cell_query`](@ref).
"""
function prepare_cell_sampler(built, plan::StructuralPlan, observations,
        u0::AbstractVector{<:Real}; backend, on_error = nothing, locality = nothing)
    layout = built.layout::LayoutTable
    length(u0) == layout.total || throw(ContractValidationError(
        "[query] exemplar length $(length(u0)) ≠ layout total $(layout.total)"))
    q = prepare_cell_query(built, plan, observations; on_error, locality)
    ad = Base.invokelatest(prepare_ad, q.kernel, backend, Vector{Float64}(u0), 1;
        active = :unconstrained)
    return CellSampler(q.kernel, ad, layout, q.observations)
end

"""
    cell_value_and_gradient!(s::CellSampler, g, u, i)

Cell `i`'s log density and its gradient with respect to the packed
unconstrained vector, written into `g`. Returns `(value, gradient)`.
"""
cell_value_and_gradient!(s::CellSampler, g::AbstractVector, u::Vector{Float64}, cell::Int) =
    (_latest_value(_cell_gradient!, s, g, u, cell), g)
cell_value_and_gradient!(s::CellSampler, g::AbstractVector, u::AbstractVector, cell::Int) =
    cell_value_and_gradient!(s, g, Vector{Float64}(u), cell)
_cell_gradient!(s, g, args::Vararg{Any,N}) where {N} =
    first(ad_value_and_gradient!(s.ad, g, args...))

# A packed point whose selected coordinates come from a compact active vector.
# `slots[j]` is packed coordinate `j`'s position in `active`, or 0 when `j`
# reads `base`. The packed base and the slot map cross the AD boundary as
# constants; the graph still contains every model calculation after this
# selection. Mutable cell samplers use dense maps: one slot load and a branch,
# where a binary search over the selection cost O(log selection) per read (snag
# `compact-cell-cal-58569833`). Native Reverse multiplied that search's cost
# inside gathers far beyond its primal cost; that is Enzyme snag
# `reverse-over-a-g-a9ee15bb`). Fixed factor queries instead use support-sized
# sparse maps; their native Reverse cost is covered by the preparation probe.
struct _CompactCellPoint{B<:AbstractVector{Float64},A<:AbstractVector{Float64},S<:AbstractVector{Int32}} <: AbstractVector{Float64}
    base::B
    active::A
    slots::S
end

Base.size(point::_CompactCellPoint) = size(point.base)
Base.IndexStyle(::Type{<:_CompactCellPoint}) = IndexLinear()
@inline function Base.getindex(point::_CompactCellPoint, index::Int)
    @boundscheck checkbounds(point.base, index)
    slot = @inbounds point.slots[index]
    iszero(slot) ? @inbounds(point.base[index]) : @inbounds(point.active[slot])
end

@inline _compact_cell_point(base, active, slots) =
    _CompactCellPoint(base, active, slots)
# The overlay is a lazy view of its base and active vectors: a cell or
# partition reads it at rows as it reads a packed HAVE vector (RK's locality
# analysis, `reactivekernels-use` §4i).
ReactiveKernels._cell_lazy_view(::typeof(_compact_cell_point)) = true

# The slot map of a selection over `total` packed coordinates.
function _compact_cell_slots(total::Int, indices)
    slots = zeros(Int32, total)
    for (slot, index) in enumerate(indices)
        slots[index] = slot
    end
    slots
end

# Fixed factor queries own only their structural support. Mutable cell samplers
# keep the dense map above, which they update in place when the selection moves.
_fixed_cell_slots(total::Int, indices) =
    sparsevec(indices, Int32.(eachindex(indices)), total)

# A parameter block read from the overlay is not strided, so a product of it
# would take LinearAlgebra's generic `mul!`; RK's native product computes it
# with a typed loop instead (`_native_mul`).
ReactiveKernels._native_mul_operand(::_CompactCellPoint) = true

# The overlay's rows are its active vector at the rows' slots: a partition
# reads only its footprint, so every row it reads has a slot, and a plan with
# the label's slot map computes those slots once per label (RK
# `_cell_lazy_rows`); the body then reads a plain view of the active vector,
# no slot load and branch per element.
function ReactiveKernels._cell_lazy_rows(::typeof(_compact_cell_point), inputs, rows, bind,
                                         constant)
    # Only where the slot map is constant per label (a partition, which reads
    # its footprint only); a cell sampler's selection varies per call and may
    # leave rows to the base.
    constant(inputs[3]) || return nothing
    slots = bind(:cell_positions,
                 Expr(:call, GlobalRef(@__MODULE__, :_overlay_slots), inputs[3], rows))
    Expr(:call, GlobalRef(ReactiveKernels, :_cell_source_rows), inputs[2], slots)
end
function _overlay_slots(slots::AbstractVector{Int32}, rows::AbstractVector{<:Integer})
    out = Vector{Int}(undef, length(rows))
    for (k, row) in enumerate(rows)
        slot = slots[row]
        iszero(slot) && _overlay_outside(rows, slots)
        out[k] = Int(slot)
    end
    out
end
function _overlay_slots(slots::AbstractVector{Int32}, row::Integer)
    slot = slots[row]
    iszero(slot) && _overlay_outside(row:row, slots)
    Int(slot)
end
@noinline function _overlay_outside(rows, slots)
    outside = Int[row for row in rows if iszero(slots[row])]
    throw(ContractValidationError("[query] a partition reads $(length(outside)) packed " *
        "coordinate(s) outside its footprint: a block's rows $(length(rows)) long over " *
        "$(extrema(rows)), outside from $(first(outside)) to $(last(outside))"))
end

function _check_compact_cell_lengths(base, active, indices, total)
    length(base) == total || throw(ContractValidationError(
        "[query] packed base length $(length(base)) ≠ layout total $total"))
    length(active) == length(indices) || throw(ContractValidationError(
        "[query] active length $(length(active)) ≠ selection length $(length(indices))"))
    nothing
end

function _check_compact_cell_indices(base, active, indices, total)
    _check_compact_cell_lengths(base, active, indices, total)
    previous = 0
    for index in indices
        previous < index <= total || throw(ContractValidationError(
            "[query] compact cell indices must be unique, sorted, and within 1:$total"))
        previous = index
    end
    nothing
end

"""
    CompactCellSampler

A prepared cell gradient over selected packed coordinates. The packed base point
is read-only, and the derivative is returned only in the order of `indices`.
The sampler keeps the current selection as a map over every packed coordinate,
rewritten when a call selects different positions. Prepare one sampler per
concurrent caller.
"""
mutable struct CompactCellSampler{P,L,O}
    const ad::P
    const layout::L
    const observations::O
    # The selection `slots` maps (`_compact_cell_slots`), compared with each
    # call's `indices`.
    const selected::Vector{Int}
    const slots::Vector{Int32}
    value::Float64
end
CompactCellSampler(ad, layout, observations, selected, slots) =
    CompactCellSampler(ad, layout, observations, selected, slots, NaN)

# Point the slot map of a `CompactCellSampler` or `CompactCellAD` at `indices`.
# Validating and rewriting cost O(selection), so they run only when the
# selection differs from the previous call's; a run of calls over one selection
# pays one comparison each.
function _select_compact_cells!(sampler, base, active, indices)
    _check_compact_cell_lengths(base, active, indices, sampler.layout.total)
    selected = sampler.selected
    _same_selection(selected, indices) && return nothing
    _check_compact_cell_indices(base, active, indices, sampler.layout.total)
    slots = sampler.slots
    for index in selected
        slots[index] = 0
    end
    copyto!(resize!(selected, length(indices)), indices)
    for (slot, index) in enumerate(selected)
        slots[index] = slot
    end
    nothing
end

function _same_selection(selected::Vector{Int}, indices::Vector{Int})
    length(selected) == length(indices) || return false
    different = false
    @inbounds @simd for k in eachindex(selected, indices)
        different |= selected[k] != indices[k]
    end
    !different
end
_same_selection(selected::Vector{Int}, indices) = selected == indices

"""
    prepare_compact_cell_sampler(built, plan, observations, base0, indices0; backend, on_error = nothing, locality = nothing)

Prepare a cell value and gradient over the sorted, unique packed positions
`indices0`. `locality` is as for [`prepare_cell_query`](@ref). At evaluation, the selected positions may change, provided the
active vector and selection retain the exemplar's type and length. The packed
base is passed read-only; selected coordinates are supplied separately. This
keeps the active AD point and gradient proportional to the selected cell's
coordinates rather than the full layout. With native `AutoEnzyme` Reverse, use
`Enzyme.set_runtime_activity(Enzyme.Reverse)`: packed reads can select either an
active coordinate or a constant base coordinate. The sampler holds one `Int32`
per packed coordinate, mapping each to its position in the selection.
"""
function prepare_compact_cell_sampler(built, plan::StructuralPlan, observations,
        base0::AbstractVector{Float64}, indices0::AbstractVector{Int};
        backend, on_error = nothing, locality = nothing)
    c = _prepare_compact_cell(:prepare_compact_cell_sampler, built, plan, observations,
        base0, indices0; on_error, locality)
    ad = Base.invokelatest(prepare_ad, c.kernel, backend, base0, c.active0, c.slots, 1;
        active = :_ppl_cell_active)
    CompactCellSampler(ad, built.layout, c.names, c.selected, c.slots)
end

# The cell kernel over the compact overlay that the compact cell sampler and
# the compact cell derivative operators differentiate: its HAVE boundary is
# `(base, active, slots, cell)`, with `active` the selected coordinates.
function _prepare_compact_cell(caller::Symbol, built, plan::StructuralPlan, observations,
        base0::AbstractVector{Float64}, indices0::AbstractVector{Int};
        on_error = nothing, locality = nothing)
    layout = built.layout::LayoutTable
    _check_compact_cell_indices(base0, indices0, indices0, layout.total)
    active0 = base0[indices0]
    selected = Vector{Int}(indices0)
    slots = _compact_cell_slots(layout.total, selected)
    isbound(plan) || throw(ContractValidationError(
        "[query] $caller requires a bound plan (bind_data first)"))
    _check_built_program(built, plan)
    names = observations isa Symbol ? (observations,) : Tuple(Symbol.(observations))
    isempty(names) && throw(ContractValidationError(
        "[query] $caller needs at least one observation name"))
    _refuse_absent_cells(plan, names)
    spec, want = _cell_query_spec(built, plan, names; locality)
    have = _compact_overlay!(spec, plan)
    kernel = _absent_refusal(plan, spec, have, want) do
        Base.invokelatest(_prepare_cached, plan, spec; have, want, bound = _query_bound(plan), on_error,
                          exemplars = (base0, active0, slots, 1))
    end
    (; kernel, names, active0, selected, slots)
end

# Overlay a compact active vector on a read-only packed base in `spec`: the
# packed point becomes `_compact_cell_point(base, active, slots)`, and the
# HAVE boundary starts with the three overlay ports. Returns that boundary.
function _compact_overlay!(spec, plan::StructuralPlan)
    for (name, type) in ((:_ppl_cell_base, AbstractVector{Float64}),
            (:_ppl_cell_active, AbstractVector{Float64}),
            (:_ppl_cell_slots, AbstractVector{Int32}))
        haskey(spec.ports, name) && throw(ContractValidationError(
            "[query] compact cell port `$name` collides with a model port"))
        spec.ports[name] = value!(spec.graph, name, type)
        push!(spec.port_order, name)
    end
    add!(spec.graph, (spec[:_ppl_cell_base], spec[:_ppl_cell_active],
        spec[:_ppl_cell_slots]) => spec[:unconstrained], _compact_cell_point)
    filter!(!=(:unconstrained), spec.have_names)
    prepend!(spec.have_names, [:_ppl_cell_base, :_ppl_cell_active, :_ppl_cell_slots])
    filter!(n -> n ∉ _absent_inputs(plan), spec.have_names)
    (spec.have_names..., _frozen_graph_names(plan)...)
end

"""
    cell_value_and_gradient!(sampler::CompactCellSampler, gradient, base, active, indices, cell)

Return one cell's density and write only the selected-coordinate gradient.
`base` and `indices` are not mutated. The active values overlay `base` at the
sorted positions in `indices`. Consecutive calls with equal `indices` check the
selection once.
"""
function cell_value_and_gradient!(sampler::CompactCellSampler, gradient::AbstractVector,
        base::AbstractVector{Float64}, active::AbstractVector{Float64},
        indices::AbstractVector{Int}, cell::Int)
    length(gradient) == length(active) || throw(ContractValidationError(
        "[query] compact gradient length $(length(gradient)) ≠ active length $(length(active))"))
    _select_compact_cells!(sampler, base, active, indices)
    return _latest_value(_cell_gradient!, sampler, gradient, base, active,
        sampler.slots, cell), gradient
end

"""
    CompactCellAD

A prepared ReactiveKernels derivative operator over one cell's log density,
active over selected packed coordinates: the boundary of
[`prepare_compact_cell_sampler`](@ref). Construct one with
[`prepare_compact_cell_ad`](@ref). The ReactiveKernels derivative calls accept
it in place of the prepared operator, with `base, active, indices, cell` (as
[`cell_value_and_gradient!`](@ref) takes them) in place of the operator's
point: `ad_gradient_and_hvp(q, tangents, base, active, indices, cell)`,
`ad_gradient_hvp_and_third(q, tangents, base, active, indices, cell)` and the
other verbs [`QueryAD`](@ref) takes. Directions, gradients and results are over
the selection, in `indices` order. `base` and `indices` are read only. Each
call carries the `Base.invokelatest` barrier. Not thread-safe (one per caller).
"""
mutable struct CompactCellAD{P,L,O,S}
    const prepared::P
    const layout::L
    const observations::O
    # The selection `slots` maps, as for `CompactCellSampler`.
    const selected::Vector{Int}
    const slots::Vector{Int32}
    const shared::S
end

"""
    prepare_compact_cell_ad(prepare_operator, built, plan, observations, base0, indices0,
                            backend, exemplars...; on_error = nothing, locality = nothing)
        -> CompactCellAD

Prepare a ReactiveKernels derivative operator over one cell of the
observations named in `observations`, differentiating with respect to the
packed coordinates at the sorted, unique positions `indices0` over the
read-only packed base `base0`, as [`prepare_compact_cell_sampler`](@ref) does
for the gradient. `prepare_operator` is the ReactiveKernels preparation to
apply (`prepare_ad`, `prepare_ad_pullback`, `prepare_ad_pushforward`,
`prepare_ad_hvp` or `prepare_ad_third`), and `exemplars` are its arguments
between the backend and the point: the directions for an HVP or a third-order
derivative, each of the selection's length.

```julia
so = SecondOrder(AutoEnzyme(; mode = Enzyme.Reverse),   # reverse over forward
                 AutoEnzyme(; mode = Enzyme.Forward))
directions = ntuple(k -> onehot(k, m), m)                # m = length(indices0)
h = prepare_compact_cell_ad(prepare_ad_hvp, built, plan, :y, u, indices0, so, directions)
gradient, columns = ad_gradient_and_hvp(h, directions, u, active, indices, i)

t = prepare_compact_cell_ad(prepare_ad_third, built, plan, :y, u, indices0,
                            SecondOrder(so, AutoEnzyme(; mode = Enzyme.Forward)), (v,))
gradient, (hv,), (tv,) = ad_gradient_hvp_and_third(t, (v,), u, active, indices, i)
```

With the `m` unit directions, the HVP columns are the cell's `m × m` Hessian
block over its selection, from one call. `tv = ∇(vᵀ H v)` is the third-order
contraction `D³ℓ[v, v, ·]` over the selection. At evaluation the selection may
change, as for the compact sampler: the active vector and the selection keep
the exemplar's type and length, and so do the directions. `locality` is as for
[`prepare_cell_query`](@ref).

With `partitioned = :z` (or the partition declaration accepted by cell
queries), cache opaque whole-array computations that depend only on the
remaining, shared coordinates. The compact selection must then contain only
partitioned coordinates: these derivatives condition on the shared point.
The cache refreshes when shared coordinates change. For exact mixed shared
derivatives, prepare a [`prepare_cell_accumulator`](@ref) and accumulate a
pass with [`cell_pullback!`](@ref), which applies the shared prefix's pullback
once after collecting all cell contributions.
Inside that callback, pass its active `CellPullback` in place of `base` to
reuse the fixed point without checking all shared coordinates per cell.
"""
function prepare_compact_cell_ad(prepare_operator, built, plan::StructuralPlan, observations,
        base0::AbstractVector{Float64}, indices0::AbstractVector{Int}, backend, exemplars...;
        on_error = nothing, locality = nothing, partitioned = ())
    isempty(_query_partition_rule(plan, built.layout, partitioned, nothing)) ||
        return _prepare_shared_compact_ad(prepare_operator, built, plan, observations,
            base0, indices0, backend, exemplars...; on_error, locality, partitioned)
    c = _prepare_compact_cell(:prepare_compact_cell_ad, built, plan, observations,
        base0, indices0; on_error, locality)
    prepared = Base.invokelatest(prepare_operator, c.kernel, backend, exemplars...,
        base0, c.active0, c.slots, 1; active = :_ppl_cell_active)
    CompactCellAD(prepared, built.layout, c.names, c.selected, c.slots, nothing)
end

for verb in (:ad_gradient, :ad_value_and_gradient, :ad_value_and_gradient!,
        :ad_pullback, :ad_value_and_pullback, :ad_value_and_pullback!,
        :ad_pushforward, :ad_value_and_pushforward,
        :ad_hvp, :ad_hvp!, :ad_gradient_and_hvp, :ad_gradient_and_hvp!,
        :ad_third, :ad_third!, :ad_gradient_hvp_and_third,
        :ad_gradient_hvp_and_third!)
    @eval ReactiveKernels.$verb(q::CompactCellAD, args...) =
        Base.invokelatest(_compact_cell_ad_call, ReactiveKernels.$verb, q, args...)
end
# The verb's trailing `base, active, indices, cell` become the operator's
# point `base, active, slots, cell`; its leading arguments (outputs, tangents,
# seeds) pass through.
function _compact_cell_ad_call(verb::F, q::CompactCellAD, args::Vararg{Any,N}) where {F,N}
    N >= 4 || throw(ContractValidationError(
        "[query] a compact cell derivative takes `base, active, indices, cell` last"))
    base_arg, active, indices, cell = args[N - 3], args[N - 2], args[N - 1], args[N]
    base = _cell_ad_base(q.shared, base_arg)
    base isa AbstractVector{Float64} && active isa AbstractVector{Float64} &&
        indices isa AbstractVector{Int} && cell isa Int || throw(ContractValidationError(
        "[query] a compact cell derivative takes `base::AbstractVector{Float64}, " *
        "active::AbstractVector{Float64}, indices::AbstractVector{Int}, cell::Int` last; got " *
        join(summary.((base, active, indices, cell)), ", ")))
    _select_compact_cells!(q, base, active, indices)
    _check_cell_local_selection(q.shared, indices)
    shared_values = _cell_ad_values(q.shared, base_arg)
    leading = ntuple(k -> args[k], Val(N - 4))
    return verb(q.prepared, leading..., base, active, q.slots, cell, shared_values...)
end

function _stack_restored_draws(values)
    value = first(values)
    if value isa NamedTuple
        names = Tuple(keys(value))
        columns = map(names) do name
            _stack_restored_draws([draw[name] for draw in values])
        end
        return NamedTuple{names}(Tuple(columns))
    elseif value isa AbstractVector && !(eltype(value) <: Number)
        # One array per index (a dotted cell latent): stacked per index.
        return [_stack_restored_draws([draw[k] for draw in values]) for k in eachindex(value)]
    elseif value isa AbstractVector
        return hcat([Vector{Float64}(draw) for draw in values]...)
    elseif value isa AbstractArray
        return [Array{Float64,ndims(value)}(draw) for draw in values]
    end
    return Float64[draw for draw in values]
end

function _empty_restored_draws(value::NamedTuple)
    return map(_empty_restored_draws, value)
end
_empty_restored_draws(value::AbstractVector) = eltype(value) <: Number ?
    zeros(Float64, length(value), 0) : map(_empty_restored_draws, value)
_empty_restored_draws(value::AbstractArray) = Array{Float64,ndims(value)}[]
_empty_restored_draws(::Real) = Float64[]

"""
    restore_draws(layout::LayoutTable, U::AbstractMatrix{<:Real}) -> NamedTuple

Restore constrained parameters from `U` (`layout.total` rows × draws
columns), with the same nested author paths as [`constrain`](@ref).
Scalar leaves become length-`draws` vectors, vector leaves become
`size × draws` matrices, and higher-rank array leaves become vectors of arrays.
Empty draws retain those keys and leaf shapes. Each column is constrained
independently through `constrain`, so transforms stay single-sourced.
"""
function restore_draws(layout::LayoutTable, U::AbstractMatrix{<:Real})
    size(U, 1) == layout.total || throw(ContractValidationError(
        "[query] draws rows $(size(U, 1)) ≠ layout total $(layout.total)"))
    n = size(U, 2)
    if n == 0
        # The host transform supplies keys/shapes without evaluating a
        # density. It also handles split coefficient blocks and derived
        # matrix draws exactly as the nonempty path does.
        return _empty_restored_draws(constrain(layout, zeros(layout.total)))
    end
    nts = map(1:n) do j
        constrain(layout, view(U, :, j))
    end
    return _stack_restored_draws(nts)
end

# ------------------------------------------------------------- partitions

"""
    PartitionQuery

The summed log density of one partition of the observations named at
preparation: all cells whose key equals a label, as one compiled pass
(RK `plate_partition`). Construct with [`prepare_partition`](@ref); call as
`q(u, label)`. Not thread-safe: prepare one per thread.
"""
mutable struct PartitionQuery{K,L,T,C}
    const kernel::K
    const layout::L
    const observations::Tuple{Vararg{Symbol}}
    const by::Dict{Symbol,Symbol}
    const labels::Vector{Any}
    # Each label's plan (`ReactiveKernels.partition_plan`), computed at
    # preparation; a label without one is planned on the spot, which refuses
    # a label no observation has.
    const plans::Dict{Any,T}
    const prefix::C
end
PartitionQuery(kernel, layout, observations, by, labels, plans) =
    PartitionQuery(kernel, layout, observations, by, labels, plans, nothing)

(q::PartitionQuery)(u::AbstractVector{Float64}, label) =
    Base.invokelatest(_partition_density, q, u, label)::Float64
(q::PartitionQuery)(u::AbstractVector{<:Real}, label) = q(Vector{Float64}(u), label)
function _partition_density(q::PartitionQuery, u, label)
    if q.prefix === nothing
        return q.kernel(u, label, get(() -> ReactiveKernels.partition_plan(q.kernel, u, label, nothing),
                                     q.plans, label))
    end
    length(u) == q.layout.total || throw(ContractValidationError(
        "[query] packed length $(length(u)) ≠ layout total $(q.layout.total)"))
    prefix = _query_prefix_kernel(q.prefix, u)
    cached = _refresh_partition_prefix!(q.prefix.cache, u, prefix)
    residual = _query_residual(q.prefix, (u, label, nothing, cached...))
    plan = get(() -> ReactiveKernels.partition_plan(residual, u, label, nothing, cached...),
               q.plans, label)
    residual(u, label, plan, cached...)
end
ReactiveKernels.partition_locality(q::PartitionQuery) =
    ReactiveKernels.partition_locality(q.prefix === nothing ? q.kernel : q.prefix.residual)

"""
    partition_labels(q) -> Vector

The distinct keys of the partitioned observations, in first-seen order over
the observations named at preparation: every label the query accepts.
"""
partition_labels(q::PartitionQuery) = copy(q.labels)

const _PARTITION_PORT = :_ppl_partition
# The plan port: the integer bookkeeping of a label (its rows, every pushed
# value's rows and positions), computed once per label by RK's
# `partition_plan` and read by the body instead of recomputed per call.
const _PLAN_PORT = :_ppl_partition_plan
# Every label's plan by the plan kernel prepared from the spec (RK
# `partition_plan_kernel`), and the type of those plans, which declares the
# plan port of the partition kernel to prepare.
function _partition_plans(planner, labels, args...)
    _partition_plans_each(label -> Base.invokelatest(planner, args..., label, nothing), labels)
end
function _partition_plans_each(plan_of, labels)
    plans = Dict{Any,Any}(label => plan_of(label) for label in labels)
    types = unique!(Any[typeof(plan) for plan in values(plans)])
    length(types) == 1 || throw(ContractValidationError(
        "[query] the partition plans have several types: $(join(types, ", "))"))
    # Preparation already requires one plan type across all labels. Preserve
    # it at lookup rather than boxing the plan again on every evaluation.
    plan_type = only(types)
    Dict{Any,plan_type}(plans), plan_type
end
const _PARTITION_TOTAL = :_ppl_partition_total

# The key column of each observation, spelled as RK's `plate_partition` reads
# `by`. Every column must be a supplied data column of the binding.
function _partition_keys(plan::StructuralPlan, names, by)
    keys_by_name = ReactiveKernels._partition_by_map(by, names; what = "observation",
        refuse = message -> ContractValidationError("[query] " * message *
            ". A key is a column name or, per observation, a vector of labels"))
    columns = _query_columns(plan)
    for name in names
        key = keys_by_name[name]
        key in columns || throw(ContractValidationError(
            "[query] the partition key `$key` of `$name` is not a supplied data column; " *
            "the columns are $(join(columns, ", "))"))
    end
    keys_by_name
end

# Keys `by` gives as their labels (`observation => labels`), one label per cell.
# A label need not be a model input, and no statement need read it, so it stays
# out of the program: each is a fresh input of this query alone, a port on a copy
# of the graph and a column of a copy of the binding, read from here on like any
# data column. Returns the copies and `by` with each such key named by its port;
# `built`, `plan` and `by` themselves when `by` gives no labels.
function _partition_label_keys(built, plan::StructuralPlan, by)
    gives_labels(entry) = entry isa Pair && first(entry) isa Union{Symbol,AbstractString} &&
        last(entry) isa AbstractVector
    entries = ReactiveKernels._partition_by_entries(by)
    any(gives_labels, entries) || return built, plan, by
    spec = ReactiveKernels._copy_cell_factor_spec(built.spec)
    columns = copy(plan.columns)
    labels_by_name = Dict{Symbol,Any}()
    named = Any[]
    for entry in entries
        if gives_labels(entry)
            name = Symbol(first(entry))
            port = Symbol(:_ppl_partition_labels_, name)
            if !haskey(labels_by_name, name)
                labels = collect(last(entry))
                labels_by_name[name] = labels
                spec.ports[port] = value!(spec.graph, port, typeof(labels))
                push!(spec.port_order, port)
                push!(spec.have_names, port)
                columns[port] = labels
            elseif !isequal(labels_by_name[name], last(entry))
                throw(ContractValidationError(
                    "[query] `by` gives observation :$name two different vectors of labels"))
            end
            push!(named, name => port)
        else
            push!(named, entry isa Pair ? Pair(first(entry), last(entry)) : entry)
        end
    end
    merge(built, (; spec)), _with(plan; columns), named
end

# Each observation's cell plate partitioned by its key, and their sum: the
# plates of `_cell_query_spec`, through RK's `plate_partition`.
function _partition_query_spec(built, plan::StructuralPlan, names, keys_by_name;
                               locality, partitioned = (), plan_type = Any,
                               plan_constant = ())
    locality in (nothing, :error, :warn, :ignore) || throw(ContractValidationError(
        "[query] locality must be :error, :warn, :ignore or nothing; got $(repr(locality))"))
    def = hasproperty(built, :program) ? built.program : kernel_expr(plan, built.layout)
    body = def.args[2].args
    statement = findfirst(st -> Meta.isexpr(st, :(=), 2) && st.args[1] === :pointwise, body)
    pointwise = Dict{Symbol,Any}()
    if statement !== nothing && Meta.isexpr(body[statement].args[2], :tuple)
        for field in body[statement].args[2].args
            pointwise[field.args[1]] = field.args[2]
        end
    end
    assigned = Set{Symbol}()
    for st in body
        Meta.isexpr(st, :(=), 2) || continue
        lhs = st.args[1]
        Meta.isexpr(lhs, :(::), 2) && (lhs = lhs.args[1])
        lhs isa Symbol && push!(assigned, lhs)
    end
    plates = Symbol[]
    keys_by_plate = Dict{Symbol,Symbol}()
    for name in names
        haskey(pointwise, name) || throw(ContractValidationError(
            "[query] `$name` is not an observation of this model; its " *
            "observations are $(join(sort!(collect(keys(pointwise))), ", "))"))
        node = pointwise[name]
        node isa Symbol || throw(ContractValidationError(
            "[query] observation `$name` has no pointwise plate, so it has no cells"))
        totals = Symbol(node, :_totals)
        node = totals in assigned ? totals : node
        push!(plates, node)
        keys_by_plate[node] = keys_by_name[name]
    end
    spec = try
        plate_partition(built.spec, plates; by = keys_by_plate, index = _PARTITION_PORT,
                        name = _PARTITION_TOTAL, locality, partitioned,
                        plan = plan_type === Any ? _PLAN_PORT : _PLAN_PORT => plan_type,
                        plan_constant)
    catch err
        err isa ArgumentError || rethrow()
        throw(ContractValidationError("[query] the observations have no partitionable " *
            "pointwise plate ($(err.msg))"))
    end
    return spec, _PARTITION_TOTAL
end

# The distinct labels of the named observations' key columns, in first-seen
# order; each key column must hold one entry per cell of its plate, which the
# prepared kernel checks at evaluation.
function _partition_labels(plan::StructuralPlan, names, keys_by_name)
    labels = Any[]
    seen = Set{Any}()
    for name in names
        for label in vec(plan.columns[keys_by_name[name]])
            label in seen || (push!(seen, label); push!(labels, label))
        end
    end
    labels
end

"""
    prepare_partition(built, plan, observations; by, locality = :error, on_error = nothing,
                      partitioned = (), u0 = nothing) -> PartitionQuery

Prepare the summed log density of ONE partition of the observations named in
`observations` (a name or a collection of names): all cells whose key equals
a label, such as one subject's observations, evaluated at an unconstrained
point `u`: `q(u, label)`. `by` names the key: a supplied data column with one
entry per cell of each observation's plate (one row per entry of an
elementwise observation, one per iteration of an `@plate for` loop), either
one column name for every observation or, per observation, a named tuple, a
dictionary, one `observation => column` pair or a collection of such pairs.
Per observation, `by` may instead give the labels themselves,
`observation => labels`: a vector with one label per cell, which need not be
data of the model and which no statement need read. A key of another length is
refused. [`partition_labels`](@ref) lists the labels.

A partition runs as one compiled pass over its cells (RK `plate_partition`):
each plate loops once over the positions holding the label, and every value
the partition's cells read at their rows is computed once per partition at
the union of those rows, so

    sum(q(u, label) for label in partition_labels(q)) == likelihood of the observations

and one partition costs its share of the whole likelihood. `locality` is as
for [`prepare_cell_query`](@ref), with the default `:error`: the query is
prepared only when every value its cells read at rows is computed at those
rows; otherwise preparation throws a `ReactiveKernels.PartitionLocalityError`
naming each such value and its statement.

With `partitioned`, use the same local/shared coordinate declaration and
separability check as [`prepare_partition_sampler`](@ref). Shared intermediates
are cached across labels and local changes, and the strict check applies to
the remaining per-partition program. Row-capable calculations stay in that
program. `u0` is an optional packed exemplar (zeros by default), as for
[`prepare_cell_query`](@ref). `partition_locality(q)` reports the executed
residual. The original `q.kernel` continues to supply packed-coordinate
footprints including all dependencies through the prefix; call `q(u, label)`
for cached evaluation. Comparing shared coordinates costs O(shared dimension)
per call; refreshing the prefix adds its graph's work once per changed shared
point. Cache storage is query-owned; prepare one query per concurrent caller.
Cached calls preserve `AbstractVector{Float64}` storage and prepare each
argument-type specialization once, without copying the packed point.

For a value-only query's packed-coordinate footprint, call
`ReactiveKernels.partition_footprint(q.kernel, u, label, nothing).unconstrained`.
The final `nothing` supplies the plan port; the footprint kernel computes its
own row bookkeeping. A result of `:whole` means all positions of `u`.
For repeated calls, prepare `ReactiveKernels.partition_footprint_kernel(q.kernel)`
once and call it with `(u, label, nothing)` through `Base.invokelatest`.
This does not prepare AD. For individual cell supports, use the
[`prepare_cell_query`](@ref) footprint recipe: preparing a partition with one
label per cell would retain a partition plan for every cell.
"""
function prepare_partition(built, plan::StructuralPlan, observations; by,
        locality = :error, on_error = nothing, partitioned = (), u0 = nothing)
    isbound(plan) || throw(ContractValidationError(
        "[query] prepare_partition requires a bound plan (bind_data first)"))
    _check_built_program(built, plan)
    names = observations isa Symbol ? (observations,) : Tuple(Symbol.(observations))
    isempty(names) && throw(ContractValidationError(
        "[query] prepare_partition needs at least one observation name"))
    _refuse_absent_cells(plan, names)
    _prepare_partition(_partition_label_keys(built, plan, by)..., names;
        locality, on_error, partitioned, u0)
end

function _prepare_partition(built, plan::StructuralPlan, by, names;
        locality, on_error, partitioned, u0)
    keys_by_name = _partition_keys(plan, names, by)
    labels = _partition_labels(plan, names, keys_by_name)
    have = (_query_have(plan)..., _PARTITION_PORT, _PLAN_PORT)
    exemplar = _query_exemplar(built.layout, u0)
    rule = _query_partition_rule(plan, built.layout, partitioned, keys_by_name)
    spec, want = _partition_query_spec(built, plan, names, keys_by_name; locality, partitioned = rule)
    parts = _query_shared_parts(spec, plan, have, want, rule,
        (_PARTITION_PORT, _PLAN_PORT), on_error)
    dense_locality = parts === nothing ? locality : :ignore
    if parts !== nothing
        spec, want = _partition_query_spec(built, plan, names, keys_by_name;
            locality = dense_locality, partitioned = rule)
    end
    planner = _absent_refusal(plan, spec, have, want) do
        Base.invokelatest(_prepare_cached_partition, plan, spec; have, want,
                          bound = _query_bound(plan), on_error)
    end
    plans, plan_type = _partition_plans(planner, labels, exemplar)
    spec, want = _partition_query_spec(built, plan, names, keys_by_name;
        locality = dense_locality, partitioned = rule, plan_type)
    kernel = _absent_refusal(plan, spec, have, want) do
        Base.invokelatest(_prepare_cached, plan, spec; have, want, bound = _query_bound(plan), on_error)
    end
    if !isempty(rule)
        for label in labels
            Base.invokelatest(ReactiveKernels.partition_separability, kernel, exemplar, label, nothing)
        end
    end
    prefix = if parts === nothing
        nothing
    else
        cache = _prepare_value_prefix(parts, built.layout, last(only(rule)), exemplar, on_error)
        residual = Base.invokelatest(prepare, parts.residual;
            exemplars = (exemplar, first(labels), nothing, cache.values...), on_error)
        result = _value_query_prefix(cache, residual, parts.residual,
            (exemplar, first(labels), nothing, cache.values...), on_error)
        plans, _ = _partition_plans_each(labels) do label
            label_kernel = _query_residual(result, (exemplar, label, nothing, cache.values...))
            Base.invokelatest(ReactiveKernels.partition_plan, label_kernel,
                exemplar, label, nothing, cache.values...)
        end
        result
    end
    return PartitionQuery(kernel, built.layout, names, keys_by_name, labels, plans, prefix)
end

"""
    PartitionSampler

A [`PartitionQuery`](@ref) with its `prepare_ad` gradient: compact, over each
partition's footprint (the packed coordinates it reads), or dense over the
whole packed vector. Construct with [`prepare_partition_sampler`](@ref); use
[`partition_value_and_gradient!`](@ref) and [`partition_footprint`](@ref).
Not thread-safe: prepare one per thread.
"""
mutable struct PartitionSampler{K,P,L,C,T}
    const kernel::K
    const ad::P
    const layout::L
    const observations::Tuple{Vararg{Symbol}}
    const by::Dict{Symbol,Symbol}
    const labels::Vector{Any}
    const exemplar::Vector{Float64}
    # Each label's exact footprint (sorted packed positions). Compact only:
    # its padded selection of one width, the selection in force, its slot
    # map, and the active and gradient buffers of that width.
    const footprints::Dict{Any,Vector{Int}}
    const local_footprints::Union{Nothing,Dict{Any,Vector{Int}}}
    const selections::Dict{Any,Vector{Int}}
    const selected::Vector{Int}
    const slots::Vector{Int32}
    const active::Vector{Float64}
    const gradient::Vector{Float64}
    # Each label's plan of the differentiated kernel (`partition_plan`).
    const plans::Dict{Any,T}
    const prefix::C
    value::Float64
end
_compact(s::PartitionSampler) = !isempty(s.slots)
partition_labels(s::PartitionSampler) = copy(s.labels)

# A label is constant to AD, but its concrete type is part of a backend's
# preparation signature. Mixed key columns need one operator per label type.
# The common homogeneous case keeps its concrete operator and direct call.
struct _PartitionADByType
    operators::Dict{DataType,Any}
end
function _partition_prepare_ad(prepare_label, labels)
    first_type = typeof(first(labels))
    all(label -> typeof(label) === first_type, labels) && return prepare_label(first(labels))
    operators = Dict{DataType,Any}()
    for label in labels
        get!(() -> prepare_label(label), operators, typeof(label))
    end
    _PartitionADByType(operators)
end
_partition_ad(ad, label) = ad
_partition_ad(ad::_PartitionADByType, label) = ad.operators[typeof(label)]
_partition_gradient!(s, g, label, args::Vararg{Any,N}) where {N} =
    first(ad_value_and_gradient!(_partition_ad(s.ad, label), g, args...))

function _partition_gradient_footprint(s::PartitionSampler, label, shared::Bool)
    footprints = shared ? s.footprints : s.local_footprints
    footprints === nothing && throw(ContractValidationError(
        "[query] `shared = false` requires `partitioned` at sampler preparation"))
    get(footprints, label) do
        throw(ContractValidationError("[query] no partition has the label $(repr(label)); " *
            "the labels are $(join(repr.(s.labels), ", "))"))
    end
end

"""
    partition_footprint(s::PartitionSampler, label; shared = true) -> Vector{Int}

The packed positions partition `label` reads (sorted, unique): the support of
its gradient, computed at preparation from the bound index data by the row
analysis that evaluates the partition (a method of ReactiveKernels'
`partition_footprint`). A compact sampler's gradient is over exactly these,
in this order. With `shared = false`, return only the coordinates declared
per-partition by `partitioned`; this requires that declaration at preparation.
"""
ReactiveKernels.partition_footprint(s::PartitionSampler, label; shared::Bool = true) =
    copy(_partition_gradient_footprint(s, label, shared))

# The separability rule as RK takes it: the label of each packed coordinate
# of the named parameters, `missing` for every other coordinate. A parameter
# named bare is indexed by the key column itself (`z[levels(subject)]`), so
# its row `r` carries the `r`-th level; `parameter => labels` gives each
# level's label when the parameter's axis is finer than the key (a group
# within a subject).
function _partition_coordinate_labels(plan::StructuralPlan, layout::LayoutTable,
                                      partitioned, keys_by_name)
    labels = Vector{Any}(missing, layout.total)
    entries = partitioned isa Union{Symbol,Pair} ? (partitioned,) : partitioned
    for entry in entries
        name, level_labels = entry isa Pair ? (Symbol(first(entry)), last(entry)) :
            (Symbol(entry), nothing)
        position = findfirst(e -> e.name === name, layout.entries)
        position === nothing && throw(ContractValidationError(
            "[query] `partitioned` names `$name`, which is not a sampled parameter; the " *
            "parameters are $(join((e.name for e in layout.entries), ", "))"))
        e = layout.entries[position]
        if level_labels === nothing
            axis = _partition_parameter_axis(plan, name)
            axis === nothing && throw(ContractValidationError(
                "[query] `partitioned` names `$name`, whose first axis is not " *
                "`levels(<column>)`; give its levels' labels as `$name => labels`"))
            (keys_by_name === nothing || axis in values(keys_by_name)) || throw(ContractValidationError(
                "[query] `$name` is indexed by `levels($axis)`, not by the partition key " *
                "$(join(unique(values(keys_by_name)), ", ")); give its levels' labels as " *
                "`$name => labels`"))
            level_labels = _ppl_levels(plan.columns[axis])
        end
        n = isempty(e.dims) ? e.size : first(e.dims)
        length(level_labels) == n || throw(ContractValidationError(
            "[query] `$name` has $n levels, but `partitioned` gives $(length(level_labels)) " *
            "labels"))
        for c in 1:e.size
            labels[e.offset + c - 1] = level_labels[mod1(c, n)]
        end
    end
    labels
end

function _query_partition_rule(plan, layout, partitioned, keys_by_name)
    declarations = partitioned isa Union{Symbol,Pair} ? (partitioned,) : partitioned
    isempty(declarations) ? () :
        (:unconstrained => _partition_coordinate_labels(plan, layout, declarations, keys_by_name),)
end

function _query_exemplar(layout, u0)
    u0 === nothing && return zeros(layout.total)
    length(u0) == layout.total || throw(ContractValidationError(
        "[query] exemplar length $(length(u0)) ≠ layout total $(layout.total)"))
    Vector{Float64}(u0)
end

_value_query_prefix(cache, residual, residual_plan, args, on_error) =
    (; cache, residual, residual_plan, args, argument_type = typeof(args), on_error,
       residuals = Dict{DataType,Any}(typeof(args) => residual),
       prefixes = Dict{DataType,Any}(Vector{Float64} => cache.kernel))

# Ordinary cell indices include CartesianIndex and multiple integer types;
# partition keys can also have mixed types. Preserve custom packed storage
# at each concrete prepared boundary instead of reading it all to copy it.
function _query_residual(prefix, args)
    typeof(args) === prefix.argument_type && return prefix.residual
    get!(prefix.residuals, typeof(args)) do
        Base.invokelatest(prepare, prefix.residual_plan;
            exemplars = args, on_error = prefix.on_error)
    end
end

function _query_prefix_kernel(prefix, u)
    u isa Vector{Float64} && return prefix.cache.kernel
    get!(prefix.prefixes, typeof(u)) do
        Base.invokelatest(prepare, prefix.cache.kernel.plan;
            exemplars = (u, prefix.cache.point, prefix.cache.slots),
            on_error = prefix.on_error)
    end
end

function _query_shared_parts(spec, plan, have, want, rule, varying, on_error)
    isempty(rule) && return nothing
    # Binding the selected plan executes data-only helpers before the shared
    # split. They may have been defined in the caller's fresh model module.
    Base.invokelatest(_query_shared_parts_latest, spec, plan, have, want,
        rule, varying, on_error)
end

function _query_shared_parts_latest(spec, plan, have, want, rule, varying, on_error)
    selected = ReactiveKernels.plan(spec; have, want)
    selected = ReactiveKernels._kernel_error_policy(selected, on_error)
    selected = ReactiveKernels._fuse_authored_plate_chains(selected; cells_only = true)
    bound_selected = _with_query_data(plan, spec, on_error) do cache
        ReactiveKernels._partial_apply(selected,
            ReactiveKernels._kernel_bound_pairs(spec, _query_bound(plan));
            _data_cache = cache)
    end
    ReactiveKernels._partition_shared_parts(bound_selected, rule;
        varying, preserve_rows = true)
end

# The column a parameter's first axis takes its levels from, `nothing` when
# that axis is not `levels(<column>)`.
function _partition_parameter_axis(plan::StructuralPlan, name::Symbol)
    axis(ex) = Meta.isexpr(ex, :call, 2) && ex.args[1] === :levels && ex.args[2] isa Symbol ?
        ex.args[2] : nothing
    for p in plan.plate_parameters
        p.name === name && return axis(p.range)
    end
    for p in plan.array_parameters
        p.name === name && return axis(first(p.dims))
    end
    nothing
end

"""
    prepare_partition_sampler(built, plan, observations, u0; by, backend, compact = true,
                              partitioned = (), locality = :error, on_error = nothing) -> PartitionSampler

[`prepare_partition`](@ref), with the same `by`, plus a `prepare_ad` gradient;
the label stays a constant. `u0` is a length-consistent exemplar and `backend` a
DifferentiationInterface backend, as for [`prepare_sampler`](@ref).

With `compact = true` (the default) the gradient is over each partition's
footprint, the packed coordinates it reads, computed at preparation for every
label from the bound index data ([`partition_footprint`](@ref)); the active
coordinates overlay a read-only packed base as in
[`prepare_compact_cell_sampler`](@ref), so with native `AutoEnzyme` Reverse
use ordinary `Enzyme.Reverse`. With `compact = false` the
gradient is over the whole packed vector, nonzero only on the footprint, at
the cost of zeroing the whole vector per call.

`partitioned` is the separability rule: the parameters whose rows belong to
one partition each (an NLME unit's effects), named bare when indexed by the
key column itself (`z[levels(subject)]`) or as `parameter => labels` giving
each level's label. Preparation then checks every label: a partition that
reads another label's coordinates of those parameters, even genuinely, is
refused with a `ReactiveKernels.PartitionSeparabilityError` naming them.

For `parameter => labels`, the label count is the first constrained dimension
(`first(parameter_metadata(built)[i].shape)`), or the packed count for a scalar.
Labels cycle in order over that parameter's packed coordinates. The packed
count need not equal or be divisible by the label count. For a whole local
coupled transform, repeat its unit's label across that constrained axis:
`L ~ LKJCholesky(2, eta)` takes `partitioned = (:L => fill(unit, 2),)`, even
though `L` has only one packed coordinate. More generally an LKJ factor of
width `K` takes `fill(unit, K)` to assign all `K*(K-1)/2` coordinates to `unit`.
This declaration assigns coordinate ownership; it does not split the transform
or its atomic prior/Jacobian factors. A partition query still selects only
the requested observations, not those prior/Jacobian terms.

The compiler also cuts out intermediates that structurally read only shared
coordinates and bound data. The sampler keeps their values across labels and
local-coordinate changes, and recomputes them when a shared coordinate changes.
The cache belongs to this sampler; changes are compared by value, including
in-place edits to `u`. Full gradients propagate cotangents through the prefix
with the chosen backend. That pullback may recompute its forward work; no
backend tape reuse is assumed. `shared = false` on the gradient call requests
only local derivatives and skips that pullback.
"""
function prepare_partition_sampler(built, plan::StructuralPlan, observations,
        u0::AbstractVector{<:Real}; by, backend, compact::Bool = true, partitioned = (),
        locality = :error, on_error = nothing)
    layout = built.layout::LayoutTable
    length(u0) == layout.total || throw(ContractValidationError(
        "[query] exemplar length $(length(u0)) ≠ layout total $(layout.total)"))
    exemplar = Vector{Float64}(u0)
    isbound(plan) || throw(ContractValidationError(
        "[query] prepare_partition_sampler requires a bound plan (bind_data first)"))
    _check_built_program(built, plan)
    names = observations isa Symbol ? (observations,) : Tuple(Symbol.(observations))
    isempty(names) && throw(ContractValidationError(
        "[query] prepare_partition_sampler needs at least one observation name"))
    _refuse_absent_cells(plan, names)
    _prepare_partition_sampler(_partition_label_keys(built, plan, by)..., names,
        layout, exemplar; backend, compact, partitioned, locality, on_error)
end

function _prepare_partition_sampler(built, plan::StructuralPlan, by, names, layout, exemplar;
        backend, compact, partitioned, locality, on_error)
    keys_by_name = _partition_keys(plan, names, by)
    labels = _partition_labels(plan, names, keys_by_name)
    isempty(labels) && throw(ContractValidationError(
        "[query] the partition key columns of $(join(names, ", ")) hold no label"))
    rule = _query_partition_rule(plan, layout, partitioned, keys_by_name)
    # Split before lowering enforces locality. A shared opaque producer may
    # have no row form even though its cached frontier is a valid row input
    # to every partition. The residual keeps the requested locality policy.
    spec, want = _partition_query_spec(built, plan, names, keys_by_name; locality,
                                       partitioned = rule)
    have = (_query_have(plan)..., _PARTITION_PORT, _PLAN_PORT)
    parts = _query_shared_parts(spec, plan, have, want, rule,
        (_PARTITION_PORT, _PLAN_PORT), on_error)
    # This dense kernel supplies the original packed-coordinate footprints
    # and separability check, including dependencies through the prefix. It
    # is not the cached sampler's executable residual. Only that residual is
    # subject to the requested locality check when a prefix exists.
    dense_locality = parts === nothing ? locality : :ignore
    if parts !== nothing
        spec, want = _partition_query_spec(built, plan, names, keys_by_name;
            locality = dense_locality, partitioned = rule)
    end
    planner = _absent_refusal(plan, spec, have, want) do
        Base.invokelatest(_prepare_cached_partition, plan, spec; have, want,
                          bound = _query_bound(plan), on_error)
    end
    plans, plan_type = _partition_plans(planner, labels, exemplar)
    spec, want = _partition_query_spec(built, plan, names, keys_by_name;
                                       locality = dense_locality, partitioned = rule, plan_type)
    kernel = _absent_refusal(plan, spec, have, want) do
        Base.invokelatest(_prepare_cached, plan, spec; have, want, bound = _query_bound(plan), on_error)
    end
    footprints = Dict{Any,Vector{Int}}()
    footprint = Base.invokelatest(ReactiveKernels.partition_footprint_kernel, kernel)
    for label in labels
        read = Base.invokelatest(footprint, exemplar, label, nothing)
        isempty(rule) || Base.invokelatest(ReactiveKernels.partition_separability, kernel,
                                           exemplar, label, nothing; footprint = read)
        positions = read.unconstrained
        footprints[label] = positions === :whole ? collect(1:layout.total) : positions
    end
    local_footprints = isempty(rule) ? nothing :
        Dict{Any,Vector{Int}}(label => filter(p -> !ismissing(last(only(rule))[p]), footprint)
                             for (label, footprint) in footprints)
    if parts !== nothing
        return _prepare_shared_partition_sampler(kernel, parts, layout, names,
            keys_by_name, labels, exemplar, footprints, local_footprints,
            last(only(rule)), backend, compact, on_error)
    end
    if !compact
        ad = _partition_prepare_ad(labels) do label
            Base.invokelatest(prepare_ad, kernel, backend, exemplar, label,
                plans[label]; active = :unconstrained)
        end
        return PartitionSampler(kernel, ad, layout, names, keys_by_name, labels, exemplar,
            footprints, local_footprints, Dict{Any,Vector{Int}}(), Int[], Int32[],
            Float64[], Float64[], plans, nothing, NaN)
    end
    # One selection width for every label: each footprint padded with the
    # smallest positions outside it (their derivative is zero).
    selections = _partition_selections(footprints)
    selected = copy(selections[first(labels)])
    width = length(selected)
    slots = _compact_cell_slots(layout.total, selected)
    active = exemplar[selected]
    # The slot map is constant per label, so the plan reads the active
    # vector's slots of every row once per label (`_cell_lazy_rows`): each
    # label's plan is computed with its own slot map.
    spec, want = _partition_query_spec(built, plan, names, keys_by_name; locality,
                                       partitioned = rule, plan_constant = (:_ppl_cell_slots,))
    have = (_compact_overlay!(spec, plan)..., _PARTITION_PORT, _PLAN_PORT)
    planner = _absent_refusal(plan, spec, have, want) do
        Base.invokelatest(_prepare_cached_partition, plan, spec; have, want,
                          bound = _query_bound(plan), on_error)
    end
    plans, plan_type = _partition_plans_each(labels) do label
        selection = selections[label]
        Base.invokelatest(planner, exemplar, exemplar[selection],
                          _compact_cell_slots(layout.total, selection), label, nothing)
    end
    spec, want = _partition_query_spec(built, plan, names, keys_by_name; locality,
                                       partitioned = rule, plan_type,
                                       plan_constant = (:_ppl_cell_slots,))
    _compact_overlay!(spec, plan)
    compact_kernel = _absent_refusal(plan, spec, have, want) do
        Base.invokelatest(_prepare_cached, plan, spec; have, want, bound = _query_bound(plan), on_error)
    end
    ad = _partition_prepare_ad(labels) do label
        selection = selections[label]
        Base.invokelatest(prepare_ad, compact_kernel, backend, exemplar,
            exemplar[selection], _compact_cell_slots(layout.total, selection),
            label, plans[label]; active = :_ppl_cell_active)
    end
    PartitionSampler(kernel, ad, layout, names, keys_by_name, labels, exemplar, footprints,
        local_footprints, selections, selected, slots, active, zeros(width), plans, nothing, NaN)
end

"""
    partition_value_and_gradient!(s::PartitionSampler, g, u, label; shared = true)

The partition `label`'s log density and its gradient, written into `g`:
over [`partition_footprint`](@ref)`(s, label)`, in that order, for a compact
sampler (`length(g)` must be the footprint's length), or over the whole
packed vector otherwise. Returns `(value, g)`. `u` is read only.

With `shared = false`, return only the derivatives of the coordinates named
by `partitioned` at preparation. A compact destination follows
`partition_footprint(s, label; shared = false)`; a dense destination retains
the full packed length, with every shared-coordinate entry zero. The value
is unchanged. This skips the shared prefix's pullback while reusing its value.
"""
function partition_value_and_gradient!(s::PartitionSampler, g::AbstractVector,
                                       u::Vector{Float64}, label; shared::Bool = true)
    length(u) == s.layout.total || throw(ContractValidationError(
        "[query] packed length $(length(u)) ≠ layout total $(s.layout.total)"))
    footprint = _partition_gradient_footprint(s, label, shared)
    expected = _compact(s) ? length(footprint) : s.layout.total
    length(g) == expected || throw(ContractValidationError(
        "[query] gradient length $(length(g)) ≠ expected length $expected " *
        "of partition $(repr(label))"))
    s.prefix === nothing || return (
        _latest_value(_shared_partition_gradient!, s, g, u, label, shared), g)
    if !_compact(s)
        value = _latest_value(_partition_gradient!, s, g, label, u, label,
                              _partition_plan_of(s, label))
        if !shared
            local_index = 1
            for position in eachindex(g)
                if local_index <= length(footprint) && position == footprint[local_index]
                    local_index += 1
                else
                    g[position] = 0
                end
            end
        end
        return value, g
    end
    selection = s.selections[label]
    _select_partition!(s, selection)
    for (k, position) in enumerate(selection)
        s.active[k] = u[position]
    end
    value = _latest_value(_partition_gradient!, s, s.gradient, label, u, s.active, s.slots, label,
                          _partition_plan_of(s, label))
    for (k, position) in enumerate(footprint)
        g[k] = s.gradient[s.slots[position]]
    end
    return value, g
end
partition_value_and_gradient!(s::PartitionSampler, g::AbstractVector, u::AbstractVector,
                              label; shared::Bool = true) =
    partition_value_and_gradient!(s, g, Vector{Float64}(u), label; shared)

# Pad each exact footprint to one AD input width across all labels.
function _partition_selections(footprints)
    width = maximum(length, values(footprints))
    selections = Dict{Any,Vector{Int}}()
    for (label, footprint) in footprints
        selection = copy(footprint)
        inside = Set(footprint)
        position = 1
        while length(selection) < width
            position in inside || push!(selection, position)
            position += 1
        end
        selections[label] = sort!(selection)
    end
    selections
end

# The existing compact overlay, applied to an exact selected Plan. Replanning
# from its original graph could recover the shared producers just cut out.
function _compact_partition_plan(p::ReactiveKernels.Plan)
    old = p.graph
    graph = Graph(copy(old.values), copy(old.recipes),
        Dict(id => copy(ids) for (id, ids) in old.producers), copy(old.aliases), old.version, nothing)
    overlay = map((:_ppl_cell_base, :_ppl_cell_active, :_ppl_cell_slots),
                  (AbstractVector{Float64}, AbstractVector{Float64}, AbstractVector{Int32})) do name, type
        any(v -> v.name === name, values(graph.values)) && throw(ContractValidationError(
            "[query] compact cell port `$name` collides with a model port"))
        value!(graph, name, type)
    end
    packed = only(v for v in p.have if v.name === :unconstrained)
    recipe = ReactiveKernels._add_recipe!(graph, overlay, (packed,), _compact_cell_point,
        1.0, nothing, false, ReactiveKernels._NO_KERNEL_SOURCE)
    recipes = ReactiveKernels.Recipe[recipe]
    for r in p.recipes
        op = r.op
        if op isa ReactiveKernels._AuthoredPlateCellOp && op.plan !== nothing
            op = ReactiveKernels._AuthoredPlateCellOp(op.plate, op.locality, op.rows,
                op.partitioned, op.plan, (op.plan_constant..., :_ppl_cell_slots))
            r = ReactiveKernels.Recipe(r.id, r.inputs, r.outputs, op, r.cost,
                r.cse_key, r.effectful, r.source)
        end
        push!(recipes, r)
    end
    template = ReactiveKernels.Plan(graph, p.have, p.want, p.recipes, p.producer,
        p.cost, p.recipes)
    ReactiveKernels._partial_subplan(template,
        ReactiveKernels.Value[overlay..., filter(!=(packed), p.have)...], p.want, recipes)
end

mutable struct _PartitionPrefix{K,A,V}
    const kernel::K
    const ad::A
    const positions::Vector{Int}
    const point::Vector{Float64}
    const slots::Vector{Int32}
    values::V
    valid::Bool
end

# Views must not keep mutable caller storage as the cached shared value.
_partition_cache_value(x::AbstractArray) = copy(x)
_partition_cache_value(x::Tuple) = map(_partition_cache_value, x)
_partition_cache_value(x::NamedTuple) = map(_partition_cache_value, x)
_partition_cache_value(x) = x
_partition_seed(x::Number) = zero(x)
_partition_seed(x::AbstractArray) = zero(x)
_partition_seed(x::Tuple) = map(_partition_seed, x)
_partition_seed(x::NamedTuple) = map(_partition_seed, x)
_partition_frontier_values(kernel, value) = length(outputs(kernel)) == 1 ? (value,) : value
_partition_frontier_seed(values::Tuple) = length(values) == 1 ? only(values) : values

function _prepare_value_prefix(parts, layout, coordinate_labels, exemplar, on_error)
    shared_positions = findall(ismissing, coordinate_labels)
    shared_point = exemplar[shared_positions]
    shared_slots = _compact_cell_slots(layout.total, shared_positions)
    prefix = Base.invokelatest(prepare, _compact_partition_plan(parts.prefix);
        exemplars = (exemplar, shared_point, shared_slots), on_error)
    cached = map(_partition_cache_value, _partition_frontier_values(prefix,
        Base.invokelatest(prefix, exemplar, shared_point, shared_slots)))
    _PartitionPrefix(prefix, nothing, shared_positions, shared_point, shared_slots, cached, true)
end

function _prepare_shared_partition_sampler(kernel, parts, layout, names, by, labels,
        exemplar, footprints, local_footprints, coordinate_labels, backend, compact, on_error)
    initial = _prepare_value_prefix(parts, layout, coordinate_labels, exemplar, on_error)
    cached = initial.values
    seed = _partition_frontier_seed(map(_partition_seed, cached))
    prefix_ad = Base.invokelatest(prepare_ad_pullback, initial.kernel, backend, seed,
        exemplar, initial.point, initial.slots; active = :_ppl_cell_active)
    cache = _PartitionPrefix(initial.kernel, prefix_ad, initial.positions, initial.point,
        initial.slots, cached, true)

    selections = compact ? _partition_selections(footprints) : Dict{Any,Vector{Int}}()
    selected = compact ? copy(selections[first(labels)]) : Int[]
    slots = compact ? _compact_cell_slots(layout.total, selected) : Int32[]
    active = exemplar[selected]
    args = compact ? (exemplar, active, slots, first(labels), nothing, cached...) :
                     (exemplar, first(labels), nothing, cached...)
    residual_plan = compact ? _compact_partition_plan(parts.residual) : parts.residual
    residual = Base.invokelatest(prepare, residual_plan; exemplars = args, on_error)
    planner = Base.invokelatest(ReactiveKernels.partition_plan_kernel, residual)
    plans, _ = _partition_plans_each(labels) do label
        label_args = compact ? (exemplar, exemplar[selections[label]],
            _compact_cell_slots(layout.total, selections[label]), label, nothing, cached...) :
            (exemplar, label, nothing, cached...)
        Base.invokelatest(planner, label_args...)
    end
    selector = (compact ? :_ppl_cell_active : :unconstrained,
                (v.name for v in parts.frontier)...)
    ad = _partition_prepare_ad(labels) do label
        label_args = compact ? (exemplar, exemplar[selections[label]],
            _compact_cell_slots(layout.total, selections[label]), label, plans[label], cached...) :
            (exemplar, label, plans[label], cached...)
        # Exemplar preparation also specializes the kernel's call boundary.
        # Keep that boundary consistent with each AD context signature.
        label_kernel = typeof(label) === typeof(first(labels)) ? residual :
            Base.invokelatest(prepare, residual_plan; exemplars = label_args, on_error)
        Base.invokelatest(prepare_ad, label_kernel, backend, label_args...; active = selector)
    end
    PartitionSampler(kernel, ad, layout, names, by, labels, exemplar, footprints,
        local_footprints, selections, selected, slots, active, zeros(length(active)), plans, cache, NaN)
end

function _refresh_partition_prefix!(cache::_PartitionPrefix, u, kernel = cache.kernel)
    changed = !cache.valid || any(eachindex(cache.positions)) do i
        !isequal(cache.point[i], u[cache.positions[i]])
    end
    changed || return cache.values
    # A failed evaluation must not leave an old value marked valid for a new
    # point. A later retry at that point recomputes the prefix.
    cache.valid = false
    for (i, position) in enumerate(cache.positions)
        cache.point[i] = u[position]
    end
    cache.values = map(_partition_cache_value, _partition_frontier_values(kernel,
        kernel(u, cache.point, cache.slots)))
    cache.valid = true
    cache.values
end

function _shared_partition_gradient!(s::PartitionSampler, g, u, label, shared)
    cache = s.prefix
    cached = _refresh_partition_prefix!(cache, u)
    plan = _partition_plan_of(s, label)
    ad = _partition_ad(s.ad, label)
    if _compact(s)
        selection = s.selections[label]
        _select_partition!(s, selection)
        for (i, position) in enumerate(selection)
            s.active[i] = u[position]
        end
        return _shared_partition_evaluate!(s, g, u, label, shared,
            ad, u, s.active, s.slots, label, plan, cached...)
    end
    _shared_partition_evaluate!(s, g, u, label, shared, ad, u, label, plan, cached...)
end

# Keep each call's argument tuple length concrete, and specialize together on
# its plan, frontier and AD operator even when label types differ.
function _shared_partition_evaluate!(s, g, u, label, shared,
        ad, args::Vararg{Any,N}) where {N}
    value, derivatives = ad_value_and_gradient(ad, args...)
    direct = first(derivatives)
    if shared
        cache = s.prefix
        seed = _partition_frontier_seed(Base.tail(derivatives))
        propagated = ad_pullback(cache.ad, seed, u, cache.point, cache.slots)
        for (i, position) in enumerate(cache.positions)
            slot = _compact(s) ? s.slots[position] : position
            iszero(slot) || (direct[slot] += propagated[i])
        end
    end
    footprint = _partition_gradient_footprint(s, label, shared)
    if _compact(s)
        for (i, position) in enumerate(footprint)
            g[i] = direct[s.slots[position]]
        end
    elseif shared
        copyto!(g, direct)
    else
        fill!(g, 0)
        for position in footprint
            g[position] = direct[position]
        end
    end
    value
end

# The plan of `label` for the sampler's differentiated kernel.
_partition_plan_of(s::PartitionSampler, label) = get(s.plans, label) do
    throw(ContractValidationError("[query] no partition has the label $(repr(label)); " *
        "the labels are $(join(repr.(s.labels), ", "))"))
end

# Point the slot map at `selection` when it differs from the one in force.
function _select_partition!(s::PartitionSampler, selection::Vector{Int})
    _same_selection(s.selected, selection) && return nothing
    for position in s.selected
        s.slots[position] = 0
    end
    copyto!(resize!(s.selected, length(selection)), selection)
    for (slot, position) in enumerate(s.selected)
        s.slots[position] = slot
    end
    nothing
end
