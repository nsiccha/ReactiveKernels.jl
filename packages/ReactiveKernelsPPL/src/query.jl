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
(other row counts and data values, or the same source lowered in another
gensym-created isolation module); one that generates another program
(other levels or parameter sizes, data element types, missing entries, an
empty response, another broadcast shape) is refused with a
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
        Base.invokelatest(prepare, built.spec; have,
            want, bound = _query_bound(plan), on_error, exemplars)
    end
end

# A built graph evaluates the program its build generated. A binding that
# generates the same program (other row counts and data values, or another
# gensym-created lowering namespace) evaluates on it exactly as on its own
# build. A binding that generates another program
# (other level labels or parameter sizes, data element types, missing
# entries, an empty response, another broadcast shape) needs its own build;
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
    # (modules among them, which `deepcopy` refuses) are only printed. A long
    # quoted array (a level axis's labels) prints as its summary and ends.
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
mutable struct CellQuery{K,L}
    const kernel::K
    const layout::L
    const observations::Tuple{Vararg{Symbol}}
end

(q::CellQuery)(u::AbstractVector{Float64}, cell) =
    Base.invokelatest(_cell_density, q, u, cell)::Float64
(q::CellQuery)(u::AbstractVector{<:Real}, cell) = q(Vector{Float64}(u), cell)
_cell_density(q::CellQuery, u, cell) = q.kernel(u, cell)

"""
    prepare_cell_query(built, plan, observations; on_error = nothing, locality = nothing) -> CellQuery

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
unless set). `ReactiveKernels.partition_locality(q.kernel)` lists every value.

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
"""
function prepare_cell_query(built, plan::StructuralPlan, observations;
        on_error = nothing, locality = nothing)
    isbound(plan) || throw(ContractValidationError(
        "[query] prepare_cell_query requires a bound plan (bind_data first)"))
    _check_built_program(built, plan)
    names = observations isa Symbol ? (observations,) : Tuple(Symbol.(observations))
    isempty(names) && throw(ContractValidationError(
        "[query] prepare_cell_query needs at least one observation name"))
    _refuse_absent_cells(plan, names)
    spec, want = _cell_query_spec(built, plan, names; locality)
    have = (_query_have(plan)..., _CELL_PORT)
    kernel = _absent_refusal(plan, spec, have, want) do
        Base.invokelatest(prepare, spec; have, want, bound = _query_bound(plan), on_error)
    end
    return CellQuery(kernel, built.layout, names)
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
# selection. A read is one slot load and a branch, where a binary search over
# the selection cost O(log selection) per read (snag
# `compact-cell-cal-58569833`). Native Reverse multiplied that search's cost
# inside gathers far beyond its primal cost; that is Enzyme snag
# `reverse-over-a-g-a9ee15bb`.
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

# A parameter block read from the overlay is not strided, so a product of it
# would take LinearAlgebra's generic `mul!`; RK's native product computes it
# with a typed loop instead (`_native_mul`).
ReactiveKernels._native_mul_operand(::_CompactCellPoint) = true

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

# Point the slot map at `indices`. Validating and rewriting cost O(selection),
# so they run only when the selection differs from the previous call's; a run
# of calls over one selection pays one comparison each.
function _select_compact_cells!(sampler::CompactCellSampler, base, active, indices)
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
    layout = built.layout::LayoutTable
    _check_compact_cell_indices(base0, indices0, indices0, layout.total)
    active0 = base0[indices0]
    selected = Vector{Int}(indices0)
    slots = _compact_cell_slots(layout.total, selected)
    isbound(plan) || throw(ContractValidationError(
        "[query] prepare_compact_cell_sampler requires a bound plan (bind_data first)"))
    _check_built_program(built, plan)
    names = observations isa Symbol ? (observations,) : Tuple(Symbol.(observations))
    isempty(names) && throw(ContractValidationError(
        "[query] prepare_compact_cell_sampler needs at least one observation name"))
    _refuse_absent_cells(plan, names)
    spec, want = _cell_query_spec(built, plan, names; locality)
    have = _compact_overlay!(spec, plan)
    kernel = _absent_refusal(plan, spec, have, want) do
        Base.invokelatest(prepare, spec; have, want, bound = _query_bound(plan), on_error)
    end
    ad = Base.invokelatest(prepare_ad, kernel, backend, base0, active0, slots, 1;
        active = :_ppl_cell_active)
    CompactCellSampler(ad, layout, names, selected, slots)
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
mutable struct PartitionQuery{K,L}
    const kernel::K
    const layout::L
    const observations::Tuple{Vararg{Symbol}}
    const by::Dict{Symbol,Symbol}
    const labels::Vector{Any}
    # Each label's plan (`ReactiveKernels.partition_plan`), computed at
    # preparation; a label without one is planned on the spot, which refuses
    # a label no observation has.
    const plans::Dict{Any,Any}
end

(q::PartitionQuery)(u::AbstractVector{Float64}, label) =
    Base.invokelatest(_partition_density, q, u, label)::Float64
(q::PartitionQuery)(u::AbstractVector{<:Real}, label) = q(Vector{Float64}(u), label)
_partition_density(q::PartitionQuery, u, label) =
    q.kernel(u, label, get(() -> ReactiveKernels.partition_plan(q.kernel, u, label, nothing),
                           q.plans, label))

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
    plans = Dict{Any,Any}(label => Base.invokelatest(planner, args..., label, nothing)
                          for label in labels)
    types = unique!(Any[typeof(plan) for plan in values(plans)])
    length(types) == 1 || throw(ContractValidationError(
        "[query] the partition plans have several types: $(join(types, ", "))"))
    plans, only(types)
end
const _PARTITION_TOTAL = :_ppl_partition_total

# The key column of each observation: one column name for all of them, or
# a named tuple / pairs `observation => column`. Every column must be a
# supplied data column of the binding.
function _partition_keys(plan::StructuralPlan, names, by)
    keys_by_name = by isa Symbol ? Dict{Symbol,Symbol}(name => by for name in names) :
        Dict{Symbol,Symbol}(Symbol(k) => Symbol(v) for (k, v) in pairs(by))
    columns = _query_columns(plan)
    for name in names
        haskey(keys_by_name, name) || throw(ContractValidationError(
            "[query] `by` gives no key column for observation `$name`"))
        key = keys_by_name[name]
        key in columns || throw(ContractValidationError(
            "[query] the partition key `$key` of `$name` is not a supplied data column; " *
            "the columns are $(join(columns, ", "))"))
    end
    keys_by_name
end

# Each observation's cell plate partitioned by its key, and their sum: the
# plates of `_cell_query_spec`, through RK's `plate_partition`.
function _partition_query_spec(built, plan::StructuralPlan, names, keys_by_name;
                               locality, partitioned = (), plan_type = Any)
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
                        plan = plan_type === Any ? _PLAN_PORT : _PLAN_PORT => plan_type)
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
    prepare_partition(built, plan, observations; by, locality = :error, on_error = nothing) -> PartitionQuery

Prepare the summed log density of ONE partition of the observations named in
`observations` (a name or a collection of names): all cells whose key equals
a label, such as one subject's observations, evaluated at an unconstrained
point `u`: `q(u, label)`. `by` names the key: a supplied data column with one
entry per cell of each observation's plate (one row per entry of an
elementwise observation, one per iteration of an `@plate for` loop), either
one column name for every observation or a named tuple or pairs
`observation => column`. [`partition_labels`](@ref) lists the labels.

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
"""
function prepare_partition(built, plan::StructuralPlan, observations; by,
        locality = :error, on_error = nothing)
    isbound(plan) || throw(ContractValidationError(
        "[query] prepare_partition requires a bound plan (bind_data first)"))
    _check_built_program(built, plan)
    names = observations isa Symbol ? (observations,) : Tuple(Symbol.(observations))
    isempty(names) && throw(ContractValidationError(
        "[query] prepare_partition needs at least one observation name"))
    _refuse_absent_cells(plan, names)
    keys_by_name = _partition_keys(plan, names, by)
    labels = _partition_labels(plan, names, keys_by_name)
    have = (_query_have(plan)..., _PARTITION_PORT, _PLAN_PORT)
    exemplar = zeros(built.layout.total)
    spec, want = _partition_query_spec(built, plan, names, keys_by_name; locality)
    planner = _absent_refusal(plan, spec, have, want) do
        Base.invokelatest(ReactiveKernels.partition_plan_kernel, spec; have, want,
                          bound = _query_bound(plan), on_error)
    end
    plans, plan_type = _partition_plans(planner, labels, exemplar)
    spec, want = _partition_query_spec(built, plan, names, keys_by_name; locality, plan_type)
    kernel = _absent_refusal(plan, spec, have, want) do
        Base.invokelatest(prepare, spec; have, want, bound = _query_bound(plan), on_error)
    end
    return PartitionQuery(kernel, built.layout, names, keys_by_name, labels, plans)
end

"""
    PartitionSampler

A [`PartitionQuery`](@ref) with its `prepare_ad` gradient: compact, over each
partition's footprint (the packed coordinates it reads), or dense over the
whole packed vector. Construct with [`prepare_partition_sampler`](@ref); use
[`partition_value_and_gradient!`](@ref) and [`partition_footprint`](@ref).
Not thread-safe: prepare one per thread.
"""
mutable struct PartitionSampler{K,P,L}
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
    const selections::Dict{Any,Vector{Int}}
    const selected::Vector{Int}
    const slots::Vector{Int32}
    const active::Vector{Float64}
    const gradient::Vector{Float64}
    # Each label's plan of the differentiated kernel (`partition_plan`).
    const plans::Dict{Any,Any}
    value::Float64
end
_compact(s::PartitionSampler) = !isempty(s.slots)
partition_labels(s::PartitionSampler) = copy(s.labels)

"""
    partition_footprint(s::PartitionSampler, label) -> Vector{Int}

The packed positions partition `label` reads (sorted, unique): the support of
its gradient, computed at preparation from the bound index data by the row
analysis that evaluates the partition (a method of ReactiveKernels'
`partition_footprint`). A compact sampler's gradient is over exactly these,
in this order.
"""
ReactiveKernels.partition_footprint(s::PartitionSampler, label) = copy(get(s.footprints, label) do
    throw(ContractValidationError("[query] no partition has the label $(repr(label)); " *
        "the labels are $(join(repr.(s.labels), ", "))"))
end)

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
            axis in values(keys_by_name) || throw(ContractValidationError(
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

[`prepare_partition`](@ref) plus a `prepare_ad` gradient; the label stays a
constant. `u0` is a length-consistent exemplar and `backend` a
DifferentiationInterface backend, as for [`prepare_sampler`](@ref).

With `compact = true` (the default) the gradient is over each partition's
footprint, the packed coordinates it reads, computed at preparation for every
label from the bound index data ([`partition_footprint`](@ref)); the active
coordinates overlay a read-only packed base as in
[`prepare_compact_cell_sampler`](@ref), so with native `AutoEnzyme` Reverse
use `Enzyme.set_runtime_activity(Enzyme.Reverse)`. With `compact = false` the
gradient is over the whole packed vector, nonzero only on the footprint, at
the cost of zeroing the whole vector per call. On BRM's nested panel at
atlas-17 size (34627 coordinates, a unit's footprint 2903) the compact
gradient took 2.6 ms and the dense one 3.1 ms per unit.

`partitioned` is the separability rule: the parameters whose rows belong to
one partition each (an NLME unit's effects), named bare when indexed by the
key column itself (`z[levels(subject)]`) or as `parameter => labels` giving
each level's label. Preparation then checks every label: a partition that
reads another label's coordinates of those parameters, even genuinely, is
refused with a `ReactiveKernels.PartitionSeparabilityError` naming them.
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
    keys_by_name = _partition_keys(plan, names, by)
    labels = _partition_labels(plan, names, keys_by_name)
    isempty(labels) && throw(ContractValidationError(
        "[query] the partition key columns of $(join(names, ", ")) hold no label"))
    rule = isempty(partitioned) ? () :
        (:unconstrained => _partition_coordinate_labels(plan, layout, partitioned, keys_by_name),)
    # The dense query: the values, the footprints and the separability check.
    spec, want = _partition_query_spec(built, plan, names, keys_by_name; locality,
                                       partitioned = rule)
    have = (_query_have(plan)..., _PARTITION_PORT, _PLAN_PORT)
    planner = _absent_refusal(plan, spec, have, want) do
        Base.invokelatest(ReactiveKernels.partition_plan_kernel, spec; have, want,
                          bound = _query_bound(plan), on_error)
    end
    plans, plan_type = _partition_plans(planner, labels, exemplar)
    spec, want = _partition_query_spec(built, plan, names, keys_by_name; locality,
                                       partitioned = rule, plan_type)
    kernel = _absent_refusal(plan, spec, have, want) do
        Base.invokelatest(prepare, spec; have, want, bound = _query_bound(plan), on_error)
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
    if !compact
        ad = Base.invokelatest(prepare_ad, kernel, backend, exemplar, first(labels),
            plans[first(labels)]; active = :unconstrained)
        return PartitionSampler(kernel, ad, layout, names, keys_by_name, labels, exemplar,
            footprints, Dict{Any,Vector{Int}}(), Int[], Int32[], Float64[], Float64[], plans,
            NaN)
    end
    # One selection width for every label: each footprint padded with the
    # smallest positions outside it (their derivative is zero).
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
    selected = copy(selections[first(labels)])
    slots = _compact_cell_slots(layout.total, selected)
    active = exemplar[selected]
    spec, want = _partition_query_spec(built, plan, names, keys_by_name; locality,
                                       partitioned = rule)
    have = (_compact_overlay!(spec, plan)..., _PARTITION_PORT, _PLAN_PORT)
    planner = _absent_refusal(plan, spec, have, want) do
        Base.invokelatest(ReactiveKernels.partition_plan_kernel, spec; have, want,
                          bound = _query_bound(plan), on_error)
    end
    plans, plan_type = _partition_plans(planner, labels, exemplar, active, slots)
    spec, want = _partition_query_spec(built, plan, names, keys_by_name; locality,
                                       partitioned = rule, plan_type)
    _compact_overlay!(spec, plan)
    compact_kernel = _absent_refusal(plan, spec, have, want) do
        Base.invokelatest(prepare, spec; have, want, bound = _query_bound(plan), on_error)
    end
    ad = Base.invokelatest(prepare_ad, compact_kernel, backend, exemplar, active, slots,
        first(labels), plans[first(labels)]; active = :_ppl_cell_active)
    PartitionSampler(kernel, ad, layout, names, keys_by_name, labels, exemplar, footprints,
        selections, selected, slots, active, zeros(width), plans, NaN)
end

"""
    partition_value_and_gradient!(s::PartitionSampler, g, u, label)

The partition `label`'s log density and its gradient, written into `g`:
over [`partition_footprint`](@ref)`(s, label)`, in that order, for a compact
sampler (`length(g)` must be the footprint's length), or over the whole
packed vector otherwise. Returns `(value, g)`. `u` is read only.
"""
function partition_value_and_gradient!(s::PartitionSampler, g::AbstractVector,
                                       u::Vector{Float64}, label)
    _compact(s) || return (_latest_value(_cell_gradient!, s, g, u, label,
                                         _partition_plan_of(s, label)), g)
    length(u) == s.layout.total || throw(ContractValidationError(
        "[query] packed length $(length(u)) ≠ layout total $(s.layout.total)"))
    footprint = get(s.footprints, label) do
        throw(ContractValidationError("[query] no partition has the label $(repr(label)); " *
            "the labels are $(join(repr.(s.labels), ", "))"))
    end
    length(g) == length(footprint) || throw(ContractValidationError(
        "[query] compact gradient length $(length(g)) ≠ footprint length " *
        "$(length(footprint)) of partition $(repr(label))"))
    selection = s.selections[label]
    _select_partition!(s, selection)
    for (k, position) in enumerate(selection)
        s.active[k] = u[position]
    end
    value = _latest_value(_cell_gradient!, s, s.gradient, u, s.active, s.slots, label,
                          _partition_plan_of(s, label))
    for (k, position) in enumerate(footprint)
        g[k] = s.gradient[s.slots[position]]
    end
    return value, g
end
partition_value_and_gradient!(s::PartitionSampler, g::AbstractVector, u::AbstractVector,
                              label) =
    partition_value_and_gradient!(s, g, Vector{Float64}(u), label)

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
