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
_query_have(plan::StructuralPlan) =
    (:unconstrained, sort!(collect(keys(plan.columns)))...)

function _query_bound(plan::StructuralPlan)
    names = sort!(collect(keys(plan.columns)))
    return NamedTuple{Tuple(names)}(Tuple(plan.columns[k] for k in names))
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

Under Reactant the barrier goes AROUND the compile, never inside the traced
call: `Reactant.@compile kernel(Reactant.to_rarray(u))` at top level, or
`Base.invokelatest(Reactant.compile, kernel, (Reactant.to_rarray(u),))` from
an older world. A traced wrapper `u -> Base.invokelatest(kernel, u)` is
opaque to Reactant's tracing overlay and to RK's tensorized lowering, so the
packed scalar reads `unconstrained[i]` reach Reactant's scalar-indexing ban
and fail with `Scalar indexing is disallowed` (measured on Reactant 0.2.285
with the earlier `sum(view(unconstrained, i:i))` spelling, which fell through
to Base's scalar `mapreduce`; `test_reactant_joint.jl` pins the working
shape).
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
    return Base.invokelatest(prepare, built.spec; have = _query_have(plan),
        want, bound = _query_bound(plan), on_error, exemplars)
end

# A built graph evaluates the program its build generated. A binding that
# generates the same program (other row counts and data values, or another
# gensym-created lowering namespace) evaluates on it exactly as on its own
# build. A binding that generates another program
# (other level labels or parameter sizes, data element types, missing
# entries, an empty response, another broadcast shape) needs its own build;
# evaluating the old graph on it can be silently wrong, so it is refused.
# Values made outside `build_kernel` carry no `program` and are not checked.
function _check_built_program(built, plan::StructuralPlan)
    hasproperty(built, :program) || return nothing
    program = _program_identity(kernel_expr(plan, assign_layout(plan)))
    _same_program(program, built.program) && return nothing
    throw(ContractValidationError("[query] this binding generates a " *
        "different program than the one this graph was built from, so it " *
        "needs its own `build_kernel(plan)`. First difference, " *
        _program_difference(built.program, program)))
end

function _program_difference(built::Expr, other::Expr)
    text(x) = x === nothing ? "(absent)" :
        sprint(print, Base.remove_linenums!(deepcopy(x)))
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

"""
    SamplerQuery

Reusable sampler-space density + gradient over a built program: the prepared
`:posterior` value kernel plus its `prepare_ad` gradient preparation.
Construct with [`prepare_sampler`](@ref); not thread-safe (one per caller).
The call paths below carry a `Base.invokelatest` barrier, so they are not
traceable by Reactant; compile the fields directly instead —
`Reactant.@compile q.kernel(traced_u)` for the value and
`compile_ad_value_and_gradient(q.ad, traced_u)` for value + gradient (see
[`prepare_query`](@ref)).

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
    prepare_sampler(built, plan, u0::AbstractVector{<:Real}; backend, on_error = nothing, retain = (), warm = false, native_forward_batch = false) -> SamplerQuery

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
operator for traced calls and a later input with a different active shape.
"""
function prepare_sampler(built, plan::StructuralPlan, u0::AbstractVector{<:Real};
        backend, on_error = nothing, retain::Tuple = (), warm::Bool = false,
        native_forward_batch::Bool = false)
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
    use_forward = native_forward_batch && isempty(wants) &&
        ReactiveKernels._native_forward_batch_eligible(
            backend, gradient_kernel, u)
    prep = Base.invokelatest(prepare_ad, gradient_kernel, backend,
        u; active = :unconstrained, retain = wants, warm,
        native_forward_batch = use_forward)
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
`ad_hvp(q, tangents, u)`, `ad_gradient_and_hvp(q, tangents, u)` and their
other value/in-place forms. Each call carries the `Base.invokelatest` barrier
the query's build-time code needs, so they are safe from compiled callers.
Not thread-safe (one per caller); not traceable by Reactant.
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
`prepare_ad_pushforward` or `prepare_ad_hvp` — and `exemplars` are that
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
```

Gradients and Hessian-vector products need a scalar preset (`:sampler`,
`:likelihood`, `:prior`, `:log_jacobian`); pullbacks and pushforwards also
accept `:pointwise` and `:observations`. Directions with disjoint supports
over `coordinate_names(built.layout)` compress block-structured Jacobians and
Hessians (see `ReactiveKernels.prepare_ad_hvp`). The backend's packages must be
loaded in the calling session.
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
        :ad_hvp, :ad_hvp!, :ad_gradient_and_hvp, :ad_gradient_and_hvp!)
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
    prepare_cell_query(built, plan, observations; on_error = nothing) -> CellQuery

Prepare the summed log density of ONE cell of the observations named in
`observations` (a name or a collection of names), evaluated at an
unconstrained point `u` and a position `i`: `q(u, i)`. Each observation's
pointwise values come from a plate (`:pointwise`'s array for that name); cell
`i` is that plate's entry `i`, so

    sum(q(u, i) for i in eachindex(pointwise.y)) == pointwise likelihood of y

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
        on_error = nothing)
    isbound(plan) || throw(ContractValidationError(
        "[query] prepare_cell_query requires a bound plan (bind_data first)"))
    _check_built_program(built, plan)
    names = observations isa Symbol ? (observations,) : Tuple(Symbol.(observations))
    isempty(names) && throw(ContractValidationError(
        "[query] prepare_cell_query needs at least one observation name"))
    spec, want = _cell_query_spec(built, plan, names)
    kernel = Base.invokelatest(prepare, spec; have = (_query_have(plan)..., _CELL_PORT),
        want, bound = _query_bound(plan), on_error)
    return CellQuery(kernel, built.layout, names)
end

const _CELL_PORT = :_ppl_cell

# Each observation's cell plate (its per-index totals when the loop observes
# one array per index), one `plate_cell` value per plate at the shared position
# port, and their sum when there are several.
function _cell_query_spec(built, plan::StructuralPlan, names)
    body = kernel_expr(plan, built.layout).args[2].args
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
            plate_cell(spec, node; index = _CELL_PORT, name = cell)
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
    prepare_cell_sampler(built, plan, observations, u0; backend, on_error = nothing) -> CellSampler

[`prepare_cell_query`](@ref) plus a `prepare_ad` gradient with
`active = :unconstrained`; the position stays a constant (an `Int`). `u0` is a
length-consistent exemplar and `backend` a DifferentiationInterface backend,
as for [`prepare_sampler`](@ref).
"""
function prepare_cell_sampler(built, plan::StructuralPlan, observations,
        u0::AbstractVector{<:Real}; backend, on_error = nothing)
    layout = built.layout::LayoutTable
    length(u0) == layout.total || throw(ContractValidationError(
        "[query] exemplar length $(length(u0)) ≠ layout total $(layout.total)"))
    q = prepare_cell_query(built, plan, observations; on_error)
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
# The packed base and coordinate selection cross the AD boundary as constants;
# the graph still contains every model calculation after this selection.
struct _CompactCellPoint{B<:AbstractVector{Float64},A<:AbstractVector{Float64},I<:AbstractVector{Int}} <: AbstractVector{Float64}
    base::B
    active::A
    indices::I
end

Base.size(point::_CompactCellPoint) = size(point.base)
Base.IndexStyle(::Type{<:_CompactCellPoint}) = IndexLinear()
@inline function Base.getindex(point::_CompactCellPoint, index::Int)
    @boundscheck checkbounds(point.base, index)
    slot = searchsortedfirst(point.indices, index)
    slot <= length(point.indices) && @inbounds(point.indices[slot] == index) ?
        @inbounds(point.active[slot]) : @inbounds(point.base[index])
end

@inline _compact_cell_point(base, active, indices) =
    _CompactCellPoint(base, active, indices)

function _check_compact_cell_indices(base, active, indices, total)
    length(base) == total || throw(ContractValidationError(
        "[query] packed base length $(length(base)) ≠ layout total $total"))
    length(active) == length(indices) || throw(ContractValidationError(
        "[query] active length $(length(active)) ≠ selection length $(length(indices))"))
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
Prepare one sampler per concurrent caller.
"""
mutable struct CompactCellSampler{P,L,O}
    const ad::P
    const layout::L
    const observations::O
    value::Float64
end
CompactCellSampler(ad, layout, observations) =
    CompactCellSampler(ad, layout, observations, NaN)

"""
    prepare_compact_cell_sampler(built, plan, observations, base0, indices0; backend)

Prepare a cell value and gradient over the sorted, unique packed positions
`indices0`. At evaluation, the selected positions may change, provided the
active vector and selection retain the exemplar's type and length. The packed
base is passed read-only; selected coordinates are supplied separately. This
keeps the active AD point and gradient proportional to the selected cell's
coordinates rather than the full layout. With native `AutoEnzyme` Reverse, use
`Enzyme.set_runtime_activity(Enzyme.Reverse)`: packed reads can select either an
active coordinate or a constant base coordinate.
"""
function prepare_compact_cell_sampler(built, plan::StructuralPlan, observations,
        base0::AbstractVector{Float64}, indices0::AbstractVector{Int};
        backend, on_error = nothing)
    layout = built.layout::LayoutTable
    _check_compact_cell_indices(base0, indices0, indices0, layout.total)
    active0 = base0[indices0]
    isbound(plan) || throw(ContractValidationError(
        "[query] prepare_compact_cell_sampler requires a bound plan (bind_data first)"))
    _check_built_program(built, plan)
    names = observations isa Symbol ? (observations,) : Tuple(Symbol.(observations))
    isempty(names) && throw(ContractValidationError(
        "[query] prepare_compact_cell_sampler needs at least one observation name"))
    spec, want = _cell_query_spec(built, plan, names)
    for (name, type) in ((:_ppl_cell_base, AbstractVector{Float64}),
            (:_ppl_cell_active, AbstractVector{Float64}),
            (:_ppl_cell_indices, AbstractVector{Int}))
        haskey(spec.ports, name) && throw(ContractValidationError(
            "[query] compact cell port `$name` collides with a model port"))
        spec.ports[name] = value!(spec.graph, name, type)
        push!(spec.port_order, name)
    end
    add!(spec.graph, (spec[:_ppl_cell_base], spec[:_ppl_cell_active],
        spec[:_ppl_cell_indices]) => spec[:unconstrained], _compact_cell_point)
    filter!(!=(:unconstrained), spec.have_names)
    prepend!(spec.have_names, [:_ppl_cell_base, :_ppl_cell_active, :_ppl_cell_indices])
    kernel = Base.invokelatest(prepare, spec; have = Tuple(spec.have_names),
        want, bound = _query_bound(plan), on_error)
    ad = Base.invokelatest(prepare_ad, kernel, backend, base0, active0, indices0, 1;
        active = :_ppl_cell_active)
    CompactCellSampler(ad, layout, names)
end

"""
    cell_value_and_gradient!(sampler::CompactCellSampler, gradient, base, active, indices, cell)

Return one cell's density and write only the selected-coordinate gradient.
`base` and `indices` are not mutated. The active values overlay `base` at the
sorted positions in `indices`.
"""
function cell_value_and_gradient!(sampler::CompactCellSampler, gradient::AbstractVector,
        base::AbstractVector{Float64}, active::AbstractVector{Float64},
        indices::AbstractVector{Int}, cell::Int)
    _check_compact_cell_indices(base, active, indices, sampler.layout.total)
    length(gradient) == length(active) || throw(ContractValidationError(
        "[query] compact gradient length $(length(gradient)) ≠ active length $(length(active))"))
    return _latest_value(_cell_gradient!, sampler, gradient, base, active, indices, cell),
        gradient
end

function _stack_restored_draws(values)
    value = first(values)
    if value isa NamedTuple
        names = Tuple(keys(value))
        columns = map(names) do name
            _stack_restored_draws([draw[name] for draw in values])
        end
        return NamedTuple{names}(Tuple(columns))
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
_empty_restored_draws(value::AbstractVector) = zeros(Float64, length(value), 0)
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
