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
"""
const WORKFLOW_WANTS = (
    sampler = :posterior,
    likelihood = :likelihood,
    prior = :prior,
    log_jacobian = :log_jacobian,
    pointwise = :pointwise,
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
    prepare_query(built, plan, preset::Symbol; on_error = nothing)

Prepare the `build_kernel` output `built` for a named workflow `preset`
over the sampler boundary (`:unconstrained` + data columns, data hoisted
via `bound=`). Thin wrapper over `ReactiveKernels.prepare` with
`want = workflow_wants(preset)` and the supplied `on_error` policy. The
returned kernel maps an unconstrained `Vector{Float64}` to the preset node's
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
function prepare_query(built, plan::StructuralPlan, preset::Symbol; on_error = nothing)
    return _prepare_wants(built, plan, workflow_wants(preset); on_error)
end

# Prepare the built graph over the sampler boundary for one WANT node or a
# tuple of them (a tuple-WANT kernel returns a tuple in that order).
function _prepare_wants(built, plan::StructuralPlan, want; on_error = nothing)
    isbound(plan) || throw(ContractValidationError(
        "[query] prepare_query requires a bound plan (bind_data first)"))
    _check_built_program(built, plan)
    # World-age barrier: `built.spec` holds closures eval'd at build time
    # (newer than any already-compiled caller), so partial evaluation must
    # run at the latest world. Same barrier guards every call below.
    return Base.invokelatest(prepare, built.spec; have = _query_have(plan),
        want, bound = _query_bound(plan), on_error)
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
    program == built.program && return nothing
    throw(ContractValidationError("[query] this binding generates a " *
        "different program than the one this graph was built from, so it " *
        "needs its own `build_kernel(plan)`. First difference, " *
        _program_difference(built.program, program)))
end

function _program_difference(built::Expr, other::Expr)
    text(x) = x === nothing ? "(absent)" : sprint(print, readable_code(x))
    sig(def) = def.args[1]
    sig(built) == sig(other) || return "in the data arguments:\n  built: " *
        text(sig(built)) * "\n  this:  " * text(sig(other))
    body(def) = Any[x for x in def.args[2].args if !(x isa LineNumberNode)]
    a, b = body(built), body(other)
    for i in 1:max(length(a), length(b))
        x, y = get(a, i, nothing), get(b, i, nothing)
        x == y && continue
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
"""
struct SamplerQuery{K,P,L}
    kernel::K
    ad::P
    layout::L
end

"""
    prepare_sampler(built, plan, u0::AbstractVector{<:Real}; backend, on_error = nothing, retain = ()) -> SamplerQuery

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

`retain` names further query presets (`:pointwise`, `:likelihood`, `:prior`,
`:log_jacobian`) whose values the gradient's own primal sweep returns through
[`sampler_value_gradient_and_retained!`](@ref), with no second primal pass.
It needs a backend that keeps that sweep's values (reverse-mode
`AutoEnzyme`; see `ReactiveKernels.ad_retains_primal_sweep`).
"""
function prepare_sampler(built, plan::StructuralPlan, u0::AbstractVector{<:Real};
        backend, on_error = nothing, retain::Tuple = ())
    layout = built.layout::LayoutTable
    length(u0) == layout.total || throw(ContractValidationError(
        "[query] exemplar length $(length(u0)) ≠ layout total $(layout.total)"))
    kern = prepare_query(built, plan, :sampler; on_error)
    gradient_kernel, wants = if isempty(retain)
        kern, ()
    else
        wants = _retained_wants(retain)
        _prepare_wants(built, plan,
            (workflow_wants(:sampler), wants...); on_error), wants
    end
    prep = Base.invokelatest(prepare_ad, gradient_kernel, backend,
        Vector{Float64}(u0); active = :unconstrained, retain = wants)
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
(q::SamplerQuery)(u::Vector{Float64}) = Base.invokelatest(q.kernel, u)
(q::SamplerQuery)(u::AbstractVector{<:Real}) =
    Base.invokelatest(q.kernel, Vector{Float64}(u))

"""
    sampler_value_and_gradient!(q::SamplerQuery, g::AbstractVector, u)

Posterior value and gradient at unconstrained `u`, writing the gradient
into `g` in place. Returns the `(value, gradient)` pair. `g` must be a
valid gradient destination for `u`; `u` is converted to `Vector{Float64}`
unless it already is one.
"""
function sampler_value_and_gradient!(q::SamplerQuery, g::AbstractVector, u::Vector{Float64})
    return Base.invokelatest(ad_value_and_gradient!, q.ad, g, u)
end
function sampler_value_and_gradient!(q::SamplerQuery, g::AbstractVector, u::AbstractVector)
    return Base.invokelatest(ad_value_and_gradient!, q.ad, g, Vector{Float64}(u))
end

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
    return Base.invokelatest(ad_value_gradient_and_retained!, q.ad, g, v)
end

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
struct QueryAD{P,L}
    prepared::P
    layout::L
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

j = prepare_query_ad(prepare_ad_pushforward, built, plan, :pointwise,
                     AutoEnzyme(; mode = Enzyme.Forward), (v,), u0)
pointwise, (jv,) = ad_value_and_pushforward(j, (v,), u)
```

Gradients and Hessian-vector products need a scalar preset (`:sampler`,
`:likelihood`, `:prior`, `:log_jacobian`); pullbacks and pushforwards also
accept `:pointwise`. Directions with disjoint supports
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
        Base.invokelatest(ReactiveKernels.$verb, q.prepared, args...)
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
