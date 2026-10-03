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
value. The preparation itself is
world-age safe (callable from compiled functions), but the RAW returned
kernel closes over build-time eval'd code: call it from top level, wrap
the call in `Base.invokelatest`, or use [`SamplerQuery`](@ref), whose
call paths carry the barrier.

Under Reactant the barrier goes AROUND the compile, never inside the traced
call: `Reactant.@compile kernel(Reactant.to_rarray(u))` at top level, or
`Base.invokelatest(Reactant.compile, kernel, (Reactant.to_rarray(u),))` from
an older world. A traced wrapper `u -> Base.invokelatest(kernel, u)` is
opaque to Reactant's tracing overlay, so the packed reads
`sum(view(unconstrained, i:i))` fall through to Base's scalar `mapreduce`
and fail with `Scalar indexing is disallowed` (measured on Reactant
0.2.285; `test_reactant_joint.jl` pins the working shape).
"""
function prepare_query(built, plan::StructuralPlan, preset::Symbol; on_error = nothing)
    isbound(plan) || throw(ContractValidationError(
        "[query] prepare_query requires a bound plan (bind_data first)"))
    # World-age barrier: `built.spec` holds closures eval'd at build time
    # (newer than any already-compiled caller), so partial evaluation must
    # run at the latest world. Same barrier guards every call below.
    return Base.invokelatest(prepare, built.spec; have = _query_have(plan),
        want = workflow_wants(preset), bound = _query_bound(plan), on_error)
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
    prepare_sampler(built, plan, u0::AbstractVector{<:Real}; backend, on_error = nothing) -> SamplerQuery

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
"""
function prepare_sampler(built, plan::StructuralPlan, u0::AbstractVector{<:Real};
        backend, on_error = nothing)
    layout = built.layout::LayoutTable
    length(u0) == layout.total || throw(ContractValidationError(
        "[query] exemplar length $(length(u0)) ≠ layout total $(layout.total)"))
    kern = prepare_query(built, plan, :sampler; on_error)
    prep = Base.invokelatest(prepare_ad, kern, backend, Vector{Float64}(u0);
        active = :unconstrained)
    return SamplerQuery(kern, prep, layout)
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
