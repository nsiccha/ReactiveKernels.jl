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
vocabulary: every program built by [`build_kernel`](@ref) defines all four.
"""
const PPL_NODES = (
    likelihood = :likelihood,
    prior = :prior,
    log_jacobian = :log_jacobian,
    posterior = :posterior,
)

"""
    WORKFLOW_WANTS

Preset `want` selections over generated programs, as a NamedTuple mapping
preset name → the `want` argument. Prefer [`workflow_wants`](@ref), which
validates the preset name.
"""
const WORKFLOW_WANTS = (
    sampler = :posterior,
    likelihood = :likelihood,
    prior = :prior,
    log_jacobian = :log_jacobian,
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
    prepare_query(built, plan, preset::Symbol)

Prepare the `build_kernel` output `built` for a named workflow `preset`
over the sampler boundary (`:unconstrained` + data columns, data hoisted
via `bound=`). Thin wrapper over `ReactiveKernels.prepare` with
`want = workflow_wants(preset)`; the returned kernel maps an unconstrained
`Vector{Float64}` to the preset node's value. The preparation itself is
world-age safe (callable from compiled functions), but the RAW returned
kernel closes over build-time eval'd code: call it from top level, wrap
the call in `Base.invokelatest`, or use [`SamplerQuery`](@ref), whose
call paths carry the barrier.
"""
function prepare_query(built, plan::StructuralPlan, preset::Symbol)
    isbound(plan) || throw(ContractValidationError(
        "[query] prepare_query requires a bound plan (bind_data first)"))
    # World-age barrier: `built.spec` holds closures eval'd at build time
    # (newer than any already-compiled caller), so partial evaluation must
    # run at the latest world. Same barrier guards every call below.
    return Base.invokelatest(prepare, built.spec; have = _query_have(plan),
        want = workflow_wants(preset), bound = _query_bound(plan))
end

"""
    SamplerQuery

Reusable sampler-space density + gradient over a built program: the prepared
`:posterior` value kernel plus its `prepare_ad` gradient preparation.
Construct with [`prepare_sampler`](@ref); not thread-safe (one per caller).
"""
struct SamplerQuery{K,P,L}
    kernel::K
    ad::P
    layout::L
end

"""
    prepare_sampler(built, plan, u0::AbstractVector{<:Real}; backend) -> SamplerQuery

Prepare a reusable [`SamplerQuery`](@ref): the `:sampler`-cut value kernel
plus a `prepare_ad` gradient with `active = :unconstrained`. `u0` is a
length-consistent type/shape exemplar (checked against `layout.total`
before any preparation work); `backend` is any
`DifferentiationInterface.AbstractADType` (e.g. `AutoEnzyme` reverse mode).
"""
function prepare_sampler(built, plan::StructuralPlan, u0::AbstractVector{<:Real};
        backend)
    layout = built.layout::LayoutTable
    length(u0) == layout.total || throw(ContractValidationError(
        "[query] exemplar length $(length(u0)) ≠ layout total $(layout.total)"))
    kern = prepare_query(built, plan, :sampler)
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

"""
    restore_draws(layout::LayoutTable, U::AbstractMatrix{<:Real}) -> NamedTuple

Restore named constrained parameters from an unconstrained draws matrix `U`
(`layout.total` rows × draws columns, e.g. HMC output). Returns a NamedTuple
with one entry per layout entry — coefficient predictors map to
`(size × draws)` matrices, sampled parameters to length-`draws` vectors —
keyed exactly as [`constrain`](@ref). Each column is constrained
independently through `constrain`, so transforms stay single-sourced.
"""
function restore_draws(layout::LayoutTable, U::AbstractMatrix{<:Real})
    size(U, 1) == layout.total || throw(ContractValidationError(
        "[query] draws rows $(size(U, 1)) ≠ layout total $(layout.total)"))
    n = size(U, 2)
    if n == 0
        pairs = Pair{Symbol,Any}[]
        for e in layout.entries
            if e.kind === :coefficient
                push!(pairs, e.predictor => Matrix{Float64}(undef, e.size, 0))
            else
                push!(pairs, e.name => Vector{Float64}(undef, 0))
            end
        end
        return NamedTuple{Tuple(first.(pairs))}(Tuple(last.(pairs)))
    end
    nts = map(1:n) do j
        constrain(layout, view(U, :, j))
    end
    first_nt = first(nts)
    names = Tuple(keys(first_nt))
    cols = map(names) do k
        v = first_nt[k]
        v isa AbstractVector ?
            hcat([Vector{Float64}(nt[k]) for nt in nts]...) :
            Float64[nt[k] for nt in nts]
    end
    return NamedTuple{names}(Tuple(cols))
end
