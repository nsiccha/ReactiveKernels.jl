# Reusable bijector library (Option A — decision 0l3dsru).
#
# Each CONSTRAINED unconstrained→constrained transform is ONE method-bearing
# @kernel exposing `constrain` / `unconstrain` / `logjac` endpoints — a single
# source of truth that BOTH the host path (`constrain`/`unconstrain`/`logjac`
# in layout.jl) and the in-graph generator use. The generator splices the
# `constrain`/`logjac` endpoints (the planner inlines them and demand-prunes),
# and the host runs the same prepared endpoints, so host≡graph agreement is
# STRUCTURAL rather than hand-synced across separate dispatch sites. Adding a
# support = author one @kernel below + one `BIJECTORS`/`BIJECTOR_NAMES` entry.
#
# `:identity` (real support) carries no transform math to unify — it is a
# genuine no-op (constrain = u, logjac = 0) and stays a direct read in both
# paths, so it is deliberately NOT in the registry.

@kernel positive_bijector() = begin
    constrain(u::Float64)::Float64 = exp(u)
    inv(constrain, x::Float64)::Float64 = log(x)
    unconstrain(x::Float64)::Float64 = inv(constrain, x)
    logjac(u::Float64)::Float64 = u
end

@kernel unit_bijector() = begin
    constrain(u::Float64)::Float64 = 1 / (1 + exp(-u))
    inv(constrain, x::Float64)::Float64 = log(x) - log1p(-x)
    unconstrain(x::Float64)::Float64 = inv(constrain, x)
    logjac(u::Float64)::Float64 = begin
        x = 1 / (1 + exp(-u))
        log(x) + log1p(-x)
    end
end

"""Transform symbol → bijector kernel object (constrained supports only)."""
const BIJECTORS = Dict{Symbol,Any}(
    :exp => positive_bijector,
    :logistic => unit_bijector,
)

"""Transform symbol → the bijector binding name the generator splices (resolved
in the `PPLGeneratedModels` eval scope)."""
const BIJECTOR_NAMES = Dict{Symbol,Symbol}(
    :exp => :positive_bijector,
    :logistic => :unit_bijector,
)

_bijector_for(transform::Symbol) = get(BIJECTORS, transform) do
    throw(ContractValidationError("[bijector] no bijector for transform $transform"))
end

_bijector_name(transform::Symbol) = get(BIJECTOR_NAMES, transform) do
    throw(ContractValidationError("[bijector] no bijector for transform $transform"))
end

# Cache of prepared scalar endpoints for the host path. Each (transform,
# endpoint) prepares once, then runs the compiled scalar kernel. Lazy —
# populated on first host-side use, never at precompile time.
const _PREPARED_ENDPOINTS = Dict{Tuple{Symbol,Symbol},Any}()

function _prepared_endpoint(transform::Symbol, endpoint::Symbol)
    get!(_PREPARED_ENDPOINTS, (transform, endpoint)) do
        prepare(getproperty(_bijector_for(transform), endpoint))
    end
end
