# Reusable bijector library (Option A — decision 0l3dsru).
#
# Each CONSTRAINED unconstrained→constrained transform is ONE method-bearing
# @kernel exposing `constrain` / `unconstrain` / `logjac` endpoints — a single
# source of truth that BOTH the host path (`constrain`/`unconstrain`/`logjac`
# in layout.jl) and the in-graph generator use. The generator splices the
# `constrain`/`logjac` endpoints (the planner inlines them and demand-prunes),
# and the host runs the same prepared endpoints, so host≡graph agreement is
# STRUCTURAL rather than hand-synced across separate dispatch sites. Adding a
# PARAMETERLESS support = author one @kernel + one `BIJECTORS`/`BIJECTOR_NAMES`
# entry; a PARAMETERIZED support (per-parameter constants, e.g. interval bounds)
# is a method-bearing @kernel with OWNER PORTS, spliced by constructing with the
# literal constants (see `interval_bijector` below).
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

# ── Parameterized bijectors ─────────────────────────────────────────────────
# Some constrained supports carry per-parameter constants (an interval's bounds),
# so they do NOT fit the Symbol-keyed parameterless registry above. They are
# method-bearing @kernel objects with OWNER PORTS (constructor args) and the same
# three endpoints. The generator splices one by constructing it with the literal
# constants — `interval_bijector(lo, hi).constrain(coord)` (the planner inlines
# it exactly like the parameterless splice); the host prepares an endpoint with
# the constants `bound=` in, so the host call is just `k(value)`.

# Bounded interval (lo, hi): affine-logistic ℝ → (lo, hi). `logjac` is written in
# the CONSTRAINED value so host and in-graph forms match bit-for-bit.
@kernel interval_bijector(lo::Float64, hi::Float64) = begin
    constrain(u::Float64)::Float64 = lo + (hi - lo) / (1 + exp(-u))
    inv(constrain, x::Float64)::Float64 = log(x - lo) - log(hi - x)
    unconstrain(x::Float64)::Float64 = inv(constrain, x)
    logjac(u::Float64)::Float64 = begin
        x = lo + (hi - lo) / (1 + exp(-u))
        log(x - lo) + log(hi - x) - log(hi - lo)
    end
end

# Cache of prepared interval endpoints, keyed by (lo, hi, endpoint): the owner
# bounds are `bound=` into the prepared scalar kernel, so the host call is
# `k(value)`. Lazy, like `_prepared_endpoint`.
const _PREPARED_INTERVAL = Dict{Tuple{Float64,Float64,Symbol},Any}()

function _prepared_interval_endpoint(lo::Float64, hi::Float64, endpoint::Symbol)
    get!(_PREPARED_INTERVAL, (lo, hi, endpoint)) do
        prepare(getproperty(interval_bijector, endpoint); bound = (; lo = lo, hi = hi))
    end
end

# Floored positive (lo, ∞): offset-exp ℝ → (lo, ∞) — Stan's lower-bound
# kernel (`real<lower=lo>`, SB `lognormal(0,1; lower=rho_lower)`): the
# Jacobian is the bare exp term (`u`), with NO truncation renormalizer
# (the varying-`tau` precedent). `logjac` reads the UNCONSTRAINED value.
@kernel floored_bijector(lo::Float64) = begin
    constrain(u::Float64)::Float64 = lo + exp(u)
    inv(constrain, x::Float64)::Float64 = log(x - lo)
    unconstrain(x::Float64)::Float64 = inv(constrain, x)
    logjac(u::Float64)::Float64 = u
end

# Cache of prepared floored endpoints, keyed by (lo, endpoint): the owner
# bound is `bound=` into the prepared scalar kernel, so the host call is
# `k(value)`. Lazy, like `_prepared_endpoint`.
const _PREPARED_FLOORED = Dict{Tuple{Float64,Symbol},Any}()

function _prepared_floored_endpoint(lo::Float64, endpoint::Symbol)
    get!(_PREPARED_FLOORED, (lo, endpoint)) do
        prepare(getproperty(floored_bijector, endpoint); bound = (; lo = lo))
    end
end
