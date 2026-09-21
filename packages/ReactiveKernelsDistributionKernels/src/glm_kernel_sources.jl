# Whole-vector GLM distribution objects (Stan's `*_glm` analog).
#
# Each object takes the design matrix and coefficients as owner ports —
# `(X, beta)` with the intercept folded into `X` (the thin layer's
# `design_recipe` output flows in unchanged), plus family params — and
# exposes:
#   - `logpdf(y)` — the summed log-density (the HMC objective;
#     differentiated by generic AD, never a per-function rule);
#   - `pointwise(y)` — per-observation log-densities (LOO etc.);
#   - `score(y)` — dℓ/dη per observation (Stan's theta_derivative);
#   - `eta` — the linear predictor `X * beta` (extractable owner node);
#   - `mu` — the inverse-link fitted mean (extractable owner node);
#   - `working_weights` — IWLS weights `(dμ/dη)²/V(μ)` (extractable owner
#     node; families with clean vector weights only).
# Every endpoint body is explicit broadcast/reduce math over ports — no
# host-function escapes: what you read here is the whole computation, and
# it is what generic AD differentiates. Spelling rules (all measured on
# the nonallocating path; violations silently re-allocate per call):
#   - dotted operators only inside vector expressions (eager vector ops
#     split into allocating fallback steps);
#   - no `Float64.()`/constructor conversions — promote through arithmetic
#     (`y .* eta`, `y .+ phi`, `loggamma.(y .+ 1)`) or `.* 1.0`;
#   - no materialized ranges — lazy ranges stay fused inside consumers
#     (`r .+ (y .- 1) .* nr`); gathering via vector getindex (`cp[y]`,
#     `eta[li]`) reuses caches;
#   - comparisons stay nested inside `ifelse` (a named BitMatrix temp
#     re-allocates); row reductions over `dims` have no in-place step
#     yet, so row sums go through ones-vector matvecs (`es * oc`);
#   - bare callees only (`logistic as _rk_logistic` — the bare name
#     collides with the exported distribution object): qualified
#     `Mod.f` calls bail the decomposer to whole-recipe fallback;
#   - row vectors via `reshape` (never a lazy adjoint: adjoint nesting
#     defeats decomposition while reshape is identical and clean).
# `eta`/`mu`/weights are y-free owner recipes shared structurally across
# have→want queries; data-only temps (lchoose, lgamma) sit over bound
# ports so `bound=` folds them to constants at prepare time. Method
# bodies fuse per-endpoint (see the NOTE in bernoulli_logit_glm): vector
# temps cannot appear inside dotted calls, so cells nest fully to ports
# and `logpdf` sums the shared `pointwise` port (scalar combos where a
# data-only normalizer must stay a named, foldable temp).

const BERNOULLI_LOGIT_GLM_KERNEL_SOURCE = raw"""
using LogExpFunctions: logistic as _rk_logistic

@kernel bernoulli_logit_glm(
        X::Matrix{Float64}, beta::Vector{Float64}) = begin
    eta::Vector{Float64} = X * beta
    mu::Vector{Float64} = _rk_logistic.(eta)
    working_weights::Vector{Float64} = mu .* (1 .- mu)

    # NOTE (method-body lowering): an endpoint body fuses into one op
    # whose dotted calls take ports only — any same-body temp (even a
    # scalar) inside a dotted call drops its edge (UndefVarError at
    # runtime; core fuser bug). What survives: port-only broadcasts,
    # plain `sum(t)` / `sum(f, t)` / scalar combos over named temps,
    # and sibling-endpoint calls. So per-cell math nests fully to ports
    # (the signed margin repeats textually; extra exps per cell), and
    # `logpdf` sums the shared `pointwise` port instead of restating
    # the cells (scalar combos where a data-only normalizer must stay a
    # named, foldable temp).
    logpdf(y::Vector{Int})::Float64 = sum(pointwise(y))
    # Signed-margin Bernoulli in Stan's cutoff-20 form: -log1p(em)
    # interior, Taylor tails past |yt| > 20 (finite at +-Inf). One
    # fused broadcast; the margin nests per branch (see NOTE).
    pointwise(y::Vector{Int})::Vector{Float64} =
        ifelse.(((2 .* y .- 1) .* eta) .> 20.0,
            0.0 .- exp.(0.0 .- ((2 .* y .- 1) .* eta)),
            ifelse.((((2 .* y .- 1) .* eta)) .< -20.0, ((2 .* y .- 1) .* eta),
                0.0 .- log1p.(exp.(0.0 .- ((2 .* y .- 1) .* eta)))))
    # dℓ/dη per cell: s*em/(em+1) interior, s*em / s past the
    # em-magnitude cutoffs (same edges as the value's +-20). Fully
    # nested (even scalar temps drop edges here — see NOTE).
    score(y::Vector{Int})::Vector{Float64} =
        ifelse.(exp.(0.0 .- (2 .* y .- 1) .* eta) .< exp(-20.0),
            (2 .* y .- 1) .* exp.(0.0 .- (2 .* y .- 1) .* eta),
            ifelse.(exp.(0.0 .- (2 .* y .- 1) .* eta) .> exp(20.0),
                (2 .* y .- 1) .* 1.0,
                (2 .* y .- 1) .* exp.(0.0 .- (2 .* y .- 1) .* eta) ./
                    (exp.(0.0 .- (2 .* y .- 1) .* eta) .+ 1.0)))
end
"""

const _BERNOULLI_LOGIT_GLM_BINDINGS = _evaluate_source_bindings(
    BERNOULLI_LOGIT_GLM_KERNEL_SOURCE, (:bernoulli_logit_glm,))
const bernoulli_logit_glm = _BERNOULLI_LOGIT_GLM_BINDINGS[1]

const BERNOULLI_LOGIT_GLM_SOURCE = BERNOULLI_LOGIT_GLM_KERNEL_SOURCE * raw"""

glm_kernel = prepare(bernoulli_logit_glm.logpdf;
    have = (:y, :X, :beta), want = :logpdf)
glm_pointwise = prepare(bernoulli_logit_glm.pointwise;
    have = (:y, :X, :beta), want = :pointwise)
glm_eta = prepare(extract(bernoulli_logit_glm; have = (:X, :beta), want = :eta))

X = [1.0 -1.5; 1.0 -0.5; 1.0 0.5; 1.0 1.5]
beta = [0.2, 0.6]
y = [0, 1, 1, 0]
inputs = (; y, X, beta)
output = glm_kernel(y, X, beta)
pointwise_output = glm_pointwise(y, X, beta)

docs_example = (;
    name = :bernoulli_logit_glm,
    origin = "Bernoulli-logit GLM object: design matrix + coefficients in, summed log-density out (build executed)",
    inputs,
    spec = bernoulli_logit_glm.logpdf,
    kernel = glm_kernel,
    output,
    pointwise = pointwise_output,
)
"""

const POISSON_LOG_GLM_KERNEL_SOURCE = raw"""
using SpecialFunctions: loggamma

@kernel poisson_log_glm(
        X::Matrix{Float64}, beta::Vector{Float64}) = begin
    eta::Vector{Float64} = X * beta
    mu::Vector{Float64} = exp.(eta)
    working_weights::Vector{Float64} = mu

    logpdf(y::Vector{Int})::Float64 = begin
        # Stan's form with no cutoff (Stan has none either): y*eta -
        # exp(eta), minus the data-only lgamma normalizer (bound y
        # folds `cterm` to a constant at prepare time). Scalar combo:
        # each temp is a port-only broadcast (see bernoulli NOTE).
        cterm::Float64 = sum(loggamma.(y .+ 1))
        et::Vector{Float64} = exp.(eta)
        sum(y .* eta) - sum(et) - cterm
    end
    # Port-only leaves throughout (see bernoulli NOTE): the normalizer
    # nests into the one fused cells broadcast.
    pointwise(y::Vector{Int})::Vector{Float64} =
        y .* eta .- exp.(eta) .- loggamma.(y .+ 1)
    score(y::Vector{Int})::Vector{Float64} = y .- exp.(eta)
end
"""

const _POISSON_LOG_GLM_BINDINGS = _evaluate_source_bindings(
    POISSON_LOG_GLM_KERNEL_SOURCE, (:poisson_log_glm,))
const poisson_log_glm = _POISSON_LOG_GLM_BINDINGS[1]

const POISSON_LOG_GLM_SOURCE = POISSON_LOG_GLM_KERNEL_SOURCE * raw"""

glm_kernel = prepare(poisson_log_glm.logpdf;
    have = (:y, :X, :beta), want = :logpdf)

X = [1.0 -1.0; 1.0 0.0; 1.0 1.0]
beta = [0.5, 0.3]
y = [1, 2, 3]
inputs = (; y, X, beta)
output = glm_kernel(y, X, beta)

docs_example = (;
    name = :poisson_log_glm,
    origin = "Poisson-log GLM object: design matrix + coefficients in, summed log-density out (build executed)",
    inputs,
    spec = poisson_log_glm.logpdf,
    kernel = glm_kernel,
    output,
)
"""

# No `working_weights` here: under the identity link they are the constant
# 1/sigma^2, carrying no per-observation information (see the header policy).
const NORMAL_ID_GLM_KERNEL_SOURCE = raw"""
@kernel normal_id_glm(
        X::Matrix{Float64}, beta::Vector{Float64},
        sigma::Float64) = begin
    eta::Vector{Float64} = X * beta
    mu::Vector{Float64} = eta

    # Identity link through the shared `eta` node: -r^2/2σ^2 plus the
    # per-cell -log σ - log2π/2 (no observation count enters). Port-only
    # leaves (see bernoulli NOTE); `logpdf` sums `pointwise`.
    logpdf(y::Vector{Float64})::Float64 = sum(pointwise(y))
    pointwise(y::Vector{Float64})::Vector{Float64} =
        -0.5 .* (y .- eta) .* (y .- eta) .* (1.0 / (sigma * sigma)) .+
            (-log(sigma) - 0.5 * log(2 * π))
    score(y::Vector{Float64})::Vector{Float64} =
        (y .- eta) .* (1.0 / (sigma * sigma))
end
"""

const _NORMAL_ID_GLM_BINDINGS = _evaluate_source_bindings(
    NORMAL_ID_GLM_KERNEL_SOURCE, (:normal_id_glm,))
const normal_id_glm = _NORMAL_ID_GLM_BINDINGS[1]

const NORMAL_ID_GLM_SOURCE = NORMAL_ID_GLM_KERNEL_SOURCE * raw"""

glm_kernel = prepare(normal_id_glm.logpdf;
    have = (:y, :X, :beta, :sigma), want = :logpdf)

X = [1.0 -1.0; 1.0 0.0; 1.0 1.0]
beta = [0.5, 0.3]
sigma = 1.5
y = [0.2, 0.6, 1.1]
inputs = (; y, X, beta, sigma)
output = glm_kernel(y, X, beta, sigma)

docs_example = (;
    name = :normal_id_glm,
    origin = "Normal-id GLM object: design matrix + coefficients in, summed log-density out (build executed)",
    inputs,
    spec = normal_id_glm.logpdf,
    kernel = glm_kernel,
    output,
)
"""

const BINOMIAL_LOGIT_GLM_KERNEL_SOURCE = raw"""
using LogExpFunctions: logistic as _rk_logistic
using SpecialFunctions: loggamma

@kernel binomial_logit_glm(
        X::Matrix{Float64}, beta::Vector{Float64},
        N::Vector{Int}) = begin
    eta::Vector{Float64} = X * beta
    Nf::Vector{Float64} = Float64.(N)
    p::Vector{Float64} = _rk_logistic.(eta)
    mu::Vector{Float64} = Nf .* p
    working_weights::Vector{Float64} = Nf .* p .* (1 .- p)

    logpdf(y::Vector{Int})::Float64 = begin
        # y*eta - N*log1pexp(eta), exact at all eta (no cutoff); the
        # log1pexp twin max(e,0)+log1p(exp(-|e|)) is bit-identical to
        # the library spelling on both sides. Scalar combo of port-only
        # temps (see bernoulli NOTE); bound (y, N) folds `lg`.
        lg::Vector{Float64} = loggamma.(N .+ 1) .-
            loggamma.(y .+ 1) .-
            loggamma.(N .- y .+ 1)
        sum(y .* eta) -
            sum(Nf .* (max.(eta, 0.0) .+ log1p.(exp.(0.0 .- abs.(eta))))) +
            sum(lg)
    end
    pointwise(y::Vector{Int})::Vector{Float64} =
        y .* eta .-
            Nf .* (max.(eta, 0.0) .+ log1p.(exp.(0.0 .- abs.(eta)))) .+
            (loggamma.(N .+ 1) .- loggamma.(y .+ 1) .- loggamma.(N .- y .+ 1))
    score(y::Vector{Int})::Vector{Float64} = y .- Nf ./ (1.0 .+ exp.(0.0 .- eta))
end
"""

const _BINOMIAL_LOGIT_GLM_BINDINGS = _evaluate_source_bindings(
    BINOMIAL_LOGIT_GLM_KERNEL_SOURCE, (:binomial_logit_glm,))
const binomial_logit_glm = _BINOMIAL_LOGIT_GLM_BINDINGS[1]

const BINOMIAL_LOGIT_GLM_SOURCE = BINOMIAL_LOGIT_GLM_KERNEL_SOURCE * raw"""

glm_kernel = prepare(binomial_logit_glm.logpdf;
    have = (:y, :X, :beta, :N), want = :logpdf)

X = [1.0 -1.0; 1.0 0.0; 1.0 1.0]
beta = [0.5, 0.3]
N = [10, 10, 10]
y = [4, 5, 7]
inputs = (; y, X, beta, N)
output = glm_kernel(y, X, beta, N)

docs_example = (;
    name = :binomial_logit_glm,
    origin = "Binomial-logit GLM object: design matrix + coefficients in, summed log-density out (build executed)",
    inputs,
    spec = binomial_logit_glm.logpdf,
    kernel = glm_kernel,
    output,
)
"""

const NEG_BINOMIAL_2_LOG_GLM_KERNEL_SOURCE = raw"""
using SpecialFunctions: loggamma

@kernel neg_binomial_2_log_glm(
        X::Matrix{Float64}, beta::Vector{Float64},
        phi::Float64) = begin
    eta::Vector{Float64} = X * beta
    mu::Vector{Float64} = exp.(eta)
    working_weights::Vector{Float64} = (mu .* phi) ./ (mu .+ phi)

    logpdf(y::Vector{Int})::Float64 = begin
        # NB2 in log-mean form through the shared `eta` node:
        # y*(eta-log phi) - (y+phi)*log1p(mu/phi), plus the lgamma
        # normalizer (bound y folds `lg1`; bound (y, phi) folds `lgp`).
        # Reciprocal (phi/mu+1) forms stay finite at extreme eta.
        # Scalar combo of port-only temps (see bernoulli NOTE).
        lg1::Vector{Float64} = loggamma.(y .+ 1)
        lgp::Vector{Float64} = loggamma.(y .+ phi)
        sum(y .* (eta .- log(phi))) -
            sum((y .+ phi) .* log1p.(exp.(eta) ./ phi)) +
            sum(lgp) - sum(lg1) - length(y) * loggamma(phi)
    end
    pointwise(y::Vector{Int})::Vector{Float64} =
        y .* (eta .- log(phi)) .- (y .+ phi) .* log1p.(exp.(eta) ./ phi) .+
            loggamma.(y .+ phi) .- loggamma.(y .+ 1) .- loggamma(phi)
    score(y::Vector{Int})::Vector{Float64} =
        y .- (y .+ phi) ./ (phi ./ exp.(eta) .+ 1)
end
"""

const _NEG_BINOMIAL_2_LOG_GLM_BINDINGS = _evaluate_source_bindings(
    NEG_BINOMIAL_2_LOG_GLM_KERNEL_SOURCE, (:neg_binomial_2_log_glm,))
const neg_binomial_2_log_glm = _NEG_BINOMIAL_2_LOG_GLM_BINDINGS[1]

const NEG_BINOMIAL_2_LOG_GLM_SOURCE = NEG_BINOMIAL_2_LOG_GLM_KERNEL_SOURCE * raw"""

glm_kernel = prepare(neg_binomial_2_log_glm.logpdf;
    have = (:y, :X, :beta, :phi), want = :logpdf)

X = [1.0 -1.0; 1.0 0.0; 1.0 1.0]
beta = [0.5, 0.3]
phi = 2.0
y = [1, 0, 3]
inputs = (; y, X, beta, phi)
output = glm_kernel(y, X, beta, phi)

docs_example = (;
    name = :neg_binomial_2_log_glm,
    origin = "NegBinomial2-log GLM object: design matrix + coefficients in, summed log-density out (build executed)",
    inputs,
    spec = neg_binomial_2_log_glm.logpdf,
    kernel = glm_kernel,
    output,
)
"""

# No `working_weights` here: multinomial IWLS weights are per-observation
# C×C blocks, not a clean vector (see the header policy).
const CATEGORICAL_LOGIT_GLM_KERNEL_SOURCE = raw"""
@kernel categorical_logit_glm(
        X::Matrix{Float64}, beta::Matrix{Float64},
        alpha::Vector{Float64}) = begin
    Xbeta::Matrix{Float64} = X * beta
    # Row vector via reshape, not alpha': a lazy adjoint nested in the
    # broadcast defeats nonallocating decomposition (4.8MB/call at
    # n=200K); reshape is bitwise-identical and decomposes (160B).
    eta::Matrix{Float64} = Xbeta .+ reshape(alpha, 1, length(alpha))
    m::Matrix{Float64} = maximum(eta, dims = 2)
    es::Matrix{Float64} = exp.(eta .- m)
    # Row sums through a ones-vector matvec: row reductions over `dims`
    # have no in-place step yet (a `sum(es, dims = 2)` here would pay one
    # fallback temp per call — see bernoulli NOTE; `maximum` above still
    # pays it, unavoidable for dynamic class counts).
    s::Vector{Float64} = es * ones(size(eta, 2))
    mu::Matrix{Float64} = es ./ (s .* ones(1, size(eta, 2)))

    # Stable rowwise softmax cross-entropy off the shared `eta`/`m`/`s`
    # owner nodes: eta[i,yi] - max - log(sum(exp)) (see bernoulli NOTE:
    # everything nests to ports). The observed logits gather by linear
    # index (ranges stay lazy inside the consumer — never materialized).
    logpdf(y::Vector{Int})::Float64 = sum(pointwise(y))
    pointwise(y::Vector{Int})::Vector{Float64} =
        eta[(1:length(y)) .+ (y .- 1) .* size(eta, 1)] .- vec(m) .- log.(s)
    # dℓ/deta per cell: 1[observed] - softmax. The indicator matrix
    # compares Int codes against a lazy class range; the comparison
    # stays nested in `ifelse` (a named BitMatrix temp re-allocates).
    score(y::Vector{Int})::Matrix{Float64} =
        ifelse.((y .+ zeros(1, size(eta, 2))) .==
            ((y .* 0) .+ reshape(1:size(eta, 2), 1, size(eta, 2))),
            1.0, 0.0) .- mu
end
"""

const _CATEGORICAL_LOGIT_GLM_BINDINGS = _evaluate_source_bindings(
    CATEGORICAL_LOGIT_GLM_KERNEL_SOURCE, (:categorical_logit_glm,))
const categorical_logit_glm = _CATEGORICAL_LOGIT_GLM_BINDINGS[1]

const CATEGORICAL_LOGIT_GLM_SOURCE = CATEGORICAL_LOGIT_GLM_KERNEL_SOURCE * raw"""

glm_kernel = prepare(categorical_logit_glm.logpdf;
    have = (:y, :X, :beta, :alpha), want = :logpdf)

X = [1.0 -1.0; 1.0 0.0; 1.0 1.0]
beta = [0.5 0.0 -0.4; 0.3 -0.2 0.1]
alpha = [0.1, -0.1, 0.2]
y = [1, 3, 2]
inputs = (; y, X, beta, alpha)
output = glm_kernel(y, X, beta, alpha)

docs_example = (;
    name = :categorical_logit_glm,
    origin = "Categorical-logit GLM object: design matrix + coefficients in, summed log-density out (build executed)",
    inputs,
    spec = categorical_logit_glm.logpdf,
    kernel = glm_kernel,
    output,
)
"""

const ORDERED_LOGISTIC_GLM_KERNEL_SOURCE = raw"""
using LogExpFunctions: logistic as _rk_logistic

@kernel ordered_logistic_glm(
        X::Matrix{Float64}, beta::Vector{Float64},
        cuts::Vector{Float64}) = begin
    eta::Vector{Float64} = X * beta
    etac::Matrix{Float64} = reshape(eta, length(eta), 1)
    G::Matrix{Float64} = _rk_logistic.(etac .- reshape(cuts, 1, length(cuts)))
    Gs::Matrix{Float64} = sum(G, dims = 2)
    Gv::Vector{Float64} = vec(Gs)
    mu::Vector{Float64} = 1 .+ Gv
    # IWLS weights (dE[y]/deta)^2/Var(y) via the survival sums E[y] =
    # 1 + sum S_k, E[y^2] = 1 + sum (2k+1) S_k with S_k = P(Y > k).
    # Zero variance (all mass on one class) maps to weight 0 (no
    # information) rather than NaN. Class sums go through ones-vector
    # matvecs; the (2k+1) row fuses a lazy class range (never
    # materialized).
    K::Int = length(cuts)
    rK::UnitRange{Int} = 1:K
    # Row vectors via reshape, not adjoints: lazy-adjoint nesting
    # defeats nonallocating decomposition (measured on categorical
    # eta: 4.8MB/call); reshape is identical and decomposes.
    Sw::Matrix{Float64} = _rk_logistic.(etac .- reshape(cuts, 1, K))
    ocw::Vector{Float64} = ones(K)
    m1::Vector{Float64} = 1.0 .+ (Sw * ocw)
    m2::Vector{Float64} = 1.0 .+ (((2.0 .* reshape(rK, 1, K) .+ 1.0) .* Sw) * ocw)
    dm::Vector{Float64} = (Sw .* (1.0 .- Sw)) * ocw
    Vw::Vector{Float64} = m2 .- m1 .* m1
    working_weights::Vector{Float64} = ifelse.(Vw .> 0.0, dm .* dm ./ Vw, 0.0)

    # Cumulative-logit cells log(F(hi-e)-F(lo-e)), `logpdf` sums
    # `pointwise` (see bernoulli NOTE: everything nests to ports). Edge
    # classes branch to closed one-sided forms — never logistic(+-Inf):
    # its reverse is 0*Inf = NaN even at benign points, while the value
    # is fine. Probability-space throughout (log F / log(1-F), bitwise
    # the oracle): past saturation the value underflows to -Inf exactly
    # as specified, and the score stays ideal-exact. All gathers are
    # vector gathers with arithmetic index clamps (branches evaluate
    # eagerly, so indices must stay in bounds). The interior log takes
    # a +1e-300 floor on edge rows only: Enzyme computes every branch
    # reverse eagerly, and the clamped 0/0 there (0 cotangent over a 0
    # difference) NaNs through 0*NaN selection — the floor makes it
    # 0/1e-300 = 0 while taken rows keep bitwise-exact log(diff).
    logpdf(y::Vector{Int})::Float64 = sum(pointwise(y))
    pointwise(y::Vector{Int})::Vector{Float64} =
        ifelse.(y .== 1, log.(_rk_logistic.(cuts[y .^ 0] .- eta)),
            ifelse.(y .== (length(cuts) + 1),
                log.(1.0 .-
                    _rk_logistic.(cuts[(y .^ 0) .* length(cuts)] .- eta)),
                log.(_rk_logistic.(cuts[y .- (y .== (length(cuts) + 1))] .-
                    eta) .-
                    _rk_logistic.(cuts[y .- 1 .+ (y .== 1)] .- eta) .+
                    (((y .== 1) .| (y .== (length(cuts) + 1))) .* 1e-300))))
    # The eta adjoint collapses exactly: (a(1-a)-b(1-b))/(b-a) =
    # a + b - 1 (no division, no cutoff, exact at all eta).
    score(y::Vector{Int})::Vector{Float64} =
        _rk_logistic.(([-Inf; cuts; Inf])[y] .- eta) .+
            _rk_logistic.(([-Inf; cuts; Inf])[y .+ 1] .- eta) .- 1.0
end
"""

const _ORDERED_LOGISTIC_GLM_BINDINGS = _evaluate_source_bindings(
    ORDERED_LOGISTIC_GLM_KERNEL_SOURCE, (:ordered_logistic_glm,))
const ordered_logistic_glm = _ORDERED_LOGISTIC_GLM_BINDINGS[1]

const ORDERED_LOGISTIC_GLM_SOURCE = ORDERED_LOGISTIC_GLM_KERNEL_SOURCE * raw"""

glm_kernel = prepare(ordered_logistic_glm.logpdf;
    have = (:y, :X, :beta, :cuts), want = :logpdf)

X = [1.0 -1.0; 1.0 0.0; 1.0 1.0]
beta = [0.5, 0.3]
cuts = [-0.5, 0.5]
y = [1, 2, 3]
inputs = (; y, X, beta, cuts)
output = glm_kernel(y, X, beta, cuts)

docs_example = (;
    name = :ordered_logistic_glm,
    origin = "Ordered-logistic GLM object: design matrix + coefficients in, summed log-density out (build executed)",
    inputs,
    spec = ordered_logistic_glm.logpdf,
    kernel = glm_kernel,
    output,
)
"""
