module Election88FullExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export E88_AGE, E88_EDU, E88_AGE_EDU, E88_STATE, E88_REGION
export E88_BLACK, E88_FEMALE, E88_VPREV, E88_Y
export E88_N_AGE, E88_N_EDU, E88_N_AGE_EDU, E88_N_STATE, E88_N_REGION
export build_election88_full_graph, demo
export ELECTION88_FULL_SOURCE, evaluate_election88_full_source

# posteriordb `election88-election88_full` — the Gelman & Hill multilevel
# logistic regression for the 1988 US presidential vote (ARM ch. 14). Five
# varying-intercept grouping factors — age (`a`), education (`b`), age×education
# (`c`), state (`d`), region (`e`) — each `~ Normal(0, sigma_·)`, plus a
# `vector[5] beta ~ Normal(0, 100)` of fixed effects and five scale
# hyperparameters `sigma_· ∈ [0, 100]`. The outcome is
# `y ~ bernoulli_logit(y_hat)` with the linear predictor combining the fixed
# effects and the five gathered group intercepts.
#
# The full dataset is N = 11566 respondents. The group DIMENSIONS are fixed
# properties of the model (n_age = 4, n_edu = 4, n_age_edu = 16, n_state = 51,
# n_region_full = 5) regardless of how many rows are used, and the parameter
# vectors a..e are sized by those dimensions and receive their per-group
# `Normal(0, sigma_·)` prior whether or not a group is observed (2 of the 51
# states and region 5 are unobserved even in the full data). So a faithful,
# self-contained subset keeps the real group counts and embeds a representative
# 60-row systematic sample of the respondents — every index stays in 1..n_·,
# the parameter dimension D = 90 is exactly the real model's, and no index
# remapping is needed. The kernel itself is data-generic: it reads the group
# sizes as (bound) ports, so the same graph re-gates against the full raw data.
const E88_AGE = [
    2, 3, 1, 3, 2, 2, 1, 2, 3, 3, 4, 2, 2, 3, 1, 1, 2, 1, 2, 2, 2, 2, 4, 2, 3, 3, 3, 1, 2, 1, 4, 1, 3, 2, 1, 4, 3, 3, 3, 2, 2, 3, 4, 4, 3, 1, 3, 2, 1, 2, 1, 1, 3, 2, 1, 1, 1, 2, 3, 3,
]
const E88_EDU = [
    2, 4, 2, 2, 3, 2, 3, 3, 2, 4, 2, 2, 4, 3, 2, 3, 2, 3, 4, 4, 2, 3, 2, 2, 4, 1, 4, 4, 3, 2, 1, 2, 3, 2, 2, 1, 3, 4, 1, 2, 4, 2, 3, 3, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 2, 1, 3, 2, 3, 3,
]
const E88_AGE_EDU = [
    6, 12, 2, 10, 7, 6, 3, 7, 10, 12, 14, 6, 8, 11, 2, 3, 6, 3, 8, 8, 6, 7, 14, 6, 12, 9, 12, 4, 7, 2, 13, 2, 11, 6, 2, 13, 11, 12, 9, 6, 8, 10, 15, 15, 9, 2, 10, 6, 2, 6, 3, 3, 12, 8, 2, 1, 3, 6, 11, 11,
]
const E88_STATE = [
    7, 36, 14, 11, 3, 33, 50, 48, 23, 23, 39, 24, 39, 1, 50, 23, 33, 10, 33, 35, 41, 4, 33, 36, 5, 29, 44, 10, 47, 38, 19, 36, 39, 32, 11, 19, 18, 3, 24, 47, 5, 33, 21, 5, 43, 22, 24, 5, 5, 19, 24, 43, 47, 41, 31, 5, 15, 36, 3, 5,
]
const E88_REGION = [
    1, 2, 2, 3, 4, 1, 2, 4, 2, 2, 1, 2, 1, 3, 2, 2, 1, 3, 1, 2, 3, 3, 1, 2, 4, 4, 3, 3, 3, 4, 3, 2, 1, 4, 3, 3, 3, 4, 2, 3, 4, 1, 1, 4, 3, 1, 2, 4, 4, 3, 2, 3, 3, 3, 1, 4, 2, 2, 4, 4,
]
const E88_BLACK = Float64[
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
]
const E88_FEMALE = Float64[
    1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
]
const E88_VPREV = Float64[
    0.564, 0.5493333, 0.5336667, 0.4513333, 0.6463333, 0.511, 0.521, 0.5523333, 0.5546667, 0.5546667, 0.5213333, 0.4713334, 0.5213333, 0.5173333, 0.521, 0.5546667, 0.511, 0.5696666, 0.511, 0.6323333, 0.5253333, 0.4883333, 0.511, 0.5493333, 0.562, 0.6316667, 0.5613334, 0.5696666, 0.567, 0.5393333, 0.538, 0.5493333, 0.5213333, 0.5713333, 0.4513333, 0.538, 0.5243334, 0.6463333, 0.4713334, 0.567, 0.562, 0.511, 0.494, 0.562, 0.5063334, 0.478, 0.4713334, 0.562, 0.562, 0.538, 0.4713334, 0.5063334, 0.567, 0.5253333, 0.5633333, 0.562, 0.5863333, 0.5493333, 0.6463333, 0.562,
]
const E88_Y = Bool[
    1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0,
]

# Group dimensions — fixed properties of the election88_full model, identical
# for the full data and this subset (the two unobserved states and region 5 are
# retained so a..e keep their true lengths and priors).
const E88_N_AGE = 4
const E88_N_EDU = 4
const E88_N_AGE_EDU = 16
const E88_N_STATE = 51
const E88_N_REGION = 5

const ELECTION88_FULL_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, bernoulli
using LogExpFunctions: logistic, log1pexp

@kernel model(unconstrained::Vector{Float64},
              age::Vector{Int}, edu::Vector{Int}, age_edu::Vector{Int},
              state::Vector{Int}, region_full::Vector{Int},
              black::Vector{Float64}, female::Vector{Float64},
              v_prev_full::Vector{Float64}, y::Vector{Bool},
              n_age::Int, n_edu::Int, n_age_edu::Int,
              n_state::Int, n_region_full::Int) = begin
    # Stan's declared unconstrained order:
    #   a[n_age], b[n_edu], c[n_age_edu], d[n_state], e[n_region_full],
    #   beta[5], u_sigma_a, u_sigma_b, u_sigma_c, u_sigma_d, u_sigma_e.
    # dim = n_age + n_edu + n_age_edu + n_state + n_region_full + 5 + 5. The
    # group sizes ride as bound integer ports (like the gather indices), so the
    # slice bounds below are concrete and the same prepared kernel stays
    # traceable as a Reactant tensor program. The group vectors a..e and the
    # fixed effects beta are unconstrained (identity transform, 0 Jacobian); only
    # the five scale hyperparameters transform.
    ob::Int = n_age
    oc::Int = ob + n_edu
    od::Int = oc + n_age_edu
    oe::Int = od + n_state
    obeta::Int = oe + n_region_full
    osig::Int = obeta + 5

    a::AbstractVector{Float64} = view(unconstrained, 1:ob)
    b::AbstractVector{Float64} = view(unconstrained, ob + 1:oc)
    c::AbstractVector{Float64} = view(unconstrained, oc + 1:od)
    d::AbstractVector{Float64} = view(unconstrained, od + 1:oe)
    e::AbstractVector{Float64} = view(unconstrained, oe + 1:obeta)
    beta_vec::AbstractVector{Float64} = view(unconstrained, obeta + 1:osig)
    beta1::Float64 = sum(view(unconstrained, obeta + 1:obeta + 1))
    beta2::Float64 = sum(view(unconstrained, obeta + 2:obeta + 2))
    beta3::Float64 = sum(view(unconstrained, obeta + 3:obeta + 3))
    beta4::Float64 = sum(view(unconstrained, obeta + 4:obeta + 4))
    beta5::Float64 = sum(view(unconstrained, obeta + 5:obeta + 5))
    u_sigma_a::Float64 = sum(view(unconstrained, osig + 1:osig + 1))
    u_sigma_b::Float64 = sum(view(unconstrained, osig + 2:osig + 2))
    u_sigma_c::Float64 = sum(view(unconstrained, osig + 3:osig + 3))
    u_sigma_d::Float64 = sum(view(unconstrained, osig + 4:osig + 4))
    u_sigma_e::Float64 = sum(view(unconstrained, osig + 5:osig + 5))

    # `real<lower=0, upper=100> sigma_·` → scaled-logit interval transform
    # sigma = 100·logistic(u); the change-of-variables Jacobian
    # log|dsigma/du| = log(100) - log1pexp(-u) - log1pexp(u) is Stan's
    # `lub_constrain`. The sigmas carry no `~` statement (implicit uniform over
    # the [0,100] box), so their prior density is a dropped constant and the
    # only sigma contribution is this Jacobian.
    sigma_a::Float64 = 100.0 * logistic(u_sigma_a)
    sigma_b::Float64 = 100.0 * logistic(u_sigma_b)
    sigma_c::Float64 = 100.0 * logistic(u_sigma_c)
    sigma_d::Float64 = 100.0 * logistic(u_sigma_d)
    sigma_e::Float64 = 100.0 * logistic(u_sigma_e)
    jac_a::Float64 = log(100.0) - log1pexp(-u_sigma_a) - log1pexp(u_sigma_a)
    jac_b::Float64 = log(100.0) - log1pexp(-u_sigma_b) - log1pexp(u_sigma_b)
    jac_c::Float64 = log(100.0) - log1pexp(-u_sigma_c) - log1pexp(u_sigma_c)
    jac_d::Float64 = log(100.0) - log1pexp(-u_sigma_d) - log1pexp(u_sigma_d)
    jac_e::Float64 = log(100.0) - log1pexp(-u_sigma_e) - log1pexp(u_sigma_e)

    parameters = (; a, b, c, d, e, beta1, beta2, beta3, beta4, beta5,
                  sigma_a, sigma_b, sigma_c, sigma_d, sigma_e)
    (parameters, log_jacobian::Float64) =
        ((; a, b, c, d, e, beta1, beta2, beta3, beta4, beta5,
          sigma_a, sigma_b, sigma_c, sigma_d, sigma_e),
         jac_a + jac_b + jac_c + jac_d + jac_e)
    (a::AbstractVector{Float64}, b::AbstractVector{Float64},
     c::AbstractVector{Float64}, d::AbstractVector{Float64},
     e::AbstractVector{Float64}, beta1::Float64, beta2::Float64,
     beta3::Float64, beta4::Float64, beta5::Float64,
     sigma_a::Float64, sigma_b::Float64, sigma_c::Float64,
     sigma_d::Float64, sigma_e::Float64) =
        (parameters.a, parameters.b, parameters.c, parameters.d, parameters.e,
         parameters.beta1, parameters.beta2, parameters.beta3, parameters.beta4,
         parameters.beta5, parameters.sigma_a, parameters.sigma_b,
         parameters.sigma_c, parameters.sigma_d, parameters.sigma_e)

    # Varying-intercept priors (proper, so they appear in the density and its
    # gradient): a_k ~ Normal(0, sigma_a), ... The shared scale rides each plate
    # as a scalar arg (broadcast across cells).
    a_pointwise = plate(a, sigma_a) do x, s
        normal(0.0, s).logpdf(x)
    end
    b_pointwise = plate(b, sigma_b) do x, s
        normal(0.0, s).logpdf(x)
    end
    c_pointwise = plate(c, sigma_c) do x, s
        normal(0.0, s).logpdf(x)
    end
    d_pointwise = plate(d, sigma_d) do x, s
        normal(0.0, s).logpdf(x)
    end
    e_pointwise = plate(e, sigma_e) do x, s
        normal(0.0, s).logpdf(x)
    end
    # Fixed-effect prior: beta_j ~ Normal(0, 100).
    beta_pointwise = plate(beta_vec) do x
        normal(0.0, 100.0).logpdf(x)
    end
    log_prior::Float64 = sum(a_pointwise) + sum(b_pointwise) + sum(c_pointwise) +
                         sum(d_pointwise) + sum(e_pointwise) + sum(beta_pointwise)

    # Hierarchical integer-array GATHERS, done OUTSIDE any plate: each varying
    # intercept is gathered by its concrete data index into a length-N vector.
    # A traced integer index does not lower, so the docs_example binds the five
    # index vectors at preparation; then each gather traces as a concrete gather
    # and the Reactant path sees only the float inputs + the parameter vector.
    mu_a = a[age]
    mu_b = b[edu]
    mu_c = c[age_edu]
    mu_d = d[state]
    mu_e = e[region_full]

    # Transformed parameter: the logit-scale linear predictor
    #   y_hat_i = beta1 + beta2·black_i + beta3·female_i + beta5·female_i·black_i
    #             + beta4·v_prev_full_i + a[age_i] + b[edu_i] + c[age_edu_i]
    #             + d[state_i] + e[region_full_i].
    # beta1..beta5 ride the plate as shared scalar args; the gathered group
    # vectors enter as ordinary length-N inputs.
    y_hat = plate(black, female, v_prev_full, mu_a, mu_b, mu_c, mu_d, mu_e,
                  beta1, beta2, beta3, beta4, beta5) do bl, fe, vp, ma, mb, mc, md, me, b1, b2, b3, b4, b5
        b1 + b2 * bl + b3 * fe + b5 * fe * bl + b4 * vp + ma + mb + mc + md + me
    end

    # Likelihood: y_i ~ bernoulli_logit(y_hat_i). The linear predictor is
    # recomputed inline (one per-cell expression, no intermediate local before
    # the nested endpoint call); the Bernoulli endpoint takes the success
    # probability, so the logit link is `logistic(·)` applied inline.
    pointwise = plate(y, black, female, v_prev_full, mu_a, mu_b, mu_c, mu_d, mu_e,
                      beta1, beta2, beta3, beta4, beta5) do yi, bl, fe, vp, ma, mb, mc, md, me, b1, b2, b3, b4, b5
        bernoulli(logistic(b1 + b2 * bl + b3 * fe + b5 * fe * bl + b4 * vp + ma + mb + mc + md + me)).logpdf(yi)
    end
    likelihood::Float64 = sum(pointwise)

    constrained_logdensity::Float64 = log_prior + likelihood
    unconstrained_prior::Float64 = log_prior + log_jacobian
    posterior::Float64 = constrained_logdensity + log_jacobian

    # Generated quantity: the modeled Republican-vote probabilities
    # p = inv_logit(y_hat).
    p = plate(y_hat) do yh
        logistic(yh)
    end

    return posterior
end

q = vcat(
    fill(0.05, E88_N_AGE),
    fill(0.05, E88_N_EDU),
    fill(0.05, E88_N_AGE_EDU),
    fill(0.05, E88_N_STATE),
    fill(0.05, E88_N_REGION),
    [0.3, -0.2, 0.1, 0.4, -0.1],
    [-1.0, -1.0, -1.0, -1.0, -1.0],
)
age = E88_AGE
edu = E88_EDU
age_edu = E88_AGE_EDU
state = E88_STATE
region_full = E88_REGION
black = E88_BLACK
female = E88_FEMALE
v_prev_full = E88_VPREV
y = E88_Y
n_age = E88_N_AGE
n_edu = E88_N_EDU
n_age_edu = E88_N_AGE_EDU
n_state = E88_N_STATE
n_region_full = E88_N_REGION

requested_nodes = (:parameters, :log_jacobian, :log_prior, :likelihood, :posterior)
# Bind the integer gathers and group sizes so `a.inputs` excludes them and the
# Reactant path traces only the float inputs + the parameter vector.
density_kernel = prepare(model;
    have = (:unconstrained, :age, :edu, :age_edu, :state, :region_full,
            :black, :female, :v_prev_full, :y,
            :n_age, :n_edu, :n_age_edu, :n_state, :n_region_full),
    want = requested_nodes,
    bound = (; age, edu, age_edu, state, region_full,
             n_age, n_edu, n_age_edu, n_state, n_region_full))

output = density_kernel(q, black, female, v_prev_full, y)
parameters, log_jacobian, log_prior, likelihood, posterior = output
@assert posterior ≈ log_prior + likelihood + log_jacobian

docs_example = (;
    name = :election88_full_posterior,
    origin = "posteriordb election88_full — multilevel logistic regression (5 varying intercepts)",
    inputs = (; q, black, female, v_prev_full, y),
    model,
    kernel = density_kernel,
    output,
    requested_nodes,
    normal_object = normal,
    bernoulli_object = bernoulli,
)
"""

function evaluate_election88_full_source()
    _evaluate_ppl_source(ELECTION88_FULL_SOURCE, @__MODULE__; bindings = (
        :E88_AGE, :E88_EDU, :E88_AGE_EDU, :E88_STATE, :E88_REGION,
        :E88_BLACK, :E88_FEMALE, :E88_VPREV, :E88_Y,
        :E88_N_AGE, :E88_N_EDU, :E88_N_AGE_EDU, :E88_N_STATE, :E88_N_REGION,
    ))
end

const _ELECTION88_FULL_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _ELECTION88_FULL_GRAPH_TEMPLATE[] = evaluate_election88_full_source().model
    nothing
end

"""
    build_election88_full_graph()

Build the posteriordb `election88_full` model (a multilevel logistic regression
with five varying-intercept grouping factors) as a declarative
`ReactiveKernels.KernelSpec`. The five scale hyperparameters `sigma_·` use the
exact scaled-logit interval (`lub_constrain`) transform and its Jacobian; the
varying-intercept priors `a..e ~ Normal(0, sigma_·)` and the fixed-effect prior
`beta ~ Normal(0, 100)` reuse the shared Normal endpoint, and each varying
intercept is gathered by its concrete integer index outside any plate. The
transform Jacobian, log prior, transformed-parameter `y_hat`, pointwise
log-likelihood, likelihood reduction, densities, posterior, and the
generated-quantity probabilities `p` are separate named nodes.
"""
function build_election88_full_graph()
    compose(_ELECTION88_FULL_GRAPH_TEMPLATE[])
end

function demo()
    model = build_election88_full_graph()
    q = vcat(
        fill(0.05, E88_N_AGE), fill(0.05, E88_N_EDU), fill(0.05, E88_N_AGE_EDU),
        fill(0.05, E88_N_STATE), fill(0.05, E88_N_REGION),
        [0.3, -0.2, 0.1, 0.4, -0.1], [-1.0, -1.0, -1.0, -1.0, -1.0],
    )
    posterior_plan = plan(model;
        have = (:unconstrained, :age, :edu, :age_edu, :state, :region_full,
                :black, :female, :v_prev_full, :y,
                :n_age, :n_edu, :n_age_edu, :n_state, :n_region_full),
        want = (:log_prior, :log_jacobian, :likelihood, :posterior))
    println(explain(posterior_plan))
    log_prior, log_jacobian, likelihood, posterior =
        prepare(posterior_plan)(q, E88_AGE, E88_EDU, E88_AGE_EDU, E88_STATE,
            E88_REGION, E88_BLACK, E88_FEMALE, E88_VPREV, E88_Y,
            E88_N_AGE, E88_N_EDU, E88_N_AGE_EDU, E88_N_STATE, E88_N_REGION)
    println("log_prior + logJ + likelihood = ", log_prior, " + ", log_jacobian,
            " + ", likelihood, " = ", posterior)
    nothing
end

end # module Election88FullExample

if abspath(PROGRAM_FILE) == @__FILE__
    Election88FullExample.demo()
end
