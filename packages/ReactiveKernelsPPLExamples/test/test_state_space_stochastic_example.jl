using ReactiveKernelsPPLExamples.StateSpaceStochasticExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, student_t
using SpecialFunctions: loggamma

# Graph-independent reference oracle: recomputes the state-space density from
# first principles, matching the Stan model block (propto = false, jacobian =
# true). The seasonal transition is computed DIRECTLY from the Stan form
# (seasonal[t] ~ Normal(-Σ seasonal[t-11:t-1], σ₁)), independent of the graph's
# trailing-window matvec.
_ss_normal(x, mu, sigma) = -0.5 * log(2π) - log(sigma) - 0.5 * ((x - mu) / sigma)^2
_ss_student_t(x, nu, mu, sigma) =
    loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * log(nu * π) - log(sigma) -
    ((nu + 1) / 2) * log1p(((x - mu) / sigma)^2 / nu)
_ss_logistic(z) = 1 / (1 + exp(-z))

function _ss_reference(q, y, x, w)
    n = length(y)
    ybar = sum(y) / n
    ysd = sqrt(sum((y .- ybar) .^ 2) / (n - 1))   # Stan sd uses n-1
    L = ybar - 3 * ysd; U = ybar + 3 * ysd
    mu_unc = q[1:n]
    seasonal = q[(n + 1):(2 * n)]
    beta = q[2 * n + 1]; lambda = q[2 * n + 2]
    z1 = q[2 * n + 3]; z2 = q[2 * n + 4]; z3 = q[2 * n + 5]
    mu = L .+ (U - L) .* _ss_logistic.(mu_unc)
    mu_jac = sum(log(U - L) + log(_ss_logistic(z)) + log(_ss_logistic(-z)) for z in mu_unc)
    sigma1 = exp(z1); sigma2 = sigma1 + exp(z2); sigma3 = sigma2 + exp(z3)
    sigma_jac = z1 + z2 + z3
    log_jacobian = mu_jac + sigma_jac
    yhat = mu .+ beta .* x .+ lambda .* w
    level_lp = sum(_ss_normal(mu[t], mu[t - 1], sigma2) for t in 2:n)
    seasonal_lp = sum(_ss_normal(seasonal[t], -sum(seasonal[(t - 11):(t - 1)]), sigma1)
                      for t in 12:n)
    obs_lp = sum(_ss_normal(y[i], yhat[i] + seasonal[i], sigma3) for i in 1:n)
    sigma_prior = _ss_student_t(sigma1, 4, 0, 1) + _ss_student_t(sigma2, 4, 0, 1) +
                  _ss_student_t(sigma3, 4, 0, 1)
    (; level_lp, seasonal_lp, obs_lp, sigma_prior, log_jacobian,
       posterior = level_lp + seasonal_lp + obs_lp + sigma_prior + log_jacobian)
end

@testset "PPL graph — state_space_stochastic" begin
    artifact = evaluate_state_space_stochastic_source()
    @test artifact.source == strip(STATE_SPACE_STOCHASTIC_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    have = (:unconstrained, :y, :x, :w)
    want = (:level_lp, :seasonal_lp, :obs_lp, :sigma_prior, :log_jacobian, :posterior)
    data = (STATE_SPACE_Y, STATE_SPACE_X, STATE_SPACE_W)
    n = length(STATE_SPACE_Y)
    dim = 2 * n + 5

    @testset "authored on the reusable distribution-object surface" begin
        @test occursin("exp.(-log1pexp.(-mu_unc))", STATE_SPACE_STOCHASTIC_SOURCE)
        @test occursin("sigma2::Float64 = sigma1 + exp(sigma_unc2)", STATE_SPACE_STOCHASTIC_SOURCE)
        @test occursin("student_t(4.0, 0.0, 1.0).logpdf", STATE_SPACE_STOCHASTIC_SOURCE)
        @test occursin("window * seasonal", STATE_SPACE_STOCHASTIC_SOURCE)
        @test occursin("plate(mu[2:n]", STATE_SPACE_STOCHASTIC_SOURCE)
        @test !occursin("struct ", STATE_SPACE_STOCHASTIC_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.student_t_object === student_t
        raw_generated = code_expr(artifact.kernel)
        readable = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false,
        )
        @test !occursin(r"__ops__\[\d+\]", readable)
        @test !occursin(r"\boperation\(", readable)
    end

    @testset "density decomposition vs the independent reference oracle" begin
        kernel = prepare(model; have = have, want = want)
        for q in ([zeros(dim)],
                  [0.01 .* collect(1:dim) .- 0.05],
                  [0.2 .* sin.(0.3 .* (1:dim))])
            qi = q[1]
            level_lp, seasonal_lp, obs_lp, sigma_prior, log_jacobian, posterior =
                kernel(qi, data...)
            ref = _ss_reference(qi, data...)
            @test isfinite(posterior)
            @test level_lp ≈ ref.level_lp
            @test seasonal_lp ≈ ref.seasonal_lp
            @test obs_lp ≈ ref.obs_lp
            @test sigma_prior ≈ ref.sigma_prior
            @test log_jacobian ≈ ref.log_jacobian
            @test posterior ≈ ref.posterior
        end
    end

    @testset "the trailing-window matvec matches the direct seasonal transition" begin
        # Cross-check the graph's seasonal_lp (a banded matvec) against the direct
        # sliding-window Stan form recomputed in the oracle.
        q = 0.1 .* collect(range(-1.0, 1.0; length = dim))
        seasonal_lp = prepare(model; have = have, want = :seasonal_lp)(q, data...)
        @test seasonal_lp ≈ _ss_reference(q, data...).seasonal_lp
    end

    @testset "wrong behavior is detectable — perturbing q moves the density" begin
        kernel = prepare(model; have = have, want = :posterior)
        q = zeros(dim)
        q2 = copy(q); q2[2 * n + 3] = 0.9   # bump the first log-scale
        @test kernel(q, data...) != kernel(q2, data...)
    end

    @testset "data-generic: a shorter series still matches the reference" begin
        m = 15
        yy = collect(range(-1.0, 2.0; length = m))
        xx = collect(range(0.0, 1.0; length = m))
        ww = cos.(0.5 .* (1:m))
        small_dim = 2 * m + 5
        q = 0.1 .* collect(range(-1.0, 1.0; length = small_dim))
        post = prepare(model; have = have, want = :posterior)(q, yy, xx, ww)
        @test post ≈ _ss_reference(q, yy, xx, ww).posterior
    end
end
