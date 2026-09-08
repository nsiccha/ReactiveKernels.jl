using ReactiveKernelsPPLExamples.DogsNonhierarchicalExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, bernoulli
using LogExpFunctions: log1pexp

# Graph-independent reference oracle for the posteriordb dogs_nonhierarchical
# model: the exp transform for sigma, the cholesky_factor_corr[2] transform from
# one unconstrained value (tanh) with its Jacobian and the analytic K=2 LKJ
# density, the non-centered per-dog logit rates, and the multiplicative
# Bernoulli likelihood over the running-count design.
_dnh_normal_ld(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
_dnh_logistic_ld(x) = -x - 2 * log1pexp(-x)   # standard Logistic(0,1) log-density

function _dnh_reference(q)
    J = DOGS_NH_J
    T = DOGS_NH_T
    mu1, mu2 = q[1], q[2]
    sigma1, sigma2 = exp(q[3]), exp(q[4])
    w = q[5]
    z1 = q[6:(5 + J)]
    z2 = q[(6 + J):(5 + 2J)]
    L21 = tanh(w)
    L22 = sqrt(1 - L21^2)
    jac_L = log(1 - L21^2)
    log_jacobian = q[3] + q[4] + jac_L
    logit_a = mu1 .+ z1 .* sigma1 .+ z2 .* (sigma2 * L21)
    logit_b = mu2 .+ z2 .* (sigma2 * L22)
    log_a = -log1pexp.(-logit_a)
    log_b = -log1pexp.(-logit_b)
    mu_prior = _dnh_logistic_ld(mu1) + _dnh_logistic_ld(mu2)
    sigma_prior = _dnh_normal_ld(sigma1, 0.0, 1.0) + _dnh_normal_ld(sigma2, 0.0, 1.0)
    lkj_prior = 2 * log(L22) - log(4.0 / 3.0)
    z_prior = sum(_dnh_normal_ld(zj, 0.0, 1.0) for zj in vcat(z1, z2))
    log_prior = mu_prior + sigma_prior + lkj_prior + z_prior
    y = DOGS_NH_Y
    likelihood = 0.0
    for j in 1:J
        ps = 0.0
        pa = 0.0
        for t in 1:T
            lp = ps * log_a[j] + pa * log_b[j]
            p = exp(lp)
            likelihood += y[j, t] ? log(p) : log1p(-p)
            ps += y[j, t] ? 1.0 : 0.0
            pa += y[j, t] ? 0.0 : 1.0
        end
    end
    (; log_prior, log_jacobian, likelihood, L21, L22,
       posterior = log_prior + likelihood + log_jacobian)
end

@testset "PPL graph — dogs_nonhierarchical (posteriordb, correlated per-dog)" begin
    artifact = evaluate_dogs_nonhierarchical_source()
    @test artifact.source == strip(DOGS_NH_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat(-1.0, 0.5, log(0.5), log(0.4), 0.2, 0.1 .* range(-1.0, 1.0; length = 2 * DOGS_NH_J))
    reference = _dnh_reference(q)

    @testset "authored on the current baseline surface" begin
        @test occursin("L21::Float64 = tanh(w)", DOGS_NH_SOURCE)
        @test occursin("bernoulli(exp(ps * la + pa * lb)).logpdf(yi)", DOGS_NH_SOURCE)
        @test occursin("prev_shock::Vector{Float64} = vec(yf * C)", DOGS_NH_SOURCE)
        @test occursin("log_a_cell::Vector{Float64} = log_a[dog_idx]", DOGS_NH_SOURCE)
        @test occursin("bound = (; y, C, dog_idx)", DOGS_NH_SOURCE)
        @test !occursin("struct ", DOGS_NH_SOURCE)
        @test artifact.bernoulli_object === bernoulli
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        pk = prepare(model;
            have = (:unconstrained, :y, :C, :dog_idx),
            want = (:parameters, :log_prior, :log_jacobian, :likelihood, :posterior),
            bound = (; y = DOGS_NH_Y, C = DOGS_NH_C, dog_idx = DOGS_NH_DOG_IDX))
        parameters, log_prior, log_jacobian, likelihood, posterior = pk(q)
        @test log_prior ≈ reference.log_prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
        @test isfinite(posterior)
        # L is a valid 2x2 cholesky_factor_corr (unit columns).
        @test parameters.L21 ≈ reference.L21
        @test parameters.L22 ≈ reference.L22
        @test parameters.L21^2 + parameters.L22^2 ≈ 1.0
    end
end
