using ReactiveKernelsPPLExamples.DiamondsExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, student_t
using SpecialFunctions: loggamma

# Graph-independent reference oracle for the posteriordb diamonds model: the brms
# centered design (drop the intercept column, subtract each remaining column's
# mean), Normal(0,1) coefficient priors, a Student_t(3,8,10) intercept prior, a
# half-Student_t(3,0,10) scale prior (the `- student_t_lccdf(0|3,0,10)` = log(2)
# normalization), the exp support transform for sigma, and the Normal-id-glm
# likelihood Yᵢ ~ Normal(Intercept + Xcᵢ·b, sigma).
_normal_ld(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
_student_ld(x, nu, loc, sc) =
    loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * log(nu * π) - log(sc) -
    ((nu + 1) / 2) * log1p(((x - loc) / sc)^2 / nu)

function _diamonds_reference(q)
    Kc = size(DIAMONDS_X, 2) - 1
    b = q[1:Kc]
    Intercept = q[Kc + 1]
    log_sigma = q[Kc + 2]
    sigma = exp(log_sigma)
    X = DIAMONDS_X
    n = size(X, 1)
    Xnoint = X[:, 2:size(X, 2)]
    means = vec(sum(Xnoint; dims = 1) ./ n)
    Xc = Xnoint .- means'
    mu = Intercept .+ Xc * b
    b_prior = sum(_normal_ld(bj, 0.0, 1.0) for bj in b)
    intercept_prior = _student_ld(Intercept, 3.0, 8.0, 10.0)
    sigma_prior = log(2.0) + _student_ld(sigma, 3.0, 0.0, 10.0)
    log_prior = b_prior + intercept_prior + sigma_prior
    log_jacobian = log_sigma
    likelihood = sum(_normal_ld(DIAMONDS_Y[i], mu[i], sigma) for i in eachindex(DIAMONDS_Y))
    b_Intercept = Intercept - sum(means .* b)
    (; log_prior, log_jacobian, likelihood, b_Intercept,
       posterior = log_prior + likelihood + log_jacobian)
end

@testset "PPL graph — diamonds (posteriordb, brms centered regression)" begin
    artifact = evaluate_diamonds_source()
    @test artifact.source == strip(DIAMONDS_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat(fill(0.05, size(DIAMONDS_X, 2) - 1), 8.0, log(0.6))
    reference = _diamonds_reference(q)

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(0.0, 1.0).logpdf", DIAMONDS_SOURCE)
        @test occursin("student_t(3.0, 8.0, 10.0).logpdf", DIAMONDS_SOURCE)
        @test occursin("log(2.0) + student_t(3.0, 0.0, 10.0).logpdf", DIAMONDS_SOURCE)
        @test occursin("Xc::Matrix{Float64} = Xnoint .- col_means", DIAMONDS_SOURCE)
        @test occursin("bound = (; X)", DIAMONDS_SOURCE)
        @test occursin("pointwise = plate(", DIAMONDS_SOURCE)
        @test !occursin("struct ", DIAMONDS_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.student_t_object === student_t
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        pk = prepare(model;
            have = (:unconstrained, :X, :Y),
            want = (:log_prior, :log_jacobian, :likelihood, :posterior),
            bound = (; X = DIAMONDS_X))
        log_prior, log_jacobian, likelihood, posterior = pk(q, DIAMONDS_Y)
        @test log_prior ≈ reference.log_prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
        @test isfinite(posterior)
    end

    @testset "generated quantity b_Intercept from a constrained HAVE" begin
        gq = prepare(model; have = (:unconstrained, :X), want = :b_Intercept,
                     bound = (; X = DIAMONDS_X))
        @test gq(q) ≈ reference.b_Intercept
    end

    @testset "the data-only centering prefix hoists under bound" begin
        plain = prepare(model; have = (:unconstrained, :X, :Y), want = :posterior)
        bound = prepare(model; have = (:unconstrained, :X, :Y), want = :posterior,
                        bound = (; X = DIAMONDS_X))
        @test plain(q, DIAMONDS_X, DIAMONDS_Y) ≈ bound(q, DIAMONDS_Y)
        @test bound(q, DIAMONDS_Y) ≈ reference.posterior
    end
end
