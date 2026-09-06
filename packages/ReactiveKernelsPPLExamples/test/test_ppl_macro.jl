using ReactiveKernelsPPLExamples: @ppl
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# First cut of the RK-native sb-like `@ppl` macro: scalar real-support
# parameters + plate observation likelihoods, lowered to the canonical PPL
# workflow node set. Validated by density parity against a direct-formula
# reference (the existing hand-written PPL examples all carry a positive-scale
# or unit-interval parameter, which is a follow-up increment).
@testset "RK-native @ppl macro (first cut)" begin
    nlp(x, mu, sig) = -0.5 * log(2π) - log(sig) - 0.5 * ((x - mu) / sig)^2

    @ppl mm(y::Vector{Float64}, sigma::Vector{Float64}) = begin
        mu ~ normal(0.0, 10.0)
        y ~ normal(mu, sigma)
    end
    @test mm isa KernelSpec

    q = [1.5]
    y = [2.0, 0.5, 1.0]
    sigma = [1.0, 1.0, 1.0]
    ref_prior = nlp(1.5, 0.0, 10.0)
    ref_ll = sum(nlp(yi, 1.5, 1.0) for yi in y)

    @testset "canonical nodes and workflow cuts" begin
        post = prepare(mm; have = (:unconstrained, :y, :sigma),
                       want = :posterior)(q, y, sigma)
        @test post ≈ ref_prior + ref_ll

        parameters, prior, likelihood = prepare(mm;
            have = (:unconstrained, :y, :sigma),
            want = (:parameters, :prior, :likelihood))(q, y, sigma)
        @test prior ≈ ref_prior
        @test likelihood ≈ ref_ll
        @test parameters.mu == 1.5

        pointwise = prepare(mm; have = (:unconstrained, :y, :sigma),
                            want = :pointwise)(q, y, sigma)
        @test sum(pointwise) ≈ ref_ll

        cld = prepare(mm; have = (:unconstrained, :y, :sigma),
                      want = :constrained_logdensity)(q, y, sigma)
        up = prepare(mm; have = (:unconstrained,),
                     want = :unconstrained_prior)(q)
        @test cld ≈ ref_prior + ref_ll
        @test up ≈ ref_prior
    end

    @testset "named-latent HAVE boundary" begin
        post2 = prepare(mm; have = (:mu, :y, :sigma),
                        want = :posterior)(1.5, y, sigma)
        @test post2 ≈ ref_prior + ref_ll
    end

    @testset "two parameters with a per-cell linear predictor" begin
        @ppl lin(y::Vector{Float64}, x::Vector{Float64},
                 sigma::Vector{Float64}) = begin
            alpha ~ normal(0.0, 10.0)
            beta ~ normal(0.0, 5.0)
            y ~ normal(alpha + beta * x, sigma)
        end
        q2 = [1.0, 2.0]
        x = [-1.0, 0.0, 1.0]
        y2 = [-1.1, 1.2, 2.9]
        sig = [1.0, 1.0, 1.0]
        ref_prior2 = nlp(1.0, 0.0, 10.0) + nlp(2.0, 0.0, 5.0)
        mu_i = [1.0 + 2.0 * xi for xi in x]
        ref_ll2 = sum(nlp(y2[i], mu_i[i], sig[i]) for i in eachindex(y2))
        post_lin = prepare(lin; have = (:unconstrained, :y, :x, :sigma),
                           want = :posterior)(q2, y2, x, sig)
        @test post_lin ≈ ref_prior2 + ref_ll2
    end

    @testset "out-of-scope constructs fail loudly" begin
        # positive-support parameter (constraint inference) is a follow-up cut.
        @test_throws Exception macroexpand(@__MODULE__, :(@ppl bad(
                y::Vector{Float64}) = begin
            scale ~ exponential(1.0)
            y ~ normal(0.0, scale)
        end))
        # a posterior-mode model needs at least one observation.
        @test_throws Exception macroexpand(@__MODULE__, :(@ppl prioronly(
                y::Vector{Float64}) = begin
            mu ~ normal(0.0, 1.0)
        end))
    end
end
