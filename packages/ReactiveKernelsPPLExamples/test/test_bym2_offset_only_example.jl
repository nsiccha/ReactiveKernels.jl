using ReactiveKernelsPPLExamples.Bym2OffsetOnlyExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, beta, poisson
using LogExpFunctions: logistic, log1pexp
using SpecialFunctions: loggamma, logbeta

# Graph-independent oracle for the BYM2 spatial-Poisson model.
function _bym2_reference(q, node1, node2, y, E, scaling_factor)
    N = length(y)
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    beta0 = q[1]
    sigma = exp(q[2])
    rho = logistic(q[3])
    theta = q[4:3 + N]
    phi = q[4 + N:3 + 2N]
    log_jacobian = q[2] + (-log1pexp(-q[3]) - log1pexp(q[3]))
    rho_prior = (0.5 - 1) * log(rho) + (0.5 - 1) * log1p(-rho) - logbeta(0.5, 0.5)
    prior = nlp(beta0, 0.0, 1.0) + nlp(sigma, 0.0, 1.0) + rho_prior +
            sum(nlp(t, 0.0, 1.0) for t in theta) +
            (-0.5 * sum((phi[node1] .- phi[node2]) .^ 2)) +
            nlp(sum(phi), 0.0, 0.001 * N)
    log_E = log.(E)
    convolved = sqrt(1 - rho) .* theta .+ sqrt(rho / scaling_factor) .* phi
    eta = log_E .+ beta0 .+ convolved .* sigma
    likelihood = sum(y[i] * eta[i] - exp(eta[i]) - loggamma(y[i] + 1.0) for i in 1:N)
    (; prior, log_jacobian, likelihood, posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — bym2_offset_only (posteriordb)" begin
    N = length(BYM2_Y)
    q = vcat([0.1, -0.3, 0.2], 0.15 .* cos.(1:N), 0.1 .* sin.(1:N))
    reference = _bym2_reference(q, BYM2_NODE1, BYM2_NODE2, BYM2_Y, BYM2_E,
                                BYM2_SCALING_FACTOR)

    # Keep the true first-use call ahead of the full-source evaluator; the latter
    # prepares and executes its demo tail and would otherwise warm this path.
    @testset "actual build+prepare+execute first use in one ordinary function" begin
        function once(q)
            k = prepare(build_bym2_offset_only_graph();
                have = (:unconstrained, :node1, :node2, :y, :E, :scaling_factor),
                want = :posterior,
                bound = (; node1 = BYM2_NODE1, node2 = BYM2_NODE2, y = BYM2_Y,
                           E = BYM2_E, scaling_factor = BYM2_SCALING_FACTOR))
            k(q)
        end
        @test once(q) ≈ reference.posterior
    end

    artifact = evaluate_bym2_offset_only_source()
    @test artifact.source == strip(BYM2_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model

    @testset "authored on the current baseline surface" begin
        @test occursin("poisson(; log_rate = e)", BYM2_SOURCE)
        @test occursin("beta(0.5, 0.5).logpdf(rho)", BYM2_SOURCE)
        @test occursin("phi[node1]", BYM2_SOURCE)
        @test occursin("log(e)", BYM2_SOURCE)          # in-graph log_E
        @test !occursin("struct ", BYM2_SOURCE)
        @test artifact.beta_object === beta
        @test artifact.poisson_object === poisson
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.node1, model.node2, model.y,
                         model.E, model.scaling_factor),
                 want = (model.prior, model.log_jacobian, model.likelihood, model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(q, BYM2_NODE1, BYM2_NODE2, BYM2_Y, BYM2_E, BYM2_SCALING_FACTOR)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "prepared graph repeat-use stable" begin
        k = prepare(build_bym2_offset_only_graph();
            have = (:unconstrained, :node1, :node2, :y, :E, :scaling_factor),
            want = :posterior,
            bound = (; node1 = BYM2_NODE1, node2 = BYM2_NODE2, y = BYM2_Y,
                       E = BYM2_E, scaling_factor = BYM2_SCALING_FACTOR))
        @test k(q) == k(q)
    end

    @testset "data-generic: alternate small graph (N=3, 2 edges)" begin
        node1 = [1, 2]; node2 = [2, 3]; y = [0, 2, 1]
        E = [10.0, 5.0, 8.0]; sf = 0.5
        qs = [0.2, -0.1, 0.3, 0.1, -0.2, 0.05, 0.0, 0.15, -0.05]  # 2N+3 = 9
        k = prepare(build_bym2_offset_only_graph();
            have = (:unconstrained, :node1, :node2, :y, :E, :scaling_factor),
            want = :posterior, bound = (; node1, node2, y, E, scaling_factor = sf))
        @test k(qs) ≈ _bym2_reference(qs, node1, node2, y, E, sf).posterior
    end
end
