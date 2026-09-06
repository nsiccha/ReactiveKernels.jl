using ReactiveKernelsPPLExamples.BetaBinomialExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: beta, binomial
using SpecialFunctions: loggamma, logbeta

# Graph-independent reference oracle. Beta(2, 2) prior (full normalization) plus
# the Binomial log-pmf, evaluated directly.
function _beta_binomial_reference(logit_rate, trials, successes)
    rate = 1 / (1 + exp(-logit_rate))
    log_jacobian = log(rate) + log1p(-rate)
    prior = (2 - 1) * log(rate) + (2 - 1) * log1p(-rate) - logbeta(2.0, 2.0)
    likelihood = sum(
        (loggamma(n + 1.0) - loggamma(k + 1.0) - loggamma(n - k + 1.0)) +
        k * log(rate) + (n - k) * log1p(-rate)
        for (n, k) in zip(trials, successes))
    (; prior, log_jacobian, likelihood,
       density = prior + log_jacobian + likelihood)
end

@testset "PPL graph — beta-binomial" begin
    artifact = evaluate_beta_binomial_source()
    @test artifact.source == strip(BETA_BINOMIAL_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    logit_rate = 0.2

    @testset "authored on the current baseline surface" begin
        @test occursin("beta(2.0, 2.0).logpdf", BETA_BINOMIAL_SOURCE)
        @test occursin("binomial(trial_count, p).logpdf", BETA_BINOMIAL_SOURCE)
        @test occursin("pointwise = plate(", BETA_BINOMIAL_SOURCE)
        @test !occursin("binomial_logpmf", BETA_BINOMIAL_SOURCE)
        @test !occursin("beta22", BETA_BINOMIAL_SOURCE)
        @test !occursin("struct ", BETA_BINOMIAL_SOURCE)
        @test artifact.beta_object === beta
        @test artifact.binomial_object === binomial

        raw_generated = code_expr(artifact.kernel)
        readable = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false,
        )
        @test !occursin(r"__ops__\[\d+\]", readable)
        @test !occursin(r"\boperation\(", readable)
    end

    @testset "constrain-only prunes the density work" begin
        p = plan(model.graph; have = (model.logit_rate,), want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.prior.id) in produced)
        parameters = prepare(p)(logit_rate)
        @test parameters isa NamedTuple
        @test parameters.rate ≈ 1 / (1 + exp(-0.2))
    end

    @testset "density decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.logit_rate, model.trials, model.successes),
                 want = (model.prior, model.log_jacobian, model.pointwise,
                         model.likelihood, model.density))
        prior, log_jacobian, pointwise, likelihood, density =
            prepare(p)(logit_rate, BETA_BINOMIAL_TRIALS, BETA_BINOMIAL_SUCCESSES)
        reference =
            _beta_binomial_reference(logit_rate, BETA_BINOMIAL_TRIALS,
                                     BETA_BINOMIAL_SUCCESSES)
        @test all(isfinite, pointwise)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test density ≈ reference.density
    end

    @testset "generated quantity prunes density work" begin
        parameters = (; rate = 0.55)
        p = plan(model.graph;
                 have = (model.parameters, model.new_trials), want = (model.expected,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.prior.id) in produced)
        @test prepare(p)(parameters, 20) ≈ 11.0
    end

    @testset "one authored plate exposes a buffer-free total" begin
        pointwise_kernel = prepare(model;
            have = (:logit_rate, :trials, :successes), want = :pointwise)
        likelihood_kernel = prepare(model;
            have = (:logit_rate, :trials, :successes), want = :likelihood)
        pw = pointwise_kernel(logit_rate, BETA_BINOMIAL_TRIALS, BETA_BINOMIAL_SUCCESSES)
        @test likelihood_kernel(logit_rate, BETA_BINOMIAL_TRIALS,
                                BETA_BINOMIAL_SUCCESSES) ≈ sum(pw)
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end

    @testset "out-of-support is diagnosed as -Inf, not a silent value" begin
        # The reused Beta object returns -Inf outside (0, 1).
        prior_kernel = prepare(model; have = (:parameters,), want = :prior)
        @test prior_kernel((; rate = 1.5)) == -Inf
    end
end
