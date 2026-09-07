using ReactiveKernelsPPLExamples.SurveyModelExample
using LogExpFunctions: logistic, log1pexp

# Graph-independent oracle for posteriordb Survey_model: a discrete-n
# marginalization. theta ∈ [0,1] (logistic transform, implicit uniform prior);
# marginal likelihood = log_sum_exp over n=1..nmax of
#   log(1/nmax) + LC[n] + SK*log(theta) + (n*m - SK)*log(1-theta),
# LC[n] = -Inf for n < nmin. Max-stabilized log_sum_exp.
function _survey_reference(q, ns, lc, sk, m, log_1_nmax)
    u = q[1]
    logtheta = -log1pexp(-u); log1mtheta = -log1pexp(u)
    log_jacobian = -log1pexp(-u) - log1pexp(u)
    lp = [log_1_nmax + lc[j] + sk * logtheta + (ns[j] * m - sk) * log1mtheta
          for j in eachindex(ns)]
    mx = maximum(lp)
    likelihood = mx + log(sum(exp(x - mx) for x in lp))
    (; theta = logistic(u), log_jacobian, likelihood,
       posterior = likelihood + log_jacobian)
end

@testset "PPL graph — Survey_model (posteriordb discrete-n marginalization)" begin
    artifact = evaluate_survey_model_source()
    @test artifact.source == strip(SURVEY_MODEL_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model

    @testset "authored surface" begin
        @test occursin("maximum(lp_parts)", SURVEY_MODEL_SOURCE)
        @test occursin("log_sum_exp", SURVEY_MODEL_SOURCE) || occursin("lp_parts = plate(", SURVEY_MODEL_SOURCE)
        @test !occursin("struct ", SURVEY_MODEL_SOURCE)
    end

    @testset "the out-of-support cells carry -Inf (n < nmin = max(k))" begin
        nmin = maximum(SURVEY_K)
        @test all(!isfinite, SURVEY_LC[1:nmin - 1])   # n < nmin → -Inf
        @test all(isfinite, SURVEY_LC[nmin:end])       # n ≥ nmin → finite
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        for u in (-0.4, 0.0, 0.7)
            q = [u]
            reference = _survey_reference(q, SURVEY_NS, SURVEY_LC, SURVEY_SK,
                                          SURVEY_M, SURVEY_LOG1NMAX)
            p = prepare(model;
                have = (:unconstrained, :ns, :lc, :sk, :m, :log_1_nmax),
                want = (:likelihood, :posterior),
                bound = (; ns = SURVEY_NS, lc = SURVEY_LC, sk = SURVEY_SK,
                          m = SURVEY_M, log_1_nmax = SURVEY_LOG1NMAX))
            likelihood, posterior = p(q)
            @test likelihood ≈ reference.likelihood
            @test posterior ≈ reference.posterior
        end
    end
end
