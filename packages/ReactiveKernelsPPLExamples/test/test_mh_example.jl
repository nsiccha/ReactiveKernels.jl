using ReactiveKernelsPPLExamples.MhExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic, log1pexp, logaddexp

# Graph-independent oracle for Mh (capture-recapture, data-augmentation
# log_sum_exp marginalization).
function _mh_reference(q, y, lchoose, T)
    M = length(y)
    om = logistic(q[1]); mp = logistic(q[2]); sg = 5.0 * logistic(q[3])
    er = q[4:3 + M]
    jac = (-log1pexp(-q[1]) - log1pexp(q[1])) +
          (-log1pexp(-q[2]) - log1pexp(q[2])) +
          (log(5.0) - log1pexp(-q[3]) - log1pexp(q[3]))
    nlp(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    prior = sum(nlp(e, 0, 1) for e in er)
    lmp = log(mp) - log1p(-mp); lo = log(om); l1 = log1p(-om)
    like = 0.0
    for i in 1:M
        eps_i = lmp + sg * er[i]
        if y[i] > 0
            like += lo + lchoose[i] + y[i] * (-log1pexp(-eps_i)) +
                    (T - y[i]) * (-log1pexp(eps_i))
        else
            like += logaddexp(lo + (-T * log1pexp(eps_i)), l1)
        end
    end
    (; prior, log_jacobian = jac, likelihood = like, posterior = prior + like + jac)
end

@testset "PPL graph — Mh (posteriordb capture-recapture)" begin
    artifact = evaluate_mh_source()
    @test artifact.source == strip(MH_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = vcat([0.2, -0.1, 0.3], 0.15 .* collect(1:MH_M) ./ MH_M)
    reference = _mh_reference(q, MH_Y, MH_LCHOOSE, MH_T)

    @testset "authored on the current baseline surface" begin
        @test occursin("logaddexp(", MH_SOURCE)
        @test occursin("ifelse(yi > 0", MH_SOURCE)
        @test !occursin("struct ", MH_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = prepare(model;
            have = (:unconstrained, :y, :lchoose, :T, :M),
            want = (:prior, :log_jacobian, :likelihood, :posterior),
            bound = (; y = MH_Y, lchoose = MH_LCHOOSE, T = MH_T, M = MH_M))
        prior, log_jacobian, likelihood, posterior = p(q)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "log_sum_exp marginalization is exercised (both branches present)" begin
        @test any(>(0), MH_Y)   # observed individuals
        @test any(==(0), MH_Y)  # augmented (never-detected) individuals
    end
end
