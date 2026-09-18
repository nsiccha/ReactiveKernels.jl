using ReactiveKernelsPPLExamples.SoilIncubationExample
using LogExpFunctions: logistic, log1pexp
using SciMLBase
using SciMLSensitivity
using SpecialFunctions: loggamma
import OrdinaryDiffEqLowOrderRK: DP5

function _soil_cauchy_logpdf(x, location, scale)
    z = (x - location) / scale
    return -log(π) - log(scale) - log1p(z^2)
end

function _soil_normal_logpdf(x, mu, standard_deviation)
    z = (x - mu) / standard_deviation
    return -log(standard_deviation) - 0.5 * log(2π) - 0.5 * z^2
end

function _soil_beta_logpdf(x, a, b)
    return (a - 1) * log(x) + (b - 1) * log(1 - x) -
        (loggamma(a) + loggamma(b) - loggamma(a + b))
end

function _soil_rhs!(du, u, p, t)
    k1, k2, alpha21, alpha12 = p
    du[1] = -k1 * u[1] + alpha12 * k2 * u[2]
    du[2] = -k2 * u[2] + alpha21 * k1 * u[1]
    return nothing
end

function _soil_reference(q)
    k1, k2, alpha21, alpha12 = exp.(q[1:4])
    gamma = logistic(q[5])
    sigma = exp(q[6])
    c_t0 = [gamma * SOIL_TOTAL_C_T0, (1 - gamma) * SOIL_TOTAL_C_T0]
    theta = [k1, k2, alpha21, alpha12]
    problem = SciMLBase.ODEProblem(
        _soil_rhs!, [3.85, 3.85], (SOIL_T0, last(SOIL_TS)),
        [0.1, 0.1, 0.1, 0.1])
    solution = SciMLBase.solve(
        problem, DP5(); u0 = c_t0, p = theta, saveat = SOIL_TS,
        reltol = 1e-6, abstol = 1e-6, maxiters = 1_000_000)
    trajectory = reduce(hcat, solution.u)'
    evolved = SOIL_TOTAL_C_T0 .- trajectory[:, 1] .- trajectory[:, 2]
    logprior =
        _soil_normal_logpdf(k1, 0.0, 1.0) +
        _soil_normal_logpdf(k2, 0.0, 1.0) +
        _soil_normal_logpdf(alpha21, 0.0, 1.0) +
        _soil_normal_logpdf(alpha12, 0.0, 1.0) +
        _soil_beta_logpdf(gamma, 10.0, 1.0) +
        _soil_cauchy_logpdf(sigma, 0.0, 1.0)
    logjacobian = sum(q[1:4]) + q[6] - log1pexp(-q[5]) - log1pexp(q[5])
    likelihood = 0.0
    for i in eachindex(SOIL_ECO2MEAN)
        likelihood += _soil_normal_logpdf(SOIL_ECO2MEAN[i], evolved[i], sigma)
    end
    return (; k1, k2, alpha21, alpha12, gamma, sigma, trajectory, evolved,
             logprior, likelihood, logjacobian,
             posterior = logprior + likelihood + logjacobian)
end

@testset "PPL graph — soil_incubation (posteriordb adaptive RK45)" begin
    modules_before = Set(String(id.name) for id in keys(Base.loaded_modules))
    @test !("Reactant" in modules_before)

    artifact = evaluate_soil_incubation_source()
    @test artifact.source == strip(SOIL_INCUBATION_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, artifact.inputs.q)

    @testset "authored natural solver shape" begin
        @test occursin(
            "const SOIL_PROBLEM = SciMLBase.ODEProblem", SOIL_INCUBATION_SOURCE)
        @test occursin("logistic(gamma_unc)", SOIL_INCUBATION_SOURCE)
        @test occursin("Ref(sigma)", SOIL_INCUBATION_SOURCE)
        @test occursin("u0 = c_t0", SOIL_INCUBATION_SOURCE)
        @test occursin("p = p", SOIL_INCUBATION_SOURCE)
        @test occursin("reltol = 1e-6", SOIL_INCUBATION_SOURCE)
        @test occursin("abstol = 1e-6", SOIL_INCUBATION_SOURCE)
        @test occursin("maxiters = 1_000_000", SOIL_INCUBATION_SOURCE)
        @test !occursin("remake(", SOIL_INCUBATION_SOURCE)
        @test !occursin("set_runtime_activity", SOIL_INCUBATION_SOURCE)
        @test !occursin("max(1.0e-16", SOIL_INCUBATION_SOURCE)
    end

    model = artifact.model
    q = [log(0.1), log(0.1), log(0.1), log(0.1), 0.0, log(1.0)]
    reference = _soil_reference(q)
    parameters, trajectory, log_prior, likelihood, log_jacobian, posterior =
        prepare(model;
            have = (:unconstrained, :ts, :total_c_t0, :eco2_mean),
            want = (:parameters, :trajectory, :log_prior, :likelihood,
                    :log_jacobian, :posterior),
            bound = (; ts = SOIL_TS, total_c_t0 = SOIL_TOTAL_C_T0,
                     eco2_mean = SOIL_ECO2MEAN))(q)

    @test collect(values(parameters)) ≈ [reference.k1, reference.k2,
                                           reference.alpha21, reference.alpha12,
                                           reference.gamma, reference.sigma]
    @test trajectory ≈ reference.trajectory
    @test log_prior ≈ reference.logprior
    @test likelihood ≈ reference.likelihood
    @test log_jacobian ≈ reference.logjacobian
    @test posterior ≈ reference.posterior
    @test posterior ≈ log_prior + likelihood + log_jacobian

    function soil_first_use()
        graph = build_soil_incubation_graph()
        kernel = prepare(graph;
            have = (:unconstrained, :ts, :total_c_t0, :eco2_mean),
            want = :posterior,
            bound = (; ts = SOIL_TS, total_c_t0 = SOIL_TOTAL_C_T0,
                     eco2_mean = SOIL_ECO2MEAN))
        return kernel(q)
    end
    @test isfinite(soil_first_use())
    template_value = soil_first_use()
    full_value = prepare(compose(evaluate_soil_incubation_source().model);
        have = (:unconstrained, :ts, :total_c_t0, :eco2_mean),
        want = :posterior,
        bound = (; ts = SOIL_TS, total_c_t0 = SOIL_TOTAL_C_T0,
                 eco2_mean = SOIL_ECO2MEAN))(q)
    @test template_value == full_value

    modules_after = Set(String(id.name) for id in keys(Base.loaded_modules))
    @test issubset(modules_before, modules_after)
    @test !("Reactant" in modules_after)
end
