using ReactiveKernelsPPLExamples.LotkaVolterraExample
using SciMLBase
using SciMLSensitivity
import OrdinaryDiffEqLowOrderRK: DP5

function _lotka_normal_logpdf(x, mean, standard_deviation)
    z = (x - mean) / standard_deviation
    return -0.5 * log(2π) - log(standard_deviation) - 0.5 * z^2
end

function _lotka_lognormal_logpdf(x, logmean, standard_deviation)
    if x <= 0
        return -Inf
    end
    z = (log(x) - logmean) / standard_deviation
    return -log(x) - log(standard_deviation) - 0.5 * log(2π) - 0.5 * z^2
end

function _lotka_rhs!(du, u, p, t)
    du[1] = (p[1] - p[2] * u[2]) * u[1]
    du[2] = (-p[3] + p[4] * u[1]) * u[2]
    return nothing
end

function _lotka_reference(q)
    theta = exp.(view(q, 1:4))
    z_init = exp.(view(q, 5:6))
    sigma = exp.(view(q, 7:8))
    problem = SciMLBase.ODEProblem(
        _lotka_rhs!, [30.0, 4.0], (0.0, last(LOTKA_TS)),
        [0.55, 0.028, 0.80, 0.024])
    solution = SciMLBase.solve(
        problem, DP5(); u0 = z_init, p = theta, saveat = LOTKA_TS,
        reltol = 1e-5, abstol = 1e-3, maxiters = 500)
    trajectory = reduce(hcat, solution.u)'
    logprior =
        _lotka_normal_logpdf(theta[1], 1.0, 0.5) +
        _lotka_normal_logpdf(theta[2], 0.05, 0.05) +
        _lotka_normal_logpdf(theta[3], 1.0, 0.5) +
        _lotka_normal_logpdf(theta[4], 0.05, 0.05) +
        _lotka_lognormal_logpdf(sigma[1], -1.0, 1.0) +
        _lotka_lognormal_logpdf(sigma[2], -1.0, 1.0) +
        _lotka_lognormal_logpdf(z_init[1], log(10.0), 1.0) +
        _lotka_lognormal_logpdf(z_init[2], log(10.0), 1.0)
    initial =
        _lotka_lognormal_logpdf(LOTKA_Y_INIT[1], log(z_init[1]), sigma[1]) +
        _lotka_lognormal_logpdf(LOTKA_Y_INIT[2], log(z_init[2]), sigma[2])
    observations = 0.0
    for species in 1:2
        for i in eachindex(LOTKA_TS)
            observations += _lotka_lognormal_logpdf(
                LOTKA_Y[i, species], log(trajectory[i, species]), sigma[species])
        end
    end
    return (; theta, z_init, sigma, trajectory, logprior, initial, observations,
             posterior = logprior + initial + observations + sum(q))
end

@testset "PPL graph — lotka_volterra (posteriordb adaptive RK45)" begin
    # Boundary: native primal/value plus ordinary-Reverse gradient is
    # accepted, supported via landed core fix 55da875/f8acaa1 for snag
    # plain-enzyme-rev-3dc5d563; compiled Reactant primal/gradient is
    # unsupported pending the separate Reactant survey — so this file
    # asserts Reactant stays absent throughout (values here, gradients in
    # the default four-model gate).
    modules_before = Set(String(id.name) for id in keys(Base.loaded_modules))
    @test !("Reactant" in modules_before)

    artifact = evaluate_lotka_volterra_source()
    @test artifact.source == strip(LOTKA_VOLTERRA_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, artifact.inputs.q)

    @testset "authored natural solver shape" begin
        @test occursin("const LOTKA_PROBLEM = SciMLBase.ODEProblem", LOTKA_VOLTERRA_SOURCE)
        @test occursin("u0 = u0", LOTKA_VOLTERRA_SOURCE)
        @test occursin("p = p", LOTKA_VOLTERRA_SOURCE)
        @test occursin("reltol = 1e-5", LOTKA_VOLTERRA_SOURCE)
        @test occursin("abstol = 1e-3", LOTKA_VOLTERRA_SOURCE)
        @test occursin("maxiters = 500", LOTKA_VOLTERRA_SOURCE)
        @test !occursin("remake(", LOTKA_VOLTERRA_SOURCE)
        @test !occursin("set_runtime_activity", LOTKA_VOLTERRA_SOURCE)
    end

    model = artifact.model
    q = [0.0, log(0.05), 0.0, log(0.05), log(30.0), log(4.0), -1.0, -1.0]
    reference = _lotka_reference(q)
    parameters, trajectory, log_prior, initial_likelihood,
        observation_likelihood, log_jacobian, posterior = prepare(model;
        have = (:unconstrained, :ts, :y_init, :y),
        want = (:parameters, :trajectory, :log_prior, :initial_likelihood,
                :observation_likelihood, :log_jacobian, :posterior),
        bound = (; ts = LOTKA_TS, y_init = LOTKA_Y_INIT, y = LOTKA_Y))(q)

    @test parameters.theta ≈ reference.theta
    @test parameters.z_init ≈ reference.z_init
    @test parameters.sigma ≈ reference.sigma
    @test trajectory ≈ reference.trajectory
    @test log_prior ≈ reference.logprior
    @test initial_likelihood ≈ reference.initial
    @test observation_likelihood ≈ reference.observations
    @test log_jacobian ≈ sum(q)
    @test posterior ≈ reference.posterior
    @test posterior ≈ log_prior + initial_likelihood + observation_likelihood + log_jacobian

    function lotka_first_use()
        graph = build_lotka_volterra_graph()
        kernel = prepare(graph;
            have = (:unconstrained, :ts, :y_init, :y), want = :posterior,
            bound = (; ts = LOTKA_TS, y_init = LOTKA_Y_INIT, y = LOTKA_Y))
        return kernel(q)
    end
    @test isfinite(lotka_first_use())
    template_value = lotka_first_use()
    full_value = prepare(compose(evaluate_lotka_volterra_source().model);
        have = (:unconstrained, :ts, :y_init, :y), want = :posterior,
        bound = (; ts = LOTKA_TS, y_init = LOTKA_Y_INIT, y = LOTKA_Y))(q)
    @test template_value == full_value

    modules_after = Set(String(id.name) for id in keys(Base.loaded_modules))
    @test issubset(modules_before, modules_after)
    @test !("Reactant" in modules_after)
end
