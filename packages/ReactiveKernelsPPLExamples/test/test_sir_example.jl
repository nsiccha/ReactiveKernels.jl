using ReactiveKernelsPPLExamples.SIRExample
using SciMLBase
using SciMLSensitivity
using SpecialFunctions: loggamma
import OrdinaryDiffEqLowOrderRK: DP5

function _sir_cauchy_logpdf(x, location, scale)
    z = (x - location) / scale
    return -log(π) - log(scale) - log1p(z^2)
end

function _sir_poisson_logpdf(count, rate)
    return count * log(rate) - rate - loggamma(count + 1)
end

function _sir_lognormal_logpdf(x, logmean, standard_deviation)
    z = (log(x) - logmean) / standard_deviation
    return -log(x) - log(standard_deviation) - 0.5 * log(2π) - 0.5 * z^2
end

function _sir_rhs!(du, u, p, t)
    beta, kappa, gamma, xi, delta = p
    force = beta * u[4] / (u[4] + kappa)
    du[1] = -force * u[1]
    du[2] = force * u[1] - gamma * u[2]
    du[3] = gamma * u[2]
    du[4] = xi * u[2] - delta * u[4]
    return nothing
end

function _sir_reference(q)
    beta, gamma, xi, delta = exp.(q)
    theta = [beta, 1.0e6, gamma, xi, delta]
    problem = SciMLBase.ODEProblem(
        _sir_rhs!, [10000.0, 0.0, 0.0, 10000.0], (0.0, last(SIR_T)),
        [0.1, 1.0e6, 0.1, 1.0, 1.0])
    solution = SciMLBase.solve(
        problem, DP5(); u0 = SIR_Y0, p = theta, saveat = SIR_T,
        reltol = 1e-6, abstol = 1e-6, maxiters = 1_000_000)
    trajectory = reduce(hcat, solution.u)'
    logprior =
        _sir_cauchy_logpdf(beta, 0.0, 2.5) +
        _sir_cauchy_logpdf(gamma, 0.0, 1.0) +
        _sir_cauchy_logpdf(xi, 0.0, 25.0) +
        _sir_cauchy_logpdf(delta, 0.0, 1.0)
    susceptible = trajectory[:, 1]
    likelihood = _sir_poisson_logpdf(SIR_STOI_HAT[1], SIR_Y0[1] - susceptible[1])
    for n in 2:length(SIR_STOI_HAT)
        likelihood += _sir_poisson_logpdf(
            SIR_STOI_HAT[n], susceptible[n - 1] - susceptible[n])
    end
    for i in eachindex(SIR_B_HAT)
        likelihood += _sir_lognormal_logpdf(
            SIR_B_HAT[i], log(trajectory[i, 4]), 0.15)
    end
    return (; beta, gamma, xi, delta, trajectory, logprior, likelihood,
             posterior = logprior + likelihood + sum(q))
end

@testset "PPL graph — sir (posteriordb adaptive RK45)" begin
    modules_before = Set(String(id.name) for id in keys(Base.loaded_modules))
    @test !("Reactant" in modules_before)

    artifact = evaluate_sir_source()
    @test artifact.source == strip(SIR_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, artifact.inputs.q)

    @testset "authored natural solver shape" begin
        @test occursin("const SIR_PROBLEM = SciMLBase.ODEProblem", SIR_SOURCE)
        @test occursin("kappa::Float64 = 1.0e6", SIR_SOURCE)
        @test occursin("u0 = y0", SIR_SOURCE)
        @test occursin("p = p", SIR_SOURCE)
        @test occursin("reltol = 1e-6", SIR_SOURCE)
        @test occursin("abstol = 1e-6", SIR_SOURCE)
        @test occursin("maxiters = 1_000_000", SIR_SOURCE)
        @test !occursin("remake(", SIR_SOURCE)
        @test !occursin("set_runtime_activity", SIR_SOURCE)
        @test !occursin("max(1.0e-16", SIR_SOURCE)
    end

    model = artifact.model
    q = [log(0.1), log(0.1), log(1.0), log(1.0)]
    reference = _sir_reference(q)
    parameters, trajectory, log_prior, likelihood, log_jacobian, posterior =
        prepare(model;
            have = (:unconstrained, :t, :y0, :stoi_hat, :B_hat),
            want = (:parameters, :trajectory, :log_prior, :likelihood,
                    :log_jacobian, :posterior),
            bound = (; t = SIR_T, y0 = SIR_Y0, stoi_hat = SIR_STOI_HAT,
                     B_hat = SIR_B_HAT))(q)

    @test collect(values(parameters)) ≈ [reference.beta, reference.gamma,
                                           reference.xi, reference.delta]
    @test trajectory ≈ reference.trajectory
    @test log_prior ≈ reference.logprior
    @test likelihood ≈ reference.likelihood
    @test log_jacobian ≈ sum(q)
    @test posterior ≈ reference.posterior
    @test posterior ≈ log_prior + likelihood + log_jacobian

    function sir_first_use()
        graph = build_sir_graph()
        kernel = prepare(graph;
            have = (:unconstrained, :t, :y0, :stoi_hat, :B_hat),
            want = :posterior,
            bound = (; t = SIR_T, y0 = SIR_Y0, stoi_hat = SIR_STOI_HAT,
                     B_hat = SIR_B_HAT))
        return kernel(q)
    end
    @test isfinite(sir_first_use())
    template_value = sir_first_use()
    full_value = prepare(compose(evaluate_sir_source().model);
        have = (:unconstrained, :t, :y0, :stoi_hat, :B_hat),
        want = :posterior,
        bound = (; t = SIR_T, y0 = SIR_Y0, stoi_hat = SIR_STOI_HAT,
                 B_hat = SIR_B_HAT))(q)
    @test template_value == full_value

    modules_after = Set(String(id.name) for id in keys(Base.loaded_modules))
    @test issubset(modules_before, modules_after)
    @test !("Reactant" in modules_after)
end
