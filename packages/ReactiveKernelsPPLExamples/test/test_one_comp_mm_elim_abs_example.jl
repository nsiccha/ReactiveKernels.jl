using ReactiveKernelsPPLExamples.OneCompMMElimAbsExample
using SciMLBase
using SciMLSensitivity
import OrdinaryDiffEqBDF: FBDF

function _onecomp_cauchy_logpdf(x, location, scale)
    z = (x - location) / scale
    return -log(π) - log(scale) - log1p(z^2)
end

function _onecomp_lognormal_logpdf(x, logmean, standard_deviation)
    z = (log(x) - logmean) / standard_deviation
    return -log(x) - log(standard_deviation) - 0.5 * log(2π) - 0.5 * z^2
end

function _onecomp_rhs!(du, u, p, t)
    k_a, K_m, V_m = p
    dose = t > 0 ? exp(-k_a * t) * ONECOMP_D * k_a / ONECOMP_V : 0.0
    elim = (V_m / ONECOMP_V) * u[1] / (K_m + u[1])
    du[1] = dose - elim
    return nothing
end

function _onecomp_reference(q)
    k_a, K_m, V_m, sigma = exp.(q)
    theta = [k_a, K_m, V_m]
    problem = SciMLBase.ODEProblem(
        _onecomp_rhs!, [0.0], (ONECOMP_T0, last(ONECOMP_TIMES)),
        [1.0, 1.0, 1.0])
    solution = SciMLBase.solve(
        problem, FBDF(); u0 = ONECOMP_C0, p = theta, saveat = ONECOMP_TIMES,
        reltol = 1e-10, abstol = 1e-10, maxiters = 100_000_000)
    trajectory = reduce(hcat, solution.u)'
    logprior =
        _onecomp_cauchy_logpdf(k_a, 0.0, 1.0) +
        _onecomp_cauchy_logpdf(K_m, 0.0, 1.0) +
        _onecomp_cauchy_logpdf(V_m, 0.0, 1.0) +
        _onecomp_cauchy_logpdf(sigma, 0.0, 1.0)
    likelihood = 0.0
    for i in eachindex(ONECOMP_C_HAT)
        likelihood += _onecomp_lognormal_logpdf(
            ONECOMP_C_HAT[i], log(trajectory[i, 1]), sigma)
    end
    return (; k_a, K_m, V_m, sigma, trajectory, logprior, likelihood,
             posterior = logprior + likelihood + sum(q))
end

@testset "PPL graph — one_comp_mm_elim_abs (posteriordb adaptive BDF)" begin
    modules_before = Set(String(id.name) for id in keys(Base.loaded_modules))
    @test !("Reactant" in modules_before)

    artifact = evaluate_one_comp_mm_elim_abs_source()
    @test artifact.source == strip(ONE_COMP_MM_ELIM_ABS_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, artifact.inputs.q)

    @testset "authored natural solver shape" begin
        @test occursin(
            "const ONECOMP_PROBLEM = SciMLBase.ODEProblem",
            ONE_COMP_MM_ELIM_ABS_SOURCE)
        @test occursin("import OrdinaryDiffEqBDF: FBDF",
                       ONE_COMP_MM_ELIM_ABS_SOURCE)
        @test occursin("ONECOMP_PROBLEM, FBDF()", ONE_COMP_MM_ELIM_ABS_SOURCE)
        @test occursin("u0 = c0", ONE_COMP_MM_ELIM_ABS_SOURCE)
        @test occursin("p = p", ONE_COMP_MM_ELIM_ABS_SOURCE)
        @test occursin("reltol = 1e-10", ONE_COMP_MM_ELIM_ABS_SOURCE)
        @test occursin("abstol = 1e-10", ONE_COMP_MM_ELIM_ABS_SOURCE)
        @test occursin("maxiters = 100_000_000", ONE_COMP_MM_ELIM_ABS_SOURCE)
        @test occursin("t > 0 ?", ONE_COMP_MM_ELIM_ABS_SOURCE)
        @test occursin("Ref(sigma)", ONE_COMP_MM_ELIM_ABS_SOURCE)
        @test !occursin("remake(", ONE_COMP_MM_ELIM_ABS_SOURCE)
        @test !occursin("set_runtime_activity", ONE_COMP_MM_ELIM_ABS_SOURCE)
        @test !occursin("max(1.0e-16", ONE_COMP_MM_ELIM_ABS_SOURCE)
    end

    model = artifact.model
    q = [log(1.0), log(1.0), log(1.0), log(1.0)]
    reference = _onecomp_reference(q)
    parameters, trajectory, log_prior, likelihood, log_jacobian, posterior =
        prepare(model;
            have = (:unconstrained, :times, :c0, :c_hat),
            want = (:parameters, :trajectory, :log_prior, :likelihood,
                    :log_jacobian, :posterior),
            bound = (; times = ONECOMP_TIMES, c0 = ONECOMP_C0,
                     c_hat = ONECOMP_C_HAT))(q)

    @test collect(values(parameters)) ≈ [reference.k_a, reference.K_m,
                                           reference.V_m, reference.sigma]
    @test trajectory ≈ reference.trajectory
    @test log_prior ≈ reference.logprior
    @test likelihood ≈ reference.likelihood
    @test log_jacobian ≈ sum(q)
    @test posterior ≈ reference.posterior
    @test posterior ≈ log_prior + likelihood + log_jacobian

    function onecomp_first_use()
        graph = build_one_comp_mm_elim_abs_graph()
        kernel = prepare(graph;
            have = (:unconstrained, :times, :c0, :c_hat),
            want = :posterior,
            bound = (; times = ONECOMP_TIMES, c0 = ONECOMP_C0,
                     c_hat = ONECOMP_C_HAT))
        return kernel(q)
    end
    @test isfinite(onecomp_first_use())
    template_value = onecomp_first_use()
    full_value = prepare(compose(evaluate_one_comp_mm_elim_abs_source().model);
        have = (:unconstrained, :times, :c0, :c_hat),
        want = :posterior,
        bound = (; times = ONECOMP_TIMES, c0 = ONECOMP_C0,
                 c_hat = ONECOMP_C_HAT))(q)
    @test template_value == full_value

    modules_after = Set(String(id.name) for id in keys(Base.loaded_modules))
    @test issubset(modules_before, modules_after)
    @test !("Reactant" in modules_after)
end
