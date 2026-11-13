# Pass the public BYM2 toy from snag rkppl-adaptive-c-2120d96a as the sole
# argument. The reference fixture used for the initial acceptance has SHA-256
# 1076770bb986328a9a7614d24e6016cded45b1d27d3b6a42fbffe32b6b8460b1.
# This script prints its actual digest before checking target agreement and
# comparing four chains per arm. Timing includes each arm's first-use compilation.
using ReactiveKernelsPPL, WarmupHMC, Enzyme, Random, Statistics, SHA, Test
using ReactiveKernelsPPL: adaptive_centering_problem
using DifferentiationInterface: AutoEnzyme
import LogDensityProblems as LDP
import MCMCDiagnosticTools as MCD

include(joinpath(pkgdir(ReactiveKernelsPPL), "test", "centering", "runtests.jl"))
const ACT = AdaptiveCenteringTests

# Reuse the reporter's exact public toy-data generator, without running the
# file's six top-level fits. Its dense graph scaling is only this N=32 oracle;
# it is not part of the adapter or a scalable model-preparation implementation.
toy_path = only(ARGS)
toy_source = read(toy_path, String)
println((; fixture_sha256 = bytes2hex(sha256(toy_source)), julia = string(VERSION),
    ppl = string(pkgversion(ReactiveKernelsPPL)), warmuphmc = string(pkgversion(WarmupHMC))))
include_string(@__MODULE__, first(split(toy_source, "\nfunction fit(")), toy_path)

function check_reporter_targets(data, built, plan)
    inputs = Base.structdiff(data, NamedTuple{(:y,)})
    u0 = zeros(built.layout.total)
    original = prepare_sampler(built, plan, u0; backend = AutoEnzyme())
    centered = adaptive_centering_problem(built, plan, original, AutoEnzyme();
        cells = (theta = (location = :theta_location, log_scale = :theta_log_scale,
            source = 1.0),), u0)
    for (label, model, candidate) in ((:noncentered, NONCENTRED, original),
            (:centered, CENTRED, centered))
        reference_plan = model(; inputs...) | (; y = data.y)
        reference_built = build_kernel(reference_plan)
        reference = prepare_sampler(reference_built, reference_plan, u0; backend = AutoEnzyme())
        names = string.(coordinate_names(built.layout))
        label === :centered && (names = replace.(names, "theta." => "total."))
        permutation = [only(findall(==(string(name)), names))
            for name in coordinate_names(reference_built.layout)]
        worst_value_error = 0.0
        worst_gradient_error = 0.0
        for phase in (0.2, 0.7, 1.3)
            x = 0.3 .* sin.(phase .+ (1:length(u0)))
            value, gradient = LDP.logdensity_and_gradient(candidate, x)
            expected_value, expected_gradient = LDP.logdensity_and_gradient(reference, x[permutation])
            expected_gradient = expected_gradient[invperm(permutation)]
            @test value ≈ expected_value rtol = 1e-10
            @test gradient ≈ expected_gradient rtol = 1e-10 atol = 1e-9
            worst_value_error = max(worst_value_error, abs(value - expected_value))
            worst_gradient_error = max(worst_gradient_error, maximum(abs, gradient - expected_gradient))
        end
        println((; phase = :reporter_target_agreement, label, worst_value_error, worst_gradient_error))
        flush(stdout)
    end
end

function sample_comparison()
    data = simulate(4, 2, 0.1; clustered = true)
    input = Base.structdiff(data, NamedTuple{(:y, :mu_scale)})
    plan = ACT.SPATIAL(; input..., mu_sd = data.mu_scale) | (; y = data.y)
    built = build_kernel(plan)
    u0 = zeros(built.layout.total)
    check_reporter_targets(data, built, plan)
    for arm in (:original, :adaptive_theta, :adaptive_phi, :fixed_theta_centered)
        results = []
        controls = []
        chain_elapsed = Float64[]
        t0 = time()
        for chain in 1:4
            chain_started = time()
            problem = prepare_sampler(built, plan, u0; backend = AutoEnzyme())
            if arm === :original
                rp = problem
            else
                cells = arm === :adaptive_phi ?
                    (phi = (location = :phi_location, log_scale = :phi_log_scale),) :
                    (theta = (location = :theta_location, log_scale = :theta_log_scale,
                        source = arm === :fixed_theta_centered ? 1.0 : 0.0),)
                rp = adaptive_centering_problem(built, plan, problem, AutoEnzyme(); cells, u0)
            end
            result = adaptive_warmup_mcmc(Xoshiro(100 + chain), rp;
                n_draws = 1000, monitor_ess = true,
                nonlinear_adapt = arm !== :fixed_theta_centered)
            push!(chain_elapsed, time() - chain_started)
            push!(results, result)
            arm === :original || push!(controls,
                [last(pair).c for pair in WarmupHMC.reparam_sources(rp)])
            println((; phase = :chain_finished, arm, chain,
                draws = size(result.posterior_position, 2), elapsed = last(chain_elapsed)))
            flush(stdout)
        end
        elapsed = time() - t0
        nd = minimum(size(r.posterior_position, 2) for r in results)
        cube = permutedims(cat([permutedims(Matrix(r.posterior_position)[:, end-nd+1:end])
            for r in results]...; dims = 3), (1, 3, 2))
        draws = sum(size(r.posterior_position, 2) for r in results)
        divergent = sum(r.n_divergent_samples for r in results)
        println((; phase = :comparison, arm, common_draws = nd,
            max_rhat = maximum(MCD.rhat(cube)), min_bulk_ess = minimum(MCD.ess(cube; kind = :bulk)),
            min_tail_ess = minimum(MCD.ess(cube; kind = :tail)), divergent,
            divergent_percent = 100 * divergent / draws,
            evaluations_per_draw = [r.sampling_evaluation_counter / size(r.posterior_position, 2)
                for r in results], elapsed, chain_elapsed, controls))
        flush(stdout)
    end
end

sample_comparison()
