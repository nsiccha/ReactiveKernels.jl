using AdvancedHMC
using DifferentiationInterface
import Enzyme
using DynamicHMC
using DynamicHMC.Diagnostics: REACHED_MAX_DEPTH, is_divergent
using LinearAlgebra
using LogDensityProblems
using MCMCDiagnosticTools
using Pkg
using Random
using ReactiveKernels
using Statistics

# Every sampler's runtime gradient goes through this shared reverse-mode Enzyme
# backend via DifferentiationInterface, so RK/AdvancedHMC/DynamicHMC differentiate
# the SAME scalar log density and the comparison is fair. Runtime activity is on
# and the differentiated function is annotated Const because only the numeric
# position is differentiated, never any captured constant model data. The
# handwritten analytic gradients below are correctness oracles only, never the
# sampled path.
const BENCHMARK_BACKEND = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const,
)

abstract type CountedTarget end

# Targets carry a CONCRETELY-TYPED DI+Enzyme gradient preparation built once (at
# construction, i.e. the setup phase, timed separately from sampling) from the
# actual initial primal, so the sampled gradient path is type-stable.
mutable struct GaussianTarget{P} <: CountedTarget
    dimension::Int
    logdensity_calls::Int
    gradient_calls::Int
    preparation::P
end
GaussianTarget(dimension::Int) = GaussianTarget(
    dimension, 0, 0,
    prepare_gradient(_gaussian_logp, BENCHMARK_BACKEND, zeros(dimension)))

mutable struct CenteredEightSchoolsTarget{P} <: CountedTarget
    logdensity_calls::Int
    gradient_calls::Int
    preparation::P
end
CenteredEightSchoolsTarget() = CenteredEightSchoolsTarget(
    0, 0,
    prepare_gradient(_centered_logp, BENCHMARK_BACKEND, [0.0, log(5.0), zeros(8)...]))

mutable struct NoncenteredEightSchoolsTarget{P} <: CountedTarget
    logdensity_calls::Int
    gradient_calls::Int
    preparation::P
end
NoncenteredEightSchoolsTarget() = NoncenteredEightSchoolsTarget(
    0, 0,
    prepare_gradient(_noncentered_logp, BENCHMARK_BACKEND, [0.0, log(5.0), zeros(8)...]))

const EIGHT_Y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
const EIGHT_SIGMA = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
const COMPARISON_PACKAGES = (
    "ReactiveKernels",
    "MutatingFunctions",
    "AdvancedHMC",
    "DynamicHMC",
    "LogDensityProblems",
    "MCMCDiagnosticTools",
    "DifferentiationInterface",
    "Enzyme",
)

LogDensityProblems.dimension(target::GaussianTarget) = target.dimension
LogDensityProblems.dimension(::CenteredEightSchoolsTarget) = 10
LogDensityProblems.dimension(::NoncenteredEightSchoolsTarget) = 10
LogDensityProblems.capabilities(::Type{<:CountedTarget}) =
    LogDensityProblems.LogDensityOrder{1}()

_gaussian(q) = (-sum(abs2, q) / 2, -copy(q))

function _centered_eight_schools(q)
    mu = q[1]
    log_tau = q[2]
    tau = exp(log_tau)
    theta = @view q[3:10]
    inverse_tau_squared = inv(tau * tau)

    logdensity = -0.5 * log(2pi) - log(5.0) - 0.5 * (mu / 5.0)^2
    ratio_squared = (tau / 5.0)^2
    logdensity += log(2.0) - log(pi) - log(5.0) -
        log1p(ratio_squared) + log_tau

    gradient = similar(q)
    gradient_mu = -mu / 25.0
    gradient_log_tau = 1.0 - 2.0 * ratio_squared / (1.0 + ratio_squared)
    for index in eachindex(theta)
        delta = theta[index] - mu
        residual = EIGHT_Y[index] - theta[index]
        logdensity += -0.5 * log(2pi) - log_tau -
            0.5 * delta^2 * inverse_tau_squared
        logdensity += -0.5 * log(2pi) - log(EIGHT_SIGMA[index]) -
            0.5 * (residual / EIGHT_SIGMA[index])^2
        gradient_mu += delta * inverse_tau_squared
        gradient_log_tau += -1.0 + delta^2 * inverse_tau_squared
        gradient[index + 2] = -delta * inverse_tau_squared +
            residual / EIGHT_SIGMA[index]^2
    end
    gradient[1] = gradient_mu
    gradient[2] = gradient_log_tau
    (logdensity, gradient)
end

function _noncentered_eight_schools(q)
    mu = q[1]
    log_tau = q[2]
    tau = exp(log_tau)
    eta = @view q[3:10]

    logdensity = -0.5 * log(2pi) - log(5.0) - 0.5 * (mu / 5.0)^2
    ratio_squared = (tau / 5.0)^2
    logdensity += log(2.0) - log(pi) - log(5.0) -
        log1p(ratio_squared) + log_tau

    gradient = similar(q)
    gradient_mu = -mu / 25.0
    gradient_log_tau = 1.0 - 2.0 * ratio_squared / (1.0 + ratio_squared)
    for index in eachindex(eta)
        theta = mu + tau * eta[index]
        residual = EIGHT_Y[index] - theta
        inverse_sigma_squared = inv(EIGHT_SIGMA[index]^2)
        logdensity += -0.5 * log(2pi) - 0.5 * eta[index]^2
        logdensity += -0.5 * log(2pi) - log(EIGHT_SIGMA[index]) -
            0.5 * residual^2 * inverse_sigma_squared
        gradient_mu += residual * inverse_sigma_squared
        gradient_log_tau += residual * tau * eta[index] * inverse_sigma_squared
        gradient[index + 2] = -eta[index] +
            residual * tau * inverse_sigma_squared
    end
    gradient[1] = gradient_mu
    gradient[2] = gradient_log_tau
    (logdensity, gradient)
end

# Scalar log densities — the public model boundary DI+Enzyme differentiates.
_gaussian_logp(q) = -sum(abs2, q) / 2

function _centered_logp(q)
    mu = q[1]
    log_tau = q[2]
    tau = exp(log_tau)
    theta = @view q[3:10]
    inverse_tau_squared = inv(tau * tau)
    logdensity = -0.5 * log(2pi) - log(5.0) - 0.5 * (mu / 5.0)^2
    ratio_squared = (tau / 5.0)^2
    logdensity += log(2.0) - log(pi) - log(5.0) - log1p(ratio_squared) + log_tau
    for index in eachindex(theta)
        delta = theta[index] - mu
        residual = EIGHT_Y[index] - theta[index]
        logdensity += -0.5 * log(2pi) - log_tau -
            0.5 * delta^2 * inverse_tau_squared
        logdensity += -0.5 * log(2pi) - log(EIGHT_SIGMA[index]) -
            0.5 * (residual / EIGHT_SIGMA[index])^2
    end
    logdensity
end

function _noncentered_logp(q)
    mu = q[1]
    log_tau = q[2]
    tau = exp(log_tau)
    eta = @view q[3:10]
    logdensity = -0.5 * log(2pi) - log(5.0) - 0.5 * (mu / 5.0)^2
    ratio_squared = (tau / 5.0)^2
    logdensity += log(2.0) - log(pi) - log(5.0) - log1p(ratio_squared) + log_tau
    for index in eachindex(eta)
        theta = mu + tau * eta[index]
        residual = EIGHT_Y[index] - theta
        inverse_sigma_squared = inv(EIGHT_SIGMA[index]^2)
        logdensity += -0.5 * log(2pi) - 0.5 * eta[index]^2
        logdensity += -0.5 * log(2pi) - log(EIGHT_SIGMA[index]) -
            0.5 * residual^2 * inverse_sigma_squared
    end
    logdensity
end

# The scalar log density DI+Enzyme differentiates for each target.
_logp(::GaussianTarget) = _gaussian_logp
_logp(::CenteredEightSchoolsTarget) = _centered_logp
_logp(::NoncenteredEightSchoolsTarget) = _noncentered_logp

_evaluate(target::GaussianTarget, q) = _gaussian(q)
_evaluate(::CenteredEightSchoolsTarget, q) = _centered_eight_schools(q)
_evaluate(::NoncenteredEightSchoolsTarget, q) = _noncentered_eight_schools(q)

function LogDensityProblems.logdensity(target::CountedTarget, q)
    target.logdensity_calls += 1
    _logp(target)(q)
end

# External LDP consumers (AdvancedHMC / DynamicHMC) may retain the gradient of a
# phasepoint, so the gradient returned here must be a FRESH array per call. Use
# DI's out-of-place value_and_gradient with the concrete preparation. Owned
# in-place gradient buffers are reserved for RK's phasepoint slots, where the
# reactive program explicitly owns the memory.
function LogDensityProblems.logdensity_and_gradient(target::CountedTarget, q)
    target.gradient_calls += 1
    value_and_gradient(_logp(target), target.preparation, BENCHMARK_BACKEND, q)
end

initial(::GaussianTarget) = zeros(4)
initial(::CenteredEightSchoolsTarget) = [0.0, log(5.0), zeros(8)...]
initial(::NoncenteredEightSchoolsTarget) = [0.0, log(5.0), zeros(8)...]

# Validate the DI+Enzyme gradient of the scalar log density against the
# closed-form analytic gradient at a fixed point. The analytic form is a
# correctness ORACLE only — never the sampled gradient path — and no numerical
# differencing is used.
function validate_target_gradient(logp, analytic_evaluate, q)
    _, analytic = analytic_evaluate(q)
    preparation = prepare_gradient(logp, BENCHMARK_BACKEND, copy(q))
    di_gradient =
        DifferentiationInterface.gradient(logp, preparation, BENCHMARK_BACKEND, q)
    @assert isapprox(analytic, di_gradient; rtol = 2.0e-6, atol = 2.0e-7)
    maximum(abs, analytic - di_gradient)
end

function validate_target_gradients()
    q = [0.3, log(1.7), -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
    println((;
        centered_gradient_max_error = validate_target_gradient(
            _centered_logp, _centered_eight_schools, q,
        ),
        noncentered_gradient_max_error = validate_target_gradient(
            _noncentered_logp, _noncentered_eight_schools, q,
        ),
    ))
end

function reset_counts!(target)
    target.logdensity_calls = 0
    target.gradient_calls = 0
    target
end

function prepare_reactive_target(target)
    initial_position = initial(target)
    timed = @timed begin
        # Public compiled-reactive boundary: the scalar potential (= -logdensity)
        # is differentiated by DI+Enzyme (prepared once on the target) into the
        # sampler's OWNED gradient buffer in place, then negated in place. No
        # handwritten sampled gradient; RK's own gradient counter is incremented.
        function potential_gradient!(gradient, position)
            target.gradient_calls += 1
            value = first(value_and_gradient!(
                _logp(target), gradient, target.preparation,
                BENCHMARK_BACKEND, position))
            gradient .*= -1
            -value
        end
        dimension = length(initial_position)
        ReactiveKernels.reactive_nuts_group(
            potential_gradient!,
            Matrix{Float64}(I, dimension, dimension),
            copy(initial_position),
            zeros(dimension))
    end
    (;
        target,
        point = timed.value,
        setup_seconds = timed.time,
        setup_bytes = timed.bytes,
    )
end

function run_reactive(context; seed, n_warmup, n_draws, max_depth,
                      target_accept)
    reset_counts!(context.target)
    timed = @timed begin
        # nuts_state on the flat reactive_nuts_group returns a CompiledNUTSState.
        sampler = ReactiveKernels.nuts_state(
            copy(context.point);
            rng = Xoshiro(seed),
            step_f = ReactiveKernels.partial(
                ReactiveKernels.leapfrog!; stepsize = 0.1,
            ),
            max_depth,
        )
        warmup = ReactiveKernels.warmup!(
            sampler, n_warmup; target_accept,
        )
        chain = ReactiveKernels.sample!(sampler, n_draws)
        (; chain, warmup)
    end
    (; chain, warmup) = timed.value
    stats = chain.diagnostics
    @assert all(isfinite, chain.samples)
    (;
        implementation = :ReactiveKernels,
        samples = chain.samples,
        seconds = timed.time,
        bytes = timed.bytes,
        gradients = context.target.gradient_calls,
        logdensities = context.target.logdensity_calls,
        divergences = count(stat -> stat.diverged, stats),
        max_depth_hits = count(stat -> stat.depth == max_depth, stats),
        mean_acceptance = mean(stat -> stat.acceptance_rate, stats),
        mean_steps = mean(stat -> stat.n_steps, stats),
        stepsize = warmup.final_stepsize,
    )
end

function run_advancedhmc(target; seed, n_warmup, n_draws, max_depth,
                         target_accept)
    initial_position = initial(target)
    timed = @timed begin
        rng = Xoshiro(seed)
        metric = AdvancedHMC.DiagEuclideanMetric(length(initial_position))
        hamiltonian = AdvancedHMC.Hamiltonian(metric, target)
        stepsize = AdvancedHMC.find_good_stepsize(
            rng, hamiltonian, initial_position,
        )
        integrator = AdvancedHMC.Leapfrog(stepsize)
        kernel = AdvancedHMC.HMCKernel(
            AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
                integrator,
                AdvancedHMC.GeneralisedNoUTurn(max_depth = max_depth),
            ),
        )
        adaptor = AdvancedHMC.StanHMCAdaptor(
            AdvancedHMC.MassMatrixAdaptor(metric),
            AdvancedHMC.StepSizeAdaptor(target_accept, integrator),
        )
        AdvancedHMC.sample(
            rng,
            hamiltonian,
            kernel,
            initial_position,
            n_warmup + n_draws,
            adaptor,
            n_warmup;
            drop_warmup = true,
            verbose = false,
            progress = false,
        )
    end
    draws, stats = timed.value
    samples = reduce(hcat, draws)
    @assert all(isfinite, samples)
    (;
        implementation = :AdvancedHMC,
        samples,
        seconds = timed.time,
        bytes = timed.bytes,
        gradients = target.gradient_calls,
        logdensities = target.logdensity_calls,
        divergences = count(stat -> stat.numerical_error, stats),
        max_depth_hits = count(stat -> stat.tree_depth == max_depth, stats),
        mean_acceptance = mean(stat -> stat.acceptance_rate, stats),
        mean_steps = mean(stat -> stat.n_steps, stats),
        stepsize = stats[end].step_size,
    )
end

function run_dynamichmc(target; seed, n_warmup, n_draws, max_depth,
                        target_accept)
    n_warmup == 900 || error(
        "DynamicHMC comparison uses its exact 900-transition default schedule",
    )
    initial_position = initial(target)
    stages = DynamicHMC.default_warmup_stages(
        stepsize_search = DynamicHMC.InitialStepsizeSearch(),
        stepsize_adaptation = DynamicHMC.DualAveraging(δ = target_accept),
    )
    timed = @timed DynamicHMC.mcmc_with_warmup(
        Xoshiro(seed),
        target,
        n_draws;
        initialization = (q = initial_position,),
        warmup_stages = stages,
        algorithm = DynamicHMC.NUTS(max_depth = max_depth),
        reporter = DynamicHMC.NoProgressReport(),
    )
    chain = timed.value
    stats = chain.tree_statistics
    @assert all(isfinite, chain.posterior_matrix)
    (;
        implementation = :DynamicHMC,
        samples = chain.posterior_matrix,
        seconds = timed.time,
        bytes = timed.bytes,
        gradients = target.gradient_calls,
        logdensities = target.logdensity_calls,
        divergences = count(stat -> is_divergent(stat.termination), stats),
        max_depth_hits = count(
            stat -> stat.termination == REACHED_MAX_DEPTH, stats,
        ),
        mean_acceptance = mean(stat -> stat.acceptance_rate, stats),
        mean_steps = mean(stat -> stat.steps, stats),
        stepsize = chain.ϵ,
    )
end

function aggregate(chains; setup_seconds = 0.0, setup_bytes = 0)
    chain_count = length(chains)
    dimension, draw_count = size(first(chains).samples)
    stacked = Array{Float64}(undef, draw_count, chain_count, dimension)
    for (chain_index, chain) in pairs(chains)
        stacked[:, chain_index, :] .= permutedims(chain.samples)
    end
    effective = vec(MCMCDiagnosticTools.ess(stacked; kind = :bulk))
    rhats = vec(MCMCDiagnosticTools.rhat(stacked))
    pooled = reduce(hcat, (chain.samples for chain in chains))
    sampling_seconds = sum(chain -> chain.seconds, chains)
    total_seconds = setup_seconds + sampling_seconds
    total_gradients = sum(chain -> chain.gradients, chains)
    (;
        implementation = first(chains).implementation,
        chains = chain_count,
        draws_per_chain = draw_count,
        setup_seconds,
        sampling_seconds,
        total_seconds,
        mebibytes =
            (setup_bytes + sum(chain -> chain.bytes, chains)) / 2.0^20,
        gradients = total_gradients,
        logdensities = sum(chain -> chain.logdensities, chains),
        divergences = sum(chain -> chain.divergences, chains),
        max_depth_hits = sum(chain -> chain.max_depth_hits, chains),
        mean_acceptance = mean(chain -> chain.mean_acceptance, chains),
        mean_steps = mean(chain -> chain.mean_steps, chains),
        stepsizes = [chain.stepsize for chain in chains],
        min_bulk_ess = minimum(effective),
        median_bulk_ess = median(effective),
        max_rhat = maximum(rhats),
        min_ess_per_second = minimum(effective) / total_seconds,
        min_ess_per_sampling_second = minimum(effective) / sampling_seconds,
        min_ess_per_gradient = minimum(effective) / total_gradients,
        means = vec(mean(pooled; dims = 2)),
        variances = vec(var(pooled; dims = 2)),
    )
end

function target_constructor(target)
    target isa GaussianTarget && return () -> GaussianTarget(4)
    target isa CenteredEightSchoolsTarget &&
        return () -> CenteredEightSchoolsTarget()
    () -> NoncenteredEightSchoolsTarget()
end

function compile_paths()
    for target in (
        GaussianTarget(4),
        CenteredEightSchoolsTarget(),
        NoncenteredEightSchoolsTarget(),
    )
        context = prepare_reactive_target(target)
        run_reactive(
            context;
            seed = 1,
            n_warmup = 100,
            n_draws = 10,
            max_depth = 8,
            target_accept = 0.8,
        )
        constructor = target_constructor(target)
        run_advancedhmc(
            constructor();
            seed = 1,
            n_warmup = 100,
            n_draws = 10,
            max_depth = 8,
            target_accept = 0.8,
        )
        run_dynamichmc(
            constructor();
            seed = 1,
            n_warmup = 900,
            n_draws = 10,
            max_depth = 8,
            target_accept = 0.8,
        )
    end
    nothing
end

function compare(name, constructor; target_accept)
    seeds = 20260825:20260828
    settings = (;
        warmup = 900,
        draws = 1_000,
        max_depth = 8,
        target_accept,
        seeds = collect(seeds),
    )
    println((; model = name, settings))
    for runner in (run_reactive, run_advancedhmc, run_dynamichmc)
        GC.gc()
        if runner === run_reactive
            # Setup = target DI+Enzyme preparation (timed) + reactive graph
            # preparation (timed inside prepare_reactive_target), both once.
            built = @timed constructor()
            context = prepare_reactive_target(built.value)
            chains = [
                runner(
                    context;
                    seed,
                    n_warmup = settings.warmup,
                    n_draws = settings.draws,
                    max_depth = settings.max_depth,
                    target_accept,
                ) for seed in seeds
            ]
            println(aggregate(
                chains;
                setup_seconds = built.time + context.setup_seconds,
                setup_bytes = built.bytes + context.setup_bytes,
            ))
        else
            # Per-chain target instances (clean call counters); the DI+Enzyme
            # preparation cost of each construction is timed and summed into
            # setup. Collect the timed builds first, then sum their fields, so no
            # mutable accumulator is captured into the sampling closure.
            builds = [@timed(constructor()) for _ in seeds]
            chains = [
                runner(
                    build.value;
                    seed,
                    n_warmup = settings.warmup,
                    n_draws = settings.draws,
                    max_depth = settings.max_depth,
                    target_accept,
                ) for (seed, build) in zip(seeds, builds)
            ]
            println(aggregate(
                chains;
                setup_seconds = sum(build -> build.time, builds),
                setup_bytes = sum(build -> build.bytes, builds),
            ))
        end
    end
end

function package_receipts()
    dependencies = Pkg.dependencies()
    receipts = map(COMPARISON_PACKAGES) do name
        info = only(info for info in values(dependencies) if info.name == name)
        (;
            name,
            version = info.version,
            tree_hash = info.tree_hash,
            git_revision = info.git_revision,
            git_source = info.git_source,
            tracking_path = info.is_tracking_path,
        )
    end
    println((;
        candidate_sha = ENV["REACTIVEKERNELS_CANDIDATE_SHA"],
        julia = string(VERSION),
        packages = receipts,
    ))
end

package_receipts()
validate_target_gradients()
compile_paths()
compare(:gaussian, () -> GaussianTarget(4); target_accept = 0.8)
compare(
    :centered_eight_schools,
    () -> CenteredEightSchoolsTarget();
    target_accept = 0.8,
)
compare(
    :centered_eight_schools,
    () -> CenteredEightSchoolsTarget();
    target_accept = 0.9,
)
compare(
    :noncentered_eight_schools,
    () -> NoncenteredEightSchoolsTarget();
    target_accept = 0.8,
)
