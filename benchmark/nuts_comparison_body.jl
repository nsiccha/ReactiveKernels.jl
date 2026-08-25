using AdvancedHMC
using DynamicHMC
using DynamicHMC.Diagnostics: REACHED_MAX_DEPTH, is_divergent
using LinearAlgebra
using LogDensityProblems
using MCMCDiagnosticTools
using Pkg
using Random
using ReactiveKernels
using Statistics

abstract type CountedTarget end

mutable struct GaussianTarget <: CountedTarget
    dimension::Int
    logdensity_calls::Int
    gradient_calls::Int
end

mutable struct CenteredEightSchoolsTarget <: CountedTarget
    logdensity_calls::Int
    gradient_calls::Int
end

mutable struct NoncenteredEightSchoolsTarget <: CountedTarget
    logdensity_calls::Int
    gradient_calls::Int
end

const EIGHT_Y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
const EIGHT_SIGMA = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]
const COMPARISON_PACKAGES = (
    "ReactiveKernels",
    "MutatingFunctions",
    "AdvancedHMC",
    "DynamicHMC",
    "LogDensityProblems",
    "MCMCDiagnosticTools",
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

_evaluate(target::GaussianTarget, q) = _gaussian(q)
_evaluate(::CenteredEightSchoolsTarget, q) = _centered_eight_schools(q)
_evaluate(::NoncenteredEightSchoolsTarget, q) = _noncentered_eight_schools(q)

function LogDensityProblems.logdensity(target::CountedTarget, q)
    target.logdensity_calls += 1
    first(_evaluate(target, q))
end

function LogDensityProblems.logdensity_and_gradient(target::CountedTarget, q)
    target.gradient_calls += 1
    _evaluate(target, q)
end

initial(::GaussianTarget) = zeros(4)
initial(::CenteredEightSchoolsTarget) = [0.0, log(5.0), zeros(8)...]
initial(::NoncenteredEightSchoolsTarget) = [0.0, log(5.0), zeros(8)...]

function validate_target_gradient(evaluate, q)
    _, analytic = evaluate(q)
    increment = 1.0e-6
    finite_difference = similar(q)
    for index in eachindex(q)
        forward = copy(q)
        backward = copy(q)
        forward[index] += increment
        backward[index] -= increment
        finite_difference[index] =
            (first(evaluate(forward)) - first(evaluate(backward))) /
            (2increment)
    end
    @assert isapprox(
        analytic, finite_difference; rtol = 2.0e-6, atol = 2.0e-7,
    )
    maximum(abs, analytic - finite_difference)
end

function validate_target_gradients()
    q = [0.3, log(1.7), -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
    println((;
        centered_gradient_max_error = validate_target_gradient(
            _centered_eight_schools, q,
        ),
        noncentered_gradient_max_error = validate_target_gradient(
            _noncentered_eight_schools, q,
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
        potential(position) = -LogDensityProblems.logdensity(target, position)
        function potential_gradient(position)
            logdensity, gradient =
                LogDensityProblems.logdensity_and_gradient(target, position)
            (-logdensity, -gradient)
        end
        model = ReactiveKernels.@kernel begin
            pos::Vector{Float64}
            mom::Vector{Float64}
            metric::Matrix{Float64}
            chol_metric::Cholesky{Float64,Matrix{Float64}} = cholesky(metric)
            pot::Float64 = potential(pos)
            (pot, dpot_dpos::Vector{Float64}) = potential_gradient(pos)
            (kin::Float64, dham_dmom::Vector{Float64}) = begin
                velocity = chol_metric \ mom
                (0.5 * (logdet(chol_metric) + dot(mom, velocity)), velocity)
            end
            ham::Float64 = pot + kin
            dham_dpos::Vector{Float64} = dpot_dpos
            return (pot, dpot_dpos, chol_metric, kin, ham,
                    dham_dpos, dham_dmom)
        end
        dimension = length(initial_position)
        ReactiveKernels.euclidean_phasepoint(model, (
            pos = copy(initial_position),
            mom = zeros(dimension),
            metric = Matrix{Float64}(I, dimension, dimension),
        ))
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
    target isa GaussianTarget && return () -> GaussianTarget(4, 0, 0)
    target isa CenteredEightSchoolsTarget &&
        return () -> CenteredEightSchoolsTarget(0, 0)
    () -> NoncenteredEightSchoolsTarget(0, 0)
end

function compile_paths()
    for target in (
        GaussianTarget(4, 0, 0),
        CenteredEightSchoolsTarget(0, 0),
        NoncenteredEightSchoolsTarget(0, 0),
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
            context = prepare_reactive_target(constructor())
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
                setup_seconds = context.setup_seconds,
                setup_bytes = context.setup_bytes,
            ))
        else
            chains = [
                runner(
                    constructor();
                    seed,
                    n_warmup = settings.warmup,
                    n_draws = settings.draws,
                    max_depth = settings.max_depth,
                    target_accept,
                ) for seed in seeds
            ]
            println(aggregate(chains))
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
compare(:gaussian, () -> GaussianTarget(4, 0, 0); target_accept = 0.8)
compare(
    :centered_eight_schools,
    () -> CenteredEightSchoolsTarget(0, 0);
    target_accept = 0.8,
)
compare(
    :centered_eight_schools,
    () -> CenteredEightSchoolsTarget(0, 0);
    target_accept = 0.9,
)
compare(
    :noncentered_eight_schools,
    () -> NoncenteredEightSchoolsTarget(0, 0);
    target_accept = 0.8,
)
