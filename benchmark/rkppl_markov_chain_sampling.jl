# A random walk observed only in its middle, with a sampled step size: the
# case WarmupHMC's MarkovChain blocks were built for (snag
# random-walk-bloc-fc175e21, design decision 1idvzrg). Compares the walk
# written non-centred and written centred, each sampled as written, with
# the same walk marked by `markov_chains`, whose MarkovChain adapts its
# centring per time step (RKPPL with `markov_chains`, WarmupHMC >= 0.3.1).
#
# Run single-threaded in an environment with ReactiveKernelsPPL, WarmupHMC,
# Enzyme (the dogfood line), DifferentiationInterface, LogDensityProblems
# and MCMCDiagnosticTools:
#
#     julia --project=<env> benchmark/rkppl_markov_chain_sampling.jl
#
# Four chains per arm run one after another. Every arm's draws are mapped to
# the same quantities (log_q and the T states) before ESS and R-hat, so the
# arms compare on one footing. Gradient evaluations are counted at the
# model's own LogDensityProblems target.
using ReactiveKernelsPPL, WarmupHMC, Enzyme, Random, Statistics
using ReactiveKernelsPPL: adaptive_centering_problem
using DifferentiationInterface: AutoEnzyme
import LogDensityProblems as LDP
import MCMCDiagnosticTools as MCD

const NONCENTRED = @rkppl begin
    log_q ~ Normal(log(0.6), 1)
    @scan begin
        w[1] ~ Normal(0, 3)
        for t in 2:T
            e ~ Normal(0, 1)
            w[t] = w[t - 1] + exp(log_q) * e
        end
    end
    y .~ Normal.(w[obs], sigma)
end

const CENTRED = @rkppl begin
    log_q ~ Normal(log(0.6), 1)
    @scan begin
        w[1] ~ Normal(0, 3)
        for t in 2:T
            w[t] ~ Normal(w[t - 1], exp(log_q))
        end
    end
    y .~ Normal.(w[obs], sigma)
end

# Counts gradient evaluations of the target it wraps.
mutable struct Counted{P}
    problem::P
    gradients::Int
end
LDP.dimension(c::Counted) = LDP.dimension(c.problem)
LDP.capabilities(::Type{Counted{P}}) where {P} = LDP.capabilities(P)
LDP.logdensity(c::Counted, x) = LDP.logdensity(c.problem, x)
function LDP.logdensity_and_gradient(c::Counted, x)
    c.gradients += 1
    return LDP.logdensity_and_gradient(c.problem, x)
end

function simulate(seed; T = 300, gap = 60, q = 0.6, sigma = 1.6)
    rng = Xoshiro(seed)
    w = cumsum([3 * randn(rng); q .* randn(rng, T - 1)])
    obs = collect((gap + 1):(T - gap))
    return (; T, obs, sigma, y = w[obs] .+ sigma .* randn(rng, length(obs)))
end

# log_q and the states of one draw, from the arm's packed coordinates.
function states(arm, layout, u)
    c = constrain(layout, u)
    arm === :noncentred || return [c.log_q; arm === :centred ? c.w : c._ppl_scan_z_w]
    z = c._ppl_scan_z_w            # the seed, then the innovations
    return [c.log_q; cumsum([z[1]; exp(c.log_q) .* z[2:end]])]
end

function compare(seed)
    data = simulate(seed)
    inputs = (; data.T, data.obs, data.sigma)
    plans = (
        noncentred = NONCENTRED(; inputs...) | (; data.y),
        centred = CENTRED(; inputs...) | (; data.y),
        chain = markov_chains(NONCENTRED(; inputs...), :w) | (; data.y))
    for (arm, plan) in pairs(plans)
        built = build_kernel(plan)
        u0 = zeros(built.layout.total)
        cubes, gradients, draws = [], 0, 0
        t0 = time()
        for chain in 1:4
            counted = Counted(prepare_sampler(built, plan, u0; backend = AutoEnzyme()), 0)
            target = arm === :chain ?
                adaptive_centering_problem(built, plan, counted, AutoEnzyme(); u0) : counted
            result = adaptive_warmup_mcmc(Xoshiro(100 + chain), target;
                n_draws = 1000, monitor_ess = true)
            positions = Matrix(result.posterior_position)
            push!(cubes, reduce(hcat, (states(arm, built.layout, positions[:, k])
                for k in axes(positions, 2))))
            gradients += counted.gradients
            draws += size(positions, 2)
        end
        elapsed = time() - t0
        nd = minimum(size(c, 2) for c in cubes)
        cube = permutedims(cat([permutedims(c[:, end-nd+1:end]) for c in cubes]...;
            dims = 3), (1, 3, 2))
        ess = MCD.ess(cube; kind = :bulk)
        rhat = MCD.rhat(cube)
        unobserved = [1:first(data.obs) - 1; last(data.obs) + 1:data.T] .+ 1
        println((; seed, arm, common_draws = nd,
            gradients_per_draw = round(gradients / draws; digits = 1),
            min_bulk_ess = round(minimum(ess); digits = 1),
            ess_log_q = round(ess[1]; digits = 1),
            min_bulk_ess_unobserved = round(minimum(ess[unobserved]); digits = 1),
            ess_per_1000_gradients = round(1000 * minimum(ess) / gradients; digits = 2),
            max_rhat = round(maximum(rhat); digits = 3),
            mean_log_q = round(mean(cube[:, :, 1]); digits = 3),
            elapsed = round(elapsed; digits = 1)))
        flush(stdout)
    end
end

println((; julia = string(VERSION), threads = Threads.nthreads(),
    ppl = string(pkgversion(ReactiveKernelsPPL)), warmuphmc = string(pkgversion(WarmupHMC)),
    libenzyme = Enzyme.dogfood_libenzyme().matches))
foreach(compare, (1, 2))
