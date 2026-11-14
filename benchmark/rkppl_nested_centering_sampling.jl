# Nested adaptive centering on the generic two-level toy of snag
# adaptive-affine-d6c2fa03: group effects z_g about mu, unit effects z_u about
# mu + tau_g * z_g[group], with data that pin every unit value much more tightly
# than its spread. Compares the reporter's two single-level shapes with both
# levels centred about their parents (RKPPL 0.1.185, WarmupHMC >= 0.3.0),
# adaptive and at two fixed centerings.
#
# Run single-threaded in an environment with ReactiveKernelsPPL, WarmupHMC,
# Enzyme, DifferentiationInterface, LogDensityProblems and MCMCDiagnosticTools:
#
#     julia --project=<env> benchmark/rkppl_nested_centering_sampling.jl
#
# Four chains per arm run one after another. Timing includes each arm's
# first-use compilation. Reported diagnostics are the warmuphmc-use §0 set.
using ReactiveKernelsPPL, WarmupHMC, Enzyme, Random, Statistics
using ReactiveKernelsPPL: adaptive_centering_problem
using DifferentiationInterface: AutoEnzyme
import MCMCDiagnosticTools as MCD

const NESTED = @rkppl begin
    mu ~ Normal(0, mu_sd)
    tau_g ~ Exponential(tau_g_scale)
    tau_u ~ Exponential(tau_u_scale)
    z_g[1:G] .~ Normal.(0, 1)
    z_u[1:U] .~ Normal.(0, 1)
    unit_location = mu .+ tau_g .* z_g[group]
    log_tau_g = log(tau_g)
    log_tau_u = log(tau_u)
    theta = unit_location .+ tau_u .* z_u
    y .~ Normal.(theta, sigma)
end

nested_cells(sg = 0.0, su = 0.0) = (
    z_g = (location = :mu, log_scale = :log_tau_g, source = sg),
    z_u = (location = :unit_location, log_scale = :log_tau_u, source = su))

# Each arm: cells, and whether WarmupHMC adapts their centering.
const ARMS = (
    # The reporter's shapes before RKPPL 0.1.185:
    unit_only = ((z_u = (location = :unit_location, log_scale = :log_tau_u),), true),
    both_about_mu = ((z_g = (location = :mu, log_scale = :log_tau_g),
        z_u = (location = :mu, log_scale = :log_tau_u)), true),
    # Both levels, each about its parent's model value:
    nested = (nested_cells(), true),
    # Fixed controls. (0, 1) is the same map as unit_only with the unit level
    # centred, the centering unit_only adapts to; (1, 1) centres both levels.
    nested_fixed_0_1 = (nested_cells(0.0, 1.0), false),
    nested_fixed_1_1 = (nested_cells(1.0, 1.0), false),
)

function simulate(seed; G = 3, per = 3, sigma = 0.05)
    rng = Xoshiro(seed)
    group = repeat(1:G; inner = per)
    U = length(group)
    mu, tau_g, tau_u = 0.3, 0.8, 0.4
    theta = mu .+ tau_g .* randn(rng, G)[group] .+ tau_u .* randn(rng, U)
    (; G, U, group, sigma, y = theta .+ sigma .* randn(rng, U))
end

function compare(seed)
    data = simulate(seed)
    plan = NESTED(; data.G, data.U, data.group, data.sigma, mu_sd = 2.0,
        tau_g_scale = 1.0, tau_u_scale = 0.5) | (; data.y)
    built = build_kernel(plan)
    u0 = zeros(built.layout.total)
    names = string.(coordinate_names(built.layout))
    for (arm, (cells, adapt)) in pairs(ARMS)
        results = []
        controls = []
        t0 = time()
        for chain in 1:4
            problem = prepare_sampler(built, plan, u0; backend = AutoEnzyme())
            rp = adaptive_centering_problem(built, plan, problem, AutoEnzyme(); cells, u0)
            result = adaptive_warmup_mcmc(Xoshiro(100 + chain), rp;
                n_draws = 1000, monitor_ess = true, nonlinear_adapt = adapt)
            push!(results, result)
            push!(controls, unique(round.([last(p).c for p in WarmupHMC.reparam_sources(rp)];
                digits = 2)))
        end
        elapsed = time() - t0
        nd = minimum(size(r.posterior_position, 2) for r in results)
        cube = permutedims(cat([permutedims(Matrix(r.posterior_position)[:, end-nd+1:end])
            for r in results]...; dims = 3), (1, 3, 2))
        rhat = MCD.rhat(cube)
        draws = sum(size(r.posterior_position, 2) for r in results)
        divergent = sum(r.n_divergent_samples for r in results)
        # Posterior means agree across arms when every arm returns model-frame draws.
        means = vec(mean(cube; dims = (1, 2)))
        mu = findfirst(==("mu"), names)
        println((; seed, arm, adapt, common_draws = nd, max_rhat = maximum(rhat),
            slowest = names[argmax(rhat)],
            min_bulk_ess = minimum(MCD.ess(cube; kind = :bulk)),
            min_tail_ess = minimum(MCD.ess(cube; kind = :tail)), divergent,
            divergent_percent = 100 * divergent / draws,
            mean_mu = means[mu], sd_mu = std(vec(cube[:, :, mu])),
            mcse_mu = MCD.mcse(cube[:, :, mu:mu])[1],
            mean_z_g = means[findall(startswith("z_g"), names)],
            elapsed, adapted_sources = controls))
        flush(stdout)
    end
end

println((; julia = string(VERSION), threads = Threads.nthreads(),
    ppl = string(pkgversion(ReactiveKernelsPPL)), warmuphmc = string(pkgversion(WarmupHMC))))
foreach(compare, (1, 2, 3))
