# Fairness check: give single-site CGGibbs a long (cheap) warmup + many draws on a
# representative n<d correlated case, and see whether it CONVERGES to the NUTS
# posterior and what its converged bulk-ESS/second is. (Its O(d) sweeps are cheap,
# so a long chain is the advantage the paper leans on.)
include("/home/n/.local/state/kb-agents-worktrees/ReactiveKernels-sampling-gibbs/benchmark/cggibbs/cggibbs_vs_nuts.jl")
using Statistics: mean

function fair()
    s0sq = 100.0
    for (n, d, rho) in ((60, 200, 0.6), (200, 100, 0.6))
        dt = make_logistic(StableRNG(21); n = n, d = d, k = min(5, d), snr = 1.0, rho = rho)
        target = LogisticTarget(dt.X, dt.y, s0sq)
        nu = run_nuts(target; seed = 7, n_warmup = 1000, n_draws = 2000)
        mn = vec(mean(nu.samples; dims = 2))
        sn = ess_summary(nu.samples, nu.seconds)
        println("="^70)
        println("n=$n d=$d rho=$rho")
        println("  NUTS: minESS ", round(Int, sn.min_ess), " in ", round(nu.seconds; digits = 1),
                "s -> ", round(sn.min_ess_per_sec; digits = 1), " ESS/s")
        for (w, nd) in ((5000, 10000), (20000, 40000))
            cg = run_cggibbs(dt.X, dt.y, s0sq, cond; seed = 7, n_warmup = w, n_draws = nd)
            mc = vec(mean(cg.samples; dims = 2))
            sc = ess_summary(cg.samples, cg.seconds)
            println("  CGGibbs warmup=", w, " draws=", nd, ": minESS ", round(Int, sc.min_ess),
                    " in ", round(cg.seconds; digits = 1), "s -> ",
                    round(sc.min_ess_per_sec; digits = 2), " ESS/s",
                    "   max|Δmean vs NUTS|=", round(maximum(abs, mn .- mc); digits = 3))
        end
    end
    println("="^70)
end
fair()
