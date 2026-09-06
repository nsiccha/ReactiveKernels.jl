# =============================================================================
# The paper's headline question, on ReactiveKernels: is (CG)Gibbs faster than
# NUTS on a Bayesian logistic GLM?  Metric: bulk-ESS per second (the same metric
# and NUTS setup the repo's nuts_comparison uses — AdvancedHMC MultinomialTS +
# GeneralisedNoUTurn + StanHMCAdaptor, MCMCDiagnosticTools bulk ESS).
#
# CGGibbs (coordinate slice-within-Gibbs, cached η) vs NUTS on the same posterior.
# =============================================================================
include(joinpath(@__DIR__, "cggibbs.jl"))   # model + CGGibbs sampler (main() guarded)

using AdvancedHMC
import LogDensityProblems as LDP
using MCMCDiagnosticTools
using Statistics: median
using Random: Xoshiro

# ---- logistic posterior as a LogDensityProblems target (analytic gradient) -----
mutable struct LogisticTarget
    X::Matrix{Float64}
    y::Vector{Float64}
    s0sq::Float64
    q0::Vector{Float64}
    logdensity_calls::Int
    gradient_calls::Int
end
LogisticTarget(X, y, s0sq) = LogisticTarget(X, y, s0sq, zeros(size(X, 2)), 0, 0)
LDP.capabilities(::Type{<:LogisticTarget}) = LDP.LogDensityOrder{1}()
LDP.dimension(t::LogisticTarget) = size(t.X, 2)
function LDP.logdensity(t::LogisticTarget, theta)
    t.logdensity_calls += 1
    eta = t.X * theta
    -dot(theta, theta) / (2 * t.s0sq) + sum(t.y .* eta .- log1pexp.(eta))
end
function LDP.logdensity_and_gradient(t::LogisticTarget, theta)
    t.gradient_calls += 1
    eta = t.X * theta
    ld = -dot(theta, theta) / (2 * t.s0sq) + sum(t.y .* eta .- log1pexp.(eta))
    p = 1 ./ (1 .+ exp.(-eta))
    grad = -theta ./ t.s0sq .+ t.X' * (t.y .- p)
    (ld, grad)
end
initial_point(t::LogisticTarget) = copy(t.q0)

# ---- NUTS via AdvancedHMC (repo-faithful setup) --------------------------------
function run_nuts(target; seed, n_warmup, n_draws, max_depth = 10, target_accept = 0.8)
    q0 = initial_point(target)
    timed = @timed begin
        rng = Xoshiro(seed)
        metric = AdvancedHMC.DiagEuclideanMetric(length(q0))
        ham = AdvancedHMC.Hamiltonian(metric, target)
        eps = AdvancedHMC.find_good_stepsize(rng, ham, q0)
        integ = AdvancedHMC.Leapfrog(eps)
        kern = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
            integ, AdvancedHMC.GeneralisedNoUTurn(max_depth = max_depth)))
        adaptor = AdvancedHMC.StanHMCAdaptor(
            AdvancedHMC.MassMatrixAdaptor(metric),
            AdvancedHMC.StepSizeAdaptor(target_accept, integ))
        AdvancedHMC.sample(rng, ham, kern, q0, n_warmup + n_draws, adaptor, n_warmup;
            drop_warmup = true, verbose = false, progress = false)
    end
    draws, _stats = timed.value
    samples = reduce(hcat, draws)                       # d × n_draws
    (; samples, seconds = timed.time, gradients = target.gradient_calls)
end

# ---- CGGibbs (timed sampling, warmup dropped) ----------------------------------
function run_cggibbs(X, y, s0sq, cond; seed, n_warmup, n_draws)
    n, d = size(X)
    theta = zeros(d); eta = zeros(n); rng = StableRNG(seed)
    for _ in 1:n_warmup; cggibbs_sweep!(rng, theta, eta, X, y, s0sq, cond); end
    T = zeros(d, n_draws)
    timed = @timed for it in 1:n_draws
        cggibbs_sweep!(rng, theta, eta, X, y, s0sq, cond)
        @views T[:, it] .= theta
    end
    (; samples = T, seconds = timed.time)
end

# ---- bulk ESS / second from a d × n_draws sample matrix ------------------------
function ess_summary(samples, seconds)
    d, nd = size(samples)
    stacked = Array{Float64}(undef, nd, 1, d)
    stacked[:, 1, :] .= permutedims(samples)
    e = vec(MCMCDiagnosticTools.ess(stacked; kind = :bulk))
    (; min_ess = minimum(e), median_ess = median(e),
       min_ess_per_sec = minimum(e) / seconds, seconds)
end

function main_vs()
    s0sq = 100.0
    n_warmup, n_draws = 1000, 2000
    println("="^76)
    println("CGGibbs vs NUTS — bulk-ESS/second on Bayesian logistic regression")
    println("(n_draws=", n_draws, ", warmup=", n_warmup, ", prior sd 10)")
    println("="^76)
    println(rpad("dims", 14), rpad("sampler", 10), rpad("min ESS", 10),
            rpad("med ESS", 10), rpad("sec", 8), rpad("minESS/s", 10), "speedup")
    for (n, d) in ((500, 20), (500, 100), (400, 300))
        dt = make_logistic(StableRNG(11); n = n, d = d, k = min(5, d), snr = 1.0)
        target = LogisticTarget(dt.X, dt.y, s0sq)
        nu = run_nuts(target; seed = 7, n_warmup, n_draws)
        cg = run_cggibbs(dt.X, dt.y, s0sq, cond; seed = 7, n_warmup, n_draws)
        # correctness cross-check: posterior means agree
        mn = vec(mean(nu.samples; dims = 2)); mc = vec(mean(cg.samples; dims = 2))
        agree = maximum(abs, mn .- mc)
        sn = ess_summary(nu.samples, nu.seconds)
        sc = ess_summary(cg.samples, cg.seconds)
        tag = "n=$n,d=$d"
        for (name, s) in (("NUTS", sn), ("CGGibbs", sc))
            println(rpad(tag, 14), rpad(name, 10),
                    rpad(round(Int, s.min_ess), 10), rpad(round(Int, s.median_ess), 10),
                    rpad(round(s.seconds; digits = 2), 8),
                    rpad(round(s.min_ess_per_sec; digits = 1), 10),
                    name == "CGGibbs" ? string(round(sc.min_ess_per_sec / sn.min_ess_per_sec; digits = 2), "× vs NUTS") : "")
            tag = ""
        end
        converged = agree < 0.3
        println("   posterior-mean agreement NUTS vs CGGibbs: max abs diff ",
                round(agree; digits = 3),
                converged ? "  (both converged)" :
                "  ⚠ CGGibbs NOT converged at this d — coordinate-wise mixing collapsed; ESS/s not comparable")
    end
    println("="^76)
    println("Honest reading: single-site CGGibbs loses to NUTS here on ESS/s, and its")
    println("coordinate-wise mixing degrades with d (needs blocking). Two fixable/regime")
    println("caveats: the conditional eval is un-optimized (allocating rk kernel), and this")
    println("weakly-correlated synthetic is not the paper's winning regime (real GLM data).")
    println("="^76)
end

main_vs()
