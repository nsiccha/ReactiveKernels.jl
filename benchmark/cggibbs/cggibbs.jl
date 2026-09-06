# =============================================================================
# CGGibbs on ReactiveKernels — Bayesian logistic regression (arXiv 2410.03630).
#
# The paper's "compute graph Gibbs" makes a coordinate Gibbs sweep O(nd) instead
# of O(nd²) by CACHING the linear predictor η = Xθ and updating it rank-1 when one
# θ_j changes. Here:
#   • the per-coordinate CONDITIONAL log-density is authored directly as an rk
#     kernel (prepared once, evaluated by the within-coordinate slice sampler);
#   • the η cache is maintained by the sampler (rank-1 update) — the CGGibbs trick.
#
# We also test the honest question: does rk's PURE reactive layer (set!/get!) give
# the O(d) caching automatically? No — it recomputes at RECIPE granularity, so a
# dense η recipe is O(nd) per coordinate (= naive). The rank-1 cache is authored
# (here explicitly; the stateful mutate!/touch! layer is the declarative route).
# =============================================================================
using ReactiveKernels
using LinearAlgebra
using Random
using Statistics: mean, std
using LogExpFunctions: log1pexp
using StableRNGs

# ---- the coordinate-conditional log-density, authored as an rk kernel ---------
# want = logpost(θ_j) given the cached "minus-j" predictor rmj = η − x_j·θ_j.
#   logpost = −θ_j²/(2 s0²) + Σ_i [ y_i·(rmj_i + x_ij θ_j) − softplus(rmj_i + x_ij θ_j) ]
function build_coord_conditional()
    g = Graph()
    tj   = value!(g, :theta_j, Float64)
    rmj  = value!(g, :rmj, Vector{Float64})   # cached minus-j linear predictor
    xj   = value!(g, :xj, Vector{Float64})
    y    = value!(g, :y, Vector{Float64})
    s0sq = value!(g, :s0sq, Float64)
    lin  = value!(g, :lin, Vector{Float64})
    ll   = value!(g, :loglik, Float64)
    lp   = value!(g, :logpost, Float64)
    add!(g, (rmj, xj, tj) => lin, (r, x, t) -> r .+ x .* t)
    add!(g, (y, lin) => ll, (yy, l) -> sum(yy .* l .- log1pexp.(l)))
    add!(g, (tj, s0sq, ll) => lp, (t, s2, L) -> -t^2 / (2 * s2) + L)
    kern = prepare(g; have = (tj, rmj, xj, y, s0sq), want = lp)
    (t, rmjv, xjv, yv, s2) -> kern(t, rmjv, xjv, yv, s2)
end

# ---- Neal (2003) 1-D slice sampler (stepping-out + shrinkage) ------------------
function slice_sample(rng, glog, x0; w = 1.0, m = 32)
    logu = glog(x0) + log(rand(rng))
    L = x0 - w * rand(rng); R = L + w
    j = floor(Int, m * rand(rng)); k = (m - 1) - j
    while j > 0 && logu < glog(L); L -= w; j -= 1; end
    while k > 0 && logu < glog(R); R += w; k -= 1; end
    while true
        x1 = L + rand(rng) * (R - L)
        logu < glog(x1) && return x1
        x1 < x0 ? (L = x1) : (R = x1)
    end
end

# ---- CGGibbs sweep: cached η, rank-1 update (O(nd) per sweep) ------------------
function cggibbs_sweep!(rng, theta, eta, X, y, s0sq, cond)
    d = length(theta)
    @inbounds for jj in 1:d
        xj = view(X, :, jj)
        rmj = eta .- xj .* theta[jj]                    # O(n)
        tj = slice_sample(rng, t -> cond(t, rmj, xj, y, s0sq), theta[jj])
        eta .= rmj .+ xj .* tj                          # O(n) rank-1 cache update
        theta[jj] = tj
    end
end

# ---- naive sweep: recompute η = Xθ each coordinate (O(nd²) per sweep) ----------
function naive_sweep!(rng, theta, X, y, s0sq, cond)
    d = length(theta)
    @inbounds for jj in 1:d
        eta = X * theta                                 # O(nd) — the naive cost
        xj = view(X, :, jj)
        rmj = eta .- xj .* theta[jj]
        theta[jj] = slice_sample(rng, t -> cond(t, rmj, xj, y, s0sq), theta[jj])
    end
end

# ---- synthetic logistic data --------------------------------------------------
function make_logistic(rng; n, d, k, snr = 3.0)
    X = randn(rng, n, d)
    for jj in 1:d; X[:, jj] .-= mean(X[:, jj]); X[:, jj] ./= std(X[:, jj]); end
    active = sort!(randperm(rng, d)[1:k])
    btrue = zeros(d); btrue[active] .= snr .* sign.(randn(rng, k))
    p = 1 ./ (1 .+ exp.(-(X * btrue)))
    y = Float64.(rand(rng, n) .< p)
    (; X, y, btrue, active)
end

function run_chain(sweep!, data, s0sq; iters, rng)
    n, d = size(data.X)
    theta = zeros(d); eta = zeros(n)
    T = zeros(iters, d)
    for it in 1:iters
        if sweep! === cggibbs_sweep!
            sweep!(rng, theta, eta, data.X, data.y, s0sq, cond)
        else
            sweep!(rng, theta, data.X, data.y, s0sq, cond)
        end
        T[it, :] .= theta
    end
    T
end

const cond = build_coord_conditional()

function main()
    s0sq = 100.0            # prior sd 10
    # well-identified case (finite MLE) so posterior means track the truth
    data = make_logistic(StableRNG(1); n = 2000, d = 12, k = 4, snr = 1.2)
    iters, warm = 4000, 1000

    # identity: cached vs naive must produce bit-identical chains (same RNG)
    Tc = run_chain(cggibbs_sweep!, data, s0sq; iters, rng = StableRNG(9))
    Tn = run_chain(naive_sweep!,  data, s0sq; iters, rng = StableRNG(9))

    println("="^70)
    println("CORRECTNESS — logistic regression (n=", size(data.X, 1), ", d=",
            size(data.X, 2), ", active=", data.active, ")")
    bhat = vec(mean(Tc[warm+1:end, :]; dims = 1))
    println("  β̂ vs β on active:")
    for a in data.active
        println("    j=", lpad(a, 2), "  true=", rpad(round(data.btrue[a]; digits=2), 6),
                "  post_mean=", round(bhat[a]; digits = 3))
    end
    off = setdiff(1:size(data.X, 2), data.active)
    println("  max |β̂| off-active: ", round(maximum(abs, bhat[off]); digits = 3))
    println("="^70)
    println("IDENTITY — cached (CGGibbs) vs naive recompute, same RNG")
    println("  chains identical: ", Tc == Tn, "  (caching changes only cost, not the numbers)")

    # timing vs d: CGGibbs O(nd) sweep should pull away from naive O(nd²)
    println("="^70)
    println("SCALING — median ms / sweep (n=400), CGGibbs vs naive")
    println("   d     cggibbs      naive     ratio")
    for d in (16, 64, 256, 1024, 2048)
        dt = make_logistic(StableRNG(2); n = 400, d = d, k = min(5, d), snr = 3.0)
        n = 400
        # warm compile
        th = zeros(d); et = zeros(n)
        cggibbs_sweep!(StableRNG(3), th, et, dt.X, dt.y, s0sq, cond)
        thn = zeros(d); naive_sweep!(StableRNG(3), thn, dt.X, dt.y, s0sq, cond)
        reps = 7
        tc = minimum(begin
            th = zeros(d); et = zeros(n); r = StableRNG(3)
            @elapsed cggibbs_sweep!(r, th, et, dt.X, dt.y, s0sq, cond)
        end for _ in 1:reps)
        tn = minimum(begin
            thn = zeros(d); r = StableRNG(3)
            @elapsed naive_sweep!(r, thn, dt.X, dt.y, s0sq, cond)
        end for _ in 1:reps)
        println("  ", lpad(d, 4), "  ", lpad(round(tc*1e3; digits=3), 9), "  ",
                lpad(round(tn*1e3; digits=3), 9), "  ", lpad(round(tn/tc; digits=2), 6), "×")
    end
    println("="^70)
end

main()
