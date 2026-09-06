# =============================================================================
# Proof of concept: efficient Gibbs sampling on ReactiveKernels.
#
# Model (George & McCulloch 1993 SSVS — the spike-and-slab family from the
# Discourse thread https://discourse.julialang.org/t/.../132816):
#
#   ω        ~ Beta(a0, b0)                      inclusion probability
#   z_j      ~ Bernoulli(ω)                      inclusion indicator, j = 1..p
#   β_j | z_j ~ Normal(0, z_j ? τ²_slab : τ²_spike)
#   σ²       ~ InverseGamma(a_σ, b_σ)
#   y | β,σ² ~ Normal(X β, σ² I)
#
# Every full conditional is conjugate, so this is a pure Gibbs sampler:
#   β  | ·  ~ Normal(mean, cov)   [needs data suff-stats, σ², z]   O(p²/p³)
#   σ² | ·  ~ InvGamma(...)       [needs β, data suff-stats]        O(p²)
#   z  | ·  ~ Bernoulli(p_j)      [needs ω, β — NOT the data]       O(p)
#   ω  | ·  ~ Beta(...)           [needs Σz]                        O(p)
#
# The point of the PoC: the whole model is authored ONCE as a have→want graph.
# Each Gibbs block update just `set!`s its block; the reactive layer recomputes
# ONLY the quantities whose actual dependency set changed (the Markov blanket),
# automatically. The two data-touching O(p²) quantities recompute at most once
# per sweep, and the z / ω updates trigger no data-touching recompute at all —
# with no hand-written caching or dependency bookkeeping.
# =============================================================================
using ReactiveKernels
using LinearAlgebra
using Random
using Statistics: mean, std, var
using LogExpFunctions: logistic, logit
using Distributions
using StableRNGs

# ---- op-execution instrumentation -------------------------------------------
const HITS = Dict{Symbol,Int}()
reset_hits!() = empty!(HITS)
hit!(n) = (HITS[n] = get(HITS, n, 0) + 1)
count!(n, x) = (hit!(n); x)   # tick a counter, return the value

# ---- build the model graph --------------------------------------------------
# Data sufficient statistics + hyperparameters + parameter blocks are SOURCE
# values; every conditional-input quantity is a recipe with an explicit,
# instrumented op. `:E` = data-touching O(p²) op, `:c` = cheap O(p) op.
function build_ssvs_graph()
    g = Graph()
    # data suff-stats (never change)
    XtX = value!(g, :XtX, Matrix{Float64})
    Xty = value!(g, :Xty, Vector{Float64})
    yty = value!(g, :yty, Float64)
    # hyperparameters (never change)
    t2spk = value!(g, :t2spike, Float64)
    t2slb = value!(g, :t2slab, Float64)
    bsig  = value!(g, :b_sig, Float64)
    # parameter blocks (the Gibbs variables)
    beta  = value!(g, :beta, Vector{Float64})
    zind  = value!(g, :z, Vector{Float64})
    omega = value!(g, :omega, Float64)
    sig2  = value!(g, :sigma2, Float64)

    # z conditional inputs: prior precision diag  d_j = 1/(z_j slab/spike var)
    dprec = value!(g, :dprec, Vector{Float64})
    add!(g, (zind, t2spk, t2slb) => dprec,
        (z, vs, vl) -> count!(:dprec, @. 1.0 / ifelse(z > 0.5, vl, vs)); cost = 1.0)

    # β conditional: posterior precision  P = XtX/σ² + diag(d)   (O(p²))
    beta_prec = value!(g, :beta_prec, Matrix{Float64})
    add!(g, (XtX, sig2, dprec) => beta_prec,
        (M, s2, d) -> count!(:beta_prec, Matrix(M ./ s2 + Diagonal(d))); cost = 50.0)
    # β conditional: precision-weighted mean rhs  = Xty/σ²
    beta_rhs = value!(g, :beta_rhs, Vector{Float64})
    add!(g, (Xty, sig2) => beta_rhs, (b, s2) -> count!(:beta_rhs, b ./ s2); cost = 1.0)

    # σ² conditional: residual sum of squares from suff-stats (O(p²), no raw data)
    #   SSR = yᵀy - 2 βᵀXᵀy + βᵀ(XᵀX)β
    ssr = value!(g, :SSR, Float64)
    add!(g, (yty, Xty, XtX, beta) => ssr,
        (yy, Xy, M, b) -> count!(:SSR, yy - 2 * dot(b, Xy) + dot(b, M * b)); cost = 50.0)

    # ω conditional: Σz
    sumz = value!(g, :sumz, Float64)
    add!(g, zind => sumz, z -> count!(:sumz, sum(z)); cost = 1.0)

    # z conditional: per-coefficient inclusion log-odds (needs ω, β — not data)
    zlo = value!(g, :z_logodds, Vector{Float64})
    add!(g, (omega, beta, t2spk, t2slb) => zlo,
        (w, b, vs, vl) -> count!(:z_logodds,
            @. logit(w) - 0.5 * log(vl / vs) - 0.5 * b^2 * (1 / vl - 1 / vs)); cost = 1.0)

    (; g, XtX, Xty, yty, t2spk, t2slb, bsig, beta, zind, omega, sig2,
       dprec, beta_prec, beta_rhs, ssr, sumz, zlo)
end

# ---- the Gibbs sampler, driven by the reactive graph ------------------------
function rk_gibbs(X, y; iters, rng, a0, b0, a_sig, b_sig, t2_spike, t2_slab,
                  measure = false)
    N, p = size(X)
    m = build_ssvs_graph()
    st = ReactiveState(m.g; materialize = (m.dprec, m.beta_prec, m.beta_rhs,
                                           m.ssr, m.sumz, m.zlo))
    # install fixed data suff-stats + hyperparameters + initial blocks
    set!(st, m.XtX, Matrix(X' * X)); set!(st, m.Xty, X' * y); set!(st, m.yty, dot(y, y))
    set!(st, m.t2spk, t2_spike); set!(st, m.t2slb, t2_slab); set!(st, m.bsig, b_sig)
    beta = zeros(p); zz = ones(p); omega = 0.5; sig2 = var(y)
    set!(st, m.beta, beta); set!(st, m.zind, zz); set!(st, m.omega, omega); set!(st, m.sig2, sig2)

    Zt = zeros(iters, p); Bt = zeros(iters, p); St = zeros(iters); Wt = zeros(iters)
    per_sweep = Dict{Symbol,Int}[]

    for it in 1:iters
        measure && reset_hits!()

        # --- β | · : Normal with precision P, mean P⁻¹ rhs ---
        P, rhs = get!(st, (m.beta_prec, m.beta_rhs))
        C = cholesky(Symmetric(P))
        beta = (C \ rhs) .+ (C.U \ randn(rng, p))
        set!(st, m.beta, beta)

        # --- σ² | · : InverseGamma(a_σ + N/2, b_σ + SSR/2) ---
        SSR = get!(st, m.ssr)
        sig2 = rand(rng, InverseGamma(a_sig + N / 2, b_sig + SSR / 2))
        set!(st, m.sig2, sig2)

        # --- z | · : independent Bernoulli(logistic(log-odds_j)) ---
        lo = get!(st, m.zlo)
        @inbounds for j in 1:p
            zz[j] = rand(rng) < logistic(lo[j]) ? 1.0 : 0.0
        end
        set!(st, m.zind, copy(zz))

        # --- ω | · : Beta(a0 + Σz, b0 + p − Σz) ---
        s = get!(st, m.sumz)
        omega = rand(rng, Beta(a0 + s, b0 + p - s))
        set!(st, m.omega, omega)

        Bt[it, :] .= beta; Zt[it, :] .= zz; St[it] = sig2; Wt[it] = omega
        measure && push!(per_sweep, copy(HITS))
    end
    (; Bt, Zt, St, Wt, per_sweep)
end

# ---- independent reference sampler (plain arrays, identical math + RNG) ------
# Proves the reactive layer changes only WHAT is recomputed, never the numbers.
function ref_gibbs(X, y; iters, rng, a0, b0, a_sig, b_sig, t2_spike, t2_slab)
    N, p = size(X)
    XtX = Matrix(X' * X); Xty = X' * y; yty = dot(y, y)
    beta = zeros(p); zz = ones(p); omega = 0.5; sig2 = var(y)
    Zt = zeros(iters, p); Bt = zeros(iters, p); St = zeros(iters); Wt = zeros(iters)
    for it in 1:iters
        d = @. 1.0 / ifelse(zz > 0.5, t2_slab, t2_spike)
        P = XtX ./ sig2 + Diagonal(d); rhs = Xty ./ sig2
        C = cholesky(Symmetric(P))
        beta = (C \ rhs) .+ (C.U \ randn(rng, p))
        SSR = yty - 2 * dot(beta, Xty) + dot(beta, XtX * beta)
        sig2 = rand(rng, InverseGamma(a_sig + N / 2, b_sig + SSR / 2))
        lo = @. logit(omega) - 0.5 * log(t2_slab / t2_spike) -
                0.5 * beta^2 * (1 / t2_slab - 1 / t2_spike)
        @inbounds for j in 1:p
            zz[j] = rand(rng) < logistic(lo[j]) ? 1.0 : 0.0
        end
        s = sum(zz)
        omega = rand(rng, Beta(a0 + s, b0 + p - s))
        Bt[it, :] .= beta; Zt[it, :] .= zz; St[it] = sig2; Wt[it] = omega
    end
    (; Bt, Zt, St, Wt)
end

# ---- measured provenance-blind baseline -------------------------------------
# A generic PPL that does not know the conditional structure re-scores the whole
# model at every block update. We measure that by fetching all conditional-input
# quantities at each of the 4 blocks against a cache-free state.
function measure_naive_sweep(X, y; a0, b0, a_sig, b_sig, t2_spike, t2_slab)
    m = build_ssvs_graph()
    st = ReactiveState(m.g)                    # materialize = () → nothing cached
    set!(st, m.XtX, Matrix(X' * X)); set!(st, m.Xty, X' * y); set!(st, m.yty, dot(y, y))
    set!(st, m.t2spk, t2_spike); set!(st, m.t2slb, t2_slab); set!(st, m.bsig, b_sig)
    p = size(X, 2)
    set!(st, m.beta, zeros(p)); set!(st, m.zind, ones(p))
    set!(st, m.omega, 0.5); set!(st, m.sig2, var(y))
    allq = (m.dprec, m.beta_prec, m.beta_rhs, m.ssr, m.sumz, m.zlo)
    reset_hits!()
    for _block in 1:4
        get!(st, allq)                         # re-score every conditional input
    end
    copy(HITS)
end

# ---- synthetic data with a known sparse truth -------------------------------
function make_data(rng; N = 200, p = 30, k = 5, snr = 4.0)
    X = randn(rng, N, p)
    for j in 1:p; X[:, j] .-= mean(X[:, j]); X[:, j] ./= std(X[:, j]); end
    active = sort!(randperm(rng, p)[1:k])
    btrue = zeros(p); btrue[active] .= snr .* sign.(randn(rng, k)) .* (1 .+ rand(rng, k))
    y = X * btrue .+ randn(rng, N)
    (; X, y, btrue, active)
end

function main()
    hyper = (a0 = 1.0, b0 = 1.0, a_sig = 2.0, b_sig = 1.0,
             t2_spike = 1e-3, t2_slab = 10.0)
    data = make_data(StableRNG(1); N = 200, p = 30, k = 5, snr = 4.0)
    iters = 4000; warm = 1000

    # correctness + identity: same seed for both samplers
    rk  = rk_gibbs(data.X, data.y; iters, rng = StableRNG(42), hyper...)
    ref = ref_gibbs(data.X, data.y; iters, rng = StableRNG(42), hyper...)

    println("="^74)
    println("CORRECTNESS — recover the sparse truth (p=", size(data.X, 2),
            ", true active = ", data.active, ")")
    pip = vec(mean(rk.Zt[warm+1:end, :]; dims = 1))     # posterior inclusion prob
    bhat = vec(mean(rk.Bt[warm+1:end, :]; dims = 1))
    sel = findall(>=(0.5), pip)
    println("  selected (PIP≥0.5) : ", sel)
    println("  true active        : ", data.active)
    println("  exact match        : ", sel == data.active)
    println("  PIP on true active : ", round.(pip[data.active]; digits = 3))
    off = setdiff(1:size(data.X, 2), data.active)
    println("  max PIP off-active : ", round(maximum(pip[off]); digits = 3))
    println("  β̂ vs β on active   :")
    for a in data.active
        println("    j=", lpad(a, 2), "  true=", rpad(round(data.btrue[a]; digits = 2), 6),
                "  post_mean=", round(bhat[a]; digits = 3))
    end

    println("="^74)
    println("IDENTITY — reactive graph vs hand-written reference (same RNG)")
    println("  β chains identical  : ", rk.Bt == ref.Bt)
    println("  z chains identical  : ", rk.Zt == ref.Zt)
    println("  σ² chains identical : ", rk.St == ref.St)
    println("  ω chains identical  : ", rk.Wt == ref.Wt)

    println("="^74)
    println("EFFICIENCY — op executions per Gibbs sweep (measured)")
    meas = rk_gibbs(data.X, data.y; iters = 20, rng = StableRNG(7), hyper..., measure = true)
    # steady-state sweep (skip the first, which has a cold cache)
    steady = meas.per_sweep[end]
    expensive = (:beta_prec, :SSR)
    cheap = (:dprec, :beta_rhs, :sumz, :z_logodds)
    println("  data-touching O(p²) ops this sweep:")
    for k in expensive
        println("    ", rpad(k, 12), " executed ", get(steady, k, 0), "×")
    end
    println("  cheap O(p) ops this sweep:")
    for k in cheap
        println("    ", rpad(k, 12), " executed ", get(steady, k, 0), "×")
    end
    rk_exp = sum(get(steady, k, 0) for k in expensive)
    naive = measure_naive_sweep(data.X, data.y; hyper...)
    naive_exp = sum(get(naive, k, 0) for k in expensive)
    println("  → RK have→want slicing     : ", rk_exp, " expensive ops/sweep",
            "  (each O(p²) quantity recomputed once, only when its inputs changed)")
    println("  → provenance-blind re-score: ", naive_exp, " expensive ops/sweep",
            "  (measured: whole model re-scored at each of 4 blocks)")
    println("  → the z & ω blocks recompute only cheap O(p) quantities; the planner's",
            " Markov blanket automatically excludes the data.")
    println("="^74)
end

main()
