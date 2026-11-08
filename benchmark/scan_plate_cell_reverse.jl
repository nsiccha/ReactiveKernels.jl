# Native Enzyme Reverse of an RK `scan` inside a plate cell, against the same
# recurrence written as a hand loop, for three recurrences:
#
# - ODL's two-state washout step (`washout_path` is ODLPlaus
#   `src/joint_washout.jl` `odl_joint_washout_path`, with its deposit weights
#   from `src/washout.jl`), with the controls that localize the cost: a hand
#   loop in RK's layout (NamedTuple carry, NTuple trajectory), an RK plate cell
#   that calls the hand loop, and a top-level RK scan over one long sequence;
# - an ARMA(1,1) error recursion (scalar NamedTuple carry, as in docs/src/scan.md);
# - a 3-state HMM forward filter (vector carry).
#
#   julia --project=<env with ReactiveKernels and Enzyme> scan_plate_cell_reverse.jl
#
# Arms are timed in shuffled order every round; ratios are paired per round
# (median [IQR]). Gradients agree with the hand loop's to 1e-8. The Enzyme-only
# shapes behind the costs are in repro_enzyme_nested_loop_reverse.jl.
#
# Recorded on gordito, 2026-11-08, RK 0.1.119, one thread, one CPU, 30 rounds,
# at host load 25-33 on 32 CPUs, as each arm's median ratio to the hand loop
# (washout: G=38 T=120 / G=113 T=120 / G=38 T=1320; ARMA, HMM: G=38 T=120 /
# T=1320; top-level scan: T=4560 / T=50160):
#
#                         Julia 1.10.12      Julia 1.10.12        Julia 1.12.7
#                         dogfood c9ab4c9    dogfood 5e2cd2bb     dogfood 5e2cd2bb
#                         libEnzyme 42a2ee9a libEnzyme 576e674a   libEnzyme 576e674a
#   washout scan cell     1.72 / 1.76 / 1.75 1.08 / 1.11 / 1.16   0.98 / 1.00 / 1.10
#   washout, cell calling
#   the hand loop         1.03 / 1.03 / 1.02 1.05 / 1.05 / 1.06   1.02 / 1.04 / 1.03
#   ARMA(1,1) scan cell   1.10 / 1.08        0.89 / 0.94          0.78 / 0.88
#   HMM scan cell         1.30 / 1.32        0.93 / 0.98          0.96 / 0.99
#   top-level scan        1.07 / 1.03        1.10 / 1.09          1.00 / 1.02
#
# Runs on RK 0.1.118 earlier the same day, at load 22-70, gave the same
# picture. Most of the washout cell's remainder is Enzyme's cost for its active
# per-path weights computed through a branch (Enzyme snag
# enzyme-reverse-i-6fb5e335); the ARMA and HMM cells' outer values reach the
# inner loop without one.
using Enzyme, Random, Printf
using ReactiveKernels
const DI = ReactiveKernels.DifferentiationInterface

# ---------------- the washout step and its hand loop ----------------
function dmean(lambda::Real)
    if lambda < 0.1
        return 0.5 - lambda * (1.0/6 - lambda * (1.0/24 - lambda * (1.0/120 - lambda *
            (1.0/720 - lambda * (1.0/5040 - lambda * (1.0/40320 - lambda *
            (1.0/362880 - lambda/3628800)))))))
    else
        return (1.0 + expm1(-lambda) / lambda) / lambda
    end
end
dend(lambda::Real) = lambda < 0.1 ? 1.0 - lambda * dmean(lambda) : -expm1(-lambda) / lambda

function washout_path(rain::AbstractVector, recovery::Real, scavenging::Real, decay::Real,
        initial_air::Real, initial_ground::Real)
    T = promote_type(eltype(rain),typeof(recovery),typeof(scavenging),typeof(decay),
        typeof(initial_air),typeof(initial_ground))
    out = Matrix{T}(undef,length(rain),3)
    air, ground = initial_air, initial_ground
    recover = -expm1(-recovery)
    remaining = exp(-decay)
    end_weight = dend(decay)
    mean_weight = dmean(decay)
    for t in eachindex(rain)
        air = air + (1 - air) * recover
        deposit = air * (-expm1(-scavenging * rain[t]))
        air = air - deposit
        out[t,1] = end_weight * ground + mean_weight * deposit
        ground = remaining * ground + end_weight * deposit
        out[t,2] = air
        out[t,3] = ground
    end
    return out
end

# The model: one plate cell per path (gather + recursion), then linear readers.
function objective(rain, idx, rec, scav, dec, a0, g0, w)
    s = 0.0
    for g in eachindex(idx)
        p = washout_path(rain[idx[g]], rec[g], scav[g], dec[g], a0[g], g0[g])
        wg = w[g]
        @inbounds for j in eachindex(p)
            s += wg[j] * p[j]
        end
    end
    s
end

function setup(G, T; seed = 1)
    rng = Xoshiro(seed)
    nrain = G * T
    rain = rand(rng, nrain) .* 2.0
    idx = [((g-1)*T+1):(g*T) |> collect for g in 1:G]
    rec = 0.1 .+ 0.2 .* rand(rng, G)
    scav = 0.05 .+ 0.1 .* rand(rng, G)
    dec = 0.01 .+ 0.3 .* rand(rng, G)    # both branches of dmean/dend
    a0 = 0.2 .+ 0.6 .* rand(rng, G)
    g0 = rand(rng, G)
    w = [randn(rng, T, 3) for _ in 1:G]
    (; rain, idx, rec, scav, dec, a0, g0, w)
end

# --- RK: the washout recurrence as a scan in a plate cell --------------------
@kernel rk_cell(rain, idx, rec, scav, dec, a0, g0, w) = begin
    cell = plate(eachindex(idx)) do g
        r = rain[idx[g]]
        recover = -expm1(-rec[g])
        remaining = exp(-dec[g])
        ew = dend(dec[g])
        mw = dmean(dec[g])
        s = scav[g]
        wg = w[g]
        traj = scan(r; init = (; air = a0[g], ground = g0[g])) do c, x
            air1 = c.air + (1 - c.air) * recover
            deposit = air1 * (-expm1(-s * x))
            air2 = air1 - deposit
            o1 = ew * c.ground + mw * deposit
            ground2 = remaining * c.ground + ew * deposit
            ((; air = air2, ground = ground2), (o1, air2, ground2))
        end
        sum(wg[t, 1] * traj[t][1] + wg[t, 2] * traj[t][2] + wg[t, 3] * traj[t][3]
            for t in eachindex(traj); init = 0.0)
    end
    total::Float64 = sum(cell)
    return total
end

# --- RK: plate cell calling the opaque hand loop (plate machinery only) -----
@kernel rk_opaque_cell(rain, idx, rec, scav, dec, a0, g0, w) = begin
    cell = plate(eachindex(idx)) do g
        p = washout_path(rain[idx[g]], rec[g], scav[g], dec[g], a0[g], g0[g])
        wg = w[g]
        sum(wg[j] * p[j] for j in eachindex(p); init = 0.0)
    end
    total::Float64 = sum(cell)
    return total
end

# --- RK: one top-level scan over one long sequence (no plate) ----------------
@kernel rk_top(rain, rec, scav, dec, a0, g0, w) = begin
    recover = -expm1(-rec)
    remaining = exp(-dec)
    ew = dend(dec)
    mw = dmean(dec)
    traj = scan(rain; init = (; air = a0, ground = g0)) do c, x
        air1 = c.air + (1 - c.air) * recover
        deposit = air1 * (-expm1(-scav * x))
        air2 = air1 - deposit
        o1 = ew * c.ground + mw * deposit
        ground2 = remaining * c.ground + ew * deposit
        ((; air = air2, ground = ground2), (o1, air2, ground2))
    end
    total::Float64 = sum(w[t, 1] * traj[t][1] + w[t, 2] * traj[t][2] + w[t, 3] * traj[t][3]
        for t in eachindex(traj); init = 0.0)
    return total
end

# --- hand loop with RK's layout: NamedTuple carry, Vector{NTuple{3}} trajectory
function mimic_path(r, recover, remaining, ew, mw, s, a0, g0)
    traj = Vector{NTuple{3,typeof(recover * a0)}}(undef, length(r))
    c = (; air = a0, ground = g0)
    for t in eachindex(r)
        x = r[t]
        air1 = c.air + (1 - c.air) * recover
        deposit = air1 * (-expm1(-s * x))
        air2 = air1 - deposit
        o1 = ew * c.ground + mw * deposit
        ground2 = remaining * c.ground + ew * deposit
        c = (; air = air2, ground = ground2)
        traj[t] = (o1, air2, ground2)
    end
    traj
end
function mimic(rain, idx, rec, scav, dec, a0, g0, w)
    s = 0.0
    for g in eachindex(idx)
        traj = mimic_path(rain[idx[g]], -expm1(-rec[g]), exp(-dec[g]), dend(dec[g]),
            dmean(dec[g]), scav[g], a0[g], g0[g])
        wg = w[g]
        acc = 0.0
        for t in eachindex(traj)
            acc += wg[t, 1] * traj[t][1] + wg[t, 2] * traj[t][2] + wg[t, 3] * traj[t][3]
        end
        s += acc
    end
    s
end
function top_hand(rain, rec, scav, dec, a0, g0, w)
    traj = mimic_path(rain, -expm1(-rec), exp(-dec), dend(dec), dmean(dec), scav, a0, g0)
    acc = 0.0
    for t in eachindex(traj)
        acc += w[t, 1] * traj[t][1] + w[t, 2] * traj[t][2] + w[t, 3] * traj[t][3]
    end
    acc
end

function enzyme_grad(f, x)
    d = map(zero, (x.rain, x.rec, x.scav, x.dec, x.a0, x.g0))
    () -> begin
        foreach(v -> fill!(v, 0.0), d)
        Enzyme.autodiff(Reverse, f, Active, Duplicated(x.rain, d[1]), Const(x.idx),
            Duplicated(x.rec, d[2]), Duplicated(x.scav, d[3]), Duplicated(x.dec, d[4]),
            Duplicated(x.a0, d[5]), Duplicated(x.g0, d[6]), Const(x.w))
        d
    end
end

# ---------------- ARMA(1,1) ----------------
@kernel rk_arma(y, idx, mu, phi, theta) = begin
    cell = plate(eachindex(idx)) do g
        m = mu[g]; f = phi[g]; h = theta[g]
        errors = scan(y[idx[g]]; init = (; y_prev = m, err_prev = 0.0)) do c, yt
            nu = m + f * c.y_prev + h * c.err_prev
            e = yt - nu
            ((; y_prev = yt, err_prev = e), e)
        end
        sum(-0.5 * e * e for e in errors; init = 0.0)
    end
    total::Float64 = sum(cell)
    return total
end
for (name, inl) in ((:arma_path_noinl, false), (:arma_path_inl, true))
    f = :(function $name(y, m, f, h)
        out = similar(y, typeof(m * f * h))
        yp, ep = m, zero(m * f * h)
        for t in eachindex(y)
            e = y[t] - (m + f * yp + h * ep)
            @inbounds out[t] = e
            yp, ep = y[t], e
        end
        out
    end)
    @eval $(inl ? :(@inline $f) : :(@noinline $f))
end
for path in (:arma_path_noinl, :arma_path_inl)
    @eval function $(Symbol(:arma_obj_, path))(y, idx, mu, phi, theta)
        s = 0.0
        for g in eachindex(idx)
            errs = $path(y[idx[g]], mu[g], phi[g], theta[g])
            acc = 0.0
            for e in errs
                acc += -0.5 * e * e
            end
            s += acc
        end
        s
    end
end

# ---------------- 3-state HMM forward filter ----------------
@inline emit(yt, m) = exp(-0.5 * (yt - m)^2)
@kernel rk_hmm(y, idx, stay, mus) = begin
    cell = plate(eachindex(idx)) do g
        s1 = stay[1]; s2 = stay[2]; s3 = stay[3]
        m1 = mus[1]; m2 = mus[2]; m3 = mus[3]
        lc = scan(y[idx[g]]; init = (; a1 = 1 / 3, a2 = 1 / 3, a3 = 1 / 3)) do a, yt
            o1 = (1 - s1) / 2; o2 = (1 - s2) / 2; o3 = (1 - s3) / 2
            b1 = (a.a1 * s1 + a.a2 * o2 + a.a3 * o3) * emit(yt, m1)
            b2 = (a.a1 * o1 + a.a2 * s2 + a.a3 * o3) * emit(yt, m2)
            b3 = (a.a1 * o1 + a.a2 * o2 + a.a3 * s3) * emit(yt, m3)
            c = b1 + b2 + b3
            ((; a1 = b1 / c, a2 = b2 / c, a3 = b3 / c), log(c))
        end
        sum(lc; init = 0.0)
    end
    total::Float64 = sum(cell)
    return total
end
for (name, inl) in ((:hmm_path_noinl, false), (:hmm_path_inl, true))
    f = :(function $name(y, s1, s2, s3, m1, m2, m3)
        out = similar(y, typeof(s1 * m1))
        a1 = a2 = a3 = 1 / 3
        o1 = (1 - s1) / 2; o2 = (1 - s2) / 2; o3 = (1 - s3) / 2
        for t in eachindex(y)
            yt = y[t]
            b1 = (a1 * s1 + a2 * o2 + a3 * o3) * emit(yt, m1)
            b2 = (a1 * o1 + a2 * s2 + a3 * o3) * emit(yt, m2)
            b3 = (a1 * o1 + a2 * o2 + a3 * s3) * emit(yt, m3)
            c = b1 + b2 + b3
            a1, a2, a3 = b1 / c, b2 / c, b3 / c
            @inbounds out[t] = log(c)
        end
        out
    end)
    @eval $(inl ? :(@inline $f) : :(@noinline $f))
end
for path in (:hmm_path_noinl, :hmm_path_inl)
    @eval function $(Symbol(:hmm_obj_, path))(y, idx, stay, mus)
        s = 0.0
        for g in eachindex(idx)
            lc = $path(y[idx[g]], stay[1], stay[2], stay[3], mus[1], mus[2], mus[3])
            acc = 0.0
            for v in lc
                acc += v
            end
            s += acc
        end
        s
    end
end

const ROUNDS = parse(Int, get(ENV, "ROUNDS", "15"))
function interleaved(arms, n; rounds = ROUNDS)
    for (_, f) in arms; f(); f(); end
    samples = fill(Inf, rounds, length(arms))
    rng = Random.Xoshiro(7)
    for r in 1:rounds, i in Random.shuffle(rng, collect(eachindex(arms)))
        f = arms[i][2]
        GC.gc(false)
        t = @elapsed for _ in 1:n; f(); end
        samples[r, i] = t / n
    end
    samples
end
quart(v) = (s = sort(v); q(p) = s[clamp(round(Int, p * (length(s) - 1)) + 1, 1, length(s))]; (q(0.25), q(0.5), q(0.75)))
function report(arms, samples, allocs)
    for (i, (label, _)) in enumerate(arms)
        r = samples[:, i] ./ samples[:, 1]
        lo, med, hi = quart(r)
        @printf("  %-42s grad min %8.1f us med %8.1f us  %8.1f KB  ratio-to-hand med %.2f [%.2f, %.2f] min-ratio %.2f\n",
            label, minimum(samples[:, i]) * 1e6, quart(samples[:, i])[2] * 1e6, allocs[i] / 1024,
            med, lo, hi, minimum(samples[:, i]) / minimum(samples[:, 1]))
    end
end
function check_and_time(title, arms, n)
    gs = [f() for (_, f) in arms]
    ref = gs[1]
    for (i, gr) in enumerate(gs)
        err = maximum(maximum(abs.(a .- b)) for (a, b) in zip(gr, ref))
        err < 1e-8 || error("$(arms[i][1]) gradient differs by $err")
    end
    samples = interleaved(arms, n)
    println("\n== ", title, " ($ROUNDS shuffled rounds, $n calls each; ratios paired per round, median [IQR])")
    report(arms, samples, [(@allocated f()) for (_, f) in arms])
    println("  samples_us = ", round.(samples .* 1e6; digits = 1))
end

println("julia ", VERSION, "  Enzyme ", pkgversion(Enzyme), "  libEnzyme ",
    basename(dirname(dirname(Enzyme.Compiler.Enzyme_jll.libEnzyme_path))),
    "  RK ", pkgversion(ReactiveKernels), "  threads ", Threads.nthreads())
backend = DI.AutoEnzyme(; mode = Enzyme.Reverse)
active8 = (:rain, :rec, :scav, :dec, :a0, :g0)
kc = prepare(rk_cell; want = :total)
ko = prepare(rk_opaque_cell; want = :total)
kt = prepare(rk_top; want = :total)

for (G, T) in ((38, 120), (113, 120), (38, 1320))
    x = setup(G, T)
    args = (x.rain, x.idx, x.rec, x.scav, x.dec, x.a0, x.g0, x.w)
    ref = objective(args...)
    @assert kc(args...) ≈ ref && ko(args...) ≈ ref && mimic(args...) ≈ ref
    adc = prepare_ad(kc, backend, args...; active = active8)
    ado = prepare_ad(ko, backend, args...; active = active8)
    arms = [
        "hand loop (Matrix out)" => enzyme_grad(objective, x),
        "hand, RK layout (NT carry, NTuple traj)" => enzyme_grad(mimic, x),
        "RK plate + opaque hand loop" => () -> ad_value_and_gradient(ado, args...),
        "RK plate + RK scan" => () -> ad_value_and_gradient(adc, args...),
    ]
    grads = [a[2]() for a in arms]
    g0ref = collect(grads[1])
    for (i, gr) in enumerate(grads)
        gi = i <= 2 ? collect(gr) : collect(gr[2])
        err = maximum(maximum(abs.(a .- b)) for (a, b) in zip(gi, g0ref))
        err < 1e-8 || error("arm $(arms[i][1]) gradient differs by $err")
    end
    n = max(1, round(Int, 1e5 / (G * T)))
    samples = interleaved(arms, n)
    allocs = [(@allocated a[2]()) for a in arms]
    println("\n== G=$G T=$T steps=$(G*T) ($ROUNDS shuffled rounds, $n calls each; ratios paired per round, median [IQR])")
    report(arms, samples, allocs)
    println("  samples_us = ", round.(samples .* 1e6; digits = 1))
end

# Top-level single long scan (no plate): one path of 4560 and 50160 steps.
for T in (4560, 50160)
    x0 = setup(1, T)
    xs = (; rain = x0.rain, rec = x0.rec[1], scav = x0.scav[1], dec = x0.dec[1],
        a0 = x0.a0[1], g0 = x0.g0[1], w = x0.w[1])
    args = (xs.rain, xs.rec, xs.scav, xs.dec, xs.a0, xs.g0, xs.w)
    @assert kt(args...) ≈ top_hand(args...)
    adt = prepare_ad(kt, backend, args...; active = active8)
    dr = zero(xs.rain)
    hand = () -> begin
        fill!(dr, 0.0)
        Enzyme.autodiff(Reverse, top_hand, Active, Duplicated(xs.rain, dr), Active(xs.rec),
            Active(xs.scav), Active(xs.dec), Active(xs.a0), Active(xs.g0), Const(xs.w))
    end
    rk = () -> ad_value_and_gradient(adt, args...)
    hr = hand(); _, rg = rk()
    @assert maximum(abs.(rg[1] .- dr)) < 1e-8
    @assert all(abs(a - b) < 1e-8 for (a, b) in zip(rg[2:end], hr[1][2:6]))
    n = max(1, round(Int, 1e5 / T))
    arms = ["hand top-level loop" => hand, "RK top-level scan" => rk]
    samples = interleaved(arms, n)
    println("\n== top-level single scan T=$T ($ROUNDS shuffled rounds, $n calls each)")
    report(arms, samples, [(@allocated hand()), (@allocated rk())])
end

ka = prepare(rk_arma; want = :total)
kh = prepare(rk_hmm; want = :total)

for (G, T) in ((38, 120), (38, 1320))
    rng = Xoshiro(3)
    y = randn(rng, G * T); idx = [collect(((g - 1) * T + 1):(g * T)) for g in 1:G]
    n = max(1, round(Int, 1e5 / (G * T)))

    mu = 0.1 .* randn(rng, G); phi = 0.5 .* rand(rng, G); theta = 0.3 .* rand(rng, G)
    aargs = (y, idx, mu, phi, theta)
    @assert ka(aargs...) ≈ arma_obj_arma_path_noinl(aargs...) ≈ arma_obj_arma_path_inl(aargs...)
    ada = prepare_ad(ka, backend, aargs...; active = (:mu, :phi, :theta))
    arma_hand(f) = let d = map(zero, (mu, phi, theta))
        () -> (foreach(v -> fill!(v, 0.0), d);
            Enzyme.autodiff(Reverse, f, Active, Const(y), Const(idx), Duplicated(mu, d[1]),
                Duplicated(phi, d[2]), Duplicated(theta, d[3])); d)
    end
    check_and_time("ARMA(1,1) G=$G T=$T", Pair{String,Any}[
        "hand loop, path @noinline" => arma_hand(arma_obj_arma_path_noinl),
        "hand loop, path @inline" => arma_hand(arma_obj_arma_path_inl),
        "RK plate + RK scan" => () -> ad_value_and_gradient(ada, aargs...)[2]], n)

    stay = [0.9, 0.8, 0.85]; mus = [-1.0, 0.0, 1.5]
    hargs = (y, idx, stay, mus)
    @assert kh(hargs...) ≈ hmm_obj_hmm_path_noinl(hargs...) ≈ hmm_obj_hmm_path_inl(hargs...)
    adh = prepare_ad(kh, backend, hargs...; active = (:stay, :mus))
    hmm_hand(f) = let d = map(zero, (stay, mus))
        () -> (foreach(v -> fill!(v, 0.0), d);
            Enzyme.autodiff(Reverse, f, Active, Const(y), Const(idx), Duplicated(stay, d[1]),
                Duplicated(mus, d[2])); d)
    end
    check_and_time("HMM K=3 forward filter G=$G T=$T", Pair{String,Any}[
        "hand loop, path @noinline" => hmm_hand(hmm_obj_hmm_path_noinl),
        "hand loop, path @inline" => hmm_hand(hmm_obj_hmm_path_inl),
        "RK plate + RK scan" => () -> ad_value_and_gradient(adh, hargs...)[2]], n)
end
