# Standalone CPU reproducer: only Enzyme is required, no ReactiveKernels code.
#
#   julia --project=<env with Enzyme> repro_enzyme_nested_loop_reverse.jl
#
# Native Enzyme Reverse of an outer loop over paths that gathers each path's
# inputs, runs a recurrence over them into a fresh vector in an inner loop and
# reads that vector back, compared with the same inner loop in a `@noinline`
# callee. Same arithmetic, same values, same gradients (checked to 1e-10).
# ReactiveKernels' native plate lowering inlines each plate cell into the
# plate loop, so an authored `scan` inside a plate cell has the inlined shape.
#
# Rows: a scalar AR recurrence; the two-state washout step of ODL
# (`expm1` per step, three outputs per step) with constant per-path weights;
# and the same step with active per-path weights, once computed through a
# branch (`dend`/`dmean` of the active decay, which branch on `lambda < 0.1`)
# and once branch-free.
#
# Recorded on gordito, 2026-11-08, one thread, one CPU, at host load 23-31:
# ratio = inlined / @noinline, median of 30 paired, shuffled rounds, at
# G = 38 paths with T = 120 / T = 1320 steps; then the inlined form's warm
# allocation against the @noinline form's at T = 120.
#
#                           Julia 1.10.12      Julia 1.10.12      Julia 1.10.12      Julia 1.12.7
#                           dogfood c9ab4c9    dogfood 5e2cd2bb   dogfood 439590b1   dogfood 439590b1
#                           libEnzyme 42a2ee9a libEnzyme 576e674a libEnzyme 10018598 libEnzyme 10018598
#   scalar AR               1.19 / 1.14        0.96 / 1.00        0.98 / 0.99        0.98 / 1.05
#   washout, const weights  1.76 / 1.90        0.99 / 1.01        0.99 / 1.00        1.18 / 1.20
#   washout, active branch  2.05 / 1.86        1.11 / 1.14        0.97 / 1.02        1.15 / 1.23
#   washout, active, no br. 2.26 / 2.22        0.99 / 1.02        0.98 / 1.00        1.11 / 1.13
#   washout alloc (KB)      419.5 vs 308.6     306.0 vs 308.6     306.0 vs 308.6     438.0 vs 302.3
#
# Dogfood 5e2cd2bb repairs the extra tape and the per-path rebuild of the
# output on Julia 1.10 (Enzyme snag enzyme-reverse-n-69c10c81). Dogfood
# 439590b1 repairs the active per-path weights computed through a branch on
# Julia 1.10: they are rebuilt once per entry of the inner loop's reverse
# (Enzyme snag enzyme-reverse-i-6fb5e335); with EnzymeReverseLoopEntryMerges
# off, the same binary gives 1.11 / 1.17 on that row. On Julia 1.12 and 1.13
# the reverse pass still rebuilds each path's gathered slice and output instead
# of keeping them (Enzyme todo 1nw1luu). Gradients agree in every row.
using Enzyme, Random, Printf

# Deposit weights of ODL's washout step: series near zero, closed form above.
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
# The lambda >= 0.1 formulas alone: the same kind of work, without a branch.
dmean_nobranch(lambda) = (1.0 + expm1(-lambda) / lambda) / lambda
dend_nobranch(lambda) = -expm1(-lambda) / lambda

# --- scalar AR recurrence -----------------------------------------------------
for (name, inl) in ((:ar_noinl, false), (:ar_inl, true))
    f = :(function $name(r, a, s)
        out = similar(r)
        c = a
        for t in eachindex(r)
            c = c + (1 - c) * s * r[t]
            @inbounds out[t] = c
        end
        out
    end)
    @eval $(inl ? :(@inline $f) : :(@noinline $f))
end
for path in (:ar_noinl, :ar_inl)
    @eval function $(Symbol(:obj_, path))(rain, idx, rec, scav, dec, a0, g0, w)
        acc = 0.0
        for g in eachindex(idx)
            p = $path(rain[idx[g]], a0[g], scav[g])
            wg = w[g]
            for t in eachindex(p)
                acc += wg[t, 1] * p[t]
            end
        end
        acc
    end
end

# --- two-state washout step ---------------------------------------------------
for (name, inl) in ((:wash_noinl, false), (:wash_inl, true))
    f = :(function $name(r, recover, remaining, ew, mw, s, a0, g0)
        traj = similar(r, NTuple{3,Float64})
        air, ground = a0, g0
        for t in eachindex(r)
            air1 = air + (1 - air) * recover
            deposit = air1 * (-expm1(-s * r[t]))
            air = air1 - deposit
            o1 = ew * ground + mw * deposit
            ground = remaining * ground + ew * deposit
            @inbounds traj[t] = (o1, air, ground)
        end
        traj
    end)
    @eval $(inl ? :(@inline $f) : :(@noinline $f))
end
const WEIGHTS = (
    const_weights = (0.9, 0.5),
    active_branch = (:(dend(dec[g])), :(dmean(dec[g]))),
    active_nobranch = (:(dend_nobranch(dec[g])), :(dmean_nobranch(dec[g]))),
)
for path in (:wash_noinl, :wash_inl), wname in keys(WEIGHTS)
    ew, mw = WEIGHTS[wname]
    @eval function $(Symbol(:obj_, path, :_, wname))(rain, idx, rec, scav, dec, a0, g0, w)
        acc = 0.0
        for g in eachindex(idx)
            traj = $path(rain[idx[g]], -expm1(-rec[g]), exp(-dec[g]), $ew, $mw, scav[g], a0[g], g0[g])
            wg = w[g]
            for t in eachindex(traj)
                acc += wg[t, 1] * traj[t][1] + wg[t, 2] * traj[t][2] + wg[t, 3] * traj[t][3]
            end
        end
        acc
    end
end

function data(G, T; seed = 1)
    rng = Xoshiro(seed)
    rain = 2.0 .* rand(rng, G * T)
    idx = [collect(((g - 1) * T + 1):(g * T)) for g in 1:G]
    rec = 0.1 .+ 0.2 .* rand(rng, G)
    scav = 0.05 .+ 0.1 .* rand(rng, G)
    dec = 0.01 .+ 0.3 .* rand(rng, G)    # both branches of dmean/dend
    a0 = 0.2 .+ 0.6 .* rand(rng, G)
    g0 = rand(rng, G)
    w = [randn(rng, T, 3) for _ in 1:G]
    (; rain, idx, rec, scav, dec, a0, g0, w)
end
function gradient_call(f, x)
    d = map(zero, (x.rain, x.rec, x.scav, x.dec, x.a0, x.g0))
    () -> (foreach(v -> fill!(v, 0.0), d);
        Enzyme.autodiff(Reverse, f, Active, Duplicated(x.rain, d[1]), Const(x.idx),
            Duplicated(x.rec, d[2]), Duplicated(x.scav, d[3]), Duplicated(x.dec, d[4]),
            Duplicated(x.a0, d[5]), Duplicated(x.g0, d[6]), Const(x.w)); d)
end
quartiles(v) = (s = sort(v); q(p) = s[round(Int, p * (length(s) - 1)) + 1]; (q(0.25), q(0.5), q(0.75)))

const ROUNDS = parse(Int, get(ENV, "ROUNDS", "30"))
println("julia ", VERSION, "  Enzyme ", pkgversion(Enzyme), "  libEnzyme ",
    basename(dirname(dirname(Enzyme.Compiler.Enzyme_jll.libEnzyme_path))), "  threads ", Threads.nthreads())
for (G, T) in ((38, 120), (38, 1320))
    x = data(G, T)
    n = max(1, round(Int, 1e5 / (G * T)))
    println("\n== G=$G paths x T=$T steps ($ROUNDS shuffled rounds of $n calls)")
    for case in (:ar, :wash_const_weights, :wash_active_branch, :wash_active_nobranch)
        names = case === :ar ? (:obj_ar_noinl, :obj_ar_inl) :
            (Symbol(:obj_wash_noinl_, chopprefix(String(case), "wash_")),
             Symbol(:obj_wash_inl_, chopprefix(String(case), "wash_")))
        fs = [gradient_call(getfield(Main, s), x) for s in names]
        g1, g2 = fs[1](), fs[2]()
        all(maximum(abs.(a .- b)) < 1e-10 for (a, b) in zip(g1, g2)) || error("$case: gradients differ")
        for f in fs; f(); f(); end
        S = fill(Inf, ROUNDS, 2)
        rng = Xoshiro(7)
        for r in 1:ROUNDS, i in shuffle(rng, [1, 2])
            GC.gc(false)
            S[r, i] = (@elapsed for _ in 1:n; fs[i](); end) / n
        end
        lo, med, hi = quartiles(S[:, 2] ./ S[:, 1])
        @printf("  %-22s @noinline %8.1f us %8.1f KB | inlined %8.1f us %8.1f KB | ratio %.2f [%.2f, %.2f]\n",
            case, minimum(S[:, 1]) * 1e6, (@allocated fs[1]()) / 1024,
            minimum(S[:, 2]) * 1e6, (@allocated fs[2]()) / 1024, med, lo, hi)
    end
end
