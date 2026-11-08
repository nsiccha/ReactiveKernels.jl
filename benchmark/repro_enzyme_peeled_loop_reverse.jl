# Standalone CPU reproducer: only Enzyme is required.
#
#   ENZYME_CONFIG=default|strict|nomerge julia --project=<env with Enzyme> repro_enzyme_peeled_loop_reverse.jl
#
# Native Enzyme Reverse of an outer loop over paths whose inner ARMA(1,1)
# error recursion writes a fresh vector that the outer loop then reads,
# written two ways with the same arithmetic, values and gradients:
#   - plain:  for t in eachindex(y) ... end
#   - peeled: the first step written out, then for t in first+1:last ... end
# each once inlined into the outer loop and once in a @noinline callee.
# ReactiveKernels lowers an authored `scan` with its first step peeled (the
# output type can come from that step), inlined into the plate loop.
#
# ENZYME_CONFIG sets the libEnzyme options before anything is compiled:
#   default  dogfood defaults
#   strict   Enzyme.API.fast_math_flags!()  (derivative arithmetic without nsz/contract)
#   nomerge  Enzyme.API.EnzymeSetCLBool(:EnzymeReverseLoopEntryMerges, false)
#
# Recorded on gordito, 2026-11-08, Julia 1.10.12, one thread, one CPU, at host
# load 20-29 on 32 CPUs: peeled / plain gradient time, median of 30 paired,
# shuffled rounds, at G = 38 paths with T = 120 / T = 1320 steps. Allocation is
# the same for both forms (122.0 KB inlined, 164.6 KB @noinline at T = 120). The
# primal of RK's peeled ARMA cell is within 2% of the plain hand loop's.
#
#                                         inlined        @noinline
#   dogfood b5daa2ac (libEnzyme 10018598), two default runs, one strict
#     default (nsz + contract)            1.01 / 1.01    0.96 / 0.96
#                                         1.06 / 1.05    0.98 / 0.98
#     strict                              0.94 / 0.91    1.05 / 1.06
#   dogfood 439590b1 (libEnzyme 10018598)
#     default (nsz + contract)            1.07 / 1.06    1.02 / 1.03
#     strict                              0.96 / 0.94    1.11 / 1.14
#     nomerge                             1.13 / 1.14    0.98 / 1.00
#   dogfood 5e2cd2bb (libEnzyme 576e674a)
#     default (no derivative flags yet)   0.98 / 0.95    1.10 / 1.13
#
# On 439590b1 the peeled loop costs up to 14% more in reverse in one placement
# or the other under every configuration, so the cost is Enzyme's: Enzyme snag
# enzyme-reverse-o-919a1558. A second default run on 439590b1 gave 1.08 / 1.05
# and 1.03 / 1.02. Enzyme's investigation of that snag found that nomerge gives
# these loops the same final code as the default, so that row differs from the
# default by run-to-run variance only (up to 0.08 between processes).
#
# Dogfood b5daa2ac (Enzyme's data_phi_loads repair, Julia 1.10 at default
# bounds) keeps the peeled carry a phi of values. Run interleaved with it at
# load 20-23, 439590b1 gave 1.08 / 1.03 and 1.07 / 1.07 inlined, 0.96 / 0.96
# and 0.96 / 0.99 @noinline, and 0.96 / 0.93 and 1.11 / 1.14 strict. What
# remains inlined is two loads per reverse step, the tape pointer and y's data
# pointer, that LLVM does not hoist. Under --check-bounds=yes the repair does
# not apply (Enzyme todo 16d9hce). RK's generated scan forms no such phi and is
# unchanged by it (scan_plate_cell_reverse.jl, Enzyme todo 08khey3).
using Enzyme, Random, Printf, Statistics

const CONFIG = get(ENV, "ENZYME_CONFIG", "default")
if CONFIG == "strict"
    Enzyme.API.fast_math_flags!()
elseif CONFIG == "nomerge"
    Enzyme.API.EnzymeSetCLBool(:EnzymeReverseLoopEntryMerges, false)
else
    CONFIG == "default" || error("unknown ENZYME_CONFIG $CONFIG")
end

for (suffix, inl) in (("inl", true), ("noinl", false))
    plain = Symbol(:path_plain_, suffix); peeled = Symbol(:path_peeled_, suffix)
    fp = :(function $plain(y, m, f, h)
        out = similar(y, typeof(m * f * h))
        yp, ep = m, zero(m * f * h)
        for t in eachindex(y)
            e = y[t] - (m + f * yp + h * ep)
            @inbounds out[t] = e
            yp, ep = y[t], e
        end
        out
    end)
    fq = :(function $peeled(y, m, f, h)
        out = similar(y, typeof(m * f * h))
        ix = eachindex(y)
        isempty(ix) && return out
        yp, ep = m, zero(m * f * h)
        t = first(ix)
        e = y[t] - (m + f * yp + h * ep)
        @inbounds out[t] = e
        yp, ep = y[t], e
        for t in (first(ix) + 1):last(ix)
            e = y[t] - (m + f * yp + h * ep)
            @inbounds out[t] = e
            yp, ep = y[t], e
        end
        out
    end)
    @eval $(inl ? :(@inline $fp) : :(@noinline $fp))
    @eval $(inl ? :(@inline $fq) : :(@noinline $fq))
end
for path in (:path_plain_inl, :path_peeled_inl, :path_plain_noinl, :path_peeled_noinl)
    @eval function $(Symbol(:obj_, path))(y, idx, mu, phi, theta)
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
const OBJS = (obj_path_plain_inl, obj_path_peeled_inl, obj_path_plain_noinl, obj_path_peeled_noinl)
const NAMES = ("plain, inlined", "peeled, inlined", "plain, @noinline", "peeled, @noinline")

function grad!(f, d, y, idx, mu, phi, theta)
    foreach(v -> fill!(v, 0.0), d)
    Enzyme.autodiff(Reverse, f, Active, Const(y), Const(idx), Duplicated(mu, d[1]),
        Duplicated(phi, d[2]), Duplicated(theta, d[3]))
    d
end

const ROUNDS = parse(Int, get(ENV, "ROUNDS", "30"))
println("julia ", VERSION, "  Enzyme ", pkgversion(Enzyme), "  libEnzyme ",
    basename(dirname(dirname(Enzyme.Compiler.Enzyme_jll.libEnzyme_path))), "  config ", CONFIG,
    "  threads ", Threads.nthreads())
for (G, T) in ((38, 120), (38, 1320))
    rng = Xoshiro(3)
    y = randn(rng, G * T); idx = [collect(((g - 1) * T + 1):(g * T)) for g in 1:G]
    mu = 0.1 .* randn(rng, G); phi = 0.5 .* rand(rng, G); theta = 0.3 .* rand(rng, G)
    args = (y, idx, mu, phi, theta)
    ds = [map(zero, (mu, phi, theta)) for _ in OBJS]
    vals = [f(args...) for f in OBJS]
    @assert all(v -> isapprox(v, vals[1]; rtol = 1e-12), vals)
    gs = [deepcopy(grad!(f, d, args...)) for (f, d) in zip(OBJS, ds)]
    err = maximum(maximum(maximum(abs.(a .- b)) for (a, b) in zip(g, gs[1])) for g in gs)
    @assert err < 1e-10
    n = max(1, round(Int, 1e5 / (G * T)))
    for (f, d) in zip(OBJS, ds); grad!(f, d, args...); end
    Sm = fill(Inf, ROUNDS, length(OBJS))
    for r in 1:ROUNDS, i in shuffle(rng, collect(eachindex(OBJS)))
        f, d = OBJS[i], ds[i]
        GC.gc(false)
        Sm[r, i] = (@elapsed for _ in 1:n; grad!(f, d, args...); end) / n
    end
    println("\n== G=$G paths x T=$T steps ($ROUNDS shuffled rounds of $n calls; gradients agree to $(round(err; sigdigits = 2)))")
    for (i, name) in enumerate(NAMES)
        base = iseven(i) ? i - 1 : i
        r = Sm[:, i] ./ Sm[:, base]
        a = @allocated grad!(OBJS[i], ds[i], args...)
        @printf("  %-18s med %8.1f us  min %8.1f us  %7.1f KB", name, median(Sm[:, i]) * 1e6,
            minimum(Sm[:, i]) * 1e6, a / 1024)
        iseven(i) ? @printf("  peeled/plain %.2f [%.2f, %.2f]\n", median(r), quantile(r, 0.25), quantile(r, 0.75)) : println()
    end
end
