# Synthetic loop attribution: ReactiveKernels-authored cells against hand loops
# of the same arithmetic, on generic data.
#
# Two shapes, each scalar and position-batched (`prepare_batched(...; reuse =
# true)`, the borrowed-output reader shape):
#   - superposition: a plate over observations summing weighted, shifted unit
#     responses (`get` cell, lattice shifts), against the dose-outer in-place
#     hand loop;
#   - relaxation scan: an `include_init` scan of an exponential relaxation
#     towards a concentration-dependent steady state, against a hand loop
#     filling one n + 1 buffer.
# Values are compared bitwise; times are per call (minimum and median of
# samples of repeated calls); bytes are one call's allocation.
#
# Undeclared array ports/WANTs exercise native column lanes when supported.
# Based on the public synthetic reporter fixture 57ea1cf1aede34a2.jl; only
# those declarations and the output directory differ from its arithmetic.
# Usage: julia --project=<env with ReactiveKernels and JSON3> \
#            benchmark/native_position_lanes.jl [output-directory]
using ReactiveKernels, JSON3
using ReactiveKernels: @kernel, prepare, plate, scan

const OUT = isempty(ARGS) ? joinpath(tempdir(), "native-position-lanes") : first(ARGS)
mkpath(OUT)
const ROWS = Any[]

function measure(f, args...; n = 41, target_s = 2e-3)
    f(args...); f(args...)
    bytes = @allocated f(args...)
    t0 = time_ns(); f(args...); single = max((time_ns() - t0) / 1e9, 1e-8)
    reps = max(1, round(Int, target_s / single))
    GC.gc()
    times = Vector{Float64}(undef, n)
    for i in 1:n
        t0 = time_ns()
        for _ in 1:reps
            f(args...)
        end
        times[i] = (time_ns() - t0) / 1e9 / reps
    end
    sort!(times)
    (; min_us = 1e6 * times[1], median_us = 1e6 * times[cld(n, 2)], bytes, reps, n)
end

function record!(shape, case, rk, hand, equal, maxabs)
    row = (; shape, case, rk, hand, ratio_min = rk.min_us / hand.min_us,
           ratio_median = rk.median_us / hand.median_us, bitwise_equal = equal, maxabs)
    push!(ROWS, row)
    println("ROW ", JSON3.write(row))
    flush(stdout)
end

# ---- superposition ---------------------------------------------------------

@kernel superpose(observations, shifts::Vector{Int}, units,
                  weights) = begin
    total = plate(observations, Ref(shifts), Ref(units), Ref(weights)) do t, s, u, w
        sum(w[j] * get(u, t - s[j], 0.0) for j in eachindex(w); init = 0.0)
    end
    return total
end
const SUPERPOSE = prepare(superpose)
const SUPERPOSE_BATCH = ReactiveKernels.prepare_batched(superpose;
    batched = (:units, :weights), want = :total, reuse = true)

function superpose_hand!(out, shifts, units, weights)
    fill!(out, 0.0)
    n, m = length(out), length(units)
    for j in eachindex(weights)
        s, w = shifts[j], weights[j]
        @inbounds for t in max(1, s + 1):min(n, s + m)
            out[t] += w * units[t - s]
        end
    end
    out
end
superpose_hand(n, shifts, units, weights) =
    superpose_hand!(Vector{Float64}(undef, n), shifts, units, weights)
function superpose_hand_batch!(out, shifts, U, W)
    for lane in axes(U, 2)
        superpose_hand!(view(out, :, lane), shifts, view(U, :, lane), view(W, :, lane))
    end
    out
end

for (nobs, ndoses) in ((6529, 3), (6529, 14), (16321, 3), (16321, 14))
    shifts = [(j - 1) * 192 for j in 1:ndoses]
    units = [exp(-1e-3 * i) * (1 - exp(-0.05 * i)) for i in 1:nobs]
    weights = [1.0 + 0.1 * j for j in 1:ndoses]
    rk = SUPERPOSE(1:nobs, shifts, units, weights)
    hand = superpose_hand(nobs, shifts, units, weights)
    record!("superposition", "scalar nobs=$nobs doses=$ndoses",
            measure(SUPERPOSE, 1:nobs, shifts, units, weights),
            measure(superpose_hand, nobs, shifts, units, weights),
            rk == hand, maximum(abs.(rk .- hand)))
    lanes = 32
    U = [units[i] * (1 + 0.01 * l) for i in 1:nobs, l in 1:lanes]
    W = [weights[j] * (1 + 0.02 * l) for j in 1:ndoses, l in 1:lanes]
    out = Matrix{Float64}(undef, nobs, lanes)
    rkb = copy(SUPERPOSE_BATCH(1:nobs, shifts, U, W))
    handb = copy(superpose_hand_batch!(out, shifts, U, W))
    record!("superposition", "batched lanes=$lanes nobs=$nobs doses=$ndoses",
            measure(SUPERPOSE_BATCH, 1:nobs, shifts, U, W),
            measure(superpose_hand_batch!, out, shifts, U, W),
            rkb == handb, maximum(abs.(rkb .- handb)))
end

# ---- relaxation scan --------------------------------------------------------

@kernel relax(drive, dts::Vector{Float64}, q) = begin
    trajectory = scan(drive, dts, Ref(q); init = q.r0,
                                       include_init = true) do previous, c, dt, p
        rate = p.k * (1 + c / (p.a * c + p.b))
        steady = p.r / rate
        next = (previous - steady) * exp(-rate * dt) + steady
        (next, next)
    end
    return trajectory
end
const RELAX = prepare(relax)
const RELAX_BATCH = ReactiveKernels.prepare_batched(relax;
    batched = (:drive, :q), want = :trajectory, reuse = true)

function relax_hand!(out, drive, dts, p)
    previous = p.r0
    out[1] = previous
    @inbounds for i in eachindex(drive)
        c, dt = drive[i], dts[i]
        rate = p.k * (1 + c / (p.a * c + p.b))
        steady = p.r / rate
        previous = (previous - steady) * exp(-rate * dt) + steady
        out[i + 1] = previous
    end
    out
end
relax_hand(drive, dts, p) = relax_hand!(Vector{Float64}(undef, length(drive) + 1), drive, dts, p)
function relax_hand_batch!(out, drive, dts, Q)
    for lane in axes(drive, 2)
        p = (; k = Q.k[lane], a = Q.a[lane], b = Q.b[lane], r = Q.r[lane], r0 = Q.r0[lane])
        relax_hand!(view(out, :, lane), view(drive, :, lane), dts, p)
    end
    out
end

for nsteps in (3264, 8160)
    dt = nsteps == 3264 ? 0.25 : 0.1
    dts = fill(dt, nsteps)
    drive = [5.0 * exp(-1e-3 * i) * (1 + 0.5 * sin(0.01 * i)) for i in 1:nsteps]
    q = (; k = 0.05, a = 0.2, b = 1.5, r = 50.0, r0 = 1000.0)
    rk = RELAX(drive, dts, q)
    hand = relax_hand(drive, dts, q)
    record!("relaxation scan", "scalar steps=$nsteps",
            measure(RELAX, drive, dts, q), measure(relax_hand, drive, dts, q),
            rk == hand, maximum(abs.(rk .- hand)))
    lanes = 32
    D = [drive[i] * (1 + 0.01 * l) for i in 1:nsteps, l in 1:lanes]
    Q = (; k = [0.05 + 0.001 * l for l in 1:lanes], a = fill(0.2, lanes), b = fill(1.5, lanes),
         r = [50.0 + l for l in 1:lanes], r0 = [1000.0 + 10l for l in 1:lanes])
    out = Matrix{Float64}(undef, nsteps + 1, lanes)
    rkb = copy(RELAX_BATCH(D, dts, Q))
    handb = copy(relax_hand_batch!(out, D, dts, Q))
    record!("relaxation scan", "batched lanes=$lanes steps=$nsteps",
            measure(RELAX_BATCH, D, dts, Q), measure(relax_hand_batch!, out, D, dts, Q),
            rkb == handb, maximum(abs.(rkb .- handb)))
end

open(joinpath(OUT, "summary.json"), "w") do io
    JSON3.write(io, (; julia = string(VERSION), threads = Threads.nthreads(), rows = ROWS))
end
open(joinpath(OUT, "summary.md"), "w") do io
    println(io, "| shape | case | RK min / median µs | hand min / median µs | RK/hand (min) | RK/hand (median) | RK bytes | hand bytes | bitwise |")
    println(io, "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in ROWS
        println(io, "| $(r.shape) | $(r.case) | $(round(r.rk.min_us; digits = 2)) / $(round(r.rk.median_us; digits = 2)) | ",
                "$(round(r.hand.min_us; digits = 2)) / $(round(r.hand.median_us; digits = 2)) | ",
                "$(round(r.ratio_min; digits = 3)) | $(round(r.ratio_median; digits = 3)) | $(r.rk.bytes) | $(r.hand.bytes) | $(r.bitwise_equal) |")
    end
end
println("LOOP_ATTRIBUTION_DONE rows=", length(ROWS))
