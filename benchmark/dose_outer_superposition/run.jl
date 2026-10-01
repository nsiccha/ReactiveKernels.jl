# Native dose-outer lowering of a gathered generator-sum plate cell, measured
# on the ShinyRK superposition shape (snag one-natural-supe-39da86a4).
#
#     julia --project=<environment with ReactiveKernels> run.jl
#
# Rows compare, in one process and on the same data:
# - `get` cell: the natural cell, which RK lowers dose-outer;
# - `get` cell, cell loop: the same cell through an alias of `get` that the
#   lowering does not recognize, i.e. the observation-outer cell loop;
# - filtered cell: `sum(... u[t - s[j]] ... if t > s[j]; init = 0.0)`;
# - hand loop: the released in-place accumulation over shifted slices.
# Each value is the minimum over `SAMPLES` calls; bytes are for one call.
# Every RK form is checked bitwise against the cell loop.
using ReactiveKernels
using ReactiveKernels: prepare, plate, prepare_batched

const SAMPLES = 300

struct LatticePlan
    shifts::Vector{Int}
    nobs::Int
end
struct ExactPlan
    rows::Matrix{Int}
end
domain(plan::LatticePlan) = 1:plan.nobs
domain(plan::ExactPlan) = eachrow(plan.rows)
dose_table(observation, plan::LatticePlan) = plan.shifts
dose_table(row, ::ExactPlan) = row
dose_row(observation, ::LatticePlan, shift) = observation - shift
dose_row(row, ::ExactPlan, entry) = entry
const fetch = Base.get

@kernel get_cell(plan, units::Vector{Float64}, weights::Vector{Float64}) = begin
    observations = domain(plan)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum((w[i] * get(u, dose_row(t, p, dose_table(t, p)[i]), 0.0) for i in eachindex(w));
            init = 0.0)
    end
    return concentration
end
@kernel cell_loop(plan, units::Vector{Float64}, weights::Vector{Float64}) = begin
    observations = domain(plan)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum((w[i] * fetch(u, dose_row(t, p, dose_table(t, p)[i]), 0.0) for i in eachindex(w));
            init = 0.0)
    end
    return concentration
end
@kernel filtered_cell(plan, units::Vector{Float64}, weights::Vector{Float64}) = begin
    observations = 1:plan.nobs
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum(w[j] * u[t - p.shifts[j]] for j in eachindex(w) if t > p.shifts[j]; init = 0.0)
    end
    return concentration
end

function hand_loop(plan::LatticePlan, units, weights)
    concentration = zeros(Float64, plan.nobs)
    @inbounds for j in eachindex(weights)
        shift = plan.shifts[j]
        for i in max(shift + 1, 1):min(plan.nobs, length(units) + shift)
            concentration[i] += weights[j] * units[i - shift]
        end
    end
    concentration
end
function hand_loop(plan::ExactPlan, units, weights)
    concentration = zeros(Float64, size(plan.rows, 1))
    @inbounds for j in eachindex(weights), i in eachindex(concentration)
        index = plan.rows[i, j]
        index > 0 && (concentration[i] += weights[j] * units[index])
    end
    concentration
end

function measure(f, args...)
    f(args...)
    best = typemax(UInt64)
    for _ in 1:SAMPLES
        start = time_ns()
        f(args...)
        best = min(best, time_ns() - start)
    end
    bytes = @allocated f(args...)
    (best / 1e3, bytes)
end
same(a, b) = length(a) == length(b) && all(map(===, a, b))
cell(x) = string(round(x[1]; digits = 2), " µs / ", x[2], " B")

const KERNELS = (get = prepare(get_cell), loop = prepare(cell_loop),
                 filtered = prepare(filtered_cell))

println("Julia ", VERSION, ", ", Sys.MACHINE, ", ", Threads.nthreads(), " thread(s)")
println("| shape | `get` cell (dose-outer) | `get` cell, cell loop | filtered cell | hand loop |")
println("| --- | --- | --- | --- | --- |")
for (ndoses, nobs) in ((3, 16321), (3, 6529), (14, 6529), (14, 16321))
    shifts = collect(0:192:(192 * (ndoses - 1)))
    units = collect(range(0.5, 2.0; length = nobs))
    weights = collect(range(10.0, 60.0; length = ndoses))
    lattice = LatticePlan(shifts, nobs)
    exact = ExactPlan([max(t - s, 0) for t in 1:nobs, s in shifts])
    for (label, plan) in (("lattice", lattice), ("stored rows", exact))
        reference = KERNELS.loop(plan, units, weights)
        @assert same(KERNELS.get(plan, units, weights), reference)
        filtered = plan isa LatticePlan ?
            (@assert same(KERNELS.filtered(plan, units, weights), reference);
             cell(measure(KERNELS.filtered, plan, units, weights))) : "—"
        @assert hand_loop(plan, units, weights) ≈ reference
        println("| $label, $ndoses doses × $nobs obs | ",
                cell(measure(KERNELS.get, plan, units, weights)), " | ",
                cell(measure(KERNELS.loop, plan, units, weights)), " | ", filtered, " | ",
                cell(measure(hand_loop, plan, units, weights)), " |")
    end
end

# The consumer's position-batched read: 32 lanes, each with its own response
# and weights, one shared schedule; borrowed output buffers.
@kernel batched_get(scale::Float64, units::Vector{Float64}, plan, doses::Vector{Float64}) = begin
    weights::Vector{Float64} = scale .* doses
    observations = domain(plan)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum((w[i] * get(u, dose_row(t, p, dose_table(t, p)[i]), 0.0) for i in eachindex(w));
            init = 0.0)
    end
    return concentration
end
@kernel batched_loop(scale::Float64, units::Vector{Float64}, plan, doses::Vector{Float64}) = begin
    weights::Vector{Float64} = scale .* doses
    observations = domain(plan)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum((w[i] * fetch(u, dose_row(t, p, dose_table(t, p)[i]), 0.0) for i in eachindex(w));
            init = 0.0)
    end
    return concentration
end
reader(graph) = prepare_batched(graph; have = (:scale, :units, :plan, :doses),
                                batched = (:scale, :units), want = :concentration, reuse = true)
println()
println("| 32 lanes, 14 doses × 6529 obs | `get` cell (dose-outer) | `get` cell, cell loop |")
println("| --- | --- | --- |")
let lanes = 32, ndoses = 14, nobs = 6529
    shifts = collect(0:192:(192 * (ndoses - 1)))
    scales = collect(range(0.5, 1.5; length = lanes))
    units = reduce(hcat, [collect(range(0.5, 2.0; length = nobs)) .* l for l in 1:lanes])
    doses = fill(60.0, ndoses)
    for (label, plan) in (("lattice", LatticePlan(shifts, nobs)),
                          ("stored rows", ExactPlan([max(t - s, 0) for t in 1:nobs, s in shifts])))
        new, old = reader(batched_get), reader(batched_loop)
        @assert same(copy(new(scales, units, plan, doses)), copy(old(scales, units, plan, doses)))
        println("| $label | ", cell(measure(new, scales, units, plan, doses)), " | ",
                cell(measure(old, scales, units, plan, doses)), " |")
    end
end
