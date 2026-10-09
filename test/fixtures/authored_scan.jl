module AuthoredScanFixtures

using ReactiveKernels

@kernel authored_scan_arma(q::Vector{Float64}, series::Vector{Float64}) = begin
    mu::Float64 = sum(view(q, 1:1))
    phi::Float64 = sum(view(q, 2:2))
    theta::Float64 = sum(view(q, 3:3))
    errors::Vector{Float64} = scan(series;
            init = (; previous = mu, error = 0.0)) do carry, y
        e = y - (mu + phi * carry.previous + theta * carry.error)
        ((; previous = y, error = e), e)
    end
    pointwise = plate(errors) do e
        -0.5 * e^2
    end
    total::Float64 = sum(pointwise)
    energy::Float64 = sum(abs2, errors)
    joint::Float64 = total + energy
    return total
end

function _authored_scan_reference(q, series)
    previous, error = q[1], 0.0
    errors = similar(series)
    for i in eachindex(series)
        error = series[i] - (q[1] + q[2] * previous + q[3] * error)
        previous = series[i]
        errors[i] = error
    end
    errors
end

# A first-order linear recurrence carry_i = a[i] * carry_{i-1} + b[i] over TWO
# co-varying per-step sequences advanced in lockstep — the shape the single-`xs`
# scan could not author (each per-step operand had to be a broadcast-invariant
# `Ref`).  It feeds a plate + sum exactly like `authored_scan_arma`, so the same
# fused/materialized/Reactant lowering paths exercise the multi-sequence step.
@kernel authored_scan_lockstep(a::Vector{Float64}, b::Vector{Float64}) = begin
    seq::Vector{Float64} = scan(a, b; init = 0.0) do carry, ai, bi
        next = ai * carry + bi
        (next, next)
    end
    pointwise = plate(seq) do s
        -0.5 * s^2
    end
    total::Float64 = sum(pointwise)
    return total
end

function _authored_scan_lockstep_reference(a, b)
    seq = similar(a)
    carry = 0.0
    for i in eachindex(a, b)
        carry = a[i] * carry + b[i]
        seq[i] = carry
    end
    seq
end

# A forward-algorithm-shaped scan over matrix ROWS with a vector carry — the
# posteriordb hmm_drive_1 shape: `scan(eachrow(mat), Ref(gain); init = seed)`
# threads a 2-vector belief state and emits a scalar per step. The Reactant
# lowering keeps a single `stablehlo.while` for this shape (regression:
# scan-while-claim-5006b5b9); `N == 1` is a first-class case with an empty
# loop body after the eager first step.
@kernel authored_scan_eachrow(mat::Matrix{Float64}, gain::Float64) = begin
    seed::Vector{Float64} = [-0.6931471805599453, -0.6931471805599453]
    seq::Vector{Float64} = scan(eachrow(mat); init = seed) do carry, row
        emit = gain .* (row[1] .+ carry .* row[2])
        newg = emit .+ row[3]
        (newg, sum(newg))
    end
    total::Float64 = sum(seq)
    return total
end

function _authored_scan_eachrow_reference(mat, gain)
    carry = [-0.6931471805599453, -0.6931471805599453]
    seq = Vector{Float64}(undef, size(mat, 1))
    for (i, row) in enumerate(eachrow(mat))
        emit = gain .* (row[1] .+ carry .* row[2])
        carry = emit .+ row[3]
        seq[i] = sum(carry)
    end
    seq
end

# Keep scan at the top level of its own authored recipe, then call the prepared
# kernel only from the live arm of a separate graph. This preserves a genuine
# empty schedule without asking scan to infer an output type from zero steps.
@kernel authored_scan_nonempty(mat::Matrix{Float64}, positions::Vector{Int}) = begin
    weights::Vector{Float64} = scan(eachrow(mat);
            init = (; prior = zeros(Float64, length(positions)), index = 1)) do carry, row
        weight = row[1] + sum(carry.prior)
        next = ifelse.(positions .== carry.index, weight, carry.prior)
        ((; prior = next, index = carry.index + 1), weight)
    end
    return weights
end

const prepared_authored_scan_nonempty = prepare(authored_scan_nonempty)

# A triangular recurrence through `history = 0.0`: each weight reads every
# earlier weight through the response at their lag (the ShinyRK dose-feedback
# shape). The plan is a host struct shared by `Ref`, and the lag helper
# dispatches on it; lags past the response read zero through `get`.
struct HistoryLattice
    shifts::Vector{Int}
end
ReactiveKernels.@traceable _history_lag(p::HistoryLattice, j, i) =
    p.shifts[j] - p.shifts[i] + 1

@kernel authored_scan_history(amounts::Vector{Float64}, plan, units::Vector{Float64}) = begin
    weights::Vector{Float64} = scan(amounts, eachindex(amounts);
            init = 0, history = 0.0) do carry, amount, j, earlier
        exposure = sum(earlier[i] * get(units, _history_lag(plan, j, i), 0.0)
                       for i in 1:j-1; init = 0.0)
        weight = amount == 0 ? 0.0 : amount / (1 + exposure)
        (carry + 1, weight)
    end
    total::Float64 = sum(weights)
    return weights
end

function _authored_scan_history_reference(amounts, plan, units)
    weights = zeros(length(amounts))
    for j in eachindex(amounts)
        amounts[j] == 0 && continue
        exposure = 0.0
        for i in 1:j-1
            exposure += weights[i] * get(units, _history_lag(plan, j, i), 0.0)
        end
        weights[j] = amounts[j] / (1 + exposure)
    end
    weights
end

# The same lags stored as a 2-D host table (`lags[j, i]`, the stored-row plan
# of the dose-feedback shape): under a tracing backend the scan's `j` and the
# sum's `i` are both traced, so the read is a gather at two traced indices.
struct HistoryTable
    lags::Matrix{Int}
end
HistoryTable(p::HistoryLattice) = HistoryTable(
    [i < j ? p.shifts[j] - p.shifts[i] + 1 : 0
     for j in eachindex(p.shifts), i in eachindex(p.shifts)])
ReactiveKernels.@traceable _history_lag(p::HistoryTable, j, i) = p.lags[j, i]

# A step reading a partly traced named tuple: its scale is traced and its
# matrix may stay host. A `@traceable` helper sums over both axes of the
# matrix, each sum a retained loop with a traced index, so `k.W[i, j]` is a
# gather at two traced indices; the host matrix must cross both loops as a host
# value, not as a matrix of traced scalars.
ReactiveKernels.@traceable _scan_surface(k, a, b) =
    sum((sum((sin(a * i) * k.W[i, j] for i in axes(k.W, 1)); init = 0.0) * sin(b * j)
         for j in axes(k.W, 2)); init = 0.0)

@kernel authored_scan_surface(W, scale::Float64, xs::Vector{Float64}) = begin
    model = (; W, scale)
    ys::Vector{Float64} = scan(xs; init = 0.0) do carry, x
        value = model.scale * _scan_surface(model, x, x + 0.5)
        (carry + value, value)
    end
    total::Float64 = sum(ys)
    return ys
end

# Nested generator folds natively.  Julia 1.10 infers a Base fold reached while
# inferring another fold's term as `Any`, so a nested sum dispatches and boxes
# per element; RK runs every authored data-length fold as its loop instead.
# `_scan_surface_base` is the same nest as an ordinary (unrewritten) function:
# Base's value, for bitwise comparison.
_scan_surface_base(k, a, b) =
    sum((sum((sin(a * i) * k.W[i, j] for i in axes(k.W, 1)); init = 0.0) * sin(b * j)
         for j in axes(k.W, 2)); init = 0.0)

# The dose-feedback shape whose effective amount reads the nested-sum surface.
@kernel authored_scan_history_surface(amounts::Vector{Float64}, plan,
                                      units::Vector{Float64}, k) = begin
    weights::Vector{Float64} = scan(amounts, eachindex(amounts);
            init = 0, history = 0.0) do carry, amount, j, earlier
        exposure = sum(earlier[i] * get(units, _history_lag(plan, j, i), 0.0)
                       for i in 1:j-1; init = 0.0)
        weight = amount == 0 ? 0.0 :
            amount / (1 + exposure * (1 + _scan_surface(k, amount, exposure)^2))
        (carry + 1, weight)
    end
    return weights
end

function _authored_scan_history_surface_reference(amounts, plan, units, k)
    weights = zeros(length(amounts))
    for j in eachindex(amounts)
        amounts[j] == 0 && continue
        exposure = 0.0
        for i in 1:j-1
            exposure += weights[i] * get(units, _history_lag(plan, j, i), 0.0)
        end
        weights[j] = amounts[j] /
            (1 + exposure * (1 + _scan_surface_base(k, amounts[j], exposure)^2))
    end
    weights
end

# The same nest written inline in a scan step and in a plate cell.
@kernel authored_scan_nested_inline(xs::Vector{Float64}, k) = begin
    ys::Vector{Float64} = scan(xs; init = 0.0) do carry, x
        value = sum((sum((sin(x * i) * k.W[i, j] for i in axes(k.W, 1)); init = 0.0) *
                     sin((x + 0.5) * j) for j in axes(k.W, 2)); init = 0.0)
        (carry + value, value)
    end
    return ys
end

@kernel authored_plate_nested_inline(xs::Vector{Float64}, k) = begin
    ys::Vector{Float64} = plate(xs) do x
        sum((sum((sin(x * i) * k.W[i, j] for i in axes(k.W, 1)); init = 0.0) *
             sin((x + 0.5) * j) for j in axes(k.W, 2)); init = 0.0)
    end
    return ys
end

# Folds the native rewrite leaves to Base: a filter over an empty range and an
# `Int` seed with `Float64` terms (rewritten; Base's value and type), a term
# that assigns, and a fold over a tuple (both kept as written).
ReactiveKernels.@traceable _fold_filtered(x, n) = sum(x[i] for i in 1:n if x[i] > 0; init = 0)
ReactiveKernels.@traceable _fold_assigning(x) =
    sum((y = x[i]; y * y) for i in eachindex(x); init = 0.0)
ReactiveKernels.@traceable _fold_tuple(t) = sum(v for v in t; init = 0)

# A module whose own `sum` is not Base's keeps its call.
module OwnSum
using ReactiveKernels
sum(itr; init) = -1.0
ReactiveKernels.@traceable total(x) = sum(x[i] for i in eachindex(x); init = 0.0)
end

# A plain scan sharing a host struct and a host named tuple by `Ref`: under a
# tracing backend both stay host values inside the retained loop, so the plan
# helper indexes the struct and `_history_scale` loops over a host bound.
ReactiveKernels.@traceable _history_shift(p::HistoryLattice, j) = p.shifts[j]
function _history_scale(cfg, x)
    total = zero(x)
    for i in 1:cfg.m
        total += cfg.W[i, i] * x
    end
    total
end

@kernel authored_scan_host_shared(xs::Vector{Float64}, plan, cfg) = begin
    ys::Vector{Float64} = scan(xs, eachindex(xs); init = 0.0) do carry, x, j
        next = carry + x * _history_shift(plan, j) + _history_scale(cfg, x)
        (next, next)
    end
    return ys
end

# The history is the whole result vector: entries at and after the current step
# read the `history` value.
@kernel authored_scan_history_ahead(xs::Vector{Float64}) = begin
    seen::Vector{Float64} = scan(xs; init = 0.0, history = -1.0) do carry, x, earlier
        (carry, x + sum(earlier))
    end
    return seen
end

# A consumer using only `import ReactiveKernels` can still author a scan by
# qualifying its callee. The macro resolves that binding before constructing
# the scan op; a typed left-hand side is supported in either spelling.
module QualifiedScanBinding
import ReactiveKernels

const spec = ReactiveKernels.@kernel qualified_scan(xs::Vector{Float64}) = begin
    values::Vector{Float64} = ReactiveKernels.scan(xs; init = 0.0) do carry, x
        next = carry + x
        (next, next)
    end
    return values
end
const prepared = ReactiveKernels.prepare(spec)

end

module BareScanWithoutBinding
import ReactiveKernels

const spec = ReactiveKernels.@kernel unbound_scan(xs::Vector{Float64}) = begin
    values::Vector{Float64} = scan(xs; init = 0.0) do carry, x
        next = carry + x
        (next, next)
    end
    return values
end
const prepared = ReactiveKernels.prepare(spec)

end

@kernel authored_scan_lazy_branch(mat::Matrix{Float64}, positions::Vector{Int}, n::Int) = begin
    weights::Vector{Float64} = if n == 0
        Float64[]
    else
        prepared_authored_scan_nonempty(mat, positions)
    end
    total::Float64 = sum(weights)
    return total
end

end
