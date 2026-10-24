module PositionCalleeLaneTests

using Test
using ReactiveKernels
using ReactiveKernels: plate, scan, code_expr, derivative_rule

# A position-batched reader reuses the buffers of RK-owned callees whose
# mathematics is a graph (snag rule-plate-splic-728756b1): a derivative rule's
# primal cut, called as a recipe or in a lazy arm, and a child kernel with a
# plate or scan called in a lazy arm. Each runs its lane kernel there: its
# result fills the call's lane slot, and its own plates, scans and dotted calls
# fill slots kept per call site. Values are those of the scalar kernel.

@kernel decay(ts, q) = begin
    y = plate(ts, Ref(q)) do t, q
        q[1] * exp(-q[2] * t)
    end
    return y
end

@kernel recurrence(ts, q) = begin
    y = scan(ts, Ref(q); init = 0.0) do c, t, q
        n = c * exp(-q[2]) + q[1] * t
        (n, n)
    end
    return y
end

# The primal is a plate; the reverse branch reads it.
@kernel decay_graph(ts::Vector{Float64}, q::Vector{Float64}, y_bar::Vector{Float64}) = begin
    y::Vector{Float64} = decay(ts, q)
    ts_bar::Vector{Float64} = -q[2] .* y_bar .* y
    q_bar::Vector{Float64} = [sum(y_bar .* y) / q[1], -sum(y_bar .* ts .* y)]
    return y, ts_bar, q_bar
end
const decay_rule = derivative_rule(decay_graph; primal = :y, covector = :y_bar,
    cotangents = (ts = :ts_bar, q = :q_bar), name = :decay_rule)

# The primal chooses between two children in a lazy branch.
@kernel choice_graph(ts::Vector{Float64}, q::Vector{Float64}, flag::Bool,
        y_bar::Vector{Float64}) = begin
    y::Vector{Float64} = flag ? recurrence(ts, q) : decay(ts, q)
    ts_bar::Vector{Float64} = y_bar .* 0.0
    q_bar::Vector{Float64} = [sum(y_bar .* y) / q[1], 0.0]
    flag_bar::Bool = false
    return y, ts_bar, q_bar, flag_bar
end
const choice_rule = derivative_rule(choice_graph; primal = :y, covector = :y_bar,
    cotangents = (ts = :ts_bar, q = :q_bar, flag = :flag_bar), name = :choice_rule)

# A subject plate whose cell runs a plate over treatments; each treatment's
# unit vector comes from one of the callees.
@kernel rule_cells(theta, subjects, ts) = begin
    totals = plate(subjects) do treatments
        units = plate(treatments) do gp
            p = gp .* theta
            decay_rule(ts, p)
        end
        sums = plate(units) do u
            sum(u)
        end
        sum(sums)
    end
    return totals
end

@kernel arm_cells(theta, subjects, ts, flag) = begin
    totals = plate(subjects) do treatments
        units = plate(treatments) do gp
            p = gp .* theta
            flag ? recurrence(ts, p) : decay(ts, p)
        end
        sums = plate(units) do u
            sum(u)
        end
        sum(sums)
    end
    return totals
end

@kernel choice_cells(theta, subjects, ts, flag) = begin
    totals = plate(subjects) do treatments
        units = plate(treatments) do gp
            p = gp .* theta
            choice_rule(ts, p, flag)
        end
        sums = plate(units) do u
            sum(u)
        end
        sum(sums)
    end
    return totals
end

# Each subject has its own times, so the unit vectors differ in length, and
# two plates read the stored units: a cell's stored result keeps its buffer.
@kernel stored_cells(theta, groups, times) = begin
    units = plate(groups, times) do g, t
        p = g .* theta
        decay_rule(t, p)
    end
    sums = plate(units) do u
        s::Float64 = sum(u)
        s
    end
    lasts = plate(units) do u
        l::Float64 = isempty(u) ? 0.0 : u[end]
        l
    end
    total::Float64 = sum(sums) + sum(lasts)
    return total
end

function subjects_input(count, treatments, positions)
    subjects = [[[1.0 + 0.01 * (s + t), 0.1 + 0.001 * t] for t in 1:treatments]
                for s in 1:count]
    theta = collect(range(0.9, 1.1; length = positions))
    theta, subjects
end
times(n) = collect(range(0.0, 50.0; length = n))

reference(spec, theta, rest...) = let scalar = prepare(spec)
    stack([scalar(th, rest...) for th in theta]; dims = ndims(scalar(first(theta), rest...)) + 1)
end
reference_scalar(spec, theta, rest...) = let scalar = prepare(spec)
    [scalar(th, rest...) for th in theta]
end
bytes(f::F, args...) where {F} = (f(args...); minimum(@allocated(f(args...)) for _ in 1:5))

@testset "callee lanes: values" begin
    schedule = NativeScheduling(workers = 2, chunk_size = 2)
    theta, subjects = subjects_input(5, 3, 4)
    ts = times(40)
    cases = ((rule_cells, (theta, subjects, ts)),
             (arm_cells, (theta, subjects, ts, true)),
             (arm_cells, (theta, subjects, ts, false)),
             (choice_cells, (theta, subjects, ts, true)),
             (choice_cells, (theta, subjects, ts, false)))
    for (spec, args) in cases
        expected = reference(spec, args...)
        readers = (vectorize(spec; batched = :theta), copy(vectorize(spec; batched = :theta, reuse = true)),
                   copy(vectorize(spec; batched = :theta, reuse = true, schedule)))
        for r in readers, call in 1:2
            @test isequal(r(args...), expected)
        end
    end
    # Ragged times and stored cell results.
    groups = [[1.0 + 0.1 * g, 0.05 * g] for g in 1:6]
    ragged = [times(n) for n in (7, 0, 12, 7, 3, 12)]
    expected = reference_scalar(stored_cells, theta, groups, ragged)
    for r in (vectorize(stored_cells; batched = :theta),
              copy(vectorize(stored_cells; batched = :theta, reuse = true)),
              copy(vectorize(stored_cells; batched = :theta, reuse = true, schedule))), call in 1:2
        @test isequal(r(theta, groups, ragged), expected)
    end
end

@testset "callee lanes: lowering" begin
    for spec in (rule_cells, arm_cells, choice_cells)
        @test occursin("_lane_call", string(code_expr(vectorize(spec; batched = :theta, reuse = true))))
        # Outside a position residual the callee keeps its ordinary call.
        @test !occursin("_lane_call", string(code_expr(prepare(spec))))
    end
    # A rule whose graph holds a plate stays an isbits value, and its primal
    # cut is the prepared graph's.
    @test isbits(decay_rule)
    ts = times(9); q = [1.2, 0.1]
    @test decay_rule(ts, q) == prepare(decay; have = (:ts, :q), want = :y)(ts, q)
end

# On RK 25ed45ff2 each unit vector was allocated at every treatment cell and
# position (16128 B per cell and position at 2000 lags, the rule's or child's
# own result). Now a borrowed read's cost does not grow with the number of
# cells, and stays far below one unit vector per cell.
@testset "callee lanes: borrowed reads reuse the callees' buffers" begin
    ts = times(500)
    unit = sizeof(Float64) * length(ts)
    for (spec, extra) in ((rule_cells, ()), (arm_cells, (true,)), (arm_cells, (false,)),
                          (choice_cells, (true,)), (choice_cells, (false,)))
        r = copy(vectorize(spec; batched = :theta, reuse = true))
        per_cell = map(((4, 4), (16, 4), (16, 16))) do (count, positions)
            theta, subjects = subjects_input(count, 3, positions)
            bytes(r, theta, subjects, ts, extra...) / (count * 3 * positions)
        end
        @test maximum(per_cell) < unit / 16
    end
end

end
