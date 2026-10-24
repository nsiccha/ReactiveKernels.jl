module PositionNestedCellTableTests

using Test
using ReactiveKernels
using ReactiveKernels: plate, scan, code_expr, derivative_rule

# A plate inside another plate's cell keeps per-cell tables for its stored
# results in a position-batched reader (todo 0c9x0bq, part 1), as a plate of
# the residual's top level does: inner cell `k` refills the buffer its result
# held at the previous outer cell. The outer cell's result drops the entries it
# may hold, so a stored inner result that leaves the outer cell is never
# overwritten. Values are those of the scalar kernel.

@kernel decay(ts, q) = begin
    y = plate(ts, Ref(q)) do t, q
        q[1] * exp(-q[2] * t)
    end
    return y
end

@kernel decay_graph(ts::Vector{Float64}, q::Vector{Float64}, y_bar::Vector{Float64}) = begin
    y::Vector{Float64} = decay(ts, q)
    ts_bar::Vector{Float64} = -q[2] .* y_bar .* y
    q_bar::Vector{Float64} = [sum(y_bar .* y) / q[1], -sum(y_bar .* ts .* y)]
    return y, ts_bar, q_bar
end
const decay_rule = derivative_rule(decay_graph; primal = :y, covector = :y_bar,
    cotangents = (ts = :ts_bar, q = :q_bar), name = :decay_rule)

# Per-treatment units stored inside a subject cell and read whole by two
# plates, as a population simulator's unit responses are.
@kernel stored_rule(theta, subjects, ts) = begin
    totals = plate(subjects) do treatments
        units = plate(treatments) do gp
            p = gp .* theta
            decay_rule(ts, p)
        end
        sums = plate(units) do u
            s::Float64 = sum(u)
            s
        end
        lasts = plate(units) do u
            l::Float64 = u[end]
            l
        end
        t::Float64 = sum(sums) + sum(lasts)
        t
    end
    return totals
end

@kernel stored_child(theta, subjects, ts) = begin
    totals = plate(subjects) do treatments
        units = plate(treatments) do gp
            p = gp .* theta
            decay(ts, p)
        end
        sums = plate(units) do u
            s::Float64 = sum(u)
            s
        end
        lasts = plate(units) do u
            l::Float64 = u[end]
            l
        end
        t::Float64 = sum(sums) + sum(lasts)
        t
    end
    return totals
end

@kernel recurrence(ts, q) = begin
    y = scan(ts, Ref(q); init = 0.0) do c, t, q
        n = c * exp(-q[2]) + q[1] * t
        (n, n)
    end
    return y
end

@kernel stored_arm(theta, subjects, ts, flag) = begin
    totals = plate(subjects) do treatments
        units = plate(treatments) do gp
            p = gp .* theta
            flag ? recurrence(ts, p) : decay(ts, p)
        end
        sums = plate(units) do u
            s::Float64 = sum(u)
            s
        end
        lasts = plate(units) do u
            l::Float64 = u[end]
            l
        end
        t::Float64 = sum(sums) + sum(lasts)
        t
    end
    return totals
end

# A rule whose primal is that branch: the rule call is the cell result.
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

@kernel stored_choice(theta, subjects, ts, flag) = begin
    totals = plate(subjects) do treatments
        units = plate(treatments) do gp
            p = gp .* theta
            choice_rule(ts, p, flag)
        end
        sums = plate(units) do u
            s::Float64 = sum(u)
            s
        end
        lasts = plate(units) do u
            l::Float64 = u[end]
            l
        end
        t::Float64 = sum(sums) + sum(lasts)
        t
    end
    return totals
end

@kernel stored_dotted(theta, subjects, ts) = begin
    totals = plate(subjects) do treatments
        units = plate(treatments) do gp
            gp[1] .* theta .* exp.(-gp[2] .* ts)
        end
        sums = plate(units) do u
            s::Float64 = sum(u)
            s
        end
        lasts = plate(units) do u
            l::Float64 = u[end]
            l
        end
        t::Float64 = sum(sums) + sum(lasts)
        t
    end
    return totals
end

# The subject cell's result IS its units: every stored inner result leaves the
# cell, and the subject plate's stored results are read afterwards. Without the
# outer release, the next subject would overwrite them.
total_of(units) = sum(sum, units)
last_of(units) = units[end][end]

@kernel held_units(theta, subjects, ts) = begin
    held = plate(subjects) do treatments
        units = plate(treatments) do gp
            p = gp .* theta
            decay(ts, p)
        end
        units
    end
    totals = plate(held) do units
        t::Float64 = total_of(units)
        t
    end
    lasts = plate(held) do units
        l::Float64 = last_of(units)
        l
    end
    total::Float64 = sum(totals) + sum(lasts)
    return total
end

# One stored inner result leaves the cell, the others do not.
@kernel held_one(theta, subjects, ts) = begin
    held = plate(subjects) do treatments
        units = plate(treatments) do gp
            p = gp .* theta
            decay(ts, p)
        end
        sums = plate(units) do u
            s::Float64 = sum(u)
            s
        end
        (units[1], sum(sums))
    end
    totals = plate(held) do pair
        t::Float64 = sum(pair[1]) + pair[2]
        t
    end
    total::Float64 = sum(totals)
    return total
end

# Three levels: stored units inside a subject cell inside a region cell.
@kernel stored_regions(theta, regions, ts) = begin
    totals = plate(regions) do subjects
        per_subject = plate(subjects) do treatments
            units = plate(treatments) do gp
                p = gp .* theta
                decay(ts, p)
            end
            sums = plate(units) do u
                s::Float64 = sum(u)
                s
            end
            lasts = plate(units) do u
                l::Float64 = u[end]
                l
            end
            t::Float64 = sum(sums) + sum(lasts)
            t
        end
        r::Float64 = sum(per_subject)
        r
    end
    return totals
end

# A child called in a lazy arm runs its lane kernel, whose own nested plates
# keep tables as well, per call site.
@kernel phase_totals(ts, q, phases) = begin
    totals = plate(phases, Ref(ts), Ref(q)) do ph, ts, q
        units = plate(ph, Ref(ts), Ref(q)) do w, ts, q
            w .* q[1] .* exp.(-q[2] .* ts)
        end
        sums = plate(units) do u
            s::Float64 = sum(u)
            s
        end
        lasts = plate(units) do u
            l::Float64 = u[end]
            l
        end
        t::Float64 = sum(sums) + sum(lasts)
        t
    end
    return totals
end

@kernel arm_phases(theta, subjects, ts, phases, flag) = begin
    totals = plate(subjects) do treatments
        p = treatments[1] .* theta
        v = flag ? phase_totals(ts, p, phases) : decay(ts, p)
        s::Float64 = sum(v)
        s
    end
    return totals
end

function subjects_input(count, positions; treatments = t -> 3)
    subjects = [[[1.0 + 0.01 * (s + t), 0.1 + 0.001 * t] for t in 1:treatments(s)]
                for s in 1:count]
    theta = collect(range(0.9, 1.1; length = positions))
    theta, subjects
end
times(n) = collect(range(0.0, 50.0; length = n))

reference(spec, theta, rest...) = let scalar = prepare(spec)
    r = [scalar(th, rest...) for th in theta]
    first(r) isa Number ? r : stack(r)
end
readers(spec; schedule = NativeScheduling(workers = 2, chunk_size = 2)) =
    (vectorize(spec; batched = :theta),
     copy(vectorize(spec; batched = :theta, reuse = true)),
     copy(vectorize(spec; batched = :theta, reuse = true, schedule)))
bytes(f::F, args...) where {F} = (f(args...); minimum(@allocated(f(args...)) for _ in 1:5))

@testset "nested cell tables: values" begin
    theta, subjects = subjects_input(5, 4; treatments = s -> 1 + s % 3)
    ts = times(30)
    for spec in (stored_rule, stored_child, stored_dotted, held_units, held_one)
        expected = reference(spec, theta, subjects, ts)
        for r in readers(spec), call in 1:3
            @test isequal(r(theta, subjects, ts), expected)
        end
    end
    for spec in (stored_arm, stored_choice), flag in (true, false)
        expected = reference(spec, theta, subjects, ts, flag)
        for r in readers(spec), call in 1:3
            @test isequal(r(theta, subjects, ts, flag), expected)
        end
    end
    regions = [subjects[1:2], subjects[3:5], subjects[2:4]]
    expected = reference(stored_regions, theta, regions, ts)
    for r in readers(stored_regions), call in 1:3
        @test isequal(r(theta, regions, ts), expected)
    end
    phases = [[1.0, 0.5, 0.25], [0.3], [0.7, 0.2]]
    for flag in (true, false)
        expected = reference(arm_phases, theta, subjects, ts, phases, flag)
        for r in readers(arm_phases), call in 1:3
            @test isequal(r(theta, subjects, ts, phases, flag), expected)
        end
    end
    @test occursin("_lane_call",
                   string(code_expr(vectorize(arm_phases; batched = :theta, reuse = true))))
end

@testset "nested cell tables: lowering" begin
    for spec in (stored_rule, stored_child, held_units, stored_regions)
        code = string(code_expr(vectorize(spec; batched = :theta, reuse = true)))
        @test occursin("_lane_tables_release!", code)
        # Outside a position residual no plate keeps tables.
        @test !occursin("_lane_tables_release!", string(code_expr(prepare(spec))))
    end
end

# On RK 5a21d2a4 a plate inside a cell kept no table: each stored unit vector
# was allocated at every cell and position (16128 B per treatment cell and
# position at 2000 lags). Now a borrowed read's cost does not grow with the
# number of cells, and stays far below one unit vector per cell.
@testset "nested cell tables: borrowed reads reuse stored inner results" begin
    ts = times(500)
    unit = sizeof(Float64) * length(ts)
    for spec in (stored_rule, stored_child, stored_dotted)
        r = copy(vectorize(spec; batched = :theta, reuse = true))
        per_cell = map(((4, 4), (16, 4), (16, 16))) do (count, positions)
            theta, subjects = subjects_input(count, positions)
            bytes(r, theta, subjects, ts) / (count * 3 * positions)
        end
        @test maximum(per_cell) < unit / 16
    end
    for flag in (true, false)
        r = copy(vectorize(stored_choice; batched = :theta, reuse = true))
        per_cell = map(((4, 4), (16, 4), (16, 16))) do (count, positions)
            theta, subjects = subjects_input(count, positions)
            bytes(r, theta, subjects, ts, flag) / (count * 3 * positions)
        end
        @test maximum(per_cell) < unit / 16
    end
    # A lazy branch with a shared condition lowers each arm into one shared
    # buffer and keeps no table at any depth, so its stored arm arrays still
    # allocate (todo 06nbgmo).
    for flag in (true, false)
        r = copy(vectorize(stored_arm; batched = :theta, reuse = true))
        per_cell = map(((4, 4), (16, 4))) do (count, positions)
            theta, subjects = subjects_input(count, positions)
            bytes(r, theta, subjects, ts, flag) / (count * 3 * positions)
        end
        @test_broken maximum(per_cell) < unit / 16
    end
    r = copy(vectorize(stored_regions; batched = :theta, reuse = true))
    per_cell = map(((4, 4), (8, 8))) do (count, positions)
        theta, subjects = subjects_input(count, positions)
        regions = [subjects, subjects]
        bytes(r, theta, regions, ts) / (2 * count * 3 * positions)
    end
    @test maximum(per_cell) < unit / 16
end

end
