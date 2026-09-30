using ReactiveKernels, Reactant, DifferentiationInterface, Test
import Enzyme
isdefined(@__MODULE__, :AuthoredPlateChains) ||
    include("fixtures/authored_plate_chains.jl")

module RefArrayPlateFixtures
using ReactiveKernels

@kernel broadcast_axes(q::Vector{Float64}, x, y) = begin
    pointwise = plate(x, Ref(q), y) do xi, whole, yi
        xi + sum(whole) * yi
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel mixed(q::Vector{Float64}) = begin
    pointwise = plate(q, Ref(q)) do qi, whole
        qi + sum(whole)
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel matrix_atom(q::Matrix{Float64}, x::Vector{Float64}) = begin
    pointwise = plate(Ref(q), x) do whole, xi
        sum(whole) * xi
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel scalar_atom(s::Float64, x::Vector{Float64}, y::Vector{Float64}) = begin
    middle = plate(x, Ref(s)) do xi, shared
        2 * xi + shared
    end
    pointwise = plate(y, middle, Ref(s)) do yi, mi, shared
        yi + mi^2 + shared * mi
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel columns(q::Vector{Float64}, x::Matrix{Float64}, weights::Vector{Float64}) = begin
    pointwise = plate(Ref(q), weights, eachcol(x)) do whole, weight, column
        weight * sum(column .* whole)
    end
    total::Float64 = sum(pointwise)
    return total
end
end

module NestedLanePlateFixtures
using ReactiveKernels

# A host schedule with a compact per-dose plan, embedded as one prepared plate
# child in a batched outer graph: the cell's first recipe is all-host data, so
# the tensorized lowering materializes one row vector per lane before the
# traced gather recipe runs.
struct GatherSchedule
    nobs::Int
    shifts::Vector{Int}
end

_superpose_row(row, response, amounts) =
    sum(response[max.(row, 1)] .* (row .> 0) .* amounts)

@kernel superposition(plan, units, weights) = begin
    observations = collect(1:plan.nobs)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do observation, schedule_plan, response, amounts
        row = observation .- schedule_plan.shifts
        _superpose_row(row, response, amounts)
    end
    return concentration
end
const SUPERPOSITION_PREPARED = prepare(superposition)

@kernel amount_outer(units, weights, schedule) = begin
    plan = schedule.plan
    concentration = SUPERPOSITION_PREPARED(plan, units, weights)
    return concentration
end

@kernel traced_superposition(obs, shifts, units, weights) = begin
    concentration::Vector{Float64} = plate(obs, Ref(shifts), Ref(units), Ref(weights)) do observation, sh, response, amounts
        row = observation .- sh
        _superpose_row(row, response, amounts)
    end
    return concentration
end
const TRACED_SUPERPOSITION_PREPARED = prepare(traced_superposition)

@kernel traced_amount_outer(obs, shifts, units, weights) = begin
    concentration = TRACED_SUPERPOSITION_PREPARED(obs, shifts, units, weights)
    return concentration
end

_ragged_rows(n) = [collect(1:i) for i in 1:n]

@kernel ragged_inner(n, units) = begin
    rows = _ragged_rows(n)
    out::Vector{Float64} = plate(rows, Ref(units)) do row, response
        sum(response[row])
    end
    return out
end
const RAGGED_PREPARED = prepare(ragged_inner)

@kernel ragged_outer(n, units) = begin
    out = RAGGED_PREPARED(n, units)
    return out
end
end

@testset "Ref array plates retain atomic parameters in Reactant" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    q = [0.7, -0.3]
    rq = Reactant.to_rarray(q)
    @testset "N=$n" for n in (8, 32, 1, 0)
        x = n == 0 ? Float64[] : n == 1 ? [0.0] :
            collect(range(-1.0, 1.0; length = n))
        y = fill(0.3, n)
        middle = 2 .* x .+ q[1]
        pointwise = y .+ middle.^2 .+ q[2] .* middle
        total = sum(pointwise)
        # This is the reporter's exact graph: the only live HAVE is Ref(q),
        # whose length is deliberately different from the bound lane count.
        k = prepare(AuthoredPlateChains.ref_atomic_chain; bound = (; x, y))
        compiled = Reactant.@compile sync = true k(rq)
        @test Float64(compiled(rq)) ≈ total
        @test k(q) ≈ total

        both = prepare(AuthoredPlateChains.ref_atomic_chain;
            want = (:pointwise, :total), bound = (; x, y))
        compiled_both = Reactant.@compile sync = true both(rq)
        values, value = compiled_both(rq)
        @test Array(values) ≈ pointwise
        @test Float64(value) ≈ total
        demanded = prepare(AuthoredPlateChains.ref_atomic_chain;
            want = :middle, bound = (; x, y))
        compiled_middle = Reactant.@compile sync = true demanded(rq)
        @test Array(compiled_middle(rq)) ≈ middle

        ad = prepare_ad(k, backend, q; active = :q)
        compiled_ad = Reactant.@compile sync = true ad_value_and_gradient(ad, rq)
        value, gradient = compiled_ad(ad, rq)
        @test Float64(value) ≈ total
        @test Array(gradient) ≈ [sum(2 .* middle .+ q[2]), sum(middle)]

        # A compiled closure must keep q dynamic, not freeze its first payload.
        q2 = [-0.2, 0.5]
        rq2 = Reactant.to_rarray(q2)
        middle2 = 2 .* x .+ q2[1]
        value2, gradient2 = compiled_ad(ad, rq2)
        @test Float64(value2) ≈ sum(y .+ middle2.^2 .+ q2[2] .* middle2)
        @test Array(gradient2) ≈ [sum(2 .* middle2 .+ q2[2]), sum(middle2)]

        unbound = prepare(AuthoredPlateChains.ref_atomic_chain)
        rx, ry = Reactant.to_rarray.((x, y))
        compiled_unbound = Reactant.@compile sync = true unbound(rq, rx, ry)
        @test Float64(compiled_unbound(rq, rx, ry)) ≈ total
        scalar = prepare(RefArrayPlateFixtures.scalar_atom; bound = (; x, y))
        rs = Reactant.to_rarray(0.5)
        compiled_scalar = Reactant.@compile sync = true scalar(rs)
        scalar_middle = 2 .* x .+ 0.5
        @test Float64(compiled_scalar(rs)) ≈
            sum(y .+ scalar_middle.^2 .+ 0.5 .* scalar_middle)
    end

    @testset "broadcast axes and shared rank are independent" begin
        for (x, y) in (([1.0], collect(1.0:32)),
                      ((1.0,), collect(1.0:32)),
                      (reshape([1.0, 2.0], 2, 1), reshape(collect(1.0:3), 1, 3)))
            k = prepare(RefArrayPlateFixtures.broadcast_axes;
                want = :pointwise, bound = (; x, y))
            compiled = Reactant.@compile sync = true k(rq)
            actual = Array(compiled(rq))
            reference = x .+ sum(q) .* y
            @test size(actual) == size(reference)
            @test actual ≈ reference
        end
        qmatrix = reshape(collect(0.1:0.1:0.6), 2, 3)
        rmatrix = Reactant.to_rarray(qmatrix)
        x = collect(1.0:32)
        k = prepare(RefArrayPlateFixtures.matrix_atom; bound = (; x))
        compiled = Reactant.@compile sync = true k(rmatrix)
        @test Float64(compiled(rmatrix)) ≈ sum(qmatrix) * sum(x)
        ad = prepare_ad(k, backend, qmatrix; active = :q)
        compiled_ad = Reactant.@compile sync = true ad_value_and_gradient(ad, rmatrix)
        _, gradient = compiled_ad(ad, rmatrix)
        @test Array(gradient) ≈ fill(sum(x), size(qmatrix))
    end

    @testset "one array is both elementwise and atomic" begin
        for n in (8, 32)
            qmixed = collect(1.0:n) ./ n
            rmixed = Reactant.to_rarray(qmixed)
            k = prepare(RefArrayPlateFixtures.mixed)
            compiled = Reactant.@compile sync = true k(rmixed)
            @test Float64(compiled(rmixed)) ≈ (n + 1) * sum(qmixed)
            ad = prepare_ad(k, backend, qmixed; active = :q)
            compiled_ad = Reactant.@compile sync = true ad_value_and_gradient(ad, rmixed)
            _, gradient = compiled_ad(ad, rmixed)
            @test Array(gradient) ≈ fill(n + 1, n)
        end
    end

    @testset "eachcol keeps its structural batch with a leading Ref" begin
        for n in (8, 32)
            x = reshape(collect(1.0:(2n)) ./ n, 2, n)
            weights = collect(1.0:n) ./ n
            rx = Reactant.to_rarray(x)
            k = prepare(RefArrayPlateFixtures.columns; bound = (; weights))
            compiled = Reactant.@compile sync = true k(rq, rx)
            reference_gradient = vec(sum(x .* reshape(weights, 1, n); dims = 2))
            @test Float64(compiled(rq, rx)) ≈ sum(q .* reference_gradient)
            @test k(q, x) ≈ sum(q .* reference_gradient)
            ad = prepare_ad(k, backend, q, x; active = :q)
            compiled_ad = Reactant.@compile sync = true ad_value_and_gradient(ad, rq, rx)
            _, gradient = compiled_ad(ad, rq, rx)
            @test Array(gradient) ≈ reference_gradient
        end
    end

    @testset "host-plan masked gather keeps lane structure under batching" begin
        # Lane count (6), dose count (2), and subject batch (3) are all
        # distinct, so any lane/batch confusion changes shapes or values.
        plan = NestedLanePlateFixtures.GatherSchedule(6, [0, 3])
        schedule = (; plan)
        units = reshape(collect(1.0:18.0), 6, 3)
        weights = [1.0 2.0 3.0; 4.0 5.0 6.0]
        B = prepare_batched(NestedLanePlateFixtures.amount_outer;
            have = (:units, :weights, :schedule),
            batched = (:units, :weights), want = :concentration)
        reference = B(units, weights, schedule)
        expected = Matrix{Float64}(undef, 6, 3)
        for s in 1:3, o in 1:6
            row = o .- plan.shifts
            expected[o, s] = sum(units[max.(row, 1), s] .* (row .> 0) .* weights[:, s])
        end
        @test reference ≈ expected
        f = (u, w, s) -> B(u, w, s)
        runits, rweights = Reactant.to_rarray(units), Reactant.to_rarray(weights)
        hlo = repr(Reactant.@code_hlo optimize = :none f(runits, rweights, schedule))
        @test occursin("stablehlo.while", hlo)
        compiled = Reactant.@compile sync = true f(runits, rweights, schedule)
        @test Array(compiled(runits, rweights, schedule)) ≈ reference
        # Changed inputs reuse the same program with fresh values.
        units2 = units .* 1.5 .+ 1.0
        weights2 = weights .+ 2.0
        runits2 = Reactant.to_rarray(units2)
        rweights2 = Reactant.to_rarray(weights2)
        reference2 = B(units2, weights2, schedule)
        @test reference2 != reference
        @test Array(compiled(runits2, rweights2, schedule)) ≈ reference2
        # Doubling the lane count must not replicate the program.
        plan12 = NestedLanePlateFixtures.GatherSchedule(12, [0, 3])
        schedule12 = (; plan = plan12)
        units12 = reshape(collect(1.0:36.0), 12, 3)
        runits12 = Reactant.to_rarray(units12)
        hlo12 = repr(Reactant.@code_hlo optimize = :none f(runits12, rweights, schedule12))
        @test count("stablehlo.while", hlo12) == count("stablehlo.while", hlo)
        @test count("\n", hlo12) == count("\n", hlo)
        compiled12 = Reactant.@compile sync = true f(runits12, rweights, schedule12)
        @test Array(compiled12(runits12, rweights, schedule12)) ≈
            B(units12, weights, schedule12)
    end

    @testset "traced lane axis keeps vector lanes whole" begin
        obs = collect(1:6)
        shifts = [0, 2, 4]
        units = reshape(collect(1.0:18.0), 6, 3)
        weights = [5.0, 6.0, 7.0]
        B = prepare_batched(NestedLanePlateFixtures.traced_amount_outer;
            have = (:obs, :shifts, :units, :weights),
            batched = (:units,), want = :concentration)
        reference = B(obs, shifts, units, weights)
        @test size(reference) == (6, 3)
        f = (o, s, u, w) -> B(o, s, u, w)
        robs = Reactant.to_rarray(obs)
        runits = Reactant.to_rarray(units)
        compiled = Reactant.@compile sync = true f(robs, shifts, runits, weights)
        got = Array(compiled(robs, shifts, runits, weights))
        @test size(got) == (6, 3)
        @test got ≈ reference
    end

    @testset "ragged and empty per-lane values raise loudly" begin
        units = [10.0 20.0; 30.0 40.0; 50.0 60.0]
        B = prepare_batched(NestedLanePlateFixtures.ragged_outer;
            have = (:n, :units), batched = (:units,), want = :out)
        @test B(3, units) ≈ [10.0 20.0; 40.0 60.0; 90.0 120.0]
        f = (nn, u) -> B(nn, u)
        runits = Reactant.to_rarray(units)
        @test_throws ArgumentError Reactant.@compile sync = true f(3, runits)
        @test isempty(B(0, units))
        @test_throws ArgumentError Reactant.@compile sync = true f(0, runits)
    end
end

# --- per-cell reductions over a few host indices (snag plate-cell-gathe-94d4a929)
module GatherGeneratorFixtures
using ReactiveKernels

struct Lattice
    shifts::Vector{Int}
    nobs::Int
end
struct Exact
    rows::Matrix{Int}
end
_domain(plan::Lattice) = collect(1:plan.nobs)
_domain(plan::Exact) = eachrow(plan.rows)
_slots(plan::Lattice) = eachindex(plan.shifts)
_slots(plan::Exact) = axes(plan.rows, 2)
_index(observation, plan::Lattice, i) = observation - plan.shifts[i]
# A stored per-lane row reaches the cell as a traced slice under batching;
# an opaque helper reads one element as a one-element reduction rather than
# `row[i]`, which Reactant's scalar-indexing guard refuses outside the
# `@kernel`-lowered `_tensorized_getindex` rewrite.
_index(row, ::Exact, i) = sum(view(row, i:i))

@kernel generator_cell(plan, units, weights) = begin
    observations = _domain(plan)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do observation, schedule_plan, response, amounts
        sum(ifelse(_index(observation, schedule_plan, i) > 0,
                   response[max(_index(observation, schedule_plan, i), 1)] * amounts[i], 0.0)
            for i in _slots(schedule_plan))
    end
    return concentration
end

# The natural spelling with a cell-local copy of the shared host vector: that
# local is loop-invariant (its only root is the `Ref`-atomic plan), so it must
# stay atomic under the tensorized lowering instead of becoming a lane axis.
@kernel invariant_local_cell(plan, units, weights) = begin
    observations = collect(1:plan.nobs)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do observation, schedule_plan, response, amounts
        shifts = schedule_plan.shifts
        sum(ifelse(observation - shifts[i] > 0,
                   response[max(observation - shifts[i], 1)] * amounts[i], 0.0)
            for i in eachindex(shifts))
    end
    return concentration
end

@kernel invariant_local(xs, shared) = begin
    ys::Vector{Float64} = plate(xs, Ref(shared)) do x, s
        w = s.vec
        x * w[1] + w[2]
    end
    return ys
end

# The inner reduction prepared as its own reducing plate: `resp` is a shared
# array HAVE that must reach every lane whole, not as a broadcast axis.
@kernel row_cell(shift, amount, obs, resp) = begin
    row = obs - shift
    contribution::Float64 = resp[max(row, 1)] * (row > 0) * amount
    return contribution
end
const ROW_TOTAL = plate(row_cell; have = (:shift, :amount, :obs, :resp),
                        want = :contribution, batched = (:shift, :amount),
                        reduce = :+)
end

@testset "per-cell generator reductions over host indices lower under batching" begin
    F = GatherGeneratorFixtures
    nobs = 40
    shifts = [0, 7, 19]
    units = collect(range(0.5, 2.0; length = nobs))
    weights = [3.0, 1.5, 0.25]
    expected = [sum(o - s > 0 ? units[o - s] * w : 0.0
                    for (s, w) in zip(shifts, weights)) for o in 1:nobs]
    runits, rweights = Reactant.to_rarray(units), Reactant.to_rarray(weights)
    units2 = units .* 1.5 .+ 1.0
    runits2 = Reactant.to_rarray(units2)

    @testset "generator cell: $(nameof(typeof(plan)))" for plan in (
            F.Lattice(shifts, nobs), F.Exact([o - s for o in 1:nobs, s in shifts]))
        k = prepare(F.generator_cell)
        @test k(plan, units, weights) ≈ expected
        compiled = Reactant.@compile sync = true k(plan, runits, rweights)
        @test Array(compiled(plan, runits, rweights)) ≈ expected
        @test Array(compiled(plan, runits2, rweights)) ≈ k(plan, units2, weights)
    end

    @testset "doubling the lane count does not replicate the program" begin
        k = prepare(F.generator_cell)
        plan = F.Lattice(shifts, nobs)
        plan2 = F.Lattice(shifts, 2 * nobs)
        runits_double = Reactant.to_rarray(vcat(units, units))
        hlo = repr(Reactant.@code_hlo optimize = :none k(plan, runits, rweights))
        hlo2 = repr(Reactant.@code_hlo optimize = :none k(plan2, runits_double, rweights))
        @test count("\n", hlo2) == count("\n", hlo)
        @test count("stablehlo.while", hlo2) == count("stablehlo.while", hlo)
    end

    @testset "loop-invariant cell-local array stays atomic" begin
        plan = F.Lattice(shifts, nobs)
        k = prepare(F.invariant_local_cell)
        @test k(plan, units, weights) ≈ expected
        compiled = Reactant.@compile sync = true k(plan, runits, rweights)
        @test Array(compiled(plan, runits, rweights)) ≈ expected

        ki = prepare(F.invariant_local)
        xs = collect(1.0:20.0)
        shared = (; vec = [2.0, 0.5])
        rxs = Reactant.to_rarray(xs)
        compiled_local = Reactant.@compile sync = true ki(rxs, shared)
        @test Array(compiled_local(rxs, shared)) ≈ ki(xs, shared)
        @test ki(xs, shared) ≈ xs .* 2.0 .+ 0.5
    end

    @testset "plate(spec; batched) keeps a shared array HAVE whole" begin
        amounts = [1.0, 2.0, 3.0]
        resp = collect(1.0:10.0)
        obs = 6
        native = F.ROW_TOTAL(shifts, amounts, obs, resp)
        @test native ≈ sum(obs - s > 0 ? resp[obs - s] * a : 0.0
                           for (s, a) in zip(shifts, amounts))
        ramounts, rresp = Reactant.to_rarray(amounts), Reactant.to_rarray(resp)
        compiled = Reactant.@compile sync = true F.ROW_TOTAL(shifts, ramounts, obs, rresp)
        @test Float64(compiled(shifts, ramounts, obs, rresp)) ≈ native
        rshifts = Reactant.to_rarray(shifts)
        compiled_traced = Reactant.@compile sync = true F.ROW_TOTAL(rshifts, ramounts, obs, rresp)
        @test Float64(compiled_traced(rshifts, ramounts, obs, rresp)) ≈ native
    end
end

# --- the natural superposition cell (snag one-natural-supe-39da86a4)
module NaturalSuperpositionFixtures
using ReactiveKernels

struct Lattice
    shifts::Vector{Int}
    nobs::Int
end
struct Exact
    rows::Matrix{Int}
end
_domain(plan::Lattice) = 1:plan.nobs
_domain(plan::Exact) = eachrow(plan.rows)
_doses(plan::Lattice) = eachindex(plan.shifts)
_doses(plan::Exact) = axes(plan.rows, 2)
_lag(t, plan::Lattice, j) = t - plan.shifts[j]
# Inside an opaque helper a stored lane row still needs the one-element
# reduction under Reactant (its scalar-indexing guard); inline `row[j]` in the
# cell body is lowered by RK.
_lag(row, ::Exact, j) = sum(view(row, j:j))

@kernel get_cell(plan, units, weights) = begin
    observations::UnitRange{Int} = 1:plan.nobs
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum(w[j] * get(u, t - p.shifts[j], 0.0) for j in eachindex(p.shifts); init = 0.0)
    end
    return concentration
end
@kernel filter_cell(plan, units, weights) = begin
    observations::UnitRange{Int} = 1:plan.nobs
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum(w[j] * u[t - p.shifts[j]] for j in eachindex(p.shifts) if t > p.shifts[j]; init = 0.0)
    end
    return concentration
end
@kernel exact_row_cell(plan, units, weights) = begin
    observations = eachrow(plan.rows)
    concentration::Vector{Float64} = plate(observations, Ref(units), Ref(weights)) do row, u, w
        sum(w[j] * get(u, row[j], 0.0) for j in eachindex(row); init = 0.0)
    end
    return concentration
end
@kernel one_graph_cell(plan, units, weights) = begin
    observations = _domain(plan)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum(w[j] * get(u, _lag(t, p, j), 0.0) for j in _doses(p); init = 0.0)
    end
    return concentration
end
# `get` outside a plate: every recipe op selects its tensorized companion on
# traced arguments, so a plain kernel reads a traced or host table the same way.
@kernel table_read(index, table::Vector{Float64}, hosted::Vector{Float64}) = begin
    value = get(table, index, -1.0) + get(hosted, index, -1.0) + get(table, 2, -1.0)
    return value
end
end

@testset "natural superposition cell: get and filtered sums lower with lazy branches" begin
    F = NaturalSuperpositionFixtures
    nobs = 40
    shifts = [0, 7, 19]
    units = collect(range(0.5, 2.0; length = nobs))
    weights = [3.0, 1.5, 0.25]
    expected = [sum(t > s ? w * units[t - s] : 0.0 for (s, w) in zip(shifts, weights))
                for t in 1:nobs]
    runits, rweights = Reactant.to_rarray(units), Reactant.to_rarray(weights)
    units2 = units .* 1.5 .+ 1.0
    runits2 = Reactant.to_rarray(units2)
    lattice = F.Lattice(shifts, nobs)
    exact = F.Exact([max(t - s, 0) for t in 1:nobs, s in shifts])
    cases = ((F.get_cell, lattice), (F.filter_cell, lattice), (F.exact_row_cell, exact),
             (F.one_graph_cell, lattice), (F.one_graph_cell, exact))
    @testset "$(nameof(typeof(plan))) $(k)" for (k, plan) in cases
        kernel = prepare(k)
        @test kernel(plan, units, weights) ≈ expected
        compiled = Reactant.@compile sync = true kernel(plan, runits, rweights)
        @test Array(compiled(plan, runits, rweights)) ≈ expected
        @test Array(compiled(plan, runits2, rweights)) ≈ kernel(plan, units2, weights)
        # The in-range test stays a lazy branch: an out-of-range lag (a dose not
        # yet given) is never read, not clamped and selected away.
        hlo = repr(Reactant.@code_hlo optimize = :none kernel(plan, runits, rweights))
        @test occursin("stablehlo.if", hlo)
    end

    @testset "doubling the lane count does not replicate the program" begin
        kernel = prepare(F.get_cell)
        runits_double = Reactant.to_rarray(vcat(units, units))
        hlo = repr(Reactant.@code_hlo optimize = :none kernel(lattice, runits, rweights))
        hlo2 = repr(Reactant.@code_hlo optimize = :none kernel(
            F.Lattice(shifts, 2 * nobs), runits_double, rweights))
        @test count("\n", hlo2) == count("\n", hlo)
    end

    @testset "get in a plain kernel: traced index, host table, concrete index" begin
        kernel = prepare(F.table_read)
        table = [1.0, 2.0, 4.0]
        hosted = [10.0, 20.0, 40.0]
        rtable = Reactant.to_rarray(table)
        for i in (0, 1, 3, 4)
            ri = Reactant.to_rarray(i; track_numbers = true)
            compiled = Reactant.@compile sync = true kernel(ri, rtable, hosted)
            @test Float64(compiled(ri, rtable, hosted)) == kernel(i, table, hosted)
        end
    end
end
