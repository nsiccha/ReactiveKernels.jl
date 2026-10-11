using ReactiveKernels
using MutatingFunctions
using Test
using ReactiveKernels: plate, scan

# Cache ownership and result typing of `prepare_nonallocating`: a cache slot is
# a write destination only for storage the kernel owns, a declared plate output
# fixes the buffer's element type, and exemplar arguments type the program from
# the runtime argument types.

function ownership_allocations(k, args)
    k(args...)
    k(args...)
    @allocated k(args...)
end

# A lazy observation domain whose representation depends on the plan type: a
# range for one, row slices of a stored index matrix for the other.
struct RangeIndexPlan
    n::Int
    shifts::Vector{Int}
end
struct RowIndexPlan
    rows::Matrix{Int}
end
ownership_domain(p::RangeIndexPlan) = 1:p.n
ownership_domain(p::RowIndexPlan) = eachrow(p.rows)
ownership_lag(t, p::RangeIndexPlan, j) = t - p.shifts[j]
ownership_lag(row, ::RowIndexPlan, j) = row[j]

const LAZY_DOMAIN = @kernel lazy_domain(plan, w::Vector{Float64},
                                        u::Vector{Float64}) = begin
    observations = ownership_domain(plan)
    y::Vector{Float64} = plate(observations) do t
        sum(w[j] * get(u, ownership_lag(t, plan, j), 0.0) for j in eachindex(w);
            init = 0.0)
    end
    odd::Vector{Float64} = y[1:2:end]
    even = y[2:2:end]
    trajectory::AbstractVector{Float64} = scan(even; init = 0.0,
                                               include_init = true) do c, v
        next = 0.5 * c + v
        (next, next)
    end
    return odd, trajectory
end

const FIELD_SCALE = @kernel field_scale(sched, factor::Float64) = begin
    xs = sched.xs
    ys::Vector{Float64} = xs .* factor
    total::Float64 = sum(ys)
end

const TYPED_FIELD = @kernel typed_field(
        sched::NamedTuple{(:xs,),Tuple{Vector{Float64}}}) = begin
    xs = sched.xs
    total::Float64 = sum(xs)
end

const ROW_COUNT = @kernel row_count(plan) = begin
    rows = ownership_domain(plan)
    n = length(rows)
end

@testset "nonallocating cache ownership and result typing" begin
    @testset "a range-valued step is recomputed, not written into" begin
        k = prepare_nonallocating(LAZY_DOMAIN; want = :y)
        reference = prepare(LAZY_DOMAIN; want = :y)
        first_args = (RangeIndexPlan(9, [0, 2, 4]), [1.0, 0.5, 0.25],
                      collect(1.0:9.0))
        second_args = (RangeIndexPlan(11, [0, 3]), [2.0, 1.0], collect(1.0:11.0))
        @test k(first_args...) == reference(first_args...)
        @test k(first_args...) == reference(first_args...)
        @test k(second_args...) == reference(second_args...)
        rows = (RowIndexPlan([1 3; 2 4; 5 0]), [1.0, 0.25], collect(1.0:6.0))
        @test k(rows...) == reference(rows...)
        @test k(first_args...) == reference(first_args...)
    end

    @testset "steps never write into caller-owned inputs" begin
        for (label, k) in (("untyped", prepare_nonallocating(FIELD_SCALE;
                                                              want = :total)),
                           ("exemplar", prepare_nonallocating(FIELD_SCALE,
                                (; xs = [0.0], tag = 0), 1.0; want = :total)))
            @testset "$label field read" begin
                first_input = (; xs = [1.0, 2.0, 3.0], tag = 1)
                second_input = (; xs = [10.0, 20.0, 30.0], tag = 2)
                @test k(first_input, 2.0) == 12.0
                @test k(second_input, 2.0) == 120.0
                @test first_input.xs == [1.0, 2.0, 3.0]
                @test second_input.xs == [10.0, 20.0, 30.0]
            end
        end
        typed = prepare_nonallocating(TYPED_FIELD)
        first_input = (; xs = [1.0, 2.0, 3.0])
        second_input = (; xs = [10.0, 20.0, 30.0])
        @test typed(first_input) == 6.0
        @test typed(second_input) == 60.0
        @test first_input.xs == [1.0, 2.0, 3.0]

        counted = prepare_nonallocating(ROW_COUNT)
        first_plan = RowIndexPlan([1 2; 3 4])
        second_plan = RowIndexPlan([5 6; 7 8; 9 10])
        @test counted(first_plan) == 2
        @test counted(second_plan) == 3
        @test first_plan.rows == [1 2; 3 4]
        @test counted(RangeIndexPlan(4, Int[])) == 4
    end

    @testset "reactive in-place slots never write into caller inputs" begin
        g = Graph()
        source = value!(g, :source, Any)
        field = value!(g, :field, Vector{Float64})
        add!(g, source => field, s -> s.xs)
        program = prepare_reactive_nonallocating(g; have = (source,),
                                                 want = (field,))
        first_input = (; xs = [1.0, 2.0])
        second_input = (; xs = [10.0, 20.0])
        state = program(first_input)
        handle = statevalue(state, field)
        @test get!(state, handle) == [1.0, 2.0]
        set!(state, statevalue(state, source), second_input)
        @test get!(state, handle) == [10.0, 20.0]
        @test first_input.xs == [1.0, 2.0]
        @test !occursin("__cache_apply__((__slots__",
                        string(code_expr(program, field)))
    end

    @testset "registered destinations still reuse owned buffers" begin
        g = Graph()
        x = value!(g, :x, Vector{Float64})
        copied = value!(g, :copied, Vector{Float64})
        add!(g, x => copied, copy)
        k = prepare_nonallocating(plan(g; have = (x,), want = (copied,)))
        input = [1.0, 2.0]
        first_result = k(input)
        @test first_result !== input
        @test k([3.0, 4.0]) === first_result
        @test first_result == [3.0, 4.0]
        @test input == [1.0, 2.0]
    end

    @testset "a declared plate output fixes the buffer element type" begin
        args = (RangeIndexPlan(9, [0, 2, 4]), [1.0, 0.5, 0.25], collect(1.0:9.0))
        for want in (:y, :odd, (:odd, :trajectory))
            k = prepare_nonallocating(LAZY_DOMAIN; want = want)
            reference = prepare(LAZY_DOMAIN; want = want)(args...)
            result = k(args...)
            @test typeof(result) == typeof(reference)
            @test result == reference
        end
        k = prepare_nonallocating(LAZY_DOMAIN; want = :y)
        @test k(args...) isa Vector{Float64}
        @test k.caches[2] isa Base.RefValue{Vector{Float64}}
    end

    @testset "exemplar arguments type the program" begin
        args = ((; xs = [1.0, 2.0, 3.0], tag = 1), 2.0)
        exemplar = prepare_nonallocating(FIELD_SCALE, args...; want = :ys)
        untyped = prepare_nonallocating(FIELD_SCALE; want = :ys)
        signature = code_expr(exemplar).args[1].args
        @test signature[4] == Expr(:(::), :sched, typeof(args[1]))
        @test signature[5] == Expr(:(::), :factor, Float64)
        @test exemplar(args...) == prepare(FIELD_SCALE; want = :ys)(args...)
        @test exemplar((; xs = [4.0, 5.0], tag = 3), 0.5) == [2.0, 2.5]
        @test ownership_allocations(exemplar, args) == 0
        println("NONALLOCATING_ALLOC_BYTES\texemplar_field_scale\t",
                ownership_allocations(exemplar, args), "\tuntyped\t",
                ownership_allocations(untyped, args))

        @test_throws ArgumentError exemplar((; xs = [1.0f0], tag = 1), 2.0)
        @test_throws ArgumentError prepare_nonallocating(FIELD_SCALE,
                                                         args[1]; want = :ys)
        @test_throws ArgumentError prepare_nonallocating(TYPED_FIELD,
                                                         (; xs = [1]))

        plan_args = (RangeIndexPlan(9, [0, 2, 4]), [1.0, 0.5, 0.25],
                     collect(1.0:9.0))
        k = prepare_nonallocating(LAZY_DOMAIN, plan_args...;
                                  want = (:odd, :trajectory))
        reference = prepare(LAZY_DOMAIN; want = (:odd, :trajectory))
        @test k(plan_args...) == reference(plan_args...)
        other = (RangeIndexPlan(11, [0, 3]), [2.0, 1.0], collect(1.0:11.0))
        @test k(other...) == reference(other...)
        @test_throws ArgumentError k(RowIndexPlan([1 2; 3 4]), [1.0, 0.5],
                                     collect(1.0:4.0))

        prepared = prepare(FIELD_SCALE; have = (:sched, :factor), want = :ys)
        from_prepared = prepare_nonallocating(prepared, args...)
        @test from_prepared(args...) == prepared(args...)
        @test ownership_allocations(from_prepared, args) == 0
    end
end
