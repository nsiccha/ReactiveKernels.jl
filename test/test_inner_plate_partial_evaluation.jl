using ReactiveKernels
using Test

isdefined(@__MODULE__, :InnerPlatePartialEvaluation) ||
    include("fixtures/inner_plate_partial_evaluation.jl")

@testset "Inner authored-plate partial evaluation" begin
    C = InnerPlatePartialEvaluation
    RK = ReactiveKernels
    plates(p) = filter(r -> r.op isa RK._AuthoredPlateOp, p.recipes)
    caches(p) = filter(r -> r.op isa RK._BoundConstant &&
        startswith(String(only(r.outputs).name), "bound_plate_"), p.recipes)

    @testset "preparation-only prefix and original graph" begin
        p = plan(C.counted)
        graph_values, graph_recipes = copy(p.graph.values), copy(p.graph.recipes)
        inner = plate_body(only(plates(p)))
        original_recipes = copy(inner.recipes)
        data = [1.0, 2.0, 4.0]
        plain = prepare(p)
        expected = plain(2.0, data)
        C.calls[] = 0
        bound = prepare(p; bound = p.have[2] => data)
        @test C.calls[] == length(data)
        @test bound(2.0) == expected
        @test bound(3.0) == 3sum(log, data)
        @test C.calls[] == length(data)
        @test inputs(bound) == (p.have[1],)
        @test outputs(bound) == Tuple(p.want)
        @test p.graph.values == graph_values
        @test p.graph.recipes == graph_recipes
        @test inner.recipes == original_recipes
        @test bound.plan.graph !== p.graph
        specialized = only(plates(bound.plan))
        @test length(plate_body(specialized).recipes) < length(inner.recipes)
        @test !occursin("counted_log", string(code_expr(bound)))
        @test length(caches(bound.plan)) == 1
        @test only(caches(bound.plan)).op.value == log.(data)
        @test eltype(only(caches(bound.plan)).op.value) === Float64
        @test prepare(p).plan === p

        rebound = prepare(p; bound = p.have[2] => [2.0, 8.0])
        @test C.calls[] == 5
        @test rebound(2.0) == 2sum(log, [2.0, 8.0])
        @test bound(2.0) == expected
        external, values = RK._externalize_bound_arrays(bound)
        @test external(2.0, values...) == expected
        @test length(values) == 2 # raw shape operand + cached scalar frontier
    end

    @testset "projected axes, singleton expansion and empty domains" begin
        data = reshape([1.0, 2.0, 4.0], 3, 1)
        shift = reshape([0.1, 0.3], 1, 2)
        q = fill(2.0, 3, 2)
        plain = prepare(C.projected)
        C.calls[] = 0
        bound = prepare(C.projected; bound = (; data))
        @test C.calls[] == 0 # untyped live inputs can make the domain empty
        @test isempty(caches(bound.plan))
        @test bound(q, shift) == sum((log.(data) .+ shift) .* q)
        @test bound(2.0, shift) == plain(2.0, data, shift)
        @test_throws DimensionMismatch bound(ones(2, 2), shift)
        @test_throws DimensionMismatch bound(q, ones(4, 1))
        C.calls[] = 0
        scalar = prepare(C.projected_scalar; bound = (; data, shift))
        @test C.calls[] == 3
        @test size(only(caches(scalar.plan)).op.value) == (3, 1)
        @test scalar(2.0) == prepare(C.projected_scalar)(2.0, data, shift)
        C.calls[] = 0
        both = prepare(C.projected_matrix; bound = (; data, shift))
        @test C.calls[] == 3 # log is not repeated across the second dimension
        @test size(only(caches(both.plan)).op.value) == (3, 2)
        @test both(q) == prepare(C.projected_matrix)(q, data, shift)
        @test_throws DimensionMismatch both(ones(2, 2))

        C.calls[] = 0
        empty = prepare(C.counted; bound = (; data = Float64[]))
        @test empty(2.0) == 0.0
        @test C.calls[] == 0
        @test prepare(C.counted; want = :pointwise,
                      bound = (; data = Float64[]))(2.0) == Float64[]
        singleton = prepare(C.projected; bound = (; data = [2.0]))
        @test singleton(Float64[], 1.0) == 0.0
        @test singleton(ones(4), 1.0) == 4(log(2.0) + 1.0)

        # Source-identical ON/OFF controls: neither unknown rank nor a known
        # rank may turn a skipped invalid-data cell into a preparation error.
        C.calls[] = 0
        for spec in (C.projected, C.vector_live)
            empty_args = spec === C.projected ? (Float64[], 1.0) : (Float64[],)
            invalid = prepare(spec; bound = (; data = [-1.0]))
            @test isempty(caches(invalid.plan))
            @test invalid(empty_args...) == 0.0
            @test prepare(spec)(Float64[], [-1.0], empty_args[2:end]...) == 0.0
            @test C.calls[] == 0
        end
        # A non-singleton bound vector does not constrain a new live matrix
        # dimension. In particular, 3×0 is compatible and visits no cells.
        for spec in (C.projected, C.projected_matrix)
            bad_data, shift = [-1.0, -2.0, -3.0], 0.0
            extra_axis = prepare(spec; bound = (; data = bad_data, shift))
            @test isempty(caches(extra_axis.plan))
            @test extra_axis(zeros(3, 0)) == 0.0
            @test prepare(spec)(zeros(3, 0), bad_data, shift) == 0.0
            @test C.calls[] == 0
        end
        vector = prepare(C.vector_live; bound = (; data = [1.0, 2.0, 4.0]))
        @test length(caches(vector.plan)) == 1
        @test vector(ones(3)) == sum(log, [1.0, 2.0, 4.0])
        @test_throws DimensionMismatch vector(Float64[])
    end

    @testset "numeric scalars, atomic data and source granularity" begin
        data = [1.0, 2.0, 4.0]
        bound = prepare(C.affine; bound = (; data, a = 2.0, b = 1.0))
        @test bound(3.0) == 3sum(2 .* data .+ 1)
        coefs = [0.1, 0.2]
        atomic = prepare(C.atomic; bound = (; data, coefficients = coefs))
        @test atomic(2.0) == prepare(C.atomic)(2.0, data, coefs)
        @test length(caches(atomic.plan)) == 1
        C.calls[] = 0
        inline = prepare(C.inline; bound = (; data))
        @test C.calls[] == 0
        @test isempty(caches(inline.plan))
        inline(2.0)
        @test C.calls[] == 3 # the mixed recipe is deliberately indivisible
        mutable_result = prepare(C.mutable_result; bound = (; data))
        @test isempty(caches(mutable_result.plan))
        @test mutable_result(2.0) == prepare(C.mutable_result)(2.0, data)
        tuple_result = prepare(C.tuple_result; bound = (; data))
        @test isempty(caches(tuple_result.plan))
        @test tuple_result(2.0) == prepare(C.tuple_result)(2.0, data)
        bool_data = [-2, 0, 3]
        boolean = prepare(C.boolean; bound = (; data = bool_data))
        @test eltype(only(caches(boolean.plan)).op.value) === Bool
        @test only(caches(boolean.plan)).op.value == Bool[false, true, true]
        @test boolean([2.0]) == prepare(C.boolean)([2.0], bool_data)
        @test_throws ArgumentError prepare(C.counted; bound = (; data=2.0))(3.0)
    end

    @testset "demanded intermediates and composed consumers" begin
        data, observations = [1.0, 2.0, 4.0], [0.1, 0.2, 0.3]
        for want in (:total, :means, :pointwise,
                     (:means, :pointwise, :total), (:total, :extra))
            plain = prepare(C.chain; want)
            bound = prepare(C.chain; want, bound = (; data, observations))
            @test bound(2.0) == plain(2.0, data, observations)
        end
        bound = prepare(C.chain; want = :total, bound = (; data, observations))
        @test !occursin("similar", string(code_expr(bound)))
        @test prepare(C.chain; have = (:means, :observations), want = :total)(
            [0.1, 0.2, 0.3], observations) == 0.0
    end
end

using DifferentiationInterface: AutoEnzyme
import Enzyme

@testset "Inner partial evaluation with plain reverse AD" begin
    C = InnerPlatePartialEvaluation
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    q, data = [0.3, -0.1], [1.0, 2.0, 4.0]
    plain = prepare_ad(C.pure, backend, q, data; active=:q, want=:total)
    bound = prepare_ad(C.pure, backend, q; active=:q, want=:total, bound=(; data))
    @test ad_value_and_gradient(bound, q) == ad_value_and_gradient(plain, q, data)
    @test ad_gradient(bound, q) ≈ fill(sum(log, data), length(q))
    @test ad_gradient(bound, 2q) == ad_gradient(bound, q)
end
