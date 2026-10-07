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

    @testset "bound data outside a live-only plate" begin
        data = [3.0, 4.0]
        p = plan(C.unbound_plate)
        original = only(plates(p))
        @test any(r -> isempty(r.inputs), plate_body(original).recipes)
        plain = prepare(p)
        bound = prepare(C.unbound_plate; bound = (; data))
        @test isempty(caches(bound.plan))
        @test plate_body(only(plates(bound.plan))).recipes ==
              plate_body(original).recipes
        for q in (Float64[], [1.0], [1.0, 2.0, 3.0])
            @test bound(q) == plain(q, data) == sum(q) + 2length(q) + sum(data)
        end
    end

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
        # A dense primitive array per cell is cached as an array of arrays.
        mutable_result = prepare(C.mutable_result; bound = (; data))
        @test only(caches(mutable_result.plan)).op.value == [fill(d, 2) for d in data]
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

    @testset "array-valued cell values: ragged index lists" begin
        kinds = [[1, 2, 1, 1, 3], [2, 1, 1, 3, 1, 1], [3, 3]]
        read_idx = [[1, 3], [2, 4], Int[]]
        subjects = 1:3
        for spec in (C.ragged_reads, C.composed_reads)
            plain = prepare(spec)
            C.calls[] = 0
            bound = prepare(spec; bound = (; kinds_by_subject = kinds, read_idx, subjects))
            @test C.calls[] == length(subjects)
            cache = only(caches(bound.plan))
            @test cache.op.value isa Vector{Vector{Int}}
            @test cache.op.value == [[1, 4], [3, 6], Int[]]
            @test !occursin("counted_findall", string(code_expr(bound)))
            lives = (collect(1.0:6.0), [0.5, -1.0, 2.0, 3.0, 0.0, 7.0])
            results = [bound(live) for live in lives]
            @test C.calls[] == length(subjects)
            @test results == [plain(live, kinds, read_idx, subjects) for live in lives]
            @test first(results) == 14.0
            rebound = prepare(spec; bound = (; kinds_by_subject = [[1, 1]],
                                             read_idx = [[2]], subjects = 1:1))
            @test only(caches(rebound.plan)).op.value == [[2]]
            @test rebound(first(lives)) == 2.0
            @test bound(first(lives)) == 14.0
            external, values = RK._externalize_bound_arrays(bound)
            @test external(first(lives), values...) == 14.0
        end
        # A condition on the cached array partitions the plate by lane.
        guard_kinds = [[1, 2, 1], [3, 3], [1]]
        C.calls[] = 0
        guarded = prepare(C.guarded_reads;
                          bound = (; kinds_by_subject = guard_kinds, subjects))
        @test C.calls[] == length(subjects)
        @test count(r -> r.op isa RK._AuthoredPlateOp, guarded.plan.recipes) == 2
        for q in ([0.5, 0.25], [-1.0, 2.0])
            @test guarded(q) == prepare(C.guarded_reads)(q, guard_kinds, subjects) ==
                  sum(q) * 5
        end
        @test C.calls[] == length(subjects) + 2length(subjects)
    end

    @testset "nested plates, scans and prepared kernels stay in the cell" begin
        kinds = [[1, 2, 1, 1, 2, 1], [2, 2, 1, 2, 1], Int[], [1, 1]]
        steps = [[0.5, 1.0, 0.25, 2.0, 1.5, 0.75], [0.0, 1.5, 0.5, 1.0, 2.0],
                 Float64[], [0.5, 0.5]]
        read_idx = [[3, 1, 4], [2, 1, 2], Int[], Int[]]
        subjects = 1:4
        selection = [[4, 1, 6], [5, 3, 5], Int[], Int[]]
        cache_names(p) = sort([only(r.outputs).name for r in caches(p)])
        structure(k) = [(e.kind, e.depth) for e in recipe_inventory(k) if e.kind !== :ordinary]
        cases = (
            (C.scan_reads, (; kinds_by_subject = kinds, steps_by_subject = steps,
                            read_idx, subjects),
             [:bound_plate_kinds, :bound_plate_observation_operations, :bound_plate_steps],
             [(:plate, 0), (:scan, 1)]),
            (C.nested_reads, (; kinds_by_subject = kinds, read_idx, subjects),
             [:bound_plate_observation_operations], [(:plate, 0), (:plate, 1)]),
            (C.embedded_reads, (; kinds_by_subject = kinds, read_idx, subjects),
             [:bound_plate_kinds, :bound_plate_observation_operations], [(:plate, 0)]))
        for (spec, data, names, nesting) in cases
            plain = prepare(spec)
            C.calls[] = 0
            bound = prepare(spec; bound = data)
            @test C.calls[] == length(subjects)
            @test cache_names(bound.plan) == names
            @test only(filter(r -> only(r.outputs).name === :bound_plate_observation_operations,
                              caches(bound.plan))).op.value == selection
            @test structure(bound) == nesting
            body = plate_body(only(plates(bound.plan)))
            @test any(r -> r.op isa Union{RK._AuthoredPlateOp,RK._AuthoredScanOp} ||
                           RK._embedded_kernel(r.op) !== nothing, body.recipes)
            @test !occursin("counted_findall", string(code_expr(bound)))
            for live in ([0.3, 0.7, 1.1, 0.2, -0.4, 0.9], [1.2, -0.4, 0.0, 2.0, 0.5, 0.1])
                expected = plain(live, data...)
                C.calls[] = 0
                @test bound(live) == expected
                @test C.calls[] == 0
            end
        end
    end

    @testset "data-only cell results" begin
        limits = [0.5, 1.0, 1.0, 0.5, 1.0, 1.0]
        rows = [[1, 2, 3], [4], [5, 6]]
        subjects = 1:3
        lives = ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1.2, -0.4, 0.0, 2.0, 0.5, 0.1])
        plain = prepare(C.data_only_result)
        expected = [plain(live, limits, rows, subjects) for live in lives]
        C.calls[] = 0
        bound = prepare(C.data_only_result; bound = (; limits, rows, subjects))
        @test C.calls[] == length(subjects)
        # Bound inputs alone fix the domain, so the whole plate result is one
        # hoisted value: no plate and no per-cell cache remain.
        @test isempty(plates(bound.plan))
        @test isempty(caches(bound.plan))
        hoisted = only(r for r in bound.plan.recipes
                       if r.op isa RK._BoundConstant &&
                          only(r.outputs).name === :weights)
        @test hoisted.op.value isa Vector{Vector{Float64}}
        @test hoisted.op.value == [limits[r] for r in rows]
        @test [bound(live) for live in lives] == expected
        @test C.calls[] == length(subjects)
        @test hoisted.op.value == [limits[r] for r in rows]
        # More subjects change only the hoisted value, not the emitted residual.
        more = prepare(C.data_only_result; bound = (; limits = repeat(limits, 2),
            rows = vcat(rows, [r .+ 6 for r in rows]), subjects = 1:6))
        @test isempty(plates(more.plan))
        @test [typeof(r.op) for r in more.plan.recipes] ==
              [typeof(r.op) for r in bound.plan.recipes]
        @test more(repeat(first(lives), 2)) ==
              plain(repeat(first(lives), 2), repeat(limits, 2),
                    vcat(rows, [r .+ 6 for r in rows]), 1:6)

        # A live non-atomic input keeps the plate, its runtime domain check
        # and a result allocated per evaluation.
        axis_plain = prepare(C.data_only_result_live_axis)
        axis = prepare(C.data_only_result_live_axis; bound = (; limits, rows))
        @test length(plates(axis.plan)) == 1
        @test isempty(caches(axis.plan))
        x = [1.0, 2.0, 3.0]
        axis_expected = axis_plain(x, limits, rows)
        C.calls[] = 0
        @test axis(x) == axis_expected
        @test C.calls[] == length(rows)
        @test_throws DimensionMismatch axis([1.0, 2.0])
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

    xs = [[1.0, 2.0], [4.0], Float64[]]
    plain = prepare_ad(C.array_weights, backend, q, xs; active=:q, want=:total)
    bound = prepare_ad(C.array_weights, backend, q; active=:q, want=:total,
                       bound=(; xs))
    cached = filter(r -> r.op isa ReactiveKernels._BoundConstant &&
        startswith(String(only(r.outputs).name), "bound_plate_"), bound.kernel.plan.recipes)
    @test only(cached).op.value == [log.(x) for x in xs]
    @test ad_value_and_gradient(bound, q) == ad_value_and_gradient(plain, q, xs)
    @test ad_gradient(bound, q) ≈ fill(sum(sum(log, x; init = 0.0) for x in xs), length(q))

    # A data-only cell result must not reach native reverse mode as cached
    # arrays stored into the plate's output (snag native-reverse-r-79fb001d).
    limits = [0.5, 1.0, 1.0, 0.5, 1.0, 1.0]
    rows = [[1, 2, 3], [4], [5, 6]]
    subjects = 1:3
    live = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    plain = prepare_ad(C.data_only_result, backend, live, limits, rows, subjects;
                       active=:live, want=:total)
    bound = prepare_ad(C.data_only_result, backend, live; active=:live,
                       want=:total, bound=(; limits, rows, subjects))
    @test ad_value_and_gradient(bound, live) ==
          ad_value_and_gradient(plain, live, limits, rows, subjects)
    @test ad_gradient(bound, live) == reduce(vcat, [limits[r] for r in rows])
    x = [1.0, 2.0, 3.0]
    plain = prepare_ad(C.data_only_result_live_axis, backend, x, limits, rows;
                       active=:live, want=:total)
    bound = prepare_ad(C.data_only_result_live_axis, backend, x; active=:live,
                       want=:total, bound=(; limits, rows))
    @test ad_value_and_gradient(bound, x) ==
          ad_value_and_gradient(plain, x, limits, rows)
    @test ad_gradient(bound, x) == fill(sum(limits), length(x))

    # A scan beside the cached index chain differentiates in the residual cell.
    data = (; kinds_by_subject = [[1, 2, 1, 1, 2, 1], [2, 2, 1, 2, 1], Int[]],
            steps_by_subject = [[0.5, 1.0, 0.25, 2.0, 1.5, 0.75], [0.0, 1.5, 0.5, 1.0, 2.0],
                                Float64[]],
            read_idx = [[3, 1, 4], [2, 1, 2], Int[]], subjects = 1:3)
    rates = [0.3, 0.7, 1.1]
    plain = prepare_ad(C.scan_reads, backend, rates, data...; active=:rates, want=:total)
    bound = prepare_ad(C.scan_reads, backend, rates; active=:rates, want=:total,
                       bound=data)
    @test length(filter(r -> r.op isa ReactiveKernels._BoundConstant &&
        startswith(String(only(r.outputs).name), "bound_plate_"),
        bound.kernel.plan.recipes)) == 3
    for point in (rates, [1.2, -0.4, 0.5])
        value, gradient = ad_value_and_gradient(bound, point)
        expected_value, expected_gradient = ad_value_and_gradient(plain, point, data...)
        @test value == expected_value
        @test gradient ≈ expected_gradient
        primal = prepare(C.scan_reads; bound=data)
        h = 1e-6
        central = [(primal(point .+ h .* (1:3 .== i)) -
                    primal(point .- h .* (1:3 .== i))) / 2h for i in 1:3]
        @test isapprox(gradient, central; rtol=1e-6, atol=1e-8)
        C.calls[] = 0
        ad_value_and_gradient(bound, point)
        @test C.calls[] == 0
    end
end
