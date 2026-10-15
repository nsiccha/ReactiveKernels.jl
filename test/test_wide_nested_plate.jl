module WideNestedPlateTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test
include("fixtures/wide_nested_plates.jl")
include("type_stability_audit.jl")

@testset "Nested allocation type queries match the runtime allocation" begin
    RK = ReactiveKernels
    # Include scalar and zero-dimensional operands before the first axis,
    # tuples and views as markers, higher rank, and interleaved atomic arrays.
    # An atomic operand contributes neither the allocation marker nor its axes.
    vector = [1.0, 2.0, 3.0]
    cases = (((), (vector,)),
             ((), (2.0, fill(1.0), vector)),
             ((), ((1.0, 2.0, 3.0), vector)),
             ((), (view(vector, :),)),
             ((1, 3), (ones(7, 2), vector, ones(5), ones(1, 4))),
             ((), (ones(3, 1), ones(1, 4))))
    for (atomic, args) in cases
        batched = [a for (i, a) in enumerate(args) if !(i in atomic)]
        marker = first(filter(RK._authored_plate_is_axis, batched))
        axes = Base.Broadcast.combine_axes(batched...)
        expected = typeof(RK._plate_similar_output(marker, Float64, axes))
        @test RK._plate_array_type(Float64, Val(atomic), map(typeof, args)...) === expected
    end
    @test RK._plate_array_type(Float64, Val(())) === Any
    @test RK._plate_array_type(Float64, Val(()), Float64) === Any

    # Type-only calls in a generated body must fold, including a wide prefix
    # of scalars and an array excluded by the caller's explicit Ref semantics.
    for width in (31, 32, 33, 40), leading_scalars in (false, true)
        types = fill(leading_scalars ? Float64 : Vector{Float64}, width - 1)
        push!(types, Matrix{Float64})
        pushfirst!(types, Vector{Float64})
        query = Core.eval(@__MODULE__, :(() -> ReactiveKernels._plate_array_type(
            Float64, Val((1,)), $(types...))))
        @test Base.invokelatest(query) === Matrix{Float64}
        @test only(Base.code_typed(query, Tuple{})).second === Type{Matrix{Float64}}
    end
end

@testset "Wide nested plate inference and executed Reverse" begin
    F = WideNestedPlates
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    for width in (31, 32, 33, 40)
        spec = F.kernel(width)
        kernel = prepare(spec)
        rows = prepare(spec; want=:rows)
        ast = deepcopy(code_expr(kernel))
        @test F.count_head(ast, :for) == 2
        weights = [0.2, 0.7]
        # Both empty levels, ragged inner domains and changed lengths reuse
        # the authored loops. Ref(weights) must remain a whole shared array,
        # even when its length differs from every batched operand's length.
        for lengths in (Int[], [0], [3, 0, 1], [17, 5, 0])
            groups = F.data(width, lengths)
            original = deepcopy((groups, weights))
            bound = prepare(spec; bound=(; groups, weights))
            @test F.count_head(code_expr(bound), :for) <= 2
            for (reader, args) in ((kernel, (groups, 0.3, weights)),
                                   (bound, (0.3,)))
                ad = prepare_ad(reader, backend, args...; active=:scale)
                for scale in (0.3, -0.8)
                    callargs = reader === kernel ? (groups, scale, weights) : (scale,)
                    expected, derivative = F.reference(groups, scale, weights)
                    @test reader(callargs...) ≈ expected
                    value, gradient = ad_value_and_gradient(ad, callargs...)
                    @test value ≈ expected
                    @test gradient ≈ derivative
                end
            end
            @test eltype(rows(groups, 0.3, weights)) === Float64
            @test (groups, weights) == original
            @test code_expr(kernel) == ast
        end

        groups = F.data(width, [3, 0, 1])
        args = (groups, 0.3, weights)
        findings = TypeStabilityAudit.findings(kernel, map(Core.Typeof, args))
        @test isempty(findings)
        ad = prepare_ad(kernel, backend, args...; active=:scale)
        @test isempty(TypeStabilityAudit.ad_call_findings(ad, args...))
    end
end

@testset "Captured shared values and nested broadcast ranks" begin
    F = WideNestedPlates
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    width = 40
    groups = [[isodd(k) ? reshape(sin.(1.0:n), n, 1) :
                          reshape(cos.(1.0:m), 1, m) for k in 1:(width - 2)]
              for (n, m) in ((3, 4), (0, 4), (2, 1))]
    weights = [0.2, 0.7]
    original = deepcopy((groups, weights))
    for explicit_refs in (false, true)
        spec = F.kernel(width; explicit_refs)
        for bound in (false, true)
            reader = bound ? prepare(spec; bound=(; groups, weights)) : prepare(spec)
            args = bound ? (0.3,) : (groups, 0.3, weights)
            ad = prepare_ad(reader, backend, args...; active=:scale)
            @test collect(ad_value_and_gradient(ad, args...)) ≈
                  collect(F.reference(groups, 0.3, weights))
            @test isempty(TypeStabilityAudit.findings(reader, map(Core.Typeof, args)))
        end
    end
    @test (groups, weights) == original
end

end
