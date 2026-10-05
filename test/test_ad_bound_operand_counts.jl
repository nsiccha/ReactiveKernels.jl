using ReactiveKernels, Test, DifferentiationInterface
import Enzyme

@kernel bound_operand_counts(scale::Float64, q::Vector{Float64}, data,
        a::Float64, b::Float64, c::Float64) = begin
    density::Float64 = scale * sum(q .* data.weights) + a * q[1] + b * q[end] + c
end

@testset "structured bound operand counts" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    # Four bound slots contribute one, four, or six array leaves. Numeric
    # slots become literals in a rewritten body, rather than hidden operands.
    for n in (3, 9), leaves in (1, 4, 6), record in (:tuple, :namedtuple)
        weights = collect(range(0.4, 1.2; length = n))
        extra = ntuple(i -> fill(Float64(i), n), leaves - 1)
        names = ntuple(i -> Symbol(:extra_, i), leaves - 1)
        nested = record === :tuple ? extra : NamedTuple{names}(extra)
        data = (; weights, nested)
        snapshot = deepcopy(data)
        a, b, c = 0.7, -0.3, 1.1
        kernel = prepare(bound_operand_counts; want = :density,
            bound = (; data, a, b, c))
        ast = deepcopy(code_expr(kernel))
        external, values = ReactiveKernels._externalize_bound_arrays(
            kernel; externalize_scalars = true)
        @test length(values) == leaves
        @test Tuple(input.name for input in inputs(kernel)) == (:scale, :q)

        point = collect(range(-0.2, 0.5; length = n))
        prepared = prepare_ad(kernel, backend, 0.8, point; active = :q)
        pullback = prepare_ad_pullback(kernel, backend, 1.5, 0.8, point;
            active = :q)
        joint = prepare_ad(kernel, backend, 0.8, point; active = (:q, :scale))
        for (scale, q) in ((0.8, point), (-0.4, point .+ 0.2))
            saved_q = copy(q)
            expected = scale * sum(q .* weights) + a * q[1] + b * q[end] + c
            derivative = scale .* weights
            derivative[1] += a
            derivative[end] += b
            @test external(scale, q, values...) ≈ expected
            @test kernel(scale, q) ≈ expected
            gradient = similar(q)
            value, returned = ad_value_and_gradient!(prepared, gradient, scale, q)
            @test value ≈ expected
            @test returned === gradient
            @test gradient ≈ derivative
            @test ad_pullback(pullback, 1.5, scale, q) ≈ 1.5 .* derivative
            joint_value, (q_gradient, scale_gradient) =
                ad_value_and_gradient(joint, scale, q)
            @test joint_value ≈ expected
            @test q_gradient ≈ derivative
            @test scale_gradient ≈ sum(q .* weights)
            @test q == saved_q
            @test data == snapshot
        end
        @test code_expr(kernel) == ast
    end

    # An opaque body cannot rewrite literal slot loads. Its fallback boundary
    # still carries one operand per replaced slot, including numeric slots.
    data = (; weights = [2.0, 3.0])
    ops = (ReactiveKernels._BoundConstant(data),
           ReactiveKernels._BoundConstant(0.5))
    body = (ops, q) -> sum(q .* ops[1]().weights) + ops[2]()
    external, values = ReactiveKernels._externalize_bound_array_call(
        body, ops; externalize_scalars = true)
    @test values == (data, 0.5)
    @test external([0.1, 0.2], values...) == body(ops, [0.1, 0.2])
end
