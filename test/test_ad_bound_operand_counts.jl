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

# Kernels whose bound data leave many hidden operands, evaluated once at file
# scope: `width` bound vectors plus `width` bound scalars, read by a scalar
# objective chained through binary `+`.
const _MANY_BOUND_OPERANDS = Dict(map((4, 40)) do width
    vectors = [Symbol(:v, i) for i in 1:width]
    scalars = [Symbol(:s, i) for i in 1:width]
    terms = [:(q[1] * $v[1] + q[2] * $v[2] + $s)
             for (v, s) in zip(vectors, scalars)]
    total = foldl((left, right) -> Expr(:call, :+, left, right), terms)
    spec = @eval @kernel $(Symbol(:many_bound_operands_, width))(
            q::Vector{Float64}, $(vectors...), $(scalars...)) = begin
        density::Float64 = $total
    end
    bound = NamedTuple{(vectors..., scalars...)}(
        ((([0.1i, -0.2i] for i in 1:width)...),
         ((0.01i for i in 1:width)...)))
    width => (spec, bound)
end)

@testset "the AD call's arity does not grow with the bound operands" begin
    # Every hidden operand used to cross as its own `Constant` context. From
    # 32 annotated arguments Enzyme's reverse `autodiff` re-splats past Julia's
    # 32-element limit, and from 32 contexts DifferentiationInterface maps
    # them with Base's unspecialized `map`, so each call dispatched
    # dynamically and boxed every operand: a BRM reader paid 13 KB per
    # gradient for one more bound operand, its plate's `eachindex` domain
    # (snag `rk-ref-indexed-p-5d0a952e`). Operands now cross as one named
    # tuple.
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    q = [0.3, -0.7]
    bytes = Dict{Int,Int}()
    for width in (4, 40)
        spec, bound = _MANY_BOUND_OPERANDS[width]
        kernel = prepare(spec; want = :density, bound)
        prepared = prepare_ad(kernel, backend, q; active = :q)
        hidden = ReactiveKernels._ad_hidden_operands(prepared)
        @test length(hidden) == 2width
        # One `Constant` named tuple, keyed by the bound ports.
        @test length(prepared.external_values) == 1
        pack = only(prepared.external_values)
        @test sort!(collect(keys(pack))) == sort!(collect(keys(bound)))
        @test all(pack[name] == bound[name] for name in keys(pack))
        expected = sum(q[1] * 0.1i - q[2] * 0.2i + 0.01i for i in 1:width)
        derivative = [sum(0.1i for i in 1:width), -sum(0.2i for i in 1:width)]
        gradient = zeros(2)
        value, returned = ad_value_and_gradient!(prepared, gradient, q)
        @test value ≈ expected
        @test returned === gradient
        @test gradient ≈ derivative
        @test ad_gradient(prepared, q) ≈ derivative
        pullback = prepare_ad_pullback(kernel, backend, 1.5, q; active = :q)
        @test ad_pullback(pullback, 1.5, q) ≈ 1.5 .* derivative
        call() = ad_value_and_gradient!(prepared, gradient, q)
        call(); call()
        bytes[width] = @allocated call()
    end
    # A wide boundary costs what a narrow one does.
    @test bytes[40] <= bytes[4]
end
