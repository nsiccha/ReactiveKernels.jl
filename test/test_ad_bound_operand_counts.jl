using ReactiveKernels, Test, DifferentiationInterface
import Enzyme

@kernel bound_operand_counts(scale::Float64, q::Vector{Float64}, data,
        a::Float64, b::Float64, c::Float64) = begin
    density::Float64 = scale * sum(q .* data.weights) + a * q[1] + b * q[end] + c
end

@testset "structured bound operand counts" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    # Four bound slots contribute one, four, or six array leaves, and each
    # numeric slot one more operand: a rewritten body holds no bound number
    # (snag `prepare-sampler-e69f1b38`).
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
        @test length(values) == leaves + 3
        @test count(value -> value isa AbstractArray, values) == leaves
        @test issetequal(filter(value -> value isa Number, values), (a, b, c))
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
    constants = (data, 0.5)
    callables = (nothing, nothing)
    body = (constants, callables, q) -> sum(q .* constants[1].weights) + constants[2]
    external, values = ReactiveKernels._externalize_bound_array_call(
        body, constants, callables; externalize_scalars = true)
    @test values == (data, 0.5)
    @test external([0.1, 0.2], values...) == body(constants, callables, [0.1, 0.2])
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

# Kernels with many inactive HAVE ports, evaluated once at file scope: `width`
# vectors and `width` scalars passed at the call, a density that is nonlinear in
# `q`, and a WANT port that reads the scalars alone.
const _MANY_INACTIVE_PORTS = Dict(map((4, 40)) do width
    vectors = [Symbol(:a, i) for i in 1:width]
    scalars = [Symbol(:s, i) for i in 1:width]
    terms = [:($s * exp(q[1] * $v[1]) + q[2] * q[2] * $v[2])
             for (v, s) in zip(vectors, scalars)]
    total = foldl((left, right) -> Expr(:call, :+, left, right), terms)
    scales = foldl((left, right) -> Expr(:call, :+, left, right), scalars)
    spec = @eval @kernel $(Symbol(:many_inactive_ports_, width))(
            q::Vector{Float64}, $(vectors...), $(scalars...)) = begin
        density::Float64 = $total
        scale_sum::Float64 = $scales
    end
    args = ((([0.1i, -0.2i] for i in 1:width)...), ((0.01i for i in 1:width)...))
    width => (spec, args)
end)

@testset "many inactive HAVE ports cross the backend packed" begin
    # Each inactive HAVE port crossed as its own `Constant` context, beside the
    # packed hidden operands above. An RKPPL cell that reads 33 shared values
    # reached DifferentiationInterface's and Enzyme's 32-element limits, and
    # inside a nested derivative their heap argument tuple stored the constant
    # operands beside the active point: Enzyme refused both Hessian-vector
    # compositions with `EnzymeRuntimeActivityError` (snag
    # `varyingsource3-b-82246291`). Past `_AD_UNPACKED_OPERANDS` the plain
    # values cross as one `Constant` tuple; below it the boundary is unchanged.
    reverse = AutoEnzyme(; mode = Enzyme.Reverse)
    forward = AutoEnzyme(; mode = Enzyme.Forward)
    q, v = [0.3, -0.7], [0.6, -1.1]
    for width in (4, 40)
        spec, args = _MANY_INACTIVE_PORTS[width]
        a = [0.1i for i in 1:width]
        b = [-0.2i for i in 1:width]
        s = [0.01i for i in 1:width]
        e = exp.(q[1] .* a)
        density = sum(s .* e) + q[2]^2 * sum(b)
        gradient = [sum(a .* s .* e), 2q[2] * sum(b)]
        hessian = [sum(a .^ 2 .* s .* e) 0.0; 0.0 2sum(b)]
        third = [v[1]^2 * sum(a .^ 3 .* s .* e), 0.0]
        packed = 2width > ReactiveKernels._AD_UNPACKED_OPERANDS

        kernel = prepare(spec; want = :density)
        prepared = prepare_ad(kernel, reverse, q, args...; active = :q)
        _, contexts = ReactiveKernels._ad_prepared_arguments(
            prepared, (q, args...), (;))
        @test length(contexts) == (packed ? 1 : 2width)
        @test (prepared.call isa ReactiveKernels._ADPackedOperandsCall) == packed
        value, g = ad_value_and_gradient(prepared, q, args...)
        @test value ≈ density
        @test g ≈ gradient
        pullback = prepare_ad_pullback(kernel, reverse, 1.5, q, args...; active = :q)
        @test ad_pullback(pullback, 1.5, q, args...) ≈ 1.5 .* gradient
        for second in (SecondOrder(reverse, forward), SecondOrder(forward, reverse))
            hvp = prepare_ad_hvp(kernel, second, (v,), q, args...; active = :q)
            hvp_gradient, (hv,) = ad_gradient_and_hvp(hvp, (v,), q, args...)
            @test hvp_gradient ≈ gradient
            @test hv ≈ hessian * v
        end
        thirds = prepare_ad_third(kernel,
            SecondOrder(SecondOrder(reverse, forward), forward), (v,), q, args...;
            active = :q)
        third_gradient, (third_hv,), (d3,) =
            ad_gradient_hvp_and_third(thirds, (v,), q, args...)
        @test third_gradient ≈ gradient
        @test third_hv ≈ hessian * v
        @test d3 ≈ third

        # A constant WANT port's box, a seeded pullback's and retained values'
        # boxes ride after the pack.
        both = prepare(spec; want = (:density, :scale_sum))
        seeded = prepare_ad_pullback(both, reverse, (1.5, 2.0), q, args...;
                                     active = :q)
        values, cotangent = ad_value_and_pullback(seeded, (1.5, 2.0), q, args...)
        @test values[1] ≈ density
        @test values[2] ≈ sum(s)
        @test cotangent ≈ 1.5 .* gradient
        retaining = prepare_ad(both, reverse, q, args...; active = :q,
                               retain = (:scale_sum,))
        retained_value, retained_gradient, retained =
            ad_value_gradient_and_retained(retaining, q, args...)
        @test retained_value ≈ density
        @test retained_gradient ≈ gradient
        @test retained.scale_sum ≈ sum(s)
    end
end
