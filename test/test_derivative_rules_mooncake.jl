# The generated Mooncake adapter (ext/ReactiveKernelsMooncakeExt.jl) for rules
# built from one pure-math graph (src/derivative_rules.jl). Mooncake's own rule
# tester checks each hand-registered primitive against finite differences in
# both modes; an end-to-end gradient through a loss that calls the rules checks
# them in context. No derivative code is written here.
import Mooncake
using Mooncake.TestUtils: test_rule
using ReactiveKernels
using ReactiveKernels: derivative_cut
using StableRNGs: StableRNG
using Test

module DerivativeRuleMooncakeGraphs
using ReactiveKernels
@kernel two_input_graph(a::Float64, b::Float64) = begin
    s::Float64 = a + b
    e::Float64 = exp(a * b)
    y::Float64 = e + log(s)
    dy_da::Float64 = b * e + 1 / s
    dy_db::Float64 = a * e + 1 / s
    return y, dy_da, dy_db
end
@kernel matvec_graph(
        A::Matrix{Float64}, x::Vector{Float64},
        A_dot::Matrix{Float64}, x_dot::Vector{Float64},
        y_bar::Vector{Float64}) = begin
    y::Vector{Float64} = A * x
    y_dot::Vector{Float64} = A_dot * x + A * x_dot
    A_bar::Matrix{Float64} = y_bar * transpose(x)
    x_bar::Vector{Float64} = transpose(A) * y_bar
    return y, y_dot, A_bar, x_bar
end
# Reverse branch only; a scalar and an array input, an array result.
@kernel scale_graph(s::Float64, v::Vector{Float64}, y_bar::Vector{Float64}) = begin
    y::Vector{Float64} = s .* v
    s_bar::Float64 = sum(y_bar .* v)
    v_bar::Vector{Float64} = s .* y_bar
    return y, s_bar, v_bar
end
# A scalar result from array inputs, both branches.
@kernel wdot_graph(w::Vector{Float64}, x::Vector{Float64},
        w_dot::Vector{Float64}, x_dot::Vector{Float64}, y_bar::Float64) = begin
    y::Float64 = sum(w .* x)
    y_dot::Float64 = sum(w_dot .* x) + sum(w .* x_dot)
    w_bar::Vector{Float64} = y_bar .* x
    x_bar::Vector{Float64} = y_bar .* w
    return y, y_dot, w_bar, x_bar
end

const two_input = scalar_derivative_rule(
    two_input_graph; primal = :y, partials = (a = :dy_da, b = :dy_db),
    name = :two_input)
const matvec = derivative_rule(matvec_graph; primal = :y,
    directions = (A = :A_dot, x = :x_dot), tangent = :y_dot,
    covector = :y_bar, cotangents = (A = :A_bar, x = :x_bar), name = :matvec)
const scale = derivative_rule(scale_graph; primal = :y,
    covector = :y_bar, cotangents = (s = :s_bar, v = :v_bar), name = :scale)
const wdot = derivative_rule(wdot_graph; primal = :y,
    directions = (w = :w_dot, x = :x_dot), tangent = :y_dot,
    covector = :y_bar, cotangents = (w = :w_bar, x = :x_bar), name = :wdot)
end # module

const _DRM = DerivativeRuleMooncakeGraphs

@testset "generated Mooncake adapter: rules are primitives with no tangent" begin
    for rule in (_DRM.two_input, _DRM.matvec, _DRM.scale, _DRM.wdot)
        @test Mooncake.tangent_type(typeof(rule)) == Mooncake.NoTangent
    end
    world = Base.get_world_counter()
    @test Mooncake.is_primitive(Mooncake.DefaultCtx, Mooncake.ReverseMode,
        Tuple{typeof(_DRM.matvec),Matrix{Float64},Vector{Float64}}, world)
    @test Mooncake.is_primitive(Mooncake.DefaultCtx, Mooncake.ForwardMode,
        Tuple{typeof(_DRM.two_input),Float64,Float64}, world)
end

@testset "generated Mooncake adapter: Mooncake's rule tester" begin
    rng = StableRNG(20260923)
    A = [1.0 2.0; 3.0 4.0]; x = [0.5, -1.0]
    # Scalar rule, both modes.
    test_rule(rng, _DRM.two_input, 0.3, 1.7; is_primitive = true)
    # Vector rule with both branches: array inputs, array result.
    test_rule(rng, _DRM.matvec, A, x; is_primitive = true)
    # Scalar result from array inputs.
    test_rule(rng, _DRM.wdot, [0.2, -0.3, 0.4], [1.0, 2.0, -1.5]; is_primitive = true)
    # Reverse-only rule: mixed scalar/array inputs.
    test_rule(rng, _DRM.scale, 2.0, x; is_primitive = true,
        mode = Mooncake.ReverseMode)
end

@testset "generated Mooncake adapter: gradients through a loss" begin
    A = [1.0 2.0; 3.0 4.0]; x = [0.5, -1.0]
    loss(A, x, a) = sum(abs2, _DRM.matvec(A, x)) + _DRM.two_input(a, x[1]^2) +
        sum(_DRM.scale(a, x))
    cache = Mooncake.prepare_gradient_cache(loss, A, x, 0.3)
    value, (_, gA, gx, ga) = Mooncake.value_and_gradient!!(cache, loss, A, x, 0.3)
    @test value ≈ loss(A, x, 0.3)
    # Reference: the graphs' authored partials, combined by hand.
    y = A * x
    _, dy_da, dy_db = derivative_cut(_DRM.two_input, (true, true), 0.3, x[1]^2)
    @test gA ≈ 2 .* y * transpose(x)
    @test gx ≈ transpose(A) * (2 .* y) .+ [dy_db * 2 * x[1], 0.0] .+ 0.3
    @test ga ≈ dy_da + sum(x)
    # A missing branch is explicit in the mode that needs it.
    @test_throws ArgumentError Mooncake.frule!!(
        Mooncake.zero_dual(_DRM.scale), Mooncake.zero_dual(2.0), Mooncake.zero_dual(x))
end
