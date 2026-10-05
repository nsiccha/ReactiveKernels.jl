using ReactiveKernels
using Test

module InverseEdgesFixture
using ReactiveKernels

# Single unary bare call: the forward edge is authored, the reverse edge is
# synthesized automatically through InverseFunctions.jl.
@kernel log_only(x::Float64) = begin
    y::Float64 = log(x)
end

# Native exp lowering is wrapped even for scalar ports. Its authored identity
# still supplies log as the inverse, independently of that execution wrapper.
@kernel exp_only(x::Float64) = begin
    y::Float64 = exp(x)
end

const exp_alias = exp
@kernel qualified_exp(x::Float64) = begin
    y::Float64 = Base.exp(x)
end
@kernel aliased_exp(x::Float64) = begin
    y::Float64 = exp_alias(x)
end
@eval @kernel generated_exp(x::Float64) = begin
    y::Float64 = $(GlobalRef(Base, :exp))(x)
end

# Mutable bindings do not establish immutable authored call identity.
mutable_exp = exp
build_mutable_exp() = @kernel begin
    x::Float64
    y::Float64 = mutable_exp(x)
    return y
end

# The hand-written bidirectional pair this feature replaces: synthesis must
# detect the already-authored reverse edge and add nothing.
@kernel manual_pair(scale::Float64) = begin
    log_scale::Float64 = log(scale)
    scale::Float64 = exp(log_scale)
end

# Self-inverse bare operator.
@kernel negate(x::Float64) = begin
    y::Float64 = -x
end

helper(x) = x^3 + one(x)

@kernel no_inverse(x::Float64) = begin
    y::Float64 = helper(x)
end

@kernel qualified_log(x::Float64) = begin
    y::Float64 = Base.log(x)
end

@kernel named_pack(x::Float64, y::Float64) = begin
    params = (; x, y)
end

@kernel named_pack_kw(x::Float64) = begin
    params = (; first = x)
end

@kernel named_pack_eq(x::Float64, y::Float64) = begin
    nt = (a = x, b = y)
end

@kernel positional_pack(x::Float64, y::Float64) = begin
    t = (x, y)
end

@kernel splat_pack(x::Float64, ys::Vector{Float64}) = begin
    t = (x, ys...)
end

@kernel computed_field_pack(x::Float64, y::Float64) = begin
    nt = (a = x, b = 2y)
end

# An explicitly user-invertible function: synthesis must honor inverses that
# users register through the InverseFunctions extension point.
ubu(x) = inv(exp(-x) + 1)
ubu_inv(y) = log(y / (1 - y))

# Built on demand (after the test registers the inverse), not at module load.
build_ubu_spec() = @kernel begin
    x::Float64
    y::Float64 = ubu(x)
    return y
end

# A user-owned function with a tracing companion also gets a source wrapper.
@traceable cube(x) = x^3
build_cube_spec() = @kernel begin
    x::Float64
    y::Float64 = cube(x)
    return y
end

end # module InverseEdgesFixture

using InverseFunctions

@testset "automatic inverse edges" begin
    F = InverseEdgesFixture

    @testset "unary bare call inverts through InverseFunctions" begin
        forward = prepare(F.log_only; have = (:x,), want = :y)
        @test forward(2.0) ≈ log(2.0)

        backward = prepare(F.log_only; have = (:y,), want = :x)
        @test backward(1.0) ≈ MathConstants.e

        rev_plan = plan(F.log_only; have = (:y,), want = :x)
        @test length(rev_plan.recipes) == 1
        @test only(rev_plan.recipes).op === exp
    end

    @testset "self-inverse operator" begin
        backward = prepare(F.negate; have = (:y,), want = :x)
        @test backward(3.0) == -3.0
    end

    @testset "wrapped exact calls retain automatic inverses" begin
        for spec in (F.exp_only, F.qualified_exp, F.aliased_exp, F.generated_exp)
            @test length(spec.graph.recipes) == 2
            @test prepare(spec; have = (:x,), want = :y)(0.7) ≈ exp(0.7)
            backward = prepare(spec; have = (:y,), want = :x)
            @test backward(exp(0.7)) ≈ 0.7
            @test only(plan(spec; have = (:y,), want = :x).recipes).op === log
        end
    end

    @testset "mutable bindings cannot change inverse identity" begin
        @eval InverseEdgesFixture mutable_exp = log
        try
            dynamic = F.build_mutable_exp()
            @test length(dynamic.graph.recipes) == 1
            @test prepare(dynamic)(0.7) ≈ exp(0.7)
            @test_throws PlanningError plan(dynamic; have = (:y,), want = :x)
        finally
            @eval InverseEdgesFixture mutable_exp = exp
        end
    end

    @testset "authored bidirectional pair gains no duplicate" begin
        recipes = F.manual_pair.graph.recipes
        @test length(recipes) == 2
        @test count(r -> r.op === log, recipes) == 1
        @test length(ReactiveKernels.producers_of(
            F.manual_pair.graph, F.manual_pair[:scale].id)) == 1

        from_scale = prepare(F.manual_pair; have = (:scale,), want = :log_scale)
        @test from_scale(2.0) ≈ log(2.0)
        from_log = prepare(F.manual_pair; have = (:log_scale,), want = :scale)
        @test from_log(1.0) ≈ MathConstants.e
    end

    @testset "HAVE authority and cycle diagnostics are unchanged" begin
        settled = plan(F.log_only; have = (:x, :y), want = :y)
        @test isempty(settled.recipes)

        @test_throws PlanningError plan(F.log_only; have = (), want = (:x, :y))
    end

    @testset "functions without an inverse gain no edge" begin
        @test inverse(typeof(F.helper)) isa NoInverse
        @test_throws PlanningError plan(F.no_inverse; have = (:y,), want = :x)
        @test length(F.no_inverse.graph.recipes) == 1
    end

    @testset "qualified calls use their retained inverse identity" begin
        @test prepare(F.qualified_log; have = (:y,), want = :x)(1.0) ≈ exp(1.0)
    end

    @testset "user-registered inverses are honored" begin
        # Qualified: the documented InverseFunctions extension point (a bare
        # `inverse(...) = ...` under `using` would shadow, not extend).
        InverseFunctions.inverse(::typeof(F.ubu)) = F.ubu_inv
        spec = F.build_ubu_spec()
        backward = prepare(spec; have = (:y,), want = :x)
        @test backward(F.ubu(4.2)) ≈ 4.2

        InverseFunctions.inverse(::typeof(F.cube)) = cbrt
        wrapped = F.build_cube_spec()
        @test length(wrapped.graph.recipes) == 2
        @test prepare(wrapped)(2.0) == 8.0
        @test prepare(wrapped; have = (:y,), want = :x)(8.0) ≈ 2.0
        combined = merge(wrapped, F.build_cube_spec())
        @test length(ReactiveKernels.producers_of(
            combined.graph, combined[:y].id)) == 1
    end
end

@testset "automatic tuple pack/unpack edges" begin
    F = InverseEdgesFixture

    @testset "named tuple shorthand" begin
        forward = prepare(F.named_pack; have = (:x, :y), want = :params)
        @test forward(1.0, 2.0) == (x = 1.0, y = 2.0)

        get_x = prepare(F.named_pack; have = (:params,), want = :x)
        @test get_x((x = 1.0, y = 2.0)) == 1.0
        get_y = prepare(F.named_pack; have = (:params,), want = :y)
        @test get_y((x = 1.0, y = 2.0)) == 2.0
        get_both = prepare(F.named_pack; have = (:params,), want = (:x, :y))
        @test get_both((x = 1.0, y = 2.0)) == (1.0, 2.0)
    end

    @testset "named tuple keyword and = forms" begin
        get_first = prepare(F.named_pack_kw; have = (:params,), want = :x)
        @test get_first((first = 4.0,)) == 4.0

        get_a = prepare(F.named_pack_eq; have = (:nt,), want = :x)
        @test get_a((a = 1.0, b = 2.0)) == 1.0
        get_b = prepare(F.named_pack_eq; have = (:nt,), want = :y)
        @test get_b((a = 1.0, b = 2.0)) == 2.0
    end

    @testset "positional tuple" begin
        forward = prepare(F.positional_pack; have = (:x, :y), want = :t)
        @test forward(1.0, 2.0) == (1.0, 2.0)

        get_x = prepare(F.positional_pack; have = (:t,), want = :x)
        @test get_x((1.0, 2.0)) == 1.0
        get_y = prepare(F.positional_pack; have = (:t,), want = :y)
        @test get_y((1.0, 2.0)) == 2.0
    end

    @testset "splat and computed fields are skipped" begin
        get_x = prepare(F.splat_pack; have = (:t,), want = :x)
        @test get_x((1.0, 2.0, 3.0)) == 1.0
        @test_throws PlanningError plan(F.splat_pack; have = (:t,), want = :ys)

        get_a = prepare(F.computed_field_pack; have = (:nt,), want = :x)
        @test get_a((a = 1.0, b = 4.0)) == 1.0
        @test_throws PlanningError plan(
            F.computed_field_pack; have = (:nt,), want = :y)
    end

    @testset "pack/unpack round trip keeps HAVE authority" begin
        settled = plan(F.named_pack; have = (:x, :y, :params), want = :params)
        @test isempty(settled.recipes)
        via_pack =
            plan(F.named_pack; have = (:x, :y), want = (:x, :params))
        @test length(via_pack.recipes) == 1
    end

    @testset "unpack edges infer like literal field access" begin
        # A value-held selector (e.g. Fix2(getproperty, :x)) widens to the
        # union of all field types; the parametric accessors must not.
        get_x = prepare(F.named_pack; have = (:params,), want = :x)
        @test @inferred(get_x((x = 1.0, y = 2))) === 1.0
        get_y = prepare(F.named_pack; have = (:params,), want = :y)
        @test @inferred(get_y((x = 1.0, y = 2))) === 2
        get_first = prepare(F.positional_pack; have = (:t,), want = :x)
        @test @inferred(get_first((1.0, 2))) === 1.0
    end
end
