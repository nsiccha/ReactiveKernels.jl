using ReactiveKernels
using Test

module KernelObjectAuthoringFixture
using ReactiveKernels

# A compact Normal-shaped standard family.  The CDF/quantile pair uses the
# common logistic approximation so this authoring fixture needs no optional
# distribution dependency; the graph/composition semantics are the subject.
@kernel standard_normal() = begin
    logpdf(z::Float64)::Float64 = -0.5 * log(2π) - 0.5z^2
    cdf(z::Float64)::Float64 = inv(1 + exp(-1.702z))
    quantile(p::Float64)::Float64 = log(p / (1 - p)) / 1.702
end

@kernel standard_cauchy() = begin
    logpdf(z::Float64)::Float64 = -log(π) - log1p(z^2)
    cdf(z::Float64)::Float64 = 0.5 + atan(z) / π
    quantile(p::Float64)::Float64 = tan(π * (p - 0.5))
end

@kernel location_scale(standard, location::Float64, scale::Float64) = begin
    # Ordinary bidirectional relation: the natural constructor HAS scale,
    # while log_scale is an alternate transparent cut.  If both are supplied,
    # HAVE authority prevents either recipe from running.
    log_scale::Float64 = log(scale)
    scale::Float64 = exp(log_scale)

    standardized(x::Float64)::Float64 = (x - location) / scale
    inv(standardized, z::Float64)::Float64 = location + scale * z

    logpdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        standard.logpdf(z) - log_scale
    end
    cdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        standard.cdf(z)
    end
    quantile(p::Float64)::Float64 = begin
        z::Float64 = standard.quantile(p)
        inv(standardized, z)
    end
end

@kernel normal = location_scale(standard_normal)
@kernel cauchy = location_scale(standard_cauchy)

# Endpoint application spells only the method argument.  Residual owner ports
# are matched by name and the distinguished endpoint return is spliced into the
# enclosing graph.
@kernel nested_normal_logpdf(x::Float64, location::Float64, scale::Float64) = begin
    logdensity::Float64 = normal.logpdf(x)
    return logdensity
end

end

module KernelObjectPureCalleeFixture
using ReactiveKernels

module PureMath
export curve
const SHIFT = 0.25
curve(x) = x^2 + SHIFT
end

using .PureMath: curve

@kernel bare_external() = begin
    endpoint(x::Float64)::Float64 = curve(x)
end

@kernel qualified_external() = begin
    endpoint(x::Float64)::Float64 = PureMath.curve(x)
end

# A qualified bang call is not silently advertised as a pure endpoint. It stays
# on the stateful/effectful method path, where its effects can be accounted for.
@kernel qualified_effect() = begin
    endpoint(xs::Vector{Int})::Vector{Int} = Base.push!(xs, 1)
end

end

@testset "transparent pure kernel objects and residual endpoints" begin
    F = KernelObjectAuthoringFixture

    @test F.normal isa KernelObjectSpec
    @test ReactiveKernels.kernel_endpoint_names(F.normal) ==
          (:standardized, :logpdf, :cdf, :quantile)
    @test Tuple(v.name for v in inputs(F.normal.logpdf)) ==
          (:location, :scale, :x)
    @test Tuple(v.name for v in outputs(F.normal.logpdf)) == (:logpdf,)
    @test !haskey(F.normal, :standard)
    @test prepare(F.nested_normal_logpdf)(1.0, 0.0, 2.0) ≈
          prepare(F.normal.logpdf)(0.0, 2.0, 1.0)
    @test !occursin("KernelSpec", sprint(show,
        code_expr(plan(F.nested_normal_logpdf))))

    P = KernelObjectPureCalleeFixture
    @test prepare(P.bare_external.endpoint)(2.0) == 4.25
    @test prepare(P.qualified_external.endpoint)(2.0) == 4.25
    @test isempty(ReactiveKernels.kernel_endpoint_names(P.qualified_effect))

    @test_throws ArgumentError @macroexpand @kernel duplicate_endpoints() = begin
        endpoint(x::Float64)::Float64 = x
        endpoint(y::Float64)::Float64 = y
    end
    @test_throws ArgumentError @macroexpand @kernel colliding_endpoint(x::Float64) = begin
        endpoint::Float64 = x + 1
        endpoint(y::Float64)::Float64 = y
    end

    @testset "ordinary and alternate location-scale cuts" begin
        outputs_of(p) = [only(recipe.outputs).name for recipe in p.recipes]
        scale_plan = plan(F.normal.logpdf;
            have = (:x, :location, :scale), want = :logpdf)
        logscale_plan = plan(F.normal.logpdf;
            have = (:x, :location, :log_scale), want = :logpdf)
        both_plan = plan(F.normal.logpdf;
            have = (:x, :location, :scale, :log_scale), want = :logpdf)

        @test :log_scale in outputs_of(scale_plan)
        @test !(:scale in outputs_of(scale_plan))
        @test :scale in outputs_of(logscale_plan)
        @test !(:log_scale in outputs_of(logscale_plan))
        @test !(:scale in outputs_of(both_plan))
        @test !(:log_scale in outputs_of(both_plan))

        expected_normal(x, location, scale) =
            -0.5 * log(2π) - log(scale) - 0.5 * ((x - location) / scale)^2
        @test prepare(scale_plan)(1.0, 0.0, 2.0) ≈
              expected_normal(1.0, 0.0, 2.0)
        @test prepare(logscale_plan)(1.0, 0.0, log(2.0)) ≈
              expected_normal(1.0, 0.0, 2.0)
        @test prepare(both_plan)(1.0, 0.0, 2.0, log(2.0)) ≈
              expected_normal(1.0, 0.0, 2.0)

        logscale_view = extract(F.normal;
            have = (:scale,), want = :log_scale)
        scale_view = extract(F.normal;
            have = (:log_scale,), want = :scale)
        @test prepare(logscale_view)(2.0) ≈ log(2.0)
        @test prepare(scale_view)(log(2.0)) ≈ 2.0
    end

    @testset "shared endpoint provenance and explicit inverse" begin
        joint = extract(F.normal;
            have = (:x, :location, :scale), want = (:logpdf, :cdf))
        joint_plan = plan(joint)
        @test count(r -> only(r.outputs).name === :standardized,
                    joint_plan.recipes) == 1
        @test all(isapprox.(prepare(joint)(1.0, 0.0, 2.0), (
            -0.5 * log(2π) - log(2.0) - 0.5 * 0.5^2,
            inv(1 + exp(-1.702 * 0.5)),
        )))

        quantile = extract(F.cauchy;
            have = (:p, :location, :scale), want = :quantile)
        qplan = plan(quantile)
        qoutputs = [only(r.outputs).name for r in qplan.recipes]
        @test :x in qoutputs
        @test !(:standardized in qoutputs)
        @test prepare(quantile)(0.75, 3.0, 2.0) ≈ 5.0

        authoritative = plan(F.normal.logpdf;
            have = (:x, :standardized, :location, :scale, :log_scale),
            want = :logpdf)
        @test !(:standardized in [only(r.outputs).name for r in authoritative.recipes])
    end

    @testset "ordinary graph execution surfaces" begin
        spec = extract(F.normal;
            have = (:x, :location, :scale), want = :logpdf)
        prepared = prepare(spec)
        @test prepared(1.0, 0.0, 2.0) ≈
              -0.5 * log(2π) - log(2.0) - 0.5 * 0.5^2

        plated = plate(spec;
            have = (:x, :location, :scale), want = :logpdf,
            batched = (:x,))
        xs = [-1.0, 0.0, 1.0]
        @test plated(xs, 0.0, 2.0) ≈ sum(prepared(x, 0.0, 2.0) for x in xs)

        state = ReactiveState(spec; materialize = :logpdf)
        set!(state, spec.x, 1.0)
        set!(state, spec.location, 0.0)
        set!(state, spec.scale, 2.0)
        @test get!(state, spec.logpdf) ≈ prepared(1.0, 0.0, 2.0)

        lowered = code_expr(plan(spec))
        @test lowered == code_expr(plan(spec))
        text = sprint(show, lowered)
        @test !occursin("KernelSpec", text)
        @test !occursin("location_scale", text)
        @test !occursin("standard_normal", text)
        @test all(!(recipe.op isa KernelSpec) for recipe in kernel_graph(spec).recipes)
    end

    @testset "automatic pure CSE never merges independent/effectful work" begin
        calls = Ref(0)
        pure_piece = @kernel pure_piece(x::Float64) = begin
            y::Float64 = x + 1
            return y
        end
        effect_piece = @kernel effect_piece(x::Float64) = begin
            @recipe (effectful = true) y::Float64 = (calls[] += 1; x + 1)
            return y
        end
        other_piece = @kernel other_piece(x::Float64) = begin
            y::Float64 = x + 1
            return y
        end

        repeated_pure = @kernel repeated_pure(x::Float64) = begin
            a::Float64 = pure_piece(x)
            b::Float64 = pure_piece(x)
            return (a, b)
        end
        repeated_effect = @kernel repeated_effect(x::Float64) = begin
            a::Float64 = effect_piece(x)
            b::Float64 = effect_piece(x)
            return (a, b)
        end
        independent = @kernel independent(x::Float64) = begin
            a::Float64 = pure_piece(x)
            b::Float64 = other_piece(x)
            return (a, b)
        end

        @test length(kernel_graph(repeated_pure).recipes) == 1
        @test length(kernel_graph(repeated_effect).recipes) == 2
        @test length(kernel_graph(independent).recipes) == 2
        @test_throws PlanningError plan(repeated_effect)
        @test calls[] == 0
    end

    docs = read(joinpath(@__DIR__, "..", "docs", "src", "index.md"), String)
    @test occursin("@kernel normal = location_scale(standard_normal)", docs)
    @test occursin("logdensity::Float64 = normal.logpdf(x)", docs)
    @test occursin("have = (:x, :location, :log_scale), want = :logpdf", docs)
end
