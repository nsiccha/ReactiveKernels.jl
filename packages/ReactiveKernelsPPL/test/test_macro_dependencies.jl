module ModelMacroDependencyTests
using ReactiveKernelsPPL, ExpressionExplorer, Test

const calls = Ref(0)
helper(a, x) = (calls[] += 1; a .+ x)

# Capturing a model must still work before its helper or data is defined.
const delayed = @rkppl begin
    a ~ Normal(0, 1)
    mu = defined_later(a, x)
    y .~ Normal.(mu, 1)
end
defined_later(a, x) = a .+ x

const never_called = @rkppl begin
    a ~ Normal(0, 1)
    mu = helper(a, x)
    y .~ Normal.(mu, 1)
end

references(ex) = ExpressionExplorer.compute_reactive_node(
    macroexpand(@__MODULE__, ex)).references

@testset "captured models expose helper dependencies" begin
    @test calls[] == 0
    @test never_called isa RKPPLModel
    @test delayed isa RKPPLModel
    for call in (:(helper(a, x)), :(helper.(a, x)),
            :(Helpers.helper(a, x)), :(Factory{Scalar}(a, x)))
        refs = references(:(@rkppl begin
            a ~ Normal(0, 1)
            mu = $call
            y .~ Normal.(mu, 1)
        end))
        expected = call.args[1] === :helper ? :helper :
            call.args[1] isa Expr && call.args[1].head === :curly ? :Factory : :Helpers
        @test expected in refs
        @test isempty(intersect(refs, Set([:a, :x, :y, :mu, :Normal])))
    end
    # The normal data-binding expression is visible independently of the
    # helpers in the quoted body. Model variables must not become globals.
    refs = references(:(@rkppl inputs begin
        a ~ Normal(0, 1)
        mu = helper(a, x)
        y .~ Normal.(mu, 1)
    end))
    @test :inputs in refs
    @test :helper in refs
    @test isempty(intersect(refs, Set([:a, :x, :y, :mu, :Normal])))

    refs = references(:(@rkppl part(x; scale = helper_scale()) = begin
        a ~ Normal(0, scale)
        mu = helper(a, x)
        mu
    end))
    @test :helper in refs
    @test :helper_scale in refs
    @test !(:x in refs)
    @test !(:scale in refs)

    refs = references(:(@rkppl begin
        @plate for i in eachindex(x)
            a[i] ~ Normal(0, 1)
            y[i] ~ Normal(helper(a[i], x[i]), 1)
        end
    end))
    @test :helper in refs
    @test isempty(intersect(refs, Set([:a, :x, :y, :i, :Normal])))
end

@testset "dependency reads preserve deferred model construction" begin
    # Even a qualified binding that does not exist is not read at capture.
    missing_module = @rkppl begin
        a ~ Normal(0, 1)
        mu = NotDefinedHere.helper(a, x)
        y .~ Normal.(mu, 1)
    end
    @test missing_module isa RKPPLModel
    @test calls[] == 0
    for n in (0, 1, 4)
        x = collect(1.0:n) ./ max(n, 1) .- 0.5
        y = x .+ 0.2
        bound = delayed(; x) | (; y)
        built = build_kernel(bound)
        density = prepare_query(built, bound, :sampler)
        for a in (-0.3, 0.0, 0.4)
            expected = -0.5log(2pi) - a^2 / 2 +
                sum(-0.5log(2pi) - (yi - (a + xi))^2 / 2
                    for (xi, yi) in zip(x, y); init = 0.0)
            @test Base.invokelatest(density, [a]) ≈ expected
        end
    end
end
end
