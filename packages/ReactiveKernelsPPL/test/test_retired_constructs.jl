using ReactiveKernelsPPL, Test

@testset "retired implicit shrinkage declarations explain the replacement" begin
    # USER ownership mapping 15rhwoy/1f0k9oz: BRM owns these models;
    # their implicit coefficient allocation no longer belongs to the PPL.
    for (ast, model) in (
            (quote
                R2 ~ Beta(1, 1)
                phi ~ Dirichlet([1.0])
                mu = a .+ b .* x
                r2d2(mu, R2, phi)
                y .~ Normal.(mu, 1)
            end, "r2d2_coefs"),
            (quote
                a ~ Normal(0, 1)
                b ~ Horseshoe()
                mu = a .+ b .* x
                y .~ Normal.(mu, 1)
            end, "horseshoe_coefs"))
        err = try
            lower_rkppl(ast, (:x, :y); mod = ReactiveKernelsPPL,
                conditioned = (:y,))
            nothing
        catch error
            error
        end
        @test err isa SurfaceLoweringError
        message = sprint(showerror, err)
        @test occursin("retired", message)
        @test occursin("priors", message)
        @test occursin("BayesianRegressionModels.rkppl_model(:$model)", message)
    end
end

@testset "retired producer catalogue points to its owner" begin
    for name in ReactiveKernelsPPL._RETIRED_MODEL_HEADS
        ast = Expr(:block, Expr(:call, :~, :q, Expr(:call, name)))
        err = try
            lower_rkppl(ast, (); mod = ReactiveKernelsPPL)
            nothing
        catch error
            error
        end
        @test err isa SurfaceLoweringError
        @test occursin("BayesianRegressionModels.rkppl_model(:$name)",
            sprint(showerror, err))
        @test !isdefined(ReactiveKernelsPPL, name)
    end
end

module RetiredCatalogueCaller
using ReactiveKernelsPPL
@rkppl horseshoe_coefs(x) = begin
    b ~ Normal(0, 3)
    return b .* x
end
monotonic(x) = x .+ 0.5
end

@testset "retired catalogue spellings remain caller-owned bindings" begin
    p = lower_rkppl(quote
        q ~ horseshoe_coefs(x)
        m = monotonic(q)
        y .~ Normal.(m, 1)
    end, (:x, :y); mod = RetiredCatalogueCaller, conditioned = (:y,))
    bound = bind_data(p, Dict(:x => [0.2, 0.4], :y => [0.3, -0.1]))
    @test coordinate_names(build_kernel(bound).layout) == [Symbol("q.b")]
end
