using Distributions, ReactiveKernelsPPL, Test

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
mo(x) = x .+ 0.1
linear_pk_read_locs(x) = x .+ 0.2
@rkppl varying_effect(x) = begin
    b ~ Normal(0, 2)
    return b .* x
end
end

@testset "retired catalogue spellings remain caller-owned bindings" begin
    p = lower_rkppl(quote
        q ~ horseshoe_coefs(x)
        m = monotonic(q)
        y .~ Normal.(m, 1)
    end, (:x, :y); mod = RetiredCatalogueCaller, conditioned = (:y,))
    bound = bind_data(p, Dict(:x => [0.2, 0.4], :y => [0.3, -0.1]))
    @test coordinate_names(build_kernel(bound).layout) == [Symbol("q.b")]
    p = lower_rkppl(quote
        a ~ Normal(0, 1)
        m = linear_pk_read_locs(x)
        y .~ Normal.(a .+ m, 1)
    end, (:x, :y); mod = RetiredCatalogueCaller, conditioned = (:y,))
    bound = bind_data(p, Dict(:x => [0.2, 0.4], :y => [0.3, -0.1]))
    built = build_kernel(bound)
    @test coordinate_names(built.layout) == [:a]
    @test Base.invokelatest(prepare_query(built, bound, :sampler), [0.1]) ≈
        logpdf(Normal(), 0.1) + sum(logpdf.(Normal.([0.5, 0.7], 1), [0.3, -0.1]))
    p = lower_rkppl(quote
        q ~ varying_effect(x)
        m = mo(q)
        y .~ Normal.(m, 1)
    end, (:x, :y); mod = RetiredCatalogueCaller, conditioned = (:y,))
    bound = bind_data(p, Dict(:x => [0.2, 0.4], :y => [0.3, -0.1]))
    @test coordinate_names(build_kernel(bound).layout) == [Symbol("q.b")]
end

@testset "retired implicit statistical constructs explain ordinary replacement" begin
    for (name, replacement) in ReactiveKernelsPPL._RETIRED_CONSTRUCT_MODELS
        ast = name in (:varying_effect, :varying_draws, :varying_slice) ?
            Expr(:block, Expr(:call, :~, :q, Expr(:call, name))) :
            Expr(:block, Expr(:(=), :q, Expr(:call, name)))
        err = try
            lower_rkppl(ast, (); mod = ReactiveKernelsPPL)
            nothing
        catch error
            error
        end
        @test err isa SurfaceLoweringError
        @test occursin("retired", sprint(showerror, err))
        @test occursin("BayesianRegressionModels.rkppl_model(:$replacement)",
            sprint(showerror, err))
    end
    for name in (:spline_basis, :hsgp_basis)
        ast = Expr(:block, Expr(:call, name, QuoteNode(:basis), :x))
        err = try
            lower_rkppl(ast, (:x,); mod = ReactiveKernelsPPL)
            nothing
        catch error
            error
        end
        @test err isa SurfaceLoweringError
        @test occursin("explicit priors", sprint(showerror, err))
    end
end

@testset "adopted PK helpers and legacy panel syntax name replacements" begin
    for name in ReactiveKernelsPPL._RETIRED_PK_HEADS
        err = try
            lower_rkppl(Expr(:block, Expr(:(=), :q, Expr(:call, name))), ();
                mod = ReactiveKernelsPPL)
            nothing
        catch error
            error
        end
        @test err isa SurfaceLoweringError
        @test occursin("RKPPLBench", sprint(showerror, err))
        @test !isdefined(ReactiveKernelsPPL, name)
    end
    for ast in (quote
            q ~ plate(x, y; subjects = N) do xx, yy
                yy .~ Normal.(xx, 1)
                xx
            end
        end, quote
            @plate q for i in 1:N
                y .~ Normal.(x, 1)
                x
            end
        end)
        err = try
            lower_rkppl(ast, (:x, :y); mod = ReactiveKernelsPPL,
                conditioned = (:y,))
            nothing
        catch error
            error
        end
        @test err isa SurfaceLoweringError
        @test occursin("retired", sprint(showerror, err))
        @test occursin("@plate for", sprint(showerror, err))
    end
end
