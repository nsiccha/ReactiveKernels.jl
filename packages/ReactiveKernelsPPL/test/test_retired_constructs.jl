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
