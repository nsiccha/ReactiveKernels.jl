using ReactiveKernels, ReactiveKernelsPPL
import BayesianRegressionModels
include(joinpath(@__DIR__, "..", "..", "packages", "ReactiveKernelsPPL", "test", "gp_binding_fixtures.jl"))

module GPBindingLibrary
using ReactiveKernels, ReactiveKernelsPPL
import BayesianRegressionModels
const gp_exp_quad_cov = BayesianRegressionModels.rk_model(:gp_exp_quad_cov)
const gp_periodic_cov = BayesianRegressionModels.rk_model(:gp_periodic_cov)
const gp_chol_latent = BayesianRegressionModels.StatisticalPreparation.gp_chol_latent
const DK = @__MODULE__
const covariance = gp_exp_quad_cov
const periodic_covariance = gp_periodic_cov
const latent = gp_chol_latent
end

function _gpb_library(kind, spelling, declaration)
    covname = kind === :exp_quad ? :gp_exp_quad_cov : :gp_periodic_cov
    covhead = _gpb_head(covname, spelling; library = true)
    lathead = _gpb_head(:gp_chol_latent, spelling; library = true)
    K = 3
    covargs = kind === :exp_quad ? Any[:x, :amp, :rho, 1e-6] : Any[:x, :amp, :rho, 1.2, 1e-6]
    covariance = Expr(:call, covhead, covargs...)
    value = Expr(:call, lathead, :covariance_matrix, :z)
    decl = declaration === :array ? :(z[1:$K] .~ Normal.(0, 1)) : quote
        @plate for i in eachindex(y)
            z[i] ~ Normal(0, 1)
        end
    end
    decl.head === :block && (decl = only(filter(x -> !(x isa LineNumberNode), decl.args)))
    cols = (; x = [-0.4, 0.2, 0.8], oi = [3, 1, 2], y = [0.1, -0.2, 0.5])
    ast = quote
        amp ~ LogNormal(0, 1)
        rho ~ LogNormal(0, 1)
        $decl
        covariance_matrix = $covariance
        f = $value
        y .~ Normal.(f[oi], 0.7)
    end
    plan = lower_rkppl(ast, cols; mod = GPBindingLibrary, conditioned = (:y,))
    bound = bind_data(plan, Dict{Symbol,Any}(pairs(cols)))
    built = build_kernel(bound)
    u = [0.1sin(i) for i in 1:built.layout.total]
    function oracle(u)
        nt = constrain(built.layout, u)
        x = cols.x
        K = [nt.amp^2 * (kind === :exp_quad ? exp(-(a-b)^2 / (2nt.rho^2)) :
             exp(-2sinpi(abs(a-b) / 1.2)^2 / nt.rho^2)) + (i == j ? 1e-6 : 0.0)
             for (i,a) in enumerate(x), (j,b) in enumerate(x)]
        f = cholesky(Symmetric(K)).L * nt.z
        jac = sum(u[i] for (i,name) in enumerate(coordinate_names(built.layout)) if name in (:amp, :rho))
        return sum(logpdf.(Normal.(f[cols.oi], 0.7), cols.y)) +
            sum(logpdf.(Normal(), nt.z)) + logpdf(LogNormal(), nt.amp) + logpdf(LogNormal(), nt.rho) + jac
    end
    return (; plan, bound, built, u, oracle)
end

@testset "BRM covariance graphs use ordinary binding and gather routes" begin
    for kind in (:exp_quad, :periodic), declaration in (:array, :plate)
        results = []
        for spelling in (:bare, :qualified, :alias)
            fx = _gpb_library(kind, spelling, declaration)
            q = prepare_sampler(fx.built, fx.bound, fx.u; backend = _GPB_BACKEND)
            value, gradient = sampler_value_and_gradient!(q, similar(fx.u), fx.u)
            @test value ≈ fx.oracle(fx.u) rtol = 1e-12
            @test gradient ≈ _gpb_findiff(fx.oracle, fx.u) rtol = 1e-5 atol = 1e-7
            push!(results, (value, gradient))
        end
        @test all(r -> r[1] ≈ results[1][1] && r[2] ≈ results[1][2], results)
    end
end
