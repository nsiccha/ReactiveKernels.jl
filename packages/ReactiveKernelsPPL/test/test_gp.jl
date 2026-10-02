# Exact-GP latent construct: e2e value vs hand reference + Enzyme gradient.
# Native path only (Reactant reverse through dense Cholesky gradients is
# upstream-blocked, so no XLA assertions here by design).
using Distributions: LogNormal, Normal, logpdf
using LinearAlgebra: Symmetric, cholesky
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    gp_exp_quad_cov, gp_periodic_cov, gp_chol_latent

function _gp_bad_add(cov)
    cols = Dict{Symbol,Any}(:x => [0.0, 0.4], :oi => [1, 2], :y => [0.1, 0.2])
    plan = lower_rkppl(quote
        z[1:2] .~ Normal.(0, 1)
        w = 1.0 + gp_chol_latent($cov, z)
        y .~ Normal.(w[oi], 0.5)
    end, cols; conditioned = (:y,))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    query = prepare_query(built, bound, :sampler)
    return Base.invokelatest(query, zeros(built.layout.total))
end

@testset "exact gp end to end" begin
    plan = lower_rkppl(quote
            a ~ Normal(0, 1)
            rho_gp ~ LogNormal(0, 1)
            sigma_gp ~ LogNormal(0, 1)
            @plate for i in eachindex(y)
                z_gp[i] ~ Normal(0, 1)
            end
            f_gp = gp_chol_latent(gp_exp_quad_cov(x, sigma_gp, rho_gp, 1e-9),
                z_gp)
            mu = a .+ f_gp[oi]
            y .~ Normal.(mu, 0.5)
        end, (:y, :x, :oi); conditioned = (:y, :x))
    cols = Dict{Symbol,AbstractVector}(
        :y => [0.5, -0.2, 0.8, 0.1, -0.5, 0.3],
        :x => [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        :oi => collect(1:6))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    @test built.layout.total == 9 # a + rho + sigma + 6 z cells
    u = [0.1, 0.2, -0.1, 0.3, -0.2, 0.1, 0.0, 0.4, -0.3]
    nt = constrain(built.layout, u)
    a, rho, sig, z =
        nt.a, Float64(nt.rho_gp), Float64(nt.sigma_gp), Vector(nt.z_gp)
    K = [sig^2 * exp(-(p - q)^2 / (2 * rho^2)) + (p == q ? 1e-9 : 0.0)
         for p in cols[:x], q in cols[:x]]
    f = cholesky(Symmetric(K)).L * z
    muv = a .+ f
    ll = sum(logpdf.(Normal.(muv, 0.5), cols[:y]))
    pr = logpdf(Normal(0, 1), a) + logpdf(LogNormal(0, 1), rho) +
        logpdf(LogNormal(0, 1), sig) + sum(logpdf.(Normal(0, 1), z))
    names = coordinate_names(built.layout)
    jac = u[findfirst(==(:rho_gp), names)] + u[findfirst(==(:sigma_gp), names)]
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
    _check_gradient(built.spec, bound, u)
end

@testset "exact gp emission failures" begin
    # Undotted scalar + vector remains a Julia MethodError (P3). A module
    # call's result shape is known at execution, rather than from its name.
    @test_throws MethodError _gp_bad_add(:(gp_exp_quad_cov(x, 1.0, 1.0, 1e-9)))
    # Aniso/matrix locations fail at first eval (loud ArgumentError).
    # capability: matrix-location (multi-dimensional) gp_exp_quad_cov (todo `0bfiemp`)
    @test_broken (gp_exp_quad_cov([0.0 1.0; 2.0 3.0], 1.0, 1.0,
        1e-9); true)
end

@testset "periodic gp end to end" begin
    plan = lower_rkppl(quote
            a ~ Normal(0, 1)
            rho_gp ~ LogNormal(0, 1)
            sigma_gp ~ LogNormal(0, 1)
            @plate for i in eachindex(y)
                z_gp[i] ~ Normal(0, 1)
            end
            f_gp = gp_chol_latent(gp_periodic_cov(x, sigma_gp, rho_gp, 1.0,
                    1e-9), z_gp)
            mu = a .+ f_gp[oi]
            y .~ Normal.(mu, 0.5)
        end, (:y, :x, :oi); conditioned = (:y, :x))
    cols = Dict{Symbol,AbstractVector}(
        :y => [0.5, -0.2, 0.8, 0.1, -0.5, 0.3],
        :x => [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        :oi => collect(1:6))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    @test built.layout.total == 9 # a + rho + sigma + 6 z cells
    u = [0.1, 0.2, -0.1, 0.3, -0.2, 0.1, 0.0, 0.4, -0.3]
    nt = constrain(built.layout, u)
    a, rho, sig, z =
        nt.a, Float64(nt.rho_gp), Float64(nt.sigma_gp), Vector(nt.z_gp)
    K = [sig^2 * exp(-2 * sin(pi * abs(p - q) / 1.0)^2 / rho^2) +
         (p == q ? 1e-9 : 0.0) for p in cols[:x], q in cols[:x]]
    f = cholesky(Symmetric(K)).L * z
    muv = a .+ f
    ll = sum(logpdf.(Normal.(muv, 0.5), cols[:y]))
    pr = logpdf(Normal(0, 1), a) + logpdf(LogNormal(0, 1), rho) +
        logpdf(LogNormal(0, 1), sig) + sum(logpdf.(Normal(0, 1), z))
    names = coordinate_names(built.layout)
    jac = u[findfirst(==(:rho_gp), names)] + u[findfirst(==(:sigma_gp), names)]
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
    _check_gradient(built.spec, bound, u)
end

@testset "periodic gp emission failures" begin
    # The same Julia arithmetic rule applies to a periodic covariance.
    @test_throws MethodError _gp_bad_add(:(gp_periodic_cov(x, 1.0, 1.0, 1.0, 1e-9)))
    # Matrix locations and non-positive periods fail at first eval.
    # capability: matrix-location gp_periodic_cov") (todo `0bfiemp`)
    @test_broken (gp_periodic_cov([0.0 1.0; 2.0 3.0], 1.0, 1.0,
        1.0, 1e-9); true)
    # refused: period = 0
    @test_throws ArgumentError gp_periodic_cov([0.0, 1.0], 1.0, 1.0, 0.0, 1e-9)
end
