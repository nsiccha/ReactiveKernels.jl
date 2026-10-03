using Test, Distributions, ReactiveKernels, ReactiveKernelsPPL
using DifferentiationInterface, Enzyme, LinearAlgebra, SpecialFunctions
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    gp_exp_quad_cov, gp_periodic_cov, gp_chol_latent

# Positive successors of the historical built-in capability pins on todo
# `0bfiemp`. Every smooth is an ordinary value and every prior is stated
# (P3/P7/P8, decisions 1cmodra and 0d5a67r). Synthetic public data only.
_sc_data(n=20) = Dict{Symbol,Any}(
    :x => [sin(0.4i) + i/n for i in 1:n],
    :z => [cos(0.7i) for i in 1:n],
    :w => [sin(1.3i) for i in 1:n],
    :g => [mod1(i, 3) for i in 1:n],
    :c => [mod1(i, 3) for i in 1:n],
    :y => [0.2cos(0.9i) for i in 1:n])

function _sc_build(ast, data=_sc_data())
    plan = lower_rkppl(ast, keys(data); mod=@__MODULE__, conditioned=keys(data))
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    u = [0.12sin(i) for i in 1:built.layout.total]
    return (; bound, built, u, nt=constrain(built.layout, u))
end
_sc_node(fx, node) = Base.invokelatest(prepare_query(fx.built, fx.bound, node), fx.u)
function _sc_gradient(fx)
    q = prepare_sampler(fx.built, fx.bound, fx.u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    v, grad = sampler_value_and_gradient!(q, similar(fx.u), fx.u)
    f = prepare_query(fx.built, fx.bound, :sampler)
    fd = map(eachindex(fx.u)) do i
        up, dn = copy(fx.u), copy(fx.u)
        up[i] += 1e-5; dn[i] -= 1e-5
        (Base.invokelatest(f, up) - Base.invokelatest(f, dn)) / 2e-5
    end
    @test all(isfinite, grad)
    @test v ≈ Base.invokelatest(f, fx.u)
    @test grad ≈ fd rtol=2e-5 atol=2e-7
end

@rkppl _sc_matern(P, lam, nu) = begin
    sigma ~ LogNormal(0, 1)
    rho ~ LogNormal(0, 1)
    z[axes(P, 2)] .~ Normal.(0, 1)
    return P * (hsgp_matern_sqrt_spd(lam, sigma, rho, nu) .* z)
end
@rkppl _sc_grouped_periodic(P, harmonics, g) = begin
    sigma[levels(g)] .~ LogNormal.(0, 1)
    rho[levels(g)] .~ LogNormal.(0, 1)
    z[axes(P, 2)] .~ Normal.(0, 1)
    return P * (hsgp_periodic_grouped_sqrt_spd(harmonics, sigma, rho) .* z)
end

# The number of margins is model structure. These seven statements give
# three margins independent penalties without a data-sized model loop.
function _sc_tensor_body(k)
    kexpr = k isa Tuple ? Expr(:tuple,k...) : k
    quote
        (X, Zrrr, Zrrn, Zrnr, Zrnn, Znrr, Znrn, Znnr) = t2_basis(x,z,w;k=$kexpr)
        b[axes(X,2)] .~ Flat.()
        sd[1:7] .~ HalfNormal.(1)
        rrr[axes(Zrrr,2)] .~ Normal.(0,1)
        rrn[axes(Zrrn,2)] .~ Normal.(0,1)
        rnr[axes(Zrnr,2)] .~ Normal.(0,1)
        rnn[axes(Zrnn,2)] .~ Normal.(0,1)
        nrr[axes(Znrr,2)] .~ Normal.(0,1)
        nrn[axes(Znrn,2)] .~ Normal.(0,1)
        nnr[axes(Znnr,2)] .~ Normal.(0,1)
        f = X * b .+ Zrrr * (sd[1] .* rrr) .+ Zrrn * (sd[2] .* rrn) .+
            Zrnr * (sd[3] .* rnr) .+ Zrnn * (sd[4] .* rnn) .+
            Znrr * (sd[5] .* nrr) .+ Znrn * (sd[6] .* nrn) .+ Znnr * (sd[7] .* nnr)
    end
end
