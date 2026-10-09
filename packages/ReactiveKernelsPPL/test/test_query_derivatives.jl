using DifferentiationInterface
using Distributions
using Enzyme
using LinearAlgebra
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Second-order derivatives of prepared queries (`prepare_query_ad`):
# Hessian-vector products of scalar presets. Oracles are analytic. Every population model has one effect per group, and groups never
# interact, so one direction colored over all groups recovers each group's
# Hessian entries at once.

# Reverse over forward (`reactivekernels-use` §7a2): forward over reverse
# fails on the design-matrix product the linear model lowers to.
const _QD_SECOND_ORDER = SecondOrder(AutoEnzyme(; mode = Enzyme.Reverse),
    AutoEnzyme(; mode = Enzyme.Forward))
const _QD_REVERSE = AutoEnzyme(; mode = Enzyme.Reverse)

const _QD_GROUPS = 5
const _QD_PER = 4
const _QD_G = repeat(1:_QD_GROUPS, inner = _QD_PER)
const _QD_T = repeat([0.5, 1.0, 2.0, 4.0], _QD_GROUPS)
const _QD_Y = [0.9 * exp(-0.4 * t) + 0.05 * sin(7i) for (i, t) in enumerate(_QD_T)]

function _qd_build(ast, data)
    plan = lower_rkppl(ast, keys(data); conditioned = (:y,))
    bound = bind_data(plan, data)
    return (; bound, built = build_kernel(bound))
end

_qd_index(layout, name) = findfirst(==(name), coordinate_names(layout))
_qd_effects(layout) = [_qd_index(layout, Symbol("eta.", j)) for j in 1:_QD_GROUPS]
_qd_unit(n, i) = (e = zeros(n); e[i] = 1.0; e)

# Linear-Gaussian population model: the log density is quadratic in
# (mu, eta), so its Hessian is constant.
const _QD_LINEAR = quote
    mu ~ Normal(0, 1)
    eta[levels(g)] .~ Normal.(0, 1)
    y .~ Normal.(mu .+ eta[g], 0.5)
end

# Nonlinear population model as a plate cell with one shared scale.
const _QD_PLATE = quote
    mu ~ Normal(0, 1)
    rate ~ Normal(-1, 0.5)
    sigma ~ Exponential(1)
    eta[levels(g)] .~ Normal.(0, 1)
    @plate for i in eachindex(y)
        c = exp(mu + eta[g[i]]) * exp(-exp(rate) * t[i])
        y[i] ~ Normal(c, sigma)
    end
end

_qd_mean(u, layout) = exp.(u[_qd_index(layout, :mu)] .+ u[_qd_effects(layout)][_QD_G]) .*
    exp.(-exp(u[_qd_index(layout, :rate)]) .* _QD_T)

function _qd_nested_hvp(u, v)
    fx = _qd_build(_QD_PLATE, Dict(:y => _QD_Y, :g => _QD_G, :t => _QD_T))
    q = prepare_query_ad(prepare_ad_hvp, fx.built, fx.bound, :sampler,
        _QD_SECOND_ORDER, (v,), u)
    return ad_gradient_and_hvp(q, (v,), u)
end

@testset "query Hessian-vector products" begin
    data = Dict(:y => _QD_Y, :g => _QD_G)
    fx = _qd_build(_QD_LINEAR, data)
    layout = fx.built.layout
    d = layout.total
    imu, ieta = _qd_index(layout, :mu), _qd_effects(layout)
    X = zeros(length(_QD_Y), d)
    for i in eachindex(_QD_G)
        X[i, imu] = 1.0
        X[i, ieta[_QD_G[i]]] = 1.0
    end
    likelihood_hessian = -(X' * X) ./ 0.25
    sampler_hessian = likelihood_hessian - I
    u = [0.2 * cos(i) for i in 1:d]
    v = [sin(i) for i in 1:d]

    q = prepare_query_ad(prepare_ad_hvp, fx.built, fx.bound, :sampler,
        _QD_SECOND_ORDER, (v,), u)
    @test q isa QueryAD
    @test q.layout === layout
    @test only(ad_hvp(q, (v,), u)) ≈ sampler_hessian * v
    gradient, (hv,) = ad_gradient_and_hvp(q, (v,), u)
    sampler = prepare_sampler(fx.built, fx.bound, u; backend = _QD_REVERSE)
    @test gradient ≈ last(sampler_value_and_gradient!(sampler, similar(u), u))
    @test hv ≈ sampler_hessian * v
    results = (similar(u),)
    @test ad_hvp!(q, results, (2v,), u) === results
    @test only(results) ≈ 2 * sampler_hessian * v

    units = ntuple(i -> _qd_unit(d, i), d)
    full = prepare_query_ad(prepare_ad_hvp, fx.built, fx.bound, :likelihood,
        _QD_SECOND_ORDER, units, u)
    @test reduce(hcat, ad_hvp(full, units, u)) ≈ likelihood_hessian

    # One direction colored over every group effect returns each effect's
    # diagonal Hessian entry at its own coordinate.
    colored = zeros(d)
    colored[ieta] .= 1.0
    (compressed,) = ad_hvp(q, (colored,), u)
    @test compressed[ieta] ≈ diag(sampler_hessian)[ieta]
end

@testset "query Hessian-vector products of a nonlinear plate model" begin
    data = Dict(:y => _QD_Y, :g => _QD_G, :t => _QD_T)
    fx = _qd_build(_QD_PLATE, data)
    layout = fx.built.layout
    d = layout.total
    ieta = _qd_effects(layout)
    u = [0.1 * cos(i) for i in 1:d]
    sigma = exp(u[_qd_index(layout, :sigma)])
    c = _qd_mean(u, layout)
    colored = zeros(d)
    colored[ieta] .= 1.0
    q = prepare_query_ad(prepare_ad_hvp, fx.built, fx.bound, :likelihood,
        _QD_SECOND_ORDER, (colored,), u)
    (compressed,) = ad_hvp(q, (colored,), u)
    # ∂c/∂eta = ∂²c/∂eta² = c, so each effect's likelihood curvature is
    # Σ_i ((y_i - c_i) c_i - c_i^2) / sigma^2 over its group's observations.
    expected = [sum(((_QD_Y[i] - c[i]) * c[i] - c[i]^2) / sigma^2
                    for i in eachindex(_QD_G) if _QD_G[i] == j) for j in 1:_QD_GROUPS]
    @test compressed[ieta] ≈ expected

    units = ntuple(i -> _qd_unit(d, i), d)
    full = prepare_query_ad(prepare_ad_hvp, fx.built, fx.bound, :sampler,
        _QD_SECOND_ORDER, units, u)
    H = reduce(hcat, ad_hvp(full, units, u))
    @test H ≈ H'

    # Built, prepared and evaluated inside a compiled caller.
    v = [cos(2i) for i in 1:d]
    nested_gradient, (nested,) = _qd_nested_hvp(u, v)
    @test nested ≈ H * v
    sampler = prepare_sampler(fx.built, fx.bound, u; backend = _QD_REVERSE)
    @test nested_gradient ≈ last(sampler_value_and_gradient!(sampler, similar(u), u))
end
