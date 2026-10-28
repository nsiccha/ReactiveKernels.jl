using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions, LinearAlgebra
import DifferentiationInterface, Enzyme

# These ordinary functions deliberately share the library GP spellings.
# The model module, not the spelling, owns their binding and result shape.
module GPBindingModels
scaled(x, s; shift = 0.0) = x .* s .+ shift
scaled(s) = exp(s)
const gp_exp_quad_cov = scaled
const gp_periodic_cov = scaled
const gp_chol_latent = scaled
const neutral = scaled
end

module GPBindingUndefined
end

const _GPB_NAMES = (:gp_exp_quad_cov, :gp_periodic_cov, :gp_chol_latent, :neutral)
const _GPB_BACKEND = DifferentiationInterface.AutoEnzyme(; mode = Enzyme.Reverse)

function _gpb_head(name, spelling; library = false)
    spelling === :bare && return name
    spelling === :alias && return library ?
        Dict(:gp_exp_quad_cov => :covariance, :gp_periodic_cov => :periodic_covariance,
            :gp_chol_latent => :latent)[name] : :neutral
    return Expr(:., library ? :DK : :GPBindingModels, QuoteNode(name))
end

function _gpb_build(name, spelling, kind; K = 3, n = 6)
    head = _gpb_head(name, spelling)
    x = [sin(0.7i) for i in 1:K]
    oi = [mod1(2i, K) for i in 1:n]
    y = [0.2cos(i) for i in 1:n]
    cols = (; x, oi, y)
    decl = kind === :array ? :(z[1:$K] .~ Normal.(0, 1)) : kind === :plate ?
        :(@plate for i in eachindex(y); z[i] ~ Normal(0, 1); end) : :(s ~ Normal(0, 1))
    value = kind === :scalar ? Expr(:call, head, :s) :
        kind === :dotted ? Expr(:., head, Expr(:tuple, :(x[oi]), :s)) :
        kind === :data ? Expr(:call, head, Expr(:parameters, Expr(:kw, :shift, 0.1)), :x, 0.4) :
        kind in (:array, :plate) ? Expr(:call, head, :z, 0.4) : Expr(:call, head, :x, :s)
    location = kind in (:scalar, :dotted) ? :f : :(f[oi])
    ast = quote
        $decl
        f = $value
        y .~ Normal.($location, 0.7)
    end
    inputs = kind === :scalar ? (; y) : kind in (:array, :plate) ? (; oi, y) : cols
    plan = lower_rkppl(ast, inputs; mod = GPBindingModels, conditioned = (:y,))
    bound = bind_data(plan, Dict{Symbol,Any}(pairs(inputs)))
    built = build_kernel(bound)
    u = [0.25sin(i) for i in 1:built.layout.total]
    return (; plan, bound, built, cols, u, kind)
end

function _gpb_oracle(fx, u)
    nt = constrain(fx.built.layout, u)
    mu = fx.kind === :scalar ? fill(exp(nt.s), length(fx.cols.y)) :
        fx.kind in (:array, :plate) ? (0.4 .* nt.z)[fx.cols.oi] :
        fx.kind === :data ? (0.4 .* fx.cols.x .+ 0.1)[fx.cols.oi] :
        nt.s .* fx.cols.x[fx.cols.oi]
    prior = fx.kind in (:array, :plate) ? sum(logpdf.(Normal(), nt.z)) : logpdf(Normal(), nt.s)
    return sum(logpdf.(Normal.(mu, 0.7), fx.cols.y)) + prior
end

function _gpb_findiff(f, u)
    h = cbrt(eps(Float64))
    return [(f(u .+ h .* (eachindex(u) .== i)) -
             f(u .- h .* (eachindex(u) .== i))) / (2h) for i in eachindex(u)]
end

function _gpb_check(fx)
    before = deepcopy(fx.cols)
    q = prepare_sampler(fx.built, fx.bound, fx.u; backend = _GPB_BACKEND)
    value, gradient = sampler_value_and_gradient!(q, similar(fx.u), fx.u)
    @test value ≈ _gpb_oracle(fx, fx.u) rtol = 1e-12
    @test gradient ≈ _gpb_findiff(u -> _gpb_oracle(fx, u), fx.u) rtol = 1e-6 atol = 1e-8
    @test fx.cols == before
    return value, gradient
end
