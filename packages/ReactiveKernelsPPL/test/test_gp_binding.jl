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

@testset "GP spellings follow model bindings and ordinary shapes" begin
    for name in _GPB_NAMES, kind in (:scalar, :whole, :array, :plate, :data, :dotted)
        results = []
        for spelling in (:bare, :qualified, :alias)
            @testset "$name $kind $spelling" begin
                fx = _gpb_build(name, spelling, kind)
                push!(results, _gpb_check(fx))
            end
        end
        @test all(r -> r[1] ≈ results[1][1] && r[2] ≈ results[1][2], results)
    end
    @test all(name -> name ∉ admitted_functions(), _GPB_NAMES)
end

@testset "GP call heads obey Julia name resolution" begin
    # Undefined callees and model values follow the ordinary callable
    # contract; a GP identifier grants neither a binding nor a shape.
    for name in _GPB_NAMES[1:3]
        ast = quote
            s ~ Normal(0, 1)
            f = $(Expr(:call, name, :s))
            y .~ Normal.(f, 0.7)
        end
        # Refused: calling an undefined binding violates Julia name resolution.
        @test_throws SurfaceLoweringError lower_rkppl(ast, (:y,);
            mod = GPBindingUndefined, conditioned = (:y,))
        ast = quote
            $name ~ Normal(0, 1)
            f = $(Expr(:call, name, 0.1))
            y .~ Normal.(f, 0.7)
        end
        # Refused: a sampled model value is not a callable (functions-as-values contract).
        @test_throws SurfaceLoweringError lower_rkppl(ast, (:y,);
            mod = GPBindingModels, conditioned = (:y,))
    end
end
