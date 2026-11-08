using DifferentiationInterface
using Enzyme
using LinearAlgebra
using Random
using ReactiveKernels
using ReactiveKernelsPPL
using Statistics
using Test

# The generated summed-plate reverse (ReactiveKernels `with_plate_reverse`,
# decision ReactiveKernels:plate-reverse 0ayxb72) against the backend's
# ordinary reverse over the same prepared program: equal values, gradients to
# rounding, and every plate program either a rule or a named fallback.

module PlateReverseModels end

_prv_stdz(x) = (x .- mean(x)) ./ std(x)

function _prv_build(model, data, names)
    plan = lower_rkppl(model, names; mod = PlateReverseModels, conditioned = names)
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    built, bound
end

function _prv_compare(built, bound; points = 5, seed = 1)
    n = built.layout.total
    pts = [0.4 .* randn(MersenneTwister(seed + k), n) for k in 1:points]
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    q0 = with_plate_reverse(false) do
        prepare_sampler(built, bound, pts[1]; backend)
    end
    q1 = with_plate_reverse(true) do
        prepare_sampler(built, bound, pts[1]; backend)
    end
    report = with_plate_reverse(true) do
        plate_reverse_report(q1.kernel)
    end
    g0, g1 = similar(pts[1]), similar(pts[1])
    for p in pts
        v0, _ = sampler_value_and_gradient!(q0, g0, p)
        v1, _ = sampler_value_and_gradient!(q1, g1, p)
        @test isapprox(v1, v0; rtol = 1e-12)
        @test isapprox(g1, g0; rtol = 1e-9, atol = 1e-12)
    end
    report
end

_prv_rules(report) = count(r -> last(r) == "rule", report)

@testset "generated summed-plate reverse" begin
    @testset "gauss_ri: gathered offset, guarded Normal density (N = $N)" for (N, J) in
            ((1, 1), (7, 3), (200, 5), (2000, 50))
        rng = MersenneTwister(12)
        x = N == 1 ? [0.3] : _prv_stdz(randn(rng, N))
        g = [mod1(i, J) for i in 1:N]
        y = 1 .+ 0.5 .* x .+ (0.7 .* randn(rng, J))[g] .+ randn(rng, N)
        built, bound = _prv_build(quote
            a ~ Normal(0, 5)
            b ~ Normal(0, 1)
            sd ~ Exponential(1)
            sigma ~ Exponential(1)
            z[1:J] .~ Normal.(0, 1)
            mu = a .+ b .* x .+ sd .* z[g]
            y .~ Normal.(mu, sigma)
        end, Dict(:y => y, :x => x, :g => g, :J => J), (:y, :x, :g, :J))
        report = _prv_compare(built, bound)
        # The likelihood plate takes the generated reverse.
        @test any(r -> first(r) === :_ppl_lik_y_resp && last(r) == "rule", report)
    end

    @testset "gauss_fe: design product feeding the plate" begin
        rng = MersenneTwister(11)
        N, K = 300, 3
        X = randn(rng, N, K)
        y = 1 .+ X * [0.3, -0.2, 0.5] .+ randn(rng, N)
        built, bound = _prv_build(quote
            a ~ Normal(0, 5)
            beta[1:K] .~ Normal.(0, 1)
            sigma ~ Exponential(1)
            mu = a .+ X * beta
            y .~ Normal.(mu, sigma)
        end, Dict(:y => y, :X => X, :K => K), (:y, :X, :K))
        report = _prv_compare(built, bound)
        @test !isempty(report)
    end

    @testset "Poisson log link" begin
        rng = MersenneTwister(13)
        N, J = 400, 8
        x = randn(rng, N); g = [mod1(i, J) for i in 1:N]
        y = [rand(rng, 0:6) for _ in 1:N]
        built, bound = _prv_build(quote
            a ~ Normal(0, 2)
            b ~ Normal(0, 1)
            sd ~ Exponential(1)
            z[1:J] .~ Normal.(0, 1)
            eta = a .+ b .* x .+ sd .* z[g]
            y .~ Poisson.(exp.(eta))
        end, Dict(:y => y, :x => x, :g => g, :J => J), (:y, :x, :g, :J))
        _prv_compare(built, bound)
    end

    @testset "Bernoulli logit link" begin
        rng = MersenneTwister(14)
        N, J = 400, 8
        x = randn(rng, N); g = [mod1(i, J) for i in 1:N]
        y = [rand(rng, Bool) for _ in 1:N]
        built, bound = _prv_build(quote
            a ~ Normal(0, 2)
            b ~ Normal(0, 1)
            sd ~ Exponential(1)
            z[1:J] .~ Normal.(0, 1)
            eta = a .+ b .* x .+ sd .* z[g]
            y .~ Bernoulli.(logistic.(eta))
        end, Dict(:y => y, :x => x, :g => g, :J => J), (:y, :x, :g, :J))
        _prv_compare(built, bound)
    end

    # Evidence wrappers, counts with trials, overdispersion and absent
    # responses over the same gathered predictor. A plate the generator does
    # not differentiate falls back to the backend's reverse, with equal results.
    @testset "families and evidence: $(name)" for (name, extra, lik, data) in (
            ("Binomial with trials", nothing, :(y .~ Binomial.(n, logistic.(eta))),
                rng -> Dict{Symbol,Any}(:n => fill(5, 120), :y => rand(rng, 0:5, 120))),
            ("NegativeBinomial2", :(phi ~ Exponential(1)),
                :(y .~ NegativeBinomial2.(exp.(eta), phi)),
                rng -> Dict{Symbol,Any}(:y => rand(rng, 0:9, 120))),
            ("truncated Normal", :(sigma ~ Exponential(1)),
                :(y .~ truncated.(Normal.(eta, sigma), lo, Inf)),
                rng -> (y = randn(rng, 120); Dict{Symbol,Any}(:y => y, :lo => y .- 1.5))),
            ("interval censored Normal", :(sigma ~ Exponential(1)),
                :(y .~ interval_censored.(Normal.(eta, sigma), hi)),
                rng -> (y = randn(rng, 120); Dict{Symbol,Any}(:y => y, :hi => y .+ 1.0))),
            ("censored counts", nothing,
                :(y .~ censored.(Poisson.(exp.(eta)), 0, hi)),
                rng -> Dict{Symbol,Any}(:y => rand(rng, 0:3, 120), :hi => fill(3, 120))),
            ("absent responses", :(sigma ~ Exponential(1)),
                :(y .~ Normal.(eta, sigma)),
                rng -> Dict{Symbol,Any}(:y => Union{Missing,Float64}[
                    isodd(i) ? missing : randn(rng) for i in 1:120])))
        rng = MersenneTwister(21)
        N, J = 120, 6
        x = randn(rng, N); g = [mod1(i, J) for i in 1:N]
        model = quote
            a ~ Normal(0, 2)
            b ~ Normal(0, 1)
            sd ~ Exponential(1)
            z[1:J] .~ Normal.(0, 1)
            eta = a .+ b .* x .+ sd .* z[g]
            $lik
        end
        extra === nothing || insert!(model.args, 1, extra)
        d = merge(Dict{Symbol,Any}(:x => x, :g => g, :J => J), data(rng))
        built, bound = _prv_build(model, d, Tuple(keys(d)))
        report = _prv_compare(built, bound)
        @info "plate reverse: $name" report
    end

    # The rule has no forward branch of its own: forward mode, and so reverse
    # over forward, differentiates the plate program as before.
    @testset "Hessian-vector products" begin
        rng = MersenneTwister(16)
        N, J = 150, 6
        x = randn(rng, N); g = [mod1(i, J) for i in 1:N]
        y = 1 .+ 0.5 .* x .+ randn(rng, N)
        built, bound = _prv_build(quote
            a ~ Normal(0, 5)
            b ~ Normal(0, 1)
            sd ~ Exponential(1)
            sigma ~ Exponential(1)
            z[1:J] .~ Normal.(0, 1)
            mu = a .+ b .* x .+ sd .* z[g]
            y .~ Normal.(mu, sigma)
        end, Dict(:y => y, :x => x, :g => g, :J => J), (:y, :x, :g, :J))
        n = built.layout.total
        u = 0.3 .* randn(MersenneTwister(3), n)
        v = randn(MersenneTwister(4), n)
        q = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
        so = SecondOrder(AutoEnzyme(; mode = Enzyme.Reverse), AutoEnzyme(; mode = Enzyme.Forward))
        h0 = with_plate_reverse(false) do
            prepare_ad_hvp(q.kernel, so, (v,), u; active = :unconstrained)
        end
        h1 = with_plate_reverse(true) do
            prepare_ad_hvp(q.kernel, so, (v,), u; active = :unconstrained)
        end
        (hv0,) = ad_hvp(h0, (v,), u)
        (hv1,) = ad_hvp(h1, (v,), u)
        @test isapprox(hv1, hv0; rtol = 1e-9, atol = 1e-12)
        gr0, (hg0,) = ad_gradient_and_hvp(h0, (v,), u)
        gr1, (hg1,) = ad_gradient_and_hvp(h1, (v,), u)
        @test isapprox(gr1, gr0; rtol = 1e-9, atol = 1e-12)
        @test isapprox(hg1, hg0; rtol = 1e-9, atol = 1e-12)
    end

    @testset "off by default" begin
        rng = MersenneTwister(15)
        N, J = 50, 5
        x = randn(rng, N); g = [mod1(i, J) for i in 1:N]; y = randn(rng, N)
        built, bound = _prv_build(quote
            a ~ Normal(0, 5)
            sd ~ Exponential(1)
            sigma ~ Exponential(1)
            z[1:J] .~ Normal.(0, 1)
            mu = a .+ sd .* z[g]
            y .~ Normal.(mu, sigma)
        end, Dict(:y => y, :x => x, :g => g, :J => J), (:y, :x, :g, :J))
        u = zeros(built.layout.total)
        q = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
        @test ReactiveKernels._plate_reverse_enabled() === false
        # Without the switch, no rule enters the AD kernel.
        @test !occursin("_PlateSumRule", string(typeof(q)))
    end
end
