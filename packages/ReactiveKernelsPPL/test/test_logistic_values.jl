# `logistic` is the inverse logit wherever a value is computed: undotted over a
# scalar, as `exp` is, and dotted over an array. A scalar definition, a scalar
# times a dotted vector, a distribution argument, a prior argument, a
# data-only definition, a plate cell and a recurrence all read it as a value,
# and a count rate or mean is any value, inline as named. Densities and Enzyme
# reverse gradients match Distributions oracles. Self-contained: runs
# standalone.
using DifferentiationInterface
using Distributions: Bernoulli, Binomial, NegativeBinomial, Normal, Poisson,
    logpdf
using Enzyme
using LogExpFunctions: logistic
using ReactiveKernels
using ReactiveKernelsPPL
using Test

const _LV_X = [-1.2, -0.4, 0.1, 0.6, 1.3, 0.0, 0.8, -0.7]
const _LV_POS = [0.5, 1.5, 2.0, 0.3, 1.1, 0.9, 2.5, 1.7]
const _LV_Y = [0, 1, 1, 0, 1, 0, 1, 1]
const _LV_W = [0, 2, 1, 0, 3, 1, 4, 2]
const _LV_C = [0, 5, 0, 2, 4, 1, 3, 1]
const _LV_G = [1, 2, 3, 1, 2, 3, 1, 2]

function _lv_model(ast, data; conditioned)
    values = Dict{Symbol,Any}(pairs(data))
    plan = bind_data(lower_rkppl(ast, values; conditioned), values)
    built = build_kernel(plan)
    kernel = prepare_query(built, plan, :sampler)
    return (; plan, built, kernel)
end

_lv_value(f, u) = Base.invokelatest(f.kernel, u)

function _lv_gradient(f, u)
    ad = Base.invokelatest(prepare_ad, f.kernel,
        AutoEnzyme(; mode = Enzyme.Reverse), u; active = :unconstrained)
    return Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
        ad, similar(u), u)
end

function _lv_findiff(f, u)
    h = cbrt(eps(Float64))
    return [(f(u .+ [j == i ? h : 0.0 for j in eachindex(u)]) -
             f(u .- [j == i ? h : 0.0 for j in eachindex(u)])) / (2h)
        for i in eachindex(u)]
end

function _lv_check(f, oracle, points)
    for u in points
        @test _lv_value(f, u) ≈ oracle(u) rtol = 2e-12
        value, gradient = _lv_gradient(f, u)
        @test value ≈ oracle(u) rtol = 2e-12
        @test gradient ≈ _lv_findiff(oracle, u) rtol = 2e-5 atol = 2e-7
    end
end

_lv_normals(u) = sum(logpdf.(Normal(), u))

@testset "logistic is a scalar value function" begin
    @test :logistic in admitted_functions()

    # A scalar definition, as `p_egkg = logistic(logit_p)`.
    f = _lv_model(quote
        a ~ Normal(0, 1)
        p = logistic(a)
        y .~ Bernoulli.(p)
    end, (; y = _LV_Y); conditioned = (:y,))
    _lv_check(f, u -> _lv_normals(u) +
        sum(logpdf.(Bernoulli(logistic(u[1])), _LV_Y)), ([0.3], [-1.7], [2.4]))

    # A scalar times a dotted vector, as `e = logistic(z) .* exp.(r)`.
    f = _lv_model(quote
        z ~ Normal(0, 1)
        r[1:3] .~ Normal.(0, 1)
        e = logistic(z) .* exp.(r)
        rate = e[g]
        w .~ Poisson.(rate)
    end, (; w = _LV_W, g = _LV_G); conditioned = (:w,))
    _lv_check(f, u -> _lv_normals(u) +
        sum(logpdf.(Poisson.(logistic(u[1]) .* exp.(u[2:4][_LV_G])), _LV_W)),
        ([0.3, -0.2, 0.1, 0.4], [-1.1, 0.5, -0.3, 0.9]))

    # Beside data in a rate, as `Poisson(logistic(rho) * anlagen)`, named and
    # inline, and inline in a Binomial probability.
    rate_oracle(u) = _lv_normals(u) +
        sum(logpdf.(Poisson.(logistic(u[1]) .* _LV_POS), _LV_W))
    f = _lv_model(quote
        a ~ Normal(0, 1)
        rate = logistic(a) .* x
        w .~ Poisson.(rate)
    end, (; w = _LV_W, x = _LV_POS); conditioned = (:w,))
    _lv_check(f, rate_oracle, ([0.3], [-2.2]))
    f = _lv_model(quote
        a ~ Normal(0, 1)
        w .~ Poisson.(logistic(a) .* x)
    end, (; w = _LV_W, x = _LV_POS); conditioned = (:w,))
    _lv_check(f, rate_oracle, ([0.3], [-2.2]))
    f = _lv_model(quote
        a ~ Normal(0, 1)
        c .~ Binomial.(5, logistic(a))
    end, (; c = _LV_C); conditioned = (:c,))
    _lv_check(f, u -> _lv_normals(u) +
        sum(logpdf.(Binomial(5, logistic(u[1])), _LV_C)), ([0.3], [-0.8]))

    # A count rate or mean is any value, inline as named (rkppl-use §9): a
    # gather, an n-ary product, a NegativeBinomial2 mean, a product of a
    # link and a sum, a dotted function that is not a link, a mixture
    # component's rate.
    f = _lv_model(quote
        z ~ Normal(0, 1)
        r[1:3] .~ Normal.(0, 1)
        e = logistic(z) .* exp.(r)
        w .~ Poisson.(e[g])
    end, (; w = _LV_W, g = _LV_G); conditioned = (:w,))
    _lv_check(f, u -> _lv_normals(u) +
        sum(logpdf.(Poisson.(logistic(u[1]) .* exp.(u[2:4][_LV_G])), _LV_W)),
        ([0.3, -0.2, 0.1, 0.4],))
    f = _lv_model(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        w .~ Poisson.(.*(0.25, logistic(a), exp.(b .* x) .* p))
    end, (; w = _LV_W, x = _LV_X, p = _LV_POS); conditioned = (:w,))
    _lv_check(f, u -> _lv_normals(u) + sum(logpdf.(Poisson.(0.25 .*
        logistic(u[1]) .* exp.(u[2] .* _LV_X) .* _LV_POS), _LV_W)),
        ([0.3, 0.2], [-1.0, -0.4]))
    nb2(mu, phi) = NegativeBinomial(phi, phi / (phi + mu))
    f = _lv_model(quote
        a ~ Normal(0, 1)
        s ~ Normal(0, 1)
        w .~ NegativeBinomial2.(logistic(a) .* x, exp(s))
    end, (; w = _LV_W, x = _LV_POS); conditioned = (:w,))
    _lv_check(f, u -> _lv_normals(u) +
        sum(logpdf.(nb2.(logistic(u[1]) .* _LV_POS, exp(u[2])), _LV_W)),
        ([0.3, 0.2], [-0.7, 1.1]))
    f = _lv_model(quote
        d ~ Normal(0, 1)
        o ~ Normal(0, 1)
        s ~ Normal(0, 1)
        w .~ NegativeBinomial2.(exp.(d) .* .+(exp.(o) .* x, 0.5 .* x), exp(s))
    end, (; w = _LV_W, x = _LV_POS); conditioned = (:w,))
    _lv_check(f, u -> _lv_normals(u) + sum(logpdf.(nb2.(exp(u[1]) .*
        (exp(u[2]) .* _LV_POS .+ 0.5 .* _LV_POS), exp(u[3])), _LV_W)),
        ([0.3, -0.4, 0.2],))
    f = _lv_model(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        w .~ Poisson.(abs.(eta) .+ 0.1)
    end, (; w = _LV_W, x = _LV_X); conditioned = (:w,))
    _lv_check(f, u -> _lv_normals(u) + sum(logpdf.(Poisson.(
        abs.(u[1] .+ u[2] .* _LV_X) .+ 0.1), _LV_W)), ([0.3, 0.2],))
    f = _lv_model(quote
        a ~ Normal(0, 1)
        w .~ MixtureModel.(vcat.(Poisson.(logistic(a) .* x), Poisson.(1.0)),
            Ref([0.3, 0.7]))
    end, (; w = _LV_W, x = _LV_POS); conditioned = (:w,))
    _lv_check(f, u -> _lv_normals(u) + sum(log.(0.3 .* exp.(logpdf.(
        Poisson.(logistic(u[1]) .* _LV_POS), _LV_W)) .+
        0.7 .* exp.(logpdf.(Poisson(1.0), _LV_W)))), ([0.3], [-1.6]))
    # An undotted inverse link over a scalar is the dotted link's value.
    f = _lv_model(quote
        a ~ Normal(0, 1)
        w .~ Poisson.(exp(a))
    end, (; w = _LV_W); conditioned = (:w,))
    _lv_check(f, u -> _lv_normals(u) + sum(logpdf.(Poisson(exp(u[1])), _LV_W)),
        ([0.3],))

    # A prior argument.
    f = _lv_model(quote
        a ~ Normal(0, 1)
        b ~ Normal(logistic(a), 1)
        y .~ Normal.(b, 1)
    end, (; y = _LV_X); conditioned = (:y,))
    _lv_check(f, u -> logpdf(Normal(), u[1]) +
        logpdf(Normal(logistic(u[1]), 1), u[2]) +
        sum(logpdf.(Normal(u[2], 1), _LV_X)), ([0.3, 0.1], [-1.4, 0.7]))

    # A data-only definition is evaluated once, at binding.
    f = _lv_model(quote
        a ~ Normal(0, 1)
        q = logistic(s)
        y .~ Normal.(a + q, 1)
    end, (; y = _LV_X, s = 0.4); conditioned = (:y,))
    _lv_check(f, u -> _lv_normals(u) +
        sum(logpdf.(Normal(u[1] + logistic(0.4), 1), _LV_X)), ([0.3],))

    # A plate cell, through a cell local and inline in the observation.
    cell_oracle(u) = _lv_normals(u) +
        sum(logpdf.(Bernoulli.(logistic.(u[1] .+ u[2] .* _LV_X)), _LV_Y))
    f = _lv_model(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        @plate for i in eachindex(y)
            p = logistic(a + b * x[i])
            y[i] ~ Bernoulli(p)
        end
    end, (; y = _LV_Y, x = _LV_X); conditioned = (:y,))
    _lv_check(f, cell_oracle, ([0.3, -0.5], [1.2, 0.8]))
    f = _lv_model(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        @plate for i in eachindex(y)
            y[i] ~ Bernoulli(logistic(a + b * x[i]))
        end
    end, (; y = _LV_Y, x = _LV_X); conditioned = (:y,))
    _lv_check(f, cell_oracle, ([0.3, -0.5],))

    # A scalar probability under `.~` broadcasts, as in Julia, also as a
    # mixture component.
    f = _lv_model(quote
        a ~ Normal(0, 1)
        y .~ Bernoulli.(logistic(a + 0.5))
    end, (; y = _LV_Y); conditioned = (:y,))
    _lv_check(f, u -> _lv_normals(u) +
        sum(logpdf.(Bernoulli(logistic(u[1] + 0.5)), _LV_Y)), ([0.3], [-1.1]))
    f = _lv_model(quote
        a ~ Normal(0, 1)
        y .~ MixtureModel.(vcat.(Bernoulli.(logistic(a)), Bernoulli.(0.2)),
            Ref([0.4, 0.6]))
    end, (; y = _LV_Y); conditioned = (:y,))
    _lv_check(f, u -> _lv_normals(u) + sum(y -> log(0.4 *
        exp(logpdf(Bernoulli(logistic(u[1])), y)) +
        0.6 * exp(logpdf(Bernoulli(0.2), y))), _LV_Y), ([0.3], [-0.9]))

    # A recurrence's seeds and steps, at top level and in a plate cell.
    function scan_path(a, n)
        h = zeros(n)
        for t in 2:n
            h[t] = logistic(h[t - 1] + a)
        end
        return h
    end
    f = _lv_model(quote
        a ~ Normal(0, 1)
        @scan begin
            h[1] = logistic(a) - logistic(a)
            for t in 2:T
                h[t] = logistic(h[t - 1] + a)
            end
        end
        y .~ Normal.(h, 1)
    end, (; y = _LV_X, T = 8); conditioned = (:y,))
    _lv_check(f, u -> _lv_normals(u) +
        sum(logpdf.(Normal.(scan_path(u[1], 8), 1), _LV_X)), ([0.3], [-1.3]))
    groups = [_LV_X[1:3], _LV_X[4:8]]
    f = _lv_model(quote
        a ~ Normal(0, 1)
        @plate for g in eachindex(ys)
            @scan begin
                h[1] = 0.0
                for t in 2:length(ys[g])
                    h[t] = logistic(h[t - 1] + a)
                end
            end
            loc[g] = h
            ys[g] .~ Normal.(loc[g], 1)
        end
    end, (; ys = groups); conditioned = (:ys,))
    _lv_check(f, u -> _lv_normals(u) + sum(ys -> sum(logpdf.(
        Normal.(scan_path(u[1], length(ys)), 1), ys)), groups), ([0.3],))

    # The value is the stable inverse logit: a finite gradient at a large
    # argument, where Enzyme through `LogExpFunctions.logistic` returns NaN
    # (reactivekernels-use §7ac).
    f = _lv_model(quote
        a ~ Normal(0, 1)
        rate = logistic(a) .* x
        w .~ Poisson.(rate)
    end, (; w = _LV_W, x = _LV_POS); conditioned = (:w,))
    value, gradient = _lv_gradient(f, [1000.0])
    @test isfinite(value) && all(isfinite, gradient)
    @test gradient ≈ [-1000.0] rtol = 1e-12

    # refused: undotted elementwise math over an array is a Julia MethodError,
    # as `exp(eta)` is (P3)
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
        p = logistic(eta)
        y .~ Bernoulli.(p)
    end, (:y, :x); conditioned = (:y,))
    # refused: an undotted inverse link over a vector in a probability is a
    # Julia MethodError, though the dotted link is the same program (P3)
    for (resp, rhs) in ((:y, :(Bernoulli.(logistic(eta)))),
            (:y, :(Bernoulli.(exp(eta)))), (:c, :(Binomial.(5, normcdf(eta)))),
            (:w, :(ZeroInflatedPoisson.(exp.(eta), logistic(eta)))),
            (:w, :(Poisson.(exp(eta)))),
            (:w, :(NegativeBinomial2.(exp(eta), 2.0))))
        err = try
            lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                eta = a .+ b .* x
                $resp .~ $rhs
            end, (resp, :x); conditioned = (resp,))
            nothing
        catch e
            e
        end
        @test err isa SurfaceLoweringError
        @test occursin("over a vector needs dots", sprint(showerror, err))
    end
end

# A location summand computed from a `logistic` value is a scalar, as one
# computed from `exp` is: named, inline, undotted, beside a product
# definition and from data. Binding failed with `term references missing
# column _rkppl_leaf_1` (snag rkppl-leaf-colum-59c22941).
@testset "a logistic value in a location summand" begin
    xs = [-0.6, 0.1, 0.8, 1.4]
    rows = [1, 3, 2, 4, 3]
    y = [0.9, 1.6, 1.1, 1.8, 1.4]
    data = (; x = xs, rows, y)
    points = ([0.3, -0.4], [-1.1, 0.8])
    located(loc) = u -> _lv_normals(u) + sum(logpdf.(Normal.(loc(u...), 0.7), y))
    p(a) = logistic.(a .+ logistic.(xs)[rows])
    e(a, b) = logistic(b) * exp(a)

    # BRM's emission: a vector definition plus a product reading a product
    # definition, the factor named, inline or a dotted `exp`.
    for (factor, value) in ((:(exp.(a)), exp), (:s, logistic),
            (:(logistic.(a)), logistic))
        f = _lv_model(quote
            q = logistic.(x)
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            p = logistic.(a .+ q[rows])
            e = logistic.(b) .* exp.(a)
            s = logistic.(a)
            y .~ Normal.(p .+ e .* $factor, 0.7)
        end, data; conditioned = (:y,))
        _lv_check(f, located((a, b) -> p(a) .+ e(a, b) * value(a)), points)
    end

    xr = xs[rows]
    for (loc, oracle) in (
            (:(x[rows] .+ e), (a, b) -> xr .+ e(a, b)),
            (:(x[rows] .+ logistic.(a)), (a, b) -> xr .+ logistic(a)),
            (:(x[rows] .+ b .* logistic.(a)), (a, b) -> xr .+ b * logistic(a)),
            (:(x[rows] .+ 2 .* logistic(a)), (a, b) -> xr .+ 2 * logistic(a)),
            (:(x[rows] .+ b .* m), (a, b) -> xr .+ b * logistic(sum(xs) / 4)))
        f = _lv_model(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            e = logistic.(b) .* exp.(a)
            m = logistic(mean(x))
            y .~ Normal.($loc, 0.7)
        end, data; conditioned = (:y,))
        _lv_check(f, located(oracle), points)
    end
end
