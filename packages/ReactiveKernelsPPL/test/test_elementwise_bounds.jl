using Test, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
import Distributions as D
using LogExpFunctions: log1pexp, logistic

# One bound per element of a declared array (snag `rkppl-per-elemen-41559f02`):
# `x[1:n] .~ restricted.(D.(…), lo, hi)`, `truncated.(…)` and `Uniform.(lo, hi)`
# with `lo`/`hi` arrays of the declared shape, as Stan's
# `vector<lower=L, upper=U>[n]`. Each element has its own transform and
# Jacobian; the program reads the bounds as data, so one program serves every
# length and every set of bound values.

# The packed coordinates of entry `name` in `u`.
_coords(built, u, name) = (e = only(filter(e -> e.name === name, built.layout.entries));
    u[e.offset:(e.offset + e.size - 1)])

# Every binding shares one program; for each one, the host transforms invert
# each other, the sampler's value and Enzyme reverse gradient equal an
# independent oracle `want(u, built, data)` (explicit transforms and Jacobians,
# not the layout's host functions) and its finite differences, and the
# bound data are unchanged.
function _elementwise_check(source, datasets, values_of, want; conditioned = (),
        same_program = true)
    plan = lower_rkppl(source, first(datasets); conditioned)
    bound = [bind_data(plan, d) for d in datasets]
    builds = build_kernel.(bound)
    programs = [Base.remove_linenums!(kernel_expr(b, built.layout))
        for (b, built) in zip(bound, builds)]
    same_program && @test all(==(first(programs)), programs)
    for (b, built, d) in zip(bound, builds, datasets)
        before = deepcopy(d)
        values = values_of(d)
        u = unconstrain(built.layout, values)
        restored = constrain(built.layout, u)
        for name in keys(values)
            @test all(isapprox.(restored[name], values[name]; atol = 1e-12, rtol = 1e-12))
        end
        oracle(z) = want(z, built, d)
        @test logjac(built.layout, u) ≈ want(u, built, d; jacobian_only = true) atol = 1e-12
        query = prepare_sampler(built, b, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(query, similar(u), u)
        @test value ≈ oracle(u) atol = 1e-11 rtol = 1e-11
        finite = map(eachindex(u)) do i
            up, down = copy(u), copy(u)
            up[i] += 1e-5
            down[i] -= 1e-5
            (oracle(up) - oracle(down)) / 2e-5
        end
        @test gradient ≈ finite atol = 2e-7 rtol = 2e-6
        @test d == before
    end
    return (; plan, bound, builds, program = first(programs))
end

# Explicit per-element transforms and their log-Jacobians.
_upper(hi, u) = hi .- exp.(u), sum(u; init = 0.0)
_lower(lo, u) = lo .+ exp.(u), sum(u; init = 0.0)
_interval(lo, hi, u) = lo .+ (hi .- lo) .* logistic.(u),
    sum(log.(hi .- lo) .- log1pexp.(-u) .- log1pexp.(u); init = 0.0)

@testset "censored rows: an upper bound per element from data" begin
    # Data augmentation of right-censored rows: each latent value lies below
    # its own row's limit, under the observation model's own density.
    source = quote
        mu ~ Normal(0, 1)
        latent[1:n] .~ restricted.(Normal.(mu, 1), -Inf, limit)
    end
    datasets = [(n = 0, limit = Float64[]), (n = 3, limit = [0.5, 1.0, -0.3]),
        (n = 9, limit = collect(range(-1.0, 2.0; length = 9)))]
    function want(u, built, d; jacobian_only = false)
        mu = only(_coords(built, u, :mu))
        x, jac = _upper(d.limit, _coords(built, u, :latent))
        jacobian_only && return jac
        D.logpdf(D.Normal(), mu) + sum(D.logpdf.(D.Normal(mu, 1), x); init = 0.0) + jac
    end
    r = _elementwise_check(source, datasets,
        d -> (mu = 0.2, latent = d.limit .- 0.25 .- 0.1 .* (1:d.n)), want)
    # The program reads the limits as data, never this binding's numbers.
    @test !occursin("-0.3", string(r.program))
    # The same density as one scalar declaration per element.
    scalar = quote
        mu ~ Normal(0, 1)
        l1 ~ restricted(Normal(mu, 1), -Inf, 0.5)
        l2 ~ restricted(Normal(mu, 1), -Inf, 1.0)
        l3 ~ restricted(Normal(mu, 1), -Inf, -0.3)
    end
    sb = bind_data(lower_rkppl(scalar, (;)), (;))
    sbuilt = build_kernel(sb)
    su = unconstrain(sbuilt.layout, (mu = 0.2, l1 = 0.1, l2 = 0.4, l3 = -0.8))
    vb = r.bound[2]
    vbuilt = r.builds[2]
    vu = unconstrain(vbuilt.layout, (mu = 0.2, latent = [0.1, 0.4, -0.8]))
    @test vu ≈ su atol = 1e-14
    sq = prepare_sampler(sbuilt, sb, su; backend = AutoEnzyme(; mode = Enzyme.Reverse))
    vq = prepare_sampler(vbuilt, vb, vu; backend = AutoEnzyme(; mode = Enzyme.Reverse))
    sv, sg = sampler_value_and_gradient!(sq, similar(su), su)
    vv, vg = sampler_value_and_gradient!(vq, similar(vu), vu)
    @test vv ≈ sv atol = 1e-12
    @test vg ≈ sg atol = 1e-12
end

@testset "interval rows: both bounds per element, data and definitions" begin
    # Interval-censored (rounded) rows: each latent value lies in its own
    # [lo, hi]. `restricted.(Flat.(), …)` contributes only the Jacobian.
    datasets = [(n = 0, lo = Float64[], hi = Float64[], y = Float64[]),
        (n = 3, lo = [0.0, -1.0, 2.0], hi = [1.0, 0.5, 3.5], y = [0.4, -0.2, 2.9]),
        (n = 5, lo = [-2.0, 0.0, 0.1, 1.0, 4.0], hi = [-1.0, 0.25, 0.2, 3.0, 9.0],
            y = [-1.2, 0.1, 0.3, 2.0, 5.0])]
    values_of(d) = (s = 0.7, position = d.lo .+ 0.3 .* (d.hi .- d.lo))
    function want(u, built, d; jacobian_only = false)
        s = exp(only(_coords(built, u, :s)))
        x, jac = _interval(d.lo, d.hi, _coords(built, u, :position))
        jac += only(_coords(built, u, :s))
        jacobian_only && return jac
        D.logpdf(D.Exponential(1), s) + sum(D.logpdf.(D.Normal.(x, s), d.y); init = 0.0) + jac
    end
    _elementwise_check(quote
        s ~ Exponential(1)
        position[1:n] .~ restricted.(Flat.(), lo, hi)
        y .~ Normal.(position, s)
    end, datasets, values_of, want; conditioned = (:y,))
    # Data-only definitions are bounds too: they fold at binding and the
    # program computes them from the data.
    shifted(d) = (n = d.n, lo = d.lo .- 1, hi = d.hi .+ 1, y = d.y)
    _elementwise_check(quote
        s ~ Exponential(1)
        lower = lo .+ 1
        upper = hi .- 1
        position[1:n] .~ restricted.(Flat.(), lower, upper)
        y .~ Normal.(position, s)
    end, shifted.(datasets), d -> values_of((lo = d.lo .+ 1, hi = d.hi .- 1)),
        (u, built, d; kw...) -> want(u, built, (lo = d.lo .+ 1, hi = d.hi .- 1, y = d.y); kw...);
        conditioned = (:y,))
end

@testset "infinite entries leave an element unbounded on that side" begin
    # Left-censored, right-censored, interval-censored and unbounded rows in
    # one latent array: each element takes the transform its own bounds
    # state. One program serves every pattern of infinite entries, an empty
    # array and an all-finite one included.
    datasets = [
        (n = 4, lo = [-Inf, 0.5, -1.0, -Inf], hi = [0.0, Inf, 1.0, Inf], y = [-0.4, 0.9, 0.2, 2.0]),
        (n = 3, lo = [-2.0, 0.0, 1.0], hi = [-1.0, 0.25, 3.0], y = [-1.5, 0.1, 2.0]),
        (n = 0, lo = Float64[], hi = Float64[], y = Float64[]),
        (n = 2, lo = [-Inf, -Inf], hi = [Inf, Inf], y = [0.3, -0.3])]
    # Stan's `lub_constrain` per element.
    function bounded(lo, hi, u)
        isfinite(lo) && isfinite(hi) && return lo + (hi - lo) * logistic(u),
            log(hi - lo) - log1pexp(-u) - log1pexp(u)
        isfinite(lo) && return lo + exp(u), u
        isfinite(hi) && return hi - exp(u), u
        return u, 0.0
    end
    function want(u, built, d; jacobian_only = false)
        mu = only(_coords(built, u, :mu))
        cells = bounded.(d.lo, d.hi, _coords(built, u, :latent))
        x, jac = first.(cells), sum(last.(cells); init = 0.0)
        jacobian_only && return jac
        D.logpdf(D.Normal(), mu) + sum(D.logpdf.(D.Normal(mu, 2), x); init = 0.0) +
            sum(D.logpdf.(D.Normal.(x, 0.5), d.y); init = 0.0) + jac
    end
    inside(lo, hi) = isfinite(lo) && isfinite(hi) ? (lo + hi) / 2 :
        isfinite(lo) ? lo + 0.7 : isfinite(hi) ? hi - 0.7 : 0.4
    r = _elementwise_check(quote
        mu ~ Normal(0, 1)
        latent[1:n] .~ restricted.(Normal.(mu, 2), lo, hi)
        y .~ Normal.(latent, 0.5)
    end, datasets, d -> (mu = 0.1, latent = inside.(d.lo, d.hi)), want; conditioned = (:y,))
    # Each element's coordinate is the scalar declaration's.
    layout = r.builds[1].layout
    @test unconstrain(layout, (mu = 0.1, latent = [-0.3, 1.5, 0.25, 0.8])) ≈
        [0.1, log(0.3), log(1.0), log(1.25) - log(0.75), 0.8] atol = 1e-14
end

@testset "truncated and Uniform priors per element" begin
    datasets = [(n = 0, lo = Float64[], hi = Float64[]),
        (n = 3, lo = [0.0, -1.0, 2.0], hi = [1.0, 0.5, 3.5])]
    # Infinite entries leave that side of the element open; the normalizer
    # takes its one-sided mass there, never reading the infinite endpoint.
    open = [(n = 4, lo = [-Inf, 0.0, 1.0, -Inf], hi = [0.5, Inf, 2.0, Inf])]
    function bounded(lo, hi, u)
        isfinite(lo) && isfinite(hi) && return lo + (hi - lo) * logistic(u),
            log(hi - lo) - log1pexp(-u) - log1pexp(u)
        isfinite(lo) && return lo + exp(u), u
        isfinite(hi) && return hi - exp(u), u
        return u, 0.0
    end
    function truncated_want(u, built, d; jacobian_only = false)
        mu = only(_coords(built, u, :mu))
        cells = bounded.(d.lo, d.hi, _coords(built, u, :x))
        x, jac = first.(cells), sum(last.(cells); init = 0.0)
        jacobian_only && return jac
        D.logpdf(D.Normal(), mu) + jac +
            sum(D.logpdf.(D.truncated.(D.Normal(mu, 1), d.lo, d.hi), x); init = 0.0)
    end
    inside(lo, hi) = isfinite(lo) && isfinite(hi) ? (lo + hi) / 2 :
        isfinite(lo) ? lo + 0.7 : isfinite(hi) ? hi - 0.7 : 0.4
    _elementwise_check(quote
        mu ~ Normal(0, 1)
        x[1:n] .~ truncated.(Normal.(mu, 1), lo, hi)
    end, [datasets; open], d -> (mu = 0.3, x = inside.(d.lo, d.hi)), truncated_want)
    function uniform_want(u, built, d; jacobian_only = false)
        x, jac = _interval(d.lo, d.hi, _coords(built, u, :x))
        jacobian_only && return jac
        sum(D.logpdf.(D.Uniform.(d.lo, d.hi), x); init = 0.0) + jac
    end
    _elementwise_check(quote
        x[1:n] .~ Uniform.(lo, hi)
    end, datasets, d -> (x = d.lo .+ 0.2 .* (d.hi .- d.lo),), uniform_want)
    # A lower bound per element on a positive family intersects its support.
    function positive_want(u, built, d; jacobian_only = false)
        x, jac = _lower(max.(d.lo, 0.0), _coords(built, u, :x))
        jacobian_only && return jac
        sum(D.logpdf.(D.Exponential(2), x); init = 0.0) + jac
    end
    _elementwise_check(quote
        x[1:n] .~ restricted.(Exponential.(2), lo, Inf)
    end, [(n = 3, lo = [-1.0, 0.5, 2.0])], d -> (x = [0.3, 0.9, 2.5],), positive_want)
end

@testset "two-axis arrays take a bound matrix of their size" begin
    datasets = [(L = [0.0 1.0 -2.0; 0.5 0.25 3.0], U = [1.0 2.0 0.0; 1.5 4.0 3.5]),
        (L = zeros(2, 0), U = ones(2, 0))]
    function want(u, built, d; jacobian_only = false)
        x, jac = _interval(vec(d.L), vec(d.U), _coords(built, u, :B))
        jacobian_only && return jac
        sum(D.logpdf.(D.truncated.(D.Normal(0, 2), vec(d.L), vec(d.U)), x); init = 0.0) + jac
    end
    _elementwise_check(quote
        B[axes(L, 1), axes(L, 2)] .~ truncated.(Normal.(0, 2), L, U)
    end, datasets, d -> (B = (d.L .+ d.U) ./ 2,), want)
    function floor_want(u, built, d; jacobian_only = false)
        x, jac = _lower(vec(d.L), _coords(built, u, :B))
        jacobian_only && return jac
        jac
    end
    _elementwise_check(quote
        B[axes(L, 1), axes(L, 2)] .~ restricted.(Flat.(), L, Inf)
    end, datasets[1:1], d -> (B = d.L .+ 0.5,), floor_want)
end

@testset "a bound per element computed from parameters" begin
    # A graph value with the array's axes: the bound moves with `a`.
    datasets = [(n = 3, limit = [0.5, 1.0, -0.3]), (n = 0, limit = Float64[])]
    function want(u, built, d; jacobian_only = false)
        a = only(_coords(built, u, :a))
        x, jac = _upper(d.limit .+ exp(a), _coords(built, u, :x))
        jacobian_only && return jac
        D.logpdf(D.Normal(), a) + sum(D.logpdf.(D.Normal(0, 3), x); init = 0.0) + jac
    end
    _elementwise_check(quote
        a ~ Normal(0, 1)
        top = limit .+ exp(a)
        x[1:n] .~ restricted.(Normal.(0, 3), -Inf, top)
    end, datasets, d -> (a = 0.1, x = d.limit .- 0.5), want)
end

@testset "a plate cell's own bounds" begin
    # `x[i] ~ restricted(D, lo[i], hi[i])` in a `@plate` bounds each cell by
    # its own row, infinite entries included, exactly as the declared array.
    datasets = [
        (lo = [0.0, -1.0, -Inf, -Inf], hi = [1.0, Inf, 2.0, Inf], y = [0.4, 0.3, 1.2, -0.5]),
        (lo = [-2.0, 0.5], hi = [-1.0, 3.0], y = [-1.5, 1.0]),
        (lo = Float64[], hi = Float64[], y = Float64[])]
    function bounded(lo, hi, u)
        isfinite(lo) && isfinite(hi) && return lo + (hi - lo) * logistic(u),
            log(hi - lo) - log1pexp(-u) - log1pexp(u)
        isfinite(lo) && return lo + exp(u), u
        isfinite(hi) && return hi - exp(u), u
        return u, 0.0
    end
    inside(lo, hi) = isfinite(lo) && isfinite(hi) ? (lo + hi) / 2 :
        isfinite(lo) ? lo + 0.7 : isfinite(hi) ? hi - 0.7 : 0.4
    for (wrapper, law) in ((:restricted, (lo, hi) -> D.Normal(0, 2)),
            (:truncated, (lo, hi) -> D.truncated(D.Normal(0, 2), lo, hi)))
        function want(u, built, d; jacobian_only = false)
            cells = bounded.(d.lo, d.hi, _coords(built, u, :x))
            x, jac = first.(cells), sum(last.(cells); init = 0.0)
            jacobian_only && return jac
            sum(D.logpdf.(law.(d.lo, d.hi), x); init = 0.0) +
                sum(D.logpdf.(D.Normal.(x, 1), d.y); init = 0.0) + jac
        end
        _elementwise_check(quote
            @plate for i in eachindex(y)
                x[i] ~ $wrapper(Normal(0, 2), lo[i], hi[i])
                y[i] ~ Normal(x[i], 1)
            end
        end, datasets, d -> (x = inside.(d.lo, d.hi),), want; conditioned = (:y,),
            # A plate over a response's rows packs a literal range per binding,
            # bounds or not (`view(unconstrained, 1:4)`): a separate limitation.
            same_program = false)
    end
end

@testset "per-element bounds are refused where they do not apply" begin
    refusal(source, data; conditioned = ()) = try
        build_kernel(bind_data(lower_rkppl(source, data; conditioned), data))
        nothing
    catch e
        e isa Union{ContractValidationError,SurfaceLoweringError} || rethrow()
        sprint(showerror, e)
    end
    # The bound array has the declared shape.
    msg = refusal(quote
        x[1:n] .~ restricted.(Flat.(), -Inf, limit)
    end, (n = 3, limit = [0.0, 1.0]))
    @test msg !== nothing && occursin("needs the declared size (3,)", msg)
    # A vector per row of a two-axis array is Julia's broadcast, not built.
    msg = refusal(quote
        B[1:2, 1:3] .~ restricted.(Flat.(), lo, Inf)
    end, (lo = [0.0, 1.0],))
    @test msg !== nothing && occursin("needs the declared size (2, 3)", msg)
    # A scalar parameter takes one bound: a vector of data is no scalar name.
    msg = refusal(quote
        x ~ restricted(Flat(), -Inf, limit)
    end, (limit = [0.0, 1.0],))
    @test msg !== nothing && occursin("unknown name limit", msg)
    # A cell's bound is its own row's (`upper[i]`), never a vector read whole.
    msg = refusal(quote
        @plate for i in eachindex(y)
            x[i] ~ restricted(Flat(), -Inf, upper)
            y[i] ~ Normal(x[i], 1)
        end
    end, (y = [0.2, 0.4], upper = [1.0, 2.0]); conditioned = (:y,))
    @test msg !== nothing && occursin("bare `upper` reads a whole vector in a cell", msg)
    # Lower bounds below upper bounds, element by element.
    msg = refusal(quote
        x[1:n] .~ restricted.(Flat.(), lo, hi)
    end, (n = 2, lo = [0.0, 2.0], hi = [1.0, 1.0]))
    @test msg !== nothing && occursin("no mass", msg)
end
