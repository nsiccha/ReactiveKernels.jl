function _scan_cap_recurrence(n, kind; lag = 2)
    seeds, loop, location = if kind === :lag
        (:(h[1] = 0.1; h[2] = 0.2), :(for t in 3:T
            e ~ Normal(0, 1)
            h[t] = phi * h[t - k] + x[t] + e
        end), :h)
    elseif kind === :mixed
        (:(h[1] ~ Normal(0, 1); g[1] = 0.0), :(for t in 2:T
            h[t] ~ Normal(phi * h[t - 1], 1)
            g[t] = g[t - 1] + h[t]
        end), :g)
    elseif kind === :data
        (:(h[1] = x[1]), :(for t in 2:T
            h[t] = phi * h[t - 1] + x[t]
        end), :h)
    elseif kind === :volatile
        (:(h[1] ~ Normal(0, 1)), :(for t in 2:T
            e ~ Normal(0, exp(h[t - 1]))
            h[t] = phi * h[t - 1] + e
        end), :h)
    else
        (:(h[1] ~ Normal(0, 1)), :(for t in 2:T
            e ~ Normal(0, 1)
            h[t] = phi * h[t - 1] + t * e
        end), :h)
    end
    fills = seeds.head === :block ? seeds.args : Any[seeds]
    block = Expr(:block, fills..., loop)
    program = quote
        phi ~ Normal(0, 1)
        @scan $block
        y .~ Normal.($location, 1.0)
    end
    data = Dict{Symbol,Any}(:y => [0.3sin(t) for t in 1:n], :T => n)
    kind in (:data, :lag) && (data[:x] = [0.2cos(t) for t in 1:(n + 2)])
    kind === :lag && (data[:k] = lag)
    snapshot = deepcopy(data)
    f = _scan_cap_sampler(program, data)
    pe = only(e for e in f.built.layout.entries if e.name === :phi)
    zs = [e for e in f.built.layout.entries if e.kind === :scan]
    function oracle(u)
        phi = u[pe.offset]
        z = isempty(zs) ? Float64[] : u[only(zs).offset:(only(zs).offset + only(zs).size - 1)]
        lp = logpdf(Normal(), phi)
        h = kind === :data ? data[:x][1] : kind === :lag ? 0.2 : z[1]
        values = kind === :lag ? [0.1, 0.2] : [h]
        g = 0.0
        kind in (:data, :lag) || (lp += logpdf(Normal(), h))
        for t in 1:n
            if t >= (kind === :lag ? 3 : 2)
                if kind === :data
                    h = phi * h + data[:x][t]
                elseif kind === :lag
                    e = z[t - 2]
                    lp += logpdf(Normal(), e)
                    h = phi * values[t - lag] + data[:x][t] + e
                    push!(values, h)
                elseif kind === :mixed
                    next_h = z[t]
                    lp += logpdf(Normal(phi * h, 1), next_h)
                    h = next_h
                    g += h
                else
                    e = z[t]
                    lp += logpdf(Normal(0, kind === :volatile ? exp(h) : 1), e)
                    h = phi * h + (kind === :index ? t * e : e)
                end
            elseif kind === :lag
                h = values[t]
            end
            lp += logpdf(Normal(kind === :mixed ? g : h, 1), data[:y][t])
        end
        return lp
    end
    return f, oracle, data, snapshot
end

function _scan_cap_support(n, seed_family, innovation_family)
    spellings = Dict(:normal => :(Normal(0, 1)), :exponential => :(Exponential(1.3)),
        :beta => :(Beta(2, 3)), :gamma => :(Gamma(2, 0.7)), :uniform => :(Uniform(-1, 2)))
    distributions = Dict(:normal => Normal(), :exponential => Exponential(1.3),
        :beta => Beta(2, 3), :gamma => Gamma(2, 0.7), :uniform => Uniform(-1,2))
    program = quote
        phi ~ Normal(0, 1)
        @scan begin
            h[1] ~ $(spellings[seed_family])
            for t in 2:T
                e ~ $(spellings[innovation_family])
                h[t] = phi * h[t - 1] + e
            end
        end
        y .~ Normal.(h, 1.0)
    end
    data = Dict{Symbol,Any}(:y => [0.3sin(t) for t in 1:n])
    f = _scan_cap_sampler(program, data)
    pe = only(e for e in f.built.layout.entries if e.name === :phi)
    ze = only(e for e in f.built.layout.entries if e.kind === :scan)
    function transform(a, family)
        family === :normal && return a, 0.0
        family === :beta && return 1 / (1 + exp(-a)), -log1p(exp(-a)) - log1p(exp(a))
        family === :uniform && return -1 + 3 / (1 + exp(-a)), log(3) - log1p(exp(-a)) - log1p(exp(a))
        return exp(a), a
    end
    function oracle(u)
        phi = u[pe.offset]
        raw = u[ze.offset:(ze.offset + ze.size - 1)]
        h, jac = transform(raw[1], seed_family)
        lp = logpdf(Normal(), phi) + logpdf(distributions[seed_family], h) + jac
        for t in 1:n
            if t > 1
                e, jac = transform(raw[t], innovation_family)
                lp += logpdf(distributions[innovation_family], e) + jac
                h = phi * h + e
            end
            lp += logpdf(Normal(h, 1), data[:y][t])
        end
        return lp
    end
    return f, oracle
end

@testset "scan capabilities: data, indices, mixed writes and conditional innovations" begin
    for kind in (:data, :index, :mixed, :volatile, :lag)
        for n in (kind === :lag ? 2 : 1, 4)
            f, oracle, data, snapshot = _scan_cap_recurrence(n, kind)
            _scan_cap_check(f, oracle)
            @test data == snapshot
        end
    end
    f, oracle, data, snapshot = _scan_cap_recurrence(4, :lag; lag = 1)
    _scan_cap_check(f, oracle)
    @test data == snapshot
    for lag in (0, 3, true, 1.5)
        @test_throws ContractValidationError _scan_cap_recurrence(4, :lag; lag)
    end

end

@testset "scan capabilities: supported seed and innovation geometry" begin
    for (seed, innovation) in ((:normal, :exponential), (:exponential, :normal),
            (:exponential, :exponential), (:beta, :gamma), (:uniform, :uniform)), n in (1, 4)
        f, oracle = _scan_cap_support(n, seed, innovation)
        _scan_cap_check(f, oracle)
        draws = constrain(f.built.layout, f.u)
        @test unconstrain(f.built.layout, draws) ≈ f.u
        ze = only(e for e in f.built.layout.entries if e.kind === :scan)
        raw = f.u[ze.offset:(ze.offset + ze.size - 1)]
        expected = sum(enumerate(raw)) do (i, a)
            family = i == 1 ? seed : innovation
            family === :normal ? 0.0 : family === :beta ?
                -log1p(exp(-a)) - log1p(exp(a)) : family === :uniform ?
                log(3) - log1p(exp(-a)) - log1p(exp(a)) : a
        end
        @test logjac(f.built.layout, f.u) ≈ expected

    end
end

# Seeds and steps take every scalar prior a declaration takes: `Flat()`, half
# priors, and `restricted` / `truncated` with bounds fixed across steps.
# `:walk` is a bounded random walk beside a flat-seeded sum-to-zero state
# (tuple carry), `:centered` a single truncated state whose upper bound is
# sampled, `:innovation` a one-sided truncated innovation.
function _scan_cap_bounded(n, kind)
    data = Dict{Symbol,Any}(:y => [0.3sin(t) + 0.5 for t in 1:n], :T => n)
    program = if kind === :walk
        data[:lo], data[:hi] = -0.4, 1.3
        quote
            s ~ restricted(Normal(0, 1), 0, Inf)
            @scan begin
                h[1] ~ restricted(Flat(), lo, hi)
                h[2] ~ restricted(Normal(h[1], s), lo, hi)
                g[1] ~ Flat()
                g[2] ~ Flat()
                for t in 3:T
                    h[t] ~ restricted(Normal(h[t - 1], s), lo, hi)
                    g[t] ~ Normal(-(g[t - 1] + g[t - 2]), s)
                end
            end
            y .~ Normal.(h .+ g, 1.0)
        end
    elseif kind === :centered
        quote
            phi ~ Normal(0, 1)
            ub ~ restricted(Normal(3, 1), 1, Inf)
            @scan begin
                h[1] ~ HalfNormal(2)
                for t in 2:T
                    h[t] ~ truncated(Normal(phi * h[t - 1], 1), 0, ub)
                end
            end
            y .~ Normal.(h, 1.0)
        end
    else
        data[:lo] = -0.5
        quote
            phi ~ Normal(0, 1)
            sd ~ HalfNormal(1)
            @scan begin
                h[1] ~ HalfCauchy(1)
                for t in 2:T
                    e ~ truncated(Normal(0, sd), lo, Inf)
                    h[t] = phi * h[t - 1] + e
                end
            end
            y .~ Normal.(h, 1.0)
        end
    end
    f = _scan_cap_sampler(program, data)
    entry(name) = only(e for e in f.built.layout.entries if e.name === name)
    ze = only(e for e in f.built.layout.entries if e.kind === :scan)
    y = data[:y]
    interval(a, lo, hi) = lo + (hi - lo) / (1 + exp(-a)),
        log(hi - lo) - log1p(exp(-a)) - log1p(exp(a))
    function oracle(u; jacobian = false)
        z = u[ze.offset:(ze.offset + ze.size - 1)]
        lp, jac = 0.0, 0.0
        if kind === :walk
            lo, hi, m = data[:lo], data[:hi], n - 2
            rs = u[entry(:s).offset]
            s = exp(rs)
            lp += logpdf(Normal(), s)
            jac += rs
            h, g = zeros(n), zeros(n)
            for t in 1:n
                a = t <= 2 ? z[t] : z[4 + t - 2]
                h[t], j = interval(a, lo, hi)
                jac += j
                t == 1 || (lp += logpdf(Normal(h[t - 1], s), h[t]))
                g[t] = t <= 2 ? z[2 + t] : z[4 + m + t - 2]
                t <= 2 || (lp += logpdf(Normal(-(g[t - 1] + g[t - 2]), s), g[t]))
                lp += logpdf(Normal(h[t] + g[t], 1), y[t])
            end
        elseif kind === :centered
            phi, rub = u[entry(:phi).offset], u[entry(:ub).offset]
            ub = 1 + exp(rub)
            lp += logpdf(Normal(), phi) + logpdf(Normal(3, 1), ub)
            jac += rub
            h = exp(z[1])
            lp += logpdf(truncated(Normal(0, 2), 0, Inf), h)
            jac += z[1]
            lp += logpdf(Normal(h, 1), y[1])
            for t in 2:n
                next, j = interval(z[t], 0.0, ub)
                lp += logpdf(truncated(Normal(phi * h, 1), 0, ub), next)
                jac += j
                h = next
                lp += logpdf(Normal(h, 1), y[t])
            end
        else
            phi, rsd = u[entry(:phi).offset], u[entry(:sd).offset]
            sd = exp(rsd)
            lp += logpdf(Normal(), phi) + logpdf(truncated(Normal(), 0, Inf), sd)
            jac += rsd
            h = exp(z[1])
            lp += logpdf(truncated(Cauchy(), 0, Inf), h)
            jac += z[1]
            lp += logpdf(Normal(h, 1), y[1])
            for t in 2:n
                e = data[:lo] + exp(z[t])
                lp += logpdf(truncated(Normal(0, sd), data[:lo], Inf), e)
                jac += z[t]
                h = phi * h + e
                lp += logpdf(Normal(h, 1), y[t])
            end
        end
        return jacobian ? jac : lp + jac
    end
    return f, oracle, data
end

@testset "scan capabilities: bounded and improper seed and step priors" begin
    for kind in (:walk, :centered, :innovation), n in (kind === :walk ? 2 : 1, 5)
        f, oracle, data = _scan_cap_bounded(n, kind)
        snapshot = deepcopy(data)
        _scan_cap_check(f, oracle)
        @test logjac(f.built.layout, f.u) ≈ oracle(f.u; jacobian = true) rtol=1e-12
        draws = constrain(f.built.layout, f.u)
        @test unconstrain(f.built.layout, draws) ≈ f.u
        z = draws[only(e.name for e in f.built.layout.entries if e.kind === :scan)]
        if kind === :walk
            @test all(data[:lo] .< z[[1, 2, (5:(n + 2))...]] .< data[:hi])
        elseif kind === :centered
            @test all(0 .< z[2:end] .< draws.ub)
        else
            @test all(z[2:end] .> data[:lo])
        end
        @test data == snapshot
    end
    # refused: `truncated` divides by the base mass inside the bounds, which an
    # improper `Flat()` lacks; `restricted(Flat(), lo, hi)` is the flat kernel.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        @scan begin
            h[1] ~ truncated(Flat(), 0, 1)
            for t in 2:T
                h[t] ~ Normal(h[t - 1], 1)
            end
        end
        y .~ Normal.(h, 1.0)
    end, (:y, :T); conditioned = (:y,))
end

function _scan_cap_data_model(n, kind)
    program = if kind === :arma
        quote
            m ~ Normal(0, 5)
            phi ~ Normal(0, 1)
            theta ~ Normal(0, 1)
            sigma ~ Exponential(1)
            @scan begin
                h[1] = y[1] - m
                for t in 2:T
                    h[t] = y[t] - m - phi * (y[t - 1] - m) - theta * h[t - 1]
                end
            end
            z .~ Normal.(h, sigma)
        end
    elseif kind === :arma_tuple
        # ARMA(1,1) as two carried arrays: a tuple carry whose step reads its
        # current write (`err[t]` reads `nu[t]`) and indexed data.
        quote
            m ~ Normal(0, 5)
            phi ~ Normal(0, 1)
            theta ~ Normal(0, 1)
            sigma ~ Exponential(1)
            @scan begin
                nu[1] = m + phi * m
                err[1] = y[1] - nu[1]
                for t in 2:T
                    nu[t] = m + phi * y[t - 1] + theta * err[t - 1]
                    err[t] = y[t] - nu[t]
                end
            end
            y .~ Normal.(nu, sigma)
        end
    else
        quote
            m ~ Normal(0, 1)
            a0 ~ Exponential(1)
            a1 ~ Beta(1, 1)
            b1 ~ Beta(1, 1)
            @scan begin
                h[1] = a0 + a1 * (y[1] - m)^2 + b1
                for t in 2:T
                    h[t] = a0 + a1 * (y[t - 1] - m)^2 + b1 * h[t - 1]
                end
            end
            sigma = sqrt.(h)
            y .~ Normal.(m, sigma)
        end
    end
    y = [0.3sin(t) for t in 1:n]
    data = Dict{Symbol,Any}(:y => y)
    kind === :arma && (data[:z] = zeros(n))
    observed = kind === :arma ? (:z,) : (:y,)
    plan = bind_data(lower_rkppl(program, Tuple(keys(data)); conditioned = observed), data)
    built = build_kernel(plan)
    u = [0.2sin(i) for i in 1:built.layout.total]
    sampler = prepare_sampler(built, plan, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
    f = (; plan, built, sampler, u)
    parameter(u, name) = u[only(e for e in built.layout.entries if e.name === name).offset]
    function oracle(u)
        m = parameter(u, :m)
        if kind === :arma_tuple
            phi, theta = parameter(u, :phi), parameter(u, :theta)
            rawsigma = parameter(u, :sigma)
            sigma = exp(rawsigma)
            lp = logpdf(Normal(0,5),m) + logpdf(Normal(),phi) +
                logpdf(Normal(),theta) + logpdf(Exponential(),sigma) + rawsigma
            err = 0.0
            for t in 1:n
                nu = t == 1 ? m + phi * m : m + phi * y[t-1] + theta * err
                err = y[t] - nu
                lp += logpdf(Normal(nu,sigma),y[t])
            end
        elseif kind === :arma
            phi, theta = parameter(u, :phi), parameter(u, :theta)
            rawsigma = parameter(u, :sigma)
            sigma = exp(rawsigma)
            lp = logpdf(Normal(0,5),m) + logpdf(Normal(),phi) +
                logpdf(Normal(),theta) + logpdf(Exponential(),sigma) + rawsigma
            h = y[1] - m
            for t in 1:n
                t == 1 || (h = y[t] - m - phi * (y[t-1] - m) - theta * h)
                lp += logpdf(Normal(h,sigma),0)
            end
        else
            rawa0, rawa1, rawb1 = parameter(u,:a0), parameter(u,:a1), parameter(u,:b1)
            a0, a1, b1 = exp(rawa0), 1/(1+exp(-rawa1)), 1/(1+exp(-rawb1))
            lp = logpdf(Normal(),m) + logpdf(Exponential(),a0) + rawa0 +
                logpdf(Beta(1,1),a1) + logpdf(Beta(1,1),b1) +
                log(a1) + log1p(-a1) + log(b1) + log1p(-b1)
            h = a0 + a1 * (y[1] - m)^2 + b1
            for t in 1:n
                t == 1 || (h = a0 + a1 * (y[t-1] - m)^2 + b1 * h)
                lp += logpdf(Normal(m,sqrt(h)),y[t])
            end
        end
        return lp
    end
    return f, oracle
end

@testset "scan capabilities: deterministic ARMA and GARCH" begin
    for kind in (:arma, :arma_tuple, :garch), n in (1,4)
        f, oracle = _scan_cap_data_model(n,kind)
        @test !any(e -> e.kind === :scan, f.built.layout.entries)
        _scan_cap_check(f,oracle)
    end
end

# A trajectory is a value in every distribution argument: read bare or inline
# there, it lowers as its named alias (`sigma = sqrt.(h)` above) does.
function _scan_cap_argument_model(n, kind)
    y = kind === :zi ? [iseven(t) ? 0 : t for t in 1:n] : [0.3sin(t) for t in 1:n]
    data = Dict{Symbol,Any}(:y => y)
    program = if kind === :garch
        data[:s1] = 0.5
        quote
            m ~ Flat()
            a0 ~ restricted(Flat(), 0, Inf)
            a1 ~ restricted(Flat(), 0, 1)
            b1 ~ restricted(Flat(), 0, 1 - a1)
            @scan begin
                sigma[1] = s1
                for t in 2:T
                    sigma[t] = sqrt(a0 + a1 * (y[t - 1] - m)^2 + b1 * sigma[t - 1]^2)
                end
            end
            y .~ Normal.(m, sigma)
        end
    else
        observe = Dict(
            :inline => :(y .~ Normal.(m, sqrt.(s))),
            :nu => :(y .~ StudentT.(s, m, 1.0)),
            :zi => :(y .~ ZeroInflatedPoisson.(exp.(m), logistic.(s))),
            :mixture => :(y .~ MixtureModel.(vcat.(Normal.(m, s), Normal.(0, 1)),
                Ref([0.3, 0.7]))))[kind]
        Expr(:block, :(m ~ Normal(0, 1)), :(a ~ Normal(0, 1)),
            :(@scan begin
                s[1] = exp(a)
                for t in 2:T
                    s[t] = exp(a) + 0.5 * s[t - 1]
                end
            end), observe)
    end
    f = _scan_cap_sampler(program, data)
    parameter(u, name) = u[only(e for e in f.built.layout.entries if e.name === name).offset]
    function oracle(u)
        m = parameter(u, :m)
        if kind === :garch
            ra0, ra1, rb1 = parameter(u, :a0), parameter(u, :a1), parameter(u, :b1)
            a0, a1, l = exp(ra0), 1 / (1 + exp(-ra1)), 1 / (1 + exp(-rb1))
            b1 = (1 - a1) * l
            lp = ra0 + log(a1) + log1p(-a1) + log(1 - a1) + log(l) + log1p(-l)
            s = data[:s1]
            for t in 1:n
                t == 1 || (s = sqrt(a0 + a1 * (y[t - 1] - m)^2 + b1 * s^2))
                lp += logpdf(Normal(m, s), y[t])
            end
            return lp
        end
        a = parameter(u, :a)
        lp = logpdf(Normal(), m) + logpdf(Normal(), a)
        s = exp(a)
        for t in 1:n
            t == 1 || (s = exp(a) + 0.5 * s)
            lp += if kind === :inline
                logpdf(Normal(m, sqrt(s)), y[t])
            elseif kind === :nu
                logpdf(LocationScale(m, 1.0, TDist(s)), y[t])
            elseif kind === :zi
                p = 1 / (1 + exp(-s))
                log(p * (y[t] == 0) + (1 - p) * pdf(Poisson(exp(m)), y[t]))
            else
                logpdf(MixtureModel([Normal(m, s), Normal(0, 1)], [0.3, 0.7]), y[t])
            end
        end
        return lp
    end
    return f, oracle
end

@testset "scan capabilities: a trajectory read in distribution arguments" begin
    for kind in (:garch, :inline, :nu, :zi, :mixture), n in (1, 4)
        f, oracle = _scan_cap_argument_model(n, kind)
        @test !any(e -> e.kind === :scan, f.built.layout.entries)
        if kind === :zi
            # Value only: native Reverse through a live per-observation
            # zero-inflation vector fails for the named spelling too
            # (`w = s; logistic.(w)`), independently of how it is read.
            @test f.sampler(f.u) ≈ oracle(f.u) rtol=1e-11
        else
            _scan_cap_check(f, oracle)
        end
    end
end

function _scan_cap_defaults(n)
    program = quote
        phi ~ Normal()
        @scan begin
            h[1] ~ Normal()
            for t in 2:T
                e ~ Normal()
                h[t] = phi * h[t - 1] + e
            end
        end
        y .~ Binomial.(2, logistic.(h))
    end
    data = Dict{Symbol,Any}(:y => [iseven(t) for t in 1:n])
    snapshot = deepcopy(data)
    f = _scan_cap_sampler(program, data)
    pe = only(e for e in f.built.layout.entries if e.name === :phi)
    ze = only(e for e in f.built.layout.entries if e.kind === :scan)
    function oracle(u)
        phi = u[pe.offset]
        z = u[ze.offset:(ze.offset + ze.size - 1)]
        lp = logpdf(Normal(), phi) + sum(logpdf.(Normal(), z))
        h = z[1]
        for t in 1:n
            t == 1 || (h = phi * h + z[t])
            lp += logpdf(Binomial(2, 1 / (1 + exp(-h))), data[:y][t])
        end
        return lp
    end
    return f, oracle, data, snapshot
end

@testset "scan capabilities: constructor defaults and Boolean counts" begin
    for n in (1, 4)
        f, oracle, data, snapshot = _scan_cap_defaults(n)
        @test f.built.layout.total == n + 1
        _scan_cap_check(f, oracle)
        @test data == snapshot
    end
end
