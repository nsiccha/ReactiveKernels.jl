module StochasticCellScanTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions
import DifferentiationInterface as DI
import Enzyme

function build(program, data)
    plan = lower_rkppl(program, Tuple(keys(data)); conditioned = (:y,))
    bound = bind_data(plan, data)
    return (; plan, bound, built = build_kernel(bound))
end

raw(f, u, name) = begin
    e = only(e for e in f.built.layout.entries if e.name === name)
    u[e.offset:(e.offset + e.size - 1)]
end
function check(f, oracle; shifts = (0.0, 0.11))
    n = f.built.layout.total
    initial = [0.13sin(i) for i in 1:n]
    sampler = prepare_sampler(f.built, f.bound, initial;
        backend = DI.AutoEnzyme(; mode = Enzyme.Reverse))
    prior_query = prepare_query(f.built, f.bound, :prior)
    likelihood_query = prepare_query(f.built, f.bound, :likelihood)
    pointwise_query = prepare_query(f.built, f.bound, :pointwise)
    for shift in shifts
        u = initial .+ shift
        expected = oracle(u)
        saved = copy(u)
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ expected.prior + expected.likelihood + expected.jac rtol = 2e-11
        h = cbrt(eps(Float64))
        posterior(x) = sum(values(oracle(x)))
        fd = map(eachindex(u)) do i
            up, dn = copy(u), copy(u)
            up[i] += h
            dn[i] -= h
            (posterior(up) - posterior(dn)) / (2h)
        end
        @test gradient ≈ fd rtol = 2e-5 atol = 2e-7
        @test Base.invokelatest(prior_query, u) ≈ expected.prior rtol = 2e-11 atol = 1e-12
        @test Base.invokelatest(likelihood_query, u) ≈ expected.likelihood rtol = 2e-11 atol = 1e-12
        pointwise = Base.invokelatest(pointwise_query, u)
        @test keys(pointwise) == (:y,)
        @test sum(x -> sum(x; init = 0.0), pointwise.y; init = 0.0) ≈
            expected.likelihood rtol = 2e-11 atol = 1e-12
        @test logjac(f.built.layout, u) ≈ expected.jac atol = 1e-12
        @test unconstrain(f.built.layout, constrain(f.built.layout, u)) ≈ u
        @test u == saved
    end
end

function normal_fixture(lengths; centered = false, history = false)
    data = (; y = [[0.2sin(g + t) for t in 1:T] for (g, T) in enumerate(lengths)])
    mean = history ? :(phi * sum(h[k] for k in 1:(t - 1)) / (t - 1)) :
        :(phi * h[t - 1])
    step = centered ? :(h[t] ~ Normal($mean, s)) : quote
        innovation ~ Normal(0, 1)
        h[t] = $mean + s * innovation
    end
    steps = Meta.isexpr(step, :block) ? step.args : Any[step]
    program = quote
        phi ~ Normal(0, 0.7)
        s ~ Exponential(1.2)
        @plate for g in eachindex(y)
            T = length(y[g])
            seed_mean = phi + 0.05g
            @scan begin
                h[1] ~ Normal(seed_mean, 1)
                for t in 2:T
                    $(steps...)
                end
            end
            loc[g] = h
            y[g] .~ Normal.(loc[g], 0.8)
        end
    end
    f = build(program, data)
    function oracle(u; fixture = f, observed = data)
        current_lengths = length.(observed.y)
        phi = only(raw(fixture, u, :phi))
        log_s = only(raw(fixture, u, :s))
        s = exp(log_s)
        seeds = raw(fixture, u, :_rkppl_scan_h_seed_h_1)
        steps = raw(fixture, u, centered ? :_rkppl_scan_h_draw_h : :_rkppl_scan_h_draw_innovation)
        prior = logpdf(Normal(0, 0.7), phi) + logpdf(Exponential(1.2), s)
        likelihood, offset = 0.0, 0
        for (g, j) in enumerate(eachindex(current_lengths))
            h = [seeds[g]]
            prior += logpdf(Normal(phi + 0.05j, 1), h[1])
            for t in 2:current_lengths[j]
                mu = phi * (history ? sum(h) / length(h) : last(h))
                z = steps[offset + t - 1]
                prior += logpdf(centered ? Normal(mu, s) : Normal(), z)
                push!(h, centered ? z : mu + s * z)
            end
            offset += max(0, current_lengths[j] - 1)
            likelihood += sum(logpdf.(Normal.(h, 0.8), observed.y[j]))
        end
        return (; prior, likelihood, jac = log_s)
    end
    return f, oracle, data
end

@testset "stochastic cell scans: independent packed trajectories" begin
    for (lengths, centered, history) in (
            ([1, 4, 2], false, false),
            ([4, 2], true, true),
            ([2, 3, 1], false, false),
            ([1, 1], true, false),
            ([0, 3], false, false),
            (Int[], false, false))
        f, oracle, data = normal_fixture(lengths; centered, history)
        snapshot = deepcopy(data)
        @test f.built.layout.total == 2 + sum(max.(lengths, 1))
        check(f, oracle)
        @test data == snapshot
    end
end

@testset "stochastic cell scans: bounds, Flat, and normalizers" begin
    for normalized in (false, true)
        prior = normalized ? :(truncated(Normal(0.2 * h[t - 1], s), lo, hi)) :
            :(restricted(Normal(0.2 * h[t - 1], s), lo, hi))
        data = (; y = [[0.1, 0.4, -0.1], [0.2]], lo = -0.5, hi = 1.3)
        f = build(quote
            s ~ Exponential(1)
            @plate for g in eachindex(y)
                @scan begin
                    h[1] ~ Flat()
                    for t in 2:length(y[g])
                        innovation ~ $prior
                        h[t] = h[t - 1] + innovation
                    end
                end
                loc[g] = h
                y[g] .~ Normal.(loc[g], 1)
            end
        end, data)
        function oracle(u)
            a = only(raw(f, u, :s))
            s, prior, likelihood, jac = exp(a), -exp(a), 0.0, a
            seeds = raw(f, u, :_rkppl_scan_h_seed_h_1)
            draws = raw(f, u, :_rkppl_scan_h_draw_innovation)
            k = 1
            for g in eachindex(data.y)
                h = seeds[g]
                for t in eachindex(data.y[g])
                    if t > 1
                        z = draws[k]
                        innovation = data.lo + (data.hi - data.lo) / (1 + exp(-z))
                        d = Normal(0.2h, s)
                        prior += logpdf(normalized ? truncated(d, data.lo, data.hi) : d, innovation)
                        jac += log(data.hi - data.lo) - log1p(exp(-z)) - log1p(exp(z))
                        h += innovation
                        k += 1
                    end
                    likelihood += logpdf(Normal(h, 1), data.y[g][t])
                end
            end
            return (; prior, likelihood, jac)
        end
        check(f, oracle)
    end
end

@testset "stochastic cell scans: rebinding and retained structure" begin
    small, oracle, data = normal_fixture([2, 3])
    larger_data = (; y = [[0.1cos(g + t) for t in 1:T] for (g, T) in enumerate([8, 1, 5, 4])])
    rebound = bind_data(small.plan, larger_data)
    larger = build_kernel(rebound)
    @test larger.layout.total == 2 + sum(length.(larger_data.y))
    check((; built = larger, bound = rebound),
        u -> oracle(u; fixture = (; built = larger), observed = larger_data);
        shifts = (0.03,))
    structure(spec) = [(e.kind, e.depth) for e in recipe_inventory(spec) if e.kind !== :ordinary]
    @test structure(larger.spec) == structure(small.built.spec)
    @test count(e -> e.kind === :scan, recipe_inventory(small.built.spec)) == 1
    body_size(x) = x isa Expr ? 1 + sum(body_size, x.args; init = 0) : 1
    @test body_size(kernel_expr(rebound, larger.layout)) ==
        body_size(kernel_expr(small.bound, small.built.layout))
    # The printable program replays the same graph, including the prior.
    source = kernel_expr(small.bound, small.built.layout)
    replayed = ReactiveKernelsPPL._eval_kernel_def(source)
    u = fill(0.1, small.built.layout.total)
    replay = prepare_query((; spec = replayed, layout = small.built.layout), small.bound, :sampler)
    @test Base.invokelatest(replay, u) ≈ sum(values(oracle(u)))
    @test data.y == [[0.2sin(g + t) for t in 1:T] for (g, T) in enumerate([2, 3])]
end

@testset "stochastic cell scans: seed depth and sampled support bounds" begin
    data = (; y = [[0.2, -0.1, 0.3, 0.2], [0.4, 0.1]])
    f = build(quote
        s ~ Exponential(1)
        ub ~ restricted(Normal(3, 1), 1, Inf)
        @plate for g in eachindex(y)
            @scan begin
                h[1] ~ HalfNormal(1)
                h[2] ~ restricted(Flat(), 0, ub)
                for t in 3:length(y[g])
                    innovation ~ Exponential(s)
                    h[t] = h[t - 2] + innovation
                end
            end
            loc[g] = h
            y[g] .~ Normal.(loc[g], 1)
        end
    end, data)
    function oracle(u)
        a, b = only(raw(f, u, :s)), only(raw(f, u, :ub))
        s, ub = exp(a), 1 + exp(b)
        prior, jac, likelihood = logpdf(Exponential(), s) + logpdf(Normal(3, 1), ub), a + b, 0.0
        seed1 = raw(f, u, :_rkppl_scan_h_seed_h_1)
        seed2 = raw(f, u, :_rkppl_scan_h_seed_h_2)
        steps = raw(f, u, :_rkppl_scan_h_draw_innovation)
        k = 1
        for g in eachindex(data.y)
            h = [exp(seed1[g]), ub / (1 + exp(-seed2[g]))]
            prior += logpdf(truncated(Normal(), 0, Inf), h[1])
            jac += seed1[g] + log(ub) - log1p(exp(-seed2[g])) - log1p(exp(seed2[g]))
            for t in 3:length(data.y[g])
                innovation = exp(steps[k])
                prior += logpdf(Exponential(s), innovation)
                jac += steps[k]
                push!(h, h[t - 2] + innovation)
                k += 1
            end
            likelihood += sum(logpdf.(Normal.(h, 1), data.y[g]))
        end
        return (; prior, likelihood, jac)
    end
    check(f, oracle)
end
end
