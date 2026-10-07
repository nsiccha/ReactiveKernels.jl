using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

# General response values, independent of the legacy grouped-model fixtures.
function _prv_fixture(kind, n; unbound = nothing)
    x = collect(range(0.2, 0.8; length = n))
    y = collect(range(-0.2, 0.4; length = n))
    g = [mod1(i, 2) for i in 1:n]
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 2)
        r[levels(g)] .~ Normal.(0, 1)
    end
    # Computed cell arguments read the top-level definition `mu` at the
    # loop index, for every observation iterator and argument position.
    alias = kind in (:alias, :alias_literal, :alias_axes, :alias_local, :alias_scale)
    alias && push!(ast.args, :(mu = a .+ r[g]))
    # `:selected` observes the whole response; binding selects its single
    # present entry y[1]. The skipped cells' `sqrt` operands are invalid.
    selected = kind === :selected
    selected && n > 1 && (x[2:end] .= -1.0)
    iterator = kind === :alias_literal ? :(1:$n) :
        kind === :alias_axes ? :(axes(y, 1)) : :(eachindex(y))
    value = selected ? :(a + r[g[i]] + b * sqrt(x[i])) :
        alias ? :(mu[i] + b * x[i]) : :(a + r[g[i]] + b * x[i])
    cell = kind === :alias_local ?
        Expr(:block, :(m = $value), :(y[i] ~ Normal(m, 0.7))) :
        kind === :alias_scale ?
        Expr(:block, :(y[i] ~ Normal(b * x[i], exp(0.2 * mu[i])))) :
        Expr(:block, :(y[i] ~ Normal($value, 0.7)))
    push!(ast.args, Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
        Expr(:for, :(i = $iterator), cell)))
    # Observations cover the whole response and binding skips missing
    # entries (provisional USER `1uhcm3b`).
    observed_y = selected ? Union{Missing,Float64}[i == 1 ? y[i] : missing
        for i in eachindex(y)] : y
    data = Dict(:x => x, :y => observed_y, :g => g)
    unbound === nothing && (unbound = lower_rkppl(ast, data; conditioned = keys(data)))
    bound = bind_data(unbound, data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; a = 0.25, b = -0.3, r = [-0.2, 0.4]))
    function pointwise(v)
        q = constrain(built.layout, v)
        kind === :alias_scale && return [logpdf(Normal(q.b * x[i],
            exp(0.2 * (q.a + q.r[g[i]]))), y[i]) for i in eachindex(y)]
        # A skipped entry leaves zero in pointwise output.
        [ismissing(observed_y[i]) ? 0.0 : logpdf(Normal(q.a + q.r[g[i]] +
            q.b * (selected ? sqrt(x[i]) : x[i]), 0.7), y[i]) for i in eachindex(y)]
    end
    function oracle(v)
        q = constrain(built.layout, v)
        logpdf(Normal(), q.a) + logpdf(Normal(0, 2), q.b) +
            sum(logpdf.(Normal(), q.r)) +
            sum(pointwise(v))
    end
    return (; kind, data, unbound, bound, built, u, oracle, pointwise)
end

function _prv_bare(n; unbound = nothing, indexed = true)
    data = Dict(:y => [mod(i, 3) for i in 1:n], :trials => fill(3, n))
    ast = Expr(:block, :(p ~ Beta(2, 3)))
    if indexed
        observation = quote
            @plate for i in eachindex(y)
                y[i] ~ Binomial(trials[i], p)
            end
        end
        push!(ast.args, observation.args[end])
    else
        push!(ast.args, :(y .~ Binomial.(trials, p)))
    end
    unbound === nothing && (unbound = lower_rkppl(ast, data; conditioned = keys(data)))
    bound = bind_data(unbound, data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; p = 0.4))
    pointwise(v) = logpdf.(Binomial.(data[:trials], constrain(built.layout, v).p), data[:y])
    function oracle(v)
        p = constrain(built.layout, v).p
        logpdf(Beta(2, 3), p) + log(p * (1 - p)) + sum(pointwise(v))
    end
    return (; kind = indexed ? :bare : :bare_broadcast,
        data, unbound, bound, built, u, oracle, pointwise)
end

function _prv_fd(f, u)
    h = cbrt(eps(Float64))
    [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end for i in eachindex(u)]
end

@testset "retained plate response values: native math and ownership" begin
    for kind in (:direct, :alias, :alias_literal, :alias_axes, :alias_local,
            :alias_scale, :selected, :bare)
        for n in (3, 9)
            fx = kind === :bare ? _prv_bare(n) : _prv_fixture(kind, n)
            saved = deepcopy(fx.data)
            sampler = prepare_sampler(fx.built, fx.bound, fx.u;
                backend = AutoEnzyme(; mode = Enzyme.Reverse))
            for shift in (0.0, 0.03)
                u = fx.u .+ shift
                value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
                @test value ≈ fx.oracle(u) rtol = 1e-10
                @test gradient ≈ _prv_fd(fx.oracle, u) rtol = 1e-5 atol = 1e-7
                pw = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), u)
                @test pw.y ≈ fx.pointwise(u)
            end
            @test isequal(fx.data, saved)
        end
    end
end
