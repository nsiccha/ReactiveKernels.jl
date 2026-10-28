using DataAPI
using Distributions
using ReactiveKernelsPPL
using Test

function _gv_build(kind, n, G; labels = false)
    pool = labels ? Any[missing, [2], NaN, -0.0, "a"][1:G] :
        [G; collect(1:(G - 1))]
    g = [pool[mod1(i, G)] for i in 1:n]
    axis = kind === :unique ? :(unique(g)) :
        kind === :stepped ? :(levels(g)[1:2:end]) :
        kind === :descending ? :(unique(g)[3:-2:1]) : :lv
    definition = kind === :alias ? :(lv = unique(g)) :
        kind === :computed ? :(lv = reverse(unique(g))) : nothing
    ast = quote
        $definition
        a ~ Normal(0, 1)
        z[$axis] .~ Normal.(0, 1)
        mu = a .+ z[g]
        y .~ Normal.(mu, 0.7)
    end
    filter!(!isnothing, ast.args)
    data = Dict{Symbol,Any}(:y => [0.2 * cos(i) for i in 1:n], :g => g)
    kind === :provided && (data[:lv] = [reverse(pool); G + 1])
    kind === :range && (data[:lv] = 1:(G + 1))
    plan = lower_rkppl(ast, data; mod = @__MODULE__, conditioned = (:y,))
    bound = bind_data(plan, data)
    return bound, build_kernel(bound)
end

function _gv_plate_build(n, G)
    ast = quote
        a ~ Normal(0, 1)
        z[unique(g), 1:2] .~ Normal.(0, 1)
        @plate for i in eachindex(g)
            mu[i] = a + sum(z[g[i], :])
        end
        y .~ Normal.(mu, 0.7)
    end
    data = Dict(:y => [0.2 * cos(i) for i in 1:n],
        :g => [mod1(i + G - 2, G) for i in 1:n])
    bound = bind_data(lower_rkppl(ast, data; conditioned = (:y,)), data)
    return bound, build_kernel(bound)
end

function _gv_crossed_build(n, G)
    ast = quote
        a ~ Normal(0, 1)
        z[unique(g), unique(h)] .~ Normal.(0, 1)
        @plate for i in eachindex(g)
            mu[i] = a + z[g[i], h[i]]
        end
        y .~ Normal.(mu, 0.7)
    end
    data = Dict(:y => [0.2 * cos(i) for i in 1:n],
        :g => [mod1(i + G - 2, G) for i in 1:n],
        :h => [mod1(2i, G + 1) for i in 1:n])
    bound = bind_data(lower_rkppl(ast, data; conditioned = (:y,)), data)
    return bound, build_kernel(bound)
end

function _gv_factor_build(n, G)
    ast = quote
        a ~ Normal(0, 1)
        z[levels(g)] .~ Normal.(0, 1)
        mu = a .+ z[g]
        y .~ Normal.(mu, 0.7)
    end
    pool = Any[collect(1:(G - 1)); missing]
    data = Dict(:y => [0.2 * cos(i) for i in 1:n],
        :g => [pool[mod1(i, G)] for i in 1:n])
    bound = bind_data(lower_rkppl(ast, data; conditioned = (:y,)), data)
    return bound, build_kernel(bound)
end

function _gv_levels(kind, columns)
    g = columns[:g]
    kind === :unique && return unique(g)
    kind === :stepped && return DataAPI.levels(g)[1:2:end]
    kind === :descending && return unique(g)[3:-2:1]
    return columns[:lv]
end

@testset "Grouping values: ordered data axes and stepped selections" begin
    for kind in (:unique, :alias, :computed, :provided, :range, :stepped, :descending),
            (n, G) in ((7, 3), (19, 5))
        bound, built = _gv_build(kind, n, G)
        u = [0.2 * sin(i) for i in 1:built.layout.total]
        nt = constrain(built.layout, u)
        lv = _gv_levels(kind, bound.columns)
        @test length(nt.z) == length(lv)
        means = map(bound.columns[:g]) do g
            k = findfirst(v -> isequal(v, g), lv)
            nt.a + (k === nothing ? 0.0 : nt.z[k])
        end
        prior = logpdf(Normal(), nt.a) + sum(logpdf.(Normal(), nt.z))
        ll = sum(logpdf.(Normal.(means, 0.7), bound.columns[:y]))
        @test _query(built.spec, bound, :prior, u) ≈ prior atol = 1e-11
        @test _query(built.spec, bound, :likelihood, u) ≈ ll atol = 1e-11
        @test unconstrain(built.layout, nt) ≈ u
        _check_gradient(built.spec, bound, u)
    end
    @test ReactiveKernelsPPL._declared_codes(Any[missing, [2], NaN, -0.0],
        Any[NaN, missing, -0.0, [2]]) == [2, 4, 1, 3]
end

@testset "Grouping values: arbitrary labels and array cells" begin
    for (n, G) in ((7, 3), (19, 5))
        bound, built = _gv_factor_build(n, G)
        u = [0.2 * sin(i) for i in 1:built.layout.total]
        nt = constrain(built.layout, u)
        lv = DataAPI.levels(bound.columns[:g])
        # DataAPI.levels excludes missing on an ordinary vector. A read at
        # an excluded label contributes zero, like a selected level axis.
        mu = map(bound.columns[:g]) do g
            k = findfirst(v -> isequal(v, g), lv)
            nt.a + (k === nothing ? 0.0 : nt.z[k])
        end
        @test _query(built.spec, bound, :likelihood, u) ≈
            sum(logpdf.(Normal.(mu, 0.7), bound.columns[:y]))
        _check_gradient(built.spec, bound, u)
    end
    for kind in (:unique, :alias), (n, G) in ((7, 3), (19, 5))
        bound, built = _gv_build(kind, n, G; labels = true)
        u = [0.2 * sin(i) for i in 1:built.layout.total]
        nt = constrain(built.layout, u)
        lv = _gv_levels(kind, bound.columns)
        mu = [nt.a + nt.z[findfirst(v -> isequal(v, g), lv)] for g in bound.columns[:g]]
        @test _query(built.spec, bound, :likelihood, u) ≈
            sum(logpdf.(Normal.(mu, 0.7), bound.columns[:y]))
        _check_gradient(built.spec, bound, u)
    end
    for (n, G) in ((7, 3), (19, 5))
        bound, built = _gv_plate_build(n, G)
        u = [0.2 * sin(i) for i in 1:built.layout.total]
        nt = constrain(built.layout, u)
        lv = unique(bound.columns[:g])
        mu = [nt.a + sum(nt.z[findfirst(==(g), lv), :]) for g in bound.columns[:g]]
        @test _query(built.spec, bound, :likelihood, u) ≈
            sum(logpdf.(Normal.(mu, 0.7), bound.columns[:y]))
        _check_gradient(built.spec, bound, u)
    end
    for (n, G) in ((7, 3), (19, 5))
        bound, built = _gv_crossed_build(n, G)
        u = [0.2 * sin(i) for i in 1:built.layout.total]
        nt = constrain(built.layout, u)
        g, h = bound.columns[:g], bound.columns[:h]
        gl, hl = unique(g), unique(h)
        mu = [nt.a + nt.z[findfirst(==(g[i]), gl), findfirst(==(h[i]), hl)]
            for i in eachindex(g)]
        @test _query(built.spec, bound, :likelihood, u) ≈
            sum(logpdf.(Normal.(mu, 0.7), bound.columns[:y]))
        _check_gradient(built.spec, bound, u)
    end
    # A label column read numerically still rejects missing observations.
    ast = quote
        a ~ Normal(0, 1)
        z[unique(g)] .~ Normal.(0, 1)
        mu = a .+ z[g] .+ a .* g
        y .~ Normal.(mu, 0.7)
    end
    data = Dict{Symbol,Any}(:y => [0.2, 0.3], :g => [1, missing])
    plan = lower_rkppl(ast, data; conditioned = (:y,))
    @test_throws ContractValidationError bind_data(plan, data)
    # A missing response entry is skipped by the whole-response statement
    # (provisional USER 1uhcm3b); the observed row keeps its density.
    observed = copy(data)
    observed[:g] = [1, 2]
    observed[:y] = [0.2, missing]
    bound = bind_data(plan, observed)
    built = build_kernel(bound)
    u = [0.2 * sin(i) for i in 1:built.layout.total]
    nt = constrain(built.layout, u)
    @test _query(built.spec, bound, :likelihood, u) ≈
        logpdf(Normal(nt.a + nt.z[1] + nt.a, 0.7), 0.2)
    _check_gradient(built.spec, bound, u)
end
