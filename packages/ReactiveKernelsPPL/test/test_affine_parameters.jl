using Distributions

function _affine_model(ast; data...)
    plan = RKPPLModel(ast, @__MODULE__)(; data...)
    built = build_kernel(plan)
    return plan, built
end

@testset "affine uses preserve parameter semantics" begin
    x = [-1.0, 0.5, 2.0]
    z = [0.2, -0.4, 1.0]
    y = [0.3, -0.2, 0.7]
    base = quote
        a ~ Normal(0, 5)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1)
        mu = a .+ b .* x
        y .~ Normal.(mu, sigma)
    end
    plan, built = _affine_model(base; x, y)
    @test coordinate_names(built.layout) == [:a, :b, :sigma]
    @test isempty(plan.population_priors)
    @test Set(p.name for p in plan.parameters) == Set((:a, :b, :sigma))
    u = [0.2, -0.3, log(0.8)]
    expected = logpdf(Normal(0, 5), u[1]) + logpdf(Normal(), u[2]) +
        logpdf(Exponential(), exp(u[3])) + u[3] +
        sum(logpdf.(Normal.(u[1] .+ u[2] .* x, exp(u[3])), y))
    @test _query(built.spec, plan, :posterior, u) ≈ expected
    @test _query(built.spec, plan, :b, u) == u[2]
    _check_gradient(built.spec, plan, u)

    extra = copy(base)
    push!(extra.args, :(q = b^2))
    p2, b2 = _affine_model(extra; x, y)
    @test coordinate_names(b2.layout) == coordinate_names(built.layout)
    @test _query(b2.spec, p2, :posterior, u) ==
        _query(built.spec, plan, :posterior, u)
    @test _query(b2.spec, p2, :q, u) == u[2]^2

    shared, bs = _affine_model(quote
        a ~ Normal(0, 5)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1)
        mu = a .+ b .* x .- b .* z
        y .~ Normal.(mu, sigma)
        eta = b .* z
        y2 .~ Normal.(eta, sigma)
        q = b^2
    end; x, z, y, y2=y)
    @test coordinate_names(bs.layout) == [:a, :b, :sigma]
    expected_shared = expected -
        sum(logpdf.(Normal.(u[1] .+ u[2] .* x, exp(u[3])), y)) +
        sum(logpdf.(Normal.(u[1] .+ u[2] .* (x .- z), exp(u[3])), y)) +
        sum(logpdf.(Normal.(u[2] .* z, exp(u[3])), y))
    @test _query(bs.spec, shared, :posterior, u) ≈ expected_shared
    _check_gradient(bs.spec, shared, u)

    overlapping, bo = _affine_model(quote
        a ~ Normal(1, 2)
        b ~ Normal(0, 3)
        c ~ Normal(-1, 4)
        d ~ Normal(2, 5)
        mu = a .- d .+ b .* x .- c .* x
        y .~ Normal.(mu, 1.0)
    end; x, y)
    uo = [0.2, -0.3, 0.4, -0.1]
    expected_overlap = sum(logpdf.(Normal.([1, 0, -1, 2], [2, 3, 4, 5]), uo)) +
        sum(logpdf.(Normal.(uo[1] - uo[4] .+ (uo[2] - uo[3]) .* x, 1.0), y))
    @test coordinate_names(bo.layout) == [:a, :b, :c, :d]
    @test _query(bo.spec, overlapping, :posterior, uo) ≈ expected_overlap
    _check_gradient(bo.spec, overlapping, uo)
end

@testset "affine vector and factor uses retain their array declarations" begin
    x = [-1.0, 0.5, 2.0]
    y = [0.3, -0.2, 0.7]
    plan, built = _affine_model(quote
        X = hcat(1, x)
        b[axes(X, 2)] .~ Normal.(0, 2)
        sigma ~ Exponential(1)
        mu = X * b
        q = sum(b .^ 2)
        y .~ Normal.(mu, sigma)
    end; x, y)
    @test isempty(plan.population_priors)
    @test only(plan.array_parameters).name === :b
    @test coordinate_names(built.layout) == [:sigma, Symbol("b.1"), Symbol("b.2")]
    u = [log(0.8), 0.2, -0.3]
    expected = logpdf(Exponential(), exp(u[1])) + u[1] +
        sum(logpdf.(Normal(0, 2), u[2:3])) +
        sum(logpdf.(Normal.(u[2] .+ u[3] .* x, exp(u[1])), y))
    @test _query(built.spec, plan, :posterior, u) ≈ expected
    @test _query(built.spec, plan, :q, u) ≈ sum(abs2, u[2:3])
    _check_gradient(built.spec, plan, u)

    g = ["b", "a", "b"]
    factor, bf = _affine_model(quote
        c[levels(g)] .~ Normal.(0, 2)
        sigma ~ Exponential(1)
        mu = c[g]
        q = sum(c .^ 2)
        y .~ Normal.(mu, sigma)
    end; g, y)
    @test only(factor.array_parameters).name === :c
    @test isempty(factor.population_priors)
    expected_factor = logpdf(Exponential(), exp(u[1])) + u[1] +
        sum(logpdf.(Normal(0, 2), u[2:3])) +
        sum(logpdf.(Normal.([u[3], u[2], u[3]], exp(u[1])), y))
    @test _query(bf.spec, factor, :posterior, u) ≈ expected_factor
    _check_gradient(bf.spec, factor, u)

    scaled, bscaled = _affine_model(quote
        a ~ Normal(0, 2)
        s ~ HalfNormal(1)
        c[levels(g)] .~ Normal.(0, 1)
        d = s .* c
        mu = a .+ d[g]
        y .~ Normal.(mu, 1.0)
    end; g, y)
    @test only(scaled.assignments).name === :d
    @test coordinate_names(bscaled.layout) == [:a, :s, Symbol("c.1"), Symbol("c.2")]
    uscaled = [0.2, log(0.7), -0.3, 0.4]
    expected_scaled = logpdf(Normal(0, 2), uscaled[1]) +
        logpdf(truncated(Normal(), 0, Inf), exp(uscaled[2])) + uscaled[2] +
        sum(logpdf.(Normal(), uscaled[3:4])) +
        sum(logpdf.(Normal.(uscaled[1] .+ exp(uscaled[2]) .* [uscaled[4], uscaled[3], uscaled[4]], 1.0), y))
    @test _query(bscaled.spec, scaled, :posterior, uscaled) ≈ expected_scaled
    _check_gradient(bscaled.spec, scaled, uscaled)
end

@testset "affine readers and prior arguments share a declaration" begin
    x = [-1.0, 0.5, 2.0]
    y = [0.3, -0.2, 0.7]
    plan, built = _affine_model(quote
        b ~ Normal(0, 1)
        c ~ Normal(b, 2)
        nu ~ Gamma(2, 1)
        mu = b .* x .+ b .* x .+ c
        y .~ StudentT.(nu, mu, 1.0)
    end; x, y)
    @test coordinate_names(built.layout) == [:b, :c, :nu]
    u = [-0.2, 0.4, log(3.0)]
    expected = logpdf(Normal(), u[1]) + logpdf(Normal(u[1], 2), u[2]) +
        logpdf(Gamma(2, 1), exp(u[3])) + u[3] +
        sum(logpdf.(LocationScale.(u[2] .+ 2u[1] .* x, 1.0,
            TDist(exp(u[3]))), y))
    @test _query(built.spec, plan, :posterior, u) ≈ expected
    _check_gradient(built.spec, plan, u)

    g = ["c", "a", "b"]
    subset, bs = _affine_model(quote
        a ~ Normal(0, 2)
        c[levels(g)[2:end]] .~ Normal.(0, 2)
        mu = a .+ c[g]
        q = sum(c .^ 2)
        shifted = c[g] .+ 1.0
        y .~ Normal.(mu, 1.0)
    end; g, y)
    @test only(subset.array_parameters).name === :c
    @test coordinate_names(bs.layout) == [:a, Symbol("c.1"), Symbol("c.2")]
    u = [0.2, -0.3, 0.4]
    expected = sum(logpdf.(Normal(0, 2), u)) +
        sum(logpdf.(Normal.(u[1] .+ [u[3], 0.0, u[2]], 1.0), y))
    @test _query(bs.spec, subset, :posterior, u) ≈ expected
    @test _query(bs.spec, subset, :shifted, u) ≈ [u[3], 0.0, u[2]] .+ 1.0
    _check_gradient(bs.spec, subset, u)

    glm, bg = _affine_model(quote
        X = hcat(x)
        alpha ~ Normal(0, 2)
        b[axes(X, 2)] .~ StudentT.(3, 0, 2)
        sigma ~ Exponential(1)
        y ~ NormalIDGLM(X, alpha, b, sigma)
        q = sum(b .^ 2)
    end; x, y)
    @test isempty(glm.population_priors)
    @test only(glm.array_parameters).name === :b
    u = [0.2, log(0.8), -0.3]
    expected = logpdf(Normal(0, 2), u[1]) +
        logpdf(Exponential(), exp(u[2])) + u[2] +
        logpdf(LocationScale(0, 2, TDist(3)), u[3]) +
        sum(logpdf.(Normal.(u[1] .+ u[3] .* x, exp(u[2])), y))
    @test _query(bg.spec, glm, :posterior, u) ≈ expected
    _check_gradient(bg.spec, glm, u)
end

@testset "affine optimization preserves matrix and level value reads" begin
    x = [-1.0, 0.5, 2.0]
    y = [0.3, -0.2, 0.7]
    plan, built = _affine_model(quote
        X = hcat(1, x)
        b[axes(X, 2)] .~ Normal.(0, 1)
        mu = exp.(X * b)
        y .~ Normal.(mu, 1.0)
        raw = X * b
        y2 .~ Normal.(raw, 1.0)
    end; x, y, y2=y)
    @test isempty(plan.population_priors)
    @test only(plan.array_parameters).name === :b
    u = [0.2, -0.3]
    eta = u[1] .+ u[2] .* x
    expected = sum(logpdf.(Normal(), u)) +
        sum(logpdf.(Normal.(exp.(eta), 1.0), y)) +
        sum(logpdf.(Normal.(eta, 1.0), y))
    @test _query(built.spec, plan, :posterior, u) ≈ expected
    _check_gradient(built.spec, plan, u)

    g = ["a", "b", "c"]
    h = ["c", "a", "b"]
    factor, bf = _affine_model(quote
        c[levels(g)] .~ Normal.(0, 1)
        mu = c[g]
        eta = c[h]
        y .~ Normal.(mu, 1.0)
        y2 .~ Normal.(eta, 1.0)
    end; g, h, y, y2=y)
    u = [0.2, -0.3, 0.4]
    @test coordinate_names(bf.layout) == Symbol.("c." .* string.(1:3))
    expected = sum(logpdf.(Normal(), u)) +
        sum(logpdf.(Normal.(u, 1.0), y)) +
        sum(logpdf.(Normal.(u[[3, 1, 2]], 1.0), y))
    @test _query(bf.spec, factor, :posterior, u) ≈ expected
    _check_gradient(bf.spec, factor, u)

    g_mixed = ["a", "b", "c"]
    mixed, bm = _affine_model(quote
        c[levels(g)] .~ Normal.(0, 1)
        d[levels(g)[2:end]] .~ Normal.(0, 2)
        mu = c[g] .+ d[g]
        y .~ Normal.(mu, 1.0)
    end; g = g_mixed, y)
    um = unconstrain(bm.layout, (c = [0.2, -0.3, 0.4], d = [0.1, -0.2]))
    expected_mixed = sum(logpdf.(Normal(), [0.2, -0.3, 0.4])) +
        sum(logpdf.(Normal(0, 2), [0.1, -0.2])) +
        sum(logpdf.(Normal.([0.2, -0.2, 0.2], 1.0), y))
    @test coordinate_names(bm.layout) ==
        Symbol.(["c.1", "c.2", "c.3", "d.1", "d.2"])
    @test _query(bm.spec, mixed, :posterior, um) ≈ expected_mixed
    _check_gradient(bm.spec, mixed, um)

    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 1)
        b ~ Horseshoe()
        mu = a .+ b .* x
        y .~ Normal.(mu, 1.0)
        q = a^2
    end, (:x, :y))
end
