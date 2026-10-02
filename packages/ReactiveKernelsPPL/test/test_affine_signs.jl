using Distributions

# Nonzero prior locations and asymmetric responses distinguish a signed use
# from changing the sampled coordinate. Oracles use only ordinary Julia and
# Distributions, independently of the generated predictor and its gradients.
function _affine_sign_fixtures(n)
    x = repeat([0.5, -1.0, 2.0], cld(n, 3))[1:n]
    y = repeat([0.2, -0.1, 0.9], cld(n, 3))[1:n]
    fixtures = []
    scalar_cases = (
        ("dotted coefficient", :(mu = .-b .* x), -1, 1),
        ("coefficient", :(mu = -b .* x), -1, 1),
        ("dotted column", :(mu = b .* (.-x)), -1, 1),
        ("column", :(mu = b .* (-x)), -1, 1),
        ("alias", quote nb = -b; mu = nb .* x end, -1, 1),
        ("dotted whole product", :(mu = .-(b .* x)), -1, 1),
        ("whole product", :(mu = -(b .* x)), -1, 1),
        ("two negative factors", :(mu = .-b .* (.-x)), 1, 1),
        ("outer and inner minus", :(mu = .-(.-b .* x)), 1, 1),
        ("positive", :(mu = b .* x), 1, 1),
        ("nonlinear coefficient", :(mu = .-(b^2) .* x), -1, 2),
    )
    for (label, definition, scale, power) in scalar_cases, reader in (false, true)
        ast = quote b ~ Normal(1.2, 0.7) end
        if definition.head === :block
            append!(ast.args, definition.args)
        else
            push!(ast.args, definition)
        end
        reader && push!(ast.args, :(q = b^2))
        push!(ast.args, :(y .~ Normal.(mu, 1.0)))
        prior = u -> logpdf(Normal(1.2, 0.7), u[1])
        oracle = u -> prior(u) + sum(logpdf.(
            Normal.(scale * u[1]^power .* x, 1.0), y))
        analytic = u -> [-(u[1] - 1.2) / 0.7^2 + sum(
            (y .- scale * u[1]^power .* x) .*
            (scale * power * u[1]^(power - 1) .* x))]
        push!(fixtures, (; label = "$label / reader=$reader", ast,
            data = (; x, y), u = [0.4], parameters = [:b], names = [:b],
            prior, oracle, analytic, reader))
    end

    # The same sign metadata serves factor, matrix and monotonic readers.
    g = repeat(["b", "a", "b"], cld(n, 3))[1:n]
    indices = [v == "a" ? 1 : 2 for v in g]
    factor_ast = quote
        c[levels(g)] .~ Normal.(1.2, 0.7)
        mu = .-c[g]
        q = sum(c .^ 2)
        y .~ Normal.(mu, 1.0)
    end
    factor_prior = u -> sum(logpdf.(Normal(1.2, 0.7), u))
    factor_oracle = u -> factor_prior(u) +
        sum(logpdf.(Normal.(-u[indices], 1.0), y))
    push!(fixtures, (; label = "factor whole minus", ast = factor_ast,
        data = (; g, y), u = [0.4, -0.2], parameters = [:c],
        names = Symbol.(["c.1", "c.2"]), prior = factor_prior,
        oracle = factor_oracle, analytic = nothing, reader = false))

    z = repeat([-0.3, 0.8, 0.1], cld(n, 3))[1:n]
    matrix_ast = quote
        X = hcat(x, z)
        b[axes(X, 2)] .~ Normal.([1.2, -0.3], [0.7, 1.1])
        mu = .-(X * b)
        q = sum(b .^ 2)
        y .~ Normal.(mu, 1.0)
    end
    matrix_prior = u -> sum(logpdf.(Normal.([1.2, -0.3], [0.7, 1.1]), u))
    matrix_oracle = u -> matrix_prior(u) + sum(logpdf.(
        Normal.(-(u[1] .* x .+ u[2] .* z), 1.0), y))
    push!(fixtures, (; label = "matrix whole minus", ast = matrix_ast,
        data = (; x, z, y), u = [0.4, -0.2], parameters = [:b],
        names = Symbol.(["b.1", "b.2"]), prior = matrix_prior,
        oracle = matrix_oracle, analytic = nothing, reader = false))

    c = repeat([1, 3, 2], cld(n, 3))[1:n]
    for (label, definition) in (
            ("monotonic coefficient", :(mu = .-b .* mo(c, s))),
            ("monotonic column", :(mu = b .* (.-mo(c, s)))),
            ("monotonic whole minus", :(mu = .-(b .* mo(c, s)))))
        ast = quote
            b ~ Normal(1.2, 0.7)
            s ~ Dirichlet([1.3, 2.4])
            q = b^2
        end
        push!(ast.args, definition, :(y .~ Normal.(mu, 1.0)))
        simplex = u -> [1 / (1 + exp(-u[2])), 1 / (1 + exp(u[2]))]
        prior = u -> logpdf(Normal(1.2, 0.7), u[1]) +
            logpdf(Dirichlet([1.3, 2.4]), simplex(u))
        oracle = u -> prior(u) + sum(log, simplex(u)) + sum(logpdf.(
            Normal.(-u[1] .* [0.0; cumsum(simplex(u))][c], 1.0), y))
        push!(fixtures, (; label, ast, data = (; c, y),
            u = [0.4, log(0.3 / 0.7)], parameters = [:b, :s],
            names = [:b, Symbol("s.1")], prior, oracle,
            analytic = nothing, reader = true))
    end
    return fixtures
end

@testset "affine factor signs preserve authored parameters" begin
    for f in _affine_sign_fixtures(3)
        @testset "$(f.label)" begin
            plan, built = _affine_model(f.ast; f.data...)
            @test coordinate_names(built.layout) == f.names
            @test isempty(plan.population_priors)
            @test sort!([p.name for p in (plan.parameters...,
                plan.array_parameters..., plan.vector_parameters...)]) ==
                sort(f.parameters)
            constrained = constrain(built.layout, f.u)
            @test unconstrain(built.layout, constrained) ≈ f.u
            if f.parameters == [:b] && length(f.u) == 1
                @test constrained.b == f.u[1]
            end
            @test _query(built.spec, plan, :prior, f.u) ≈ f.prior(f.u)
            sampler = prepare_sampler(built, plan, f.u; backend = _GEN_BACKEND)
            grad = similar(f.u)
            val, _ = sampler_value_and_gradient!(sampler, grad, f.u)
            @test val ≈ f.oracle(f.u)
            @test grad ≈ _findiff_grad(f.oracle, f.u) rtol=1e-5 atol=1e-7
            if f.analytic !== nothing
                @test grad ≈ f.analytic(f.u)
            end
            if f.reader
                @test _query(built.spec, plan, :q, f.u) == f.u[1]^2
            end
        end
    end
end
