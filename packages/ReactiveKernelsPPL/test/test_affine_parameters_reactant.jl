using Reactant

@testset "ordinary affine parameters compile with their gradients" begin
    scalar_ops = Vector{String}[]
    for n in (4, 9)
        x = collect(range(-1.0, 1.0; length=n))
        y = 0.2 .+ 0.3 .* x
        plan, built = _affine_model(quote
            a ~ Normal(0, 2)
            b ~ Normal(0, 1)
            sigma ~ Exponential(1)
            mu = a .+ b .* x
            q = b^2
            y .~ Normal.(mu, sigma)
        end; x, y)
        u = [0.2, -0.3, log(0.8)]
        post = prepare_query(built, plan, :sampler)
        ru = Reactant.to_rarray(u)
        compiled = Reactant.@compile post(ru)
        push!(scalar_ops, [m.match for m in eachmatch(r"stablehlo\.[a-z_]+",
            string(Reactant.@code_hlo post(ru)))])
        @test Float64(compiled(ru)) ≈ post(u)
        sampler = prepare_sampler(built, plan, u; backend=_GEN_BACKEND)
        grad = similar(u)
        val, _ = sampler_value_and_gradient!(sampler, grad, u)
        cad = compile_ad_value_and_gradient(sampler.ad, ru)
        rval, rgrad = cad(ru)
        @test Float64(rval) ≈ val
        @test Array(rgrad) ≈ grad
    end
    @test !isempty(scalar_ops[1])
    @test scalar_ops[1] == scalar_ops[2]

    factor_ops = Vector{String}[]
    for k in (3, 6)
        g = collect(1:k)
        y = zeros(k)
        x = collect(range(0.5, 1.5; length=k))
        plan, built = _affine_model(quote
            a ~ Normal(0, 2)
            c[levels(g)[2:end]] .~ Normal.(0, 1)
            mu = a .+ c[g]
            shifted = c[g] .* x
            y .~ Normal.(mu, 1.0)
            y2 .~ Normal.(shifted, 1.0)
        end; g, y, x, y2=y)
        u = collect(range(-0.2, 0.3; length=k))
        post = prepare_query(built, plan, :sampler)
        ru = Reactant.to_rarray(u)
        compiled = Reactant.@compile post(ru)
        push!(factor_ops, [m.match for m in eachmatch(r"stablehlo\.[a-z_]+",
            string(Reactant.@code_hlo post(ru)))])
        @test Float64(compiled(ru)) ≈ post(u)
        sampler = prepare_sampler(built, plan, u; backend=_GEN_BACKEND)
        grad = similar(u)
        val, _ = sampler_value_and_gradient!(sampler, grad, u)
        cad = compile_ad_value_and_gradient(sampler.ad, ru)
        rval, rgrad = cad(ru)
        @test Float64(rval) ≈ val
        @test Array(rgrad) ≈ grad
    end
    @test !isempty(factor_ops[1])
    @test factor_ops[1] == factor_ops[2]

    @testset "matrix and GLM readers retain the array prior" begin
        for glm in (false, true)
            operations = Vector{String}[]
            for n in (4, 9)
                x = collect(range(-1.0, 1.0; length=n))
                y = 0.2 .+ 0.3 .* x
                ast = glm ? quote
                    X = hcat(x)
                    alpha ~ Normal(0, 2)
                    b[axes(X, 2)] .~ StudentT.(3, 0, 2)
                    sigma ~ Exponential(1)
                    y ~ NormalIDGLM(X, alpha, b, sigma)
                    q = sum(b .^ 2)
                end : quote
                    X = hcat(ones(length(x)), x)
                    b[axes(X, 2)] .~ Normal.(0, 2)
                    sigma ~ Exponential(1)
                    mu = X * b
                    q = sum(b .^ 2)
                    y .~ Normal.(mu, sigma)
                end
                plan, built = _affine_model(ast; x, y)
                u = collect(range(-0.2, 0.3; length=built.layout.total))
                _check_gradient(built.spec, plan, u)
                post = prepare_query(built, plan, :sampler)
                ru = Reactant.to_rarray(u)
                compiled = Reactant.@compile post(ru)
                push!(operations, [m.match for m in eachmatch(r"stablehlo\.[a-z_]+",
                    string(Reactant.@code_hlo post(ru)))])
                @test Float64(compiled(ru)) ≈ post(u)
                sampler = prepare_sampler(built, plan, u; backend=_GEN_BACKEND)
                grad = similar(u)
                val, _ = sampler_value_and_gradient!(sampler, grad, u)
                cad = compile_ad_value_and_gradient(sampler.ad, ru)
                rval, rgrad = cad(ru)
                @test Float64(rval) ≈ val
                @test Array(rgrad) ≈ grad
            end
            @test !isempty(operations[1])
            @test operations[1] == operations[2]
        end
    end

    @testset "mixed packs and transformed scalars" begin
        for ast in (quote
                b ~ Laplace(0, 2)
                c[levels(g)] .~ Cauchy.(0, 1)
                mu = c[g] .- b .* x
                y .~ Normal.(mu, 1.0)
            end, quote
                a ~ Uniform(-2, 2)
                b ~ HalfNormal(1)
                mu = a .- b .* x
                y .~ Normal.(mu, 1.0)
            end, quote
                c[levels(g)] .~ Normal.(0, 1)
                d[levels(g)[2:end]] .~ Normal.(0, 2)
                mu = c[g] .+ d[g]
                y .~ Normal.(mu, 1.0)
            end, quote
                a ~ Normal(0, 2)
                s ~ HalfNormal(1)
                c[levels(g)] .~ Normal.(0, 1)
                d = s .* c
                mu = a .+ d[g]
                y .~ Normal.(mu, 1.0)
            end)
            g = ["b", "a", "c", "b"]
            x = [-1.0, 0.5, 2.0, 0.2]
            y = zeros(4)
            plan, built = _affine_model(ast; g, x, y)
            u = collect(range(-0.2, 0.3; length=built.layout.total))
            _check_gradient(built.spec, plan, u)
            post = prepare_query(built, plan, :sampler)
            ru = Reactant.to_rarray(u)
            compiled = Reactant.@compile post(ru)
            @test Float64(compiled(ru)) ≈ post(u)
            sampler = prepare_sampler(built, plan, u; backend=_GEN_BACKEND)
            grad = similar(u)
            val, _ = sampler_value_and_gradient!(sampler, grad, u)
            cad = compile_ad_value_and_gradient(sampler.ad, ru)
            rval, rgrad = cad(ru)
            @test Float64(rval) ≈ val
            @test Array(rgrad) ≈ grad
        end
    end
end
