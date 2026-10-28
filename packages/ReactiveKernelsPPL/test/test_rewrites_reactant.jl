using ReactiveKernels, ReactiveKernelsPPL, Distributions, Test
using DifferentiationInterface, Enzyme, Reactant

@testset "whole-data GLM conditioning: compiled parity and retained structure" begin
    for head in (:NormalIDGLM, :BernoulliLogitGLM, :PoissonLogGLM)
        structures = Dict{String,Int}[]
        for n in (4, 9)
            data = _rewrite_glm_data(head, n)
            model = ReactiveKernelsPPL.RKPPLModel(_rewrite_glm_ast(head), @__MODULE__)
            plan = model(; data.x1, data.x2) | (; data.y)
            built = build_kernel(plan)
            u = [0.1, -0.2, 0.3]
            kernel = prepare_query(built, plan, :sampler)
            query = prepare_sampler(built, plan, u;
                backend = AutoEnzyme(mode = Enzyme.Reverse,
                    function_annotation = Enzyme.Const))
            value, gradient = sampler_value_and_gradient!(query, similar(u), u)
            @test value ≈ _rewrite_glm_oracle(head, data, u[1], u[2:3])
            ru = Reactant.to_rarray(u)
            hlo = repr(Reactant.@code_hlo optimize = false kernel(ru))
            operations = Dict{String,Int}()
            for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
                operations[m.match] = get(operations, m.match, 0) + 1
            end
            @test get(operations, "stablehlo.reduce", 0) > 0
            push!(structures, operations)
            compiled = Reactant.@compile kernel(ru)
            @test Float64(compiled(ru)) ≈ value rtol = 1e-8
            cad = compile_ad_value_and_gradient(query.ad, ru)
            rvalue, rgradient = cad(ru)
            @test Float64(rvalue) ≈ value rtol = 1e-8
            @test Array(rgradient) ≈ gradient rtol = 1e-6
        end
        @test structures[1] == structures[2]
    end
end

@testset "conditioning: native and compiled density, gradients and structure" begin
    structures = Dict{String,Int}[]
    for K in (3, 13)
        model = @rkppl begin
            a ~ Normal(0, 1)
            tau ~ HalfNormal(1)
            b[axes(X, 2)] .~ truncated.(Normal.(a, 1), -1, 1)
            y .~ Normal.(a, tau)
        end
        X = zeros(5, K)
        b = fill(0.2, K)
        y = [0.1, -0.2, 0.3, 0.0, 0.4]
        plan = model(; X) | (; b, y, tau = 0.3)
        built = build_kernel(plan)
        u = [0.4]
        kernel = prepare_query(built, plan, :sampler)
        oracle = logpdf(Normal(), u[1]) +
            logpdf(truncated(Normal(), 0, Inf), 0.3) +
            sum(logpdf.(truncated(Normal(u[1], 1), -1, 1), b)) +
            sum(logpdf.(Normal(u[1], 0.3), y))
        @test Base.invokelatest(kernel, u) ≈ oracle
        query = prepare_sampler(built, plan, u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(query, similar(u), u)
        z = cdf(Normal(u[1], 1), 1) - cdf(Normal(u[1], 1), -1)
        dz = pdf(Normal(), -1 - u[1]) - pdf(Normal(), 1 - u[1])
        @test gradient ≈ [-u[1] + sum(b .- u[1]) - K * dz / z +
            sum(y .- u[1]) / 0.3^2]
        ru = Reactant.to_rarray(u)
        hlo = repr(Reactant.@code_hlo optimize = false kernel(ru))
        operations = Dict{String,Int}()
        for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
            operations[m.match] = get(operations, m.match, 0) + 1
        end
        @test get(operations, "stablehlo.reduce", 0) > 0
        push!(structures, operations)
        compiled = Reactant.@compile kernel(ru)
        @test Float64(compiled(ru)) ≈ value rtol = 1e-9
        cad = compile_ad_value_and_gradient(query.ad, ru)
        rvalue, rgradient = cad(ru)
        @test Float64(rvalue) ≈ value rtol = 1e-9
        @test Array(rgradient) ≈ gradient rtol = 1e-9
        @test b == fill(0.2, K)
    end
    @test structures[1] == structures[2]
end

@testset "conditioned support guards preserve inactive arithmetic" begin
    model = @rkppl begin
        a ~ Normal(0, 1)
        tau ~ HalfNormal(1)
        y .~ Normal.(a, 1)
    end
    y = [0.1, -0.2]
    u = [0.4]
    # The HalfNormal density's invalid side must return -Inf without
    # evaluating transform arithmetic or introducing a sampled tau coordinate.
    plan = model() | (; y, tau = -0.3)
    built = build_kernel(plan)
    query = prepare_sampler(built, plan, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    value, gradient = sampler_value_and_gradient!(query, similar(u), u)
    @test value == -Inf
    @test gradient ≈ [-u[1] + sum(y .- u[1])]
    ru = Reactant.to_rarray(u)
    cad = compile_ad_value_and_gradient(query.ad, ru)
    rvalue, rgradient = cad(ru)
    @test Float64(rvalue) == -Inf
    @test Array(rgradient) ≈ gradient
end

@testset "pins: compiled array values and removed densities" begin
    structures = Dict{String,Int}[]
    for K in (3, 13)
        model = @rkppl begin
            a ~ Normal(0, 1)
            tau ~ HalfNormal(1)
            b[axes(X, 2)] .~ Normal.(0, 1)
            y .~ Normal.(a .+ X * b, tau)
        end
        X = fill(0.1, 5, K)
        b = fill(0.2, K)
        y = [0.1, -0.2, 0.3, 0.0, 0.4]
        plan = model(; X, b, tau = 0.3) | (; y)
        built = build_kernel(plan)
        u = [0.4]
        kernel = prepare_query(built, plan, :sampler)
        query = prepare_sampler(built, plan, u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(query, similar(u), u)
        @test value ≈ logpdf(Normal(), u[1]) +
            sum(logpdf.(Normal.(u[1] .+ X * b, 0.3), y))
        @test gradient ≈ [-u[1] + sum(y .- u[1] .- X * b) / 0.3^2]
        ru = Reactant.to_rarray(u)
        hlo = repr(Reactant.@code_hlo optimize = false kernel(ru))
        operations = Dict{String,Int}()
        for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
            operations[m.match] = get(operations, m.match, 0) + 1
        end
        push!(structures, operations)
        compiled = Reactant.@compile kernel(ru)
        @test Float64(compiled(ru)) ≈ value rtol = 1e-9
        cad = compile_ad_value_and_gradient(query.ad, ru)
        rvalue, rgradient = cad(ru)
        @test Float64(rvalue) ≈ value rtol = 1e-9
        @test Array(rgradient) ≈ gradient rtol = 1e-9
        @test b == fill(0.2, K)
    end
    @test structures[1] == structures[2]
end
