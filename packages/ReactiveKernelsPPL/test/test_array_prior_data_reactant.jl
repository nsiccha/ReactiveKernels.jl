using Reactant

# Native density and independent gradient oracles live in test_array_prior_data.jl.
# Centered multivariate Normal slices retain their existing native-only boundary.
@testset "Reactant: whole data in declared-array priors" begin
    direct = :(begin
        z[levels(k)] .~ Normal.(mu0, 1)
        y .~ Normal.(z[k], 0.7)
    end)
    composed = :(begin
        a ~ Normal(0, 1)
        m0 = mu0 .+ a
        z[levels(k)] .~ Normal.(m0, 1)
        y .~ Normal.(z[k], 0.7)
    end)
    concentration = :(begin
        eachrow(P[levels(k), 1:3]) .~ Dirichlet(alpha)
        y .~ Normal.(P[k, 1], 0.7)
    end)
    # Two-axis arrays read per-element arguments of their own size.
    elementwise = :(begin
        B[levels(k), 1:2] .~ Normal.(LOC, SC)
        y .~ Normal.(B[k, 1], 0.7)
    end)
    live = :(begin
        a ~ Normal(0, 1)
        M = LOC .+ a
        B[levels(k), 1:2] .~ Normal.(M, SC)
        y .~ Normal.(B[k, 2], 0.7)
    end)
    means(K) = Dict(:mu0 => [0.2 * sin(i) for i in 1:K])
    matrices(K) = Dict(:LOC => [0.2 * sin(i + j) for i in 1:K, j in 1:2],
        :SC => [1.0 + 0.1 * (i + j) for i in 1:K, j in 1:2])
    for (ast, extra, coordinates) in ((direct, means, identity),
            (composed, means, K -> K + 1),
            (concentration, K -> Dict(:alpha => [1.2, 2.1, 0.8]), K -> 2K),
            (elementwise, matrices, K -> 2K), (live, matrices, K -> 2K + 1))
        structures = Dict{String,Int}[]
        recipes = Int[]
        for (K, n) in ((3, 8), (5, 17))
            data = merge(_apd_data(n, K), extra(K))
            original = deepcopy(data)
            fx = _apd_build(ast, data)
            @test fx.bound.n_obs == n
            @test fx.built.layout.total == coordinates(K)
            push!(recipes, length(fx.built.spec.graph.recipes))
            u = [0.25 * cos(i) for i in 1:fx.built.layout.total]
            kern = prepare_query(fx.built, fx.bound, :sampler)
            ru = Reactant.to_rarray(u)
            hlo = repr(Reactant.@code_hlo optimize = false kern(ru))
            ops = Dict{String,Int}()
            for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
                ops[m.match] = get(ops, m.match, 0) + 1
            end
            @test get(ops, "stablehlo.reduce", 0) > 0
            push!(structures, ops)
            compiled = Reactant.@compile kern(ru)
            @test Float64(compiled(ru)) ≈ Base.invokelatest(kern, u) rtol = 1e-9
            q = prepare_sampler(fx.built, fx.bound, u;
                backend = AutoEnzyme(; mode = Enzyme.Reverse))
            value, grad = sampler_value_and_gradient!(q, similar(u), u)
            cad = compile_ad_value_and_gradient(q.ad, ru)
            rvalue, rgrad = cad(ru)
            @test Float64(rvalue) ≈ value rtol = 1e-9
            @test Array(rgrad) ≈ grad rtol = 1e-9
            @test data == original
        end
        # Growing both observation count and the data-derived array axis
        # must retain the same graph and backend operations/control regions.
        @test recipes[1] == recipes[2]
        @test structures[1] == structures[2]
    end
end
