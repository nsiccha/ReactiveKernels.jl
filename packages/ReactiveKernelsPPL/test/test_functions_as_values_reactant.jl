using Reactant

@testset "Reactant: absorbed declaration dependencies" begin
    for chained in (false, true)
        structures = []
        recipes = Int[]
        for n in (7, 19)
            _FV.POOLS[] = 0
            _, bound, built, cols = _fv_group_axis_case(n; chained)
            original = deepcopy(cols)
            u = [0.15 * sin(i) for i in 1:built.layout.total]
            q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
            ru = Reactant.to_rarray(u)
            kernel, ad = q.kernel, q.ad
            both(w) = ad_value_and_gradient(ad, w)
            ops(hlo) = begin
                counts = Dict{String,Int}()
                for m in eachmatch(r"(?:stablehlo|enzyme)\.[a-z_]+", repr(hlo))
                    counts[m.match] = get(counts, m.match, 0) + 1
                end
                counts
            end
            push!(structures, (
                ops(Reactant.@code_hlo optimize = false kernel(ru)),
                ops(Reactant.@code_hlo optimize = false both(ru)),
                ops(Reactant.@code_hlo optimize = true kernel(ru)),
                ops(Reactant.@code_hlo optimize = true both(ru))))
            @test all(inventory -> !isempty(inventory), structures[end])
            @test get(structures[end][1], "stablehlo.reduce", 0) > 0
            push!(recipes, length(built.spec.graph.recipes))
            primal = Reactant.@compile kernel(ru)
            compiled = compile_ad_value_and_gradient(ad, ru)
            for w in (u, u .+ 0.1)
                rw = Reactant.to_rarray(w)
                value, grad = sampler_value_and_gradient!(q, similar(w), w)
                cvalue, cgrad = compiled(rw)
                @test value ≈ _fv_group_axis_reference(built, cols, w)
                @test Float64(primal(rw)) ≈ value rtol = 1e-9
                @test Float64(cvalue) ≈ value rtol = 1e-9
                @test Array(cgrad) ≈ grad rtol = 1e-8 atol = 1e-9
                @test Array(rw) == w
            end
            @test _FV.POOLS[] == Int(chained)
            @test cols == original
        end
        @test recipes[1] == recipes[2]
        @test structures[1] == structures[2]
    end
end

@testset "Reactant: prepared records beside declared arrays" begin
    for weights in (:vector, :matrix), kind in (:namedtuple, :tuple)
        structures = Dict{String,Int}[]
        for n in (3, 7)
            a, k = collect(range(0.5, 1.5; length = n)), collect(1:n)
            sched = kind === :namedtuple ? (; a, k) : (a, k)
            oi = [n, 1, n, 2]
            cols = Dict{Symbol,ColumnData}(:s => [sched], :oi => oi,
                :y => [0.2, -0.1, 0.3, 0.5])
            ast = quote
                b ~ Normal(0, 1)
                sc = take_schedule(s)
                reads = weighted_reads(sc, w, b)
                y .~ Normal.(reads[oi], 1)
            end
            pushfirst!(ast.args, weights === :vector ? :(w[1:$n] .~ Normal.(0, 1)) :
                :(w[1:$n, 1:1] .~ Normal.(0, 1)))
            _FV.UNWRAPS[] = 0
            _, bound, built = _fv_build(ast, cols)
            u = [0.1 * i - 0.2 for i in 1:built.layout.total]
            q = prepare_sampler(built, bound, u; backend = _FV_BACKEND)
            @test _FV.UNWRAPS[] == 1
            value, grad = sampler_value_and_gradient!(q, similar(u), u)
            ru = Reactant.to_rarray(u)
            hlo = repr(Reactant.@code_hlo optimize = false q.kernel(ru))
            ops = Dict{String,Int}()
            for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
                ops[m.match] = get(ops, m.match, 0) + 1
            end
            @test get(ops, "stablehlo.reduce", 0) > 0
            push!(structures, ops)
            compiled = Reactant.@compile q.kernel(ru)
            @test Float64(compiled(ru)) ≈ value rtol = 1e-9
            cad = compile_ad_value_and_gradient(q.ad, ru)
            rvalue, rgrad = cad(ru)
            @test Float64(rvalue) ≈ value rtol = 1e-9
            @test Array(rgrad) ≈ grad rtol = 1e-9
            @test _FV.UNWRAPS[] == 1
            @test a == collect(range(0.5, 1.5; length = n))
            @test k == collect(1:n)
        end
        @test structures[1] == structures[2]
    end
end
