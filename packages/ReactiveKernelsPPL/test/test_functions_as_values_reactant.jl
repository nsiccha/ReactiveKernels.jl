using Reactant

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
