using Reactant

function _rlkj_hlo_ops(hlo)
    ops = Dict{String,Int}()
    for m in eachmatch(r"stablehlo\.[a-z_]+", repr(hlo))
        ops[m.match] = get(ops, m.match, 0) + 1
    end
    ops
end

@testset "Reactant: retained LKJ transform and reverse" begin
    for eta in (1.0, 2.3)
        structures = Dict{String,Int}[]
        for K in (2, 4, 8)
            fx = _rlkj_case(K, eta)
            u = [0.3 * sin(i) for i in 1:fx.built.layout.total]
            original = deepcopy(fx.data)
            k = prepare_query(fx.built, fx.bound, :sampler)
            ru = Reactant.to_rarray(u)
            ops = _rlkj_hlo_ops(Reactant.@code_hlo optimize = false k(ru))
            @test get(ops, "stablehlo.while", 0) > 0
            push!(structures, ops)
            compiled = Reactant.@compile k(ru)
            @test Float64(compiled(ru)) ≈ _rlkj_oracle(fx, u) rtol = 1e-10
            q = prepare_sampler(fx.built, fx.bound, u;
                backend = AutoEnzyme(; mode = Enzyme.Reverse))
            cad = compile_ad_value_and_gradient(q.ad, ru)
            value, grad = cad(ru)
            @test Float64(value) ≈ _rlkj_oracle(fx, u) rtol = 1e-10
            @test Array(grad) ≈ _rlkj_gradient(fx, u) rtol = 1e-9 atol = 1e-10
            @test fx.data == original
            @test Array(ru) == u
            if eta == 1.0
                factor = prepare(fx.built.spec; have = :unconstrained, want = :L)
                cf = Reactant.@compile factor(ru)
                @test Array(cf(ru)) ≈ lkj_chol_constrain(u, K) rtol = 1e-12
                weights = reshape(collect(1.0:(K * K)), K, K) ./ (K * K)
                loss = v -> sum(weights .* factor(v))
                gradient = v -> only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(loss), v))
                cg = Reactant.@compile gradient(ru)
                h = 1e-5
                fd = map(eachindex(u)) do i
                    up, um = copy(u), copy(u)
                    up[i] += h
                    um[i] -= h
                    (sum(weights .* lkj_chol_constrain(up, K)) -
                     sum(weights .* lkj_chol_constrain(um, K))) / (2h)
                end
                @test Array(cg(ru)) ≈ fd rtol = 1e-7 atol = 1e-8
            end
        end
        @test allequal(structures)
    end
    fx = _rlkj_case(1, 2.3)
    k = prepare_query(fx.built, fx.bound, :sampler)
    ru = Reactant.to_rarray(Float64[])
    compiled = Reactant.@compile k(ru)
    @test Float64(compiled(ru)) ≈ _rlkj_oracle(fx, Float64[])
end
