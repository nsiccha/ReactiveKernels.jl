using Reactant

function _sm_operations(hlo)
    operations = Dict{String,Int}()
    for m in eachmatch(r"(?:stablehlo|enzyme)\.[a-z_]+", repr(hlo))
        operations[m.match] = get(operations, m.match, 0) + 1
    end
    return operations
end

@testset "Reactant: scalar varying margin parity and retained structure" begin
    for kind in (:library, :body)
        recipes, traces, optimized = Int[], Dict{String,Int}[], Any[]
        for (n, G) in ((7, 3), (19, 5))
            bound, built, data = _sm_build(kind, n, G)
            u = [0.2 * sin(i) for i in 1:built.layout.total]
            oracle(w) = _sm_oracle(built, data, kind, w).posterior
            expected = (oracle(u), _findiff_grad(oracle, u))
            push!(recipes, length(built.spec.graph.recipes))
            push!(traces, Base.invokelatest(_pcr_measure, built, bound, u;
                expected, reference = oracle, structure_ad = true))
            ru = Reactant.to_rarray(u)
            kernel = prepare_query(built, bound, :sampler)
            q = prepare_sampler(built, bound, u;
                backend = AutoEnzyme(; mode = Enzyme.Reverse))
            ad = q.ad
            both(w) = ad_value_and_gradient(ad, w)
            push!(optimized, (
                _sm_operations(Reactant.@code_hlo optimize = true kernel(ru)),
                _sm_operations(Reactant.@code_hlo optimize = true both(ru))))
        end
        @test recipes[1] == recipes[2]
        @test traces[1] == traces[2]
        @test optimized[1] == optimized[2]
    end
    bound, built, data = _sm_build(:library, 7, 3; labels = true)
    u = [0.2 * sin(i) for i in 1:built.layout.total]
    oracle(w) = _sm_oracle(built, data, :library, w).posterior
    Base.invokelatest(_pcr_measure, built, bound, u;
        expected = (oracle(u), _findiff_grad(oracle, u)),
        reference = oracle, structure_ad = true)
end
