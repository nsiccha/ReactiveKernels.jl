using Reactant

function _cap_ordinal_operations(hlo)
    out = Dict{String, Int}()
    for m in eachmatch(r"stablehlo\.[a-z_]+", hlo)
        out[m.match] = get(out, m.match, 0)+1
    end
    out
end

@testset "Reactant: declared ordinal categories and incomplete samples" begin
    for kind in (:ordered, :cumulative, :stopping)
        primal, reverse = Dict{String, Int}[], Dict{String, Int}[]
        for n in (3, 6, 18)
            fx = _cap_ordinal_fixture(kind, n, (1, 3))
            ru = Reactant.to_rarray(fx.u)
            kernel = fx.sampler.kernel
            # Compare repeated category profiles (three versus nine rows
            # per observed category). The three-row sample also exercises
            # a single-row category arm in the numerical checks below.
            n == 3 || push!(primal, _cap_ordinal_operations(repr(Reactant.@code_hlo optimize=false kernel(ru))))
            cp = Reactant.@compile kernel(ru)
            @test Float64(cp(ru)) ≈ fx.oracle(fx.u)
            cad = compile_ad_value_and_gradient(fx.sampler.ad, ru)
            rv, rg = cad(ru)
            @test Float64(rv) ≈ fx.oracle(fx.u)
            @test Array(rg) ≈ _cap_ordinal_fd(fx.oracle, fx.u) rtol=5e-6
            grad = v -> only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(kernel), v))
            n == 3 || push!(reverse, _cap_ordinal_operations(repr(Reactant.@code_hlo grad(ru))))
        end
        @test primal[1] == primal[2]
        @test reverse[1] == reverse[2]
    end
end
