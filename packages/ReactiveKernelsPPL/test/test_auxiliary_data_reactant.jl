using Reactant

function _aux_data_inventory(hlo)
    counts=Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|tensor|cf|math|linalg|memref)\.\w+",hlo)
        counts[m.match]=get(counts,m.match,0)+1
    end
    counts
end

@testset "auxiliary data and cell scales retain default primal and reverse graphs" begin
    previous=Dict{Any,Any}()
    for n in (15,31)
        fixtures=Any[_aux_data_fixture(family,n; surface) for family in (:StudentT,:ZIP)
            for surface in (false,true)]
        push!(fixtures,_aux_cell_scale_fixture(n))
        for (index,f) in enumerate(fixtures)
            @testset "fixture $index / n=$n" begin
                kernel=f.kernel
                ru=Reactant.to_rarray(f.u)
                ad=Base.invokelatest(prepare_ad,kernel,AutoEnzyme(; mode=Enzyme.Reverse),f.u;
                    active=:unconstrained)
                compiled=Reactant.@compile kernel(ru)
                reverse=compile_ad_value_and_gradient(ad,ru)
                value,gradient=reverse(ru)
                @test Float64(compiled(ru)) ≈ f.oracle(f.u)
                @test Float64(value) ≈ f.oracle(f.u)
                @test Array(gradient) ≈ _distributional_findiff(f.oracle,f.u) rtol=2e-5 atol=2e-7
                both=reverse.f
                pair=(_aux_data_inventory(repr(Reactant.@code_hlo kernel(ru))),
                    _aux_data_inventory(repr(Reactant.@code_hlo both(ru))))
                @test !isempty(pair[1]) && !isempty(pair[2])
                n==15 ? (previous[index]=pair) : (@test pair==previous[index])
                if index==5
                    theta=fill(-1.2,n)
                    invalid=unconstrain(f.built.layout,(; mu=0.2,tau=0.8,theta))
                    ri=Reactant.to_rarray(invalid)
                    badvalue,badgradient=reverse(ri)
                    @test Float64(compiled(ri)) == -Inf
                    @test Float64(badvalue) == -Inf
                    @test Array(badgradient) ≈ _distributional_findiff(f.prior,invalid) rtol=2e-5 atol=2e-7
                end
            end
        end
    end
end
