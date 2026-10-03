using Reactant

@testset "scalar Case-A families retain default primal and reverse graphs" begin
    previous=Dict{Any,Any}()
    for n in (15,31), (kind,variant) in [[(k,:plain) for k in _SCALAR_MI_KINDS]...,(:stopping,:gathered),(:stopping,:censored)]
        @testset "$kind / $variant / n=$n" begin
            f=_scalar_mi_fixture(kind,n;variant)
            kernel=f.kernel
            ru=Reactant.to_rarray(f.u)
            ad=Base.invokelatest(prepare_ad,kernel,AutoEnzyme(; mode=Enzyme.Reverse),f.u;active=:unconstrained)
            compiled=Reactant.@compile kernel(ru)
            reverse=compile_ad_value_and_gradient(ad,ru)
            value,gradient=reverse(ru)
            @test Float64(compiled(ru)) ≈ f.oracle(f.u)
            @test Float64(value) ≈ f.oracle(f.u)
            @test Array(gradient) ≈ _distributional_findiff(f.oracle,f.u) rtol=2e-5 atol=2e-7
            @test f.data == f.saved
            both=reverse.f
            pair=(_probability_value_inventory(repr(Reactant.@code_hlo kernel(ru))),
                _probability_value_inventory(repr(Reactant.@code_hlo both(ru))))
            @test !isempty(pair[1]) && !isempty(pair[2])
            n==15 ? (previous[(kind,variant)]=pair) : (@test pair==previous[(kind,variant)])
        end
    end
end
