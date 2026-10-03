using Reactant

function _probability_value_inventory(hlo)
    counts=Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|tensor|cf|math|linalg|memref)\.\w+",hlo)
        counts[m.match]=get(counts,m.match,0)+1
    end
    counts
end

@testset "probability arithmetic retains default primal and reverse graphs" begin
    previous=Dict{Any,Any}()
    for n in (15,31), family in (:Bernoulli,:Binomial), named in (false,true), scalar in (false,true)
        f=_probability_value_fixture(family,n; named,scalar)
        kernel=f.kernel
        ru=Reactant.to_rarray(f.u)
        ad=Base.invokelatest(prepare_ad,kernel,AutoEnzyme(; mode=Enzyme.Reverse),f.u;active=:unconstrained)
        compiled=Reactant.@compile kernel(ru)
        reverse=compile_ad_value_and_gradient(ad,ru)
        value,gradient=reverse(ru)
        @test Float64(compiled(ru)) ≈ f.oracle(f.u)
        @test Float64(value) ≈ f.oracle(f.u)
        @test Array(gradient) ≈ _distributional_findiff(f.oracle,f.u) rtol=2e-5 atol=2e-7
        both=reverse.f
        pair=(_probability_value_inventory(repr(Reactant.@code_hlo kernel(ru))),
            _probability_value_inventory(repr(Reactant.@code_hlo both(ru))))
        @test !isempty(pair[1]) && !isempty(pair[2])
        key=(family,named,scalar)
        n==15 ? (previous[key]=pair) : (@test pair==previous[key])
    end
end
