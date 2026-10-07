# Compiled (Reactant) checks for the cases in test_owned_evidence_tails.jl.
using Reactant

function _owned_tail_measure(f,v,expected,data...)
    native=only(Enzyme.gradient(Enzyme.Reverse,x->f(x,data...),v))
    rv=Reactant.to_rarray(v)
    rdata=map(Reactant.to_rarray,data)
    hlo=repr(Reactant.@code_hlo optimize=false f(rv,rdata...))
    @test Float64((Reactant.@compile f(rv,rdata...))(rv,rdata...))≈expected atol=2e-12
    gradient(v,d...)=only(Enzyme.gradient(Enzyme.Reverse,x->f(x,d...),v))
    @test Array((Reactant.@compile gradient(rv,rdata...))(rv,rdata...))≈native atol=1e-9 rtol=1e-8
    return count("stablehlo.while",hlo),count("stablehlo.if",hlo)
end

@testset "owned beta-binomial tails retain count loops" begin
    structures=[_owned_tail_measure(f,v,expected,data...)
        for (f,v,expected,data) in _owned_beta_binomial_tail_cases()]
    @test all(first(s)>0 for s in structures)
    @test structures[1]==structures[2]
end

@testset "owned stopping-ordinal tails retain level loops" begin
    structures=[_owned_tail_measure(f,v,expected,data...)
        for (f,v,expected,data) in _owned_stopping_ordinal_tail_cases()]
    @test all(first(s)>0 for s in structures)
    @test structures[1]==structures[2]
end

@testset "owned inverse-Gaussian tails" begin
    for (f,v,expected,data) in _owned_inverse_gaussian_tail_cases()
        _owned_tail_measure(f,v,expected,data...)
    end
end

@testset "categorical tails retain support loops" begin
    structures=[_owned_tail_measure(f,v,expected,data...)
        for (f,v,expected,data) in _owned_categorical_tail_cases()]
    @test all(first(s)>0 for s in structures)
    @test structures[1]==structures[2]
end
