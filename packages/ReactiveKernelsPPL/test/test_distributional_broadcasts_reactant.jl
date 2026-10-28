# Compiled (Reactant) checks for the cases in test_distributional_broadcasts.jl.
using Reactant

@testset "distributional links retain Julia broadcast axes under Reactant" begin
    operations = Dict{Symbol,Tuple{Vector{String},Vector{String}}}()
    for n in (3, 7), family in (:StudentT, :ZeroInflatedBinomial, :Normal,
            :Poisson, :Bernoulli)
        f = _distributional_broadcast_fixture(n, family)
        model = _distributional_model(f.ast, f.data)
        kernel = model.kernel
        ru = Reactant.to_rarray(f.u)
        ad = Base.invokelatest(prepare_ad, kernel,
            AutoEnzyme(; mode = Enzyme.Reverse), f.u; active = :unconstrained)
        compiled = Reactant.@compile kernel(ru)
        cad = compile_ad_value_and_gradient(ad, ru)
        value, gradient = cad(ru)
        @test Float64(compiled(ru)) ≈ f.oracle(f.u)
        @test Float64(value) ≈ f.oracle(f.u)
        @test Array(gradient) ≈ _distributional_findiff(f.oracle, f.u) rtol=2e-5 atol=2e-7
        adcall = cad.f
        ops = [m.match for m in eachmatch(r"stablehlo\.[a-z_]+", string(Reactant.@code_hlo optimize=false kernel(ru)))]
        adops = [m.match for m in eachmatch(r"stablehlo\.[a-z_]+", string(Reactant.@code_hlo optimize=false adcall(ru)))]
        n == 3 ? (operations[family] = (ops, adops)) : (@test (ops, adops) == operations[family])
    end
end
