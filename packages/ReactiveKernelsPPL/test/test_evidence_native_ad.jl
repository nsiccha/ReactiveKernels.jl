using Test, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
@isdefined(_EVIDENCE_FAMILY_CASES) || include("evidence_fixtures.jl")
function _evidence_native_gradient(bound,built,k,u)
    sampler=prepare_sampler(built,bound,u;backend=AutoEnzyme(;mode=Enzyme.Reverse))
    value,g=sampler_value_and_gradient!(sampler,similar(u),u)
    @test value ≈ k(u) atol=1e-12
    fd=(k(u .+ 1e-5)-k(u .- 1e-5))/2e-5
    @test only(g) ≈ fd atol=1e-7 rtol=1e-6
end
@testset "native reverse mode for evidence on every univariate family" begin
 for (name,ctor,make_dist,lo,hi,interior) in _EVIDENCE_FAMILY_CASES
  @testset "$name" begin
   for kind in (:truncated,:censored,:interval_censored)
    println("NATIVE_AD_BEGIN ",name," ",kind);flush(stdout)
    ys=kind===:censored ? [lo,interior[end],hi] : kind===:interval_censored ? [lo,(lo+hi)/2] : interior
    wrap=kind===:interval_censored ? Expr(:.,kind,Expr(:tuple,ctor,hi)) : Expr(:.,kind,Expr(:tuple,ctor,lo,hi))
    expr=quote a ~ Normal(0,1); eta=a .+ 0*x; y .~ $wrap end
    bound,built,k,u=_evidence_query(expr,Dict(:y=>ys,:x=>zeros(length(ys))),(a=.2,))
    Base.invokelatest(_evidence_native_gradient,bound,built,k,u)
   end
  end
 end
end
