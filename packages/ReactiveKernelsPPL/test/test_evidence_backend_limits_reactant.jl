using Test, Reactant, SpecialFunctions, ReactiveKernelsPPL
@isdefined(_EVIDENCE_FAMILY_CASES) || include("evidence_fixtures.jl")

function _evidence_missing_primitive(k,u,primitive)
    ru=Reactant.to_rarray(u)
    err=try
        Reactant.@code_hlo optimize=false k(ru)
        nothing
    catch e
        e
    end
    @test_broken err===nothing
    @test err isa MethodError && err.f===primitive &&
        any(arg->arg isa Reactant.TracedRNumber,err.args)
end

@testset "evidence numerical backend limitations remain precise" begin
    for (name,ctor,make_dist,lo,hi,interior) in _EVIDENCE_FAMILY_CASES
        name in (:gamma,:beta,:poisson,:nb1,:nb2,:zip,:hurdle,:vonmises) || continue
        expr=quote a ~ Normal(0,1);eta=a .+ 0*x;y .~ interval_censored.($ctor,$hi) end
        data=Dict(:y=>[lo,(lo+hi)/2],:x=>zeros(2))
        bound,built,k,u=_evidence_query(expr,data,(a=.2,))
        @test isfinite(Base.invokelatest(k,u))
        primitive=name in (:gamma,:poisson,:zip,:hurdle) ? SpecialFunctions.gamma_inc :
            name===:vonmises ? SpecialFunctions.besselix : SpecialFunctions.beta_inc
        Base.invokelatest(_evidence_missing_primitive,k,u,primitive)
    end
end
