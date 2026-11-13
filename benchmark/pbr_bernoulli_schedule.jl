# Diagnostic controls only. No production handwritten derivative or adapter.
include("pbr_matched_targets.jl")
using InteractiveUtils
const SCHEDULE_OUTPUT_DIR = get(ENV, "PBR_SCHEDULE_OUTPUT_DIR", mktempdir())
mkpath(SCHEDULE_OUTPUT_DIR)
println("ARTIFACTS ", SCHEDULE_OUTPUT_DIR)
@assert Base.get_extension(RK,:ReactiveKernelsEnzymeExt)!==nothing
@assert Base.get_extension(RK,:ReactiveKernelsMutatingFunctionsExt)!==nothing
println("INTEGRATION_EXTENSIONS loaded Enzyme and MutatingFunctions; RK source=",pkgdir(RK))
println("PROVENANCE Julia=",VERSION," Enzyme=",pkgdir(Enzyme)," library=",Enzyme.Compiler.Enzyme_jll.libEnzyme_path)
flush(stdout)

module LikelihoodSchedule
using ReactiveKernels
@kernel likelihood(eta::Vector{Float64}, y::Vector{Float64}) = begin
    cells = plate(eta,y) do e,yi
        yi*e-rk_log1pexp(e)
    end
    value::Float64=sum(cells)
    return value
end
function native(eta,y)
    value=0.0
    @inbounds @simd for i in eachindex(eta,y)
        value+=y[i]*eta[i]-rk_log1pexp(eta[i])
    end
    value
end
end

# Same scalar arithmetic as PBR; independently timed scheduling controls.
@inline function parts(e)
    q=exp(-abs(e))
    max(e,zero(e))+log1p(q), ifelse(e>=0,1.0,q)/(1+q)
end
function fused!(g,eta,y)
    value=0.0
    @inbounds @simd for i in eachindex(eta,y,g)
        v,d=parts(eta[i])
        value+=y[i]*eta[i]-v
        g[i]=y[i]-d
    end
    value,g
end
function owned_fused!(g,eta,y)
    value=0.0
    @inbounds @simd for i in eachindex(eta,y,g)
        v,d=derivative_cut(rk_log1pexp,(true,),eta[i])
        value+=y[i]*eta[i]-v
        g[i]=y[i]-d
    end
    value,g
end
function recompute!(g,eta,y)
    value=0.0
    @inbounds @simd for i in eachindex(eta,y)
        v,_=parts(eta[i]); value+=y[i]*eta[i]-v
    end
    @inbounds @simd for i in eachindex(eta,y,g)
        _,d=parts(eta[i]);g[i]=y[i]-d
    end
    value,g
end
function saved!(g,buffer,eta,y)
    v,_=fused!(buffer,eta,y)
    copyto!(g,buffer)
    v,g
end
function native_reverse!(g,eta,y)
    fill!(g,0.0)
    result=Enzyme.autodiff(Enzyme.ReverseWithPrimal,LikelihoodSchedule.native,
        Enzyme.Active,Enzyme.Duplicated(eta,g),Enzyme.Const(y))
    result[2],g
end
function full_fused!(g,eta,deta,theta,X,y)
    mul!(eta,X,theta)
    v,_=fused!(deta,eta,y)
    mul!(g,transpose(X),deta)
    @inbounds for j in eachindex(theta,g)
        v-=theta[j]^2/8
        g[j]-=theta[j]/4
    end
    v,g
end

eta_thunk(f,g,eta,y)=()->f(g,eta,y)
saved_thunk(g,b,e,y)=()->saved!(g,b,e,y)
rk_eta_thunk(op,g,e,y)=()->ad_value_and_gradient!(op,g,e,y)
pbr_eta_thunk(ld,e)=()->(PBR._obs_loop!(ld,(e,)),ld.ws[1].w)
full_thunk(g,e,d,t,X,y)=()->full_fused!(g,e,d,t,X,y)
pbr_full_thunk(ld,g,t)=()->(PBR.logdensity_and_gradient!(g,ld,t),g)
rk_full_thunk(op,g,args)=()->ad_value_and_gradient!(op,g,args...)

function compare_schedule(n)
    rng=Xoshiro(819)
    X=randn(rng,n,2); y=Float64.(rand(rng,n).<.5); theta=[.2,-.15]
    eta=X*theta; original=deepcopy((eta,y,X,theta))
    ld=pbr_model(:bernoulli,X,y)
    expected=(sum(y.*eta-log1pexp.(eta)),y-logistic.(eta))
    backend=AutoEnzyme(;mode=Enzyme.Reverse)
    cases=Any[("pbr_likelihood",pbr_eta_thunk(ld,eta)),
        ("fused",eta_thunk(fused!,zero(eta),eta,y)),
        ("owned_fused",eta_thunk(owned_fused!,zero(eta),eta,y)),
        ("recompute",eta_thunk(recompute!,zero(eta),eta,y)),
        ("saved_copy",saved_thunk(zero(eta),zero(eta),eta,y)),
        ("native_reverse",eta_thunk(native_reverse!,zero(eta),eta,y))]
    for enabled in (false,true)
        println("PREP likelihood plate=",enabled," n=",n);flush(stdout)
        k=with_plate_reverse(enabled) do
            prepare_nonallocating(LikelihoodSchedule.likelihood,eta,y)
        end
        op=with_plate_reverse(enabled) do
            prepare_ad(k,backend,eta,y;active=:eta)
        end
        push!(cases,("rk_plate$enabled",rk_eta_thunk(op,zero(eta),eta,y)))
        for m in (0,3,17)
            e=sin.(collect(1:m));b=Float64.(isodd.(1:m));g=fill(99.0,m)
            v,got=ad_value_and_gradient!(op,g,e,b)
            @test v≈sum(b.*e-log1pexp.(e);init=0.0)
            @test got≈b-logistic.(e)
            @test got===g
        end
        if n==2048
            open(joinpath(SCHEDULE_OUTPUT_DIR,"schedule-rk-plate$enabled.txt"),"w") do io
                print(io,code_expr(k))
            end
        end
    end
    for (label,f) in cases
        @test all(T->T!==Core.Box&&T!==Any,fieldtypes(typeof(f)))
        v,g=f();@test v≈expected[1];@test g≈expected[2]
    end
    for round in 1:5, idx in randperm(rng,length(cases))
        label,f=cases[idx];measurement("schedule/n$n/round$round/$label",f)
    end
    args=(theta,X,y)
    full=Any[("pbr",pbr_full_thunk(ld,zero(theta),theta)),
        ("fused_blas",full_thunk(zero(theta),zero(eta),zero(eta),theta,X,y))]
    for enabled in (false,true)
        op=prepare_mode(MatchedTargets.bernoulli_shared,backend,args,true,enabled)
        push!(full,("rk_plate$enabled",rk_full_thunk(op,zero(theta),args)))
    end
    ref=MatchedTargets.reference(:bernoulli,args...)
    for (_,f) in full
        v,g=f();@test v≈ref[1];@test g≈ref[2]
    end
    for round in 1:5,idx in randperm(rng,length(full))
        label,f=full[idx];measurement("full/n$n/round$round/$label",f)
    end
    @test (eta,y,X,theta)==original
end

@testset "Bernoulli schedule attribution" begin
    for f in (fused!,owned_fused!,recompute!,native_reverse!)
        v,g=f(Float64[],Float64[],Float64[])
        @test v==0.0;@test isempty(g)
    end
    for n in (2048,8192)
        compare_schedule(n)
    end
end
println("NUMERICAL AND TIMING COMPLETE");flush(stdout)
open(joinpath(SCHEDULE_OUTPUT_DIR,"schedule-native-ad.ll"),"w") do io
    Enzyme.Compiler.enzyme_code_llvm(io,LikelihoodSchedule.native,Enzyme.Active,
        Tuple{Enzyme.Duplicated{Vector{Float64}},Enzyme.Const{Vector{Float64}}};
        returnPrimal=true,dump_module=true,debuginfo=:none)
end
open(joinpath(SCHEDULE_OUTPUT_DIR,"schedule-fused.ll"),"w") do io
    code_llvm(io,fused!,Tuple{Vector{Float64},Vector{Float64},Vector{Float64}};
        dump_module=true,debuginfo=:none)
end
println("COMPLETE")
