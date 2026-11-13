# Production PPL and distribution paths on the same target as PBR.
# Run before and after a primitive change; BENCH_LABEL identifies the source.
include("pbr_matched_targets.jl")
using ReactiveKernelsPPL
using ReactiveKernelsDistributionKernels
const DKS = ReactiveKernelsDistributionKernels.DistributionKernelSources

module ProductionBernoulli
using ReactiveKernels, ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: bernoulli
@kernel model(theta::Vector{Float64}, X::Matrix{Float64}, y::Vector{Bool}) = begin
    eta::Vector{Float64} = X * theta
    cells = plate(eta, y) do e, observed
        bernoulli(; logit=e).logpdf(observed)
    end
    value::Float64 = sum(cells) - sum(abs2,theta)/8
    return value
end
end

const production_ppl = @rkppl begin
    beta[1:2] .~ Normal.(0, 2)
    eta = X * beta
    y .~ BernoulliLogit.(eta)
end
production_ppl_thunk(q,g,u) = () -> sampler_value_and_gradient!(q,g,u)
production_rk_thunk(q,g,args) = () -> ad_value_and_gradient!(q,g,args...)
production_pbr_thunk(q,g,u) = () -> (PBR.logdensity_and_gradient!(g,q,u),g)

function production_comparison(n)
    rng=Xoshiro(819)
    X=randn(rng,n,2); y=Vector{Bool}(rand(rng,n).<.5); theta=[.2,-.15]
    expected=MatchedTargets.reference(:bernoulli,theta,X,Float64.(y))
    backend=AutoEnzyme(;mode=Enzyme.Reverse)
    cases=Any[]
    for on in (false,true)
        println("PREP distribution plate=",on," n=",n); flush(stdout)
        op=prepare_mode(ProductionBernoulli.model,backend,(theta,X,y),true,on)
        push!(cases,("distribution_plate$on",production_rk_thunk(op,zero(theta),(theta,X,y)),0.0))
    end
    println("PREP PPL n=",n); flush(stdout)
    bound=production_ppl(;X) | (;y)
    built=build_kernel(bound)
    u=unconstrain(built.layout,(;beta=theta))
    q=prepare_sampler(built,bound,u;backend,native_forward_batch=false)
    push!(cases,("ppl",production_ppl_thunk(q,zero(u),u),-2log(2sqrt(2pi))))
    if n>0
        ld=pbr_model(:bernoulli,X,Float64.(y))
        push!(cases,("pbr",production_pbr_thunk(ld,zero(theta),theta),0.0))
    end
    for (_,f,c) in cases
        value,grad=f()
        @test value ≈ expected[1]+c
        @test grad ≈ expected[2]
    end
    if n>=2048
        for round in 1:5, idx in randperm(rng,length(cases))
            label,f,_=cases[idx]
            measurement("$(get(ENV,"BENCH_LABEL","unspecified"))/n$n/round$round/$label",f)
        end
    end
end

@testset "Production Bernoulli PPL comparison" begin
    for n in (0,3,2048,8192)
        production_comparison(n)
    end
end
println("COMPLETE")
