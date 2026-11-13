# Attribute the matched Bernoulli gap: original primitive, stable primal
# through ordinary reverse, and its joint value/partial graph. No backend
# adapter replacement and no handwritten model derivative.
include("pbr_matched_targets.jl")
using InteractiveUtils

function native_softplus_model(f, theta, X, y)
    eta = X * theta
    total = -sum(abs2, theta)/8
    @inbounds @simd for i in eachindex(y)
        total += y[i]*eta[i] - f(eta[i])
    end
    total
end
native_call(f) = NativeGradient((theta,X,y) -> native_softplus_model(f,theta,X,y))
native_thunk(f, g, args) = () -> f(g, args...)
rk_thunk(op, g, args) = () -> ad_value_and_gradient!(op, g, args...)
pbr_thunk(ld, g, theta) = () -> (PBR.logdensity_and_gradient!(g,ld,theta),g)

function paired(n)
    rng = Xoshiro(819)
    X = randn(rng,n,2); y = Float64.(rand(rng,n) .< .5); theta=[.2,-.15]
    args = (theta,X,y)
    expected = MatchedTargets.reference(:bernoulli,args...)
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    cases = Any[]
    ld = pbr_model(:bernoulli,X,y)
    push!(cases, ("pbr",pbr_thunk(ld,zero(theta),theta)))
    for (label,f) in (("native_original",log1pexp),
            ("native_stable_primal",MatchedTargets.softplus_primal),
            ("native_joint_graph",MatchedTargets.shared_softplus))
        push!(cases,(label,native_thunk(native_call(f),zero(theta),args)))
    end
    for (label,spec) in (("rk_original",MatchedTargets.bernoulli),
            ("rk_stable_primal",MatchedTargets.bernoulli_shared_primal),
            ("rk_joint_graph",MatchedTargets.bernoulli_shared))
        println("PREP ",label," n=",n); flush(stdout)
        op=prepare_mode(spec,backend,args,true,false)
        push!(cases,(label,rk_thunk(op,zero(theta),args)))
    end
    for (label,f) in cases
        @test all(T->T !== Core.Box && T !== Any,fieldtypes(typeof(f)))
        v,g = f()
        @test v ≈ expected[1]
        @test g ≈ expected[2]
        if n == 2048 && haskey(ENV,"SOFTPLUS_IR_DIR")
            mkpath(ENV["SOFTPLUS_IR_DIR"])
            open(joinpath(ENV["SOFTPLUS_IR_DIR"],label*".ll"),"w") do io
                code_llvm(io,f,Tuple{};raw=true,debuginfo=:none)
            end
        end
    end
    for round in 1:5
        for idx in randperm(rng,length(cases))
            label,f=cases[idx]
            measurement("n$n/round$round/$label",f)
        end
    end
end

@testset "Shared softplus attribution" begin
    for x in (-1000.0,-40.0,-1.0,-0.0,0.0,1.0,40.0,1000.0)
        @test MatchedTargets.softplus_primal(x) ≈ log1pexp(x)
        for seed in (-1.3,0.0,1.0)
            f = z -> seed * MatchedTargets.softplus_primal(z)
            derivative = Enzyme.autodiff(Enzyme.Reverse,f,Enzyme.Active(x))[1][1]
            @test derivative ≈ seed * logistic(x)
        end
    end
    for n in (2048,8192)
        paired(n)
    end
end
println("COMPLETE")
