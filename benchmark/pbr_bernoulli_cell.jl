# Compare a joint local likelihood graph with a joint softplus graph.
# All model gradients still use the backend's ordinary reverse entry point.
include("pbr_matched_targets.jl")

module BernoulliCell
using ReactiveKernels

# The continuous extension in y also exposes its partial, so the owned rule
# remains compositional when both arguments are active. The sign selection
# avoids cancellation in an observed success's positive-logit tail.
@kernel cell_graph(eta::Float64, y::Float64) = begin
    positive = eta >= 0
    q::Float64 = exp(ifelse(positive, -eta, eta))
    offset::Float64 = ifelse(positive, y - 1, y)
    value::Float64 = offset * eta - log1p(q)
    deta::Float64 = offset + ifelse(positive, q, -q) / (1 + q)
    dy::Float64 = eta
    return value, deta, dy
end
const cell = scalar_derivative_rule(cell_graph; primal=:value,
    partials=(eta=:deta, y=:dy), name=:bernoulli_cell)

@kernel model(theta::Vector{Float64}, X::Matrix{Float64}, y::Vector{Float64}) = begin
    eta::Vector{Float64} = X * theta
    terms = plate(eta, y) do ei, yi
        cell(ei, yi)
    end
    total::Float64 = sum(terms) - sum(abs2, theta) / 8
    return total
end

function native(theta, X, y)
    eta = X * theta
    total = -sum(abs2,theta)/8
    @inbounds @simd for i in eachindex(y)
        total += cell(eta[i], y[i])
    end
    total
end
end

cell_rk_thunk(op,g,args) = () -> ad_value_and_gradient!(op,g,args...)
cell_native_thunk(f,g,args) = () -> f(g,args...)
cell_pbr_thunk(ld,g,theta) = () -> (PBR.logdensity_and_gradient!(g,ld,theta),g)

function validate_cell()
    # Independent high precision expression, not the production arithmetic.
    setprecision(BigFloat, 512) do
        for eta in (-1000.0,-40.0,-1.0,-0.0,0.0,1.0,40.0,1000.0),
                y in (0.0,0.3,1.0)
            b = BigFloat(eta); by = BigFloat(y)
            ref = Float64(by*b - log1p(exp(b)))
            dref = Float64(by - 1/(1+exp(-b)))
            v,de,dy = derivative_cut(BernoulliCell.cell,(true,true),eta,y)
            @test v ≈ ref rtol=2e-14
            @test de ≈ dref rtol=2e-14
            @test dy == eta
            for seed in (-1.3,0.0,1.0)
                f = (a,b) -> seed*BernoulliCell.cell(a,b)
                d = Enzyme.autodiff(Enzyme.Reverse,f,Enzyme.Active(eta),Enzyme.Active(y))[1]
                @test d[1] ≈ seed*dref rtol=2e-14
                @test d[2] ≈ seed*eta rtol=2e-14
            end
        end
    end
end

function compare_cell(n)
    rng = Xoshiro(819)
    X=randn(rng,n,2); y=Float64.(rand(rng,n).<.5); theta=[.2,-.15]
    args=(theta,X,y)
    expected=MatchedTargets.reference(:bernoulli,args...)
    backend=AutoEnzyme(;mode=Enzyme.Reverse)
    cases=Any[]
    ld=pbr_model(:bernoulli,X,y)
    push!(cases,("pbr",cell_pbr_thunk(ld,zero(theta),theta)))
    push!(cases,("native_cell",cell_native_thunk(NativeGradient(BernoulliCell.native),zero(theta),args)))
    for (label,spec,on) in (("rk_softplus",MatchedTargets.bernoulli_shared,false),
            ("rk_cell",BernoulliCell.model,false),
            ("rk_cell_plate",BernoulliCell.model,true))
        println("PREP ",label," n=",n); flush(stdout)
        op=prepare_mode(spec,backend,args,true,on)
        push!(cases,(label,cell_rk_thunk(op,zero(theta),args)))
        # Runtime lengths and changed parameters reuse this exact executable.
        for m in (0,3,17)
            a=theta.+.11; A=randn(rng,m,2); b=Float64.(rand(rng,m).<.5)
            original=deepcopy((a,A,b))
            ref=MatchedTargets.reference(:bernoulli,a,A,b)
            v,g=ad_value_and_gradient(op,a,A,b)
            @test v ≈ ref[1]
            @test g ≈ ref[2]
            @test (a,A,b)==original
        end
    end
    for (label,f) in cases
        v,g=f()
        @test v ≈ expected[1]
        @test g ≈ expected[2]
        @test all(T->T !== Core.Box && T !== Any,fieldtypes(typeof(f)))
    end
    for round in 1:5, idx in randperm(rng,length(cases))
        label,f=cases[idx]
        measurement("cell/n$n/round$round/$label",f)
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    @testset "Shared Bernoulli likelihood cell" begin
        validate_cell()
        for n in (2048,8192)
            compare_cell(n)
        end
    end
    println("COMPLETE")
end
