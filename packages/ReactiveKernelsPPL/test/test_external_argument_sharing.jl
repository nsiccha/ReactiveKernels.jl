using Test, ReactiveKernelsPPL, ReactiveKernels, DifferentiationInterface, Enzyme

module ArgumentSharingFixtures
using ReactiveKernelsPPL
import Distributions
normal_shared(p::AbstractVector, x::Real) = Distributions.Normal(sum(p)+x, 1.0)
normal_scalar(p::Real, x::Real) = Distributions.Normal(p+x, 1.0)
const MIXED = @rkppl begin
    beta[1:3] .~ Normal.(0,1)
    @plate for i in eachindex(y)
        y[i] ~ normal_shared(beta,x[i])
    end
end
const SHARED_SLICE = @rkppl begin
    beta[1:3] .~ Normal.(0,1)
    @plate for i in eachindex(y)
        y[i] ~ normal_shared(beta[1:2],x[i])
    end
end
const SELECTED = @rkppl begin
    beta[1:3] .~ Normal.(0,1)
    @plate for i in 2:3
        y[i] ~ normal_shared(beta,x[i])
    end
end
const GROUPED = @rkppl begin
    beta[1:3] .~ Normal.(0,1)
    @plate for i in eachindex(y)
        y[i] .~ normal_shared(beta,x[i])
    end
end
const SHARED = @rkppl begin
    beta[1:3] .~ Normal.(0,1)
    y .~ normal_shared(beta,0.5)
end
const DOTTED = @rkppl begin
    beta[1:3] .~ Normal.(0,1)
    y .~ normal_scalar.(beta,x)
end
end

_argument_expr_size(x) = 1
_argument_expr_size(x::Expr) = 1 + sum(_argument_expr_size, x.args; init=0)
const _ARGUMENT_SHARING_SHAPES = Dict{String,Tuple{Int,Int}}()

function _argument_sharing_check(label,model,data,observations,kind,used=3; indices=eachindex(observations))
    before=deepcopy((data,observations))
    plan=model(;data...) | (;y=observations)
    built=build_kernel(plan)
    point=[-0.3,0.2,0.4]
    ad=prepare_sampler(built,plan,point;backend=AutoEnzyme(;mode=Enzyme.Reverse),native_forward_batch=false)
    for beta in (point,point .+ 0.1)
        residuals = if kind==:grouped
            [yi-(sum(beta)+data.x[i]) for i in eachindex(observations) for yi in observations[i]]
        elseif kind==:dotted
            observations .- (beta .+ data.x)
        else
            observations[indices] .- (sum(beta[1:used]) .+
                (kind==:shared ? 0.5 : data.x[indices]))
        end
        expected=-0.5*(sum(abs2,beta)+sum(abs2,residuals))-(length(beta)+length(residuals))*log(2pi)/2
        expected_gradient=-beta
        if kind==:dotted
            expected_gradient .+= residuals
        else
            expected_gradient[1:used] .+= sum(residuals)
        end
        gradient=similar(beta)
        value,_=sampler_value_and_gradient!(ad,gradient,beta)
        @test value ≈ expected atol=1e-12
        @test gradient ≈ expected_gradient atol=1e-12
    end
    @test (data,observations)==before
    ppl=kernel_expr(plan,built.layout)
    prepared=code_expr(ad.kernel)
    _ARGUMENT_SHARING_SHAPES[label]=(_argument_expr_size(ppl),_argument_expr_size(prepared))
    out=get(ENV,"RKPPL_ARGUMENT_EVIDENCE","")
    if !isempty(out)
        mkpath(out)
        for (name,ex) in (("ppl",ppl),("prepared",prepared))
            open(joinpath(out,label*"-"*name*".jl"),"w") do io
                show(io,MIME"text/plain"(),ex)
            end
        end
    end
    println("ARGUMENT_SHARING ",label," PASS");flush(stdout)
end

@testset "shared and per-index external constructor arguments" begin
    F=ArgumentSharingFixtures
    for n in (0,2,7)
        x=n==0 ? Float64[] : collect(range(-0.5,0.5;length=n))
        y=cos.(collect(1:n))
        _argument_sharing_check("mixed-"*string(n),F.MIXED,(;x),y,:mixed)
        _argument_sharing_check("slice-"*string(n),F.SHARED_SLICE,(;x),y,:mixed,2)
        groups=[i==1 ? Float64[] : [y[i],y[i]+0.2] for i in 1:n]
        _argument_sharing_check("grouped-"*string(n),F.GROUPED,(;x),groups,:grouped)
    end
    _argument_sharing_check("shared",F.SHARED,(;),[0.2,0.6],:shared)
    _argument_sharing_check("dotted",F.DOTTED,(;x=[-0.2,0.3,0.4]),[0.2,0.6,0.1],:dotted)
    _argument_sharing_check("prefix",F.MIXED,(;x=[-0.2,0.3,0.4]),[0.2,0.6],:mixed)
    _argument_sharing_check("selected",F.SELECTED,(;x=[100.0,0.3,0.4,200.0]),
        [300.0,0.6,0.1,400.0],:mixed;indices=2:3)
    # An authored x[i] is an indexed read, not a broadcastable singleton.
    caught=nothing
    try
        plan=F.MIXED(;x=[0.2]) | (;y=[0.1,0.3])
        built=build_kernel(plan)
        query=prepare_query(built,plan,:sampler)
        Base.invokelatest(query,[0.1,0.2,0.3])
    catch err
        caught=err
    end
    @test caught !== nothing
    println("INDEX_BOUNDARY ",typeof(caught));flush(stdout)
    for prefix in ("mixed-","slice-","grouped-")
        @test _ARGUMENT_SHARING_SHAPES[prefix*"2"] == _ARGUMENT_SHARING_SHAPES[prefix*"7"]
    end
end
