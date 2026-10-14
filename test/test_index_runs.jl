module IndexRunAcceptance
using ReactiveKernels, Enzyme, DifferentiationInterface, Test
include("type_stability_audit.jl")
using .TypeStabilityAudit: findings, ad_call_findings
const RK=ReactiveKernels
@kernel subject(q::Vector{Float64},idx::Vector{Int},x::Vector{Float64})=begin
    gathered=ReactiveKernels._readonly_gather(q,idx)
    terms=plate(gathered,x) do a,z
        -0.5*(z-a)^2
    end
    total::Float64=sum(terms)
    return total
end
@kernel multiple(q::Vector{Float64},idx::Vector{Int},other::Vector{Int},x::Vector{Float64})=begin
    a=ReactiveKernels._readonly_gather(q,idx)
    b=ReactiveKernels._readonly_gather(q,other)
    terms=plate(a,b,x) do left,right,z
        -0.5*(z-left-right)^2
    end
    total::Float64=sum(terms)
    return total
end
function oracle(q,idx,x)
    value=0.;g=zero(q)
    for i in eachindex(idx)
        d=x[i]-q[idx[i]];value-=.5*d*d;g[idx[i]]+=d
    end
    value,g
end
function verify(k,q,ref)
    pristine=copy(q)
    @test k(q)≈ref[1] rtol=1e-12 atol=1e-12
    ad=prepare_ad(k,AutoEnzyme(;mode=Enzyme.Reverse),q;active=:q,native_forward_batch=false)
    @test ad.backend.mode===Enzyme.Reverse
    g=zero(q);v,_=ad_value_and_gradient!(ad,g,q)
    @test v≈ref[1] rtol=1e-12 atol=1e-12
    @test g≈ref[2] rtol=1e-11 atol=1e-11
    @test q==pristine
end
@testset "index runs: binding, native semantics and Reverse" begin
    cache=PreparationCache();q=[.1,-.4,.3]
    shapes=(Int[],[1],repeat([1,2,3],inner=2),repeat([1,2,3],inner=16),
        [fill(2,32);1;3;fill(1,23)],repeat([3,1],inner=64))
    kernels=Any[];references=Any[];source=nothing
    for idx in shapes
        x=[sin(i/3)/2 for i in eachindex(idx)]
        oldidx=copy(idx);oldx=copy(x)
        k=prepare!(cache,subject;bound=(;idx,x))
        push!(kernels,k);push!(references,oracle(q,idx,x))
        verify(k,q,last(references))
        @test idx==oldidx && x==oldx
        if source===nothing
            source=k.f
        else
            @test k.f===source
            @test k.ast==first(kernels).ast
        end
        metas=[op.value for op in k.ops if op isa RK._BoundConstant && op.value isa RK._IndexRunData]
        @test length(metas)==1
        @test only(metas)[4]==length(idx)
        @test only(metas)[5]==any(all(idx[j]==idx[i] for j in i:(i+15)) for i in 1:max(0,length(idx)-15))
    end
    # The active array length is a runtime property too.
    q4=[q;0.7];idx4=fill(4,32);x4=fill(.1,32)
    k4=prepare!(cache,subject;bound=(idx=idx4,x=x4))
    @test k4.f===source
    verify(k4,q4,oracle(q4,idx4,x4))
    # Earlier kernels retain their own constants after later rebindings.
    for (k,ref) in zip(kernels,references)
        @test k(q)≈ref[1] rtol=1e-12 atol=1e-12
    end
    @test occursin("run_partial",string(last(kernels).ast))
    @test RK._lower_index_runs_native(last(kernels).ast,last(kernels).ops)==last(kernels).ast
    @test isempty(findings(last(kernels),(typeof(q),)))
    audit_ad=prepare_ad(last(kernels),AutoEnzyme(;mode=Enzyme.Reverse),q;
        active=:q,native_forward_batch=false)
    @test isempty(ad_call_findings(audit_ad,q))
    idx=repeat([1,2],inner=32);x=fill(.25,length(idx))
    query=prepare(subject;bound=(;idx,x),want=(:gathered,:terms,:total))
    result=query(q)
    @test result[1] isa SubArray
    @test result[1]==q[idx]
    @test result[2]≈[-.5*(x[i]-q[idx[i]])^2 for i in eachindex(idx)]
    @test isempty([op for op in query.ops if op===RK._readonly_gather_runs])
    gather_query=prepare(subject;bound=(;idx,x),want=(:gathered,:total))
    gathered,total=gather_query(q)
    @test gathered isa SubArray
    @test parent(gathered)===q
    @test gathered==q[idx]
    @test total≈oracle(q,idx,x)[1]
    # Distinct index maps vary inside runs: only the selected map is hoisted.
    other=repeat([3,1],length(idx)÷2)
    k=prepare(multiple;bound=(;idx,other,x))
    ref=0.;g=zero(q)
    for i in eachindex(idx)
        d=x[i]-q[idx[i]]-q[other[i]];ref-=.5*d*d
        g[idx[i]]+=d;g[other[i]]+=d
    end
    verify(k,q,(ref,g))
    # Original invalid-index behavior is retained (todo0arze4h).
    invalid=prepare(subject;bound=(idx=fill(4,32),x=fill(.25,32)))
    @test_throws BoundsError invalid(q)
    # Native/tensor bodies keep the same operation table without tracing.
    _,nativeops,_=RK._lower_with_ops(k.plan;inline_embedded=true)
    _,tensorops,_=RK._lower_with_ops(k.plan;inline_embedded=true,tensorized=true)
    @test nativeops==tensorops
end
end
