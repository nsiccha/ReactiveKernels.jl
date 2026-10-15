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
@kernel matrix_subject(q::Vector{Float64},idx::Vector{Int},x::Matrix{Float64})=begin
    gathered=ReactiveKernels._readonly_gather(q,idx)
    terms=plate(gathered,x) do a,z
        -0.5*(z-a)^2
    end
    total::Float64=sum(terms)
    return total
end
@kernel nonlinear_subject(q::Vector{Float64},idx::Vector{Int},x::Vector{Float64})=begin
    gathered=ReactiveKernels._readonly_gather(q,idx)
    terms=plate(gathered,x) do a,z
        sin(a)*z+exp(a)/(1+z*z)
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
    # Include both sides of the mean-run threshold of 8, unequal run lengths,
    # and a group id that reappears after another group (not globally sorted).
    shapes=(Int[],[1],repeat([1,2,3],inner=2),repeat([1,2,3],inner=7),
        repeat([1,2,3],inner=8),repeat([1,2,3],inner=16),
        [fill(2,16);1;3;fill(1,7)],
        [fill(2,17);fill(1,8);fill(3,9);fill(1,13)],repeat([3,1],inner=64))
    kernels=Any[];references=Any[];sources=Dict{Bool,Any}();asts=Dict{Bool,Expr}()
    for (idx,enabled) in zip(shapes,(false,false,false,false,true,true,false,true,true))
        x=[sin(i/3)/2 for i in eachindex(idx)]
        oldidx=copy(idx);oldx=copy(x)
        k=prepare!(cache,subject;bound=(;idx,x))
        push!(kernels,k);push!(references,oracle(q,idx,x))
        verify(k,q,last(references))
        changed=[-.8,.6,1.2]
        verify(k,changed,oracle(changed,idx,x))
        @test idx==oldidx && x==oldx
        if !haskey(sources,enabled)
            sources[enabled]=k.f
            asts[enabled]=k.ast
        else
            @test k.f===sources[enabled]
            @test k.ast==asts[enabled]
        end
        @test occursin("run_partial",string(k.ast))==enabled
        metas=[op.value for op in k.ops if op isa RK._BoundConstant && op.value isa RK._IndexRunData]
        @test length(metas)==1
        @test only(metas)[3]==length(idx)
        @test only(metas)[4]==enabled
        starts,stops=only(metas)[1:2]
        if enabled
            # Runs partition every observation exactly once, each with one
            # constant index, and boundaries separate different indices.
            @test reduce(vcat,(collect(a:b) for (a,b) in zip(starts,stops)))==collect(eachindex(idx))
            @test all(all(==(idx[a]),idx[a:b]) for (a,b) in zip(starts,stops))
            @test all(idx[stops[i]]!=idx[starts[i+1]] for i in 1:length(starts)-1)
        else
            @test isempty(starts) && isempty(stops)
        end
    end
    @test length(sources)==2
    @test sources[false]!==sources[true]
    # The active array length is a runtime property too.
    q4=[q;0.7];idx4=fill(4,32);x4=fill(.1,32)
    k4=prepare!(cache,subject;bound=(idx=idx4,x=x4))
    @test k4.f===sources[true]
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
    # A gather that binding can compute entirely has no residual gather to
    # optimize. Do not derive run metadata just to discard it in the prefix.
    bound_cache=PreparationCache()
    bound_gather=prepare!(bound_cache,subject;bound=(;q,idx))
    @test bound_gather(x)≈oracle(q,idx,x)[1]
    @test all(r->r.op!==RK._index_run_metadata && r.op!==RK._readonly_gather_runs,
        only(values(bound_cache.bound)).plan.recipes)
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

@testset "index runs: retained loops across fresh data sizes" begin
    # No cached template can conceal data-dependent body replication here:
    # prepare afresh as both the number of runs and observations per run grow.
    inventories=Dict{Symbol,Int}[]
    for (nruns,runlength) in ((1,8),(3,16),(9,64))
        idx=repeat(mod1.(1:nruns,3),inner=runlength)
        x=sin.(eachindex(idx))./2
        k=prepare(subject;bound=(;idx,x))
        @test occursin("run_partial",string(k.ast))
        counts=Dict{Symbol,Int}()
        RK._index_run_visit(k.ast) do ex
            counts[ex.head]=get(counts,ex.head,0)+1
        end
        @test get(counts,:for,0)>0
        @test get(counts,:while,0)>0
        push!(inventories,counts)
        verify(k,[.1,-.4,.3],oracle([.1,-.4,.3],idx,x))
    end
    @test all(==(first(inventories)),inventories)
end

@testset "index runs: multidimensional plate keeps its flat fallback" begin
    idx=repeat([3,1,2],inner=8)
    x=reshape(sin.(1:3length(idx)),length(idx),3)
    k=prepare(matrix_subject;bound=(;idx,x))
    for q in ([.1,-.4,.3],[-.8,.6,1.2])
        refs=[oracle(q,idx,x[:,j]) for j in axes(x,2)]
        verify(k,q,(sum(first,refs),sum(last,refs)))
    end
    @test RK._lower_index_runs_native(k.ast,k.ops)==k.ast
end

@testset "index runs: nonlinear cells reuse the same transformation" begin
    for idx in (repeat([2,1,3,1],inner=8),[fill(1,11);fill(3,19)])
        x=cos.(eachindex(idx))
        k=prepare(nonlinear_subject;bound=(;idx,x))
        @test occursin("run_partial",string(k.ast))
        for q in ([.1,-.4,.3],[-.8,.6,1.2])
            value=0.;g=zero(q)
            for i in eachindex(idx)
                a=q[idx[i]];z=x[i]
                value+=sin(a)*z+exp(a)/(1+z*z)
                g[idx[i]]+=cos(a)*z+exp(a)/(1+z*z)
            end
            verify(k,q,(value,g))
        end
    end
end
end
