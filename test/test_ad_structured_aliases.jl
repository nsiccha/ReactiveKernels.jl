module ADStructuredAliasTests
using ReactiveKernels,Enzyme,DifferentiationInterface,Test,Random
const EXT=Base.get_extension(ReactiveKernels,:ReactiveKernelsEnzymeExt)
const ShadowIntervals = EXT._ShadowIntervals
const ShadowInterval = EXT._ShadowInterval
const interval_destinations! = EXT._interval_destinations!
const sort_intervals! = EXT._sort_shadow_intervals!
function guard_bytes(w,p,g,c)
    interval_destinations!(w,p,g,c)
    @allocated interval_destinations!(w,p,g,c)
end
function small_guard_bytes(p,g,c)
    EXT._structured_destinations(p,g,c)
    @allocated EXT._structured_destinations(p,g,c)
end
@testset "small selectors with large context tuples" begin
    point=(ones(3),Ref(1.)); gradient=(zeros(3),Ref(0.))
    for count in (1,8,128)
        arrays=ntuple(_->ones(3),count)
        mixed=ntuple(i->isodd(i) ? ones(3) : Ref(1.),count)
        for context in (arrays,mixed)
            contexts=(Constant(context),)
            @test EXT._structured_destinations(point,gradient,contexts)
            @test small_guard_bytes(point,gradient,contexts)==0
            aliased=Base.setindex(context,gradient[1],1)
            @test !EXT._structured_destinations(point,gradient,(Constant(aliased),))
        end
    end
end
@testset "interval guard equivalence and ownership" begin
    rng=MersenneTwister(819)
    work=ShadowIntervals()
    for count in (1,2,3,8,32,128)
        for trial in 1:20
            len=trial%5==0 ? 0 : 8
            pool=[randn(rng,len) for _ in 1:2count+2]
            point=Tuple(pool[1:count]); gradient=Tuple(pool[count+1:2count])
            contexts=(Constant(pool[end-1]),Constant((pool[end],Ref((pool[end-1],)))))
            @test interval_destinations!(work,point,gradient,contexts)
            @test guard_bytes(work,point,gradient,contexts)==0
            if trial%2==0
                gradient=Base.setindex(gradient,point[1],rand(rng,1:count))
            else
                contexts=(Constant(first(gradient)),)
            end
            expected=count<=8 ? EXT._structured_destinations(point,gradient,contexts) : len==0
            @test interval_destinations!(work,point,gradient,contexts)==expected
        end
    end
    a=collect(1.0:12)
    GC.@preserve a begin
        x=unsafe_wrap(Array,pointer(a),(6,);own=false)
        overlap=unsafe_wrap(Array,pointer(a,4),(6,);own=false)
        adjacent=unsafe_wrap(Array,pointer(a,7),(6,);own=false)
        @test !interval_destinations!(work,(x,),(overlap,),(Constant(0.),))
        @test interval_destinations!(work,(x,),(adjacent,),(Constant(0.),))
    end
    r=Ref(2.)
    GC.@preserve r begin
        alias=unsafe_wrap(Array,Base.unsafe_convert(Ptr{Float64},r),(1,);own=false)
        @test !interval_destinations!(work,(r,ones(1)),(Ref(0.),alias),(Constant(0.),))
    end
    for n in (0,1,2,7,128,4096)
        ranges=[ShadowInterval(rand(rng,UInt),0,0x01) for _ in 1:n]
        expected=sort([r.lo for r in ranges])
        @test [r.lo for r in sort_intervals!(ranges)]==expected
    end
end

function mixed_interval_bytes(w,p,g,c)
    interval_destinations!(w,p,g,c)
    @allocated interval_destinations!(w,p,g,c)
end
@testset "mixed scalar and array interval inputs" begin
    for n in (3,8,32,128)
        point=ntuple(i->isodd(i) ? ones(1) : Ref(1.),n)
        gradient=ntuple(i->isodd(i) ? zeros(1) : Ref(0.),n)
        contexts=(Constant((ones(1),Ref((ones(1),)))),)
        work=ShadowIntervals()
        @test interval_destinations!(work,point,gradient,contexts)
        bytes=mixed_interval_bytes(work,point,gradient,contexts)
        println((;n,bytes))
        @test bytes==0
        @test !interval_destinations!(work,point,Base.setindex(gradient,point[1],1),contexts)
        @test !interval_destinations!(work,point,gradient,(Constant(Ref{Any}(1.)),))
    end
end

using ReactiveKernels,Enzyme,DifferentiationInterface,Test
@kernel many_buffer_objective(x::Vector{Float64},z::Vector{Float64},u::Vector{Float64},v::Vector{Float64},w::Vector{Float64},data::Vector{Float64}) = begin
    total::Float64=sum(abs2,x)+2sum(abs2,z)+3sum(abs2,u)+4sum(abs2,v)+5sum(abs2,w)+sum(data)
    return total
end
function many_bytes(p,d,args::Vararg{Any,N}) where N
    ad_value_and_gradient!(p,d,args...)
    @allocated ad_value_and_gradient!(p,d,args...)
end
@testset "prepared gradients with interval alias storage" begin
    backend=AutoEnzyme(mode=Enzyme.Reverse)
    # Registered Enzyme still allocates its tuple-zeroing IdSet (snag
    # make-zero-tuple-d96fee1c); keep numerical checks on both backends.
    zeroing_repaired=isdefined(Enzyme,:has_dogfood_repair) &&
        Enzyme.has_dogfood_repair(:make_zero_tuple_points)
    kernel=prepare(many_buffer_objective;want=:total)
    p=prepare_ad(kernel,backend,ntuple(_->Float64[],6)...;active=(:v,:x,:u,:z,:w))
    p2=prepare_ad(kernel,backend,ntuple(_->Float64[],6)...;active=(:v,:x,:u,:z,:w))
    @test p.storage.buffers.ranges !== p2.storage.buffers.ranges
    for n in (0,1,7,128,2048,7)
        args=ntuple(j->fill(Float64(j),n),6)
        d=ntuple(_->zeros(n),5)
        value,reference=ad_value_and_gradient(p,args...)
        actual,out=ad_value_and_gradient!(p,d,args...)
        @test value≈actual && out===d
        @test all(map(isapprox,d,reference))
        bytes=many_bytes(p,d,args...)
        if zeroing_repaired
            @test bytes==0
        else
            @test_broken bytes==0
        end
        # Destination aliases an inactive context: copy after differentiation.
        aliased=Base.setindex(d,args[end],3)
        actual,out=ad_value_and_gradient!(p,aliased,args...)
        @test value≈actual && out===aliased
        @test all(map(isapprox,aliased,reference))
        # Aliased active inputs and destinations preserve ordinary copy order.
        same_inputs=(args[1],args[1],args[3],args[4],args[5],args[6])
        value,reference=ad_value_and_gradient(p,same_inputs...)
        actual,out=ad_value_and_gradient!(p,d,same_inputs...)
        @test value≈actual && all(map(isapprox,d,reference))
        same_dest=Base.setindex(d,d[1],4)
        actual,out=ad_value_and_gradient!(p,same_dest,same_inputs...)
        @test value≈actual && same_dest[1]≈reference[4]
    end
end

end
