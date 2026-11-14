using ReactiveKernels, DifferentiationInterface, Enzyme, Test

@kernel structured_buffer_objective(x::Vector{Float64}, z::Vector{Float64}, a::Float64) = begin
    total::Float64 = a * (sum(abs2,x) + sum(z))
    return total
end

@kernel structured_no_context(x::Vector{Float64}, a::Float64) = begin
    total::Float64 = a * sum(abs2,x)
    return total
end

@kernel structured_tuple_port(xs::Tuple, scale::Float64) = begin
    total::Float64 = scale * sum(sum, xs)
    return total
end

_structured_outcome(f) = try
    (; value=f(), error=nothing)
catch e
    (; value=nothing, error=e)
end
@testset "tuple-valued ports preserve backend preparation checks" begin
    backend=AutoEnzyme(mode=Enzyme.Reverse)
    p=prepare_ad(structured_tuple_port,backend,(ones(3),ones(3)),2.;
        active=:xs,want=:total)
    for count in (1,2,4,2)
        xs=ntuple(j->fill(Float64(j),3),count)
        dest=ntuple(_->zeros(3),count)
        reference=_structured_outcome(()->ad_value_and_gradient(p,xs,2.))
        actual=_structured_outcome(()->ad_value_and_gradient!(p,dest,xs,2.))
        # Changing tuple length changes the point type. Preserve whatever
        # check the backend's ordinary prepared call makes; the alias guard
        # must not replace that diagnostic with a missing-workspace error.
        @test typeof(actual.error)===typeof(reference.error)
        if reference.error===nothing
            value,gradient=reference.value
            result,returned=actual.value
            @test value≈result && returned===dest
            @test all(map(isapprox,dest,gradient))
        end
        count==2 && @test reference.error===nothing
    end
end

@testset "structured destinations without a DI context" begin
    x=[1.,2.,3.]; a=2.
    p=prepare_ad(structured_no_context,AutoEnzyme(mode=Enzyme.Reverse),x,a;
        active=(:x,:a),want=:total)
    d=(zero(x),Ref(0.))
    value,reference=ad_value_and_gradient(p,x,a)
    actual,returned=ad_value_and_gradient!(p,d,x,a)
    @test actual≈value
    @test returned===d
    @test d[1]≈reference[1] && d[2][]≈reference[2]
    @test x==[1.,2.,3.]
end
function _structured_buffer_bytes(p,d,args::Vararg{Any,N}) where N
    ad_value_and_gradient!(p,d,args...)
    @allocated ad_value_and_gradient!(p,d,args...)
end

@testset "caller-owned structured reverse destinations" begin
    backend=AutoEnzyme(mode=Enzyme.Reverse)
    # Registered Enzyme still allocates its tuple-zeroing IdSet (snag
    # make-zero-tuple-d96fee1c); keep numerical checks on both backends.
    zeroing_repaired=isdefined(Enzyme,:has_dogfood_repair) &&
        Enzyme.has_dogfood_repair(:make_zero_tuple_points)
    kernel=prepare(structured_buffer_objective;want=:total)
    p=prepare_ad(kernel,backend,Float64[],Float64[],1.;active=(:z,:x))
    mixed=prepare_ad(kernel,backend,Float64[],Float64[],1.;active=(:a,:x))
    for n in (0,7,128,2048,7)
        x=sin.(collect(1.0:n));z=cos.(collect(1.0:n));a=1.7
        d=(fill(17.,n),fill(-19.,n))
        dm=(Ref(NaN),fill(17.,n))
        for aa in (a,-2.1,.3)
            value,reference=ad_value_and_gradient(p,x,z,aa)
            actual,returned=ad_value_and_gradient!(p,d,x,z,aa)
            @test actual≈value
            @test returned===d
            @test d[1]≈reference[1] && d[2]≈reference[2]
            v,ref=ad_value_and_gradient(mixed,x,z,aa)
            v2,out=ad_value_and_gradient!(mixed,dm,x,z,aa)
            @test v2≈v && out===dm
            @test dm[1][]≈ref[1] && dm[2]≈ref[2]
        end
        bytes=_structured_buffer_bytes(p,d,x,z,a)
        if zeroing_repaired
            @test bytes==0
        else
            @test_broken bytes==0
        end
        bytes=_structured_buffer_bytes(mixed,dm,x,z,a)
        if zeroing_repaired
            @test bytes==0
        else
            @test_broken bytes==0
        end
    end
    x=[1.,2.,3.];z=[4.,5.,6.];a=2.
    value,reference=ad_value_and_gradient(p,x,z,a)
    d=zero(x)
    @test first(ad_value_and_gradient!(p,(d,x),x,z,a))≈value
    @test d≈reference[1] && x≈reference[2]
    x=[1.,2.,3.];d=zero(x)
    value,reference=ad_value_and_gradient(p,x,z,a)
    @test first(ad_value_and_gradient!(p,(d,d),x,z,a))≈value
    @test d≈reference[2]
    value,reference=ad_value_and_gradient(p,x,x,a)
    d1=zero(x);d2=zero(x)
    @test first(ad_value_and_gradient!(p,(d1,d2),x,x,a))≈value
    @test d1≈reference[1] && d2≈reference[2]
    # Inactive operands must survive until after differentiation too.
    value,reference=ad_value_and_gradient(mixed,x,z,a)
    da=Ref(0.)
    @test first(ad_value_and_gradient!(mixed,(da,z),x,z,a))≈value
    @test da[]≈reference[1] && z≈reference[2]
    # Unsupported destination types retain the copy path and its errors.
    @test_throws DimensionMismatch ad_value_and_gradient!(p,(zero(x),),x,z,a)
    viewdest=(view(zeros(5),2:4),view(zeros(5),2:4))
    value,reference=ad_value_and_gradient(p,x,z,a)
    @test first(ad_value_and_gradient!(p,viewdest,x,z,a))≈value
    @test viewdest[1]≈reference[1] && viewdest[2]≈reference[2]
    # Different dense headers can wrap the same memory.
    x=[1.,2.,3.];z=[4.,5.,6.]
    GC.@preserve x begin
        alias=unsafe_wrap(Array,pointer(x),length(x);own=false)
        value,reference=ad_value_and_gradient(p,x,z,a)
        @test first(ad_value_and_gradient!(p,(zero(x),alias),x,z,a))≈value
        @test x≈reference[2]
    end
end
