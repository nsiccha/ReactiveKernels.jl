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

function _structured_buffer_bytes(p,d,args::Vararg{Any,N}) where N
    ad_value_and_gradient!(p,d,args...)
    @allocated ad_value_and_gradient!(p,d,args...)
end

@kernel structured_no_context_pair(x::Vector{Float64}, z::Vector{Float64}) = begin
    total::Float64 = sum(abs2,x) + 3 * sum(z)
    return total
end

@kernel structured_no_context_wide(x1::Vector{Float64}, x2::Vector{Float64},
        x3::Vector{Float64}, x4::Vector{Float64}, x5::Vector{Float64}) = begin
    total::Float64 = sum(abs2,x1) + 2 * sum(x2) + sum(abs2,x3) + sum(x4) + sum(abs2,x5)
    return total
end

@testset "structured destinations without a DI context" begin
    backend=AutoEnzyme(mode=Enzyme.Reverse)
    # Enzyme dogfood's structured_gradient_inplace repair lets a no-context
    # tuple point write its gradient straight into the destination. Without it,
    # RK keeps the copy path, so the numerical checks hold on every Enzyme.
    direct=isdefined(Enzyme,:has_dogfood_repair) &&
        Enzyme.has_dogfood_repair(:structured_gradient_inplace)
    zeroing_repaired=isdefined(Enzyme,:has_dogfood_repair) &&
        Enzyme.has_dogfood_repair(:make_zero_tuple_points)
    p=prepare_ad(structured_no_context,backend,[1.,2.,3.],2.;
        active=(:x,:a),want=:total)
    pz=prepare_ad(structured_no_context_pair,backend,[1.,2.],[3.,4.];
        active=(:x,:z),want=:total)
    pw=prepare_ad(structured_no_context_wide,backend,(ones(2) for _ in 1:5)...;
        active=(:x1,:x2,:x3,:x4,:x5),want=:total)
    # Changed lengths, empty inputs and repeated calls through one preparation;
    # stale destination contents must be overwritten, never accumulated.
    for n in (3,0,1,129,3,0)
        x=sin.(collect(1.0:n)); z=cos.(collect(1.0:n))
        for a in (2.,-0.5)
            d=(fill(NaN,n),Ref(NaN))
            for _ in 1:2
                actual,returned=ad_value_and_gradient!(p,d,x,a)
                @test actual≈a*sum(abs2,x)
                @test returned===d
                @test d[1]≈2a .* x && d[2][]≈sum(abs2,x)
            end
            @test x==sin.(collect(1.0:n))
        end
        dz=(fill(17.,n),fill(-19.,n))
        @test first(ad_value_and_gradient!(pz,dz,x,z))≈sum(abs2,x)+3sum(z)
        @test dz[1]≈2 .* x && dz[2]≈fill(3.,n)
        xs=ntuple(j->j .* x,5)
        dw=ntuple(_->fill(NaN,n),5)
        @test first(ad_value_and_gradient!(pw,dw,xs...))≈
            sum(abs2,xs[1])+2sum(xs[2])+sum(abs2,xs[3])+sum(xs[4])+sum(abs2,xs[5])
        @test dw[1]≈2 .* xs[1] && dw[2]≈fill(2.,n) && dw[3]≈2 .* xs[3] &&
            dw[4]≈fill(1.,n) && dw[5]≈2 .* xs[5]
        n>0 || continue
        bytes=_structured_buffer_bytes(p,(zero(x),Ref(0.)),x,2.)
        direct && zeroing_repaired && @test bytes==0
        bytes=_structured_buffer_bytes(pz,(zero(x),zero(z)),x,z)
        direct && zeroing_repaired && @test bytes==0
        bytes=_structured_buffer_bytes(pw,map(zero,xs),xs...)
        direct && zeroing_repaired && @test bytes==0
    end
    # Aliased destinations or inputs keep the copy path and its results.
    x=[1.,2.,3.];z=[4.,5.,6.]
    d=zero(x)
    @test first(ad_value_and_gradient!(pz,(d,x),x,z))≈sum(abs2,[1.,2.,3.])+45
    @test d≈[2.,4.,6.] && x≈fill(3.,3)
    x=[1.,2.,3.];d=zero(x)
    @test first(ad_value_and_gradient!(pz,(d,d),x,z))≈14+45
    @test d≈fill(3.,3)
    # One array bound to both active ports shares its shadow, as out of place.
    value,reference=ad_value_and_gradient(pz,x,x)
    d1=fill(NaN,3);d2=fill(NaN,3)
    @test first(ad_value_and_gradient!(pz,(d1,d2),x,x))≈value≈14+18
    @test d1≈reference[1] && d2≈reference[2] && x==[1.,2.,3.]
    GC.@preserve x begin
        alias=unsafe_wrap(Array,pointer(x),length(x);own=false)
        @test first(ad_value_and_gradient!(pz,(zero(x),alias),x,z))≈14+45
        @test x≈fill(3.,3)
    end
    xs=ntuple(j->fill(Float64(j),2),5); shared=zeros(2)
    dw=(shared,zeros(2),zeros(2),zeros(2),shared)
    @test first(ad_value_and_gradient!(pw,dw,xs...))≈2+8+18+8+50
    @test shared≈2 .* xs[5] && dw[2]≈fill(2.,2)
    # Destinations of another representation keep the copy path; a mismatch
    # is still rejected.
    x=[1.,2.,3.]
    viewdest=(view(zeros(5),2:4),Ref(0.))
    @test first(ad_value_and_gradient!(p,viewdest,x,2.))≈28
    @test viewdest[1]≈4 .* x && viewdest[2][]≈14
    @test_throws DimensionMismatch ad_value_and_gradient!(p,(zero(x),),x,2.)
    @test_throws ArgumentError ad_value_and_gradient!(p,(zero(x),0.),x,2.)
end

@testset "copied cotangents reject a mismatched destination before writing" begin
    backend=AutoEnzyme(mode=Enzyme.Reverse)
    x=[1.,2.,3.];z=[4.,5.,6.];a=2.
    p=prepare_ad(structured_no_context,backend,x,a;active=(:x,:a),want=:total)
    pz=prepare_ad(structured_no_context_pair,backend,x,z;active=(:x,:z),want=:total)
    pc=prepare_ad(structured_buffer_objective,backend,x,z,a;active=(:x,:z),want=:total)
    pt=prepare_ad(structured_tuple_port,backend,(x,z),a;active=:xs,want=:total)
    pb=prepare_ad_pullback(structured_buffer_objective,backend,1.,x,z,a;
        active=(:x,:z),want=:total)
    sentinel(n)=fill(-7.,n)
    untouched(d)=all(c->c isa Base.RefValue ? c[]==-7. : all(==(-7.),c),d)
    # Each component before the invalid one must stay untouched.
    for (prepared,args) in ((p,(x,a)),(pz,(x,z)),(pc,(x,z,a)))
        second=prepared===p ? Ref(-7.) : sentinel(3)
        for (d,E) in (((sentinel(4),second),DimensionMismatch),  # oversized
                      ((sentinel(2),second),DimensionMismatch),  # undersized
                      ((sentinel(3),prepared===p ? -7. : (-7.,)),ArgumentError),
                      ((sentinel(3),sentinel(prepared===p ? 2 : 5)),DimensionMismatch),
                      ((sentinel(3),second,sentinel(3)),DimensionMismatch))
            @test_throws E ad_value_and_gradient!(prepared,d,args...)
            @test untouched(d)
        end
    end
    for d in ((sentinel(3),sentinel(4)),(sentinel(3),-7.))
        @test_throws Union{DimensionMismatch,ArgumentError} ad_value_and_gradient!(pt,d,(x,z),a)
        @test untouched(d)
    end
    for d in ((sentinel(3),sentinel(2)),(sentinel(3),-7.))
        @test_throws Union{DimensionMismatch,ArgumentError} ad_value_and_pullback!(pb,d,1.,x,z,a)
        @test untouched(d)
    end
    # Matching layouts the copy path supports keep working: a view, a matrix
    # of the same length, and a destination that aliases an input.
    value,reference=ad_value_and_gradient(pc,x,z,a)
    d=(view(sentinel(5),2:4),reshape(sentinel(3),3,1))
    @test first(ad_value_and_gradient!(pc,d,x,z,a))≈value
    @test d[1]≈reference[1] && vec(d[2])≈reference[2]
    _,cot=ad_value_and_pullback(pb,1.,x,z,a)
    d=(view(sentinel(5),1:3),sentinel(3))
    @test ad_value_and_pullback!(pb,d,1.,x,z,a)[2]===d
    @test d[1]≈cot[1] && d[2]≈cot[2]
    zz=copy(z)
    @test first(ad_value_and_gradient!(pc,(sentinel(3),zz),x,zz,a))≈value
    @test zz≈reference[2]
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
