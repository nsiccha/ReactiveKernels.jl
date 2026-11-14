using ReactiveKernels,Enzyme,DifferentiationInterface,Test
@kernel installed_point_objective(x::Vector{Float64},z::Vector{Float64},a::Float64) = begin
    total::Float64=a*(sum(abs2,x)+sum(z))
    return total
end
@kernel installed_point_keywords(x::Vector{Float64};z::Vector{Float64},a::Float64) = begin
    total::Float64=a*(sum(abs2,x)+sum(z))
    return total
end
function stored_bytes(p,d,args::Vararg{Any,N}) where N
    ad_value_and_gradient!(p,d,args...)
    @allocated ad_value_and_gradient!(p,d,args...)
end
function keyword_bytes(p,d,x,z,a)
    ad_value_and_gradient!(p,d,x;z,a)
    @allocated ad_value_and_gradient!(p,d,x;z,a)
end
@testset "installed scalar point ownership" begin
    backend=AutoEnzyme(mode=Enzyme.Reverse)
    # Registered Enzyme still allocates its tuple-zeroing IdSet (snag
    # make-zero-tuple-d96fee1c); keep numerical checks on both backends.
    zeroing_repaired=isdefined(Enzyme,:has_dogfood_repair) &&
        Enzyme.has_dogfood_repair(:make_zero_tuple_points)
    for active in ((:x,:a),(:a,:x),(:a,))
        p=prepare_ad(installed_point_objective,backend,Float64[],Float64[],1.;active,want=:total)
        p2=prepare_ad(p.kernel,backend,Float64[],Float64[],1.;active)
        for (j,name) in enumerate(active)
            name===:a && @test p.storage.point[j] !== p2.storage.point[j]
        end
        for n in (0,7,128,2048,7)
            x=sin.(collect(1.0:n));z=cos.(collect(1.0:n))
            d=map(name->name===:a ? Ref(NaN) : zero(x),active)
            for a in (1.7,-2.1,.3)
                value,reference=ad_value_and_gradient(p,x,z,a)
                actual,returned=ad_value_and_gradient!(p,d,x,z,a)
                @test value≈actual && returned===d
                @test all(map((got,want)->(got isa Ref ? got[] : got)≈want,d,reference))
                # Interleaving another preparation cannot change this one's box.
                slots=map(s->s isa Ref ? s[] : nothing,p.storage.point)
                ad_value_and_gradient!(p2,d,x,z,-a)
                @test map(s->s isa Ref ? s[] : nothing,p.storage.point)==slots
            end
            bytes=stored_bytes(p,d,x,z,1.7)
            if zeroing_repaired
                @test bytes==0
            else
                @test_broken bytes==0
            end
        end
    end
    p=prepare_ad(installed_point_keywords,backend,Float64[];z=Float64[],a=1.,active=(:a,:x),want=:total)
    for n in (0,7,128,7)
        x=ones(n);z=fill(2.,n);d=(Ref(0.),zero(x))
        v,g=ad_value_and_gradient(p,x;z,a=2.)
        v2,out=ad_value_and_gradient!(p,d,x;z,a=2.)
        @test v≈v2 && out===d && d[1][]≈g[1] && d[2]≈g[2]
        bytes=keyword_bytes(p,d,x,z,2.)
        if zeroing_repaired
            @test bytes==0
        else
            @test_broken bytes==0
        end
    end
end
