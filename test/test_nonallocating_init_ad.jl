using ReactiveKernels, MutatingFunctions, DifferentiationInterface, Enzyme, Test

@kernel reusable_init_ad(x::Vector{Float64}, z::Vector{Float64}, initial::Float64) = begin
    total::Float64 = sum(x .* z; init = initial)
    return total
end
@kernel reusable_mapped_init_ad(x::Vector{Float64}, initial::Float64) = begin
    total::Float64 = sum(abs2, x .+ 1; init = initial)
    return total
end
function _init_gradient_bytes(p,d,args::Vararg{Any,N}) where N
    ad_value_and_gradient!(p,d,args...)
    @allocated ad_value_and_gradient!(p,d,args...)
end

@testset "ordinary reverse of runtime-init reductions" begin
    backend=AutoEnzyme(mode=Enzyme.Reverse)
    plain=prepare(reusable_init_ad;want=:total)
    reused=prepare_nonallocating(plain,Float64[],Float64[],0.)
    p=prepare_ad(reused,backend,Float64[],Float64[],0.;active=:x)
    allp=prepare_ad(reused,backend,Float64[],Float64[],0.;active=(:x,:z,:initial))
    reference=prepare_ad(plain,backend,Float64[],Float64[],0.;active=(:x,:z,:initial))
    mapped=prepare_ad(prepare_nonallocating(reusable_mapped_init_ad,Float64[],0.;want=:total),
        backend,Float64[],0.;active=:x)
    for n in (0,7,128,2048,7)
        x=sin.(collect(1.0:n)); z=cos.(collect(1.0:n)); dx=zero(x)
        for initial in (-3.,0.,2.5)
            value,gradient=ad_value_and_gradient(allp,x,z,initial)
            native_value,native_gradient=ad_value_and_gradient(reference,x,z,initial)
            @test value≈native_value
            @test all(map(isapprox,gradient,native_gradient))
            @test gradient[3]==1.
            v,out=ad_value_and_gradient!(p,dx,x,z,initial)
            @test v≈native_value && out===dx && dx≈z
            mv,mg=DifferentiationInterface.value_and_gradient(
                t->sum(abs2,t .+ 1;init=initial),backend,x)
            v,out=ad_value_and_gradient!(mapped,dx,x,initial)
            @test v≈mv && out===dx && dx≈mg
        end
        # Numerical checks above retain the empty-seeded preparation across
        # changed lengths. The read-only AD cache driver does not retain a
        # replacement buffer, so allocation checks use matching exemplars.
        # Automatic cache resizing remains separate work.
        sized=prepare_ad(prepare_nonallocating(plain,x,z,0.),backend,x,z,0.;active=:x)
        sized_mapped=prepare_ad(prepare_nonallocating(reusable_mapped_init_ad,x,0.;want=:total),
            backend,x,0.;active=:x)
        @test _init_gradient_bytes(sized,dx,x,z,0.)==0
        @test _init_gradient_bytes(sized_mapped,dx,x,0.)==0
    end
end
