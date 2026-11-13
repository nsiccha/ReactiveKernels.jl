using ReactiveKernels, MutatingFunctions, Enzyme, DifferentiationInterface
using ReactiveKernelsDistributionKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: _student_log_kernel
using Test
include("prototypes.jl")
const RK = ReactiveKernels

@kernel stored_transform(u::Vector{Float64}, nu::Float64) = begin
    w = exp.(u)
    squared = w .^ 2
    prior = plate(w, Ref(nu)) do wi, ni
        _student_log_kernel(ni, wi)
    end
    total = sum(squared) + sum(prior)
    return total
end

function smoke()
    println((; julia=VERSION, enzyme=pkgversion(Enzyme),
        enzyme_source=pathof(Enzyme), libenzyme=Enzyme.Compiler.Enzyme_jll.libEnzyme_path))
    u = [0.2sin(i) for i in 1:17]; nu = 3.0
    p = plan(stored_transform)
    reference = sum(exp(2z) - (nu+1)/2*log1p(exp(2z)/nu) for z in u)
    slope = [2exp(2z) - (nu+1)*exp(2z)/(nu+exp(2z)) for z in u]
    for mode in (:control, :early, :early_snapshot, :late, :late_snapshot)
        println((; mode, stage="prepare")); flush(stdout)
        k = mode === :control ? prepare_nonallocating(prepare(p;bound_statistics=false),u,nu) :
            mode in (:early,:early_snapshot) ? StorageAlternatives.early(p,u,nu;snapshot=mode===:early_snapshot) :
            StorageAlternatives.late(p,u,nu;snapshot=mode===:late_snapshot)
        @test only(k(u,nu)) ≈ reference
        println((; mode, stage="ordinary Reverse")); flush(stdout)
        ad = prepare_ad(k,AutoEnzyme(;mode=Enzyme.Reverse),u,nu;active=:u)
        d = similar(u)
        v,_ = ad_value_and_gradient!(ad,d,u,nu)
        @test v ≈ reference
        @test d ≈ slope
        println((; mode, stage="pass", value=v)); flush(stdout)
    end
end
with_plate_reverse(false) do
    @testset "storage alternative smoke" smoke()
end
