using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random, Test, LinearAlgebra
const EXT = Base.get_extension(ReactiveKernelsPPL,:ReactiveKernelsPPLSMCExt)

function retained_and_allocated(d)
    data = (; diagonal=collect(range(0.5,1.5;length=d)),
        indices=[collect(1:d) for _ in 1:4],
        reflections=[normalize!([sin(i*j) for i in 1:d]) for j in 1:4])
    reference = SMCReference(zeros(d),data)
    scale = WarmupHMC.scale_from_data(reference.scale)
    kernel = prepare(EXT._structured_normal_reference;want=:density)
    u, g = fill(0.1,d), zeros(d)
    args = (reference.location,scale,first(logabsdet(scale)))
    ad = Base.invokelatest(prepare_ad,kernel,AutoEnzyme(mode=Enzyme.Reverse),u,args...;active=:u)
    for _ in 1:2
        Base.invokelatest(ad_value_and_gradient!,ad,g,u,args...)
    end
    allocated = @allocated Base.invokelatest(ad_value_and_gradient!,ad,g,u,args...)
    @test all(isfinite,g)
    direction = normalize!([sin(i) for i in 1:d])
    step = 1e-5
    finite_difference = (Base.invokelatest(kernel,u.+step.*direction,args...) -
        Base.invokelatest(kernel,u.-step.*direction,args...))/(2step)
    @test dot(g,direction) ≈ finite_difference atol=1e-8
    (; retained=Base.summarysize(reference), allocated)
end

@testset "Bounded-reflection reference AD scales linearly" begin
    small, large = retained_and_allocated(128), retained_and_allocated(256)
    @test large.retained < 2.1small.retained
    @test large.allocated < 2.5small.allocated
    @info "Structured reference scaling" small large
end
