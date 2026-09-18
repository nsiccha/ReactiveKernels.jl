# The native functional stage path lowers through the PUBLIC functional
# surface (`lower_with_ops`), not the private `_lower_with_ops` the lane
# pinned before core published it (snag kernel-functiona-82e0c460).
import ReactiveKernels
using ReactiveKernels: lower, plan

@testset "functional lowering resolves through the public API" begin
    @test isdefined(ReactiveKernels, :lower_with_ops)
    @test :lower_with_ops in names(ReactiveKernels)
    # The generator's exact call shape: positional plan, default keywords.
    spec_plan = plan(RKRO.tsit5_stage; have=RKRO.TSIT5_STAGE_PORTS,
        want=(:u, :kk, :EEst))
    triple = ReactiveKernels.lower_with_ops(spec_plan)
    @test triple isa Tuple && length(triple) == 3
    @test first(triple) == lower(spec_plan)
end

@testset "no private lowering dependency remains" begin
    # Brittle by design: the private entry still exists upstream, so only the
    # source can prove the generator stopped calling it. If this trips,
    # migrate the call site back to public `lower_with_ops`; do not weaken
    # this test.
    src = read(joinpath(@__DIR__, "..", "src", "kernels.jl"), String)
    @test !occursin("_lower_with_ops", src)
    @test occursin("lower_with_ops(spec_plan)", src)
end
