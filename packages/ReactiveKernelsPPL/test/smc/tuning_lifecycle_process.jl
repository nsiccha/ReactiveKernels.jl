include("tuning_lifecycle_fixture.jl")
using .SMCTuningLifecycleFixture, ReactiveKernelsPPL, Serialization, Test
const F = SMCTuningLifecycleFixture
mode, directory = ARGS
if mode == "write"
    expected = F.signature(F.tune())
    serialize(joinpath(directory,"expected.jls"), expected)
    for phase in (:reference, :grid)
        stop = (p,b) -> p.phase == phase && b == :window && (phase == :reference || p.knot == 2)
        partial = F.tune(; checkpoint_dir=joinpath(directory,string(phase)), callback=stop)
        @test partial isa SMCTuningPartial
        @test partial.progress.phase == phase
        serialize(joinpath(directory,string(phase)*"-partial.jls"), partial)
    end
elseif mode == "read"
    expected = deserialize(joinpath(directory,"expected.jls"))
    for phase in (:reference, :grid)
        partial = deserialize(joinpath(directory,string(phase)*"-partial.jls"))
        @test F.EXT._same(F.signature(F.tune(; resume=partial)), expected)
        result = F.tune(; checkpoint_dir=joinpath(directory,string(phase)), resume=true)
        @test F.EXT._same(F.signature(result), expected)
    end
else
    error("expected write or read")
end
