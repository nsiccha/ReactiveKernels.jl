using ReactiveKernels, Reactant
include("../../test/test_transpiled_program.jl")
check_transpiled_program(:reactant)

include("prepared_hmc.jl")
using Random
@testset "HMC prepared state and RNG continuation" for backend in (:native, :reactant)
    rng = backend === :native ? Xoshiro(91) : Reactant.ReactantRNG(Reactant.to_rarray(UInt64[91, 77]))
    saved = copy(rng)
    program = PreparedHMCExample.prepare_hmc(rng; backend, transitions=10)
    initial = initial_transpiled_state(program)
    first = program(initial, rng)
    continued = program(first.state, first.argument)
    @test all(isfinite, Array(first.outputs.position))
    @test all(isfinite, Array(continued.outputs.position))
    if backend === :native
        @test rand(copy(rng), UInt64) == rand(saved, UInt64)
    else
        @test Array(rng.seed) == Array(saved.seed)
        @test_throws ArgumentError program(initial,
            Reactant.ReactantRNG(copy(rng.seed), "PHILOX"))
        @test_throws ArgumentError program(initial,
            Reactant.ReactantRNG(Reactant.to_rarray(UInt64[91])))
    end
    # Replay is a same-executable ownership check, not cross-backend parity.
    replayed = program(initial, rng)
    @test Array(replayed.outputs.position) == Array(first.outputs.position)
    first.outputs.position .= 100.0
    replayed_continuation = program(first.state, first.argument)
    @test Array(replayed_continuation.outputs.position) == Array(continued.outputs.position)
end
