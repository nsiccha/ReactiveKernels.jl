module PreparedHMCExample
using ReactiveKernels, LinearAlgebra, Random
include("eight_schools_density.jl")
using .EightSchoolsDensity: build_density
include("hmc_benchmark.jl")

# BEGIN HMC consumer
function prepare_hmc(rng; backend=:native, transitions=1000, steps=4)
    density, ad, position = build_density()
    HMCBenchmark.prepare_hmc(density, ad, position, rng; backend, transitions, steps)
end

function hmc_example(rng=Xoshiro(91); backend=:native, transitions=1000, steps=4)
    program = prepare_hmc(rng; backend, transitions, steps)
    state = initial_transpiled_state(program)
    first = program(state, rng)
    continued = program(first.state, first.argument)
    (; first, continued)
end
# END HMC consumer
end

if abspath(PROGRAM_FILE) == @__FILE__
    backend = isempty(ARGS) ? :native : Symbol(ARGS[1])
    if backend === :reactant
        @eval using Reactant
        rng = Reactant.ReactantRNG(Reactant.to_rarray(UInt64[91, 77]))
    else
        rng = PreparedHMCExample.Xoshiro(91)
    end
    result = PreparedHMCExample.hmc_example(rng; backend)
    println("final position: ", Array(result.continued.outputs.position))
end
