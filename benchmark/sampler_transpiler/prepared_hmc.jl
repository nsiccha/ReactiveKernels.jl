module PreparedHMCExample
using ReactiveKernels, LinearAlgebra, Random
include("eight_schools_density.jl")
using .EightSchoolsDensity: build_density
include("hmc_benchmark.jl")

# BEGIN HMC consumer
function prepare_hmc(rng; transitions=1000, steps=4)
    density, ad, position = build_density()
    HMCBenchmark.prepare_hmc(density, ad, position, rng; transitions, steps)
end

function hmc_example(rng=Xoshiro(91); transitions=1000, steps=4)
    program = prepare_hmc(rng; transitions, steps)
    state = initial_transpiled_state(program)
    first = program(state, rng)
    continued = program(first.state, first.argument)
    (; first, continued)
end
# END HMC consumer
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = PreparedHMCExample.hmc_example()
    println("final position: ", Array(result.continued.outputs.position))
end
