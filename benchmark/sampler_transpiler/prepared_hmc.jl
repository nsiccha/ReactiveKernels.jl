module PreparedHMCExample
using ReactiveKernels, LinearAlgebra, Random
include("eight_schools_density.jl")
using .EightSchoolsDensity: build_density, Potential, Gradient, CallbackHandle
include("../nuts_kernel_authoring_fixture_b.jl")
include("position_multinomial_hmc_kernel.jl")
const F = NUTSBMutationAuthoringFixture

# BEGIN HMC consumer
function prepare_hmc(rng; backend=:native, transitions=1000, steps=4)
    density, ad, position = build_density()
    point = transpiled_endpoint(F.euclidean_phasepoint, F.leapfrog!,
        CallbackHandle(Potential(density)), CallbackHandle(Gradient(ad)),
        Diagonal(ones(length(position))), position, zeros(length(position)))
    prepare_transpiled(PositionMultinomialHMCAuthoring.multinomial_hmc_state, point;
        backend, method=:step!, argument=rng, iterations=transitions,
        kernel_kwargs=(n_steps=steps, step_f=F.leapfrog!, stepsize=0.03),
        outputs=(position=(:init, :pos),))
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
