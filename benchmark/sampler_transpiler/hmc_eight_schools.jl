# Executable prototype, not a public sampler API or performance claim.
# The sampler and integrator are the existing authored mathematical kernels.
using ReactiveKernels, ReactiveKernelsPPLExamples, Enzyme
using DifferentiationInterface, LinearAlgebra, Random
import AdvancedHMC
const RK = ReactiveKernels
const E = ReactiveKernelsPPLExamples.EightSchoolsExample
include(joinpath(@__DIR__, "..", "nuts_kernel_authoring_fixture_b.jl"))
include(joinpath(@__DIR__, "..", "reactivehmc_hmc_kernel_fixture_b.jl"))
const F = NUTSBMutationAuthoringFixture
const H = ReactiveHMCHMCBMutationAuthoringFixture

function measured(f, label)
    println("BEGIN ", label); flush(stdout)
    result = @timed f()
    println(label, " seconds=", result.time, " bytes=", result.bytes)
    flush(stdout)
    result.value
end

include(joinpath(@__DIR__, "eight_schools_density.jl"))
using .EightSchoolsDensity: build_density, Potential, Gradient, CallbackHandle

struct StepAuthority <: Function end
struct StatsAuthority <: Function end
(::StepAuthority)(args...) = error("use the compiled endpoint lowering")
(::StatsAuthority)(args...) = error("use the compiled observation lowering")
function statistics(effect, state)
    acceptance = exp(min(zero(state.dham), state.dham))
    (arguments=(state,), result=nothing,
     effect_state=(steps=effect.steps + 1,
                   acceptance_sum=effect.acceptance_sum + acceptance))
end

rng_state(rng::Xoshiro) = UInt64[rng.s0, rng.s1, rng.s2, rng.s3, rng.s4]
function native_normal(state, destination)
    rng = Xoshiro(state...)
    value = randn!(rng, similar(destination))
    (state=rng_state(rng), value, valid=true)
end
function native_bool(state)
    rng = Xoshiro(state...)
    value = rand(rng, Bool)
    (state=rng_state(rng), value, valid=true)
end
function native_exp(state)
    rng = Xoshiro(state...)
    value = randexp(rng)
    (state=rng_state(rng), value, valid=true)
end


function build_prototype(backend; L=4, callback_handles=true)
    density, ad, q = build_density()
    potential, gradient = Potential(density), Gradient(ad)
    if callback_handles
        potential, gradient = CallbackHandle(potential), CallbackHandle(gradient)
    end
    endpoint_inputs = (potential, gradient, Diagonal(ones(10)), q, zeros(10))
    endpoint = measured("prepare_endpoint") do
        RK.compile_state_transition(F.euclidean_phasepoint,
            RK.partial(F.leapfrog!; stepsize=0.03),
            endpoint_inputs)
    end
    point = RK.initial_transition_state(endpoint)
    structured = RK.structured_state_port(endpoint)
    step_source = StepAuthority()
    stats_source = backend === :native_slots ? nothing : StatsAuthority()
    step_port = RK.effect_lowering_port(step_source, Tuple{typeof(point)}, Nothing;
        written_arguments=(1,), initial_effect_state=nothing,
        functional_lowering=RK.total_functional_lowering(
            RK._SMCompiledTransitionEffect(endpoint)))
    stats_port = stats_source === nothing ? nothing : RK.effect_lowering_port(stats_source, Tuple{RK.StatefulStateValue}, Nothing;
        written_arguments=(), initial_effect_state=(steps=0, acceptance_sum=0.0),
        functional_lowering=RK.total_functional_lowering(statistics))
    bindings = RK.stateful_compiler_bindings(
        init=structured, fwd=structured, step_f=step_port, stats_f=stats_port)
    kernel = measured("prepare_hmc") do
        RK.compile_stateful(H.hmc_state, bindings, point;
            n_steps=L, step_f=step_source, stats_f=stats_source)
    end
    snapshot = RK.stateful_snapshot(kernel(point;
        n_steps=L, step_f=step_source, stats_f=stats_source))
    backend === :native_slots && return (; kernel, snapshot, endpoint, endpoint_inputs, L,
                                          density, ad, q)
    provider = RK.rng_provider(Vector{UInt64};
        normal_fill=RK.total_functional_lowering(native_normal),
        bool_draw=RK.total_functional_lowering(native_bool),
        exp_draw=RK.total_functional_lowering(native_exp))
    transition = measured("lower_hmc") do
        RK._functionalize_stateful(kernel, Val(:step!); max_iterations=L,
            argument_types=Tuple{Vector{UInt64}}, rng_providers=(rng=provider,),
            native_state_types=true)
    end
    (; transition, snapshot, L)
end

function native_chain(transition, state, seed, n)
    for _ in 1:n
        result = transition(state, seed)
        result.control_overflow && error("compiled control bound exhausted")
        state, seed = result.state, result.arguments[1]
    end
    state, seed
end

function run_native(prototype; n=100)
    transition, snapshot = prototype.transition, prototype.snapshot
    seed = rng_state(Xoshiro(91))
    result = measured("native_first_transition") do
        transition(snapshot, seed)
    end
    result.control_overflow && error("compiled control bound exhausted")
    all(isfinite, result.state.init.pos) || error("nonfinite position")
    native_chain(transition, snapshot, seed, 2)
    for _ in 1:3
        measured("native_$(n)_transitions") do
            native_chain(transition, snapshot, seed, n)
        end
    end
end

function ahmc_chain(rng, hamiltonian, kernel, phasepoint, n)
    for _ in 1:n
        result = AdvancedHMC.transition(rng, hamiltonian, kernel, phasepoint)
        result.stat.numerical_error && error("AdvancedHMC numerical failure")
        phasepoint = result.z
    end
    phasepoint
end

function run_ahmc(; n=100)
    density, ad, q = build_density()
    # Capture the exact same prepared model and gradient, avoiding global lookup.
    value_gradient = position -> RK.ad_value_and_gradient(ad, position)
    hamiltonian = AdvancedHMC.Hamiltonian(
        AdvancedHMC.DiagEuclideanMetric(length(q)), position -> density(position),
        value_gradient)
    kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.EndPointTS}(
        AdvancedHMC.Leapfrog(0.03), AdvancedHMC.FixedNSteps(4)))
    hamiltonian, initial = AdvancedHMC.sample_init(Xoshiro(91), hamiltonian, q)
    measured("ahmc_first_chain") do
        ahmc_chain(Xoshiro(91), hamiltonian, kernel, deepcopy(initial.z), 2)
    end
    for _ in 1:5
        # Initialization/copy is outside the timed transition loop in every mode.
        rng, phasepoint = Xoshiro(91), deepcopy(initial.z)
        final = measured("ahmc_$(n)_transitions") do
            ahmc_chain(rng, hamiltonian, kernel, phasepoint, n)
        end
        all(isfinite, final.θ) || error("AdvancedHMC nonfinite position")
    end
end

include("native_slots.jl")
include("native_slots_factory.jl")
include("native_comparison.jl")

function main(args=ARGS)
    length(args) <= 3 || error("usage: hmc_eight_schools.jl [native|native-slots|compare|ahmc] [batch_length] [leapfrog_steps]")
    backend = isempty(args) ? :native_slots : Symbol(replace(first(args), '-' => '_'))
    n = length(args) >= 2 ? parse(Int, args[2]) : 100
    L = length(args) == 3 ? parse(Int,args[3]) : 4
    n > 0 || error("batch length must be positive")
    L > 0 || error("leapfrog count must be positive")
    backend in (:native, :native_slots, :compare, :ahmc) || error("unknown backend")
    length(args)==3 && !(backend in (:native_slots,:compare)) &&
        error("leapfrog count is configurable in the slot-based modes")
    backend === :ahmc && return run_ahmc(; n)
    if backend in (:native_slots,:compare)
        prototype=build_fast_prototype(;L)
        return run_prepared_comparison(prototype.prepared,
            backend===:compare ? prototype.comparator : nothing,L,n)
    end
    run_native(build_prototype(backend); n)
end
abspath(PROGRAM_FILE) == (@__FILE__) && main()
