# Executable prototype, not a public sampler API or performance claim.
# The sampler and integrator are the existing authored mathematical kernels.
using ReactiveKernels, ReactiveKernelsPPLExamples, Reactant, Enzyme
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

struct Potential{K} <: Function
    kernel::K
end
(f::Potential)(q) = -f.kernel(q)
struct Gradient{A} <: Function
    ad::A
end
function (f::Gradient)(q)
    value, gradient = RK.ad_value_and_gradient(f.ad, q)
    -value, -gradient
end
function (f::Gradient)(destination, q)
    value, _ = RK.ad_value_and_gradient!(f.ad, destination, q)
    destination .*= -1
    -value
end

# Prototype of compiler-owned callback storage: numerical state carries a
# typed handle instead of copying a prepared program's metadata by value.
# The captured callback is initialized once and never replaced.
struct CallbackHandle{F} <: Function
    storage::Base.RefValue{F}
end
CallbackHandle(f::F) where {F} = CallbackHandle{F}(Ref(f))
(f::CallbackHandle)(args...) = getfield(f, :storage)[](args...)

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

function build_density()
    model = E.build_eight_schools_graph()
    q = [0.0, log(5.0), zeros(8)...]
    density = prepare(model;
        have=(:unconstrained, :observations, :observation_scales), want=:posterior,
        bound=(observations=E.EIGHT_SCHOOLS_Y,
               observation_scales=E.EIGHT_SCHOOLS_SIGMA))
    ad_backend = AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Const)
    ad = prepare_ad(density, ad_backend, q; active=:unconstrained)
    (; density, ad, q)
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
    provider = backend === :reactant ? RK.rng_provider(Val(:reactant)) :
        RK.rng_provider(Vector{UInt64};
            normal_fill=RK.total_functional_lowering(native_normal),
            bool_draw=RK.total_functional_lowering(native_bool),
            exp_draw=RK.total_functional_lowering(native_exp))
    transition = measured("lower_hmc") do
        RK._functionalize_stateful(kernel, Val(:step!); max_iterations=L,
            argument_types=Tuple{Vector{UInt64}}, rng_providers=(rng=provider,),
            native_state_types=backend === :native)
    end
    (; transition, snapshot, L)
end

# Backend-boundary adapter using the compiler's existing structural contract.
# Only numerical leaves are runtime inputs; callable identities are preparation metadata.
struct NumericStateTransition{T,C}
    transition::T
    contract::C
end
function (program::NumericStateTransition)(raw, seed)
    state = RK._sm_finite_structural_read(program.contract, raw, 1).value
    result = program.transition(state, seed)
    packed = RK._sm_finite_structural_write(
        program.contract, raw, 1, result.state).storage
    merge(result, (state=packed,))
end

function native_chain(transition, state, seed, n)
    for _ in 1:n
        result = transition(state, seed)
        result.control_overflow && error("compiled control bound exhausted")
        state, seed = result.state, result.arguments[1]
    end
    state, seed
end

# This is a driver loop around the generated transition, not another HMC implementation.
function traced_chain(program, state, seed, overflow, steps, n)
    Reactant.@trace for _ in 1:n
        result = program(state, seed)
        state = result.state
        seed = result.arguments[1]
        overflow = overflow | result.control_overflow
        steps = steps + result.effects.stats_f.steps
    end
    (; state, seed, overflow, steps)
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

function run_reactant(prototype; n=100)
    transition, snapshot = prototype.transition, prototype.snapshot
    contract = RK._sm_finite_structural_contract([snapshot];
        static_values=(snapshot.init.pot_f, snapshot.init.grad_f,
                       snapshot.step_f, snapshot.stats_f))
    state = Reactant.to_rarray(RK._sm_finite_structural_pack(contract, [snapshot]))
    seed = Reactant.to_rarray(UInt64[91, 77])
    overflow = Reactant.to_rarray(false; track_numbers=true)
    steps = Reactant.to_rarray(0; track_numbers=true)
    program = NumericStateTransition(transition, contract)
    driver = (state, seed, overflow, steps) ->
        traced_chain(program, state, seed, overflow, steps, n)
    inputs = (state, seed, overflow, steps)
    compiled = measured("compile_reactant_$(n)_transitions") do
        Reactant.compile(driver, inputs; sync=true,
            donated_args=:none, serializable=true)
    end
    println("mlir_bytes=", sizeof(compiled.module_string),
            " while_regions=", count("stablehlo.while", compiled.module_string),
            " triangular_solves=", count("stablehlo.triangular_solve", compiled.module_string))
    result = measured("reactant_first_chain") do
        compiled(inputs...)
    end
    Bool(result.overflow) && error("compiled control bound exhausted")
    println("integration_steps=", Int(result.steps))
    Int(result.steps) == n * prototype.L || error("unexpected integration-step count")
    decoded = RK._sm_finite_structural_read(contract, result.state, 1).value
    all(isfinite, Array(decoded.init.pos)) || error("nonfinite position")
    for _ in 1:5
        measured("reactant_$(n)_transitions") do
            compiled(inputs...)
        end
    end
    compiled
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
    length(args) <= 2 || error("usage: hmc_eight_schools.jl [native|native-slots|compare|reactant|ahmc] [batch_length]")
    backend = isempty(args) ? :native_slots : Symbol(replace(first(args), '-' => '_'))
    n = length(args) == 2 ? parse(Int, args[2]) : 100
    n > 0 || error("batch length must be positive")
    backend in (:native, :native_slots, :compare, :reactant, :ahmc) || error("unknown backend")
    backend === :ahmc && return run_ahmc(; n)
    if backend in (:native_slots,:compare)
        prototype=build_fast_prototype()
        return run_prepared_comparison(prototype.prepared,
            backend===:compare ? prototype.comparator : nothing,4,n)
    end
    prototype = build_prototype(backend)
    backend === :native ? run_native(prototype; n) : run_reactant(prototype; n)
end
abspath(PROGRAM_FILE) == (@__FILE__) && main()
