using ReactiveKernels
using ReactiveKernelsNUTSExamples
using Reactant
using LinearAlgebra
using Random
using Test
import Reactant: @compile
import TOML

include(joinpath(@__DIR__, "fixtures",
                 "reactivehmc_hmc_compiler_support.jl"))

module _MutationProfileBReactantNUTS
include(joinpath(@__DIR__, "..", "benchmark",
                 "nuts_kernel_authoring_fixture_b.jl"))
end

const _MPBR_HMC = ReactiveHMCHMCCompilerSupport
const _MPBR_NUTS =
    _MutationProfileBReactantNUTS.NUTSBMutationAuthoringFixture
const _MPBR_TESTSET = get(ENV, "RK_MPB_REACTANT_TESTSET", "all")
_mpbr_enabled(name) = _MPBR_TESTSET == "all" || _MPBR_TESTSET == name

struct _MPBRCallable{F}
    f::F
end
(callable::_MPBRCallable)(arguments...) = callable.f(arguments...)

function _mpbr_trace(value)
    _mpbr_trace(value, IdDict{Any,Any}())
end
function _mpbr_trace(value::AbstractArray, seen)
    get!(seen, value) do
        Reactant.to_rarray(value)
    end
end
_mpbr_trace(value::Number, seen) =
    Reactant.to_rarray(value; track_numbers=true)
_mpbr_trace(value::Diagonal, seen) = Diagonal(_mpbr_trace(value.diag, seen))
_mpbr_trace(value::LinearAlgebra.Cholesky, seen) = LinearAlgebra.Cholesky(
    _mpbr_trace(value.factors, seen), value.uplo, value.info)
_mpbr_trace(value::NamedTuple, seen) = map(child -> _mpbr_trace(child, seen), value)
_mpbr_trace(value::Tuple, seen) = map(child -> _mpbr_trace(child, seen), value)
_mpbr_trace(value, seen) = value

function _mpbr_trace_replay(replay)
    ReactiveKernels._sm_ordered_rng_reconstruct(
        _mpbr_trace(replay.normals),
        _mpbr_trace(replay.uniforms),
        _mpbr_trace(replay.exponentials),
        _mpbr_trace(replay.event_tokens),
        Reactant.to_rarray(replay.normal_index; track_numbers=true),
        Reactant.to_rarray(replay.uniform_index; track_numbers=true),
        Reactant.to_rarray(replay.exponential_index; track_numbers=true),
        Reactant.to_rarray(replay.event_index; track_numbers=true),
        Reactant.to_rarray(replay.overflow; track_numbers=true))
end

function _mpbr_nuts_factory()
    ReactiveKernels._prepare_factory(
        _MPBR_NUTS.euclidean_phasepoint,
        ReactiveKernels.kernel_registration(_MPBR_NUTS.leapfrog!))
end

function _mpbr_nuts_values(pf)
    metric = Float64[2 0; 0 2]
    values = Dict{Int,Any}()
    for slot in ReactiveKernels.kernel_plan_slots(
            ReactiveKernels.kernel_prepared_plan(pf))
        name = String(slot.path[1])
        values[slot.canon] = name == "pot_f" ? (p -> sum(abs2, p)) :
            name == "grad_f" ?
                ((dst, p) -> (dst .= 2 .* p; sum(abs2, p))) :
            name == "metric" ? metric :
            name == "chol_metric" ? cholesky(metric) :
            startswith(name, "##node") ? 0.0 :
            name == "pos" ? [1.0, 2.0] :
            name == "mom" ? [3.0, 4.0] :
            name in ("dpot_dpos", "dham_dpos", "dkin_dmom", "dham_dmom") ?
                [0.0, 0.0] : 0.0
    end
    values
end

function _mpbr_nuts_frame(pf)
    frame = ReactiveKernelsNUTSExamples._construct_nuts_frame(
        pf, _mpbr_nuts_values(pf), 0;
        step_f=ReactiveKernels.partial(_MPBR_NUTS.leapfrog!; stepsize=0.1),
        stats_f=_MPBR_NUTS.nuts_stats!, min_dham=-1000.0)
    ReactiveKernels.compile_prepared_initialization(
        pf, typeof(frame.init), typeof(frame.shared))(
            frame.init, frame.shared,
            ReactiveKernels.kernel_prepared_handles(pf))
    ReactiveKernelsNUTSExamples._seed_nuts_children!(frame)
    frame
end

if _mpbr_enabled("hmc")
@testset "mutation profile B — fixed-step HMC Reactant" begin
    receipt = TOML.parsefile(joinpath(
        @__DIR__, "..", "benchmark", "receipts",
        "reactivehmc-hmc-ca9-v1.toml"))
    case = only(filter(item -> item["name"] == "accepted", receipt["cases"]))
    program = _MPBR_HMC.build_case(
        case;
        hmc_spec=_MPBR_HMC.ReactiveHMCHMCBMutationAuthoringFixture.hmc_state,
        potential_f=_MPBRCallable(_MPBR_HMC.potential),
        gradient_f=_MPBRCallable(_MPBR_HMC.gradient))
    state = _mpbr_trace(program.snapshot)
    replay = _mpbr_trace_replay(program.replay)
    compiled = @compile program.transition(state, replay)
    guarded = ReactiveKernels.validated_compiled_transition(
        compiled, program.transition)
    actual = _MPBR_HMC.result_values(program, guarded(state, replay))

    @test actual.init_pos ≈ case["init_pos"] atol=128eps(Float64) rtol=0
    @test actual.init_mom ≈ case["init_mom"] atol=128eps(Float64) rtol=0
    @test actual.fwd_pos ≈ case["fwd_pos"] atol=128eps(Float64) rtol=0
    @test actual.fwd_mom ≈ case["fwd_mom"] atol=128eps(Float64) rtol=0
    @test actual.energy_errors ≈ case["energy_errors"] atol=128eps(Float64) rtol=0
    @test !Bool(actual.control_overflow)
    @test !Bool(actual.rng_overflow)
end
end

if _mpbr_enabled("nuts")
@testset "mutation profile B — specialized NUTS Reactant" begin
    pf = _mpbr_nuts_factory()
    frame = _mpbr_nuts_frame(pf)
    compiled = ReactiveKernelsNUTSExamples.compile_nuts_reactant(
        pf, _MPBR_NUTS.nuts_state, _MPBR_NUTS.refresh_momentum!!,
        _MPBR_NUTS.nuts!!, frame)
    momentum = [0.25, -0.5]
    bundle = ReactiveKernelsNUTSExamples.nuts_reactant_bundle(
        momentum, Bool[], Float64[], 0)
    state = map(Reactant.to_rarray,
        ReactiveKernelsNUTSExamples.nuts_reactant_state(
            compiled, frame, bundle))
    executable = ReactiveKernelsNUTSExamples.nuts_reactant_compile(
        compiled, state; sync=true)
    output = executable(state)

    @test Array(output.pp_pos)[:, 1] == [1.0, 2.0]
    @test Array(output.pp_mom)[:, 1] ≈ sqrt(2.0) .* momentum
    @test Array(output.n_steps)[1] == 0
    @test Array(output.reached_depth)[1] == 0
    @test Array(output.csp)[1] == 0
    @test all(iszero, Array(output.overflow))
end
end
