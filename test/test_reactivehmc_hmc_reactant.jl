using ReactiveKernels
using Reactant
using LinearAlgebra
using Test
import Reactant: @compile
import TOML

if !isdefined(@__MODULE__, :ReactiveHMCHMCCompilerSupport)
    include(joinpath(@__DIR__, "fixtures",
                     "reactivehmc_hmc_compiler_support.jl"))
end
const _RHMC_HMC_REACTANT = ReactiveHMCHMCCompilerSupport

_rhmc_hmc_trace(value::AbstractArray) = Reactant.to_rarray(value)
_rhmc_hmc_trace(value::Diagonal) = Diagonal(_rhmc_hmc_trace(value.diag))
_rhmc_hmc_trace(value::LinearAlgebra.Cholesky) = LinearAlgebra.Cholesky(
    _rhmc_hmc_trace(value.factors), value.uplo, value.info)
_rhmc_hmc_trace(value::NamedTuple) = map(_rhmc_hmc_trace, value)
_rhmc_hmc_trace(value::Tuple) = map(_rhmc_hmc_trace, value)
_rhmc_hmc_trace(value) = value

function _rhmc_hmc_traced_replay(replay)
    ReactiveKernels.OrderedRNGReplay(
        _rhmc_hmc_trace(replay.normals),
        _rhmc_hmc_trace(replay.uniforms),
        _rhmc_hmc_trace(replay.exponentials),
        Reactant.to_rarray(replay.normal_index; track_numbers=true),
        Reactant.to_rarray(replay.uniform_index; track_numbers=true),
        Reactant.to_rarray(replay.exponential_index; track_numbers=true),
        Reactant.to_rarray(replay.overflow; track_numbers=true))
end

function _rhmc_hmc_host_values(result)
    count = Int(result.effects.stats_f.count)
    (
        init_pos=Array(result.state.init.pos),
        init_mom=Array(result.state.init.mom),
        init_ham=Float64(result.state.init.ham),
        fwd_pos=Array(result.state.fwd.pos),
        fwd_mom=Array(result.state.fwd.mom),
        fwd_ham=Float64(result.state.fwd.ham),
        dham=Float64(result.state.dham),
        diverged=Bool(result.state.diverged),
        energy_errors=Tuple(Float64(value)
                            for value in result.effects.stats_f.errors[1:count]),
        normal_calls=Int(result.arguments[1].normal_index) - 1,
        exponential_calls=Int(result.arguments[1].exponential_index) - 1,
        rng_overflow=Bool(result.arguments[1].overflow),
        control_overflow=Bool(result.control_overflow),
    )
end

@testset "fixed-step HMC compiles through generic Reactant lowering" begin
    float_atol = 128eps(Float64)
    receipt = TOML.parsefile(joinpath(
        @__DIR__, "..", "benchmark", "receipts",
        "reactivehmc-hmc-ca9-v1.toml"))
    for case in receipt["cases"]
        program = _RHMC_HMC_REACTANT.build_case(case)
        state = _rhmc_hmc_trace(program.snapshot)
        replay = _rhmc_hmc_traced_replay(program.replay)
        compiled = @compile program.transition(state, replay)
        actual = _rhmc_hmc_host_values(compiled(state, replay))

        @test actual.init_pos ≈ case["init_pos"] atol=float_atol rtol=0
        @test actual.init_mom ≈ case["init_mom"] atol=float_atol rtol=0
        @test actual.init_ham ≈ case["init_ham"] atol=float_atol rtol=0
        @test actual.fwd_pos ≈ case["fwd_pos"] atol=float_atol rtol=0
        @test actual.fwd_mom ≈ case["fwd_mom"] atol=float_atol rtol=0
        @test actual.fwd_ham ≈ case["fwd_ham"] atol=float_atol rtol=0
        @test actual.dham ≈ case["dham"] atol=float_atol rtol=0
        @test actual.diverged == case["diverged"]
        @test collect(actual.energy_errors) ≈ case["energy_errors"] atol=float_atol rtol=0
        @test actual.normal_calls == case["normal_calls"]
        @test actual.exponential_calls == case["exponential_calls"]
        @test !actual.rng_overflow
        @test !actual.control_overflow

        if case["name"] == "accepted"
            exhausted_host = ReactiveKernels.OrderedRNGReplay(
                program.replay.normals, program.replay.uniforms,
                program.replay.exponentials,
                program.replay.normal_index,
                program.replay.uniform_index,
                length(program.replay.exponentials) + 1, false)
            exhausted = _rhmc_hmc_traced_replay(exhausted_host)
            overflow = compiled(state, exhausted)
            @test Bool(overflow.control_overflow)
            @test Bool(overflow.arguments[1].overflow)
            @test Int(overflow.effects.stats_f.count) == 0
            @test Array(overflow.state.init.pos) ==
                  Array(state.init.pos)
            @test Array(overflow.state.fwd.pos) ==
                  Array(state.fwd.pos)
        end
    end
end
