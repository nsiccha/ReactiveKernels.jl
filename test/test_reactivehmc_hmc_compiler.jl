using ReactiveKernels
using Test
import TOML

include(joinpath(@__DIR__, "fixtures",
                 "reactivehmc_hmc_compiler_support.jl"))
const _RHMC_HMC_COMPILER = ReactiveHMCHMCCompilerSupport

function _assert_hmc_receipt(actual, expected)
    @test collect(actual.init_pos) == expected["init_pos"]
    @test collect(actual.init_mom) == expected["init_mom"]
    @test actual.init_ham == expected["init_ham"]
    @test collect(actual.fwd_pos) == expected["fwd_pos"]
    @test collect(actual.fwd_mom) == expected["fwd_mom"]
    @test actual.fwd_ham == expected["fwd_ham"]
    @test actual.dham == expected["dham"]
    @test actual.diverged == expected["diverged"]
    @test collect(actual.energy_errors) == expected["energy_errors"]
    @test actual.normal_calls == expected["normal_calls"]
    @test actual.exponential_calls == expected["exponential_calls"]
    @test !actual.rng_overflow
    @test !actual.control_overflow
end

@testset "generic structured compiler reproduces fixed-step HMC" begin
    receipt = TOML.parsefile(joinpath(
        @__DIR__, "..", "benchmark", "receipts",
        "reactivehmc-hmc-ca9-v1.toml"))
    for case in receipt["cases"]
        compiled = _RHMC_HMC_COMPILER.build_case(case)
        @test propertynames(compiled.structured.repairs) == (:mom, :pos)
        @test compiled.snapshot.init.dpot ===
              compiled.snapshot.init.dham_dpos
        @test compiled.snapshot.fwd.dpot ===
              compiled.snapshot.fwd.dham_dpos

        result = compiled.transition(compiled.snapshot, compiled.replay)
        _assert_hmc_receipt(
            _RHMC_HMC_COMPILER.result_values(result), case)
        @test result.state.init.dpot === result.state.init.dham_dpos
        @test result.state.fwd.dpot === result.state.fwd.dham_dpos
        @test result.state.init.pos !== result.state.fwd.pos
        @test result.state.init.mom !== result.state.fwd.mom
        @test result.state.init.pot_f ===
              compiled.snapshot.init.pot_f
        @test result.state.fwd.grad_f ===
              compiled.snapshot.fwd.grad_f
    end
end

@testset "structured state and ordered effects fail closed" begin
    case = only(filter(
        case -> case["name"] == "accepted",
        TOML.parsefile(joinpath(
            @__DIR__, "..", "benchmark", "receipts",
            "reactivehmc-hmc-ca9-v1.toml"))["cases"]))
    compiled = _RHMC_HMC_COMPILER.build_case(case)

    # Exhausting the conditional exponential stream is observable and atomic:
    # the entire state/effect surface rolls back while overflow stays sticky on
    # the replay argument.
    replay = compiled.replay
    exhausted = ReactiveKernels.OrderedRNGReplay(
        replay.normals, replay.uniforms, replay.exponentials,
        replay.normal_index, replay.uniform_index,
        length(replay.exponentials) + 1, false)
    result = compiled.transition(compiled.snapshot, exhausted)
    @test result.control_overflow
    @test result.arguments[1].overflow
    @test result.arguments[1].normal_index ==
          exhausted.normal_index + 1
    @test result.arguments[1].exponential_index ==
          exhausted.exponential_index
    @test result.state == compiled.snapshot
    @test result.effects.stats_f.count == 0
    @test all(iszero, result.effects.stats_f.errors)

    @test_throws ArgumentError ReactiveKernels.OrderedRNGReplay(
        reshape([0.1, -0.2], :, 1), Bool[], [0.5])
    @test_throws ArgumentError ReactiveKernels.OrderedRNGReplay(
        reshape([0.1, -0.2], :, 1), Bool[false], Float64[])

    wrong = merge(compiled.snapshot.init,
                  (; pot_f = x -> sum(abs2, x)))
    @test_throws ArgumentError ReactiveKernels._sm_validate_structured_state_port(
        compiled.structured, wrong)

    # A compiled transition freezes its repair programs. Later mutation of the
    # authoring graph cannot change a newly constructed structured port.
    @test !(:spec in fieldnames(typeof(compiled.endpoint)))
    replacement = compiled.snapshot.init.mom .+ 0.125
    changed = ReactiveKernels._sm_structured_set(
        compiled.structured, compiled.snapshot.init,
        Val((:mom,)), replacement)
    expected = compiled.structured.repairs.mom(changed)
    empty!(compiled.bundle.spec.graph.recipes)
    empty!(compiled.bundle.spec.graph.producers)
    frozen = ReactiveKernels.structured_state_port(compiled.endpoint)
    actual = frozen.repairs.mom(changed)
    @test actual.mom == replacement
    @test actual.dham_dmom == expected.dham_dmom
    @test actual.ham == expected.ham
end
