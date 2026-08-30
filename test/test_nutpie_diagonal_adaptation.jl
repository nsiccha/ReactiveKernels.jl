using SHA
using TOML

include(joinpath(@__DIR__, "..", "examples", "nutpie_diagonal_adaptation.jl"))
using .NutpieDiagonalAdaptationExample:
    NUTS_RS_REVISION, NUTS_RS_SOURCE_DIGESTS, ORACLE_INPUTS,
    initial_state, advance

const _NUTPIE_ORACLE_PATH =
    joinpath(@__DIR__, "..", "benchmark", "nutpie_diag_oracle.toml")
const _NUTPIE_ORACLE_RECEIPT_PATH =
    joinpath(@__DIR__, "..", "benchmark", "nutpie_diag_oracle_receipt.toml")
const _NUTPIE_ORACLE_SOURCE =
    joinpath(@__DIR__, "..", "benchmark", "nutpie_diag_oracle.rs")

_nutpie_float(bits::AbstractString) = reinterpret(Float64, parse(UInt64, bits; base = 16))
_nutpie_floats(bits) = _nutpie_float.(bits)

function _nutpie_assert_state(state, stage)
    for name in (
        :draw_mean, :draw_variance, :grad_mean, :grad_variance,
        :background_draw_mean, :background_draw_variance,
        :background_grad_mean, :background_grad_variance,
        :stds, :inv_stds, :transformation_mean,
    )
        @test getproperty(state, name) ≈ _nutpie_floats(stage[string(name)]) rtol = 2e-15
    end
    @test state.count == stage["count"]
    @test state.background_count == stage["background_count"]
    @test state.logdet ≈ _nutpie_float(stage["logdet"]) rtol = 2e-15
    @test state.transformation_id == stage["transformation_id"]
end

@testset "nutpie diagonal adaptation is an external mathematical RK kernel" begin
    example_source = read(joinpath(@__DIR__, "..", "examples",
                                   "nutpie_diagonal_adaptation.jl"), String)
    oracle_source = read(_NUTPIE_ORACLE_SOURCE, String)

    @test occursin("@kernel nutpie_diagonal_initialize(", example_source)
    @test occursin("@kernel nutpie_diagonal_adaptation(", example_source)
    @test occursin(NUTS_RS_REVISION, example_source)
    for (path, digest) in NUTS_RS_SOURCE_DIGESTS
        @test occursin(path, oracle_source)
        @test occursin(digest, oracle_source)
    end

    # Domain words remain in the opt-in example/corpus, never compiler handlers.
    for root in (joinpath(@__DIR__, "..", "src"), joinpath(@__DIR__, "..", "ext"))
        for (dir, _, files) in walkdir(root), file in files
            endswith(file, ".jl") || continue
            @test !occursin(r"(?i)nutpie|runningvariance|diagmassmatrix",
                            read(joinpath(dir, file), String))
        end
    end

    oracle = TOML.parsefile(_NUTPIE_ORACLE_PATH)
    receipt = TOML.parsefile(_NUTPIE_ORACLE_RECEIPT_PATH)
    @test oracle["upstream_revision"] == NUTS_RS_REVISION
    @test oracle["rng_consumption"] == "none"
    @test receipt["oracle_source_sha256"] == bytes2hex(sha256(oracle_source))
    @test receipt["oracle_output_sha256"] ==
          bytes2hex(sha256(read(_NUTPIE_ORACLE_PATH)))
    @test receipt["rustc_version"] == "rustc 1.98.0 (88d9e12ae 2026-08-18)"
    @test receipt["execution_host"] == "official Rust Playground stable/release/edition-2021"
    @test Dict(file["path"] => file["sha256"] for file in oracle["upstream_files"]) ==
          Dict(NUTS_RS_SOURCE_DIGESTS)

    inputs = oracle["inputs"]
    stages = oracle["stages"]
    @test getindex.(inputs, "name") == ["init", "step1", "step2", "step3", "step4", "step5"]
    @test getindex.(stages, "name") == getindex.(inputs, "name")

    init = first(inputs)
    @test _nutpie_floats(init["position"]) == ORACLE_INPUTS.init_position
    @test _nutpie_floats(init["gradient"]) == ORACLE_INPUTS.init_gradient
    state = initial_state(ORACLE_INPUTS.init_position, ORACLE_INPUTS.init_gradient)
    _nutpie_assert_state(state, first(stages))

    # Explicitly pin the important source semantics: adaptation starts at three
    # samples; a rejected draw does not update either estimator but does rerun the
    # transform; a switch consumes the just-updated background and resets it.
    @test stages[2]["adapted"] == false
    @test stages[3]["adapted"] == true
    @test stages[4]["count"] == stages[3]["count"]
    @test stages[4]["transformation_id"] == stages[3]["transformation_id"] + 1
    @test stages[5]["background_count"] == 0
    @test stages[6]["background_count"] == 1

    for (index, step) in enumerate(ORACLE_INPUTS.steps)
        physical_input = inputs[index + 1]
        @test _nutpie_floats(physical_input["position"]) == step.position
        @test _nutpie_floats(physical_input["gradient"]) == step.gradient
        @test physical_input["is_good"] == step.is_good
        @test physical_input["switch_now"] == step.switch_now
        @test physical_input["adapt_now"] == step.adapt_now
        state = advance(state, step.position, step.gradient;
            is_good = step.is_good,
            switch_now = step.switch_now,
            adapt_now = step.adapt_now)
        _nutpie_assert_state(state, stages[index + 1])
    end

    # Dimensions 2:4 respectively produce a zero, infinity, and NaN ratio.
    # fill_invalid=None preserves their initialized scales through every update.
    initialized_stds = _nutpie_floats(stages[1]["stds"])
    @test state.stds[2:4] == initialized_stds[2:4]
    @test state == NutpieDiagonalAdaptationExample.run()
end
