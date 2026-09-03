using SHA
using Test
using TOML

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const RECEIPT_PATH = joinpath(@__DIR__, "sum-to-zero-reactant-v1.toml")
const NATIVE_RECEIPT_PATH = joinpath(@__DIR__, "sum-to-zero-native-v1.toml")

function _normalized_sha256(path)
    text = replace(read(path, String), "\r\n" => "\n", "\r" => "\n")
    bytes2hex(sha256(text))
end

@testset "sum-to-zero Reactant benchmark receipt" begin
    receipt = TOML.parsefile(RECEIPT_PATH)
    @test receipt["schema"] == "sum-to-zero-reactant-v1"
    @test receipt["pins"]["reactivekernels_dirty"] == false
    @test occursin(
        r"^[0-9a-f]{40}$", receipt["pins"]["reactivekernels_sha"])

    protocol = receipt["protocol"]
    @test protocol["boundary"] == "packed unconstrained parameters"
    @test protocol["outcome"] == "full unconstrained posterior"
    @test protocol["hot_loop_excludes_recovery"] == true
    @test protocol["source_reused"] == true
    @test protocol["reactant_sync"] == true
    @test protocol["reactant_transfers_in_timed_region"] == false
    @test protocol["reactant_compile_time_in_timed_region"] == false
    @test protocol["reactant_readback_in_timed_region"] == false
    @test protocol["first_execution_in_steady_state_region"] == false
    @test protocol["rounds"] >= 20

    expected = Set((
        "rk_primal_reactant", "manual_primal_reactant", "turing_reactant",
        "rk_ad_reactant", "manual_ad_reactant", "turing_reactant_ad",
    ))
    measurements = receipt["measurements"]
    @test Set(row["configuration"] for row in measurements) == expected
    @test all(row["state"] == "supported" for row in measurements
              if row["provider"] != "turing")
    @test all(row["state"] == "unsupported" for row in measurements
              if row["provider"] == "turing")
    @test all(length(row["result"]["times_ns"]) == protocol["rounds"]
              for row in measurements if row["state"] == "supported")
    @test all(row["value_abs_error"] <= 1e-10
              for row in measurements if row["state"] == "supported")
    @test all(row["gradient_max_abs_error"] <= 1e-10
              for row in measurements
              if row["differentiation"] == "value_and_gradient" &&
                 row["state"] == "supported")

    pins = receipt["pins"]
    @test pins["native_receipt_sha256"] ==
        bytes2hex(sha256(read(NATIVE_RECEIPT_PATH)))
    native = TOML.parsefile(NATIVE_RECEIPT_PATH)
    @test pins["native_receipt_reactivekernels_sha"] ==
        native["pins"]["reactivekernels_sha"]
    for pin in (
        "model_source", "native_comparator_source",
        "reactant_comparator_source",
    )
        source = pins[pin]
        path = joinpath(ROOT, source["path"])
        @test isfile(path)
        @test _normalized_sha256(path) == source["text_sha256"]
        @test readchomp(`git -C $ROOT hash-object $path`) == source["git_blob"]
    end
end
