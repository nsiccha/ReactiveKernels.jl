using SHA
using Test
using TOML

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const RECEIPT_PATH = joinpath(@__DIR__, "sum-to-zero-native-v1.toml")

function _normalized_sha256(path)
    text = replace(read(path, String), "\r\n" => "\n", "\r" => "\n")
    bytes2hex(sha256(text))
end

function _measurement(receipt, configuration)
    only(filter(
        row -> row["configuration"] == configuration,
        receipt["measurements"],
    ))
end

@testset "sum-to-zero native benchmark receipt" begin
    receipt = TOML.parsefile(RECEIPT_PATH)
    @test receipt["schema"] == "sum-to-zero-hotloop-v1"
    @test receipt["pins"]["reactivekernels_dirty"] == false
    @test occursin(
        r"^[0-9a-f]{40}$", receipt["pins"]["reactivekernels_sha"])

    protocol = receipt["protocol"]
    @test protocol["boundary"] == "packed unconstrained parameters"
    @test protocol["outcome"] == "full unconstrained posterior"
    @test protocol["hot_loop_excludes_recovery"] == true
    @test protocol["setup_in_timed_region"] == false
    @test protocol["preparation_in_timed_region"] == false
    @test protocol["first_execution_in_steady_state_region"] == false
    @test protocol["rounds"] >= 10

    expected = Set((
        "rk_primal_native", "manual_primal", "turing_primal",
        "rk_ad_native", "manual_ad", "turing_ad",
    ))
    @test Set(row["configuration"] for row in receipt["measurements"]) == expected
    @test all(row["state"] == "supported" for row in receipt["measurements"])
    @test all(length(row["result"]["times_ns"]) == protocol["rounds"]
              for row in receipt["measurements"])

    reference = _measurement(receipt, "manual_primal")["value"]
    @test all(isapprox(row["value"], reference; rtol = 1e-11, atol = 1e-12)
              for row in receipt["measurements"]
              if row["differentiation"] == "primal")
    @test all(row["result"]["value_abs_error"] <= 1e-11
              for row in receipt["measurements"]
              if row["differentiation"] == "value_and_gradient")
    @test all(row["result"]["gradient_max_abs_error"] <= 1e-8
              for row in receipt["measurements"]
              if row["differentiation"] == "value_and_gradient")

    for pin in ("model_source", "comparator_source")
        source = receipt["pins"][pin]
        path = joinpath(ROOT, source["path"])
        @test isfile(path)
        @test _normalized_sha256(path) == source["text_sha256"]
        @test readchomp(`git -C $ROOT hash-object $path`) == source["git_blob"]
    end
end
