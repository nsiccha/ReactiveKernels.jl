#!/usr/bin/env julia

import TOML

const EXPECTED_EVAL_SIZES = (16, 256, 4096)
const EXPECTED_EVAL_MODES = ("primal", "gradient", "gq")
const EXPECTED_EVAL_REPLICAS = 256

function validate_eval_throughput_receipt(path::AbstractString)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "eval-throughput-v1",
            "schema must be eval-throughput-v1")
    for section in ("pins", "environment", "protocol", "support", "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    for key in (
        "reactivekernels_sha", "reactivekernels_version", "reactant_version",
        "turing_version", "dynamicppl_version", "distributions_version",
        "differentiationinterface_version", "enzyme_version", "julia_version",
    )
        require(haskey(pins, key) && !isempty(string(pins[key])), "pins.$key missing")
    end
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")),
            "ReactiveKernels receipt pin must be a full commit SHA")

    protocol = receipt["protocol"]
    require(get(protocol, "reactant_sync", false) == true,
            "Reactant timings must use sync=true")
    require(get(protocol, "reactant_transfers_included", true) == false,
            "Reactant device-transfer policy must be explicit and false")
    require(get(protocol, "reactant_compile_time_in_timed_region", true) == false,
            "Reactant execution timings must exclude compilation")
    require(Int(get(protocol, "rounds", 0)) >= 50,
            "published receipt must use at least 50 timing rounds")
    require(Int(get(protocol, "replicas", 0)) == EXPECTED_EVAL_REPLICAS,
            "published receipt must use $EXPECTED_EVAL_REPLICAS replicas")

    support = receipt["support"]
    require(get(support, "turing_reactant", true) == false,
            "Turing + Reactant must remain explicitly unsupported")
    require(!isempty(get(support, "turing_reactant_reason", "")),
            "unsupported Turing + Reactant path needs a diagnostic")

    measurements = receipt["measurements"]
    require(length(measurements) ==
            length(EXPECTED_EVAL_SIZES) * length(EXPECTED_EVAL_MODES) * 4,
            "unexpected measurement count")
    for n in EXPECTED_EVAL_SIZES, mode in EXPECTED_EVAL_MODES
        expected = (
            ("reactivekernels", "native"),
            ("reactivekernels", "reactant"),
            ("turing", "native"),
            ("reactivekernels", "reactant_replicated"),
        )
        for (implementation, variant) in expected
            matching = filter(measurements) do row
                Int(row["size"]) == n && row["mode"] == mode &&
                    row["implementation"] == implementation && row["variant"] == variant
            end
            require(length(matching) == 1,
                    "n=$n mode=$mode expected one $implementation/$variant row")
            length(matching) == 1 || continue
            row = only(matching)
            require(Float64(row["median_ns"]) > 0,
                    "n=$n mode=$mode $variant median_ns must be positive")
            if variant == "reactant_replicated"
                require(Int(get(row, "batch_size", 0)) == EXPECTED_EVAL_REPLICAS,
                        "n=$n mode=$mode replicated batch size mismatch")
                require(Float64(get(row, "median_batch_ns", 0.0)) > 0,
                        "n=$n mode=$mode median_batch_ns must be positive")
                require(isapprox(
                    Float64(row["median_ns"]),
                    Float64(row["median_batch_ns"]) / EXPECTED_EVAL_REPLICAS;
                    rtol = 1e-12,
                ), "n=$n mode=$mode replicated normalization mismatch")
            end
        end
    end

    errors
end

function main(path)
    errors = validate_eval_throughput_receipt(path)
    if isempty(errors)
        println("VALIDATE OK — eval-throughput-v1: single-call and " *
                "$EXPECTED_EVAL_REPLICAS-replica rows accepted")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_eval_throughput.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
