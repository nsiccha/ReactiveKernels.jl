#!/usr/bin/env julia

import SHA
import Statistics
import TOML

const EXPECTED_MNIST_REACTANT_BOUNDARIES =
    ("packed_unconstrained", "structured_parameters")
const EXPECTED_MNIST_REACTANT_OUTCOMES =
    ("joint", "prior", "likelihood", "pointwise")
# Cells the published receipt must show compiling through Reactant; a receipt
# where one of these regressed to unsupported is rejected, not silently landed.
# The likelihood-bearing cells currently fail to trace (the constant
# reference-logits row reaches Base's generic elementwise vcat — scalar
# indexing); extend this set when that upstream capability lands.
const EXPECTED_MNIST_REACTANT_COMPILED = (
    ("packed_unconstrained", "prior"),
    ("structured_parameters", "prior"),
)

_mnist_reactant_median(values) = Statistics.median(Float64.(values))
_mnist_reactant_text_sha256(path) = bytes2hex(SHA.sha256(
    replace(read(path, String), "\r\n" => "\n", "\r" => "\n")))

function validate_mnist_reactant_receipt(
    path::AbstractString;
    primal_path::AbstractString = joinpath(
        dirname(path), "mnist-logistic-v1.toml"),
)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "mnist-reactant-v1",
            "schema must be mnist-reactant-v1")
    for section in ("pins", "environment", "setup", "protocol", "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    require(isfile(primal_path), "missing matched primal receipt: $primal_path")
    isempty(errors) || return errors

    primal = TOML.parsefile(primal_path)
    require(get(primal, "schema", "") == "mnist-logistic-v1",
            "matched receipt must use schema mnist-logistic-v1")
    haskey(primal, "protocol") && haskey(primal, "measurements") ||
        require(false, "matched primal receipt lacks protocol or measurements")

    pins = receipt["pins"]
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a clean ReactiveKernels tree")
    for key in (
        "reactivekernels_sha", "reactivekernels_version",
        "reactivekernelspplexamples_version",
        "reactivekernelsdistributionkernels_version", "reactant_version",
        "reactant_jll_version", "benchmarktools_version", "mldatasets_version",
        "julia_version", "source_authority_path", "source_authority_blob",
        "source_text_sha256", "primal_receipt_sha256",
        "primal_receipt_reactivekernels_sha",
    )
        require(haskey(pins, key) && !isempty(string(pins[key])), "pins.$key missing")
    end
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")),
            "ReactiveKernels receipt pin must be a full commit SHA")
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "source_authority_blob", "")),
            "source authority must be a full Git blob SHA")
    require(occursin(r"^[0-9a-f]{64}$", get(pins, "source_text_sha256", "")),
            "source text must carry a SHA-256 digest")
    require(get(pins, "source_authority_path", "") ==
            "packages/ReactiveKernelsPPLExamples/src/mnist_logistic.jl",
            "unexpected MNIST source-authority path")
    require(get(pins, "primal_receipt_sha256", "") ==
            _mnist_reactant_text_sha256(primal_path),
            "matched primal receipt digest mismatch")
    if haskey(primal, "pins")
        require(get(pins, "primal_receipt_reactivekernels_sha", "") ==
                get(primal["pins"], "reactivekernels_sha", ""),
                "matched primal receipt code pin mismatch")
    end

    protocol = receipt["protocol"]
    require(Tuple(get(protocol, "input_boundaries", String[])) ==
            EXPECTED_MNIST_REACTANT_BOUNDARIES,
            "input-boundary matrix mismatch")
    require(Tuple(get(protocol, "outcomes", String[])) ==
            EXPECTED_MNIST_REACTANT_OUTCOMES,
            "outcome matrix mismatch")
    if haskey(primal, "protocol")
        require(Tuple(get(primal["protocol"], "input_boundaries", String[])) ==
                EXPECTED_MNIST_REACTANT_BOUNDARIES,
                "matched primal receipt input-boundary matrix mismatch")
        require(Tuple(get(primal["protocol"], "outcomes", String[])) ==
                EXPECTED_MNIST_REACTANT_OUTCOMES,
                "matched primal receipt outcome matrix mismatch")
        require(Int(get(protocol, "num_observations", 0)) ==
                Int(get(primal["protocol"], "num_observations", -1)),
                "observation count must match the primal receipt")
    end
    require(Int(get(protocol, "num_features", 0)) == 784,
            "MNIST Reactant receipt must use full-resolution features")
    require(Int(get(protocol, "num_classes", 0)) == 10,
            "MNIST Reactant receipt must cover ten classes")
    require(get(protocol, "source_reused", false) == true,
            "benchmark must reuse the authored MNIST source")
    require(get(protocol, "matrix_source", "") ==
            "benchmark/receipts/mnist-logistic-v1.toml",
            "benchmark must name the matched primal receipt")
    require(Int(get(protocol, "rounds", 0)) >= 10,
            "published receipt must contain at least ten raw rounds")
    require(Int(get(protocol, "samples_per_round", 0)) >= 1,
            "per-round sample budget missing")
    require(Float64(get(protocol, "seconds_per_round", 0.0)) > 0,
            "per-round timing budget must be positive")
    require(get(protocol, "reactant_sync", false) == true,
            "Reactant timings must use sync=true")
    for key in (
        "setup_in_timed_region", "preparation_in_timed_region",
        "reactant_compile_time_in_timed_region",
        "reactant_transfers_in_timed_region",
        "reactant_readback_in_timed_region",
    )
        require(get(protocol, key, true) == false, "$key must be false")
    end
    require(get(protocol, "unsupported_cells_recorded", false) == true,
            "unsupported Reactant cells must retain diagnostics")
    require(get(protocol, "gradients_included", true) == false,
            "MNIST Reactant receipt must not contain gradients")
    require(get(protocol, "generated_predictions_included", true) == false,
            "predictive generated quantities must not enter this comparison")

    setup = receipt["setup"]
    for key in (
        "environment_seconds", "package_precompile_seconds",
        "data_load_seconds", "kernel_preparation_seconds",
    )
        require(Float64(get(setup, key, -1.0)) >= 0,
                "setup.$key must be nonnegative")
    end

    rows = receipt["measurements"]
    require(length(rows) ==
            length(EXPECTED_MNIST_REACTANT_BOUNDARIES) *
            length(EXPECTED_MNIST_REACTANT_OUTCOMES),
            "matrix must contain exactly one row per boundary/outcome pair")
    primal_rows = haskey(primal, "measurements") ? primal["measurements"] : Any[]
    supported_reactant = Set{Tuple{String,String}}()
    function checked_measurement(result, label)
        require(length(get(result, "times_ns", Float64[])) >= 10,
                "$label needs ten timing rounds")
        if haskey(result, "times_ns") && !isempty(result["times_ns"])
            require(all(>(0), result["times_ns"]),
                    "$label has non-positive timing")
            require(haskey(result, "min_ns") &&
                    isapprox(Float64(result["min_ns"]),
                             minimum(Float64.(result["times_ns"])); rtol = 1e-12),
                    "$label minimum mismatch")
            require(haskey(result, "median_ns") &&
                    isapprox(Float64(result["median_ns"]),
                             _mnist_reactant_median(result["times_ns"]); rtol = 1e-12),
                    "$label median mismatch")
        end
    end
    for boundary in EXPECTED_MNIST_REACTANT_BOUNDARIES,
        outcome in EXPECTED_MNIST_REACTANT_OUTCOMES
        matches = filter(rows) do row
            row["boundary"] == boundary && row["outcome"] == outcome
        end
        require(length(matches) == 1, "expected one row for $boundary / $outcome")
        length(matches) == 1 || continue
        row = only(matches)

        primal_matches = filter(primal_rows) do candidate
            candidate["boundary"] == boundary && candidate["outcome"] == outcome
        end
        require(length(primal_matches) == 1,
                "matched primal receipt needs one $boundary / $outcome row")
        require(length(primal_matches) == 1 &&
                haskey(only(primal_matches), "rk_native"),
                "matched primal receipt lost native $boundary / $outcome support")
        require(get(row, "rk_native_supported", false) == true,
                "$boundary / $outcome native support drifted from primal receipt")
        require(haskey(row, "rk_native"),
                "$boundary / $outcome native measurement missing")
        haskey(row, "rk_native") &&
            checked_measurement(row["rk_native"], "$boundary / $outcome native")
        require(Float64(get(row, "reactant_transfer_seconds", -1.0)) >= 0,
                "$boundary / $outcome transfer setup missing")
        require(Float64(get(row, "reactant_compile_seconds", -1.0)) >= 0,
                "$boundary / $outcome compile attempt missing")

        reactant_supported = get(row, "rk_reactant_supported", false)
        if reactant_supported
            push!(supported_reactant, (boundary, outcome))
            require(haskey(row, "rk_reactant"),
                    "$boundary / $outcome supported without a Reactant measurement")
            require(!haskey(row, "rk_reactant_error"),
                    "$boundary / $outcome supported but retains an error")
            require(Float64(get(row, "reactant_first_execution_seconds", -1.0)) >= 0,
                    "$boundary / $outcome first execution missing")
            require(Float64(get(row, "max_rel_error", Inf)) <= 1e-9,
                    "$boundary / $outcome exceeds native/Reactant parity tolerance")
            require(haskey(row, "max_abs_error"),
                    "$boundary / $outcome parity magnitude missing")
            haskey(row, "rk_reactant") && checked_measurement(
                row["rk_reactant"], "$boundary / $outcome Reactant")
        else
            require(!haskey(row, "rk_reactant"),
                    "$boundary / $outcome unsupported but contains a measurement")
            require(!isempty(get(row, "rk_reactant_error", "")),
                    "$boundary / $outcome unsupported without a diagnostic")
        end
    end
    for (boundary, outcome) in EXPECTED_MNIST_REACTANT_COMPILED
        require((boundary, outcome) in supported_reactant,
                "$boundary / $outcome must compile through Reactant")
    end
    errors
end

function main(path)
    errors = validate_mnist_reactant_receipt(path)
    if isempty(errors)
        println("VALIDATE OK — mnist-reactant-v1: matched 2×4 matrix accepted")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_mnist_reactant.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
