#!/usr/bin/env julia

import SHA
import Statistics
import TOML

const EXPECTED_MNIST_REACTANT_AD_BOUNDARIES =
    ("packed_unconstrained", "structured_parameters")
const EXPECTED_MNIST_REACTANT_AD_OUTCOMES =
    ("joint", "prior", "likelihood", "pointwise")
# Native-AD scalar support published by mnist-logistic-ad-v1.
const EXPECTED_MNIST_REACTANT_AD_NATIVE = Set((
    ("packed_unconstrained", "joint"),
    ("packed_unconstrained", "prior"),
    ("packed_unconstrained", "likelihood"),
))
# Cells the published receipt must show compiling through Reactant-compiled AD.
# These are exactly the scalar cells supported by the matched native AD receipt;
# canonical eachcol/gather lowering makes all three mandatory.
const EXPECTED_MNIST_REACTANT_AD_COMPILED = (
    ("packed_unconstrained", "joint"),
    ("packed_unconstrained", "prior"),
    ("packed_unconstrained", "likelihood"),
)

_mnist_reactant_ad_median(values) = Statistics.median(Float64.(values))
_mnist_reactant_ad_text_sha256(path) = bytes2hex(SHA.sha256(
    replace(read(path, String), "\r\n" => "\n", "\r" => "\n")))

function validate_mnist_reactant_ad_receipt(
    path::AbstractString;
    ad_path::AbstractString = joinpath(
        dirname(path), "mnist-logistic-ad-v1.toml"),
)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "mnist-reactant-ad-v1",
            "schema must be mnist-reactant-ad-v1")
    for section in ("pins", "environment", "setup", "protocol", "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    require(isfile(ad_path), "missing matched AD receipt: $ad_path")
    isempty(errors) || return errors

    ad_receipt = TOML.parsefile(ad_path)
    require(get(ad_receipt, "schema", "") == "mnist-logistic-ad-v1",
            "matched receipt must use schema mnist-logistic-ad-v1")
    haskey(ad_receipt, "protocol") && haskey(ad_receipt, "measurements") ||
        require(false, "matched AD receipt lacks protocol or measurements")

    pins = receipt["pins"]
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a clean ReactiveKernels tree")
    for key in (
        "reactivekernels_sha", "reactivekernels_version",
        "reactivekernelspplexamples_version",
        "reactivekernelsdistributionkernels_version", "reactant_version",
        "reactant_jll_version", "enzyme_version",
        "differentiationinterface_version", "benchmarktools_version",
        "mldatasets_version", "julia_version", "source_authority_path",
        "source_authority_blob", "source_text_sha256", "ad_receipt_path",
        "ad_receipt_sha256", "ad_receipt_reactivekernels_sha",
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
    require(get(pins, "ad_receipt_path", "") ==
            "benchmark/receipts/mnist-logistic-ad-v1.toml",
            "benchmark must name the matched AD receipt")
    require(get(pins, "ad_receipt_sha256", "") ==
            _mnist_reactant_ad_text_sha256(ad_path),
            "matched AD receipt digest mismatch")
    if haskey(ad_receipt, "pins")
        require(get(pins, "ad_receipt_reactivekernels_sha", "") ==
                get(ad_receipt["pins"], "reactivekernels_sha", ""),
                "matched AD receipt code pin mismatch")
    end

    protocol = receipt["protocol"]
    require(Tuple(get(protocol, "input_boundaries", String[])) ==
            EXPECTED_MNIST_REACTANT_AD_BOUNDARIES,
            "input-boundary matrix mismatch")
    require(Tuple(get(protocol, "outcomes", String[])) ==
            EXPECTED_MNIST_REACTANT_AD_OUTCOMES,
            "outcome matrix mismatch")
    if haskey(ad_receipt, "protocol")
        require(Tuple(get(ad_receipt["protocol"], "input_boundaries", String[])) ==
                EXPECTED_MNIST_REACTANT_AD_BOUNDARIES,
                "matched AD receipt input-boundary matrix mismatch")
        require(Tuple(get(ad_receipt["protocol"], "outcomes", String[])) ==
                EXPECTED_MNIST_REACTANT_AD_OUTCOMES,
                "matched AD receipt outcome matrix mismatch")
        require(Int(get(protocol, "num_observations", 0)) ==
                Int(get(ad_receipt["protocol"], "num_observations", -1)),
                "observation count must match the AD receipt")
    end
    require(Int(get(protocol, "num_features", 0)) == 784,
            "MNIST Reactant-AD receipt must use full-resolution features")
    require(Int(get(protocol, "num_classes", 0)) == 10,
            "MNIST Reactant-AD receipt must cover ten classes")
    require(get(protocol, "source_reused", false) == true,
            "benchmark must reuse the authored MNIST source")
    require(get(protocol, "matrix_source", "") ==
            "benchmark/receipts/mnist-logistic-ad-v1.toml",
            "benchmark must name the matched AD receipt as its matrix source")
    require(get(protocol, "gradient_operation", "") == "value and gradient",
            "receipt must record the value-and-gradient operation")
    require(get(protocol, "rk_reactant_ad_surface", "") ==
            "prepare_ad + compile_ad_value_and_gradient",
            "receipt must consume the first-class Reactant-compiled AD verb")
    require(get(protocol, "partial_evaluation_enabled", false) == true,
            "benchmark must enable preparation-time partial evaluation")
    require(Tuple(get(protocol, "bound_ports", String[])) ==
            ("X", "y", "num_classes"),
            "benchmark must bind the complete dataset boundary")
    require(get(protocol, "native_and_reactant_use_same_bound_kernel", false) == true,
            "native and Reactant AD timings must share one bound kernel")
    require(get(protocol, "bound_values_in_timed_region", true) == false,
            "bound data setup must stay outside steady-state timing")
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
        "ad_preparation_in_timed_region",
        "reactant_compile_time_in_timed_region",
        "reactant_transfers_in_timed_region",
        "reactant_readback_in_timed_region",
        "first_execution_in_steady_state_region",
    )
        require(get(protocol, key, true) == false, "$key must be false")
    end
    require(get(protocol, "unsupported_cells_recorded", false) == true,
            "unsupported cells must retain diagnostics")
    require(get(protocol, "pointwise_jacobian_or_vjp_invented", true) == false,
            "pointwise must remain unsupported without a matched public contract")
    require(get(protocol, "structured_multi_active_boundary_invented", true) == false,
            "structured two-active-port AD must not be invented")

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
            length(EXPECTED_MNIST_REACTANT_AD_BOUNDARIES) *
            length(EXPECTED_MNIST_REACTANT_AD_OUTCOMES),
            "matrix must contain exactly one row per boundary/outcome pair")
    ad_supported = Set{Tuple{String,String}}()
    for row in get(ad_receipt, "measurements", Any[])
        get(row, "supported", false) &&
            push!(ad_supported, (row["boundary"], row["outcome"]))
    end
    require(ad_supported == EXPECTED_MNIST_REACTANT_AD_NATIVE,
            "matched AD receipt supported inventory drifted")
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
                             _mnist_reactant_ad_median(result["times_ns"]); rtol = 1e-12),
                    "$label median mismatch")
        end
    end
    for boundary in EXPECTED_MNIST_REACTANT_AD_BOUNDARIES,
        outcome in EXPECTED_MNIST_REACTANT_AD_OUTCOMES
        matches = filter(rows) do row
            row["boundary"] == boundary && row["outcome"] == outcome
        end
        require(length(matches) == 1, "expected one row for $boundary / $outcome")
        length(matches) == 1 || continue
        row = only(matches)

        native_expected = (boundary, outcome) in EXPECTED_MNIST_REACTANT_AD_NATIVE
        require(get(row, "rk_native_ad_supported", false) == native_expected,
                "$boundary / $outcome native AD support drifted from AD receipt")
        require(haskey(row, "rk_native_ad") == native_expected,
                "$boundary / $outcome native AD measurement support mismatch")

        if native_expected
            require(get(row, "active_port", "") == "unconstrained",
                    "$boundary / $outcome active port drifted")
            haskey(row, "rk_native_ad") &&
                checked_measurement(row["rk_native_ad"],
                                    "$boundary / $outcome native AD")
            require(Float64(get(row, "ad_preparation_seconds", -1.0)) >= 0,
                    "$boundary / $outcome AD preparation missing")
            require(Float64(get(row, "reactant_transfer_seconds", -1.0)) >= 0,
                    "$boundary / $outcome transfer setup missing")
            require(Float64(get(row, "reactant_ad_compile_seconds", -1.0)) >= 0,
                    "$boundary / $outcome compile attempt missing")
        end

        reactant_supported = get(row, "rk_reactant_ad_supported", false)
        if reactant_supported
            push!(supported_reactant, (boundary, outcome))
            require(native_expected,
                    "$boundary / $outcome compiled without native AD support")
            require(haskey(row, "rk_reactant_ad"),
                    "$boundary / $outcome supported without a Reactant measurement")
            require(!haskey(row, "rk_reactant_ad_error"),
                    "$boundary / $outcome supported but retains an error")
            require(Float64(get(row, "reactant_first_execution_seconds", -1.0)) >= 0,
                    "$boundary / $outcome first execution missing")
            require(Float64(get(row, "max_rel_error", Inf)) <= 1e-9,
                    "$boundary / $outcome exceeds gradient parity tolerance")
            require(Float64(get(row, "value_rel_error", Inf)) <= 1e-9,
                    "$boundary / $outcome exceeds value parity tolerance")
            require(haskey(row, "max_abs_error") && haskey(row, "value_abs_error"),
                    "$boundary / $outcome parity magnitudes missing")
            haskey(row, "rk_reactant_ad") && checked_measurement(
                row["rk_reactant_ad"], "$boundary / $outcome Reactant AD")
        else
            require(!haskey(row, "rk_reactant_ad"),
                    "$boundary / $outcome unsupported but contains a measurement")
            require(!isempty(get(row, "rk_reactant_ad_error", "")),
                    "$boundary / $outcome unsupported without a diagnostic")
        end
    end
    require(supported_reactant == Set(EXPECTED_MNIST_REACTANT_AD_COMPILED),
            "all native scalar AD cells must compile through Reactant")
    errors
end

function main(path)
    errors = validate_mnist_reactant_ad_receipt(path)
    if isempty(errors)
        println("VALIDATE OK — mnist-reactant-ad-v1: matched 2×4 matrix accepted")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_mnist_reactant_ad.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
