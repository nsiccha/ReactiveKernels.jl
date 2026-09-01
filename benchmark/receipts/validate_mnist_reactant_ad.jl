#!/usr/bin/env julia

module MNISTReactantADV1Validation

import SHA
import Statistics
import TOML

include(joinpath(@__DIR__, "_validate_mnist_dataset_profile.jl"))

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
    end
    _validate_mnist_dataset_profile!(require, protocol, ad_receipt["protocol"])
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
    require(get(protocol, "partial_evaluation_enabled", true) == false,
            "Reactant-AD receipt must keep data as runtime inputs")
    require(Tuple(get(protocol, "runtime_data_ports", String[])) ==
            ("X", "y", "num_classes"),
            "benchmark must record the complete runtime data boundary")
    require(get(protocol, "native_and_reactant_use_same_runtime_boundary", false) == true,
            "native and Reactant AD timings must share one runtime boundary")
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
end # module MNISTReactantADV1Validation

import SHA
import Statistics
import TOML

include(joinpath(@__DIR__, "_validate_mnist_dataset_profile.jl"))

isdefined(@__MODULE__, :MNISTLogisticMatrixSpec) ||
    include(joinpath(dirname(@__DIR__), "mnist_logistic_matrix_spec.jl"))
using .MNISTLogisticMatrixSpec

const EXPECTED_MNIST_REACTANT_AD_CONFIGURATIONS = Tuple(
    configuration for configuration in MNIST_RK_CONFIGURATIONS
    if configuration.differentiation == "value_and_gradient" &&
       configuration.compiler == "reactant"
)

_mnist_reactant_ad_median(values) = Statistics.median(Float64.(values))
_mnist_reactant_ad_text_sha256(path) = bytes2hex(SHA.sha256(
    replace(read(path, String), "\r\n" => "\n", "\r" => "\n")))

function _validate_mnist_reactant_ad_v2_receipt(
    path::AbstractString;
    ad_path::AbstractString = joinpath(dirname(path), "mnist-logistic-ad-v2.toml"),
)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "mnist-reactant-ad-v2",
            "schema must be mnist-reactant-ad-v2")
    for section in ("pins", "environment", "setup", "protocol", "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    require(isfile(ad_path), "missing matched native-AD receipt: $ad_path")
    isempty(errors) || return errors

    native_ad = TOML.parsefile(ad_path)
    require(get(native_ad, "schema", "") == "mnist-logistic-ad-v2",
            "matched receipt must use schema mnist-logistic-ad-v2")
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
        "source_authority_blob", "idiomatic_source_text_sha256",
        "vcat_free_source_text_sha256", "ad_receipt_path",
        "ad_receipt_sha256", "ad_receipt_reactivekernels_sha",
    )
        require(haskey(pins, key) && !isempty(string(pins[key])), "pins.$key missing")
    end
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")),
            "ReactiveKernels receipt pin must be a full commit SHA")
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "source_authority_blob", "")),
            "source authority must be a full Git blob SHA")
    for key in ("idiomatic_source_text_sha256", "vcat_free_source_text_sha256")
        require(occursin(r"^[0-9a-f]{64}$", get(pins, key, "")),
                "$key must be a SHA-256 digest")
    end
    require(get(pins, "source_authority_path", "") ==
            "packages/ReactiveKernelsPPLExamples/src/mnist_logistic.jl",
            "unexpected MNIST source-authority path")
    require(get(pins, "ad_receipt_path", "") ==
            "benchmark/receipts/mnist-logistic-ad-v2.toml",
            "matched native-AD path mismatch")
    require(get(pins, "ad_receipt_sha256", "") ==
            _mnist_reactant_ad_text_sha256(ad_path),
            "matched native-AD receipt digest mismatch")
    require(get(pins, "ad_receipt_reactivekernels_sha", "") ==
            get(native_ad["pins"], "reactivekernels_sha", ""),
            "matched native-AD receipt code pin mismatch")

    protocol = receipt["protocol"]
    require(Tuple(get(protocol, "input_boundaries", String[])) == MNIST_BOUNDARIES,
            "input-boundary matrix mismatch")
    require(Tuple(get(protocol, "outcomes", String[])) == MNIST_OUTCOMES,
            "outcome matrix mismatch")
    require(Tuple(get(protocol, "models", String[])) == MNIST_MODELS,
            "model inventory mismatch")
    require(Tuple(get(protocol, "rk_configurations", String[])) ==
            Tuple(configuration.id for configuration in
                  EXPECTED_MNIST_REACTANT_AD_CONFIGURATIONS),
            "Reactant-AD configuration inventory mismatch")
    require(get(protocol, "matrix_layout", "") ==
            "long-form provider/model/configuration/boundary/outcome rows",
            "Reactant-AD receipt must use the long-form capability matrix")
    require(Tuple(get(protocol, "bound_ports", String[])) ==
            ("X", "y", "num_classes"), "bound-port inventory mismatch")
    require(Int(get(protocol, "num_observations", 0)) ==
            Int(get(native_ad["protocol"], "num_observations", -1)),
            "observation count must match native AD")
    require(Int(get(protocol, "num_features", 0)) == 784,
            "MNIST Reactant-AD receipt must use all 784 pixels")
    _validate_mnist_dataset_profile!(require, protocol, native_ad["protocol"])
    require(Int(get(protocol, "num_classes", 0)) == 10,
            "MNIST Reactant-AD receipt must cover ten classes")
    require(get(protocol, "matrix_source", "") ==
            "benchmark/receipts/mnist-logistic-ad-v2.toml",
            "benchmark must name the matched native-AD receipt")
    require(Int(get(protocol, "rounds", 0)) >= 10,
            "published receipt must contain at least ten rounds")
    require(get(protocol, "reactant_sync", false) == true,
            "Reactant timings must use sync=true")
    for key in ("setup_in_timed_region", "preparation_in_timed_region",
                "ad_preparation_in_timed_region",
                "reactant_compile_time_in_timed_region",
                "reactant_transfers_in_timed_region",
                "reactant_readback_in_timed_region",
                "first_execution_in_steady_state_region")
        require(get(protocol, key, true) == false, "$key must be false")
    end
    require(get(protocol, "pointwise_jacobian_or_vjp_invented", true) == false,
            "pointwise Jacobian/VJP must not be invented")
    require(get(protocol, "structured_multi_active_boundary_invented", true) == false,
            "structured multi-active AD must not be invented")

    setup = receipt["setup"]
    for key in ("environment_seconds", "package_precompile_seconds",
                "data_load_seconds", "kernel_preparation_seconds")
        require(Float64(get(setup, key, -1)) >= 0, "setup.$key must be nonnegative")
    end

    function checked_measurement(result, label)
        times = Float64.(get(result, "times_ns", Float64[]))
        require(length(times) >= 10, "$label needs ten timing rounds")
        isempty(times) && return
        require(all(>(0), times), "$label has non-positive timing")
        require(isapprox(Float64(get(result, "min_ns", NaN)), minimum(times);
                         rtol = 1e-12), "$label minimum mismatch")
        require(isapprox(Float64(get(result, "median_ns", NaN)),
                         _mnist_reactant_ad_median(times); rtol = 1e-12),
                "$label median mismatch")
    end

    rows = receipt["measurements"]
    expected_count = length(MNIST_MODELS) *
        length(EXPECTED_MNIST_REACTANT_AD_CONFIGURATIONS) *
        length(MNIST_BOUNDARIES) * length(MNIST_OUTCOMES)
    require(length(rows) == expected_count,
            "long-form Reactant-AD matrix must contain $expected_count rows")
    keys = [(row["model"], row["configuration"], row["boundary"], row["outcome"])
            for row in rows]
    require(length(Set(keys)) == length(keys),
            "Reactant-AD matrix contains duplicate cells")

    compiled_prior_models = Set{String}()
    for model in MNIST_MODELS,
        configuration in EXPECTED_MNIST_REACTANT_AD_CONFIGURATIONS,
        boundary in MNIST_BOUNDARIES, outcome in MNIST_OUTCOMES
        matches = filter(rows) do row
            row["provider"] == "rk" && row["model"] == model &&
                row["configuration"] == configuration.id &&
                row["boundary"] == boundary && row["outcome"] == outcome
        end
        require(length(matches) == 1,
                "expected one RK / $model / $(configuration.id) / " *
                "$boundary / $outcome row")
        length(matches) == 1 || continue
        row = only(matches)
        expected_state, _ = matrix_support(configuration, boundary, outcome)

        native_configuration = configuration.data == "bound" ?
            "ad_native_bound" : "ad_native"
        native_matches = filter(native_ad["measurements"]) do candidate
            candidate["provider"] == "rk" && candidate["model"] == model &&
                candidate["configuration"] == native_configuration &&
                candidate["boundary"] == boundary && candidate["outcome"] == outcome
        end
        require(length(native_matches) == 1,
                "matched native-AD row missing for $model / " *
                "$native_configuration / $boundary / $outcome")
        length(native_matches) == 1 && require(
            get(only(native_matches), "state", "") == expected_state,
            "matched native-AD support drift")

        if expected_state != "supported"
            require(get(row, "state", "") == expected_state,
                    "deliberate Reactant-AD support state drift")
            require(!haskey(row, "result"),
                    "non-measured Reactant-AD cell has data")
            require(!isempty(get(row, "reason", "")),
                    "non-measured Reactant-AD cell lacks a reason")
            continue
        end

        require(get(row, "active_port", "") == "unconstrained",
                "Reactant-AD active port mismatch")
        require(Float64(get(row, "ad_preparation_seconds", -1.0)) >= 0,
                "native AD preparation time missing")
        require(haskey(row, "native_control"),
                "supported Reactant-AD cell lacks native control")
        haskey(row, "native_control") && checked_measurement(
            row["native_control"], "$model / $(configuration.id) / " *
            "$boundary / $outcome native AD")
        require(Float64(get(row, "reactant_transfer_seconds", -1.0)) >= 0,
                "Reactant-AD transfer setup missing")
        require(Float64(get(row, "reactant_ad_compile_seconds", -1.0)) >= 0,
                "Reactant-AD compile attempt missing")
        state = get(row, "state", "")
        require(state in ("supported", "unsupported_runtime"),
                "runtime Reactant-AD state is invalid")
        if state == "supported"
            require(haskey(row, "result"),
                    "supported Reactant-AD cell lacks a measurement")
            haskey(row, "result") && checked_measurement(
                row["result"], "$model / $(configuration.id) / " *
                "$boundary / $outcome Reactant AD")
            require(Float64(get(row, "reactant_first_execution_seconds", -1.0)) >= 0,
                    "Reactant-AD first execution missing")
            require(Float64(get(row, "max_rel_error", Inf)) <= 1e-9,
                    "native/Reactant gradient parity tolerance exceeded")
            require(Float64(get(row, "value_rel_error", Inf)) <= 1e-9,
                    "native/Reactant value parity tolerance exceeded")
            configuration.data == "unbound" && outcome == "prior" &&
                push!(compiled_prior_models, model)
        else
            require(!haskey(row, "result"),
                    "runtime-unsupported Reactant-AD cell contains data")
            require(!isempty(get(row, "reason", "")),
                    "runtime-unsupported Reactant-AD cell lacks a diagnostic")
        end
    end
    require(compiled_prior_models == Set(MNIST_MODELS),
            "both public models must retain unbound prior Reactant-AD support")
    errors
end

function validate_mnist_reactant_ad_receipt(
    path::AbstractString;
    ad_path::Union{Nothing,AbstractString} = nothing,
)
    schema = get(TOML.parsefile(path), "schema", "")
    if schema == "mnist-reactant-ad-v1"
        companion = isnothing(ad_path) ?
            joinpath(dirname(path), "mnist-logistic-ad-v1.toml") : String(ad_path)
        return MNISTReactantADV1Validation.validate_mnist_reactant_ad_receipt(
            path; ad_path = companion)
    elseif schema == "mnist-reactant-ad-v2"
        companion = isnothing(ad_path) ?
            joinpath(dirname(path), "mnist-logistic-ad-v2.toml") : String(ad_path)
        return _validate_mnist_reactant_ad_v2_receipt(
            path; ad_path = companion)
    end
    ["schema must be mnist-reactant-ad-v1 or mnist-reactant-ad-v2"]
end

function main(path)
    errors = validate_mnist_reactant_ad_receipt(path)
    if isempty(errors)
        println("VALIDATE OK — mnist-reactant-ad-v2: " *
                "two-model unbound/bound Reactant-AD matrix accepted")
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
