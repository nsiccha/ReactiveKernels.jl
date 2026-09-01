#!/usr/bin/env julia

import SHA
import Statistics
import TOML

isdefined(@__MODULE__, :MNISTLogisticMatrixSpec) ||
    include(joinpath(dirname(@__DIR__), "mnist_logistic_matrix_spec.jl"))
using .MNISTLogisticMatrixSpec

const EXPECTED_MNIST_AD_CONFIGURATIONS = Tuple(
    configuration for configuration in MNIST_RK_CONFIGURATIONS
    if configuration.differentiation == "value_and_gradient" &&
       configuration.compiler == "native"
)

_mnist_ad_median(values) = Statistics.median(Float64.(values))
_mnist_ad_text_sha256(path) = bytes2hex(SHA.sha256(
    replace(read(path, String), "\r\n" => "\n", "\r" => "\n")))

function validate_mnist_logistic_ad_receipt(path::AbstractString;
        root::AbstractString = normpath(joinpath(dirname(path), "..", "..")))
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "mnist-logistic-ad-v2",
            "schema must be mnist-logistic-ad-v2")
    for section in ("pins", "environment", "setup", "protocol", "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a tracked-clean candidate")
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")),
            "candidate pin must be a full commit SHA")
    for key in ("model_source", "primal_comparator_source")
        require(haskey(pins, key), "pins.$key missing")
    end

    primal_path = joinpath(root, get(pins, "primal_receipt_path", ""))
    require(isfile(primal_path), "matched primal receipt is missing")
    if isfile(primal_path)
        primal = TOML.parsefile(primal_path)
        require(get(primal, "schema", "") == "mnist-logistic-primal-v3",
                "matched receipt must be mnist-logistic-primal-v3")
        require(get(pins, "primal_receipt_sha256", "") ==
                _mnist_ad_text_sha256(primal_path),
                "matched primal receipt digest mismatch")
        haskey(primal, "pins") && require(
            get(pins, "primal_receipt_reactivekernels_sha", "") ==
            get(primal["pins"], "reactivekernels_sha", ""),
            "matched primal receipt code pin mismatch")
    end

    for (key, expected_path) in (
            "model_source" =>
                "packages/ReactiveKernelsPPLExamples/src/mnist_logistic.jl",
            "primal_comparator_source" =>
                "benchmark/mnist_logistic_comparison_body.jl",
        )
        haskey(pins, key) || continue
        pin = pins[key]
        require(get(pin, "path", "") == expected_path, "$key path mismatch")
        require(occursin(r"^[0-9a-f]{40}$", get(pin, "commit", "")),
                "$key published commit missing")
        require(occursin(r"^[0-9a-f]{40}$", get(pin, "git_blob", "")),
                "$key published blob missing")
        require(occursin(r"^[0-9a-f]{64}$", get(pin, "text_sha256", "")),
                "$key published text digest missing")
        current = get(pin, "current", Dict{String,Any}())
        absolute_path = joinpath(root, expected_path)
        require(get(current, "path", "") == expected_path,
                "$key current path mismatch")
        require(isfile(absolute_path), "$key current source is missing")
        isfile(absolute_path) && require(
            get(current, "text_sha256", "") ==
            _mnist_ad_text_sha256(absolute_path),
            "$key current text digest mismatch")
    end

    protocol = receipt["protocol"]
    require(Tuple(get(protocol, "input_boundaries", String[])) == MNIST_BOUNDARIES,
            "input-boundary matrix mismatch")
    require(Tuple(get(protocol, "outcomes", String[])) == MNIST_OUTCOMES,
            "outcome matrix mismatch")
    require(Tuple(get(protocol, "models", String[])) == MNIST_MODELS,
            "model inventory mismatch")
    require(Tuple(get(protocol, "rk_configurations", String[])) ==
            Tuple(configuration.id for configuration in EXPECTED_MNIST_AD_CONFIGURATIONS),
            "native AD configuration inventory mismatch")
    require(get(protocol, "matrix_layout", "") ==
            "long-form provider/model/configuration/boundary/outcome rows",
            "AD receipt must use the long-form capability matrix")
    require(Tuple(get(protocol, "bound_ports", String[])) ==
            ("X", "y", "num_classes"), "bound-port inventory mismatch")
    require(get(protocol, "source_reused", false) == true,
            "benchmark must reuse the published model source")
    require(get(protocol, "matrix_source", "") ==
            "benchmark/receipts/mnist-logistic-primal-v3.toml",
            "benchmark must name the matched primal matrix")
    require(Int(get(protocol, "num_observations", 0)) == 60000,
            "published receipt must use the full 60000-image train split")
    require(Int(get(protocol, "num_features", 0)) == 784,
            "MNIST input must retain all 784 pixels")
    require(Int(get(protocol, "num_classes", 0)) == 10,
            "MNIST must retain ten classes")
    require(Int(get(protocol, "active_parameter_count", 0)) == 7065,
            "packed active vector must contain 7065 coefficients")
    require(get(protocol, "pointwise_jacobian_or_vjp_invented", true) == false,
            "pointwise must remain unsupported without a matched public contract")
    require(get(protocol, "structured_multi_active_boundary_invented", true) == false,
            "structured two-active-port AD must not be invented")
    for key in ("setup_in_timed_region", "preparation_in_timed_region",
                "first_execution_in_steady_state_region")
        require(get(protocol, key, true) == false, "$key must be false")
    end
    require(Int(get(protocol, "rounds", 0)) >= 10,
            "published receipt must retain at least ten rounds")
    relative_tolerance = Float64(get(protocol, "parity_relative_tolerance", 0.0))
    absolute_tolerance = Float64(get(protocol, "parity_absolute_tolerance", 0.0))
    require(relative_tolerance > 0, "relative parity tolerance must be positive")
    require(absolute_tolerance > 0, "absolute parity tolerance must be positive")

    function checked_result(row, label; caller_owned)
        haskey(row, "result") || (require(false, "$label lacks a result"); return)
        result = row["result"]
        for key in ("preparation_seconds", "preparation_bytes",
                    "first_execution_seconds", "first_execution_bytes")
            require(Float64(get(result, key, -1)) >= 0, "$label $key missing")
        end
        times = Float64.(get(result, "times_ns", Float64[]))
        bytes = Int.(get(result, "bytes", Int[]))
        allocs = Int.(get(result, "allocs", Int[]))
        require(length(times) >= 10, "$label needs ten timing rounds")
        require(length(bytes) == length(times) && length(allocs) == length(times),
                "$label raw measurement vector lengths mismatch")
        isempty(times) && return
        require(all(>(0), times), "$label has non-positive timing")
        require(all(>=(0), bytes) && all(>=(0), allocs),
                "$label has negative allocations")
        require(Float64(get(result, "min_ns", NaN)) == minimum(times),
                "$label minimum mismatch")
        require(isapprox(Float64(get(result, "median_ns", NaN)),
                         _mnist_ad_median(times); rtol = 1e-12),
                "$label timing median mismatch")
        require(Int(get(result, "median_bytes", -1)) ==
                round(Int, _mnist_ad_median(bytes)), "$label byte median mismatch")
        require(Int(get(result, "median_allocs", -1)) ==
                round(Int, _mnist_ad_median(allocs)),
                "$label allocation median mismatch")
        require(Int(get(result, "gradient_length", 0)) == 7065,
                "$label gradient length mismatch")
        absolute_error = Float64(get(result, "gradient_max_abs_error", Inf))
        relative_error = Float64(get(result, "gradient_max_rel_error", Inf))
        require(absolute_error <= absolute_tolerance ||
                relative_error <= relative_tolerance,
                "$label exceeds gradient parity tolerance")
        require(Float64(get(result, "value_abs_error", Inf)) <= 1e-7,
                "$label value parity failed")
        require(get(result, "caller_owned_gradient", !caller_owned) == caller_owned,
                "$label ownership mismatch")
    end

    rows = receipt["measurements"]
    expected_count =
        length(MNIST_MODELS) * length(EXPECTED_MNIST_AD_CONFIGURATIONS) *
            length(MNIST_BOUNDARIES) * length(MNIST_OUTCOMES) +
        length(MNIST_BOUNDARIES) * length(MNIST_OUTCOMES) +
        length(MNIST_MODELS) * length(MNIST_BOUNDARIES) * length(MNIST_OUTCOMES)
    require(length(rows) == expected_count,
            "long-form AD matrix must contain $expected_count rows")
    keys = [(row["provider"], row["model"], row["configuration"],
             row["boundary"], row["outcome"]) for row in rows]
    require(length(Set(keys)) == length(keys), "AD matrix contains duplicate cells")

    function only_cell(provider, model, configuration, boundary, outcome)
        matches = filter(rows) do row
            row["provider"] == provider && row["model"] == model &&
                row["configuration"] == configuration &&
                row["boundary"] == boundary && row["outcome"] == outcome
        end
        require(length(matches) == 1,
                "expected one $provider / $model / $configuration / " *
                "$boundary / $outcome row")
        length(matches) == 1 ? only(matches) : nothing
    end

    for boundary in MNIST_BOUNDARIES, outcome in MNIST_OUTCOMES
        base_state, _ = matrix_support(
            first(EXPECTED_MNIST_AD_CONFIGURATIONS), boundary, outcome)
        for model in MNIST_MODELS,
            configuration in EXPECTED_MNIST_AD_CONFIGURATIONS
            row = only_cell("rk", model, configuration.id, boundary, outcome)
            row === nothing && continue
            expected_state, _ = matrix_support(configuration, boundary, outcome)
            require(get(row, "state", "") == expected_state,
                    "RK AD support drift for $model / $(configuration.id) / " *
                    "$boundary / $outcome")
            if expected_state == "supported"
                require(get(row, "active_port", "") == "unconstrained",
                        "RK AD active port mismatch")
                require(Int(get(row, "analytic_gradient_length", 0)) == 7065,
                        "RK AD oracle length mismatch")
                checked_result(row,
                    "RK $model / $(configuration.id) / $boundary / $outcome";
                    caller_owned = true)
            else
                require(!haskey(row, "result"), "unsupported RK AD cell has data")
                require(!isempty(get(row, "reason", "")),
                        "unsupported RK AD cell lacks a reason")
            end
        end

        manual = only_cell(
            "manual_julia", "implicit_reference", "manual_ad", boundary, outcome)
        manual === nothing || if base_state == "supported"
            require(get(manual, "state", "") == "supported",
                    "manual AD supported cell drifted")
            checked_result(manual, "manual / $boundary / $outcome";
                           caller_owned = true)
        else
            require(get(manual, "state", "") == base_state,
                    "manual AD unsupported state drifted")
            require(!haskey(manual, "result"), "unsupported manual AD cell has data")
        end

        for model in MNIST_MODELS
            row = only_cell(
                "turing", model, "turing_$(model)_ad", boundary, outcome)
            row === nothing && continue
            require(get(row, "state", "") == base_state,
                    "Turing AD support drift for $model / $boundary / $outcome")
            if base_state == "supported"
                checked_result(row, "Turing $model / $boundary / $outcome";
                               caller_owned = false)
            else
                require(!haskey(row, "result"), "unsupported Turing AD cell has data")
                require(!isempty(get(row, "reason", "")),
                        "unsupported Turing AD cell lacks a reason")
            end
        end
    end
    errors
end

function main(path)
    errors = validate_mnist_logistic_ad_receipt(path)
    if isempty(errors)
        println("VALIDATE OK — mnist-logistic-ad-v2: " *
                "two-model native/bound AD matrix accepted")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_mnist_logistic_ad.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
