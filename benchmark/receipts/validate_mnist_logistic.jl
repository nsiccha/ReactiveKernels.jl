#!/usr/bin/env julia

import Statistics
import TOML

include(joinpath(@__DIR__, "validate_mnist_logistic_v1.jl"))

isdefined(@__MODULE__, :MNISTLogisticMatrixSpec) ||
    include(joinpath(dirname(@__DIR__), "mnist_logistic_matrix_spec.jl"))
using .MNISTLogisticMatrixSpec

const EXPECTED_PRIMAL_CONFIGURATIONS = Tuple(
    configuration for configuration in MNIST_RK_CONFIGURATIONS
    if configuration.differentiation == "primal" &&
       configuration.compiler == "native"
)

function _validate_mnist_logistic_v3_receipt(path::AbstractString)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "mnist-logistic-primal-v3",
            "schema must be mnist-logistic-primal-v3")
    for section in ("pins", "environment", "protocol", "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    for key in (
        "reactivekernels_sha", "reactivekernels_version",
        "reactivekernelspplexamples_version",
        "reactivekernelsdistributionkernels_version", "turing_version",
        "dynamicppl_version", "distributions_version", "nnlib_version",
        "mldatasets_version", "benchmarktools_version",
        "mutatingfunctions_version", "mutatingfunctions_rev", "julia_version",
    )
        require(haskey(pins, key) && !isempty(string(pins[key])),
                "pins.$key missing")
    end
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")),
            "ReactiveKernels receipt pin must be a full commit SHA")
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a clean detached ReactiveKernels tree")

    protocol = receipt["protocol"]
    require(Tuple(get(protocol, "input_boundaries", String[])) == MNIST_BOUNDARIES,
            "input-boundary matrix mismatch")
    require(Tuple(get(protocol, "outcomes", String[])) == MNIST_OUTCOMES,
            "outcome matrix mismatch")
    require(Tuple(get(protocol, "models", String[])) == MNIST_MODELS,
            "model inventory mismatch")
    require(Tuple(get(protocol, "rk_configurations", String[])) ==
            Tuple(configuration.id for configuration in EXPECTED_PRIMAL_CONFIGURATIONS),
            "native primal configuration inventory mismatch")
    require(get(protocol, "matrix_layout", "") ==
            "long-form provider/model/configuration/boundary/outcome rows",
            "receipt must use the long-form capability matrix")
    require(Tuple(get(protocol, "bound_ports", String[])) ==
            ("X", "y", "num_classes"), "bound-port inventory mismatch")
    require(Int(get(protocol, "rounds", 0)) >= 10,
            "published receipt must use at least 10 rounds")
    require(Int(get(protocol, "num_observations", 0)) > 0,
            "num_observations must be positive")
    require(Int(get(protocol, "num_features", 0)) == 784,
            "MNIST is full-resolution: num_features must be 784")
    require(Int(get(protocol, "num_classes", 0)) == 10,
            "MNIST has ten classes")
    require(get(protocol, "setup_in_timed_region", true) == false,
            "setup must be outside timing")
    require(get(protocol, "preparation_in_timed_region", true) == false,
            "kernel/LDF preparation must be outside timing")
    require(get(protocol, "turing_transform_strategy", "") == "fixed transforms",
            "Turing sampler-space rows must use fixed transforms")
    require(get(protocol, "turing_pointwise_supported", true) == false,
            "the @addlogprob! likelihood has no public Turing pointwise view")
    require(get(protocol, "gradients_included", true) == false,
            "MNIST logistic primal receipt must not contain gradients")

    function checked_result(row, label)
        haskey(row, "result") || (require(false, "$label lacks a result"); return)
        result = row["result"]
        times = Float64.(get(result, "times_ns", Float64[]))
        bytes = Int.(get(result, "bytes", Int[]))
        allocs = Int.(get(result, "allocs", Int[]))
        require(length(times) >= 10, "$label needs ten raw timing rounds")
        require(length(bytes) == length(times) && length(allocs) == length(times),
                "$label raw measurement vector lengths differ")
        isempty(times) && return
        require(all(>(0), times), "$label has non-positive timing")
        require(all(>=(0), bytes), "$label has negative allocated bytes")
        require(all(>=(0), allocs), "$label has negative allocation counts")
        require(isapprox(Float64(get(result, "min_ns", NaN)), minimum(times);
                         rtol = 1e-12), "$label minimum mismatch")
        require(isapprox(Float64(get(result, "median_ns", NaN)),
                         Statistics.median(times); rtol = 1e-12),
                "$label timing median mismatch")
        require(Int(get(result, "median_bytes", -1)) == Int(Statistics.median(bytes)),
                "$label byte median mismatch")
        require(Int(get(result, "median_allocs", -1)) == Int(Statistics.median(allocs)),
                "$label allocation median mismatch")
    end

    measurements = receipt["measurements"]
    expected_count =
        length(MNIST_MODELS) * length(EXPECTED_PRIMAL_CONFIGURATIONS) *
            length(MNIST_BOUNDARIES) * length(MNIST_OUTCOMES) +
        length(MNIST_BOUNDARIES) * length(MNIST_OUTCOMES) +
        2 * length(MNIST_BOUNDARIES) * length(MNIST_OUTCOMES)
    require(length(measurements) == expected_count,
            "long-form primal matrix must contain $expected_count rows")
    keys = [
        (row["provider"], row["model"], row["configuration"],
         row["boundary"], row["outcome"]) for row in measurements
    ]
    require(length(Set(keys)) == length(keys), "matrix contains duplicate cells")

    function only_cell(provider, model, configuration, boundary, outcome)
        matches = filter(measurements) do row
            row["provider"] == provider && row["model"] == model &&
                row["configuration"] == configuration &&
                row["boundary"] == boundary && row["outcome"] == outcome
        end
        require(length(matches) == 1,
                "expected one $provider / $model / $configuration / " *
                "$boundary / $outcome row")
        length(matches) == 1 ? only(matches) : nothing
    end

    for model in MNIST_MODELS, configuration in EXPECTED_PRIMAL_CONFIGURATIONS,
        boundary in MNIST_BOUNDARIES, outcome in MNIST_OUTCOMES
        row = only_cell("rk", model, configuration.id, boundary, outcome)
        row === nothing && continue
        expected_state, _ = matrix_support(configuration, boundary, outcome)
        state = get(row, "state", "")
        require(state == expected_state,
                "RK support drift for $model / $(configuration.id) / $boundary / $outcome")
        if state == "supported"
            checked_result(row,
                "RK $model / $(configuration.id) / $boundary / $outcome")
        else
            require(!haskey(row, "result"), "non-measured RK cell contains a result")
            require(!isempty(get(row, "reason", "")),
                    "non-measured RK cell lacks a reason")
        end
    end

    for boundary in MNIST_BOUNDARIES, outcome in MNIST_OUTCOMES
        manual = only_cell(
            "manual_julia", "implicit_reference", "manual_primal",
            boundary, outcome)
        manual === nothing || checked_result(
            manual, "manual / $boundary / $outcome")
        for model in MNIST_MODELS
            configuration = "turing_$(model)_primal"
            row = only_cell("turing", model, configuration, boundary, outcome)
            row === nothing && continue
            expected_state = outcome == "pointwise" ? "unsupported" : "supported"
            require(get(row, "state", "") == expected_state,
                    "Turing support drift for $model / $boundary / $outcome")
            if expected_state == "supported"
                checked_result(row, "Turing $model / $boundary / $outcome")
            else
                require(!haskey(row, "result"),
                        "unsupported Turing pointwise cell contains a result")
                require(!isempty(get(row, "reason", "")),
                        "unsupported Turing pointwise cell lacks a reason")
            end
        end
    end

    errors
end

function validate_mnist_logistic_receipt(path::AbstractString)
    schema = get(TOML.parsefile(path), "schema", "")
    schema == "mnist-logistic-v1" &&
        return MNISTLogisticV1Validation.validate_mnist_logistic_receipt(path)
    schema == "mnist-logistic-primal-v3" &&
        return _validate_mnist_logistic_v3_receipt(path)
    ["schema must be mnist-logistic-v1 or mnist-logistic-primal-v3"]
end

function main(path)
    errors = validate_mnist_logistic_receipt(path)
    if isempty(errors)
        println("VALIDATE OK — MNIST logistic primal receipt accepted")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_mnist_logistic.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
