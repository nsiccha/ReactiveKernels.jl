#!/usr/bin/env julia

module MNISTLogisticV1Validation

import TOML

include(joinpath(@__DIR__, "_validate_mnist_dataset_profile.jl"))

const EXPECTED_BOUNDARIES = ("packed_unconstrained", "structured_parameters")
const EXPECTED_OUTCOMES = ("joint", "prior", "likelihood", "pointwise")
const EXPECTED_BACKENDS = ("rk_native", "manual_julia", "turing_native")

# Turing's public interfaces cover the joint, prior, and summed likelihood, but
# the model's likelihood is a single `@addlogprob!` term, so there is no public
# per-observation Turing pointwise view. That cell is omitted, not synthesized.
function _expected_support(boundary, outcome, backend)
    backend == "turing_native" && outcome == "pointwise" && return false
    true
end

function validate_mnist_logistic_receipt(path::AbstractString)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "mnist-logistic-v1",
            "schema must be mnist-logistic-v1")
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
        "mldatasets_version", "benchmarktools_version", "julia_version",
    )
        require(haskey(pins, key) && !isempty(string(pins[key])),
                "pins.$key missing")
    end
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")),
            "ReactiveKernels receipt pin must be a full commit SHA")
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a clean detached ReactiveKernels tree")

    protocol = receipt["protocol"]
    require(Tuple(protocol["input_boundaries"]) == EXPECTED_BOUNDARIES,
            "input-boundary matrix mismatch")
    require(Tuple(protocol["outcomes"]) == EXPECTED_OUTCOMES,
            "outcome matrix mismatch")
    require(Int(get(protocol, "rounds", 0)) >= 10,
            "published receipt must use at least 10 rounds")
    require(Int(get(protocol, "num_observations", 0)) > 0,
            "num_observations must be positive")
    _validate_mnist_dataset_profile!(require, protocol, protocol)
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

    measurements = receipt["measurements"]
    require(length(measurements) ==
            length(EXPECTED_BOUNDARIES) * length(EXPECTED_OUTCOMES),
            "matrix must contain exactly one row per boundary/outcome pair")
    for boundary in EXPECTED_BOUNDARIES, outcome in EXPECTED_OUTCOMES
        matching = filter(measurements) do row
            row["boundary"] == boundary && row["outcome"] == outcome
        end
        require(length(matching) == 1,
                "expected one row for $boundary / $outcome")
        length(matching) == 1 || continue
        row = only(matching)
        for backend in EXPECTED_BACKENDS
            supported = _expected_support(boundary, outcome, backend)
            require(haskey(row, backend) == supported,
                    "$boundary / $outcome / $backend support mismatch")
            supported || continue
            result = row[backend]
            require(length(result["times_ns"]) >= 10,
                    "$boundary / $outcome / $backend needs ten raw timing rounds")
            require(Float64(result["median_ns"]) > 0,
                    "$boundary / $outcome / $backend median_ns must be positive")
            require(haskey(result, "min_ns") && Float64(result["min_ns"]) > 0,
                    "$boundary / $outcome / $backend min_ns must be positive")
            require(Int(result["median_bytes"]) >= 0,
                    "$boundary / $outcome / $backend bytes must be nonnegative")
            require(Int(result["median_allocs"]) >= 0,
                    "$boundary / $outcome / $backend allocs must be nonnegative")
        end
    end

    errors
end

function main(path)
    errors = validate_mnist_logistic_receipt(path)
    if isempty(errors)
        println("VALIDATE OK — mnist-logistic-v1: 2×4 capability/timing matrix accepted")
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

end # module MNISTLogisticV1Validation
