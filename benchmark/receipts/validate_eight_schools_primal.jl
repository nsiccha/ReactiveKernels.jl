#!/usr/bin/env julia

import TOML

const EXPECTED_BOUNDARIES = (
    "packed_unconstrained", "constrained_parameters", "minimal_likelihood")
const EXPECTED_OUTCOMES = ("joint", "prior", "likelihood", "pointwise")
const EXPECTED_BACKENDS = ("rk_native", "manual_julia", "turing_native")

function _expected_support(boundary, outcome, backend)
    if boundary == "minimal_likelihood"
        return outcome in ("likelihood", "pointwise") &&
               backend in ("rk_native", "manual_julia")
    end
    true
end

function validate_eight_schools_primal_receipt(path::AbstractString)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "eight-schools-primal-v1",
            "schema must be eight-schools-primal-v1")
    for section in ("pins", "environment", "protocol", "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    for key in (
        "reactivekernels_sha", "reactivekernels_version",
        "reactivekernelspplexamples_version",
        "reactivekernelsdistributionkernels_version", "turing_version",
        "dynamicppl_version", "distributions_version", "benchmarktools_version",
        "julia_version",
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
    require(get(protocol, "setup_in_timed_region", true) == false,
            "setup must be outside timing")
    require(get(protocol, "preparation_in_timed_region", true) == false,
            "kernel/LDF preparation must be outside timing")
    require(get(protocol, "turing_transform_strategy", "") == "fixed transforms",
            "Turing sampler-space rows must use fixed transforms")
    require(get(protocol, "turing_pointwise_api", "") ==
            "DynamicPPL.pointwise_loglikelihoods",
            "Turing pointwise row must use its public pointwise API")
    require(get(protocol, "gradients_included", true) == false,
            "Eight Schools primal receipt must not contain gradients")
    require(get(protocol, "generated_predictions_included", true) == false,
            "predictive generated quantities must not enter this comparison")
    require(get(protocol, "unsupported_cells_omitted", false) == true,
            "unsupported cells must be omitted rather than synthesized")

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
            require(Int(result["median_bytes"]) >= 0,
                    "$boundary / $outcome / $backend bytes must be nonnegative")
            require(Int(result["median_allocs"]) >= 0,
                    "$boundary / $outcome / $backend allocs must be nonnegative")
        end
    end

    errors
end

function main(path)
    errors = validate_eight_schools_primal_receipt(path)
    if isempty(errors)
        println("VALIDATE OK — eight-schools-primal-v1: 3×4 capability/timing matrix accepted")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_eight_schools_primal.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
