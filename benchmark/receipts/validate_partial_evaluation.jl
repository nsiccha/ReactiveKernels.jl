#!/usr/bin/env julia

import Statistics
import TOML

const EXPECTED_PARTIAL_EVALUATION_OUTCOMES = ("density", "likelihood", "prior")
const EXPECTED_PARTIAL_EVALUATION_BOUND_PORTS = ("X", "y", "num_classes")
# Outcomes whose data-only closure is non-trivial: the receipt must show the
# bound preparation strictly reducing per-call allocation by at least the
# hoisted reference-logits row (8 bytes per observation). Timing is retained
# per raw round but deliberately not gated — shared-host noise must not turn
# an allocation guarantee into a flaky wall-clock assertion.
const EXPECTED_PARTIAL_EVALUATION_HOISTED_OUTCOMES = ("density", "likelihood")

_partial_evaluation_median(values) = Statistics.median(Float64.(values))

function validate_partial_evaluation_receipt(path::AbstractString)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "partial-evaluation-mnist-v1",
            "schema must be partial-evaluation-mnist-v1")
    for section in ("pins", "environment", "protocol", "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a clean ReactiveKernels tree")
    for key in (
        "reactivekernels_sha", "reactivekernels_version",
        "reactivekernelspplexamples_version",
        "reactivekernelsdistributionkernels_version",
        "benchmarktools_version", "mldatasets_version", "julia_version",
    )
        require(haskey(pins, key) && !isempty(string(pins[key])), "pins.$key missing")
    end
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")),
            "ReactiveKernels receipt pin must be a full commit SHA")

    protocol = receipt["protocol"]
    require(Tuple(get(protocol, "outcomes", String[])) ==
            EXPECTED_PARTIAL_EVALUATION_OUTCOMES, "outcome matrix mismatch")
    require(Tuple(get(protocol, "bound_ports", String[])) ==
            EXPECTED_PARTIAL_EVALUATION_BOUND_PORTS, "bound-port set mismatch")
    observations = Int(get(protocol, "num_observations", 0))
    require(observations > 0, "observation count missing")
    require(Int(get(protocol, "num_features", 0)) == 784,
            "receipt must use full-resolution MNIST features")
    require(Int(get(protocol, "num_classes", 0)) == 10,
            "receipt must cover ten classes")
    require(Int(get(protocol, "rounds", 0)) >= 10,
            "published receipt must contain at least ten raw rounds")
    require(Int(get(protocol, "samples_per_round", 0)) >= 1,
            "per-round sample budget missing")
    require(Float64(get(protocol, "seconds_per_round", 0.0)) > 0,
            "per-round timing budget must be positive")
    for key in ("setup_in_timed_region", "preparation_in_timed_region")
        require(get(protocol, key, true) == false, "$key must be false")
    end
    total = Int(get(protocol, "plan_recipes_total", 0))
    residual = Int(get(protocol, "residual_recipes", -1))
    hoisted = Int(get(protocol, "hoisted_constants", 0))
    require(total > 0, "plan recipe census missing")
    require(0 <= residual < total,
            "the pass must retire at least one per-call recipe")
    require(hoisted >= 1, "the pass must hoist at least one constant")
    require(get(protocol, "parity_reference", "") ==
            "landed docs model (build_mnist_logistic_graph)",
            "parity must anchor to the landed docs model")
    require(Float64(get(protocol, "parity_rtol", 1.0)) <= 1e-9 &&
            Float64(get(protocol, "parity_atol", 1.0)) <= 1e-9,
            "parity tolerances drifted")

    function checked_measurement(result, label)
        require(length(get(result, "times_ns", Float64[])) >= 10,
                "$label needs ten timing rounds")
        haskey(result, "times_ns") && !isempty(result["times_ns"]) || return
        require(all(>(0), result["times_ns"]), "$label has non-positive timing")
        require(haskey(result, "min_ns") &&
                isapprox(Float64(result["min_ns"]),
                         minimum(Float64.(result["times_ns"])); rtol = 1e-12),
                "$label minimum mismatch")
        require(haskey(result, "median_ns") &&
                isapprox(Float64(result["median_ns"]),
                         _partial_evaluation_median(result["times_ns"]);
                         rtol = 1e-12),
                "$label median mismatch")
        require(haskey(result, "median_bytes") && haskey(result, "median_allocs"),
                "$label allocation summary missing")
    end

    rows = receipt["measurements"]
    require(length(rows) == length(EXPECTED_PARTIAL_EVALUATION_OUTCOMES),
            "matrix must contain exactly one row per outcome")
    for outcome in EXPECTED_PARTIAL_EVALUATION_OUTCOMES
        matches = filter(row -> row["outcome"] == outcome, rows)
        require(length(matches) == 1, "expected one row for $outcome")
        length(matches) == 1 || continue
        row = only(matches)
        require(haskey(row, "reference_value"), "$outcome parity value missing")
        for key in ("rk_unbound", "rk_bound")
            require(haskey(row, key), "$outcome $key measurement missing")
            haskey(row, key) && checked_measurement(row[key], "$outcome $key")
        end
        haskey(row, "rk_unbound") && haskey(row, "rk_bound") || continue
        unbound_bytes = Int(get(row["rk_unbound"], "median_bytes", 0))
        bound_bytes = Int(get(row["rk_bound"], "median_bytes", typemax(Int)))
        unbound_allocs = Int(get(row["rk_unbound"], "median_allocs", 0))
        bound_allocs = Int(get(row["rk_bound"], "median_allocs", typemax(Int)))
        require(bound_bytes <= unbound_bytes,
                "$outcome bound preparation must not allocate more per call")
        require(bound_allocs <= unbound_allocs,
                "$outcome bound preparation must not add per-call allocations")
        if outcome in EXPECTED_PARTIAL_EVALUATION_HOISTED_OUTCOMES
            # The hoisted reference-logits row alone is one Float64 per
            # observation each call.
            require(unbound_bytes - bound_bytes >= 8 * observations,
                    "$outcome must save at least the hoisted reference row " *
                    "(8 bytes per observation) per call")
        end
    end
    errors
end

function main(path)
    errors = validate_partial_evaluation_receipt(path)
    if isempty(errors)
        println("VALIDATE OK — partial-evaluation-mnist-v1: " *
                "bound preparation allocation guarantee accepted")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_partial_evaluation.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
