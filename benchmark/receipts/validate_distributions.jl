#!/usr/bin/env julia

import Statistics
import TOML

const EXPECTED_PROBABILITY_MEASURES_SHA =
    "7cf3a6e112aaae2097b8d401b256d1bce635e03e"
const EXPECTED_SIZES = (1, 1_000, 10_000, 30_000, 100_000, 1_000_000)

_median(values) = Statistics.median(Float64.(values))

function validate_distribution_receipt(path::AbstractString)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "distribution-logdensity-v1",
            "schema must be distribution-logdensity-v1")
    for section in (
        "pins", "environment", "protocol", "support", "measurements",
        "reactant_amortization",
    )
        require(haskey(receipt, section), "missing $section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    for key in (
        "reactivekernels_sha", "probability_measures_sha",
        "probability_measures_version", "distributions_version",
        "reactant_version", "reactant_jll_version", "julia_version",
    )
        require(haskey(pins, key) && !isempty(string(pins[key])), "pins.$key missing")
    end
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a clean ReactiveKernels tree")
    require(get(pins, "probability_measures_sha", "") ==
            EXPECTED_PROBABILITY_MEASURES_SHA,
            "ProbabilityMeasures commit does not match the reviewed pin")
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")),
            "ReactiveKernels receipt pin must be a full commit SHA")

    protocol = receipt["protocol"]
    require(get(protocol, "reactant_sync", false) == true,
            "Reactant timings must use sync=true")
    require(get(protocol, "reactant_transfers_included", true) == false,
            "Reactant device-transfer policy must be explicit and false")
    require(get(protocol, "reactant_parameters_traced", false) == true,
            "Reactant parameters must be traced runtime values")
    require(get(protocol, "reactant_compile_time_in_timed_region", true) == false,
            "Reactant execution timings must exclude compilation")
    require(get(protocol, "reactant_compile_times_include_first_service_startup", false) == true,
            "receipt must disclose first-service-startup compile timing")
    require(Int(get(protocol, "rounds", 0)) >= 5,
            "published receipt must contain at least five raw rounds")
    require(Tuple(Int.(get(protocol, "reactant_replica_counts", Int[]))) ==
            (1, 16, 256), "published receipt must use replica counts (1, 16, 256)")

    support = receipt["support"]
    require(get(support, "rk_reactant", false) == true,
            "RK Reactant compatibility did not pass")
    require(get(support, "rk_authored_reactant", false) == true,
            "authored RK Reactant compatibility did not pass")
    require(get(support, "probability_measures_reactant", false) == true,
            "ProbabilityMeasures Reactant compatibility did not pass")
    if !get(support, "distributions_reactant", false)
        require(!isempty(get(support, "distributions_reactant_error", "")),
                "unsupported Distributions Reactant path needs a diagnostic")
    end

    required_measurements = (
        "rk_native", "rk_authored_native", "rk_direct_native",
        "distributions_native", "probability_measures_native",
        "distributions_loop", "probability_measures_loop", "hand_hoisted",
        "rk_reactant", "rk_authored_reactant", "probability_measures_reactant",
    )
    observed_sizes = Tuple(Int(row["n"]) for row in receipt["measurements"])
    require(observed_sizes == EXPECTED_SIZES,
            "measurement sizes must equal $(EXPECTED_SIZES)")
    previous_n = 0
    for row in receipt["measurements"]
        n = Int(row["n"])
        require(n > previous_n, "measurement sizes must be strictly increasing")
        previous_n = n
        require(Float64(row["max_relative_error"]) <= 1e-10,
                "N=$n exceeds relative value-parity tolerance")

        names = support["distributions_reactant"] ?
            (required_measurements..., "distributions_reactant") :
            required_measurements
        for name in names
            require(haskey(row, name), "N=$n missing $name") || continue
            measurement = row[name]
            for raw_key in ("times_ns", "bytes", "allocs")
                require(haskey(measurement, raw_key) && !isempty(measurement[raw_key]),
                        "N=$n $name.$raw_key missing")
            end
            all(haskey(measurement, key) for key in ("times_ns", "bytes", "allocs")) ||
                continue
            require(all(>(0), measurement["times_ns"]),
                    "N=$n $name has non-positive timing")
            require(all(>=(0), measurement["bytes"]),
                    "N=$n $name has negative allocation bytes")
            require(all(>=(0), measurement["allocs"]),
                    "N=$n $name has negative allocation count")
            require(isapprox(
                Float64(measurement["median_ns"]),
                _median(measurement["times_ns"]);
                rtol = 1e-12,
            ), "N=$n $name median_ns is not re-derived from raw rounds")
            require(Int(measurement["median_bytes"]) ==
                    round(Int, _median(measurement["bytes"])),
                    "N=$n $name median_bytes mismatch")
            require(Int(measurement["median_allocs"]) ==
                    round(Int, _median(measurement["allocs"])),
                    "N=$n $name median_allocs mismatch")
            if name in ("rk_native", "rk_authored_native", "rk_direct_native")
                require(measurement["median_bytes"] == 0,
                        "N=$n $name reduction must remain zero-allocation")
                require(measurement["median_allocs"] == 0,
                        "N=$n $name reduction must remain zero-allocation")
            end
        end
        if haskey(row, "rk_native") && haskey(row, "rk_direct_native")
            shared_ratio = Float64(row["rk_native"]["median_ns"]) /
                           Float64(row["rk_direct_native"]["median_ns"])
            require(shared_ratio <= 1.10,
                    "N=$n shared RK object exceeds the one-off control by more than 10%")
        end
        if n > 1 && haskey(row, "rk_native") &&
                haskey(row, "rk_authored_native")
            authored_ratio = Float64(row["rk_authored_native"]["median_ns"]) /
                             Float64(row["rk_native"]["median_ns"])
            require(authored_ratio <= 1.10,
                    "N=$n authored return exceeds legacy plate by more than 10%")
        end
    end

    amortization = receipt["reactant_amortization"]
    require(Tuple(Int(row["replicas"]) for row in amortization) == (1, 16, 256),
            "amortization rows must use replica counts (1, 16, 256)")
    wrapper_bytes = Int[]
    wrapper_allocs = Int[]
    for row in amortization
        replicas = Int(row["replicas"])
        require(Int(row["n"]) == 1,
                "replicas=$replicas amortization row must represent one observation")
        require(Float64(row["max_abs_error"]) <= 1e-10,
                "replicas=$replicas exceeds value-parity tolerance")
        require(haskey(row, "rk_reactant_replicated"),
                "replicas=$replicas missing replicated measurement") || continue
        measurement = row["rk_reactant_replicated"]
        for raw_key in ("times_ns", "bytes", "allocs")
            require(haskey(measurement, raw_key) && !isempty(measurement[raw_key]),
                    "replicas=$replicas $raw_key missing")
        end
        all(haskey(measurement, key) for key in ("times_ns", "bytes", "allocs")) ||
            continue
        require(isapprox(
            Float64(measurement["median_ns"]), _median(measurement["times_ns"]);
            rtol = 1e-12,
        ), "replicas=$replicas median_ns is not re-derived from raw rounds")
        require(isapprox(
            Float64(row["median_ns_per_evaluation"]),
            Float64(measurement["median_ns"]) / replicas;
            rtol = 1e-12,
        ), "replicas=$replicas per-evaluation normalization mismatch")
        require(Int(measurement["median_bytes"]) ==
                round(Int, _median(measurement["bytes"])),
                "replicas=$replicas median_bytes mismatch")
        require(Int(measurement["median_allocs"]) ==
                round(Int, _median(measurement["allocs"])),
                "replicas=$replicas median_allocs mismatch")
        push!(wrapper_bytes, Int(measurement["median_bytes"]))
        push!(wrapper_allocs, Int(measurement["median_allocs"]))
    end
    require(length(unique(wrapper_bytes)) <= 1,
            "host wrapper bytes must be independent of replica count")
    require(length(unique(wrapper_allocs)) <= 1,
            "host wrapper allocations must be independent of replica count")

    errors
end

function main(path)
    errors = validate_distribution_receipt(path)
    if isempty(errors)
        receipt = TOML.parsefile(path)
        sizes = join((row["n"] for row in receipt["measurements"]), ", ")
        println("VALIDATE OK — distribution-logdensity-v1: sizes [$sizes], " *
                "RK+ProbabilityMeasures Reactant paths and replica amortization accepted")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || (println("usage: validate_distributions.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
