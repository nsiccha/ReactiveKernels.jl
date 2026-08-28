#!/usr/bin/env julia

import Statistics
import TOML

const EXPECTED_STRUCTURED_PM_SHA =
    "7cf3a6e112aaae2097b8d401b256d1bce635e03e"
const EXPECTED_STRUCTURED_SIZES = (4, 16, 64, 128)
const EXPECTED_STRUCTURED_FAMILIES = ("mvnormal_cholesky", "stationary_ar1")

_structured_median(values) = Statistics.median(Float64.(values))

function validate_structured_distribution_receipt(path::AbstractString)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "structured-distribution-logdensity-v1",
            "schema must be structured-distribution-logdensity-v1")
    for section in ("pins", "environment", "protocol", "support",
                    "support_errors", "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a clean ReactiveKernels tree")
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")),
            "ReactiveKernels receipt pin must be a full commit SHA")
    require(get(pins, "probability_measures_sha", "") == EXPECTED_STRUCTURED_PM_SHA,
            "ProbabilityMeasures commit does not match the reviewed pin")
    for key in ("probability_measures_version", "distributions_version",
                "reactant_version", "reactant_jll_version", "julia_version")
        require(haskey(pins, key) && !isempty(string(pins[key])), "pins.$key missing")
    end

    protocol = receipt["protocol"]
    require(get(protocol, "reactant_sync", false) == true,
            "Reactant timings must use sync=true")
    require(get(protocol, "reactant_transfers_included", true) == false,
            "Reactant host/device transfers must be excluded")
    require(get(protocol, "reactant_compile_time_in_timed_region", true) == false,
            "Reactant compilation must be excluded from execution timings")
    require(get(protocol, "construction_and_factorization_timed", true) == false,
            "distribution construction/factorization exclusion must be explicit")
    require(Int(get(protocol, "rounds", 0)) >= 5,
            "published receipt must contain at least five raw rounds")

    support = receipt["support"]
    support_errors = receipt["support_errors"]
    for family in EXPECTED_STRUCTURED_FAMILIES
        require(haskey(support, family), "support missing $family") || continue
        family_support = support[family]
        require(get(family_support, "rk_reactant", false) == true,
                "$family RK Reactant compatibility did not pass")
        for library in ("probability_measures_reactant", "distributions_reactant")
            if !get(family_support, library, false)
                diagnostic = get(get(support_errors, family, Dict()), library, "")
                require(!isempty(diagnostic),
                        "$family $library unsupported without a diagnostic")
            end
        end
    end

    rows = receipt["measurements"]
    require(length(rows) == length(EXPECTED_STRUCTURED_SIZES) *
                            length(EXPECTED_STRUCTURED_FAMILIES),
            "unexpected structured measurement row count")
    required = ("rk_native", "distributions_native",
                "probability_measures_native", "rk_reactant")
    for family in EXPECTED_STRUCTURED_FAMILIES
        family_rows = [row for row in rows if row["family"] == family]
        observed_sizes = Tuple(Int(row["n"]) for row in family_rows)
        require(observed_sizes == EXPECTED_STRUCTURED_SIZES,
                "$family sizes must equal $(EXPECTED_STRUCTURED_SIZES)")
        for row in family_rows
            n = Int(row["n"])
            require(Float64(row["max_abs_error"]) <= 1e-9,
                    "$family n=$n exceeds value-parity tolerance")
            names = String[required...]
            family_support = support[family]
            get(family_support, "probability_measures_reactant", false) &&
                push!(names, "probability_measures_reactant")
            get(family_support, "distributions_reactant", false) &&
                push!(names, "distributions_reactant")
            for name in names
                require(haskey(row, name), "$family n=$n missing $name") || continue
                measurement = row[name]
                for raw_key in ("times_ns", "bytes", "allocs")
                    require(haskey(measurement, raw_key) && !isempty(measurement[raw_key]),
                            "$family n=$n $name.$raw_key missing")
                end
                all(haskey(measurement, key) for key in
                    ("times_ns", "bytes", "allocs")) || continue
                require(all(>(0), measurement["times_ns"]),
                        "$family n=$n $name has non-positive timing")
                require(all(>=(0), measurement["bytes"]),
                        "$family n=$n $name has negative bytes")
                require(all(>=(0), measurement["allocs"]),
                        "$family n=$n $name has negative allocations")
                require(isapprox(Float64(measurement["median_ns"]),
                                 _structured_median(measurement["times_ns"]);
                                 rtol = 1e-12),
                        "$family n=$n $name median_ns mismatch")
                require(Int(measurement["median_bytes"]) ==
                        round(Int, _structured_median(measurement["bytes"])),
                        "$family n=$n $name median_bytes mismatch")
                require(Int(measurement["median_allocs"]) ==
                        round(Int, _structured_median(measurement["allocs"])),
                        "$family n=$n $name median_allocs mismatch")
            end
        end
    end
    errors
end

function main(path)
    errors = validate_structured_distribution_receipt(path)
    isempty(errors) && (println("VALIDATE OK — structured-distribution-logdensity-v1");
                       return 0)
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_structured_distributions.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
