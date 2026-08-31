#!/usr/bin/env julia

import Statistics
import TOML

const EXPECTED_SCALAR_GALLERY_PROBABILITY_MEASURES_SHA =
    "7cf3a6e112aaae2097b8d401b256d1bce635e03e"
const EXPECTED_SCALAR_GALLERY_FAMILIES =
    (
        "cauchy_location_scale",
        "laplace_location_scale",
        "bernoulli_logit",
        "lognormal_logscale",
        "exponential_logscale",
        "geometric_logit",
        "uniform_bounded",
    )
const EXPECTED_SCALAR_GALLERY_SIZES = (1_000, 100_000)

_scalar_gallery_median(values) = Statistics.median(Float64.(values))

function validate_scalar_gallery_distribution_receipt(path::AbstractString)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "scalar-distribution-gallery-v1",
            "schema must be scalar-distribution-gallery-v1")
    for section in (
            "pins", "environment", "protocol", "support", "support_errors",
            "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    for key in (
            "reactivekernels_sha", "probability_measures_sha",
            "probability_measures_version", "distributions_version",
            "reactant_version", "reactant_jll_version", "julia_version")
        require(haskey(pins, key) && !isempty(string(pins[key])), "pins.$key missing")
    end
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a clean ReactiveKernels tree")
    require(get(pins, "probability_measures_sha", "") ==
            EXPECTED_SCALAR_GALLERY_PROBABILITY_MEASURES_SHA,
            "ProbabilityMeasures commit does not match the reviewed pin")
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")),
            "ReactiveKernels receipt pin must be a full commit SHA")

    protocol = receipt["protocol"]
    require(Tuple(protocol["families"]) == EXPECTED_SCALAR_GALLERY_FAMILIES,
            "protocol family inventory mismatch")
    require(Tuple(Int.(protocol["sizes"])) == EXPECTED_SCALAR_GALLERY_SIZES,
            "protocol size inventory mismatch")
    require(get(protocol, "reactant_sync", false) == true,
            "Reactant timings must use sync=true")
    require(get(protocol, "reactant_transfers_included", true) == false,
            "Reactant device-transfer policy must be explicit and false")
    require(get(protocol, "reactant_parameters_traced", false) == true,
            "Reactant parameters must be traced runtime values")
    require(get(protocol, "reactant_compile_time_in_timed_region", true) == false,
            "Reactant execution timings must exclude compilation")
    require(Int(get(protocol, "rounds", 0)) >= 5,
            "published receipt must contain at least five raw rounds")

    support = receipt["support"]
    support_errors = receipt["support_errors"]
    for family in EXPECTED_SCALAR_GALLERY_FAMILIES
        require(haskey(support, family), "missing support.$family") || continue
        family_support = support[family]
        require(get(family_support, "rk_reactant", false),
                "$family RK Reactant path did not pass")
        require(get(family_support, "probability_measures_reactant", false),
                "$family ProbabilityMeasures Reactant path did not pass")
        if !get(family_support, "distributions_reactant", false)
            diagnostic = get(get(support_errors, family, Dict()),
                             "distributions_reactant", "")
            require(!isempty(diagnostic),
                    "$family unsupported Distributions Reactant path needs a diagnostic")
        end
    end

    expected_rows = [
        (family, n)
        for family in EXPECTED_SCALAR_GALLERY_FAMILIES
        for n in EXPECTED_SCALAR_GALLERY_SIZES
    ]
    observed_rows = [
        (String(row["family"]), Int(row["n"])) for row in receipt["measurements"]
    ]
    require(observed_rows == expected_rows, "measurement family/size inventory mismatch")

    required_measurements = (
        "rk_native", "distributions_native", "probability_measures_native",
        "rk_reactant", "probability_measures_reactant",
    )
    for row in receipt["measurements"]
        family = String(row["family"])
        n = Int(row["n"])
        require(Float64(row["max_relative_error"]) <= 1e-10,
                "$family N=$n exceeds relative value-parity tolerance")
        names = get(support[family], "distributions_reactant", false) ?
            (required_measurements..., "distributions_reactant") :
            required_measurements
        for name in names
            require(haskey(row, name), "$family N=$n missing $name") || continue
            measurement = row[name]
            for raw_key in ("times_ns", "bytes", "allocs")
                require(haskey(measurement, raw_key) && !isempty(measurement[raw_key]),
                        "$family N=$n $name.$raw_key missing")
            end
            all(haskey(measurement, key) for key in
                ("times_ns", "bytes", "allocs")) || continue
            require(all(>(0), measurement["times_ns"]),
                    "$family N=$n $name has non-positive timing")
            require(all(>=(0), measurement["bytes"]),
                    "$family N=$n $name has negative allocation bytes")
            require(all(>=(0), measurement["allocs"]),
                    "$family N=$n $name has negative allocation count")
            require(isapprox(
                Float64(measurement["median_ns"]),
                _scalar_gallery_median(measurement["times_ns"]); rtol = 1e-12,
            ), "$family N=$n $name median_ns mismatch")
            require(Int(measurement["median_bytes"]) == round(
                Int, _scalar_gallery_median(measurement["bytes"])),
                "$family N=$n $name median_bytes mismatch")
            require(Int(measurement["median_allocs"]) == round(
                Int, _scalar_gallery_median(measurement["allocs"])),
                "$family N=$n $name median_allocs mismatch")
            if name == "rk_native"
                require(measurement["median_bytes"] == 0,
                        "$family N=$n RK native reduction must remain zero-allocation")
                require(measurement["median_allocs"] == 0,
                        "$family N=$n RK native reduction must remain zero-allocation")
            end
        end
    end

    errors
end

function main(path)
    errors = validate_scalar_gallery_distribution_receipt(path)
    if isempty(errors)
        println("VALIDATE OK — scalar-distribution-gallery-v1: " *
                "7 families × 2 sizes, RK+ProbabilityMeasures Reactant accepted")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_scalar_gallery_distributions.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
