#!/usr/bin/env julia

import Statistics
import TOML

include(joinpath(@__DIR__, "..", "distribution_benchmark_cases.jl"))
using .DistributionBenchmarkCases:
    NORMAL_SIZES, SCALAR_GALLERY_FAMILIES, SCALAR_GALLERY_SIZES, STRUCTURED_SIZES

_distribution_gradient_median(values) = Statistics.median(Float64.(values))

const _DISTRIBUTION_GRADIENT_ACTIVE = Dict(
    "cauchy_location_scale" => ("x", "vector"),
    "laplace_location_scale" => ("x", "vector"),
    "bernoulli_logit" => ("logit", "scalar"),
    "lognormal_logscale" => ("x", "vector"),
    "exponential_logscale" => ("x", "vector"),
    "geometric_logit" => ("logitp", "scalar"),
    "uniform_bounded" => ("x", "vector"),
)

function _expected_distribution_gradient_rows()
    rows = Tuple{String,String,Int,String,String}[]
    append!(rows, [
        ("normal_plate", "normal", n, "x", "vector") for n in NORMAL_SIZES
    ])
    for family in SCALAR_GALLERY_FAMILIES, n in SCALAR_GALLERY_SIZES
        active, kind = _DISTRIBUTION_GRADIENT_ACTIVE[family]
        push!(rows, ("scalar_gallery", family, n, active, kind))
    end
    append!(rows, [
        ("structured", "mvnormal_cholesky", n, "x", "vector")
        for n in STRUCTURED_SIZES
    ])
    rows
end

function validate_distribution_gradient_receipt(path::AbstractString)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "distribution-gradient-v1",
            "schema must be distribution-gradient-v1")
    for section in ("pins", "environment", "protocol", "source_receipts",
                    "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a clean ReactiveKernels tree")
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")),
            "ReactiveKernels receipt pin must be a full commit SHA")
    for key in ("differentiationinterface_version", "enzyme_version", "julia_version")
        require(haskey(pins, key) && !isempty(string(pins[key])), "pins.$key missing")
    end

    protocol = receipt["protocol"]
    rounds = Int(get(protocol, "rounds", 0))
    require(rounds >= 5, "published receipt must contain at least five raw rounds")
    require(get(protocol, "preparation_in_timed_region", true) == false,
            "AD preparation must be outside the timed region")
    require(get(protocol, "returned_surface", "") == "ad_gradient",
            "returned-gradient surface mismatch")
    require(get(protocol, "caller_owned_surface", "") == "ad_value_and_gradient!",
            "caller-owned surface mismatch")
    require(Tuple(Int.(get(protocol, "normal_sizes", Int[]))) == NORMAL_SIZES,
            "Normal size inventory mismatch")
    require(Tuple(get(protocol, "scalar_gallery_families", String[])) ==
            SCALAR_GALLERY_FAMILIES, "scalar-gallery family inventory mismatch")
    require(Tuple(Int.(get(protocol, "scalar_gallery_sizes", Int[]))) ==
            SCALAR_GALLERY_SIZES, "scalar-gallery size inventory mismatch")
    require(Tuple(Int.(get(protocol, "structured_sizes", Int[]))) ==
            STRUCTURED_SIZES, "structured size inventory mismatch")

    source_receipts = receipt["source_receipts"]
    expected_schemas = Dict(
        "normal" => "distribution-logdensity-v1",
        "scalar_gallery" => "scalar-distribution-gallery-v1",
        "structured" => "structured-distribution-logdensity-v1",
    )
    for (name, schema) in expected_schemas
        require(haskey(source_receipts, name), "missing source_receipts.$name") ||
            continue
        require(get(source_receipts[name], "schema", "") == schema,
                "source_receipts.$name schema mismatch")
    end

    rows = receipt["measurements"]
    observed_rows = [(
        String(row["group"]), String(row["family"]), Int(row["n"]),
        String(row["active"]), String(row["active_kind"]),
    ) for row in rows]
    require(observed_rows == _expected_distribution_gradient_rows(),
            "distribution-gradient case inventory mismatch")

    for row in rows
        group = String(row["group"])
        family = String(row["family"])
        n = Int(row["n"])
        kind = String(row["active_kind"])
        label = "$group/$family N=$n"
        require(Float64(row["max_abs_error"]) <= 1e-8,
                "$label exceeds analytic-gradient tolerance")
        if kind == "vector"
            require(haskey(row, "caller_owned_gradient"),
                    "$label lacks caller-owned measurement")
            require(Float64(get(row, "max_value_error", Inf)) <= 1e-10,
                    "$label value-and-gradient value mismatch")
        else
            require(!haskey(row, "caller_owned_gradient"),
                    "$label must not claim a mutable scalar destination")
        end

        surfaces = kind == "vector" ?
            ("returned_gradient", "caller_owned_gradient") : ("returned_gradient",)
        for surface in surfaces
            require(haskey(row, surface), "$label missing $surface") || continue
            measurement = row[surface]
            for raw_key in ("times_ns", "bytes", "allocs")
                require(haskey(measurement, raw_key) && !isempty(measurement[raw_key]),
                        "$label $surface.$raw_key missing")
                haskey(measurement, raw_key) && require(
                    length(measurement[raw_key]) == rounds,
                    "$label $surface.$raw_key must retain every round",
                )
            end
            all(haskey(measurement, key) for key in
                ("times_ns", "bytes", "allocs")) || continue
            require(all(>(0), measurement["times_ns"]),
                    "$label $surface has non-positive timing")
            require(all(>=(0), measurement["bytes"]),
                    "$label $surface has negative bytes")
            require(all(>=(0), measurement["allocs"]),
                    "$label $surface has negative allocations")
            require(isapprox(
                Float64(measurement["median_ns"]),
                _distribution_gradient_median(measurement["times_ns"]); rtol = 1e-12,
            ), "$label $surface median_ns mismatch")
            require(Int(measurement["median_bytes"]) == round(
                Int, _distribution_gradient_median(measurement["bytes"])),
                "$label $surface median_bytes mismatch")
            require(Int(measurement["median_allocs"]) == round(
                Int, _distribution_gradient_median(measurement["allocs"])),
                "$label $surface median_allocs mismatch")
        end

        if group in ("normal_plate", "scalar_gallery")
            zero_surface = kind == "vector" ?
                row["caller_owned_gradient"] : row["returned_gradient"]
            require(all(==(0), zero_surface["bytes"]),
                    "$label steady-state gradient must remain zero-byte")
            require(all(==(0), zero_surface["allocs"]),
                    "$label steady-state gradient must remain zero-allocation")
        end
        if kind == "vector"
            require(Int(row["returned_gradient"]["median_bytes"]) > 0,
                    "$label returned vector must account for owned result storage")
        end
    end

    errors
end

function main(path)
    errors = validate_distribution_gradient_receipt(path)
    if isempty(errors)
        println("VALIDATE OK — distribution-gradient-v1: exact distribution corpus, " *
                "analytic parity, and zero-allocation scalar/plate steady state")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_distribution_gradients.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
