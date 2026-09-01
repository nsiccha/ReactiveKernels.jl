#!/usr/bin/env julia

import SHA
import Statistics
import TOML

const EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_BOUNDARIES = (
    "packed_unconstrained", "constrained_parameters", "minimal_likelihood")
const EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_OUTCOMES =
    ("joint", "prior", "likelihood", "pointwise")
# Native RK AD matches all nine cells in eight-schools-ad-v1: seven scalar
# gradients plus two fixed-cotangent pointwise pullbacks.
const EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_NATIVE_SUPPORTED = Set((
    ("packed_unconstrained", "joint"),
    ("packed_unconstrained", "prior"),
    ("packed_unconstrained", "likelihood"),
    ("packed_unconstrained", "pointwise"),
    ("constrained_parameters", "joint"),
    ("constrained_parameters", "prior"),
    ("constrained_parameters", "likelihood"),
    ("minimal_likelihood", "likelihood"),
    ("minimal_likelihood", "pointwise"),
))
# Reactant compile attempts remain limited to the public array-backed scalar
# value-and-gradient surface. Structured gradients and pullbacks are native-only
# here; the receipt must not manufacture compiled APIs for them.
const EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_COMPILE_CANDIDATES = Set((
    ("packed_unconstrained", "joint"),
    ("packed_unconstrained", "prior"),
    ("packed_unconstrained", "likelihood"),
    ("minimal_likelihood", "likelihood"),
))
const EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_REACTANT_REQUIRED = Set((
    ("packed_unconstrained", "likelihood"),
    ("minimal_likelihood", "likelihood"),
))

_eight_schools_reactant_ad_median(values) = Statistics.median(Float64.(values))
_eight_schools_reactant_ad_text_sha256(path) = bytes2hex(SHA.sha256(
    replace(read(path, String), "\r\n" => "\n", "\r" => "\n")))

function validate_eight_schools_reactant_ad_receipt(path::AbstractString;
        root::AbstractString = normpath(joinpath(dirname(path), "..", "..")))
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "eight-schools-reactant-ad-v1",
            "schema must be eight-schools-reactant-ad-v1")
    for section in ("pins", "environment", "setup", "protocol", "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a clean ReactiveKernels tree")
    for key in (
        "reactivekernels_sha", "reactivekernels_version",
        "reactivekernelspplexamples_version",
        "reactivekernelsdistributionkernels_version", "reactant_version",
        "reactant_jll_version", "enzyme_version",
        "differentiationinterface_version", "julia_version",
        "source_authority_path", "source_authority_blob", "source_text_sha256",
        "ad_receipt_path", "ad_receipt_sha256",
        "ad_receipt_reactivekernels_sha",
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
            "packages/ReactiveKernelsPPLExamples/src/eight_schools.jl",
            "unexpected Eight Schools source-authority path")

    ad_path = joinpath(root, get(pins, "ad_receipt_path", ""))
    require(get(pins, "ad_receipt_path", "") ==
            "benchmark/receipts/eight-schools-ad-v1.toml",
            "unexpected matched AD receipt path")
    require(isfile(ad_path), "matched AD receipt is missing: $ad_path")
    if isfile(ad_path)
        ad_receipt = TOML.parsefile(ad_path)
        require(get(ad_receipt, "schema", "") == "eight-schools-ad-v1",
                "matched receipt must use schema eight-schools-ad-v1")
        require(get(pins, "ad_receipt_sha256", "") ==
                _eight_schools_reactant_ad_text_sha256(ad_path),
                "matched AD receipt digest mismatch")
        if haskey(ad_receipt, "pins")
            require(get(pins, "ad_receipt_reactivekernels_sha", "") ==
                    get(ad_receipt["pins"], "reactivekernels_sha", ""),
                    "matched AD receipt code pin mismatch")
        end
        # The matched receipt owns the full native reverse-AD inventory.
        ad_supported = Set{Tuple{String,String}}()
        for row in get(ad_receipt, "measurements", Any[])
            get(row, "supported", false) &&
                push!(ad_supported, (row["boundary"], row["outcome"]))
        end
        require(ad_supported == EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_NATIVE_SUPPORTED,
                "matched AD receipt supported inventory drifted")
    end

    protocol = receipt["protocol"]
    require(Tuple(get(protocol, "input_boundaries", String[])) ==
            EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_BOUNDARIES,
            "input-boundary matrix mismatch")
    require(Tuple(get(protocol, "outcomes", String[])) ==
            EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_OUTCOMES,
            "outcome matrix mismatch")
    require(get(protocol, "source_reused", false) == true,
            "benchmark must reuse the authored Eight Schools source")
    require(get(protocol, "matrix_source", "") ==
            "benchmark/receipts/eight-schools-ad-v1.toml",
            "benchmark must name the matched AD receipt")
    require(get(protocol, "gradient_operation", "") ==
            "value and gradient for scalar WANTs; value and reverse pullback for pointwise WANTs",
            "gradient/pullback operation contract mismatch")
    require(get(protocol, "compiled_pullback_exposed", true) == false,
            "receipt must not claim a compiled pullback surface")
    require(get(protocol, "compiled_structured_active_exposed", true) == false,
            "receipt must not claim compiled structured active arguments")
    require(Int(get(protocol, "rounds", 0)) >= 20,
            "published receipt must contain at least twenty raw rounds")
    require(Float64(get(protocol, "target_seconds_per_round", 0.0)) > 0,
            "timing target must be positive")
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
            "pointwise VJP must use the public native contract")
    require(occursin("public prepared reverse pullbacks",
                     get(protocol, "pointwise_vjp_contract", "")),
            "pointwise VJP contract is missing")
    parity_atol = Float64(get(protocol, "parity_atol", 0.0))
    require(parity_atol > 0, "parity tolerance must be positive")

    rows = receipt["measurements"]
    require(length(rows) ==
            length(EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_BOUNDARIES) *
            length(EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_OUTCOMES),
            "matrix must contain exactly one row per boundary/outcome pair")
    reactant_supported = Set{Tuple{String,String}}()
    for boundary in EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_BOUNDARIES,
        outcome in EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_OUTCOMES
        matches = filter(rows) do row
            row["boundary"] == boundary && row["outcome"] == outcome
        end
        require(length(matches) == 1, "expected one row for $boundary / $outcome")
        length(matches) == 1 || continue
        row = only(matches)

        native_expected =
            (boundary, outcome) in EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_NATIVE_SUPPORTED
        require(get(row, "rk_native_ad_supported", false) == native_expected,
                "$boundary / $outcome native AD support drifted")

        if !native_expected
            require(!haskey(row, "rk_native_ad"),
                    "$boundary / $outcome unsupported but contains native AD data")
            require(!isempty(get(row, "rk_native_ad_error", "")),
                    "$boundary / $outcome native AD unsupported without a diagnostic")
            require(get(row, "rk_reactant_ad_supported", false) == false,
                    "$boundary / $outcome Reactant AD cannot outrun native AD support")
            require(!haskey(row, "rk_reactant_ad"),
                    "$boundary / $outcome unsupported but contains Reactant AD data")
            require(!isempty(get(row, "rk_reactant_ad_error", "")),
                    "$boundary / $outcome Reactant AD unsupported without a diagnostic")
            continue
        end

        expected_active = boundary == "minimal_likelihood" ? "θ" :
            boundary == "constrained_parameters" ? "parameters" : "unconstrained"
        expected_operation = outcome == "pointwise" ?
            "value and pullback" : "value and gradient"
        expected_sensitivity_length = boundary == "minimal_likelihood" ? 8 : 10
        require(get(row, "active_port", "") == expected_active,
                "$boundary / $outcome active port mismatch")
        require(get(row, "operation", "") == expected_operation,
                "$boundary / $outcome operation mismatch")
        require(Int(get(row, "native_sensitivity_length", 0)) ==
                expected_sensitivity_length,
                "$boundary / $outcome native sensitivity length mismatch")
        if outcome == "pointwise"
            require(length(get(row, "output_cotangent", Float64[])) == 8,
                    "$boundary / $outcome output cotangent mismatch")
        else
            require(!haskey(row, "output_cotangent"),
                    "$boundary / $outcome scalar cell has an output cotangent")
        end
        require(Float64(get(row, "ad_preparation_seconds", -1.0)) >= 0,
                "$boundary / $outcome AD preparation time missing")
        require(haskey(row, "rk_native_ad"),
                "$boundary / $outcome native AD supported without a measurement")
        if haskey(row, "rk_native_ad")
            native = row["rk_native_ad"]
            require(length(get(native, "times_ns", Float64[])) >= 20,
                    "$boundary / $outcome native AD needs twenty timing rounds")
            if haskey(native, "times_ns") && !isempty(native["times_ns"])
                require(all(>(0), native["times_ns"]),
                        "$boundary / $outcome native AD has non-positive timing")
                require(haskey(native, "median_ns"),
                        "$boundary / $outcome native AD median missing")
                haskey(native, "median_ns") && require(
                    isapprox(Float64(native["median_ns"]),
                             _eight_schools_reactant_ad_median(native["times_ns"]);
                             rtol = 1e-12),
                    "$boundary / $outcome native AD median mismatch")
            end
            require(Int(get(native, "calls_per_round", 0)) > 0,
                    "$boundary / $outcome native AD calls_per_round missing")
        end

        compile_candidate =
            (boundary, outcome) in EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_COMPILE_CANDIDATES
        reactant_ok = get(row, "rk_reactant_ad_supported", false)
        if !compile_candidate
            require(reactant_ok == false,
                    "$boundary / $outcome claims an unexposed compiled AD surface")
            require(!haskey(row, "rk_reactant_ad"),
                    "$boundary / $outcome unexposed but has Reactant timing")
            require(!isempty(get(row, "rk_reactant_ad_error", "")),
                    "$boundary / $outcome unexposed without a diagnostic")
            for key in ("reactant_transfer_seconds", "reactant_ad_compile_seconds",
                        "reactant_first_execution_seconds", "max_abs_error",
                        "value_abs_error")
                require(!haskey(row, key),
                        "$boundary / $outcome unexposed but records $key")
            end
            continue
        end

        require(Float64(get(row, "reactant_ad_compile_seconds", -1.0)) >= 0,
                "$boundary / $outcome Reactant AD compile attempt missing")
        require(Float64(get(row, "reactant_transfer_seconds", -1.0)) >= 0,
                "$boundary / $outcome Reactant AD transfer setup missing")
        if reactant_ok
            push!(reactant_supported, (boundary, outcome))
            require(haskey(row, "rk_reactant_ad"),
                    "$boundary / $outcome supported without a Reactant AD measurement")
            require(!haskey(row, "rk_reactant_ad_error"),
                    "$boundary / $outcome supported but retains an error")
            require(Float64(get(row, "reactant_first_execution_seconds", -1.0)) >= 0,
                    "$boundary / $outcome Reactant AD first execution missing")
            require(Float64(get(row, "max_abs_error", Inf)) <= parity_atol,
                    "$boundary / $outcome exceeds native/Reactant AD gradient parity")
            require(Float64(get(row, "value_abs_error", Inf)) <= 1e-10,
                    "$boundary / $outcome exceeds native/Reactant AD value parity")
            if haskey(row, "rk_reactant_ad")
                result = row["rk_reactant_ad"]
                require(length(get(result, "times_ns", Float64[])) >= 20,
                        "$boundary / $outcome Reactant AD needs twenty timing rounds")
                if haskey(result, "times_ns") && !isempty(result["times_ns"])
                    require(all(>(0), result["times_ns"]),
                            "$boundary / $outcome Reactant AD has non-positive timing")
                    require(haskey(result, "median_ns"),
                            "$boundary / $outcome Reactant AD median missing")
                    haskey(result, "median_ns") && require(
                        isapprox(Float64(result["median_ns"]),
                                 _eight_schools_reactant_ad_median(result["times_ns"]);
                                 rtol = 1e-12),
                        "$boundary / $outcome Reactant AD median mismatch")
                end
                require(Int(get(result, "calls_per_round", 0)) > 0,
                        "$boundary / $outcome Reactant AD calls_per_round missing")
            end
        else
            require(!haskey(row, "rk_reactant_ad"),
                    "$boundary / $outcome Reactant AD unsupported but has a measurement")
            require(!isempty(get(row, "rk_reactant_ad_error", "")),
                    "$boundary / $outcome Reactant AD unsupported without a diagnostic")
        end
    end
    for cell in EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_REACTANT_REQUIRED
        require(cell in reactant_supported,
                "$(cell[1]) / $(cell[2]) must compile through Reactant AD")
    end
    errors
end

function main(path)
    errors = validate_eight_schools_reactant_ad_receipt(path)
    if isempty(errors)
        println("VALIDATE OK — eight-schools-reactant-ad-v1: matched 3×4 native inventory and honest compiled subset")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_eight_schools_reactant_ad.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
