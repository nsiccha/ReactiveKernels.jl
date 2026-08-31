#!/usr/bin/env julia

import SHA
import Statistics
import TOML

const EXPECTED_EIGHT_SCHOOLS_REACTANT_BOUNDARIES = (
    "packed_unconstrained", "constrained_parameters", "minimal_likelihood")
const EXPECTED_EIGHT_SCHOOLS_REACTANT_OUTCOMES =
    ("joint", "prior", "likelihood", "pointwise")

_eight_schools_reactant_median(values) = Statistics.median(Float64.(values))

function validate_eight_schools_reactant_receipt(
    path::AbstractString;
    primal_path::AbstractString = joinpath(
        dirname(path), "eight-schools-primal-v1.toml"),
)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "eight-schools-reactant-v1",
            "schema must be eight-schools-reactant-v1")
    for section in ("pins", "environment", "setup", "protocol", "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    require(isfile(primal_path), "missing matched primal receipt: $primal_path")
    isempty(errors) || return errors

    primal = TOML.parsefile(primal_path)
    require(get(primal, "schema", "") == "eight-schools-primal-v1",
            "matched receipt must use schema eight-schools-primal-v1")
    haskey(primal, "protocol") && haskey(primal, "measurements") ||
        require(false, "matched primal receipt lacks protocol or measurements")

    pins = receipt["pins"]
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a clean ReactiveKernels tree")
    for key in (
        "reactivekernels_sha", "reactivekernels_version",
        "reactivekernelspplexamples_version",
        "reactivekernelsdistributionkernels_version", "reactant_version",
        "reactant_jll_version", "julia_version", "source_authority_path",
        "source_authority_blob", "source_text_sha256",
        "primal_receipt_sha256", "primal_receipt_reactivekernels_sha",
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
    require(get(pins, "primal_receipt_sha256", "") ==
            bytes2hex(SHA.sha256(read(primal_path))),
            "matched primal receipt digest mismatch")
    if haskey(primal, "pins")
        require(get(pins, "primal_receipt_reactivekernels_sha", "") ==
                get(primal["pins"], "reactivekernels_sha", ""),
                "matched primal receipt code pin mismatch")
    end

    protocol = receipt["protocol"]
    require(Tuple(get(protocol, "input_boundaries", String[])) ==
            EXPECTED_EIGHT_SCHOOLS_REACTANT_BOUNDARIES,
            "input-boundary matrix mismatch")
    require(Tuple(get(protocol, "outcomes", String[])) ==
            EXPECTED_EIGHT_SCHOOLS_REACTANT_OUTCOMES,
            "outcome matrix mismatch")
    if haskey(primal, "protocol")
        require(Tuple(get(primal["protocol"], "input_boundaries", String[])) ==
                EXPECTED_EIGHT_SCHOOLS_REACTANT_BOUNDARIES,
                "matched primal receipt input-boundary matrix mismatch")
        require(Tuple(get(primal["protocol"], "outcomes", String[])) ==
                EXPECTED_EIGHT_SCHOOLS_REACTANT_OUTCOMES,
                "matched primal receipt outcome matrix mismatch")
    end
    require(get(protocol, "source_reused", false) == true,
            "benchmark must reuse the authored Eight Schools source")
    require(get(protocol, "matrix_source", "") ==
            "benchmark/receipts/eight-schools-primal-v1.toml",
            "benchmark must name the matched primal receipt")
    require(Int(get(protocol, "rounds", 0)) >= 20,
            "published receipt must contain at least twenty raw rounds")
    require(Float64(get(protocol, "target_seconds_per_round", 0.0)) > 0,
            "timing target must be positive")
    require(get(protocol, "reactant_sync", false) == true,
            "Reactant timings must use sync=true")
    for key in (
        "setup_in_timed_region", "preparation_in_timed_region",
        "reactant_compile_time_in_timed_region",
        "reactant_transfers_in_timed_region",
        "reactant_readback_in_timed_region",
    )
        require(get(protocol, key, true) == false, "$key must be false")
    end
    require(get(protocol, "unsupported_cells_recorded", false) == true,
            "unsupported Reactant cells must retain diagnostics")
    require(get(protocol, "gradients_included", true) == false,
            "Eight Schools Reactant receipt must not contain gradients")
    require(get(protocol, "generated_predictions_included", true) == false,
            "predictive generated quantities must not enter this comparison")

    setup = receipt["setup"]
    for key in (
        "environment_seconds", "package_precompile_seconds",
        "kernel_preparation_seconds",
    )
        require(Float64(get(setup, key, -1.0)) >= 0,
                "setup.$key must be nonnegative")
    end

    rows = receipt["measurements"]
    require(length(rows) ==
            length(EXPECTED_EIGHT_SCHOOLS_REACTANT_BOUNDARIES) *
            length(EXPECTED_EIGHT_SCHOOLS_REACTANT_OUTCOMES),
            "matrix must contain exactly one row per boundary/outcome pair")
    primal_rows = haskey(primal, "measurements") ? primal["measurements"] : Any[]
    supported_reactant = Set{Tuple{String,String}}()
    for boundary in EXPECTED_EIGHT_SCHOOLS_REACTANT_BOUNDARIES,
        outcome in EXPECTED_EIGHT_SCHOOLS_REACTANT_OUTCOMES
        matches = filter(rows) do row
            row["boundary"] == boundary && row["outcome"] == outcome
        end
        require(length(matches) == 1, "expected one row for $boundary / $outcome")
        length(matches) == 1 || continue
        row = only(matches)

        primal_matches = filter(primal_rows) do candidate
            candidate["boundary"] == boundary && candidate["outcome"] == outcome
        end
        require(length(primal_matches) == 1,
                "matched primal receipt needs one $boundary / $outcome row")
        native_expected = length(primal_matches) == 1 &&
            haskey(only(primal_matches), "rk_native")
        require(get(row, "rk_native_supported", false) == native_expected,
                "$boundary / $outcome native support drifted from primal receipt")
        native_present = haskey(row, "rk_native")
        require(native_present == native_expected,
                "$boundary / $outcome native measurement support mismatch")

        if native_expected && native_present
            native = row["rk_native"]
            require(length(get(native, "times_ns", Float64[])) >= 20,
                    "$boundary / $outcome native needs twenty timing rounds")
            if haskey(native, "times_ns") && !isempty(native["times_ns"])
                require(all(>(0), native["times_ns"]),
                        "$boundary / $outcome native has non-positive timing")
                require(haskey(native, "median_ns"),
                        "$boundary / $outcome native median missing")
                haskey(native, "median_ns") && require(
                    isapprox(Float64(native["median_ns"]),
                             _eight_schools_reactant_median(native["times_ns"]);
                             rtol = 1e-12),
                    "$boundary / $outcome native median mismatch")
            end
            require(Int(get(native, "calls_per_round", 0)) > 0,
                    "$boundary / $outcome native calls_per_round missing")
            require(Float64(get(row, "reactant_transfer_seconds", -1.0)) >= 0,
                    "$boundary / $outcome transfer setup missing")
            require(Float64(get(row, "reactant_compile_seconds", -1.0)) >= 0,
                    "$boundary / $outcome compile attempt missing")
        end

        reactant_supported = get(row, "rk_reactant_supported", false)
        if reactant_supported
            push!(supported_reactant, (boundary, outcome))
            require(haskey(row, "rk_reactant"),
                    "$boundary / $outcome supported without a Reactant measurement")
            require(!haskey(row, "rk_reactant_error"),
                    "$boundary / $outcome supported but retains an error")
            require(Float64(get(row, "reactant_first_execution_seconds", -1.0)) >= 0,
                    "$boundary / $outcome first execution missing")
            require(Float64(get(row, "max_abs_error", Inf)) <= 1e-10,
                    "$boundary / $outcome exceeds native/Reactant parity tolerance")
            if haskey(row, "rk_reactant")
                result = row["rk_reactant"]
                require(length(get(result, "times_ns", Float64[])) >= 20,
                        "$boundary / $outcome Reactant needs twenty timing rounds")
                if haskey(result, "times_ns") && !isempty(result["times_ns"])
                    require(all(>(0), result["times_ns"]),
                            "$boundary / $outcome Reactant has non-positive timing")
                    require(haskey(result, "median_ns"),
                            "$boundary / $outcome Reactant median missing")
                    haskey(result, "median_ns") && require(
                        isapprox(Float64(result["median_ns"]),
                                 _eight_schools_reactant_median(result["times_ns"]);
                                 rtol = 1e-12),
                        "$boundary / $outcome Reactant median mismatch")
                end
                require(Int(get(result, "calls_per_round", 0)) > 0,
                        "$boundary / $outcome Reactant calls_per_round missing")
            end
        else
            require(!haskey(row, "rk_reactant"),
                    "$boundary / $outcome unsupported but contains a measurement")
            require(!isempty(get(row, "rk_reactant_error", "")),
                    "$boundary / $outcome unsupported without a diagnostic")
        end
    end
    require(("minimal_likelihood", "likelihood") in supported_reactant,
            "minimal likelihood must compile through Reactant")
    require(("minimal_likelihood", "pointwise") in supported_reactant,
            "minimal pointwise likelihood must compile through Reactant")
    errors
end

function main(path)
    errors = validate_eight_schools_reactant_receipt(path)
    if isempty(errors)
        println("VALIDATE OK — eight-schools-reactant-v1: matched 3×4 matrix accepted")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_eight_schools_reactant.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
