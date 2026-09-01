#!/usr/bin/env julia

import SHA
import Statistics
import TOML

const EXPECTED_EIGHT_SCHOOLS_AD_BOUNDARIES = (
    "packed_unconstrained", "constrained_parameters", "minimal_likelihood")
const EXPECTED_EIGHT_SCHOOLS_AD_OUTCOMES =
    ("joint", "prior", "likelihood", "pointwise")
const EXPECTED_EIGHT_SCHOOLS_AD_SUPPORTED = Set((
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

_eight_schools_ad_median(values) = Statistics.median(Float64.(values))
_eight_schools_ad_text_sha256(path) = bytes2hex(SHA.sha256(
    replace(read(path, String), "\r\n" => "\n", "\r" => "\n")))

function validate_eight_schools_ad_receipt(path::AbstractString;
        root::AbstractString = normpath(joinpath(dirname(path), "..", "..")))
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "eight-schools-ad-v1",
            "schema must be eight-schools-ad-v1")
    for section in ("pins", "environment", "setup", "protocol", "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a tracked-clean candidate")
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")),
            "candidate pin must be a full commit SHA")
    for key in ("model_source", "primal_comparator_source")
        require(haskey(pins, key), "pins.$key missing")
    end
    primal_path = joinpath(root, get(pins, "primal_receipt_path", ""))
    require(isfile(primal_path), "matched primal receipt is missing")
    if isfile(primal_path)
        primal = TOML.parsefile(primal_path)
        require(get(primal, "schema", "") == "eight-schools-primal-v1",
                "matched receipt must be eight-schools-primal-v1")
        require(get(pins, "primal_receipt_sha256", "") ==
                _eight_schools_ad_text_sha256(primal_path),
                "matched primal receipt digest mismatch")
        haskey(primal, "pins") && require(
            get(pins, "primal_receipt_reactivekernels_sha", "") ==
            get(primal["pins"], "reactivekernels_sha", ""),
            "matched primal receipt code pin mismatch")
    end

    for (key, expected_path) in (
            "model_source" =>
                "packages/ReactiveKernelsPPLExamples/src/eight_schools.jl",
            "primal_comparator_source" =>
                "benchmark/eight_schools_primal_comparison_body.jl",
        )
        haskey(pins, key) || continue
        pin = pins[key]
        require(get(pin, "path", "") == expected_path,
                "$key path mismatch")
        require(occursin(r"^[0-9a-f]{40}$", get(pin, "commit", "")),
                "$key published commit missing")
        require(occursin(r"^[0-9a-f]{40}$", get(pin, "git_blob", "")),
                "$key published blob missing")
        require(occursin(r"^[0-9a-f]{64}$", get(pin, "text_sha256", "")),
                "$key published text digest missing")
        current = get(pin, "current", Dict{String,Any}())
        require(get(current, "path", "") == expected_path,
                "$key current path mismatch")
        absolute_path = joinpath(root, expected_path)
        require(isfile(absolute_path), "$key current source is missing")
        isfile(absolute_path) && require(
            get(current, "text_sha256", "") ==
            _eight_schools_ad_text_sha256(absolute_path),
            "$key current text digest mismatch")
    end
    comparator = get(pins, "primal_comparator_source", Dict{String,Any}())
    require(get(comparator, "current_delta", "") ==
            "documentation-only baseline markers plus terminal definition-only include guard",
            "comparator reuse delta is not narrowly attested")

    protocol = receipt["protocol"]
    require(Tuple(get(protocol, "input_boundaries", String[])) ==
            EXPECTED_EIGHT_SCHOOLS_AD_BOUNDARIES,
            "input-boundary matrix mismatch")
    require(Tuple(get(protocol, "outcomes", String[])) ==
            EXPECTED_EIGHT_SCHOOLS_AD_OUTCOMES,
            "outcome matrix mismatch")
    require(get(protocol, "source_reused", false) == true,
            "benchmark must reuse the published model source")
    require(get(protocol, "matrix_source", "") ==
            "benchmark/receipts/eight-schools-primal-v1.toml",
            "benchmark must name the published primal matrix")
    require(get(protocol, "pointwise_jacobian_or_vjp_invented", true) == false,
            "pointwise VJP must use the public contract, not a benchmark-only surrogate")
    require(occursin("public prepared reverse pullbacks",
                     get(protocol, "pointwise_vjp_contract", "")),
            "pointwise VJP contract is missing")
    require(get(protocol, "preparation_in_timed_region", true) == false,
            "preparation must remain outside steady-state timing")
    require(get(protocol, "first_execution_in_steady_state_region", true) == false,
            "first execution must remain outside steady-state timing")
    rounds = Int(get(protocol, "rounds", 0))
    require(rounds >= 10, "published receipt must retain at least ten rounds")
    relative_tolerance = Float64(get(
        protocol, "parity_relative_tolerance", 0.0))
    absolute_tolerance = Float64(get(
        protocol, "parity_absolute_tolerance", 0.0))
    require(relative_tolerance > 0, "relative parity tolerance must be positive")
    require(absolute_tolerance > 0, "absolute parity tolerance must be positive")

    rows = receipt["measurements"]
    require(length(rows) ==
            length(EXPECTED_EIGHT_SCHOOLS_AD_BOUNDARIES) *
            length(EXPECTED_EIGHT_SCHOOLS_AD_OUTCOMES),
            "receipt must contain the complete 3×4 matrix")
    supported = Set{Tuple{String,String}}()
    for boundary in EXPECTED_EIGHT_SCHOOLS_AD_BOUNDARIES,
        outcome in EXPECTED_EIGHT_SCHOOLS_AD_OUTCOMES
        matches = filter(rows) do row
            row["boundary"] == boundary && row["outcome"] == outcome
        end
        require(length(matches) == 1, "expected one row for $boundary / $outcome")
        length(matches) == 1 || continue
        row = only(matches)
        expected_support = (boundary, outcome) in EXPECTED_EIGHT_SCHOOLS_AD_SUPPORTED
        require(get(row, "supported", false) == expected_support,
                "$boundary / $outcome support mismatch")
        if !expected_support
            require(!isempty(get(row, "unsupported_reason", "")),
                    "$boundary / $outcome lacks unsupported reason")
            for implementation in ("rk_native", "manual_enzyme", "turing_enzyme")
                require(!haskey(row, implementation),
                        "$boundary / $outcome contains unsupported $implementation data")
            end
            continue
        end
        push!(supported, (boundary, outcome))
        expected_active = boundary == "minimal_likelihood" ? "θ" :
            boundary == "constrained_parameters" ? "parameters" :
            "unconstrained"
        expected_length = boundary == "minimal_likelihood" ? 8 : 10
        require(get(row, "active_port", "") == expected_active,
                "$boundary / $outcome active port mismatch")
        require(length(get(row, "finite_difference_gradient", Float64[])) ==
                expected_length,
                "$boundary / $outcome finite-difference oracle length mismatch")

        implementations = boundary == "packed_unconstrained" &&
                          outcome != "pointwise" ?
            ("rk_native", "manual_enzyme", "turing_enzyme") :
            ("rk_native", "manual_enzyme")
        for implementation in implementations
            require(haskey(row, implementation),
                    "$boundary / $outcome lacks $implementation")
            haskey(row, implementation) || continue
            result = row[implementation]
            for key in (
                    "preparation_seconds", "preparation_bytes",
                    "first_execution_seconds", "first_execution_bytes")
                require(Float64(get(result, key, -1)) >= 0,
                        "$boundary / $outcome $implementation $key missing")
            end
            times = get(result, "times_ns", Float64[])
            bytes = get(result, "bytes", Int[])
            allocs = get(result, "allocs", Int[])
            require(length(times) >= 10,
                    "$boundary / $outcome $implementation needs ten timing rounds")
            require(length(bytes) == length(times) && length(allocs) == length(times),
                    "$boundary / $outcome $implementation raw vector lengths mismatch")
            require(all(>(0), times),
                    "$boundary / $outcome $implementation has non-positive timing")
            require(all(>=(0), bytes) && all(>=(0), allocs),
                    "$boundary / $outcome $implementation has negative allocations")
            isempty(times) || require(isapprox(
                Float64(get(result, "median_ns", NaN)),
                _eight_schools_ad_median(times); rtol = 1e-12),
                "$boundary / $outcome $implementation timing median mismatch")
            isempty(bytes) || require(Int(get(result, "median_bytes", -1)) ==
                round(Int, _eight_schools_ad_median(bytes)),
                "$boundary / $outcome $implementation byte median mismatch")
            isempty(allocs) || require(Int(get(result, "median_allocs", -1)) ==
                round(Int, _eight_schools_ad_median(allocs)),
                "$boundary / $outcome $implementation allocation median mismatch")
            require(Int(get(result, "gradient_length", 0)) == expected_length,
                    "$boundary / $outcome $implementation gradient length mismatch")
            absolute_error = Float64(get(
                result, "gradient_max_abs_error", Inf))
            relative_error = Float64(get(
                result, "gradient_max_rel_error", Inf))
            require(absolute_error <= absolute_tolerance ||
                    relative_error <= relative_tolerance,
                    "$boundary / $outcome $implementation exceeds parity tolerance")
            require(Float64(get(result, "value_abs_error", Inf)) <= 1e-11,
                    "$boundary / $outcome $implementation value parity failed")
            caller_owned = implementation != "turing_enzyme" &&
                boundary != "constrained_parameters"
            require(get(result, "caller_owned_gradient", !caller_owned) == caller_owned,
                    "$boundary / $outcome $implementation ownership mismatch")
            if caller_owned && outcome != "pointwise"
                require(all(==(0), bytes) && all(==(0), allocs),
                        "$boundary / $outcome $implementation must remain zero-allocation")
            end
        end
        if boundary != "packed_unconstrained" || outcome == "pointwise"
            require(get(row, "turing_supported", true) == false,
                    "$boundary / $outcome must not fabricate a Turing comparison")
            require(!isempty(get(row, "turing_unsupported_reason", "")),
                    "$boundary / $outcome needs a Turing unsupported reason")
        end
    end
    require(supported == EXPECTED_EIGHT_SCHOOLS_AD_SUPPORTED,
            "supported scalar-cell inventory drifted")
    errors
end

function main(path)
    errors = validate_eight_schools_ad_receipt(path)
    if isempty(errors)
        println("VALIDATE OK — eight-schools-ad-v1: exact 3×4 matrix, " *
                "structured gradients and pointwise reverse pullbacks")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_eight_schools_ad.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
