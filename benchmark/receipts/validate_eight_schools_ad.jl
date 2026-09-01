#!/usr/bin/env julia

import SHA
import Statistics
import TOML

isdefined(@__MODULE__, :EightSchoolsMatrixSpec) ||
    include(joinpath(dirname(@__DIR__), "eight_schools_matrix_spec.jl"))
using .EightSchoolsMatrixSpec

const EXPECTED_EIGHT_SCHOOLS_AD_CONFIGURATIONS = Tuple(
    configuration for configuration in EIGHT_SCHOOLS_RK_CONFIGURATIONS
    if configuration.differentiation == "value_and_gradient" &&
       configuration.compiler == "native"
)

_eight_schools_ad_median(values) = Statistics.median(Float64.(values))
_eight_schools_ad_text_sha256(path) = bytes2hex(SHA.sha256(
    replace(read(path, String), "\r\n" => "\n", "\r" => "\n")))

function validate_eight_schools_ad_receipt(
    path::AbstractString;
    primal_path::AbstractString = joinpath(
        dirname(path), "eight-schools-primal-v2.toml"),
    root::AbstractString = normpath(joinpath(dirname(path), "..", "..")),
)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "eight-schools-ad-v2",
            "schema must be eight-schools-ad-v2")
    for section in ("pins", "environment", "setup", "protocol", "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    require(isfile(primal_path), "missing matched primal receipt: $primal_path")
    isempty(errors) || return errors

    primal = TOML.parsefile(primal_path)
    require(get(primal, "schema", "") == "eight-schools-primal-v2",
            "matched receipt must be eight-schools-primal-v2")
    pins = receipt["pins"]
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a clean candidate")
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")),
            "candidate pin must be a full commit SHA")
    require(get(pins, "primal_receipt_sha256", "") ==
            _eight_schools_ad_text_sha256(primal_path),
            "matched primal receipt digest mismatch")
    require(get(pins, "primal_receipt_reactivekernels_sha", "") ==
            get(primal["pins"], "reactivekernels_sha", ""),
            "matched primal receipt code pin mismatch")
    for (key, expected_path) in (
        "model_source" =>
            "packages/ReactiveKernelsPPLExamples/src/eight_schools.jl",
        "primal_comparator_source" =>
            "benchmark/eight_schools_primal_comparison_body.jl",
    )
        pin = get(pins, key, Dict{String,Any}())
        require(get(pin, "path", "") == expected_path, "$key path mismatch")
        current = get(pin, "current", Dict{String,Any}())
        absolute = joinpath(root, expected_path)
        require(isfile(absolute), "$key current source is missing")
        isfile(absolute) && require(get(current, "text_sha256", "") ==
            _eight_schools_ad_text_sha256(absolute),
            "$key current text digest mismatch")
    end

    protocol = receipt["protocol"]
    require(Tuple(get(protocol, "models", String[])) == EIGHT_SCHOOLS_MODELS,
            "model inventory mismatch")
    require(Tuple(get(protocol, "input_boundaries", String[])) ==
            EIGHT_SCHOOLS_BOUNDARIES, "input-boundary matrix mismatch")
    require(Tuple(get(protocol, "outcomes", String[])) == EIGHT_SCHOOLS_OUTCOMES,
            "outcome matrix mismatch")
    require(Tuple(get(protocol, "rk_configurations", String[])) == Tuple(
            configuration.id for configuration in
            EXPECTED_EIGHT_SCHOOLS_AD_CONFIGURATIONS),
            "native AD configuration inventory mismatch")
    require(get(protocol, "matrix_layout", "") ==
            "long-form provider/model/configuration/boundary/outcome rows",
            "AD receipt must use the long-form matrix")
    require(Tuple(get(protocol, "bound_ports", String[])) ==
            ("observations", "observation_scales"),
            "bound-port inventory mismatch")
    require(get(protocol, "matrix_source", "") ==
            get(pins, "primal_receipt_path", ""),
            "AD receipt must name its matched primal matrix")
    require(get(protocol, "pointwise_jacobian_or_vjp_invented", true) == false,
            "pointwise must use a public VJP rather than an invented Jacobian")
    require(occursin("prepared reverse pullbacks",
                     get(protocol, "pointwise_vjp_contract", "")),
            "pointwise VJP contract is missing")
    require(get(protocol, "preparation_in_timed_region", true) == false,
            "preparation must remain outside timing")
    require(get(protocol, "first_execution_in_steady_state_region", true) == false,
            "first execution must remain outside timing")
    require(Int(get(protocol, "rounds", 0)) >= 10,
            "published receipt must retain ten rounds")
    rtol = Float64(get(protocol, "parity_relative_tolerance", 0.0))
    atol = Float64(get(protocol, "parity_absolute_tolerance", 0.0))
    require(rtol > 0 && atol > 0, "parity tolerances must be positive")

    function checked_result(result, label, expected_length; caller_owned)
        times = Float64.(get(result, "times_ns", Float64[]))
        bytes = Int.(get(result, "bytes", Int[]))
        allocs = Int.(get(result, "allocs", Int[]))
        require(length(times) >= 10, "$label needs ten timing rounds")
        require(length(bytes) == length(times) && length(allocs) == length(times),
                "$label raw vector lengths mismatch")
        isempty(times) || require(isapprox(
            Float64(get(result, "median_ns", NaN)),
            _eight_schools_ad_median(times); rtol = 1e-12),
            "$label timing median mismatch")
        require(Int(get(result, "gradient_length", 0)) == expected_length,
                "$label gradient length mismatch")
        require(Float64(get(result, "gradient_max_abs_error", Inf)) <= atol ||
                Float64(get(result, "gradient_max_rel_error", Inf)) <= rtol,
                "$label gradient parity failed")
        require(Float64(get(result, "value_abs_error", Inf)) <= 1e-11,
                "$label value parity failed")
        require(get(result, "caller_owned_gradient", !caller_owned) == caller_owned,
                "$label ownership mismatch")
    end

    rows = receipt["measurements"]
    expected_count = (length(EXPECTED_EIGHT_SCHOOLS_AD_CONFIGURATIONS) + 2) *
        length(EIGHT_SCHOOLS_BOUNDARIES) * length(EIGHT_SCHOOLS_OUTCOMES)
    require(length(rows) == expected_count,
            "long-form AD matrix must contain $expected_count rows")
    keys = [(row["provider"], row["model"], row["configuration"],
             row["boundary"], row["outcome"]) for row in rows]
    require(length(Set(keys)) == length(keys), "AD matrix has duplicate cells")

    function only_cell(provider, configuration, boundary, outcome)
        matches = filter(rows) do row
            row["provider"] == provider && row["model"] == "centered" &&
                row["configuration"] == configuration &&
                row["boundary"] == boundary && row["outcome"] == outcome
        end
        require(length(matches) == 1,
                "expected one $provider / $configuration / $boundary / $outcome row")
        length(matches) == 1 ? only(matches) : nothing
    end

    for configuration in EXPECTED_EIGHT_SCHOOLS_AD_CONFIGURATIONS,
        boundary in EIGHT_SCHOOLS_BOUNDARIES, outcome in EIGHT_SCHOOLS_OUTCOMES
        row = only_cell("rk", configuration.id, boundary, outcome)
        row === nothing && continue
        expected_state, _ = matrix_support(configuration, boundary, outcome)
        require(get(row, "state", "") == expected_state,
                "RK AD support drift for $(configuration.id) / $boundary / $outcome")
        if expected_state == "supported"
            expected_length = boundary == "minimal_likelihood" ? 8 : 10
            require(haskey(row, "result"), "supported RK AD cell lacks result")
            haskey(row, "result") && checked_result(
                row["result"], "RK $(configuration.id) / $boundary / $outcome",
                expected_length;
                caller_owned = boundary != "constrained_parameters")
        else
            require(!haskey(row, "result"), "non-measured RK AD cell has data")
            require(!isempty(get(row, "reason", "")),
                    "non-measured RK AD cell lacks a reason")
        end
    end

    unbound = first(EXPECTED_EIGHT_SCHOOLS_AD_CONFIGURATIONS)
    for boundary in EIGHT_SCHOOLS_BOUNDARIES, outcome in EIGHT_SCHOOLS_OUTCOMES
        expected_state, _ = matrix_support(unbound, boundary, outcome)
        manual = only_cell("manual_julia", "manual_ad", boundary, outcome)
        manual === nothing || require(get(manual, "state", "") == expected_state,
            "manual AD support drift for $boundary / $outcome")
        if manual !== nothing && expected_state == "supported"
            expected_length = boundary == "minimal_likelihood" ? 8 : 10
            checked_result(manual["result"], "manual / $boundary / $outcome",
                           expected_length;
                           caller_owned = boundary != "constrained_parameters")
        end

        turing = only_cell("turing", "turing_ad", boundary, outcome)
        turing_state = expected_state == "supported" && outcome != "pointwise" &&
            boundary == "packed_unconstrained" ? "supported" : "unsupported"
        turing === nothing || require(get(turing, "state", "") == turing_state,
            "Turing AD support drift for $boundary / $outcome")
        if turing !== nothing && turing_state == "supported"
            checked_result(turing["result"], "Turing / $boundary / $outcome",
                           10; caller_owned = false)
        end
    end
    errors
end

function main(path)
    errors = validate_eight_schools_ad_receipt(path)
    isempty(errors) &&
        (println("VALIDATE OK — eight-schools-ad-v2 matrix accepted"); return 0)
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_eight_schools_ad.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
