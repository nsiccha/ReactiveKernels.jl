#!/usr/bin/env julia

import SHA
import Statistics
import TOML

isdefined(@__MODULE__, :EightSchoolsMatrixSpec) ||
    include(joinpath(dirname(@__DIR__), "eight_schools_matrix_spec.jl"))
using .EightSchoolsMatrixSpec

const EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_CONFIGURATIONS = Tuple(
    configuration for configuration in EIGHT_SCHOOLS_RK_CONFIGURATIONS
    if configuration.differentiation == "value_and_gradient" &&
       configuration.compiler == "reactant"
)

_eight_schools_reactant_ad_text_sha256(path) = bytes2hex(SHA.sha256(
    replace(read(path, String), "\r\n" => "\n", "\r" => "\n")))

function validate_eight_schools_reactant_ad_receipt(
    path::AbstractString;
    ad_path::AbstractString = joinpath(
        dirname(path), "eight-schools-ad-v2.toml"),
)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "eight-schools-reactant-ad-v2",
            "schema must be eight-schools-reactant-ad-v2")
    for section in ("pins", "environment", "setup", "protocol", "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    require(isfile(ad_path), "missing matched AD receipt: $ad_path")
    isempty(errors) || return errors

    ad = TOML.parsefile(ad_path)
    require(get(ad, "schema", "") == "eight-schools-ad-v2",
            "matched receipt must be eight-schools-ad-v2")
    pins = receipt["pins"]
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a clean candidate")
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")),
            "candidate pin must be a full commit SHA")
    require(get(pins, "ad_receipt_sha256", "") ==
            _eight_schools_reactant_ad_text_sha256(ad_path),
            "matched AD receipt digest mismatch")
    require(get(pins, "ad_receipt_reactivekernels_sha", "") ==
            get(ad["pins"], "reactivekernels_sha", ""),
            "matched AD receipt code pin mismatch")

    protocol = receipt["protocol"]
    require(Tuple(get(protocol, "models", String[])) == EIGHT_SCHOOLS_MODELS,
            "model inventory mismatch")
    require(Tuple(get(protocol, "input_boundaries", String[])) ==
            EIGHT_SCHOOLS_BOUNDARIES, "input-boundary matrix mismatch")
    require(Tuple(get(protocol, "outcomes", String[])) == EIGHT_SCHOOLS_OUTCOMES,
            "outcome matrix mismatch")
    require(Tuple(get(protocol, "rk_configurations", String[])) == Tuple(
            configuration.id for configuration in
            EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_CONFIGURATIONS),
            "Reactant AD configuration inventory mismatch")
    require(get(protocol, "matrix_layout", "") ==
            "long-form provider/model/configuration/boundary/outcome rows",
            "Reactant AD receipt must use the long-form matrix")
    require(Tuple(get(protocol, "bound_ports", String[])) ==
            ("observations", "observation_scales"),
            "bound-port inventory mismatch")
    require(get(protocol, "pointwise_jacobian_or_vjp_invented", true) == false,
            "pointwise must remain unsupported without a public contract")
    require(Int(get(protocol, "rounds", 0)) >= 10,
            "published receipt must retain ten rounds")
    for key in ("setup_in_timed_region", "preparation_in_timed_region",
                "ad_preparation_in_timed_region",
                "reactant_compile_time_in_timed_region",
                "reactant_transfers_in_timed_region",
                "reactant_readback_in_timed_region",
                "first_execution_in_steady_state_region")
        require(get(protocol, key, true) == false, "$key must be false")
    end

    function checked_measurement(result, label)
        times = Float64.(get(result, "times_ns", Float64[]))
        require(length(times) >= 10, "$label needs ten timing rounds")
        isempty(times) && return
        require(all(>(0), times), "$label has non-positive timing")
        require(isapprox(Float64(get(result, "median_ns", NaN)),
                         Statistics.median(times); rtol = 1e-12),
                "$label median mismatch")
    end

    rows = receipt["measurements"]
    expected_count = length(EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_CONFIGURATIONS) *
        length(EIGHT_SCHOOLS_BOUNDARIES) * length(EIGHT_SCHOOLS_OUTCOMES)
    require(length(rows) == expected_count,
            "long-form Reactant AD matrix must contain $expected_count rows")
    keys = [(row["model"], row["configuration"], row["boundary"], row["outcome"])
            for row in rows]
    require(length(Set(keys)) == length(keys),
            "Reactant AD matrix has duplicate cells")

    for configuration in EXPECTED_EIGHT_SCHOOLS_REACTANT_AD_CONFIGURATIONS,
        boundary in EIGHT_SCHOOLS_BOUNDARIES, outcome in EIGHT_SCHOOLS_OUTCOMES
        matches = filter(rows) do row
            row["provider"] == "rk" && row["model"] == "centered" &&
                row["configuration"] == configuration.id &&
                row["boundary"] == boundary && row["outcome"] == outcome
        end
        require(length(matches) == 1,
                "expected one RK / $(configuration.id) / $boundary / $outcome row")
        length(matches) == 1 || continue
        row = only(matches)
        expected_state, _ = matrix_support(configuration, boundary, outcome)
        if expected_state != "supported"
            require(get(row, "state", "") == expected_state,
                    "deliberate Reactant AD support state drift")
            require(!haskey(row, "result"), "non-measured Reactant AD cell has data")
            require(!isempty(get(row, "reason", "")),
                    "non-measured Reactant AD cell lacks reason")
            continue
        end
        require(haskey(row, "native_control"),
                "supported Reactant AD cell lacks native control")
        haskey(row, "native_control") && checked_measurement(
            row["native_control"], "native AD $(configuration.id) / $boundary / $outcome")
        require(Float64(get(row, "ad_preparation_seconds", -1.0)) >= 0,
                "native AD preparation missing")
        require(Float64(get(row, "reactant_transfer_seconds", -1.0)) >= 0,
                "Reactant AD transfer missing")
        require(Float64(get(row, "reactant_ad_compile_seconds", -1.0)) >= 0,
                "Reactant AD compile attempt missing")
        state = get(row, "state", "")
        require(state in ("supported", "unsupported_runtime"),
                "runtime Reactant AD state invalid")
        if state == "supported"
            require(haskey(row, "result"),
                    "supported Reactant AD cell lacks result")
            haskey(row, "result") && checked_measurement(
                row["result"], "Reactant AD $(configuration.id) / $boundary / $outcome")
            require(Float64(get(row, "reactant_first_execution_seconds", -1.0)) >= 0,
                    "Reactant AD first execution missing")
            require(Float64(get(row, "max_abs_error", Inf)) <= 1e-10,
                    "native/Reactant gradient parity failed")
            require(Float64(get(row, "value_abs_error", Inf)) <= 1e-10,
                    "native/Reactant value parity failed")
        else
            require(!haskey(row, "result"),
                    "runtime-unsupported Reactant AD cell has data")
            require(!isempty(get(row, "reason", "")),
                    "runtime-unsupported Reactant AD cell lacks diagnostic")
        end
    end
    errors
end

function main(path)
    errors = validate_eight_schools_reactant_ad_receipt(path)
    isempty(errors) &&
        (println("VALIDATE OK — eight-schools-reactant-ad-v2 matrix accepted"); return 0)
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_eight_schools_reactant_ad.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
