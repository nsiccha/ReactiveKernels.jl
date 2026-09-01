#!/usr/bin/env julia

import Statistics
import TOML

isdefined(@__MODULE__, :EightSchoolsMatrixSpec) ||
    include(joinpath(dirname(@__DIR__), "eight_schools_matrix_spec.jl"))
using .EightSchoolsMatrixSpec

const EXPECTED_EIGHT_SCHOOLS_PRIMAL_CONFIGURATIONS = Tuple(
    configuration for configuration in EIGHT_SCHOOLS_RK_CONFIGURATIONS
    if configuration.differentiation == "primal" &&
       configuration.compiler == "native"
)

function validate_eight_schools_primal_receipt(path::AbstractString)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "eight-schools-primal-v2",
            "schema must be eight-schools-primal-v2")
    for section in ("pins", "environment", "protocol", "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    for key in (
        "reactivekernels_sha", "reactivekernels_version",
        "reactivekernelspplexamples_version",
        "reactivekernelsdistributionkernels_version", "turing_version",
        "dynamicppl_version", "distributions_version", "benchmarktools_version",
        "mutatingfunctions_version", "mutatingfunctions_rev", "julia_version",
    )
        require(haskey(pins, key) && !isempty(string(pins[key])),
                "pins.$key missing")
    end
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")),
            "ReactiveKernels receipt pin must be a full commit SHA")
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a clean ReactiveKernels tree")

    protocol = receipt["protocol"]
    require(Tuple(get(protocol, "models", String[])) == EIGHT_SCHOOLS_MODELS,
            "model inventory mismatch")
    require(Tuple(get(protocol, "input_boundaries", String[])) ==
            EIGHT_SCHOOLS_BOUNDARIES, "input-boundary matrix mismatch")
    require(Tuple(get(protocol, "outcomes", String[])) == EIGHT_SCHOOLS_OUTCOMES,
            "outcome matrix mismatch")
    require(Tuple(get(protocol, "rk_configurations", String[])) == Tuple(
            configuration.id for configuration in
            EXPECTED_EIGHT_SCHOOLS_PRIMAL_CONFIGURATIONS),
            "native primal configuration inventory mismatch")
    require(get(protocol, "matrix_layout", "") ==
            "long-form provider/model/configuration/boundary/outcome rows",
            "receipt must use the long-form capability matrix")
    require(Tuple(get(protocol, "bound_ports", String[])) ==
            ("observations", "observation_scales"),
            "bound-port inventory mismatch")
    require(Int(get(protocol, "rounds", 0)) >= 10,
            "published receipt must use at least ten rounds")
    for key in ("setup_in_timed_region", "preparation_in_timed_region")
        require(get(protocol, key, true) == false, "$key must be false")
    end
    require(get(protocol, "unsupported_cells_recorded", false) == true,
            "unsupported cells must be recorded")
    require(get(protocol, "gradients_included", true) == false,
            "primal receipt must not contain gradients")

    function checked_result(row, label)
        haskey(row, "result") || (require(false, "$label lacks a result"); return)
        result = row["result"]
        times = Float64.(get(result, "times_ns", Float64[]))
        bytes = Int.(get(result, "bytes", Int[]))
        allocs = Int.(get(result, "allocs", Int[]))
        require(length(times) >= 10, "$label needs ten timing rounds")
        require(length(bytes) == length(times) && length(allocs) == length(times),
                "$label raw measurement lengths differ")
        isempty(times) && return
        require(all(>(0), times), "$label has non-positive timing")
        require(isapprox(Float64(get(result, "median_ns", NaN)),
                         Statistics.median(times); rtol = 1e-12),
                "$label timing median mismatch")
        require(Int(get(result, "median_bytes", -1)) ==
                Int(Statistics.median(bytes)), "$label byte median mismatch")
        require(Int(get(result, "median_allocs", -1)) ==
                Int(Statistics.median(allocs)), "$label allocation median mismatch")
    end

    rows = receipt["measurements"]
    expected_count =
        length(EXPECTED_EIGHT_SCHOOLS_PRIMAL_CONFIGURATIONS) *
            length(EIGHT_SCHOOLS_BOUNDARIES) * length(EIGHT_SCHOOLS_OUTCOMES) +
        2 * length(EIGHT_SCHOOLS_BOUNDARIES) * length(EIGHT_SCHOOLS_OUTCOMES)
    require(length(rows) == expected_count,
            "long-form primal matrix must contain $expected_count rows")
    keys = [(row["provider"], row["model"], row["configuration"],
             row["boundary"], row["outcome"]) for row in rows]
    require(length(Set(keys)) == length(keys), "matrix contains duplicate cells")

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

    for configuration in EXPECTED_EIGHT_SCHOOLS_PRIMAL_CONFIGURATIONS,
        boundary in EIGHT_SCHOOLS_BOUNDARIES, outcome in EIGHT_SCHOOLS_OUTCOMES
        row = only_cell("rk", configuration.id, boundary, outcome)
        row === nothing && continue
        expected_state, _ = matrix_support(configuration, boundary, outcome)
        require(get(row, "state", "") == expected_state,
                "RK support drift for $(configuration.id) / $boundary / $outcome")
        if expected_state == "supported"
            checked_result(row, "RK $(configuration.id) / $boundary / $outcome")
        else
            require(!haskey(row, "result"), "non-measured RK cell has a result")
            require(!isempty(get(row, "reason", "")),
                    "non-measured RK cell lacks a reason")
        end
    end

    for boundary in EIGHT_SCHOOLS_BOUNDARIES, outcome in EIGHT_SCHOOLS_OUTCOMES
        manual = only_cell("manual_julia", "manual_primal", boundary, outcome)
        manual_state = boundary == "minimal_likelihood" &&
            outcome in ("joint", "prior") ? "unsupported" : "supported"
        manual === nothing || require(get(manual, "state", "") == manual_state,
            "manual support drift for $boundary / $outcome")
        manual === nothing || (manual_state == "supported" ?
            checked_result(manual, "manual / $boundary / $outcome") :
            require(!haskey(manual, "result"),
                    "unsupported manual cell has a result"))

        turing = only_cell("turing", "turing_primal", boundary, outcome)
        turing_state = boundary == "minimal_likelihood" ?
            "unsupported" : "supported"
        turing === nothing || require(get(turing, "state", "") == turing_state,
            "Turing support drift for $boundary / $outcome")
        turing === nothing || (turing_state == "supported" ?
            checked_result(turing, "Turing / $boundary / $outcome") :
            require(!haskey(turing, "result"),
                    "unsupported Turing cell has a result"))
    end
    errors
end

function main(path)
    errors = validate_eight_schools_primal_receipt(path)
    isempty(errors) &&
        (println("VALIDATE OK — eight-schools-primal-v2 matrix accepted"); return 0)
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_eight_schools_primal.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
