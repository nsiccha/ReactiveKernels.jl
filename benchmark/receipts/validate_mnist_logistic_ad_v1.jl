#!/usr/bin/env julia

module MNISTLogisticADV1Validation

import SHA
import Statistics
import TOML

include(joinpath(@__DIR__, "_validate_mnist_dataset_profile.jl"))

const EXPECTED_MNIST_AD_BOUNDARIES =
    ("packed_unconstrained", "structured_parameters")
const EXPECTED_MNIST_AD_OUTCOMES =
    ("joint", "prior", "likelihood", "pointwise")
const EXPECTED_MNIST_AD_SUPPORTED = Set((
    ("packed_unconstrained", "joint"),
    ("packed_unconstrained", "prior"),
    ("packed_unconstrained", "likelihood"),
))

_mnist_ad_median(values) = Statistics.median(Float64.(values))
_mnist_ad_text_sha256(path) = bytes2hex(SHA.sha256(
    replace(read(path, String), "\r\n" => "\n", "\r" => "\n")))
_mnist_ad_text_sha256_value(text) = bytes2hex(SHA.sha256(
    replace(String(text), "\r\n" => "\n", "\r" => "\n")))

function validate_mnist_logistic_ad_receipt(path::AbstractString;
        root::AbstractString = normpath(joinpath(dirname(path), "..", "..")))
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "mnist-logistic-ad-v1",
            "schema must be mnist-logistic-ad-v1")
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
    primal = nothing
    require(isfile(primal_path), "matched primal receipt is missing")
    if isfile(primal_path)
        primal = TOML.parsefile(primal_path)
        require(get(primal, "schema", "") == "mnist-logistic-v1",
                "matched receipt must be mnist-logistic-v1")
        require(get(pins, "primal_receipt_sha256", "") ==
                _mnist_ad_text_sha256(primal_path),
                "matched primal receipt digest mismatch")
        haskey(primal, "pins") && require(
            get(pins, "primal_receipt_reactivekernels_sha", "") ==
            get(primal["pins"], "reactivekernels_sha", ""),
            "matched primal receipt code pin mismatch")
    end

    for (key, expected_paths) in (
            "model_source" => (
                "packages/ReactiveKernelsPPLExamples/src/mnist_logistic.jl",),
            "primal_comparator_source" => (
                "benchmark/mnist_logistic_comparison_body.jl",
                "benchmark/mnist_logistic_wren_pca40_comparison_body.jl",
            ),
        )
        haskey(pins, key) || continue
        pin = pins[key]
        published_path = get(pin, "path", "")
        require(published_path in expected_paths,
                "$key path mismatch")
        require(occursin(r"^[0-9a-f]{40}$", get(pin, "commit", "")),
                "$key published commit missing")
        require(occursin(r"^[0-9a-f]{40}$", get(pin, "git_blob", "")),
                "$key published blob missing")
        require(occursin(r"^[0-9a-f]{64}$", get(pin, "text_sha256", "")),
                "$key published text digest missing")
        published_commit = get(pin, "commit", "")
        if occursin(r"^[0-9a-f]{40}$", published_commit)
            published_source = string(published_commit, ":", published_path)
            published_blob = try
                readchomp(pipeline(
                    `git -C $root rev-parse --verify $published_source`;
                    stderr = devnull))
            catch
                nothing
            end
            require(published_blob !== nothing,
                    "$key published source commit is unavailable")
            if published_blob !== nothing
                require(get(pin, "git_blob", "") == published_blob,
                        "$key published blob mismatch")
                published_text = try
                    read(pipeline(`git -C $root cat-file blob $published_blob`;
                                  stderr = devnull), String)
                catch
                    nothing
                end
                require(published_text !== nothing,
                        "$key published source blob is unavailable")
                published_text === nothing || require(
                    get(pin, "text_sha256", "") ==
                    _mnist_ad_text_sha256_value(published_text),
                    "$key published text digest mismatch")
            end
        end
        current = get(pin, "current", Dict{String,Any}())
        require(get(current, "path", "") == published_path,
                "$key current path mismatch")
        current_blob = get(current, "git_blob", "")
        require(occursin(r"^[0-9a-f]{40}$", current_blob),
                "$key current blob missing")
        current_text = try
            read(`git -C $root cat-file blob $current_blob`, String)
        catch
            nothing
        end
        require(current_text !== nothing,
                "$key current source blob is unavailable")
        current_text === nothing || require(
            get(current, "text_sha256", "") ==
            _mnist_ad_text_sha256_value(current_text),
            "$key current text digest mismatch")
    end
    comparator = get(pins, "primal_comparator_source", Dict{String,Any}())
    comparator_delta = get(comparator, "current_delta", "")
    require(comparator_delta in (
                "none",
                "documentation-only baseline markers plus terminal definition-only include guard",
            ),
            "comparator reuse delta is not narrowly attested")
    comparator_delta == "none" && require(
        get(comparator, "git_blob", "") ==
            get(get(comparator, "current", Dict{String,Any}()), "git_blob", ""),
        "an exact comparator pin must reuse the published source blob")

    protocol = receipt["protocol"]
    require(Tuple(get(protocol, "input_boundaries", String[])) ==
            EXPECTED_MNIST_AD_BOUNDARIES,
            "input-boundary matrix mismatch")
    require(Tuple(get(protocol, "outcomes", String[])) ==
            EXPECTED_MNIST_AD_OUTCOMES,
            "outcome matrix mismatch")
    require(get(protocol, "source_reused", false) == true,
            "benchmark must reuse the published model source")
    require(get(protocol, "matrix_source", "") ==
            get(pins, "primal_receipt_path", ""),
            "benchmark must name the published primal matrix")
    primal === nothing || _validate_mnist_dataset_profile!(
        require, protocol, primal["protocol"])
    require(Int(get(protocol, "num_classes", 0)) == 10,
            "MNIST must retain ten classes")
    expected_active_parameter_count =
        9 * Int(get(protocol, "num_features", 0)) + 9
    require(Int(get(protocol, "active_parameter_count", 0)) ==
            expected_active_parameter_count,
            "packed active-vector length does not match the feature count")
    require(get(protocol, "pointwise_jacobian_or_vjp_invented", true) == false,
            "pointwise must remain unsupported without a matched public contract")
    require(get(protocol, "structured_multi_active_boundary_invented", true) == false,
            "structured two-active-port AD must not be invented")
    require(get(protocol, "setup_in_timed_region", true) == false,
            "setup must remain outside steady-state timing")
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
            length(EXPECTED_MNIST_AD_BOUNDARIES) *
            length(EXPECTED_MNIST_AD_OUTCOMES),
            "receipt must contain the complete 2×4 matrix")
    supported = Set{Tuple{String,String}}()
    for boundary in EXPECTED_MNIST_AD_BOUNDARIES,
        outcome in EXPECTED_MNIST_AD_OUTCOMES
        matches = filter(rows) do row
            row["boundary"] == boundary && row["outcome"] == outcome
        end
        require(length(matches) == 1,
                "expected one row for $boundary / $outcome")
        length(matches) == 1 || continue
        row = only(matches)
        expected_support = (boundary, outcome) in EXPECTED_MNIST_AD_SUPPORTED
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
        require(get(row, "active_port", "") == "unconstrained",
                "$boundary / $outcome active port mismatch")
        require(Int(get(row, "analytic_gradient_length", 0)) ==
                expected_active_parameter_count,
                "$boundary / $outcome analytic oracle length mismatch")
        for implementation in ("rk_native", "manual_enzyme", "turing_enzyme")
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
            isempty(times) || require(Float64(get(result, "min_ns", NaN)) ==
                                      minimum(Float64.(times)),
                    "$boundary / $outcome $implementation minimum mismatch")
            isempty(times) || require(isapprox(
                Float64(get(result, "median_ns", NaN)),
                _mnist_ad_median(times); rtol = 1e-12),
                "$boundary / $outcome $implementation timing median mismatch")
            isempty(bytes) || require(Int(get(result, "median_bytes", -1)) ==
                round(Int, _mnist_ad_median(bytes)),
                "$boundary / $outcome $implementation byte median mismatch")
            isempty(allocs) || require(Int(get(result, "median_allocs", -1)) ==
                round(Int, _mnist_ad_median(allocs)),
                "$boundary / $outcome $implementation allocation median mismatch")
            require(Int(get(result, "gradient_length", 0)) ==
                    expected_active_parameter_count,
                    "$boundary / $outcome $implementation gradient length mismatch")
            absolute_error = Float64(get(
                result, "gradient_max_abs_error", Inf))
            relative_error = Float64(get(
                result, "gradient_max_rel_error", Inf))
            require(absolute_error <= absolute_tolerance ||
                    relative_error <= relative_tolerance,
                    "$boundary / $outcome $implementation exceeds parity tolerance")
            require(Float64(get(result, "value_abs_error", Inf)) <= 1e-7,
                    "$boundary / $outcome $implementation value parity failed")
            caller_owned = implementation != "turing_enzyme"
            require(get(result, "caller_owned_gradient", !caller_owned) == caller_owned,
                    "$boundary / $outcome $implementation ownership mismatch")
        end
    end
    require(supported == EXPECTED_MNIST_AD_SUPPORTED,
            "supported packed scalar-cell inventory drifted")
    errors
end

function main(path)
    errors = validate_mnist_logistic_ad_receipt(path)
    if isempty(errors)
        println("VALIDATE OK — mnist-logistic-ad-v1: exact 2×4 matrix, " *
                "three packed scalar gradients accepted")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_mnist_logistic_ad.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end

end # module MNISTLogisticADV1Validation
