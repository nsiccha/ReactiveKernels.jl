#!/usr/bin/env julia

import SHA
import TOML

const PRACTICALBAYES_SCHEMA = "practicalbayes-comparison-v1"
const PRACTICALBAYES_REVISION =
    "c6b340baef4f4a9e3d26cd0ea5082a2baf26dcf9"
const PRACTICALBAYES_EVAL_SIZES = (16, 256, 4096)
const PRACTICALBAYES_EVAL_MODES = ("primal", "gradient", "gq")
const PRACTICALBAYES_MCMC_MODELS =
    ("eight_schools", "mnist_logistic_wren_pca40")

function _body_sha256(root)
    bytes2hex(SHA.sha256(read(joinpath(
        root, "benchmark", "practicalbayes_comparison_body.jl"))))
end

function _model_supported(workload, model, boundary, outcome, differentiation)
    if workload == "eight_schools"
        return (boundary == "packed_unconstrained" && outcome == "joint") ||
            (boundary == "constrained_parameters" &&
             differentiation == "primal")
    end
    (boundary == "packed_unconstrained" && outcome == "joint") ||
        (boundary == "structured_parameters" &&
         differentiation == "primal" &&
         !(model == "vcat_free" && outcome == "pointwise"))
end

function validate_practicalbayes_receipt(path::AbstractString;
                                         root = normpath(joinpath(
                                             dirname(@__DIR__), "..")))
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == PRACTICALBAYES_SCHEMA,
            "schema must be $PRACTICALBAYES_SCHEMA")
    for section in (
            "pins", "environment", "compatibility", "protocol",
            "preparations", "datasets", "model_measurements",
            "eval_measurements", "sampling_measurements")
        require(haskey(receipt, section), "missing $section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a clean candidate")
    require(occursin(r"^[0-9a-f]{40}$",
                     get(pins, "reactivekernels_sha", "")),
            "ReactiveKernels pin must be a full commit SHA")
    require(get(pins, "practicalbayes_revision", "") ==
            PRACTICALBAYES_REVISION,
            "PracticalBayes requested revision mismatch")
    require(get(pins, "practicalbayes_git_revision", "") ==
            PRACTICALBAYES_REVISION,
            "PracticalBayes resolved Git revision mismatch")
    require(get(pins, "practicalbayes_version", "") == "0.1.0",
            "PracticalBayes version must be 0.1.0")
    require(get(pins, "benchmark_body_sha256", "") == _body_sha256(root),
            "benchmark body digest does not match the executed source")
    for pin in (
            "adtypes_version", "advancedhmc_version", "benchmarktools_version",
            "distributions_version", "enzyme_version",
            "logdensityproblems_version", "mcmcdiagnostictools_version",
            "mldatasets_version", "nnlib_version", "julia_version")
        require(!isempty(get(pins, pin, "")), "missing pin $pin")
    end

    compatibility = receipt["compatibility"]
    require(get(compatibility, "separate_environment", false) == true,
            "PracticalBayes must run in its own external environment")
    require(get(compatibility, "coexists_with_pinned_turing_environment", true) == false,
            "the pinned Turing incompatibility must remain explicit")
    for key in ("solver_diagnostic", "reproduction", "source_policy")
        require(!isempty(get(compatibility, key, "")),
                "compatibility.$key must be recorded")
    end
    require(get(compatibility, "upstream_license_file_at_pin", "") == "absent",
            "the absent upstream license file must remain explicit")

    protocol = receipt["protocol"]
    require(Tuple(get(protocol, "sections", String[])) ==
            ("models", "eval", "sampling"),
            "publication receipt must contain all three sections")
    require(get(protocol, "setup_in_timed_region", true) == false,
            "setup entered steady-state timings")
    require(get(protocol, "preparation_in_timed_region", true) == false,
            "preparation entered steady-state timings")
    require(get(protocol, "warmup_in_timed_region", true) == false,
            "benchmark warmup entered steady-state timings")
    require(get(protocol, "unsupported_cells_recorded", false) == true,
            "unsupported cells must be explicit")
    require(get(protocol, "deterministic_seeds", false) == true,
            "deterministic seeds must be recorded")
    require(Int(get(protocol, "mnist_full_observations", 0)) == 60000,
            "model matrix must use all 60000 raw MNIST images")
    require(Int(get(protocol, "rounds", 0)) >= 10,
            "publication receipt requires at least ten timing rounds")
    require(Int(get(protocol, "samples_per_round", 0)) >= 20,
            "publication receipt requires at least 20 samples per round")
    require(Int(get(protocol, "mcmc_num_warmup", 0)) >= 1000,
            "publication receipt requires at least 1000 warmup transitions")
    require(Int(get(protocol, "mcmc_num_samples", 0)) >= 1000,
            "publication receipt requires at least 1000 retained draws")
    require(Int(get(protocol, "mcmc_max_tree_depth", 0)) == 10,
            "PracticalBayes NUTS depth must be 10")
    require(get(protocol, "mcmc_metric", "") == "diagonal",
            "PracticalBayes NUTS metric must be diagonal")
    require(Float64(get(protocol, "mcmc_target_accept", 0.0)) == 0.8,
            "PracticalBayes NUTS target acceptance must be 0.8")
    require(Float64(get(protocol, "mcmc_divergence_threshold", 0.0)) == 1000.0,
            "PracticalBayes NUTS divergence threshold must be 1000")

    model_rows = receipt["model_measurements"]
    expected_model_keys = Tuple{String,String,String,String,String}[]
    for boundary in ("packed_unconstrained", "constrained_parameters",
                     "minimal_likelihood"),
        outcome in ("joint", "prior", "likelihood", "pointwise"),
        differentiation in ("primal", "value_and_gradient")
        push!(expected_model_keys, (
            "eight_schools", "centered", boundary, outcome, differentiation))
    end
    for model in ("idiomatic", "vcat_free"),
        boundary in ("packed_unconstrained", "structured_parameters"),
        outcome in ("joint", "prior", "likelihood", "pointwise"),
        differentiation in ("primal", "value_and_gradient")
        push!(expected_model_keys, (
            "mnist_logistic", model, boundary, outcome, differentiation))
    end
    require(length(model_rows) == length(expected_model_keys),
            "model matrix must contain $(length(expected_model_keys)) rows")
    seen_model = Set{Tuple{String,String,String,String,String}}()
    for row in model_rows
        key = (String(get(row, "workload", "")),
               String(get(row, "model", "")),
               String(get(row, "boundary", "")),
               String(get(row, "outcome", "")),
               String(get(row, "differentiation", "")))
        require(key in expected_model_keys, "unknown model row $key")
        require(!(key in seen_model), "duplicate model row $key")
        push!(seen_model, key)
        require(get(row, "provider", "") == "practical_bayes",
                "$key has the wrong provider")
        expected_configuration = if key[1] == "eight_schools"
            key[5] == "primal" ?
                "practicalbayes_primal" : "practicalbayes_ad"
        else
            "practicalbayes_$(key[2])_" *
                (key[5] == "primal" ? "primal" : "ad")
        end
        require(get(row, "configuration", "") == expected_configuration,
                "$key has the wrong configuration")
        supported = _model_supported(key...)
        state = get(row, "state", "")
        require(state == (supported ? "supported" : "unsupported"),
                "$key has state $state, expected $(supported ? "supported" : "unsupported")")
        if supported
            result = get(row, "result", Dict{String,Any}())
            require(Float64(get(result, "min_ns", 0.0)) > 0,
                    "$key lacks a positive runtime")
            require(length(get(result, "times_ns", Any[])) >= 10,
                    "$key lacks timing rounds")
            require(Int(get(result, "median_bytes", -1)) >= 0,
                    "$key lacks allocation bytes")
            require(Int(get(result, "median_allocs", -1)) >= 0,
                    "$key lacks allocation count")
            require(Float64(get(row, "parity_max_abs", Inf)) <= 1e-8,
                    "$key failed the parity gate")
        else
            require(!isempty(get(row, "reason", "")),
                    "$key unsupported cell lacks a reason")
            if key == ("mnist_logistic", "vcat_free",
                       "structured_parameters", "pointwise", "primal")
                require(!isempty(get(row, "diagnostic", "")),
                        "$key must retain the public pointwise diagnostic")
            end
        end
    end
    require(length(seen_model) == length(expected_model_keys),
            "model matrix inventory is incomplete")

    eval_rows = receipt["eval_measurements"]
    require(length(eval_rows) == length(PRACTICALBAYES_EVAL_SIZES) *
            length(PRACTICALBAYES_EVAL_MODES) * 2,
            "evaluation matrix must contain 18 rows")
    seen_eval = Set{Tuple{Int,String,String}}()
    for row in eval_rows
        key = (Int(get(row, "size", 0)), String(get(row, "mode", "")),
               String(get(row, "variant", "")))
        require(key[1] in PRACTICALBAYES_EVAL_SIZES &&
                key[2] in PRACTICALBAYES_EVAL_MODES &&
                key[3] in ("native", "reactant"),
                "unknown evaluation row $key")
        require(!(key in seen_eval), "duplicate evaluation row $key")
        push!(seen_eval, key)
        if key[3] == "native"
            require(get(row, "state", "") == "supported",
                    "$key native cell must be supported")
            result = get(row, "result", Dict{String,Any}())
            require(Float64(get(result, "min_ns", 0.0)) > 0,
                    "$key lacks a positive runtime")
            require(Int(get(result, "median_bytes", -1)) >= 0,
                    "$key lacks allocation bytes")
            require(Float64(get(row, "parity_max_abs", Inf)) <= 1e-8,
                    "$key failed parity")
        else
            require(get(row, "state", "") == "unsupported",
                    "$key Reactant cell must be unsupported")
            require(!isempty(get(row, "reason", "")),
                    "$key Reactant cell lacks a reason")
        end
    end

    sampling_rows = receipt["sampling_measurements"]
    require(length(sampling_rows) == length(PRACTICALBAYES_MCMC_MODELS),
            "sampling matrix must contain two rows")
    seen_sampling = Set{String}()
    for row in sampling_rows
        model = String(get(row, "model", ""))
        require(model in PRACTICALBAYES_MCMC_MODELS,
                "unknown sampling model $model")
        require(!(model in seen_sampling), "duplicate sampling row $model")
        push!(seen_sampling, model)
        require(get(row, "harness", "") == "practicalbayes_nuts",
                "$model uses the wrong harness")
        require(get(row, "state", "") == "measured",
                "$model sampling row must be measured")
        for key in ("compile_or_warm_start_seconds", "sampling_seconds",
                    "min_ess", "median_ess")
            require(Float64(get(row, key, 0.0)) > 0,
                    "$model lacks positive $key")
        end
        require(Int(get(row, "divergences", -1)) >= 0,
                "$model lacks a divergence count")
        require(Float64(get(row, "density_parity_max_abs", Inf)) <= 1e-8,
                "$model sampling density parity failed")
    end

    datasets = receipt["datasets"]
    require(get(get(datasets, "mnist_logistic_full_raw", Dict()),
                "num_observations", 0) == 60000,
            "full-raw MNIST dataset record must contain 60000 observations")
    wren = get(datasets, "mnist_logistic_wren_pca40", Dict())
    require(get(wren, "dataset_profile", "") == "wren-pca40",
            "sampling dataset must be Wren PCA-40")
    require(get(wren, "num_observations", 0) == 1000 &&
            get(wren, "num_features", 0) == 40,
            "sampling dataset must be 1000 × 40")
    errors
end

function main(path)
    errors = validate_practicalbayes_receipt(path)
    isempty(errors) && begin
        println("VALIDATE OK — PracticalBayes model, evaluation, and NUTS matrices accepted")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || begin
        println("usage: validate_practicalbayes.jl <receipt.toml>")
        exit(2)
    end
    exit(main(only(ARGS)))
end
