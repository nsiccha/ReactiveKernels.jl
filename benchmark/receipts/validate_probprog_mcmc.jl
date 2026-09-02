#!/usr/bin/env julia

import TOML

const PROBPROG_MCMC_MODELS = ("eight_schools", "mnist_logistic_wren_pca40")
const PROBPROG_MCMC_HARNESSES = ("probprog_nuts", "advancedhmc_nuts", "turing_nuts")
const PROBPROG_MCMC_PUBLICATION_WARMUP = 1000
const PROBPROG_MCMC_PUBLICATION_SAMPLES = 1000

function validate_probprog_mcmc_receipt(path::AbstractString)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "probprog-mcmc-v1",
            "schema must be probprog-mcmc-v1")
    for section in ("pins", "environment", "protocol", "datasets", "measurements")
        require(haskey(receipt, section), "missing $section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    require(get(pins, "reactivekernels_dirty", true) == false,
            "receipt must come from a clean candidate")
    require(occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")),
            "candidate pin must be a full commit SHA")
    for pin in ("reactant_version", "advancedhmc_version", "turing_version",
                "dynamicppl_version", "mcmcdiagnostictools_version",
                "enzyme_version", "julia_version")
        require(!isempty(get(pins, pin, "")), "missing pin $pin")
    end
    sources = get(pins, "source_text_sha256", Dict{String,Any}())
    for model in PROBPROG_MCMC_MODELS
        require(occursin(r"^[0-9a-f]{64}$", get(sources, model, "")),
                "missing source authority digest for $model")
    end

    protocol = receipt["protocol"]
    require(Tuple(get(protocol, "models", String[])) == PROBPROG_MCMC_MODELS,
            "model inventory mismatch")
    require(Tuple(get(protocol, "harnesses", String[])) ==
            PROBPROG_MCMC_HARNESSES, "harness inventory mismatch")
    require(get(protocol, "num_warmup", 0) >= PROBPROG_MCMC_PUBLICATION_WARMUP,
            "publication receipts require at least " *
            "$PROBPROG_MCMC_PUBLICATION_WARMUP warmup transitions")
    require(get(protocol, "num_samples", 0) >= PROBPROG_MCMC_PUBLICATION_SAMPLES,
            "publication receipts require at least " *
            "$PROBPROG_MCMC_PUBLICATION_SAMPLES retained draws")
    require(get(protocol, "max_tree_depth", 0) == 10,
            "the comparison protocol fixes max_tree_depth = 10")
    require(get(protocol, "matched_initial_position", false) == true,
            "the comparison protocol requires matched initial positions")
    for note in ("target_accept", "metric", "divergence_threshold", "timing",
                 "density_parity_gate", "moment_gate", "ess")
        require(!isempty(get(protocol, note, "")), "missing protocol note $note")
    end

    datasets = receipt["datasets"]
    for model in PROBPROG_MCMC_MODELS
        require(haskey(datasets, model), "missing dataset record for $model")
    end
    if haskey(datasets, "mnist_logistic_wren_pca40")
        mnist = datasets["mnist_logistic_wren_pca40"]
        require(get(mnist, "dataset_profile", "") == "wren-pca40",
                "MNIST sampling workload must be the wren-pca40 profile")
        require(get(mnist, "num_observations", 0) == 1000,
                "the wren-pca40 sampling workload is the first 1000 images")
        require(get(mnist, "num_features", 0) == 40,
                "the wren-pca40 sampling workload has 40 PCA components")
    end

    measurements = receipt["measurements"]
    seen = Set{Tuple{String,String}}()
    for row in measurements
        model = get(row, "model", "")
        harness = get(row, "harness", "")
        key = (model, harness)
        require(model in PROBPROG_MCMC_MODELS, "unknown model $model")
        require(harness in PROBPROG_MCMC_HARNESSES, "unknown harness $harness")
        require(!(key in seen), "duplicate measurement for $key")
        push!(seen, key)
        state = get(row, "state", "")
        require(state in ("measured", "unsupported_runtime"),
                "$key has unknown state $state")
        if state == "unsupported_runtime"
            require(harness == "probprog_nuts",
                    "$key: only the ProbProg compile boundary may degrade to " *
                    "an unsupported cell")
            require(!isempty(get(row, "reason", "")),
                    "$key unsupported cell lacks its compiler diagnostic")
            continue
        end
        require(get(row, "sampling_seconds", 0.0) > 0,
                "$key lacks a positive sampling time")
        require(get(row, "compile_or_warm_start_seconds", 0.0) > 0,
                "$key lacks a positive compile/warm-start time")
        divergences = get(row, "divergences", -1)
        require(divergences isa Integer && divergences >= 0,
                "$key lacks a divergence count")
        min_ess = get(row, "min_ess", 0.0)
        median_ess = get(row, "median_ess", 0.0)
        require(min_ess > 0, "$key lacks a positive minimum ESS")
        require(median_ess >= min_ess,
                "$key median ESS must dominate the minimum")
        if harness == "probprog_nuts"
            gap = get(row, "density_parity_max_abs", 1.0)
            require(gap <= 1e-8,
                    "$key per-draw density parity exceeds the 1e-8 gate")
        end
        if model == "eight_schools"
            for moment in ("posterior_mean_mu", "posterior_sd_mu",
                           "posterior_mean_tau")
                require(haskey(row, moment), "$key lacks $moment")
            end
        else
            for moment in ("posterior_mean_first_coefficient",
                           "posterior_sd_first_coefficient")
                require(haskey(row, moment), "$key lacks $moment")
            end
        end
    end
    for model in PROBPROG_MCMC_MODELS, harness in PROBPROG_MCMC_HARNESSES
        require((model, harness) in seen, "missing measurement for $((model, harness))")
    end
    errors
end

function main(path)
    errors = validate_probprog_mcmc_receipt(path)
    isempty(errors) &&
        (println("VALIDATE OK — probprog-mcmc-v1 comparison accepted"); return 0)
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: validate_probprog_mcmc.jl <receipt.toml>"); exit(2))
    exit(main(only(ARGS)))
end
