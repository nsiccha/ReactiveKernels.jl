#!/usr/bin/env julia

import TOML

include(joinpath(dirname(@__DIR__), "mnist_logistic_matrix_spec.jl"))
using .MNISTLogisticMatrixSpec

module PrimalValidation
include(joinpath(@__DIR__, "validate_mnist_logistic.jl"))
end
module ADValidation
include(joinpath(@__DIR__, "validate_mnist_logistic_ad.jl"))
end
module ReactantValidation
include(joinpath(@__DIR__, "validate_mnist_reactant.jl"))
end
module ReactantADValidation
include(joinpath(@__DIR__, "validate_mnist_reactant_ad.jl"))
end

const DEFAULT_MNIST_SUITE_PATHS = (
    primal = joinpath(@__DIR__, "mnist-logistic-primal-v3.toml"),
    ad = joinpath(@__DIR__, "mnist-logistic-ad-v2.toml"),
    reactant = joinpath(@__DIR__, "mnist-reactant-v2.toml"),
    reactant_ad = joinpath(@__DIR__, "mnist-reactant-ad-v2.toml"),
)

function validate_mnist_logistic_suite(paths = DEFAULT_MNIST_SUITE_PATHS)
    errors = String[]
    append!(errors, ("primal: " * error for error in
        PrimalValidation.validate_mnist_logistic_receipt(paths.primal)))
    append!(errors, ("AD: " * error for error in
        ADValidation.validate_mnist_logistic_ad_receipt(paths.ad)))
    append!(errors, ("Reactant: " * error for error in
        ReactantValidation.validate_mnist_reactant_receipt(
            paths.reactant; primal_path = paths.primal)))
    append!(errors, ("Reactant AD: " * error for error in
        ReactantADValidation.validate_mnist_reactant_ad_receipt(
            paths.reactant_ad; ad_path = paths.ad)))
    isempty(errors) || return errors

    receipts = (
        primal = TOML.parsefile(paths.primal),
        ad = TOML.parsefile(paths.ad),
        reactant = TOML.parsefile(paths.reactant),
        reactant_ad = TOML.parsefile(paths.reactant_ad),
    )
    pins = Set(
        receipt["pins"]["reactivekernels_sha"] for receipt in receipts)
    length(pins) == 1 || push!(errors,
        "all four receipts must pin the same ReactiveKernels candidate")
    observations = Set(
        Int(receipt["protocol"]["num_observations"]) for receipt in receipts)
    observations == Set((60000,)) || push!(errors,
        "all four receipts must use the full 60000-image train split")

    configuration_receipt = Dict(
        "primal_native" => receipts.primal,
        "primal_native_bound" => receipts.primal,
        "primal_nonallocating" => receipts.primal,
        "primal_nonallocating_bound" => receipts.primal,
        "primal_reactant" => receipts.reactant,
        "primal_reactant_bound" => receipts.reactant,
        "ad_native" => receipts.ad,
        "ad_native_bound" => receipts.ad,
        "ad_reactant" => receipts.reactant_ad,
        "ad_reactant_bound" => receipts.reactant_ad,
    )
    accounted = Set{Tuple{String,String}}()
    for cell in headline_cells()
        receipt = configuration_receipt[cell.configuration]
        matches = filter(receipt["measurements"]) do row
            row["provider"] == "rk" && row["model"] == cell.model &&
                row["configuration"] == cell.configuration &&
                row["boundary"] == cell.boundary &&
                row["outcome"] == cell.outcome
        end
        if length(matches) != 1
            push!(errors,
                "headline cell $(cell.model) / $(cell.configuration) is not " *
                "accounted for exactly once")
            continue
        end
        state = get(only(matches), "state", "")
        state == "supported" || push!(errors,
            "headline cell $(cell.model) / $(cell.configuration) must work; " *
            "recorded state is $state")
        push!(accounted, (cell.model, cell.configuration))
    end
    length(accounted) == length(headline_cells()) || push!(errors,
        "the packed-unconstrained → full-joint headline matrix is incomplete")

    # The matrix deliberately stops at public combinations. These absences are
    # part of the contract, not forgotten benchmark columns.
    any(configuration -> configuration.allocation == "nonallocating" &&
        configuration.differentiation == "value_and_gradient",
        MNIST_RK_CONFIGURATIONS) && push!(errors,
        "nonallocating AD must not appear without a public composition surface")
    any(configuration -> configuration.allocation == "nonallocating" &&
        configuration.compiler == "reactant",
        MNIST_RK_CONFIGURATIONS) && push!(errors,
        "nonallocating Reactant must not appear without a public composition surface")
    errors
end

function main()
    errors = validate_mnist_logistic_suite()
    if isempty(errors)
        println("VALIDATE OK — complete MNIST suite: 20/20 two-model " *
                "packed-unconstrained → joint modifier cells accounted for")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) ||
        (println("usage: validate_mnist_logistic_suite.jl"); exit(2))
    exit(main())
end
