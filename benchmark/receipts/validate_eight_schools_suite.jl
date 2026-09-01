#!/usr/bin/env julia

import TOML

include(joinpath(dirname(@__DIR__), "eight_schools_matrix_spec.jl"))
using .EightSchoolsMatrixSpec

module PrimalValidation
include(joinpath(@__DIR__, "validate_eight_schools_primal.jl"))
end
module ADValidation
include(joinpath(@__DIR__, "validate_eight_schools_ad.jl"))
end
module ReactantValidation
include(joinpath(@__DIR__, "validate_eight_schools_reactant.jl"))
end
module ReactantADValidation
include(joinpath(@__DIR__, "validate_eight_schools_reactant_ad.jl"))
end

const DEFAULT_EIGHT_SCHOOLS_SUITE_PATHS = (
    primal = joinpath(@__DIR__, "eight-schools-primal-v2.toml"),
    ad = joinpath(@__DIR__, "eight-schools-ad-v2.toml"),
    reactant = joinpath(@__DIR__, "eight-schools-reactant-v2.toml"),
    reactant_ad = joinpath(@__DIR__, "eight-schools-reactant-ad-v2.toml"),
)

function validate_eight_schools_suite(paths = DEFAULT_EIGHT_SCHOOLS_SUITE_PATHS)
    errors = String[]
    append!(errors, ("primal: " * error for error in
        PrimalValidation.validate_eight_schools_primal_receipt(paths.primal)))
    append!(errors, ("AD: " * error for error in
        ADValidation.validate_eight_schools_ad_receipt(
            paths.ad; primal_path = paths.primal)))
    append!(errors, ("Reactant: " * error for error in
        ReactantValidation.validate_eight_schools_reactant_receipt(
            paths.reactant; primal_path = paths.primal)))
    append!(errors, ("Reactant AD: " * error for error in
        ReactantADValidation.validate_eight_schools_reactant_ad_receipt(
            paths.reactant_ad; ad_path = paths.ad)))
    isempty(errors) || return errors

    receipts = (
        primal = TOML.parsefile(paths.primal),
        ad = TOML.parsefile(paths.ad),
        reactant = TOML.parsefile(paths.reactant),
        reactant_ad = TOML.parsefile(paths.reactant_ad),
    )
    pins = Set(receipt["pins"]["reactivekernels_sha"] for receipt in receipts)
    length(pins) == 1 || push!(errors,
        "all four receipts must pin the same ReactiveKernels candidate")

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
    accounted = Set{String}()
    for cell in headline_cells()
        receipt = configuration_receipt[cell.configuration]
        matches = filter(receipt["measurements"]) do row
            row["provider"] == "rk" && row["model"] == cell.model &&
                row["configuration"] == cell.configuration &&
                row["boundary"] == cell.boundary && row["outcome"] == cell.outcome
        end
        if length(matches) != 1
            push!(errors,
                "headline cell $(cell.configuration) is not accounted for exactly once")
            continue
        end
        state = get(only(matches), "state", "")
        state == "supported" || push!(errors,
            "headline cell $(cell.configuration) must work; recorded state is $state")
        push!(accounted, cell.configuration)
    end
    length(accounted) == length(headline_cells()) || push!(errors,
        "the packed-unconstrained → full-joint headline matrix is incomplete")

    any(configuration -> configuration.allocation == "nonallocating" &&
        configuration.differentiation == "value_and_gradient",
        EIGHT_SCHOOLS_RK_CONFIGURATIONS) && push!(errors,
        "nonallocating AD must not appear without a public composition surface")
    any(configuration -> configuration.allocation == "nonallocating" &&
        configuration.compiler == "reactant",
        EIGHT_SCHOOLS_RK_CONFIGURATIONS) && push!(errors,
        "nonallocating Reactant must not appear without a public composition surface")
    errors
end

function main()
    errors = validate_eight_schools_suite()
    isempty(errors) &&
        (println("VALIDATE OK — complete Eight Schools suite: 10/10 " *
                 "packed-unconstrained → joint modifier cells accounted for"); return 0)
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) ||
        (println("usage: validate_eight_schools_suite.jl"); exit(2))
    exit(main())
end
