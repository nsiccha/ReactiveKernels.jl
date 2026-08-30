#!/usr/bin/env julia

import TOML

function validate_reactivehmc_hmc_receipt(path)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "reactivehmc-hmc-ca9-v1",
            "schema must be reactivehmc-hmc-ca9-v1")
    require(haskey(receipt, "pins"), "missing [pins]")
    require(haskey(receipt, "cases"), "missing [[cases]]")
    isempty(errors) || return errors

    pins = receipt["pins"]
    require(get(pins, "reactivehmc_revision", "") ==
            "ca9ea4ca41924bb0e1fadc01c717e1333916aba6",
            "wrong ReactiveHMC revision")
    require(get(pins, "hmc_sha256", "") ==
            "5d341facd929201ada08800e8d0194ec187f637ae036dd448461022a2bb577ea",
            "wrong hmc.jl digest")

    cases = receipt["cases"]
    require(length(cases) == 3, "expected accepted, rejected, and diverged cases")
    length(cases) == 3 || return errors
    by_name = Dict(case["name"] => case for case in cases)
    require(Set(keys(by_name)) == Set(("accepted", "rejected", "diverged")),
            "case names must be accepted, rejected, and diverged")

    for case in cases
        name = case["name"]
        require(case["normal_calls"] == 1,
                "$name: momentum refresh must consume exactly one normal vector")
        require(length(case["normal_draw"]) == length(case["initial_position"]),
                "$name: momentum and position dimensions differ")
        require(length(case["energy_errors"]) ==
                (case["diverged"] ? 1 : case["n_steps"]),
                "$name: statistics count does not match the physical control path")
        expected_events = case["diverged"] ? ["normal"] :
            ["normal", "exponential"]
        require(get(case, "rng_events", String[]) == expected_events,
                "$name: RNG effects do not match the physical source order")
        require(!isempty(case["energy_errors"]) &&
                last(case["energy_errors"]) == case["dham"],
                "$name: final recorded statistic must equal dham")
    end

    accepted = by_name["accepted"]
    require(!accepted["diverged"], "accepted case unexpectedly diverged")
    require(accepted["exponential_calls"] == 1,
            "accepted case must consume one Metropolis draw")
    require(accepted["init_pos"] == accepted["fwd_pos"],
            "accepted proposal was not copied into init")

    rejected = by_name["rejected"]
    require(!rejected["diverged"], "rejected case unexpectedly diverged")
    require(rejected["exponential_calls"] == 1,
            "rejected case must consume one Metropolis draw")
    require(rejected["init_pos"] == rejected["initial_position"],
            "rejected proposal changed the retained position")
    require(rejected["fwd_pos"] != rejected["initial_position"],
            "rejected fixture did not produce a distinct proposal")

    diverged = by_name["diverged"]
    require(diverged["diverged"], "diverged case did not diverge")
    require(diverged["exponential_calls"] == 0,
            "divergence exit must not consume a Metropolis draw")
    require(isempty(diverged["exponential_draws"]),
            "divergence case must declare no Metropolis draw")
    require(diverged["init_pos"] == diverged["initial_position"],
            "divergence changed the retained position")
    errors
end

function main(args=ARGS)
    length(args) == 1 || error("usage: validate_reactivehmc_hmc.jl <receipt.toml>")
    errors = validate_reactivehmc_hmc_receipt(only(args))
    if isempty(errors)
        println("VALIDATE OK — pinned ReactiveHMC fixed-step HMC receipt is self-consistent")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
