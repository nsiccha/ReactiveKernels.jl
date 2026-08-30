#!/usr/bin/env julia

import TOML

function validate_reactivehmc_integrator_receipt(path)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "reactivehmc-integrators-ca9-v1",
            "schema must be reactivehmc-integrators-ca9-v1")
    for section in ("pins", "inputs", "cases")
        require(haskey(receipt, section), "missing $section section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    require(get(pins, "reactivehmc_revision", "") ==
            "ca9ea4ca41924bb0e1fadc01c717e1333916aba6",
            "wrong ReactiveHMC revision")
    require(get(pins, "integrators_sha256", "") ==
            "39503d11f870d5942f9fe4a06065ea75d822b0702cc56c0824bff9f5d2c02b92",
            "wrong integrators.jl digest")
    require(get(pins, "phasepoints_sha256", "") ==
            "b2eb1d28c347412fafcf6e9e5cac6b4c6c08801e5b6e2db83826806a79bdaaba",
            "wrong phasepoints.jl digest")

    inputs = receipt["inputs"]
    require(length(get(inputs, "position", [])) == 2,
            "initial position must have dimension two")
    require(length(get(inputs, "momentum", [])) == 2,
            "initial momentum must have dimension two")

    cases = receipt["cases"]
    expected_names = Set(("leapfrog", "generalized_leapfrog",
                          "implicit_midpoint", "multistep"))
    require(length(cases) == 4, "expected four integrator cases")
    require(Set(case["name"] for case in cases) == expected_names,
            "integrator case names are incomplete")
    for case in cases
        name = case["name"]
        require(case["stepsize"] > 0, "$name: stepsize must be positive")
        require(case["n_steps"] >= 1, "$name: n_steps must be positive")
        require(case["n_fi_steps"] >= 0,
                "$name: fixed-point iteration count must be nonnegative")
        for field in ("pos", "mom", "dham_dpos", "dham_dmom")
            values = get(case, field, [])
            require(length(values) == 2, "$name: $field must have dimension two")
            require(all(isfinite, values), "$name: $field contains a nonfinite value")
        end
        require(isfinite(case["ham"]), "$name: Hamiltonian is nonfinite")
    end
    errors
end

function main(args=ARGS)
    length(args) == 1 || error(
        "usage: validate_reactivehmc_integrators.jl <receipt.toml>")
    errors = validate_reactivehmc_integrator_receipt(only(args))
    if isempty(errors)
        println("VALIDATE OK — pinned ReactiveHMC integrator receipt is self-consistent")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
