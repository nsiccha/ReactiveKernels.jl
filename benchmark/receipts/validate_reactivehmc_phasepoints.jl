#!/usr/bin/env julia

import TOML

function validate_reactivehmc_phasepoint_receipt(path)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "reactivehmc-phasepoints-ca9-v1",
            "schema must be reactivehmc-phasepoints-ca9-v1")
    for section in ("pins", "inputs", "cases")
        require(haskey(receipt, section), "missing $section section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    require(get(pins, "reactivehmc_revision", "") ==
            "ca9ea4ca41924bb0e1fadc01c717e1333916aba6",
            "wrong ReactiveHMC revision")
    require(get(pins, "phasepoints_sha256", "") ==
            "b2eb1d28c347412fafcf6e9e5cac6b4c6c08801e5b6e2db83826806a79bdaaba",
            "wrong phasepoints.jl digest")

    inputs = receipt["inputs"]
    require(length(get(inputs, "position", [])) == 2,
            "initial position must have dimension two")
    require(length(get(inputs, "momentum", [])) == 2,
            "initial momentum must have dimension two")
    require(get(inputs, "speed", 0) > 0, "speed must be positive")
    require(get(inputs, "mass", 0) > 0, "mass must be positive")
    require(get(inputs, "alpha", 0) > 0, "SoftAbs alpha must be positive")

    cases = receipt["cases"]
    expected_names = Set((
        "euclidean", "riemannian", "softabs",
        "relativistic_euclidean", "relativistic_riemannian",
        "relativistic_softabs",
    ))
    require(length(cases) == 6, "expected six phase-point variants")
    require(Set(case["name"] for case in cases) == expected_names,
            "phase-point case names are incomplete")
    for case in cases
        name = case["name"]
        require(isfinite(case["pot"]), "$name: potential is nonfinite")
        require(isfinite(case["ham"]), "$name: Hamiltonian is nonfinite")
        for field in ("dham_dpos", "dham_dmom")
            values = get(case, field, [])
            require(length(values) == 2, "$name: $field must have dimension two")
            require(all(isfinite, values), "$name: $field contains a nonfinite value")
        end
    end
    errors
end

function main(args=ARGS)
    length(args) == 1 || error(
        "usage: validate_reactivehmc_phasepoints.jl <receipt.toml>")
    errors = validate_reactivehmc_phasepoint_receipt(only(args))
    if isempty(errors)
        println("VALIDATE OK — pinned ReactiveHMC phase-point receipt is self-consistent")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
