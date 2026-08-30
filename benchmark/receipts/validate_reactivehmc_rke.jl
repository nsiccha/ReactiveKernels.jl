#!/usr/bin/env julia

import TOML

function validate_reactivehmc_rke_receipt(path)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "reactivehmc-rke-ca9-v1",
            "schema must be reactivehmc-rke-ca9-v1")
    require(haskey(receipt, "pins"), "missing [pins]")
    require(haskey(receipt, "cases"), "missing [[cases]]")
    isempty(errors) || return errors

    pins = receipt["pins"]
    require(get(pins, "reactivehmc_revision", "") ==
            "ca9ea4ca41924bb0e1fadc01c717e1333916aba6",
            "wrong ReactiveHMC revision")
    require(get(pins, "energies_sha256", "") ==
            "1d051da9f1ca56b46e6802f66bbf36c21d58dd4442e6ad6d8118e52a93d492de",
            "wrong energies.jl digest")

    cases = receipt["cases"]
    require(length(cases) == 2, "expected two parameter cases")
    for (index, case) in enumerate(cases)
        for field in ("x_sq", "e_sq", "p_sq", "cdf_sq")
            require(length(get(case, field, [])) == 4,
                    "case $index: $field must contain four values")
        end
        for field in ("q", "quantile_sq", "roundtrip_cdf")
            require(length(get(case, field, [])) == 4,
                    "case $index: $field must contain four values")
        end
        length(get(case, "e_sq", [])) == 4 || continue

        require(issorted(case["e_sq"]), "case $index: energy must be monotone")
        require(issorted(case["cdf_sq"]), "case $index: CDF must be monotone")
        require(all(0.0 <= value <= 1.0 for value in case["cdf_sq"]),
                "case $index: CDF outside [0,1]")
        require(all(isapprox(p, exp(-e); rtol=8eps(Float64), atol=0.0)
                    for (p, e) in zip(case["p_sq"], case["e_sq"])),
                "case $index: density is inconsistent with recorded energy")
        require(all(isapprox(actual, expected; rtol=0.0, atol=8eps(Float64))
                    for (actual, expected) in zip(case["roundtrip_cdf"], case["q"])),
                "case $index: recorded quantile/CDF round trip is inconsistent")
    end
    errors
end

function main(args=ARGS)
    length(args) == 1 || error("usage: validate_reactivehmc_rke.jl <receipt.toml>")
    errors = validate_reactivehmc_rke_receipt(only(args))
    if isempty(errors)
        println("VALIDATE OK — pinned ReactiveHMC RKE receipt is self-consistent")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
