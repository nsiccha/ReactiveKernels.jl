#!/usr/bin/env julia

import TOML

function validate_reactivehmc_statistics_receipt(path)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)

    require(get(receipt, "schema", "") == "reactivehmc-statistics-ca9-v1",
            "schema must be reactivehmc-statistics-ca9-v1")
    for section in ("pins", "inputs", "events", "second_trajectory",
                    "trajectory", "samples", "sampling")
        require(haskey(receipt, section), "missing $section section")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    require(get(pins, "reactivehmc_revision", "") ==
            "ca9ea4ca41924bb0e1fadc01c717e1333916aba6",
            "wrong ReactiveHMC revision")
    require(get(pins, "statistics_sha256", "") ==
            "20baff1337a3e7c5926f01e104484168dd9783fe397366ebcb78ad3501eb1f69",
            "wrong statistics.jl digest")

    dimension = get(receipt["inputs"], "dimension", 0)
    require(dimension > 0, "dimension must be positive")
    for field in ("reset_pos", "reset_dham_dpos")
        require(length(get(receipt["inputs"], field, [])) == dimension,
                "$field must match dimension")
    end

    events = receipt["events"]
    require(length(events) == 3, "expected three ordered trajectory events")
    require(map(event -> event["name"], events) ==
            ["append_one", "prepend", "append_two"],
            "trajectory event order changed")
    require(map(event -> event["go_forward"], events) == [true, false, true],
            "trajectory direction script changed")
    for event in events
        for field in ("pos", "dham_dpos")
            require(length(get(event, field, [])) == dimension,
                    "$(event["name"]): $field must match dimension")
        end
    end

    trajectory = receipt["trajectory"]
    count = length(events) + 1
    for field in ("positions", "gradients")
        columns = get(trajectory, field, [])
        require(length(columns) == count, "$field must have $count columns")
        require(all(column -> length(column) == dimension, columns),
                "$field columns must match dimension")
    end
    for field in ("dhams", "pots", "idxs")
        require(length(get(trajectory, field, [])) == count,
                "$field must have $count entries")
    end
    require(sort(trajectory["idxs"]) == collect(0:(count - 1)),
            "trajectory indices must preserve reveal ownership")

    samples = receipt["samples"]
    require(length(samples) == 2, "expected two sampling callbacks")
    sampling = receipt["sampling"]
    for field in ("draws", "n_steps", "stepsizes", "acc_rate", "diverged",
                  "full_history", "full_idxs")
        require(length(get(sampling, field, [])) == length(samples),
                "$field must have one entry per sampling callback")
    end
    require(all(isfinite, sampling["acc_rate"]),
            "acceptance rates must be finite")
    require(all(rate -> 0 <= rate <= 1, sampling["acc_rate"]),
            "acceptance rates must lie in [0, 1]")
    require(sampling["full_history"][1] == trajectory["positions"],
            "first full-history snapshot must preserve the first trajectory")
    require(sampling["full_idxs"][1] == trajectory["idxs"],
            "first full-index snapshot must preserve the first trajectory")
    errors
end

function main(args=ARGS)
    length(args) == 1 || error(
        "usage: validate_reactivehmc_statistics.jl <receipt.toml>")
    errors = validate_reactivehmc_statistics_receipt(only(args))
    if isempty(errors)
        println("VALIDATE OK — pinned ReactiveHMC statistics receipt is self-consistent")
        return 0
    end
    foreach(println, errors)
    1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
