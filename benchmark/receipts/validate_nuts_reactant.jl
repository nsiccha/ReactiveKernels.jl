#!/usr/bin/env julia

# Re-derive the adaptive Reactant NUTS receipt medians and acceptance invariants
# from its raw rounds. The benchmark itself is opt-in; docs and tests validate
# the frozen receipt deterministically with TOML + Statistics only.

import Statistics
import TOML

_median(values) = Statistics.median(Float64.(values))

function validate_nuts_reactant_receipt(path)
    receipt = TOML.parsefile(path)
    errors = String[]
    require(condition, message) = condition || push!(errors, message)
    require(get(receipt, "schema", "") == "nuts-reactant-v1",
            "schema must be nuts-reactant-v1")
    for section in ("pins", "environment", "protocol", "compilation", "raw",
                    "medians", "acceptance")
        require(haskey(receipt, section), "missing [$section]")
    end
    isempty(errors) || return errors

    pins = receipt["pins"]
    protocol = receipt["protocol"]
    raw = receipt["raw"]
    medians = receipt["medians"]
    acceptance = receipt["acceptance"]
    require(!isempty(get(pins, "reactivekernels_sha", "")),
            "pins.reactivekernels_sha missing")
    require(get(pins, "accepted_implementation_sha", "") ==
                "0c96826957b9eee177be5818239e30e587109d8c",
            "pins.accepted_implementation_sha mismatch")
    require(!isempty(get(pins, "authored_fixture_blob", "")),
            "pins.authored_fixture_blob missing")
    require(get(pins, "reactivekernels_dirty", true) == false,
            "published receipt must come from a tracked-clean checkout")
    require(get(protocol, "max_depth", 0) == 10,
            "publication protocol must exercise max_depth=10")
    require(get(protocol, "dimension", 0) == 5,
            "publication protocol must exercise dimension=5")
    require(get(protocol, "reactant_sync", false),
            "Reactant timing must be synchronous")
    for exclusion in (
            "compile_time_in_steady_state", "host_device_transfers_in_steady_state",
            "rng_generation_in_steady_state", "result_readback_in_steady_state",
            "input_rebundle_in_steady_state",
            "per_transition_state_setup_in_steady_state",
            "parity_screening_in_steady_state",
            "adaptation_measured", "ess_measured",
            "end_to_end_sampling_measured")
        require(get(protocol, exclusion, true) == false,
                "protocol.$exclusion must be false")
    end

    rounds = Int(protocol["rounds"])
    transitions = Int(protocol["transitions_per_round"])
    require(rounds > 0 && transitions > 0, "round and transition counts must be positive")
    examined = Int(get(protocol, "candidate_bundles_examined", 0))
    rejected = Int(get(protocol, "candidate_bundles_rejected", -1))
    require(examined == rounds * transitions + rejected,
            "candidate counts must report every accepted and rejected bundle")
    require(rejected >= 0, "candidate rejection count must be nonnegative")
    for key in (
            "round_steps", "round_directions", "round_exponentials",
            "round_max_reached_depth", "round_divergences",
            "native_round_seconds", "reactant_round_seconds", "native_transition_ms",
            "reactant_transition_ms", "native_steps_per_second",
            "reactant_steps_per_second")
        require(haskey(raw, key) && length(raw[key]) == rounds,
                "raw.$key must have one value per round")
    end
    isempty(errors) || return errors

    require(all(>(0), raw["round_steps"]), "every round must execute leapfrog work")
    require(all(depth -> 0 <= depth <= 10, raw["round_max_reached_depth"]),
            "reached depths must stay inside the admitted maximum")
    require(any(>(1), raw["round_max_reached_depth"]),
            "benchmark corpus must exercise adaptive depth beyond one")
    require(all(count -> 0 <= count <= transitions, raw["round_divergences"]),
            "divergence counts must fit the transition count")
    require(all(>(0), raw["native_round_seconds"]), "native times must be positive")
    require(all(>(0), raw["reactant_round_seconds"]), "Reactant times must be positive")
    derived_native_ms = 1e3 .* raw["native_round_seconds"] ./ transitions
    derived_reactant_ms = 1e3 .* raw["reactant_round_seconds"] ./ transitions
    derived_native_steps = raw["round_steps"] ./ raw["native_round_seconds"]
    derived_reactant_steps = raw["round_steps"] ./ raw["reactant_round_seconds"]
    for (key, derived) in (
            "native_transition_ms" => derived_native_ms,
            "reactant_transition_ms" => derived_reactant_ms,
            "native_steps_per_second" => derived_native_steps,
            "reactant_steps_per_second" => derived_reactant_steps)
        require(all(isapprox.(Float64.(raw[key]), derived; rtol=1e-12, atol=0)),
                "raw.$key is not re-derived from round work/time")
        require(isapprox(Float64(medians[key]), _median(derived); rtol=1e-12, atol=0),
                "medians.$key is not the raw-round median")
    end
    require(isapprox(
        Float64(medians["reactant_over_native_transition_time"]),
        Float64(medians["reactant_transition_ms"]) /
            Float64(medians["native_transition_ms"]); rtol=1e-12, atol=0),
        "transition-time ratio mismatch")
    require(isapprox(
        Float64(medians["reactant_over_native_steps_per_second"]),
        Float64(medians["reactant_steps_per_second"]) /
            Float64(medians["native_steps_per_second"]); rtol=1e-12, atol=0),
        "steps/s ratio mismatch")
    for key in (
            "same_authored_transition", "same_target_metric_state_depth_randomness",
            "matched_independent_start_states",
            "matched_control_flow_corpus", "parity_screening_reported",
            "per_transition_observable_parity", "random_consumption_parity",
            "all_overflow_flags_zero")
        require(get(acceptance, key, false), "acceptance.$key must be true")
    end
    require(get(acceptance, "stablehlo_while_count", 0) == 1,
            "receipt must prove exactly one stablehlo.while")

    errors
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 ||
        (println("usage: julia validate_nuts_reactant.jl <receipt.toml>"); exit(2))
    errors = validate_nuts_reactant_receipt(only(ARGS))
    if isempty(errors)
        println(
            "VALIDATE OK — nuts-reactant-v1 receipt is self-consistent: ",
            "max_depth=10, synchronous execution, exact work/time medians, ",
            "native/Reactant parity from independent matched starts, matched randomness, ",
            "zero overflow, one stablehlo.while.",
        )
    else
        foreach(println, errors)
        exit(1)
    end
end
