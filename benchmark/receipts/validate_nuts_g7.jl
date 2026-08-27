#!/usr/bin/env julia
# Validator for benchmark/receipts/nuts-g7-v1.toml. RE-DERIVES the G7 verdict from the receipt's RAW rounds + work
# counters — docs/CI parse this static receipt instead of running noisy perf. Exit 0 iff self-consistent.
# Usage: julia validate_nuts_g7.jl <receipt.toml>   (TOML stdlib only, no package deps)
import TOML, Statistics
_median(v) = (s = sort(Float64.(v)); n = length(s); isodd(n) ? s[n÷2+1] : (s[n÷2]+s[n÷2+1])/2)
function main(path)
    r = TOML.parsefile(path); errs = String[]
    push_if(c, m) = c || push!(errs, m)
    push_if(get(r, "schema", "") == "nuts-g7-v1", "schema must be nuts-g7-v1")
    for s in ("pins","env","work","medians","raw","verdict"); push_if(haskey(r, s), "missing [$s]"); end
    isempty(errs) || (foreach(println, errs); exit(1))
    for k in ("reactivekernels_sha","nuts_jl_sha","julia_version"); push_if(haskey(r["pins"],k) && !isempty(r["pins"][k]), "pins.$k missing/empty"); end
    arms = ("RK","NUTS.jl","AHMC","DHMC")
    # (1) re-derive each arm's median from its raw rounds; must match the recorded median.
    for a in arms
        push_if(haskey(r["raw"],a) && !isempty(r["raw"][a]), "raw.$a missing/empty") || continue
        push_if(haskey(r["medians"],a), "medians.$a missing") || continue
        rederived = _median(r["raw"][a]); recorded = Float64(r["medians"][a])
        push_if(abs(rederived - recorded) <= 1e-6*max(1.0,abs(recorded)), "medians.$a=$recorded != re-derived $(rederived) from raw")
    end
    isempty(errs) || (foreach(println, errs); exit(1))
    med = Dict(a => _median(r["raw"][a]) for a in arms)
    # (2) THE GATE (user-relaxed): RK > AHMC && RK > DHMC — re-derived, must equal verdict.gate_clears.
    gate = med["RK"] > med["AHMC"] && med["RK"] > med["DHMC"]
    push_if(gate == Bool(r["verdict"]["gate_clears"]), "verdict.gate_clears=$(r["verdict"]["gate_clears"]) != re-derived $gate")
    push_if(gate, "GATE FAILS: RK ($(med["RK"])) is not > both AHMC ($(med["AHMC"])) and DHMC ($(med["DHMC"]))")
    # (3) exact work: RK grads==steps, NUTS grads==steps.
    rkw = r["work"]["RK"]; nuw = r["work"]["NUTS.jl"]
    push_if(rkw["grads"] == rkw["steps"], "RK work not exact: grads $(rkw["grads"]) != steps $(rkw["steps"])")
    push_if(nuw["grads"] == nuw["steps"], "NUTS.jl work not exact: grads $(nuw["grads"]) != steps $(nuw["steps"])")
    push_if(Bool(r["verdict"]["rk_work_exact"]) == (rkw["grads"]==rkw["steps"]), "verdict.rk_work_exact mismatch")
    push_if(Bool(r["verdict"]["nuts_work_exact"]) == (nuw["grads"]==nuw["steps"]), "verdict.nuts_work_exact mismatch")
    # (4) reported RK/NUTS reference ratio consistency.
    ratio = med["RK"]/med["NUTS.jl"]
    push_if(abs(ratio - Float64(r["verdict"]["rk_nuts_ratio"])) <= 0.02, "verdict.rk_nuts_ratio $(r["verdict"]["rk_nuts_ratio"]) != re-derived $(round(ratio,digits=3))")
    if isempty(errs)
        println("VALIDATE OK — nuts-g7-v1 receipt self-consistent: gate RK>AHMC&&RK>DHMC = $gate (re-derived from raw); ",
                "RK/NUTS = $(round(ratio,digits=3)) (reference); RK & NUTS work exact; medians match raw rounds.")
        exit(0)
    else
        foreach(println, errs); exit(1)
    end
end
length(ARGS) == 1 || (println("usage: julia validate_nuts_g7.jl <receipt.toml>"); exit(2))
main(ARGS[1])
