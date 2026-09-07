# EXECUTABLE structural gate for the all-82 registry (NOT density parity — that is the
# driver's run-time hard gate). For every registry key, against the pinned env + FULL
# real posteriordb data, this checks:
#   (1) the PosteriorDB dataset loads;
#   (2) the upstream Turing make_model is APPLICABLE by DISPATCH (applicable(make_model,
#       Val(Symbol(key)), data)) — NOT a text grep, which misses multiline signatures;
#   (3) bind(data) runs and its NamedTuple keys EXACTLY equal have ∖ {:unconstrained}
#       (rejects empty / best-effort / placeholder binds);
#   (4) the RK graph builds and `prepare(graph; have, want=:posterior, bound=bind(data))`
#       succeeds on the full data (bound shapes flow through the graph).
# It also confirms registry keys == the implemented posteriordb Example-module inventory.
import Pkg
const ENV_DIR = joinpath(@__DIR__, "all80-env")
Pkg.activate(ENV_DIR); Pkg.instantiate()
const UP = get(ENV, "RK_ALL80_UPSTREAM", joinpath(ENV_DIR, "upstream"))

import PosteriorDB
using ReactiveKernels, ReactiveKernelsPPLExamples
const RKE = ReactiveKernelsPPLExamples
include(joinpath(@__DIR__, "all80_registry.jl"))
using .All80Registry: REGISTRY
include(joinpath(UP, "posteriordb.jl"))   # main-guarded; defines make_model + pdb_ models + PDB

# ---- inventory reconciliation: registry modules == implemented posteriordb modules ----
allmods = Set(n for n in names(RKE) if endswith(String(n), "Example"))
# The seven non-posteriordb infra/demo Example modules (not faithful posteriordb ports):
nonpdb = Set(Symbol.(["SumToZeroExample", "LinearRegressionExample", "BetaBinomialExample",
    "PoissonGammaExample", "MNISTLogisticExample", "MVNormalRegressionExample", "BoundRegressionExample"]))
inventory = setdiff(allmods, nonpdb)
regmods = Set(e.mod for e in values(REGISTRY))
miss = sort(String.(collect(setdiff(inventory, regmods))))
extra = sort(String.(collect(setdiff(regmods, inventory))))
println("REGISTRY keys: ", length(REGISTRY), " | inventory modules: ", length(inventory),
        " | registry modules: ", length(regmods))
println("MODULES in inventory but NOT in registry (", length(miss), "): ", miss)
println("MODULES in registry but NOT in inventory (", length(extra), "): ", extra)

# ---- per-key executable gate ----
const PDB2 = PosteriorDB.database()
# Optional ARGS key filter: `julia all80_validate.jl <key> [<key>...]` checks only those
# keys (fast fix-verify loop); no args checks all 82.
const SELECT = String[a for a in ARGS if !startswith(a, "-")]
const TARGETS = isempty(SELECT) ? sort(collect(REGISTRY); by = first) :
    [(k => REGISTRY[k]) for k in SELECT if haskey(REGISTRY, k)]
const N_TARGET = length(TARGETS)
fails = String[]
n_ok = 0
for (k, e) in TARGETS
    # (1) dataset load
    data = try
        PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(PDB2, k)))
    catch err
        push!(fails, "$k  DATASET: $(sprint(showerror, err))"); continue
    end
    # (2) upstream make_model applicability by DISPATCH
    if !applicable(make_model, Val(Symbol(k)), data)
        push!(fails, "$k  NO make_model dispatch (Val{Symbol}, data)")
    end
    # (3) bind + exact key check
    want = Set(p for p in e.have if p != :unconstrained)
    nt = try
        e.bind(data)
    catch err
        push!(fails, "$k  BIND threw: $(sprint(showerror, err))"); continue
    end
    got = Set(keys(nt))
    if got != want
        push!(fails, "$k  KEYS: got $(sort(String.(collect(got)))) want $(sort(String.(collect(want))))")
        continue
    end
    if isempty(want) && !isempty(nt) === false && length(nt) == 0
        push!(fails, "$k  EMPTY bind"); continue
    end
    # (4) build + prepare on full data
    try
        graph = getproperty(getproperty(RKE, e.mod), e.build)()
        prepare(graph; have = e.have, want = :posterior, bound = nt)
    catch err
        push!(fails, "$k  BUILD/PREPARE: $(sprint(showerror, err))"); continue
    end
    global n_ok += 1
end

println("\nEXECUTABLE GATE: $n_ok / $N_TARGET keys passed (dataset+dispatch+bind-keys+build+prepare)",
        isempty(SELECT) ? "." : " [SUBSET: $(join(SELECT, ", "))].")
if !isempty(fails)
    println("FAILURES (", length(fails), "):")
    for f in fails; println("  ", f); end
end
# On a subset run, skip the full inventory reconciliation from the verdict.
ok = (isempty(SELECT) ? (isempty(miss) && isempty(extra)) : true) && isempty(fails) && n_ok == N_TARGET
println(ok ? "VALIDATE_PASS" : "VALIDATE_ISSUES")
println("ALL80_VALIDATE_DONE")
