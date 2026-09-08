# AlgebraOfVega plots for the all-82 posteriordb benchmark, rendered from an
# all80-benchmark-v1 receipt (native ∪ Reactant cells; benchmark/all80_receipt.jl).
# Included INTO the ReactiveKernelsDocs module (like all80_comparison_tables.jl), so
# `data`, `mapping`, `visual`, `config`, `scales`, `_plot_block`, `_interactive_block`,
# and `h` are already in scope. The extra AoV marks this file needs are imported here.
#
# The receipt cells are honest by construction (benchmark/all80_reactant_body.jl,
# all80_axes.jl): a numeric timing where the operation ran, or an exact diagnostic
# STRING where it failed. These plots therefore only ever plot the NUMERIC cells, and
# the coverage plot COUNTS the string cells so the failures stay visible — never hidden,
# so they can be improved or fixed.
using AlgebraOfVega: BarPlot, Scatter
import TOML

# `_ALL80_BENCHMARK_PATH` (the aggregated all80-benchmark-v1 receipt) is defined in
# all80_comparison_tables.jl, which make.jl includes into this module first.
_all80_models(path = _ALL80_BENCHMARK_PATH) = get(TOML.parsefile(path), "models", Dict{String,Any}())

# A plot over an empty row set would break AoV at render time; degrade to an honest note
# instead, so a benchmark where (say) nothing lowered through Reactant still builds cleanly.
_all80_plot_note(text) = Markdown.parse("*" * text * "*")

# log2 speedup of `baseline` over `rk` (>0 ⇒ the RK/Reactant side is faster). Both
# operands must be finite positive Reals; anything else (a diagnostic string, a missing
# cell, a non-finite number) yields `missing` and is dropped from the ratio plots.
function _log2_speedup(fast, baseline)
    (fast isa Real && baseline isa Real && isfinite(fast) && isfinite(baseline) &&
        fast > 0 && baseline > 0) || return missing
    log2(baseline / fast)
end

# ---- Plot 1: single-eval speedup (CURRENT COMMITTED faithful RK graphs vs reference Stan / Turing) ----
# x = model dimension (log), y = log2(comparator / RK) so >0 ⇒ RK faster, faceted by comparator,
# coloured by primal-vs-gradient, SHAPED by workload-match. These are the graphs COMMITTED at the
# historical benchmark base (8f09780) — NOT the natural-source cleanup baseline (reserved separately).
# A numerically/parity-valid ratio is NOT a matched-workload claim: models where the RK graph
# evaluates a fundamentally different amount of work than the comparator (e.g. Mb's rich O(M)
# per-individual plate vs the comparator's O(1) sufficient statistics) are shown as a DISTINCT
# "workload-mismatch" series, never merged into a matched-workload win/loss. Set curated from the
# performance workload audit (extend as it establishes more); Mb is the confirmed flagship.
const _ALL80_WORKLOAD_MISMATCH = Set(["Mb_data-Mb_model"])

# TIMING-QUARANTINED models (performance steer 2026-09-08, option c): their native timings were
# measured under KNOWN competing host load (telemetry-evidenced), so their timing VALUES are
# EXCLUDED from win/loss ratio plots, ratio aggregates, and comparative conclusions. Their
# structural/value/diagnostic receipts still stand and are shown; the raw contended timings +
# provenance are surfaced SEPARATELY (a quarantine note), never promoted into a ranking. This is
# NOT a claim that every absolute timing is inflated — differential interference is unknown.
const _ALL80_TIMING_QUARANTINED = Set([
    "earnings-earn_height", "dogs-dogs_hierarchical",
    "wells_data-wells_dist", "GLMM_Poisson_data-GLMM_Poisson_model"])

function _all80_speedup_rows(models)
    rows = NamedTuple[]
    for (name, m) in models
        haskey(m, "error") && continue
        # OMIT known-contended rows ENTIRELY from the ranking (timing quarantine) — their raw values
        # live only in the Measurements/evidence section, never in a win/loss ratio (performance
        # review 2026-09-08: the caption's omission claim must be enforced here, not just described).
        name in _ALL80_TIMING_QUARANTINED && continue
        d = get(m, "dim", missing)
        (d isa Real && d > 0) || continue
        family = String(get(m, "family", ""))
        for (metric, rk_key, comparators) in (
                ("primal", "primal_rk",
                    (("reference Stan", "primal_stan"), ("upstream Turing", "primal_turing"))),
                ("gradient", "gradient_rk",
                    (("reference Stan", "gradient_stan"), ("upstream Turing", "gradient_turing"))))
            rk = get(m, rk_key, missing)
            for (comparator, cmp_key) in comparators
                # Fix E: a NON-EQUIVALENT Turing support (turing_support_ok=false) is not a valid
                # reference — never emit an RK/Turing ratio for it (RK/Stan stays valid). Old rows
                # without the flag default to true (unchanged behavior).
                comparator == "upstream Turing" && get(m, "turing_support_ok", true) == false && continue
                s = _log2_speedup(rk, get(m, cmp_key, missing))
                s === missing && continue
                push!(rows, (; model = name, family, dim = Float64(d), metric, comparator, speedup = s,
                    workload = name in _ALL80_WORKLOAD_MISMATCH ? "gross-workload mismatch (Mb)" :
                               "workload NOT certified matched"))
            end
        end
    end
    rows
end

function render_all80_speedup_plot(path = _ALL80_BENCHMARK_PATH)
    rows = _all80_speedup_rows(_all80_models(path))
    isempty(rows) && return _all80_plot_note(
        "No numerically-gated single-evaluation rows to plot.")
    spec = data(rows) *
        mapping(:dim => "Model dimension", :speedup => "log₂(comparator / RK)   ·   >0 ⇒ RK faster";
            color = :metric => "Evaluation", col = :comparator => "Comparator",
            marker = :workload => "Workload") *
        visual(Scatter)
    _plot_block(spec * config(width = 360, height = 300,
            title = "Current committed faithful RK graphs (base 8f09780) vs reference Stan and upstream Turing",
            scales = scales(X = (; scale = log10)));
        id = "all80-speedup",
        title = "Where the current committed faithful RK graphs win and lose",
        description = "Each point is one posteriordb model at the CURRENT COMMITTED faithful graphs " *
            "(base 8f09780; the natural-source cleanup baseline is reserved separately). " *
            "y = log₂(comparator median / RK median). KNOWN-CONTENDED rows are OMITTED entirely (not " *
            "disclaimed in-plot) — their raw values + load provenance are in the separate " *
            "Measurements/evidence section. The plotted timings are UN-AUDITED for host isolation " *
            "(not clean-certified, not a performance verdict); read directionally, treat near-parity " *
            "cautiously. A distinct marker flags the one CONFIRMED gross-workload mismatch (Mb: rich " *
            "O(M) per-individual plate vs the comparator's O(1) sufficient statistics); no other model " *
            "is positively certified same-workload. GLMM's RK/Turing ratio is excluded (non-equivalent " *
            "Turing support). See the reading guide for the preprocessing/endpoint/HMC categories.")
end

# ---- Plot 2: Reactant compiled-loop regime — native vs Reactant HMC throughput ----
# The load-bearing Reactant insight: a single small gradient loses under Reactant, but the
# WHOLE HMC loop compiled as one program can win. y = log2(native / reactant) µs/transition
# so >0 ⇒ Reactant faster; x = dimension. Only models whose Reactant HMC loop LOWERED
# (a numeric hmc_rk_reactant) appear.
function _all80_reactant_hmc_rows(models)
    rows = NamedTuple[]
    for (name, m) in models
        haskey(m, "error") && continue
        name in _ALL80_TIMING_QUARANTINED && continue   # timing quarantine (same as the speedup ranking)
        d = get(m, "dim", missing)
        (d isa Real && d > 0) || continue
        s = _log2_speedup(get(m, "hmc_rk_reactant", missing), get(m, "hmc_rk_native", missing))
        s === missing && continue
        nt = get(m, "hmc_transitions", missing)
        push!(rows, (; model = name, family = String(get(m, "family", "")),
            dim = Float64(d), speedup = s,
            native_T = nt isa Real ? Float64(nt) : missing))   # native batch size (Reactant is fixed T=4)
    end
    rows
end

function render_all80_reactant_hmc_plot(path = _ALL80_BENCHMARK_PATH)
    rows = _all80_reactant_hmc_rows(_all80_models(path))
    isempty(rows) && return _all80_plot_note(
        "No model's Reactant HMC loop lowered in this run — see the coverage plot and the " *
        "HMC-throughput table diagnostics.")
    spec = data(rows) *
        mapping(:dim => "Model dimension",
            :speedup => "log₂(native µs/transition / Reactant µs/transition)";
            color = :native_T => "native batch T (Reactant fixed at 4)") *
        visual(Scatter)
    _plot_block(spec * config(width = 480, height = 320,
            title = "Reactant HMC-loop CAPABILITY PROBE (T=4) vs native throughput (calibrated T)",
            scales = scales(X = (; scale = log10)));
        id = "all80-reactant-hmc",
        title = "Reactant compiled-HMC-loop capability probe — NOT a matched long-chain ranking",
        description = "One point per model whose Reactant HMC loop lowered. **Batch sizes DIFFER**: " *
            "Reactant runs a FIXED 4 transitions per compiled call (with per-round RNG marshalling " *
            "inside the timer), while native runs its calibrated batch T (30–1000; ~39/59 at 1000, " *
            "shown by colour). The per-call XLA-launch overhead is amortized over T, so at T=4 " *
            "Reactant's per-transition figure is inflated and this ratio is NOT a matched long-chain " *
            "backend verdict — it is a lowering CAPABILITY probe (does the whole loop compile+run). " *
            "The fair same-T comparison (T∈{4,100,1000}, same q/metric/L/ε both backends) is a tracked " *
            "investigation. Models whose loop does not lower are absent here and counted in the coverage plot.")
end

# ---- Plot 3: Reactant coverage — how many models lower, how many fail (visibly) ----
# For each Reactant cell (primal / gradient / HMC), stack {lowered, failed} across all 82
# models so the honest coverage — and the size of the still-failing set — is one glance.
const _ALL80_REACTANT_CELLS = (
    ("primal_rk_reactant", "primal"),
    ("gradient_rk_reactant", "gradient"),
    ("hmc_rk_reactant", "HMC loop"))

# `models` is the receipt's `models` Dict (name => cell-dict).
function _all80_reactant_coverage_rows(models)
    counts = Dict{Tuple{String,String},Int}()
    for (_, m) in models
        for (key, label) in _ALL80_REACTANT_CELLS
            v = get(m, key, nothing)
            outcome = (v isa Real && isfinite(v)) ? "lowered" :
                v === nothing ? "not attempted" : "failed (diagnostic recorded)"
            counts[(label, outcome)] = get(counts, (label, outcome), 0) + 1
        end
    end
    [(; operation = op, outcome, count) for ((op, outcome), count) in counts]
end

function render_all80_reactant_coverage_plot(path = _ALL80_BENCHMARK_PATH)
    rows = _all80_reactant_coverage_rows(_all80_models(path))
    spec = data(rows) *
        mapping(:operation => "Reactant operation", :count => "Models (of 82)";
            color = :outcome => "Outcome") *
        visual(BarPlot)
    _plot_block(spec * config(width = 420, height = 300,
            title = "Reactant lowering coverage across the 82 faithful graphs",
            scales = scales(Y = (; zero = true)));
        id = "all80-reactant-coverage",
        title = "Reactant coverage — what lowered, what still fails",
        description = "For each operation, how many of the 82 faithful graphs lowered through " *
            "Reactant versus recorded an exact lowering diagnostic. The failing set is kept " *
            "explicit (its errors are in the gradient/HMC tables) so it can be improved or fixed.")
end
