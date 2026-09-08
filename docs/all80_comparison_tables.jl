# 82-row sortable comparison tables (primal / gradient / HMC-throughput) for the
# all-82 posteriordb benchmark, rendered from an all80-benchmark-v1 receipt
# (benchmark/all80_receipt.jl). Included INTO the ReactiveKernelsDocs module (like
# result_views.jl), so `_column`, `_result_table`, and `h` are in scope. Each table is a
# `data-rk-artifact-kind="sortable-table"` section: rows = the 82 models, columns = the
# implementations. Values are shown to ~2 significant figures; N/A cells carry a
# provenance string and sort to the bottom.
import TOML

# The aggregated all80-benchmark-v1 receipt (native phase ∪ Reactant phase). The
# native-only frozen checkpoint (all80-native-checkpoint-v1.toml) remains committed as
# historical provenance; this page renders the aggregate so the Reactant cells appear.
const _ALL80_BENCHMARK_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "all80-benchmark-v1.toml")

# ~2-sig-fig timing formatter (nanoseconds in the receipt). Non-Real = N/A provenance.
function _all80_ns(value, _)
    value isa Real || return string(value)
    value < 1e3  ? string(round(value; sigdigits = 2), " ns") :
    value < 1e6  ? string(round(value / 1e3; sigdigits = 2), " µs") :
                   string(round(value / 1e6; sigdigits = 2), " ms")
end
# HMC throughput formatter: receipt stores µs/transition (lower = faster).
_all80_us(value, _) = value isa Real ? string(round(value; sigdigits = 2), " µs/it") : string(value)
# Numbers sort numerically; N/A provenance strings sort to the bottom.
_all80_sort(value, _) = value isa Real ? value : nothing

# Build one table's rows (NamedTuples) from the receipt `models` dict, sorted by key.
function _all80_rows(models, cellkeys)
    ks = sort(collect(keys(models)))
    map(ks) do k
        m = models[k]
        diagnostic = get(m, "error", "")
        fallback(c) = !isempty(diagnostic) ? "unavailable — " * first(diagnostic, 180) :
            occursin("reactant", c) ? "not attempted" : "not measured"
        base = (; model = k,
            note = isempty(diagnostic) ? get(m, "note", "") : "Benchmark gate failure: " * diagnostic)
        cells = NamedTuple{Tuple(Symbol.(cellkeys))}(
            Tuple(haskey(m, c) ? m[c] : fallback(c) for c in cellkeys))
        merge(base, cells)
    end
end

_all80_num(key, label) = _column(key, label; format = _all80_ns, sort = _all80_sort)
_all80_hmc(key, label) = _column(key, label; format = _all80_us, sort = _all80_sort)

# --- the three tables ------------------------------------------------------------
function all80_primal_table(models; id = "all80-primal")
    cells = ["primal_rk", "primal_rk_reactant", "primal_turing", "primal_stan",
             "primal_opt_stan", "primal_further_turing"]
    cols = (_column(:model, "Model"),
            _all80_num(:primal_rk, "idiomatic RK"),
            _all80_num(:primal_rk_reactant, "RK + Reactant"),
            _all80_num(:primal_turing, "upstream Turing"),
            _all80_num(:primal_stan, "reference Stan"),
            _all80_num(:primal_opt_stan, "optimized Stan"),
            _all80_num(:primal_further_turing, "further Turing"),
            _column(:note, "Structural difference"))
    _result_table(_all80_rows(models, cells), cols; id,
        title = "Primal log-density evaluation — median, lower is faster")
end

function all80_gradient_table(models; id = "all80-gradient")
    cells = ["gradient_rk", "gradient_rk_reactant", "gradient_turing", "gradient_stan",
             "gradient_opt_stan", "gradient_further_turing"]
    cols = (_column(:model, "Model"),
            _all80_num(:gradient_rk, "idiomatic RK"),
            _all80_num(:gradient_rk_reactant, "RK + Reactant"),
            _all80_num(:gradient_turing, "upstream Turing"),
            _all80_num(:gradient_stan, "reference Stan"),
            _all80_num(:gradient_opt_stan, "optimized Stan"),
            _all80_num(:gradient_further_turing, "further Turing"),
            _column(:note, "Structural difference"))
    _result_table(_all80_rows(models, cells), cols; id,
        title = "Value+gradient evaluation — median, lower is faster")
end

function all80_hmc_table(models; id = "all80-hmc")
    cells = ["hmc_rk_native", "hmc_rk_reactant", "hmc_ahmc_turing"]
    cols = (_column(:model, "Model"),
            _all80_hmc(:hmc_rk_native, "RK native"),
            _all80_hmc(:hmc_rk_reactant, "RK + Reactant"),
            _all80_hmc(:hmc_ahmc_turing, "AHMC + Turing"),
            _column(:note, "Structural difference"))
    _result_table(_all80_rows(models, cells), cols; id,
        title = "HMC throughput — median µs per transition (multinomial HMC, fixed L), lower is faster")
end

"""Render all three all-82 comparison tables from a receipt file (or a parsed
`models` dict). Returns the concatenated static blocks for embedding in the docs page."""
function all80_comparison_sections(receipt)
    models = receipt isa AbstractDict ? receipt : get(TOML.parsefile(receipt), "models", Dict())
    (all80_primal_table(models), all80_gradient_table(models), all80_hmc_table(models))
end

"""Render the published native checkpoint summary. This deliberately accepts gate-error
rows: an exact diagnostic is a benchmark result, whereas an invented timing is not."""
function render_all80_native_checkpoint_summary(path = _ALL80_BENCHMARK_PATH)
    receipt = TOML.parsefile(path)
    models = get(receipt, "models", Dict())
    failures = count(m -> haskey(m, "error"), values(models))
    passing = count(m -> get(m, "parity_pass", false) === true, values(models))
    pending_offsets = length(models) - failures - passing
    Markdown.parse("""
    **Checkpoint receipt:** $(length(models)) of 82 models recorded; **$passing** pass the
    complete declared-offset/value/gradient/support gate, **$pending_offsets** have complete
    measurements awaiting a source-declared constant replay, and **$failures** retain an
    exact structural or AD diagnostic instead of a fabricated timing.
    """)
end

"""Render the three sortable tables from the committed native-checkpoint receipt."""
function render_all80_native_checkpoint(path = _ALL80_BENCHMARK_PATH)
    receipt = TOML.parsefile(path)
    models = get(receipt, "models", Dict())
    Markdown.MD(Any[
        all80_primal_table(models),
        all80_gradient_table(models),
        all80_hmc_table(models),
    ])
end
