# Fail-closed rendered contract for the published native all-82 checkpoint.
using Test
import TOML, Markdown
# make.jl includes this file at Main top level while the table builders live in
# ReactiveKernelsDocs (via Base.include); import them explicitly so the contract
# holds under make.jl's own load order, not just inside the module.
using .ReactiveKernelsDocs: all80_gradient_table, all80_hmc_table, all80_primal_table

const _ALL80_RECEIPT = get(ENV, "RK_ALL80_RECEIPT",
    normpath(joinpath(@__DIR__, "..", "benchmark", "receipts",
        "all80-benchmark-v1.toml")))

# _result_table returns a RawHTML struct wrapping the HTML string (the shape the
# established test_benchmark_views.jl contracts assert on via `.content`); there is
# no show(::MIME"text/html", ::RawHTML) method, so read the content, don't sprint it.
_all80_html(node) = node.content

function _all80_assert_table(node, id, n_rows, headers)
    html = _all80_html(node)
    @test occursin("data-rk-artifact-kind=\"sortable-table\"", html)
    @test occursin("table:" * id, html)
    for hdr in headers
        @test occursin(hdr, html)
    end
    @test count("<tr", html) >= n_rows + 1
end

_finite_number(v) = v isa Real && isfinite(v)
_number_or_reason(v) = _finite_number(v) || (v isa AbstractString && !isempty(v))

@testset "all-82 benchmark rendered contracts" begin
    @test isfile(_ALL80_RECEIPT)
    receipt = TOML.parsefile(_ALL80_RECEIPT)
    @test get(receipt, "schema", "") == "all80-benchmark-v1"
    models = get(receipt, "models", Dict())
    @test length(models) == 82

    for (name, row) in models
        if haskey(row, "error")
            @test row["error"] isa AbstractString && !isempty(row["error"])
            continue
        end
        # MANDATORY reference cells (Turing/Stan) — always finite numbers; no RK-defect can touch them.
        for cell in ("primal_turing", "primal_stan",
                     "gradient_turing", "gradient_stan", "hmc_ahmc_turing")
            @test _finite_number(get(row, cell, nothing))
        end
        # RK cells (Fix D per-cell preservation) + conditionals — a finite number OR a nonempty
        # defect/reason diagnostic. `primal_rk` moved here (was MANDATORY): a genuine RK primal
        # defect (e.g. wells saturation) is a diagnostic + parity_pass=false, never a missing cell.
        for cell in ("primal_rk", "gradient_rk", "hmc_rk_native",
                     "primal_rk_reactant", "gradient_rk_reactant", "hmc_rk_reactant",
                     "primal_opt_stan", "primal_further_turing",
                     "gradient_opt_stan", "gradient_further_turing")
            @test _number_or_reason(get(row, cell, nothing))
        end
        @test haskey(row, "parity_pass")
    end

    n = length(models)
    _all80_assert_table(all80_primal_table(models), "all80-primal", n,
        ("idiomatic RK", "RK + Reactant", "upstream Turing", "reference Stan"))
    _all80_assert_table(all80_gradient_table(models), "all80-gradient", n,
        ("idiomatic RK", "RK + Reactant", "upstream Turing", "reference Stan"))
    _all80_assert_table(all80_hmc_table(models), "all80-hmc", n,
        ("RK native", "RK + Reactant", "AHMC + Turing"))
end

# Render-path filtering PROVEN (not merely described): the REAL row-builders
# `ReactiveKernelsDocs._all80_speedup_rows` / `_all80_reactant_hmc_rows` (docs/all80_benchmark_plots.jl)
# must (a) drop the RK/Turing ratio on a NON-EQUIVALENT Turing support, and (b) OMIT known-contended
# (timing-quarantined) rows ENTIRELY even when their cells are numeric.
@testset "render-path filtering — Turing-support guard AND timing quarantine (enforced, not described)" begin
    speedup_rows = ReactiveKernelsDocs._all80_speedup_rows
    hmc_rows = ReactiveKernelsDocs._all80_reactant_hmc_rows
    base = Dict("dim" => 3.0, "family" => "x",
        "primal_rk" => 1.0, "primal_stan" => 2.0, "primal_turing" => 2.0,
        "gradient_rk" => 1.0, "gradient_stan" => 2.0, "gradient_turing" => 2.0,
        "hmc_rk_native" => 5.0, "hmc_rk_reactant" => 2.0)
    models = Dict(
        "ok_model"             => merge(base, Dict("turing_support_ok" => true)),
        "bad_turing"           => merge(base, Dict("turing_support_ok" => false)),
        "earnings-earn_height" => merge(base, Dict("turing_support_ok" => true)))  # in the QUARANTINE set
    rows = speedup_rows(models)
    # (a) Turing-support guard.
    bad = [r for r in rows if r.model == "bad_turing"]
    @test !isempty(bad) && !any(r -> r.comparator == "upstream Turing", bad) &&
          all(r -> r.comparator == "reference Stan", bad)
    @test any(r -> r.comparator == "upstream Turing", [r for r in rows if r.model == "ok_model"])
    # (b) Timing quarantine — earn_height is in _ALL80_TIMING_QUARANTINED ⇒ OMITTED despite numeric cells.
    @test "earnings-earn_height" in ReactiveKernelsDocs._ALL80_TIMING_QUARANTINED
    @test isempty([r for r in rows if r.model == "earnings-earn_height"])            # omitted from speedup ranking
    @test isempty([r for r in hmc_rows(models) if r.model == "earnings-earn_height"]) # omitted from reactant-HMC ranking
    @test !isempty([r for r in hmc_rows(models) if r.model == "ok_model"])            # non-quarantined kept
end

# Batch-1 incremental render (todo 1x4pytu): the batch-1 plot fns satisfy the Documenter @eval
# contract (Markdown.MD) against a SYNTHETIC batch-1 receipt, mark diamonds as the workload-mismatch
# series, SURFACE (not rank) a failed reactant cell, degrade to a note when the receipt is absent, and
# read the batch-1 path — never the frozen-82 path (which stays untouched).
@testset "batch-1 incremental render contract (synthetic receipt)" begin
    RKD = ReactiveKernelsDocs
    bm(dim; failed = false) = begin
        m = Dict{String,Any}("dim" => dim, "family" => "batch1",
            "primal_rk" => 100.0, "gradient_rk" => 200.0, "hmc_rk_native" => 10.0,
            "primal_stan" => 150.0, "gradient_stan" => 400.0,
            "primal_turing" => 160.0, "gradient_turing" => 420.0,
            "primal_rk_reactant" => 90.0, "gradient_rk_reactant" => 180.0, "hmc_rk_reactant" => 8.0,
            "parity_pass" => true, "turing_support_ok" => true, "hmc_transitions" => 1000)
        failed && (m["gradient_rk_reactant"] = "gradient @compile: MethodError …")
        m
    end
    receipt = Dict("schema" => "all80-benchmark-v1", "generated_at" => "synthetic",
        "models" => Dict("diamonds-diamonds" => bm(9), "dogs-dogs_nonhierarchical" => bm(4),
            "ovarian-logistic_regression_rhs" => bm(50), "normal_5-normal_mixture_k" => bm(3; failed = true)))
    path = tempname() * ".toml"; open(path, "w") do io; TOML.print(io, receipt; sorted = true); end
    # @eval contract: both plot fns return Markdown.MD against a real receipt, and a note when absent.
    @test RKD.render_all80_batch1_coverage_plot(path) isa Markdown.MD
    @test RKD.render_all80_batch1_speedup_plot(path) isa Markdown.MD
    @test RKD.render_all80_batch1_coverage_plot(tempname() * ".toml") isa Markdown.MD
    # data contract: 4 present; diamonds is the workload-mismatch series.
    models = RKD._all80_models(path)
    @test length(models) == 4
    @test "diamonds-diamonds" in RKD._ALL80_WORKLOAD_MISMATCH
    dia = [r for r in RKD._all80_speedup_rows(models) if r.model == "diamonds-diamonds"]
    @test !isempty(dia) && all(r -> occursin("Mb", r.workload) || occursin("mismatch", r.workload), dia)
    # a failed reactant cell is SURFACED in coverage, not ranked.
    cov = RKD._all80_reactant_coverage_rows(models)
    @test any(r -> r.operation == "gradient" && r.outcome == "failed (diagnostic recorded)", cov)
    # path isolation.
    @test RKD._ALL80_BATCH1_PATH != RKD._ALL80_BENCHMARK_PATH
end
