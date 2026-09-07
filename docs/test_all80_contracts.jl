# Rendered-contract skeleton for the all-82 comparison tables (docs/all80_comparison_tables.jl).
# Asserts each of the three tables renders as a sortable-table artifact carrying the SIX
# MANDATORY columns and one row per model, and that the underlying receipt passes the
# All80Receipt publication gate (every mandatory cell a real number for all 82).
#
# Skeleton: it runs against a real all80-benchmark-v1 receipt once one exists
# (RK_ALL80_RECEIPT env or benchmark/receipts/all80-benchmark-v1.toml). While the receipt
# does not yet exist the testset SKIPS with an explanatory message — acceptable ONLY as a
# temporary scaffold. At publication the docs build MUST set RK_ALL80_REQUIRE_RECEIPT=1,
# which makes a missing receipt a HARD FAILURE (fail-closed): the page cannot ship without
# a populated 82-row receipt.
using Test
import TOML

const _ALL80_RECEIPT = get(ENV, "RK_ALL80_RECEIPT",
    normpath(joinpath(@__DIR__, "..", "benchmark", "receipts", "all80-benchmark-v1.toml")))

# The six MANDATORY column headers that must appear in the rendered tables.
const _ALL80_MANDATORY_HEADERS = (
    "idiomatic RK", "RK + Reactant", "upstream Turing", "reference Stan",  # primal+gradient
    "RK native", "AHMC + Turing",                                          # HMC
)

_all80_html(node) = sprint(show, MIME"text/html"(), node)

function _all80_assert_table(node, id, n_rows, headers)
    html = _all80_html(node)
    @test occursin("data-rk-artifact-kind=\"sortable-table\"", html)
    @test occursin("table:" * id, html)
    for hdr in headers
        @test occursin(hdr, html)
    end
    # header row + one row per model
    @test count("<tr", html) >= n_rows + 1
end

const _ALL80_REQUIRE = get(ENV, "RK_ALL80_REQUIRE_RECEIPT", "") == "1"

@testset "all-82 rendered comparison contracts" begin
    if !isfile(_ALL80_RECEIPT)
        if _ALL80_REQUIRE
            @error "all80 receipt REQUIRED for publication but missing at $_ALL80_RECEIPT"
            @test isfile(_ALL80_RECEIPT)   # fail-closed
        else
            @info "all80 contract test SKIPPED — no receipt yet at $_ALL80_RECEIPT (set RK_ALL80_REQUIRE_RECEIPT=1 to fail-closed)"
            @test_skip isfile(_ALL80_RECEIPT)
        end
    else
        include(joinpath(@__DIR__, "..", "benchmark", "all80_receipt.jl"))
        issues = Main.All80Receipt.validate(_ALL80_RECEIPT; expected_models = 82)
        @test isempty(issues)
        isempty(issues) || foreach(i -> @info("all80 receipt issue: $i"), issues)

        models = get(TOML.parsefile(_ALL80_RECEIPT), "models", Dict())
        n = length(models)
        _all80_assert_table(all80_primal_table(models), "all80-primal", n,
            ("idiomatic RK", "RK + Reactant", "upstream Turing", "reference Stan"))
        _all80_assert_table(all80_gradient_table(models), "all80-gradient", n,
            ("idiomatic RK", "RK + Reactant", "upstream Turing", "reference Stan"))
        _all80_assert_table(all80_hmc_table(models), "all80-hmc", n,
            ("RK native", "RK + Reactant", "AHMC + Turing"))
    end
end
