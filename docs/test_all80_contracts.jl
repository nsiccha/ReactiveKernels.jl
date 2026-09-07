# Fail-closed rendered contract for the published native all-82 checkpoint.
using Test
import TOML
# make.jl includes this file at Main top level while the table builders live in
# ReactiveKernelsDocs (via Base.include); import them explicitly so the contract
# holds under make.jl's own load order, not just inside the module.
using .ReactiveKernelsDocs: all80_gradient_table, all80_hmc_table, all80_primal_table

const _ALL80_RECEIPT = get(ENV, "RK_ALL80_RECEIPT",
    normpath(joinpath(@__DIR__, "..", "benchmark", "receipts",
        "all80-native-checkpoint-v1.toml")))

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

@testset "all-82 native-checkpoint rendered contracts" begin
    @test isfile(_ALL80_RECEIPT)
    receipt = TOML.parsefile(_ALL80_RECEIPT)
    @test get(receipt, "schema", "") == "all80-benchmark-v1"
    @test get(receipt, "phase", "") == "native"
    models = get(receipt, "models", Dict())
    @test length(models) == 82

    for (name, row) in models
        if haskey(row, "error")
            @test row["error"] isa AbstractString && !isempty(row["error"])
            continue
        end
        for cell in ("primal_rk", "primal_turing", "primal_stan",
                     "gradient_turing", "gradient_stan", "hmc_ahmc_turing")
            @test _finite_number(get(row, cell, nothing))
        end
        for cell in ("gradient_rk", "hmc_rk_native",
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
