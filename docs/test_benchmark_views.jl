using Test

include(joinpath(@__DIR__, "kernel_examples.jl"))
Base.include(ReactiveKernelsDocs, joinpath(@__DIR__, "result_views.jl"))

function artifact_contract(result)
    blocks = result.content
    raw = [block.content for block in blocks
           if block isa ReactiveKernelsDocs.RawHTML]
    plots = filter(html -> occursin("artifact-kind=\"aov-panel\"", html), raw)
    tables = filter(html -> occursin("artifact-kind=\"sortable-table\"", html), raw)
    ids = String[]
    for html in raw
        match_result = match(r"data-rk-artifact-id=\"([^\"]+)\"", html)
        isnothing(match_result) || push!(ids, only(match_result.captures))
    end
    (; plots, tables, ids)
end

const FOCUSED_BENCHMARK_VIEWS = (
    :render_distribution_benchmarks => (2, 2),
    :render_distribution_gradient_benchmarks => (9, 9),
    :render_distribution_amortization => (1, 1),
    :render_eval_throughput_amortization => (3, 3),
    :render_scalar_gallery_benchmarks => (7, 7),
    :render_structured_distribution_benchmarks => (1, 1),
    :render_batched_benchmarks => (2, 2),
    :render_eight_schools_primal_benchmarks => (4, 4),
    :render_eight_schools_ad_benchmarks => (4, 4),
    :render_sum_to_zero_benchmarks => (4, 4),
    :render_nuts_g7_benchmark => (1, 1),
    :render_nuts_reactant_benchmark => (1, 2),
    :render_eight_schools_reactant_benchmark => (4, 4),
    :render_eight_schools_reactant_ad_benchmark => (3, 3),
    :render_mnist_logistic_benchmarks => (4, 4),
    :render_mnist_logistic_wren_benchmarks => (4, 4),
    :render_mnist_logistic_ad_benchmarks => (3, 3),
    :render_mnist_logistic_ad_wren_benchmarks => (3, 3),
    :render_mnist_reactant_benchmark => (4, 4),
    :render_mnist_reactant_wren_benchmark => (4, 4),
    :render_mnist_reactant_ad_benchmark => (3, 3),
    :render_mnist_reactant_ad_wren_benchmark => (3, 3),
    :render_probprog_mcmc_benchmark => (1, 3),
    :render_practicalbayes_eight_schools_benchmark => (0, 1),
    :render_practicalbayes_mnist_benchmark => (0, 1),
    :render_practicalbayes_eval_benchmark => (0, 1),
    :render_practicalbayes_mcmc_benchmark => (0, 1),
)

@testset "focused benchmark renderer inventory" begin
    all_ids = String[]
    for (name, (expected_plots, expected_tables)) in FOCUSED_BENCHMARK_VIEWS
        contract = artifact_contract(getfield(ReactiveKernelsDocs, name)())
        @test length(contract.plots) == expected_plots
        @test length(contract.tables) == expected_tables
        @test length(contract.ids) == expected_plots + expected_tables
        @test length(unique(contract.ids)) == length(contract.ids)
        append!(all_ids, contract.ids)
    end

    throughput = artifact_contract(ReactiveKernelsDocs.eval_throughput_chart())
    @test length(throughput.plots) == 3
    @test length(throughput.tables) == 3
    @test length(throughput.ids) == 6
    append!(all_ids, throughput.ids)

    @test length(unique(all_ids)) == length(all_ids)
    @test all(html -> occursin("Runtime ÷ baseline", html),
              artifact_contract(
                  ReactiveKernelsDocs.render_eight_schools_ad_benchmarks()).tables)
    @test all(html -> occursin("Runtime ÷ baseline", html), throughput.tables)
end
