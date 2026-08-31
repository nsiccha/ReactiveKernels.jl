using Test

@testset "WALNUTS-D mathematical source is wired into rendered docs" begin
    root = joinpath(@__DIR__, "..")
    fixture = read(joinpath(root, "benchmark",
                            "walnuts_kernel_authoring_fixture.jl"), String)
    page = read(joinpath(root, "docs", "src", "walnuts.md"), String)
    helpers = read(joinpath(root, "docs", "kernel_examples.jl"), String)
    make = read(joinpath(root, "docs", "make.jl"), String)
    rendered_check = read(joinpath(root, "docs", "check_rendered.jl"), String)
    interactions = read(
        joinpath(root, "benchmark", "reactivehmc_docs_interactions.jl"), String,
    )

    for source_marker in (
        "@kernel walnuts_state(init; step_f, macro_time,",
        "macro_step!(ep) = begin",
        "reverse_num_steps = div(reverse_num_steps, 2)",
        "step!(directions, exponentials) = begin",
        "accepted = macro_step!(__self__, ep)",
        "@kernel walnuts!!(state; momentum, directions, exponentials)",
    )
        @test occursin(source_marker, fixture)
        @test occursin(source_marker, helpers) ||
              occursin(source_marker, rendered_check)
    end

    @test occursin("render_walnuts_source(:state)", page)
    @test occursin("render_walnuts_source(:macro_step)", page)
    @test occursin("render_walnuts_source(:nuts_step)", page)
    @test occursin("render_walnuts_source(:nuts_leaf)", page)
    @test occursin("render_walnuts_source(:entry)", page)
    @test occursin("render_walnuts_source_interaction()", page)
    @test occursin("render_walnuts_complete_source()", page)
    @test occursin("WALNUTS_INSPECTION", interactions)
    @test occursin("captured_method_count", interactions)
    @test occursin("max_depth = 10", interactions)
    @test occursin("proposal_capacity", interactions)
    @test occursin("tree_capacity", interactions)
    @test occursin("structural_container_roundtrip = true", interactions)
    @test occursin("recursive_scc", interactions)
    @test occursin("compiler_frontier_executed = true", interactions)
    @test occursin(
        "recursive functional state-machine SCC lowering is not implemented",
        interactions,
    )
    @test occursin("for root `step!`", interactions)
    @test occursin("compiler_execution_claimed = false", interactions)
    @test occursin("does not claim full", page)
    @test occursin("recursive-SCC or WALNUTS compiler support", page)
    @test occursin("\"WALNUTS-D mathematical kernel\" => \"walnuts.md\"", make)
    @test occursin("walnuts_kernel_authoring_fixture.jl", make)
    @test occursin("walnuts_compiler_support.jl", make)
    @test occursin("\"walnuts.md\" =>", rendered_check)
end
