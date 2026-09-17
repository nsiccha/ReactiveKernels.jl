using LinearAlgebra
using Test

include(joinpath(@__DIR__, "..", "examples", "manual_derivative_rule.jl"))
using .ManualDerivativeRuleExample

@testset "one pure graph supplies pruned primal, JVP, and VJP cuts" begin
    result = ManualDerivativeRuleExample.run()

    @test result.y == result.y_forward == result.y_reverse == result.y_x_only
    @test result.y == [-1.5, -2.5]
    @test result.pullback_A_bar == result.A_bar
    @test result.pullback_x_bar == result.x_bar == result.x_only_bar
    @test result.adjoint_pair[1] ≈ result.adjoint_pair[2]

    @test result.recipe_ids == (
        primal = (1,),
        forward = (1, 2, 3, 4),
        reverse = (5, 6),
        x_reverse = (6,),
    )
    @test result.captured_fields == (
        both = (:vjp, :A, :x),
        x_only = (:vjp, :A),
    )

    A = copy(EXAMPLE_INPUTS.A)
    x = copy(EXAMPLE_INPUTS.x)
    y, pullback = value_and_pullback(A, x)
    @test pullback.A === A
    @test pullback.x === x
    @test y == [-1.5, -2.5]

    source = read(joinpath(@__DIR__, "..", "examples",
                           "manual_derivative_rule.jl"), String)
    graph_start = first(findfirst("@kernel matvec_rule(", source))
    graph_stop = first(findfirst(
        "# -- END DOCS: pure mathematical derivative rule --", source,
    ))
    graph_source = source[graph_start:(graph_stop - 1)]
    for backend_term in (
            "ChainRules", "Mooncake", "Enzyme", "NoTangent", "Pullback",
        )
        @test !occursin(backend_term, graph_source)
    end

    docs_page = read(joinpath(@__DIR__, "..", "docs", "src",
                              "manual-derivative-rules.md"), String)
    docs_make = read(joinpath(@__DIR__, "..", "docs", "make.jl"), String)
    docs_helpers = read(joinpath(@__DIR__, "..", "docs",
                                 "kernel_examples.jl"), String)
    @test occursin("Executable design example, not a shipped adapter generator",
                   docs_page)
    @test occursin("examples/manual_derivative_rule.jl", docs_page)
    @test occursin("render_manual_derivative_rule_cuts()", docs_page)
    @test occursin("render_manual_derivative_pullback_source()", docs_page)
    @test occursin(
        "\"Manual derivative rules (design)\" => \"manual-derivative-rules.md\"",
        docs_make,
    )
    @test occursin("function render_manual_derivative_rule_cuts()", docs_helpers)
    @test occursin("function render_manual_derivative_pullback_source()",
                   docs_helpers)
end
