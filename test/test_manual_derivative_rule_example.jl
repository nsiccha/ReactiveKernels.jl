using LinearAlgebra
using Test

include(joinpath(@__DIR__, "..", "examples", "manual_derivative_rule.jl"))
# `import` (not `using`): keep the example module's exports (`run`, `matvec`)
# out of Main and qualify below, so the fixture is robust to suite order.
import .ManualDerivativeRuleExample

@testset "one pure graph generates the vector rule and its selected cuts" begin
    example = ManualDerivativeRuleExample
    result = example.run()
    (; A, x, A_dot, x_dot, y_bar) = example.EXAMPLE_INPUTS

    @test example.matvec isa ReactiveKernels.DerivativeRule
    @test ReactiveKernels.rule_inputs(example.matvec) == (:A, :x)
    @test result.y == result.y_forward == result.prepared.y == A * x
    @test result.y == [-1.5, -2.5]
    @test result.y_dot == A_dot * x + A * x_dot
    @test (result.y, result.y_dot) == result.prepared.forward
    @test (result.A_bar, result.x_bar) == result.prepared.reverse
    @test result.A_bar == y_bar * transpose(x)
    @test result.x_bar == result.x_only_bar == result.prepared.x_reverse ==
        transpose(A) * y_bar
    @test result.adjoint_pair[1] ≈ result.adjoint_pair[2]

    # The generator's residual sets: A_bar reads x only, x_bar reads A only.
    @test result.residuals == (
        A_only = (false, true),
        x_only = (true, false),
        both = (true, true),
    )
    # The staged residual sets and the two-stage reverse result.
    @test result.staged_residuals == (A_only = (:x,), x_only = (:A,), both = (:A, :x))
    @test result.staged_y == result.y && result.staged_x_bar == result.x_bar
    @test result.recipe_ids == (
        primal = (1,),
        forward = (1, 2, 3, 4),
        reverse = (5, 6),
        x_reverse = (6,),
    )

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
    @test occursin("const matvec = derivative_rule(", source)
    @test occursin("# -- END DOCS: generated rule --", source)

    docs_page = read(joinpath(@__DIR__, "..", "docs", "src",
                              "manual-derivative-rules.md"), String)
    docs_make = read(joinpath(@__DIR__, "..", "docs", "make.jl"), String)
    docs_helpers = read(joinpath(@__DIR__, "..", "docs",
                                 "kernel_examples.jl"), String)
    @test occursin("# Derivative rules from one pure-math graph", docs_page)
    @test occursin("examples/manual_derivative_rule.jl", docs_page)
    @test occursin("render_manual_derivative_rule_cuts()", docs_page)
    @test occursin("render_manual_derivative_rule_source()", docs_page)
    @test occursin(
        "\"Derivative rules\" => \"manual-derivative-rules.md\"",
        docs_make,
    )
    @test occursin("function render_manual_derivative_rule_cuts()", docs_helpers)
    @test occursin("function render_manual_derivative_rule_source()",
                   docs_helpers)
end
