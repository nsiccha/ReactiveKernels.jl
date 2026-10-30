using Test

_compiler_docs_lf(text) = replace(text, "\r\n" => "\n", "\r" => "\n")

@testset "compiler capability page is navigated, explicit, and prose-first" begin
    root = joinpath(@__DIR__, "..")
    page_path = joinpath(root, "docs", "src", "compiler.md")
    make_path = joinpath(root, "docs", "make.jl")
    index_path = joinpath(root, "docs", "src", "index.md")
    readme_path = joinpath(root, "README.md")
    ad_path = joinpath(root, "docs", "src", "automatic-differentiation.md")
    manual_rules_path = joinpath(
        root, "docs", "src", "manual-derivative-rules.md",
    )
    distributions_ad_path = joinpath(root, "docs", "src", "distributions-ad.md")
    batched_path = joinpath(root, "docs", "src", "batched.md")

    @test isfile(page_path)
    @test isfile(ad_path)
    @test isfile(manual_rules_path)
    @test isfile(distributions_ad_path)
    page = _compiler_docs_lf(read(page_path, String))
    make = _compiler_docs_lf(read(make_path, String))
    index = _compiler_docs_lf(read(index_path, String))
    readme = _compiler_docs_lf(read(readme_path, String))
    ad_docs = _compiler_docs_lf(read(ad_path, String))
    manual_rules_docs = _compiler_docs_lf(read(manual_rules_path, String))
    distributions_ad_docs = _compiler_docs_lf(read(distributions_ad_path, String))
    batched_docs = _compiler_docs_lf(read(batched_path, String))

    # Preparation-time partial evaluation is a public compiler capability,
    # with one prose authority that covers its API shape and frozen-data
    # semantics across ordinary, nonallocating, and AD preparation.
    for marker in (
            "### Preparation-time partial evaluation",
            "bound = (; observations)",
            "`Value => data` pair",
            "executes exactly once during\npreparation",
            "Rebind by preparing again",
            "`prepare_nonallocating`, whose",
        )
        @test occursin(marker, page)
    end

    for marker in (
            "## A plate is a pure RK subgraph",
            "narrowest valid loop\nboundary",
            "cannot enter a\nplate",
            "It is not\n`Distributions.jl.Normal`",
            "`plate(eachcol(logits), y) do column, observed ... end`",
        )
        @test occursin(marker, batched_docs)
    end

    # The exported prepared value+in-place-gradient surface has one public
    # prose authority. It shows the kernel, the interaction, caller-owned
    # storage, and Constant rebinding in a build-executed example.
    for marker in (
            "@kernel objective",
            "ad_value_and_gradient!",
            "caller-owned",
            "returned_gradient === gradient_buffer",
            "Constant",
            "AutoEnzyme(; mode = Enzyme.Reverse)",
            "test_batched_nonallocating.jl",
            "## Freeze data-only work during preparation",
            "bound = (; data)",
            "data-only prefix executes exactly once",
            "Rebind by preparing again",
            "## Jacobian-vector and Hessian-vector products",
            "prepare_ad_pushforward",
            "prepare_ad_hvp",
            "ad_gradient_and_hvp",
            "### Third-order directional derivatives",
            "prepare_ad_third",
            "ad_gradient_hvp_and_third",
            "test_ad_tangent_operators.jl",
        )
        @test occursin(marker, ad_docs)
    end
    @test occursin("Main.BatchedExamples.BATCHED_AD_SOURCE", distributions_ad_docs)
    @test occursin("test_batched_nonallocating.jl", distributions_ad_docs)
    for marker in (
            "# Derivative rules from one pure-math graph",
            "## Generated scalar rules",
            "## Generated vector rules",
            "## Two-stage reverse staging",
            "## Generated adapters",
            "RK has neither reverse-mode AD nor program transposition",
            "prepare_ad_pullback",
            "does not consume a rule graph",
            "render_manual_derivative_rule_cuts()",
            "render_manual_derivative_rule_source()",
            "result.residuals",
        )
        @test occursin(marker, manual_rules_docs)
    end

    @test occursin("\"Compiler capability and limits\" => \"compiler.md\"", make)
    @test occursin("\"Automatic differentiation\" => [", make)
    @test occursin("\"Prepared gradients\" => \"automatic-differentiation.md\"", make)
    @test occursin(
        "\"Derivative rules\" => \"manual-derivative-rules.md\"",
        make,
    )
    @test occursin("compiler.md", index)
    @test occursin("warnonly = false", make)

    @test !occursin("adaptive NUTS remains CPU-only", readme)
    @test !occursin("stays on the CPU because", index)

    for heading in (
        "## Pure kernels",
        "## The stateless compiler",
        "## Batch and replica lowering",
        "## Incremental and compiled reactive execution",
        "## Definitive support matrix",
    )
        @test occursin(heading, page)
    end

    for contract in (
        "branch-and-bound",
        "availability-based Kahn ordering",
        "Structural CSE is explicit and conservative",
        "narrowest valid nested-loop boundary",
        "Effectful\nrecipes are excluded by the planner and cannot form a plate plan",
    )
        @test occursin(contract, page)
    end

    # The compiler specification keeps its four build-executed result/API
    # panels and allows ordinary source examples for its inspection APIs.
    # Every opening fence still needs a closing fence.
    @test count(==("```@eval"), split(page, '\n')) == 4
    @test count(==("```"), split(page, '\n')) ==
          count(line -> startswith(line, "```") && line != "```", split(page, '\n'))
    @test !occursin("~~~", page)
end
