using Test

@testset "review-first docs hierarchy and Reactant ownership" begin
    root = joinpath(@__DIR__, "..")
    make = read(joinpath(root, "docs", "make.jl"), String)

    top_level = (
        "\"Home\" => \"index.md\"",
        "\"Compiler capability and limits\" => \"compiler.md\"",
        "\"Distributions\" => [",
        "\"Probabilistic programming\" => [",
        "\"Automatic differentiation\" => [",
        "\"Reactant\" => [",
        "\"Non-allocating kernels\" => \"nonallocating.md\"",
        "\"Tools and reference\" => [",
        "\"Bijectors\" => \"bijectors.md\"",
        "\"Sampling (experimental)\" => [",
    )
    positions = map(top_level) do marker
        position = findfirst(marker, make)
        @test position !== nothing
        first(position)
    end
    @test issorted(positions)

    @test first(findfirst("\"Eight schools\" => \"eight-schools.md\"", make)) <
          first(findfirst("\"MNIST multinomial logistic\" => \"mnist-logistic.md\"", make)) <
          first(findfirst("\"Other model walkthroughs\" => [", make))
    @test !occursin("practicalbayes.md", make)
    @test !isfile(joinpath(root, "docs", "src", "practicalbayes.md"))

    for page in (
        "reactant.md", "distributions-reactant.md",
        "eight-schools-reactant.md", "mnist-reactant.md",
        "reactant-ad.md", "eval-throughput.md", "nuts-reactant.md",
    )
        @test occursin("\"$page\"", make)
    end

    distributions = read(joinpath(root, "docs", "src", "distributions.md"), String)
    batched = read(joinpath(root, "docs", "src", "batched.md"), String)
    reactant_distributions = read(
        joinpath(root, "docs", "src", "distributions-reactant.md"), String,
    )
    for renderer in (
        "render_scalar_gallery_benchmarks()",
        "render_structured_distribution_benchmarks()",
        "render_distribution_benchmarks()",
        "render_distribution_amortization()",
        "render_batched_benchmarks()",
    )
        @test !occursin(renderer, distributions)
        @test !occursin(renderer, batched)
        @test length(split(reactant_distributions, renderer)) - 1 == 1
    end

    eight_schools_reactant = read(
        joinpath(root, "docs", "src", "eight-schools-reactant.md"), String,
    )
    mnist_reactant = read(
        joinpath(root, "docs", "src", "mnist-reactant.md"), String,
    )
    @test occursin("[Eight Schools kernel page](eight-schools.md)",
                   eight_schools_reactant)
    @test occursin("[MNIST multinomial-logistic kernel page](mnist-logistic.md)",
                   mnist_reactant)
    @test !occursin("execute_ppl_example", eight_schools_reactant)
    @test !occursin("execute_ppl_example", mnist_reactant)

    nuts = read(joinpath(root, "docs", "src", "nuts.md"), String)
    walnuts = read(joinpath(root, "docs", "src", "walnuts.md"), String)
    @test occursin("Compiler/runtime execution is disabled in docs", nuts)
    @test occursin("Compiler status: not executed during the docs build", walnuts)
    @test occursin("NUTS source and receipts (not executed)", make)
    @test occursin("WALNUTS-D source (not executed)", make)

    @test !occursin("Building blocks", make)
    @test !occursin("Kernel execution and tools", make)

    review_states = Dict(
        :frozen_ppl => (
            "linear-regression.md", "beta-binomial.md", "poisson-gamma.md",
            "dugongs-growth.md", "arma11.md", "gaussian-mixture.md",
        ),
        :frozen_sampling => (
            "pathfinder.md", "reactivehmc-corpus.md", "nuts.md",
            "nutpie-diagonal.md", "walnuts.md", "online-stats.md",
            "nuts-reactant.md",
        ),
        :frozen_bijectors => ("bijectors.md",),
        :review_pending_nonallocating => ("nonallocating.md",),
    )
    for (state, pages) in review_states, page in pages
        body = read(joinpath(root, "docs", "src", page), String)
        marker = "render_review_status(:$state)"
        @test length(split(body, marker)) - 1 == 1
    end
    for page in ("eight-schools.md", "mnist-logistic.md")
        body = read(joinpath(root, "docs", "src", page), String)
        @test !occursin("render_review_status", body)
    end

    views = read(joinpath(root, "docs", "result_views.jl"), String)
    styles = read(
        joinpath(root, "docs", "src", ".vitepress", "theme", "overrides.css"),
        String,
    )
    @test occursin("data_rk_review_state = status.state", views)
    @test occursin("review status, not a known runtime failure", views)
    @test occursin(".rk-review-state--frozen", styles)
    @test occursin(".rk-review-state--review-pending", styles)

    rendered_guard = read(joinpath(root, "docs", "check_rendered.jl"), String)
    sortable_contract = match(
        r"(?s)expected_sortable_tables = Dict\((.*?)\n    \)", rendered_guard,
    ).captures[1]
    aov_contract = match(
        r"(?s)expected_aov_panels = Dict\((.*?)\n    \)", rendered_guard,
    ).captures[1]
    @test occursin("\"distributions.md\" => 1", sortable_contract)
    @test occursin("\"distributions-reactant.md\" => 13", sortable_contract)
    @test occursin("\"distributions.md\" => 2", aov_contract)
    @test occursin("\"distributions-reactant.md\" => 13", aov_contract)
end
