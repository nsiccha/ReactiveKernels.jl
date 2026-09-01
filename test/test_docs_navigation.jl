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
        "\"Kernel execution and tools\" => [",
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
end
