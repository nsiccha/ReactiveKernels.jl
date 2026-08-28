using Test

@testset "compiler capability page is navigated, explicit, and code-free" begin
    root = joinpath(@__DIR__, "..")
    page_path = joinpath(root, "docs", "src", "compiler.md")
    make_path = joinpath(root, "docs", "make.jl")
    index_path = joinpath(root, "docs", "src", "index.md")

    @test isfile(page_path)
    page = read(page_path, String)
    make = read(make_path, String)
    index = read(index_path, String)

    @test occursin("\"Compiler capability and limits\" => \"compiler.md\"", make)
    @test occursin("compiler.md", index)

    nuts = read(joinpath(root, "docs", "src", "nuts.md"), String)
    @test occursin("not an RK package API", nuts)
    @test occursin("Transitional and removal-bound, not an intended RK API", nuts)

    for heading in (
        "## The stateless compiler",
        "## Batch and replica lowering",
        "## Incremental and compiled reactive execution",
        "## Source-captured method compiler",
        "## What the NUTS proof does and does not establish",
        "## Definitive support matrix",
    )
        @test occursin(heading, page)
    end

    for contract in (
        "branch-and-bound",
        "availability-based Kahn ordering",
        "Structural CSE is explicit and conservative",
        "does not call\n`code_lowered`",
        "Validity changes are exception-safe but values are not transactionally rolled\nback",
        "external compilation examples and acceptance evidence",
        "currently exported `nuts_state` / `CompiledNUTSState` compatibility path",
        "NUTS, log-density, or PPL domain API",
        "General Julia compiler replacement",
    )
        @test occursin(contract, page)
    end

    # The compiler specification is deliberately algorithmic prose plus an API
    # table. Executable/source examples belong on the focused example pages.
    @test !occursin("```", page)
    @test !occursin("~~~", page)
end
