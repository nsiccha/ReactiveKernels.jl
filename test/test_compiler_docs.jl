using Test

_compiler_docs_lf(text) = replace(text, "\r\n" => "\n", "\r" => "\n")

@testset "compiler capability page is navigated, explicit, and prose-first" begin
    root = joinpath(@__DIR__, "..")
    page_path = joinpath(root, "docs", "src", "compiler.md")
    make_path = joinpath(root, "docs", "make.jl")
    index_path = joinpath(root, "docs", "src", "index.md")

    @test isfile(page_path)
    page = _compiler_docs_lf(read(page_path, String))
    make = _compiler_docs_lf(read(make_path, String))
    index = _compiler_docs_lf(read(index_path, String))

    @test occursin("\"Compiler capability and limits\" => \"compiler.md\"", make)
    @test occursin("compiler.md", index)

    nuts = _compiler_docs_lf(read(joinpath(root, "docs", "src", "nuts.md"), String))
    @test occursin("not an RK package API", nuts)
    @test occursin("external NUTS compiler-acceptance exemplar", nuts)
    @test occursin("The former `@rk_pure`,", nuts)
    @test occursin("declarations have been removed", nuts)
    @test occursin("The verbatim source below is the macro-free executable fixture", nuts)

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
        "a bare `using ReactiveKernels` does not load or\nexport them",
        "former `@rk_pure`, `@rk_borrows`, and `@rk_rng` declarations have\nbeen removed",
        "`@node` is unrelated to this removal boundary",
        "NUTS, log-density, and\nPPL artifacts are external compilation examples and acceptance evidence",
        "not** a general Julia-recursion compiler proof",
    )
        @test occursin(contract, page)
    end

    # The compiler specification is deliberately algorithmic prose plus three
    # build-executed result/API panels. Source-code examples remain on the
    # focused example pages.
    @test count(==("```@eval"), split(page, '\n')) == 3
    @test count(==("```"), split(page, '\n')) == 3
    @test !occursin("```julia", page)
    @test !occursin("~~~", page)
end
