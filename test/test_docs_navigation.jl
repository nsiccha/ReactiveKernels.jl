using Test

@testset "documentation source contracts" begin
    docs = joinpath(@__DIR__, "..", "docs")
    include(joinpath(docs, "pages.jl"))
    include(joinpath(docs, "lint.jl"))
    @test !isempty(lint_docs(SITE_PAGES; root=docs))
end
