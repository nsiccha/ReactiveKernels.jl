using Documenter, DocumenterVitepress, ReactiveKernels
using ReactiveKernelsPPL, ReactiveKernelsDistributionKernels, AlgebraOfVega

include("pages.jl")
include("lint.jl")
lint_docs(SITE_PAGES)
include("Tutorials.jl")

makedocs(
    sitename = "ReactiveKernels.jl",
    repo = Documenter.Remotes.GitHub("nsiccha", "ReactiveKernels.jl"),
    modules = [ReactiveKernels, ReactiveKernelsPPL, ReactiveKernelsDistributionKernels],
    plugins = [Tutorials.BuildAssets()],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/ReactiveKernels.jl",
        devurl = "dev",
        devbranch = "main",
    ),
    pages = SITE_PAGES,
    checkdocs = :exports,
    warnonly = false,
)

DocumenterVitepress.deploydocs(
    repo = "github.com/nsiccha/ReactiveKernels.jl",
    devbranch = "main",
    push_preview = true,
)
