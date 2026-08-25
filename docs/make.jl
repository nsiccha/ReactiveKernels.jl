using Documenter, DocumenterVitepress, ReactiveKernels

include("kernel_examples.jl")
include(joinpath(@__DIR__, "..", "examples", "distributions.jl"))

makedocs(
    sitename = "ReactiveKernels.jl",
    modules  = [ReactiveKernels],
    format   = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/ReactiveKernels.jl",
        devurl = "dev",
        devbranch = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Visualization" => "visualization.md",
        "Distribution log densities" => "distributions.md",
        "Eight schools" => "eight-schools.md",
        "NUTS sampling" => "nuts.md",
        "Online statistics" => "online-stats.md",
        "Non-allocating kernels" => "nonallocating.md",
        "API"  => "api.md",
    ],
    checkdocs = :none,
    warnonly = true,
)

# Ensure a root index.html redirect exists
let redirect = joinpath(@__DIR__, "build", "index.html")
    isfile(redirect) || write(redirect, """
    <!DOCTYPE html>
    <html><head>
    <meta http-equiv="refresh" content="0; url=dev/">
    </head><body>Redirecting to <a href="dev/">dev</a>...</body></html>
    """)
end

DocumenterVitepress.deploydocs(
    repo = "github.com/nsiccha/ReactiveKernels.jl",
    devbranch = "main",
    push_preview = true,
)
