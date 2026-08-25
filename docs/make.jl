using Documenter, DocumenterVitepress, ReactiveKernels

makedocs(
    sitename = "ReactiveKernels.jl",
    modules  = [ReactiveKernels],
    format   = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/ReactiveKernels.jl",
        devurl = "dev",
        devbranch = "dev",
    ),
    pages = [
        "Home" => "index.md",
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
    devbranch = "dev",
    push_preview = true,
)
