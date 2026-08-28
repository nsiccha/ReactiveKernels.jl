using Documenter, DocumenterVitepress, ReactiveKernels

include("kernel_examples.jl")
include("check_rendered.jl")
include(joinpath(@__DIR__, "..", "examples", "distributions.jl"))
include(joinpath(@__DIR__, "..", "examples", "batched.jl"))

site_pages = [
    "Home" => "index.md",
    "Compiler capability and limits" => "compiler.md",
    "Building blocks" => [
        "Distribution log densities" => "distributions.md",
        "Batched log densities" => "batched.md",
        "Non-allocating kernels" => "nonallocating.md",
    ],
    "Probabilistic programming" => [
        "Eight schools" => "eight-schools.md",
        "Linear regression" => "linear-regression.md",
        "Beta-binomial" => "beta-binomial.md",
        "Poisson-Gamma" => "poisson-gamma.md",
        "Dugongs (nonlinear growth)" => "dugongs-growth.md",
        "ARMA(1,1) time series" => "arma11.md",
        "Gaussian mixture" => "gaussian-mixture.md",
    ],
    "Sampling" => [
        "NUTS sampling" => "nuts.md",
        "Online statistics" => "online-stats.md",
    ],
    "Visualization" => "visualization.md",
    "API" => "api.md",
]

makedocs(
    sitename = "ReactiveKernels.jl",
    repo = "https://github.com/nsiccha/ReactiveKernels.jl",
    modules  = [ReactiveKernels],
    format   = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/ReactiveKernels.jl",
        devurl = "dev",
        devbranch = "main",
    ),
    pages = site_pages,
    checkdocs = :none,
    # Build-executed examples must fail closed instead of silently losing their panel.
    warnonly = Documenter.except(:eval_block),
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

check_rendered_docs(joinpath(@__DIR__, "build"), site_pages)

DocumenterVitepress.deploydocs(
    repo = "github.com/nsiccha/ReactiveKernels.jl",
    devbranch = "main",
    push_preview = true,
)
