using Documenter, DocumenterVitepress, ReactiveKernels

include("kernel_examples.jl")
include(joinpath(@__DIR__, "..", "examples", "distributions.jl"))
include(joinpath(@__DIR__, "..", "examples", "batched.jl"))

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
