using Documenter, DocumenterVitepress, ReactiveKernels
using ReactiveKernelsBatchingExamples
using ReactiveKernelsCompatibilityExamples
using ReactiveKernelsDistributionKernels
using ReactiveKernelsKernelExamples
using ReactiveKernelsPPLExamples

# Opt into the external NUTS/HMC compiler-acceptance exemplar for the sampling
# and online-statistics pages. ReactiveKernels itself deliberately does not load it.
include(joinpath(@__DIR__, "..", "examples", "nuts_runtime.jl"))
using .ReactiveKernelsNUTSExample

# Opt into the standalone nutpie/nuts-rs diagonal-adaptation compiler corpus.
# It remains an external mathematical example rather than a package API.
include(joinpath(@__DIR__, "..", "examples", "nutpie_diagonal_adaptation.jl"))
import .NutpieDiagonalAdaptationExample

# Parse the external WALNUTS-D mathematical authoring fixture during the docs
# build.  The docs renderer below reads the exact source file for display, while
# this include makes malformed or no-longer-admitted @kernel source fail the
# GitHub Pages build instead of publishing a stale code listing.
include(joinpath(@__DIR__, "..", "benchmark", "walnuts_kernel_authoring_fixture.jl"))

include("kernel_examples.jl")
Base.include(ReactiveKernelsDocs, joinpath(@__DIR__, "result_views.jl"))
include("check_rendered.jl")

site_pages = [
    "Home" => "index.md",
    "Compiler capability and limits" => "compiler.md",
    "Building blocks" => [
        "Distribution log densities" => "distributions.md",
        "Batched log densities" => "batched.md",
        "Non-allocating kernels" => "nonallocating.md",
        "Evaluation throughput vs Turing.jl" => "eval-throughput.md",
    ],
    "Probabilistic programming" => [
        "Bijectors and constrained parameters" => "bijectors.md",
        "Eight schools" => "eight-schools.md",
        "Linear regression" => "linear-regression.md",
        "Beta-binomial" => "beta-binomial.md",
        "Poisson-Gamma" => "poisson-gamma.md",
        "Dugongs (nonlinear growth)" => "dugongs-growth.md",
        "ARMA(1,1) time series" => "arma11.md",
        "Gaussian mixture" => "gaussian-mixture.md",
    ],
    "Sampling" => [
        "Pathfinder approximation" => "pathfinder.md",
        "NUTS sampling" => "nuts.md",
        "Nutpie diagonal adaptation" => "nutpie-diagonal.md",
        "WALNUTS-D mathematical kernel" => "walnuts.md",
        "Online statistics" => "online-stats.md",
    ],
    "Visualization" => "visualization.md",
    "API" => "api.md",
]

makedocs(
    sitename = "ReactiveKernels.jl",
    repo = "https://github.com/nsiccha/ReactiveKernels.jl",
    modules  = [
        ReactiveKernels,
        ReactiveKernelsDistributionKernels,
        ReactiveKernelsKernelExamples,
        ReactiveKernelsBatchingExamples,
        ReactiveKernelsPPLExamples,
        ReactiveKernelsCompatibilityExamples,
    ],
    format   = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/ReactiveKernels.jl",
        devurl = "dev",
        devbranch = "main",
    ),
    pages = site_pages,
    checkdocs = :none,
    # Treat every docs warning as an error; build-executed examples must fail closed.
    warnonly = false,
)

# A successful build must have executed and rendered every PPL walkthrough
# exactly once. This catches an omitted page/block even when no eval error fires.
ReactiveKernelsDocs.assert_ppl_examples_executed!()

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
