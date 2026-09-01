using Documenter, DocumenterVitepress, ReactiveKernels
using ReactiveKernelsBatchingExamples
using ReactiveKernelsCompatibilityExamples
using ReactiveKernelsDistributionKernels
using ReactiveKernelsKernelExamples
using ReactiveKernelsHMCDiagnostics
using ReactiveKernelsNUTSExamples
using ReactiveKernelsPPLExamples
using ReactiveKernelsStreamingStats

# Opt into the standalone nutpie/nuts-rs diagonal-adaptation compiler corpus.
# It remains an external mathematical example rather than a package API.
include(joinpath(@__DIR__, "..", "examples", "nutpie_diagonal_adaptation.jl"))
import .NutpieDiagonalAdaptationExample

# NUTS and WALNUTS authoring fixtures are deliberately not included here.
# Their pages read the source files as inert text; docs builds must not parse,
# lower, compile, or execute either moving compiler/runtime surface.

# Load the accepted ReactiveHMC compiler-corpus authorities themselves. The
# corpus page extracts source from these exact files and inspects their captured
# MethodIR; malformed or rejected source therefore fails the docs build.
include(joinpath(@__DIR__, "..", "benchmark", "reactivehmc_algorithm_corpus.jl"))
include(joinpath(@__DIR__, "..", "benchmark", "reactivehmc_rke_kernel_fixture.jl"))
include(joinpath(@__DIR__, "..", "benchmark", "reactivehmc_rke_functional_lowering.jl"))
include(joinpath(@__DIR__, "..", "benchmark", "reactivehmc_integrator_kernel_fixture.jl"))
include(joinpath(@__DIR__, "..", "benchmark", "reactivehmc_statistics_kernel_fixture.jl"))
include(joinpath(@__DIR__, "..", "benchmark", "reactivehmc_hmc_kernel_fixture.jl"))
include(joinpath(@__DIR__, "..", "benchmark", "reactivehmc_docs_interactions.jl"))

include("kernel_examples.jl")
Base.include(ReactiveKernelsDocs, joinpath(@__DIR__, "result_views.jl"))
include("check_rendered.jl")

site_pages = [
    "Home" => "index.md",
    "Compiler capability and limits" => "compiler.md",
    "Distributions" => [
        "Distribution kernels" => "distributions.md",
        "Batched log densities" => "batched.md",
    ],
    "Building blocks" => [
        "Non-allocating kernels" => "nonallocating.md",
    ],
    "Automatic differentiation" => [
        "Prepared gradients" => "automatic-differentiation.md",
        "Evaluation throughput vs Turing.jl" => "eval-throughput.md",
    ],
    "Probabilistic programming" => [
        "Bijectors and constrained parameters" => "bijectors.md",
        "Eight schools" => "eight-schools.md",
        "Eight schools with Reactant" => "eight-schools-reactant.md",
        "Linear regression" => "linear-regression.md",
        "Beta-binomial" => "beta-binomial.md",
        "Poisson-Gamma" => "poisson-gamma.md",
        "Dugongs (nonlinear growth)" => "dugongs-growth.md",
        "ARMA(1,1) time series" => "arma11.md",
        "Gaussian mixture" => "gaussian-mixture.md",
        "MNIST multinomial logistic" => "mnist-logistic.md",
        "MNIST with Reactant" => "mnist-reactant.md",
    ],
    "Sampling" => [
        "Pathfinder approximation" => "pathfinder.md",
        "ReactiveHMC kernel corpus" => "reactivehmc-corpus.md",
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
        ReactiveKernelsNUTSExamples,
        ReactiveKernelsStreamingStats,
        ReactiveKernelsHMCDiagnostics,
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
ReactiveKernelsDocs.assert_reactivehmc_docs_interacted!()

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
