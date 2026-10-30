using Documenter, DocumenterVitepress, ReactiveKernels
using ReactiveKernelsBatchingExamples
using ReactiveKernelsCompatibilityExamples
using ReactiveKernelsDistributionKernels
using ReactiveKernelsKernelExamples
using ReactiveKernelsPPLExamples
using ReactiveKernelsPPL

# Load the executable derivative-rule example: one pure-math graph, the
# `derivative_rule` it generates, and the same cuts prepared explicitly so the
# page can render each one.
include(joinpath(@__DIR__, "..", "examples", "manual_derivative_rule.jl"))
import .ManualDerivativeRuleExample

include("kernel_examples.jl")
Base.include(ReactiveKernelsDocs, joinpath(@__DIR__, "result_views.jl"))
Base.include(ReactiveKernelsDocs, joinpath(@__DIR__, "rkppl_page.jl"))
Base.include(ReactiveKernelsDocs, joinpath(@__DIR__, "all80_comparison_tables.jl"))
Base.include(ReactiveKernelsDocs, joinpath(@__DIR__, "all80_benchmark_plots.jl"))
include("check_rendered.jl")
include("test_all80_contracts.jl")
include("gibbs_page.jl")

site_pages = [
    "Home" => "index.md",
    "Core constraints" => "constraints.md",
    "Compiler capability and limits" => "compiler.md",
    "Position batching (vectorize)" => "position-batching.md",
    "Sequential recurrences (scan)" => "scan.md",
    "Distributions" => [
        "Distribution kernels" => "distributions.md",
        "Batched log densities" => "batched.md",
    ],
    "Probabilistic programming" => [
        "Writing models with @rkppl" => "rkppl.md",
        "Smooths and HSGPs with @rkppl" => "rkppl-smooths.md",
        "Eight schools" => "eight-schools.md",
        "Sum-to-zero recovery" => "sum-to-zero.md",
        "MNIST multinomial logistic" => "mnist-logistic.md",
        "Other model walkthroughs" => [
            "Linear regression" => "linear-regression.md",
            "Beta-binomial" => "beta-binomial.md",
            "Poisson-Gamma" => "poisson-gamma.md",
            "Dugongs (nonlinear growth)" => "dugongs-growth.md",
            "ARMA(1,1) time series" => "arma11.md",
            "Imperial covid19 renewal model" => "covid19imperial.md",
            "GARCH(1,1)" => "garch11.md",
            "HMM (forward algorithm)" => "hmm-example.md",
            "Gaussian HMM (K states)" => "hmm-gaussian.md",
            "Input-output HMM" => "iohmm-reg.md",
            "Correlated (MvNormal) regression" => "mvnormal-regression.md",
            "Partial evaluation (bound data)" => "bound-regression.md",
            "Diamonds (brms centered regression)" => "diamonds.md",
            "Normal mixture (marginalized simplex)" => "normal-mixture-k.md",
            "Dogs (correlated avoidance learning)" => "dogs-nonhierarchical.md",
            "Logistic regression (regularized horseshoe)" => "logistic-regression-rhs.md",
            "LDA (latent Dirichlet allocation, ldaK2/ldaK5)" => "lda.md",
            "Neural RBM softmax (nn_rbm1b J10/J100)" => "nn-rbm.md",
            "Losscurve (insurance loss development)" => "losscurve-sislob.md",
            "Accel splines (brms penalized splines)" => "accel-splines.md",
            "State space (structural DLM)" => "state-space-stochastic.md",
            "Prophet (piecewise-trend forecasting)" => "prophet.md",
            "IRT 2PL (item response)" => "irt-2pl.md",
            "IRT 2PL + latent regression" => "2pl-latent-reg-irt.md",
            "IRT hierarchical 2PL (correlated items)" => "hier-2pl.md",
            "IRT GPCM (ordinal + latent regression)" => "gpcm-latent-reg-irt.md",
            "IRT rating scale (GRSM + latent regression)" => "grsm-latent-reg-irt.md",
            "GLMM1 (per-site Poisson-log GLMM)" => "glmm1.md",
            "BYM2 (spatial-Poisson convolution)" => "bym2-offset-only.md",
            "Bones (graded-response latent trait)" => "bones.md",
            "Multi-occupancy (marginalized)" => "multi-occupancy.md",
            "Lotka–Volterra adaptive ODE" => "lotka-volterra.md",
            "SIR with environmental bacteria" => "sir.md",
            "Soil incubation two-pool carbon" => "soil-incubation.md",
        ],
        "Gaussian processes" => [
            "GP regression (marginal)" => "gp-regr.md",
            "GP Poisson regression (latent)" => "gp-pois-regr.md",
            "Approximate GP (HSGP, distributional)" => "accel-gp.md",
            "BRM motorcycle (partial HSGP coordinates)" => "brm-hsgp.md",
            "Hierarchical GP" => "hierarchical-gp.md",
            "Kronecker-structured GP (grid)" => "kronecker-gp.md",
            "Basketball-drive HMM (hmm_drive_1)" => "hmm-drive-1.md",
        ],
        "Declarative @ppl front-end + Gibbs (experimental)" => "ppl-front-end.md",
    ],
    "Automatic differentiation" => [
        "Prepared gradients" => "automatic-differentiation.md",
        "Derivative rules" => "manual-derivative-rules.md",
        "Distributions: scalar and batched" => "distributions-ad.md",
        "PPL: Eight Schools and MNIST" => "ppl-ad.md",
    ],
    "Historical Reactant receipts" => [
        "Distributions: scalar and batched" => "distributions-reactant.md",
        "Probabilistic programming" => [
            "Eight Schools" => "eight-schools-reactant.md",
            "MNIST multinomial logistic" => "mnist-reactant.md",
        ],
        "Evaluation throughput vs Turing.jl" => "eval-throughput.md",
    ],
    "Non-allocating kernels" => "nonallocating.md",
    "posteriordb benchmark (RK vs Stan vs Turing)" => "posteriordb-comparison.md",
    "Tools and reference" => [
        "DAG visualization" => "visualization.md",
        "JAX target" => "jax.md",
        "API" => "api.md",
    ],
    "Bijectors" => "bijectors.md",
    "Sampling (experimental)" => [
        "Gibbs on reactive kernels" => "gibbs.md",
    ],
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
