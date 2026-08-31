module MNISTLogisticExample

using ReactiveKernels
using ..ReactiveKernelsPPLExamples: _evaluate_ppl_source

export NUM_CLASSES
export MNIST_LOGISTIC_X, MNIST_LOGISTIC_Y
export build_mnist_logistic_graph, mnist_logistic_fixture, demo
export MNIST_LOGISTIC_SOURCE, evaluate_mnist_logistic_source

# Small real-MNIST fixture (first eight training images) for the docs example and
# tests. The BENCHMARK loads full-resolution real MNIST via MLDatasets; the model
# graph below is data-agnostic (X, y, and the class count are HAVE ports), so the
# same authored source is exercised on the fixture here and on full MNIST there.
include("mnist_logistic_fixture.jl")

const NUM_CLASSES = 10
const _PIXELS = 28 * 28

"Fixture design matrix: `n × 784` in `[0, 1]`, one flattened image per row."
const MNIST_LOGISTIC_X =
    Matrix{Float64}(transpose(reshape(_MNIST_FIXTURE_PIXELS, _PIXELS, _MNIST_FIXTURE_N))) ./ 255
"Fixture labels as one-based class indices `1..NUM_CLASSES` (MNIST digit + 1)."
const MNIST_LOGISTIC_Y = _MNIST_FIXTURE_LABELS .+ 1

"""
    mnist_logistic_fixture()

The committed small real-MNIST fixture as `(; X, y, num_classes)` — `X` is
`8 × 784` in `[0, 1]`, `y` are one-based class indices. Reproduces the docs and
test inputs; the benchmark loads full MNIST via MLDatasets instead.
"""
mnist_logistic_fixture() =
    (; X = MNIST_LOGISTIC_X, y = MNIST_LOGISTIC_Y, num_classes = NUM_CLASSES)

const MNIST_LOGISTIC_SOURCE = raw"""
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, categorical_logit

@kernel model(unconstrained::Vector{Float64},
              W::Matrix{Float64}, b::Vector{Float64},
              X::Matrix{Float64}, y::Vector{Int}, num_classes::Int) = begin
    # Bidirectional packed <-> structured coefficients (two producers for the
    # same ports, matching the distribution objects' HAVE-authority policy). A
    # packed query starts from the flat sampler vector [vec(W); b]; a structured
    # query starts from (W, b). Supplying both cuts both recipes.
    feature_count::Int = size(X, 2)
    nonreference::Int = num_classes - 1
    W::Matrix{Float64} =
        reshape(view(unconstrained, 1:(nonreference * feature_count)),
                nonreference, feature_count)
    b::Vector{Float64} =
        unconstrained[(nonreference * feature_count + 1):length(unconstrained)]
    unconstrained::Vector{Float64} = vcat(vec(W), b)

    # Log prior: every coefficient in W and b is Normal(0, 1). Reuse the shared
    # `normal` distribution object over the flat coefficient vector; there is no
    # hand-written Normal formula and no duplicated prior code.
    prior_terms = plate(unconstrained) do coefficient
        normal(0.0, 1.0).logpdf(coefficient)
    end
    prior::Float64 = sum(prior_terms)

    # Softmax categorical likelihood. The linear predictor gives the C-1
    # non-reference logits per observation; class 0 is the reference (logit 0),
    # prepended as the first row. Each observation's likelihood then reuses the
    # `categorical_logit` distribution object over that observation's logits
    # column — exactly as Eight Schools reuses `normal` per observation.
    nonreference_logits = W * transpose(X) .+ b
    logits = vcat(zeros(1, size(nonreference_logits, 2)), nonreference_logits)
    logit_columns = eachcol(logits)
    pointwise = plate(logit_columns, y) do observation_logits, observed_class
        categorical_logit(observation_logits).logpdf(observed_class)
    end
    likelihood::Float64 = sum(pointwise)

    density::Float64 = prior + likelihood
    return density
end

X = MNIST_LOGISTIC_X
y = MNIST_LOGISTIC_Y
num_classes = NUM_CLASSES
feature_count = size(X, 2)
nonreference = num_classes - 1
W = reshape(0.01 .* collect(1.0:(nonreference * feature_count)), nonreference, feature_count)
b = 0.01 .* collect(1.0:nonreference)
unconstrained = vcat(vec(W), b)

# One plan extracts the coefficients, the log prior, the summed log likelihood,
# and the full log density from the packed sampler vector — the Eight Schools
# extraction pattern, one specialized kernel rather than several evaluators.
requested_nodes = (:prior, :likelihood, :density)
evaluation_kernel = prepare(model;
    have = (:unconstrained, :X, :y, :num_classes),
    want = requested_nodes)
output = evaluation_kernel(unconstrained, X, y, num_classes)
prior, likelihood, density = output

# Pointwise likelihoods and their total are alternate cuts through the same
# authored plate; a total-only query fuses the sum without materializing the
# per-observation vector.
pointwise_extraction = prepare(model;
    have = (:W, :b, :X, :y, :num_classes),
    want = :pointwise)
pointwise = pointwise_extraction(W, b, X, y, num_classes)
@assert likelihood ≈ sum(pointwise)

docs_example = (;
    name = :mnist_logistic_density,
    origin = "PPL graph-output extraction (build executed) — multinomial-logistic MNIST classifier",
    inputs = (; unconstrained, X, y, num_classes),
    model,
    kernel = evaluation_kernel,
    output,
    requested_nodes,
    pointwise_extraction,
    pointwise,
    normal_object = normal,
    categorical_logit_object = categorical_logit,
)
"""

function evaluate_mnist_logistic_source()
    # Bind only the data. The authored source imports the reusable `normal` and
    # `categorical_logit` distribution objects itself; the density is entirely
    # composed from those objects and RK plate/array primitives.
    _evaluate_ppl_source(MNIST_LOGISTIC_SOURCE, @__MODULE__; bindings = (
        :MNIST_LOGISTIC_X, :MNIST_LOGISTIC_Y, :NUM_CLASSES,
    ))
end

# Evaluate the authored source from `__init__`, after precompilation has closed
# the module, then clone the runtime template per call (the Eight Schools
# pattern) so every caller gets an independent mutable graph without crossing a
# fresh `Core.eval` world-age boundary inside its own compiled function.
const _MNIST_LOGISTIC_GRAPH_TEMPLATE = Ref{KernelSpec}()

function __init__()
    _MNIST_LOGISTIC_GRAPH_TEMPLATE[] = evaluate_mnist_logistic_source().model
    nothing
end

"""
    build_mnist_logistic_graph()

Build the multinomial-logistic (softmax) MNIST classifier as a declarative
`ReactiveKernels.KernelSpec`. `W` (`(C-1) × D`) and `b` (`C-1`) carry a
`Normal(0, 1)` prior; class 0 is the reference (logit 0); each observation is a
softmax categorical over its logits column.

The density is composed entirely from reusable, transparently-authored
distribution objects: the shared `normal` object for the coefficient prior and
the `categorical_logit` object for the per-observation likelihood (reused inside
one authored `plate`). There is no hand-written density formula. Packed
unconstrained (`unconstrained`), structured (`W`, `b`), and likelihood-only
boundaries are HAVE cuts of the same graph; prior, pointwise likelihood, summed
likelihood, and the joint density are selectable named nodes.
"""
build_mnist_logistic_graph() = compose(_MNIST_LOGISTIC_GRAPH_TEMPLATE[])

function demo()
    model = build_mnist_logistic_graph()
    fixture = mnist_logistic_fixture()
    X, y, num_classes = fixture.X, fixture.y, fixture.num_classes
    feature_count = size(X, 2)
    nonreference = num_classes - 1
    W = reshape(0.01 .* collect(1.0:(nonreference * feature_count)),
                nonreference, feature_count)
    b = 0.01 .* collect(1.0:nonreference)
    unconstrained = vcat(vec(W), b)

    println("Packed-vector joint log density (sampler boundary):")
    packed = prepare(model;
        have = (:unconstrained, :X, :y, :num_classes), want = :density)
    println("log density = ", packed(unconstrained, X, y, num_classes))

    println("\nExtract prior, likelihood, and density in one plan:")
    plan_ = plan(model;
        have = (:W, :b, :X, :y, :num_classes),
        want = (:prior, :likelihood, :density))
    println(explain(plan_))
    prior, likelihood, density = prepare(plan_)(W, b, X, y, num_classes)
    println("log prior = ", prior, ", log likelihood = ", likelihood,
            ", log density = ", density)

    pointwise = prepare(model;
        have = (:W, :b, :X, :y, :num_classes), want = :pointwise)(
            W, b, X, y, num_classes)
    println("pointwise log likelihoods = ", pointwise)
    nothing
end

end # module MNISTLogisticExample

if abspath(PROGRAM_FILE) == @__FILE__
    MNISTLogisticExample.demo()
end
