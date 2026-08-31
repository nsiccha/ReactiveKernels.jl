using ReactiveKernelsPPLExamples.MNISTLogisticExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, categorical_logit

_mnist_logsumexp(v) = (m = maximum(v); m + log(sum(x -> exp(x - m), v)))

@testset "MNIST multinomial-logistic PPL graph" begin
    artifact = evaluate_mnist_logistic_source()
    @test artifact.source == strip(MNIST_LOGISTIC_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(
        artifact.kernel, Tuple(artifact.inputs)...)

    # Transparent authoring: the density is composed from reusable distribution
    # objects; the model source carries no hand-written density math.
    @test occursin("normal(0.0, 1.0).logpdf", MNIST_LOGISTIC_SOURCE)
    @test occursin("categorical_logit(", MNIST_LOGISTIC_SOURCE)
    @test !occursin("logsumexp", MNIST_LOGISTIC_SOURCE)   # lives inside the object
    @test !occursin("log(2π)", MNIST_LOGISTIC_SOURCE)     # no hand-written Normal
    @test artifact.normal_object === normal
    @test artifact.categorical_logit_object === categorical_logit
    @test length(findall("@kernel model(", MNIST_LOGISTIC_SOURCE)) == 1
    authored = first(split(MNIST_LOGISTIC_SOURCE, "\n\nX = MNIST_LOGISTIC_X"; limit = 2))
    @test !occursin("prepare(", authored)

    # Parity against a numerically-stable manual reference on the committed
    # fixture, across the packed and structured HAVE boundaries.
    fixture = mnist_logistic_fixture()
    X, y, C = fixture.X, fixture.y, fixture.num_classes
    D = size(X, 2); K = C - 1
    W = reshape(0.01 .* collect(1.0:(K * D)), K, D)
    b = 0.01 .* collect(1.0:K)
    unconstrained = vcat(vec(W), b)
    logits = vcat(zeros(1, size(X, 1)), W * transpose(X) .+ b)
    manual_prior =
        -0.5 * log(2π) * (length(W) + length(b)) - 0.5 * (sum(abs2, W) + sum(abs2, b))
    manual_pointwise =
        [logits[y[j], j] - _mnist_logsumexp(@view logits[:, j]) for j in axes(logits, 2)]
    manual_likelihood = sum(manual_pointwise)

    model = build_mnist_logistic_graph()
    packed = prepare(model;
        have = (:unconstrained, :X, :y, :num_classes),
        want = (:prior, :likelihood, :density))(unconstrained, X, y, C)
    @test packed[1] ≈ manual_prior
    @test packed[2] ≈ manual_likelihood
    @test packed[3] ≈ manual_prior + manual_likelihood

    structured = prepare(model;
        have = (:W, :b, :X, :y, :num_classes), want = :density)(W, b, X, y, C)
    @test structured ≈ manual_prior + manual_likelihood

    rk_pointwise = prepare(model;
        have = (:W, :b, :X, :y, :num_classes), want = :pointwise)(W, b, X, y, C)
    @test rk_pointwise ≈ manual_pointwise

    # The prior boundary prunes the observation likelihood (no matmul).
    rk_prior = prepare(model; have = :unconstrained, want = :prior)(unconstrained)
    @test rk_prior ≈ manual_prior
end
