using ReactiveKernelsPPLExamples.MNISTLogisticExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal

_mnist_logsumexp(v) = (m = maximum(v); m + log(sum(x -> exp(x - m), v)))

@testset "MNIST multinomial-logistic PPL graph" begin
    artifact = evaluate_mnist_logistic_source()
    @test artifact.source == strip(MNIST_LOGISTIC_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(
        artifact.kernel, Tuple(artifact.inputs)...)

    # Transparent authoring: the density is composed from reusable distribution
    # objects; the model source carries no hand-written density math.
    @test occursin("normal(0.0, 1.0).logpdf", MNIST_LOGISTIC_SOURCE)
    @test occursin("_categorical_logit_columns_kernel(logits, y)",
                   MNIST_LOGISTIC_SOURCE)
    @test !occursin("log(2π)", MNIST_LOGISTIC_SOURCE)     # no hand-written Normal
    @test artifact.normal_object === normal
    @test length(findall("@kernel model(", MNIST_LOGISTIC_SOURCE)) == 1
    authored = "@kernel model" * first(split(
        split(MNIST_LOGISTIC_SOURCE, "@kernel model"; limit = 2)[2],
        "\n\nX = MNIST_LOGISTIC_X"; limit = 2))
    @test !occursin("logsumexp", authored)  # lives inside the reused kernel
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

@testset "MNIST multinomial-logistic optimized (vcat-free) PPL graph" begin
    artifact = evaluate_mnist_logistic_optimized_source()
    @test artifact.source == strip(MNIST_LOGISTIC_OPTIMIZED_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(
        artifact.kernel, Tuple(artifact.inputs)...)

    # Transparent authoring: the reference coding lives inside the
    # reference-coded object; the model source still carries no density math
    # and, deliberately, no padded-logits vcat and no zeros row.
    @test occursin("normal(0.0, 1.0).logpdf", MNIST_LOGISTIC_OPTIMIZED_SOURCE)
    @test occursin("_categorical_logit_ref_columns_kernel(",
                   MNIST_LOGISTIC_OPTIMIZED_SOURCE)
    @test !occursin("log(2π)", MNIST_LOGISTIC_OPTIMIZED_SOURCE)
    @test !occursin("zeros(", MNIST_LOGISTIC_OPTIMIZED_SOURCE)
    authored_optimized = "@kernel model" * first(split(
        split(MNIST_LOGISTIC_OPTIMIZED_SOURCE, "@kernel model"; limit = 2)[2],
        "\n\nX = MNIST_LOGISTIC_X"; limit = 2))
    @test !occursin("logsumexp", authored_optimized)
    @test length(findall("vcat(", authored_optimized)) == 1   # packed producer only
    @test occursin(
        "_categorical_logit_ref_columns_kernel(nonreference_logits, y)",
        authored_optimized)
    @test artifact.normal_object === normal
    @test !occursin("prepare(", authored_optimized)

    # Exact parity with the idiomatic graph on every boundary and outcome,
    # including the reference-class-only edge (every observation is class 1,
    # the implicit zero-logit reference).
    fixture = mnist_logistic_fixture()
    X, y, C = fixture.X, fixture.y, fixture.num_classes
    D = size(X, 2); K = C - 1
    W = reshape(0.01 .* collect(1.0:(K * D)), K, D)
    b = 0.01 .* collect(1.0:K)
    unconstrained = vcat(vec(W), b)
    have_packed = (:unconstrained, :X, :y, :num_classes)

    idiomatic = build_mnist_logistic_graph()
    optimized = build_mnist_logistic_optimized_graph()
    for want in (:density, :prior, :likelihood)
        a = prepare(idiomatic; have = have_packed, want = want)(
            unconstrained, X, y, C)
        o = prepare(optimized; have = have_packed, want = want)(
            unconstrained, X, y, C)
        @test isapprox(a, o; rtol = 1e-12, atol = 1e-12)
    end
    a_pointwise = prepare(idiomatic; have = have_packed, want = :pointwise)(
        unconstrained, X, y, C)
    o_pointwise = prepare(optimized; have = have_packed, want = :pointwise)(
        unconstrained, X, y, C)
    @test isapprox(a_pointwise, o_pointwise; rtol = 1e-12, atol = 1e-12)

    o_structured = prepare(optimized;
        have = (:W, :b, :X, :y, :num_classes), want = :density)(W, b, X, y, C)
    a_structured = prepare(idiomatic;
        have = (:W, :b, :X, :y, :num_classes), want = :density)(W, b, X, y, C)
    @test isapprox(a_structured, o_structured; rtol = 1e-12, atol = 1e-12)

    y_reference = fill(1, length(y))
    a_reference = prepare(idiomatic; have = have_packed, want = :likelihood)(
        unconstrained, X, y_reference, C)
    o_reference = prepare(optimized; have = have_packed, want = :likelihood)(
        unconstrained, X, y_reference, C)
    @test isapprox(a_reference, o_reference; rtol = 1e-12, atol = 1e-12)

    # The prior boundary still prunes the likelihood on the optimized graph.
    o_prior = prepare(optimized; have = :unconstrained, want = :prior)(unconstrained)
    @test o_prior ≈ prepare(idiomatic;
        have = :unconstrained, want = :prior)(unconstrained)
end
