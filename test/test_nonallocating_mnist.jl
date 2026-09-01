# Acceptance for the destination-passing non-allocating pass on the real MNIST
# multinomial-logistic graph (packages/ReactiveKernelsPPLExamples): the packed
# joint and likelihood boundaries reach a small, batch-size-independent
# steady-state allocation with parity preserved. Runs inside the pinned
# MutatingFunctions integration environment (test/run_nonallocating_integration.jl).
using ReactiveKernels
using MutatingFunctions
using ReactiveKernelsPPLExamples
using Test

const MNIST = ReactiveKernelsPPLExamples.MNISTLogisticExample

function mnist_inputs(n)
    nonreference, feature_count = MNIST.NUM_CLASSES - 1, 28 * 28
    W = reshape(0.01 .* collect(1.0:(nonreference * feature_count)),
                nonreference, feature_count)
    b = 0.01 .* collect(1.0:nonreference)
    unconstrained = vcat(vec(W), b)
    X = rand(n, feature_count)
    y = rand(1:MNIST.NUM_CLASSES, n)
    (unconstrained, X, y, MNIST.NUM_CLASSES)
end

steady_bytes(k, args...) = (k(args...); k(args...); @allocated k(args...))

@testset "MNIST optimized graph non-allocating acceptance" begin
    g = MNIST.build_mnist_logistic_optimized_graph()
    have = (:unconstrained, :X, :y, :num_classes)
    fixture = MNIST.mnist_logistic_fixture()
    fixture_args = (mnist_inputs(8)[1], fixture.X, fixture.y, fixture.num_classes)

    for want in (:density, :likelihood)
        plain = prepare(g; have, want)
        kernel = prepare_nonallocating(g; have, want)
        @test isapprox(kernel(fixture_args...), plain(fixture_args...);
                       rtol = 1e-12)
        small = steady_bytes(kernel, fixture_args...)
        large = steady_bytes(kernel, mnist_inputs(96)...)
        println("NONALLOCATING_ALLOC_BYTES\tmnist_optimized_$(want)_n8\t", small)
        println("NONALLOCATING_ALLOC_BYTES\tmnist_optimized_$(want)_n96\t", large)
        @test small == large
        @test small <= 512
    end

    kernel = prepare_nonallocating(g; have, want = :density)
    @test !any(op -> op isa ReactiveKernels._KernelSourceOp, kernel.ops)
end

@testset "MNIST graph non-allocating acceptance" begin
    g = MNIST.build_mnist_logistic_graph()
    have = (:unconstrained, :X, :y, :num_classes)
    fixture = MNIST.mnist_logistic_fixture()
    fixture_args = (mnist_inputs(8)[1], fixture.X, fixture.y, fixture.num_classes)

    for want in (:density, :likelihood)
        plain = prepare(g; have = have, want = want)
        k = prepare_nonallocating(g; have = have, want = want)

        # Value agreement on the committed real-MNIST fixture. The plain
        # kernel fuses each plate reduction into an accumulator loop while the
        # non-allocating kernel materializes the pointwise plate then sums it,
        # so totals agree up to summation association; the materialized total
        # is pinned bit-exactly below.
        @test isapprox(k(fixture_args...), plain(fixture_args...);
                       rtol = 1e-12)

        small = steady_bytes(k, fixture_args...)
        args_large = mnist_inputs(96)
        large = steady_bytes(k, args_large...)
        println("NONALLOCATING_ALLOC_BYTES\tmnist_$(want)_n8\t", small)
        println("NONALLOCATING_ALLOC_BYTES\tmnist_$(want)_n96\t", large)
        @test small == large            # zero data-sized reallocation
        @test small <= 512              # fixed near-zero per-call residue

        # Reseeding across batch-size changes preserves agreement.
        for n in (8, 32, 8)
            args = mnist_inputs(n)
            @test isapprox(k(args...), plain(args...); rtol = 1e-12)
        end
    end

    # The likelihood total is bit-exactly the sum of the plain pointwise
    # values — only the plain kernel's fused accumulation associates
    # differently.
    plain_pointwise = prepare(g; have = have, want = :pointwise)
    k_lik = prepare_nonallocating(g; have = have, want = :likelihood)
    args = mnist_inputs(96)
    @test k_lik(args...) == sum(plain_pointwise(args...))

    # Every fused captured source in this graph decomposes: no opaque fused
    # closure remains in the step table (plates stay as plate operations).
    k_joint = prepare_nonallocating(g; have = have, want = :density)
    @test !any(op -> op isa ReactiveKernels._KernelSourceOp, k_joint.ops)

    # The natural consumer entry: wrap an already-prepared kernel.
    prepared = prepare(g; have = have, want = :density)
    k_wrapped = prepare_nonallocating(prepared)
    @test k_wrapped isa NonAllocatingKernel
    @test isapprox(k_wrapped(fixture_args...), prepared(fixture_args...);
                   rtol = 1e-12)
end
