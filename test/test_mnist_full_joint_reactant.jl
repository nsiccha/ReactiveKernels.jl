using ReactiveKernels
using Reactant
using Reactant: @compile
using Test
import Enzyme
using DifferentiationInterface: AutoEnzyme
using ReactiveKernelsPPLExamples.MNISTLogisticExample:
    NUM_CLASSES, build_mnist_logistic_graph, mnist_logistic_fixture
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    categorical_logit, categorical_logit_ref

const _MNIST_FULL_JOINT_AD_BACKEND =
    AutoEnzyme(; mode = Enzyme.Reverse)

@kernel _reactant_traced_index_gather(
        values::Vector{Float64}, index::Int) = begin
    selected::Float64 = values[index]
end

@kernel _reactant_batched_traced_index_gather(
        values::Matrix{Float64}, indices::Vector{Int}) = begin
    selected = plate(eachcol(values), indices) do column, index
        column[index]
    end
end

@kernel _reactant_natural_categorical_columns(
        logits::Matrix{Float64}, observed::Vector{Int}) = begin
    pointwise = plate(eachcol(logits), observed) do column, label
        categorical_logit(column).logpdf(label)
    end
    return sum(pointwise)
end

@kernel _reactant_natural_categorical_ref_columns(
        nonreference_logits::Matrix{Float64}, observed::Vector{Int}) = begin
    pointwise = plate(eachcol(nonreference_logits), observed) do column, label
        categorical_logit_ref(column).logpdf(label)
    end
    return sum(pointwise)
end

_trace_mnist(value::AbstractArray) = Reactant.to_rarray(value)
_trace_mnist(value::Integer) = value

@testset "MNIST full joint through Reactant" begin
    @testset "traced scalar index lowers as a gather" begin
        values_host = [0.5, -1.25, 3.0, 2.5]
        index_host = 3
        kernel = prepare(_reactant_traced_index_gather;
            have = (:values, :index), want = :selected)
        values = Reactant.to_rarray(values_host)
        index = Reactant.to_rarray(index_host; track_numbers = true)
        compiled = @compile sync = true kernel(values, index)
        @test Float64(compiled(values, index)) == values_host[index_host]
    end

    @testset "eachcol gather keeps lane-varying indices" begin
        values_host = reshape(collect(1.0:20.0), 5, 4)
        indices_host = [5, 1, 4, 2]
        expected = [values_host[indices_host[j], j]
                    for j in eachindex(indices_host)]
        kernel = prepare(_reactant_batched_traced_index_gather;
            have = (:values, :indices), want = :selected)
        values = Reactant.to_rarray(values_host)
        indices = Reactant.to_rarray(indices_host)
        compiled = @compile sync = true kernel(values, indices)
        @test Array(compiled(values, indices)) == expected
    end

    @testset "natural categorical eachcol plates" begin
        observed_host = [5, 1, 4, 2]
        for (spec, logits_host) in (
                (_reactant_natural_categorical_columns,
                 reshape(collect(0.1:0.1:2.0), 5, 4)),
                (_reactant_natural_categorical_ref_columns,
                 reshape(collect(0.1:0.1:1.6), 4, 4)),
            )
            kernel = prepare(spec)
            expected = kernel(logits_host, observed_host)
            logits = Reactant.to_rarray(logits_host)
            observed = Reactant.to_rarray(observed_host)
            compiled = @compile sync = true kernel(logits, observed)
            @test Float64(compiled(logits, observed)) ≈ expected rtol = 1e-9 atol = 1e-9
        end
    end

    fixture = mnist_logistic_fixture()
    X, y = fixture.X, fixture.y
    feature_count = size(X, 2)
    nonreference = NUM_CLASSES - 1
    W = reshape(
        0.001 .* collect(1.0:(nonreference * feature_count)),
        nonreference, feature_count)
    b = 0.001 .* collect(1.0:nonreference)
    unconstrained = vcat(vec(W), b)

    model = build_mnist_logistic_graph()
    kernel = prepare(model;
        have = (:unconstrained, :X, :y, :num_classes),
        want = :density)
    args = (unconstrained, X, y, NUM_CLASSES)
    native_value = kernel(args...)
    traced = map(_trace_mnist, args)

    @testset "packed full-joint primal" begin
        compiled = @compile sync = true kernel(traced...)
        @test Float64(compiled(traced...)) ≈ native_value rtol = 1e-9 atol = 1e-9
    end

    @testset "packed full-joint value and gradient" begin
        prepared = prepare_ad(
            kernel, _MNIST_FULL_JOINT_AD_BACKEND, args...;
            active = :unconstrained)
        native_gradient = similar(unconstrained)
        native_ad_value, native_gradient =
            ad_value_and_gradient!(prepared, native_gradient, args...)

        compiled = compile_ad_value_and_gradient(prepared, traced...)
        compiled_value, compiled_gradient = compiled(traced...)
        @test Float64(compiled_value) ≈ native_ad_value rtol = 1e-9 atol = 1e-9
        @test Array(compiled_gradient) ≈ native_gradient rtol = 1e-9 atol = 1e-9
    end
end
