# Inner body for the MNIST multinomial-logistic (softmax) primal comparison.
# Mirrors the Eight Schools primal comparison: one authored RK graph is cut into
# HAVE/WANT boundaries and compared, at parity, against a handwritten Julia
# control and Turing's native public interfaces. Setup, data loading, kernel and
# LogDensityFunction preparation, and parameter linking are all outside timing.

using BenchmarkTools
using Dates
using Pkg
using Random
using Statistics
using TOML
using ReactiveKernels
using ReactiveKernelsPPLExamples.MNISTLogisticExample:
    build_mnist_logistic_graph, NUM_CLASSES
using LogExpFunctions: logsumexp
import DynamicPPL
import Turing
using Turing: filldist
using Distributions: Normal
using NNlib: softmax
import MLDatasets

include(joinpath(@__DIR__, "_mnist_dataset_profiles.jl"))

const LDP = DynamicPPL.LogDensityProblems
const DEFAULT_MNIST_ROUNDS = 10
const COMPARISON_PACKAGES = (
    "ReactiveKernels", "ReactiveKernelsPPLExamples",
    "ReactiveKernelsDistributionKernels", "Turing", "DynamicPPL",
    "Distributions", "NNlib", "MLDatasets", "BenchmarkTools",
)

# DOCS-BASELINE-BEGIN: turing
Turing.@model function turing_mnist_logistic(X, y, C)
    N, D = size(X)
    W ~ filldist(Normal(), C - 1, D)
    b ~ filldist(Normal(), C - 1)
    nonreference_logits = W * X' .+ b
    logits = vcat(zeros(eltype(nonreference_logits), 1, N), nonreference_logits)
    probabilities = softmax(logits; dims = 1)
    linear_indices = y .+ (eachindex(y) .- 1) .* size(probabilities, 1)
    Turing.@addlogprob! sum(log, probabilities[linear_indices])
end
# DOCS-BASELINE-END: turing

# ---- handwritten Julia control ------------------------------------------------
# DOCS-BASELINE-BEGIN: manual
# Standard-normal log prior over every coefficient (flat [vec(W); b]).
_manual_prior(coefficients) =
    -0.5 * length(coefficients) * log(2π) - 0.5 * sum(abs2, coefficients)

# Softmax categorical log likelihoods. Class 0 is the reference (logit 0); the
# normalizer is a numerically stable log-sum-exp over [0; Wxⱼ + b].
function _manual_pointwise(W, b, X, y)
    nonreference = W * transpose(X) .+ b            # (C-1) × N
    output = similar(y, Float64)
    @inbounds for j in eachindex(y)
        column = @view nonreference[:, j]
        maximum_logit = max(0.0, maximum(column))
        accumulated = exp(-maximum_logit)
        for logit in column
            accumulated += exp(logit - maximum_logit)
        end
        log_normalizer = maximum_logit + log(accumulated)
        true_logit = y[j] == 1 ? 0.0 : column[y[j] - 1]
        output[j] = true_logit - log_normalizer
    end
    output
end
_manual_likelihood(W, b, X, y) = sum(_manual_pointwise(W, b, X, y))

_unpack(coefficients, nonreference, features) = (
    reshape(view(coefficients, 1:(nonreference * features)), nonreference, features),
    coefficients[(nonreference * features + 1):end],
)

_manual_packed_prior(q, nonreference, features) = _manual_prior(q)
function _manual_packed_likelihood(q, X, y, nonreference, features)
    W, b = _unpack(q, nonreference, features)
    _manual_likelihood(W, b, X, y)
end
function _manual_packed_pointwise(q, X, y, nonreference, features)
    W, b = _unpack(q, nonreference, features)
    _manual_pointwise(W, b, X, y)
end
_manual_packed_joint(q, X, y, nonreference, features) =
    _manual_packed_prior(q, nonreference, features) +
    _manual_packed_likelihood(q, X, y, nonreference, features)

_manual_structured_prior(W, b) = _manual_prior(vcat(vec(W), b))
_manual_structured_likelihood(W, b, X, y) = _manual_likelihood(W, b, X, y)
_manual_structured_pointwise(W, b, X, y) = _manual_pointwise(W, b, X, y)
_manual_structured_joint(W, b, X, y) =
    _manual_structured_prior(W, b) + _manual_structured_likelihood(W, b, X, y)
# DOCS-BASELINE-END: manual

# ---- Turing native views ------------------------------------------------------
function _turing_vector(ldf, parameters)
    accumulator = DynamicPPL.OnlyAccsVarInfo(
        DynamicPPL.VectorParamAccumulator(ldf))
    _, accumulator = DynamicPPL.init!!(
        ldf.model, accumulator,
        DynamicPPL.InitFromParams(parameters), ldf.transform_strategy)
    collect(DynamicPPL.get_vector_params(accumulator))
end
_turing_density(ldf, parameters) = LDP.logdensity(ldf, parameters)
_turing_joint(model, parameters) = DynamicPPL.logjoint(model, parameters)
_turing_prior(model, parameters) = DynamicPPL.logprior(model, parameters)
_turing_likelihood(model, parameters) = DynamicPPL.loglikelihood(model, parameters)

_evaluate(spec) = first(spec)(last(spec)...)
_comparable(outcome, value) =
    outcome == "pointwise" ? collect(Float64, value) : Float64(value)

function _measurement(f, args...; rounds::Int)
    invocation = let f = f, args = args
        () -> f(args...)
    end
    benchmark = @benchmarkable $invocation()
    times_ns = Float64[]; bytes = Int[]; allocs = Int[]
    for _ in 1:rounds
        estimate = minimum(run(benchmark; samples = 200, seconds = 0.2))
        push!(times_ns, estimate.time)
        push!(bytes, estimate.memory)
        push!(allocs, estimate.allocs)
    end
    Dict(
        # `min_ns` is the headline estimator: the minimum of per-round
        # BenchmarkTools minimums estimates the uncontended cost, which a
        # median cannot on a shared host once a load episode spans more than
        # half the rounds (ms-scale matmul cells are exposed to exactly that).
        # The median and every raw round are retained for transparency.
        "times_ns" => times_ns, "min_ns" => minimum(times_ns),
        "median_ns" => median(times_ns),
        "bytes" => bytes, "median_bytes" => Int(median(bytes)),
        "allocs" => allocs, "median_allocs" => Int(median(allocs)),
    )
end

function _package_version(name)
    for info in values(Pkg.dependencies())
        info.name == name && return string(info.version)
    end
    error("package $name absent from the benchmark environment")
end

function _output_path()
    for arg in ARGS
        startswith(arg, "--output=") && return split(arg, '='; limit = 2)[2]
    end
    nothing
end

_rounds() = parse(Int, get(
    ENV, "RK_MNIST_ROUNDS", string(DEFAULT_MNIST_ROUNDS)))
_observations(profile) = parse(Int, get(
    ENV, "RK_MNIST_N", string(_mnist_default_observations(profile))))

function run_comparison()
    rounds = _rounds()
    dataset_profile = _mnist_dataset_profile()
    n = _observations(dataset_profile)
    X, y, dataset_metadata = _load_mnist_dataset(
        dataset_profile, n; wren_reference = _mnist_wren_reference_path())
    GC.gc()
    features = size(X, 2)
    nonreference = NUM_CLASSES - 1

    # A small, valid coefficient point. The Turing baseline forms softmax
    # probabilities explicitly, so large logits would underflow log(p) to -Inf;
    # a modest scale keeps every backend finite and in exact agreement while the
    # timed operation (the same dense linear predictor + normalization) is
    # unaffected.
    Random.seed!(20260901)
    W = 0.01 .* randn(nonreference, features)
    b = 0.01 .* randn(nonreference)
    unconstrained = vcat(vec(W), b)
    parameters = (; W, b)

    rk = build_mnist_logistic_graph()
    rk_packed_joint = prepare(rk;
        have = (:unconstrained, :X, :y, :num_classes), want = :density)
    rk_packed_prior = prepare(rk; have = :unconstrained, want = :prior)
    rk_packed_likelihood = prepare(rk;
        have = (:unconstrained, :X, :y, :num_classes), want = :likelihood)
    rk_packed_pointwise = prepare(rk;
        have = (:unconstrained, :X, :y, :num_classes), want = :pointwise)
    rk_structured_joint = prepare(rk;
        have = (:W, :b, :X, :y, :num_classes), want = :density)
    rk_structured_prior = prepare(rk; have = (:W, :b), want = :prior)
    rk_structured_likelihood = prepare(rk;
        have = (:W, :b, :X, :y, :num_classes), want = :likelihood)
    rk_structured_pointwise = prepare(rk;
        have = (:W, :b, :X, :y, :num_classes), want = :pointwise)

    model = turing_mnist_logistic(X, y, NUM_CLASSES)
    turing_packed_joint = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll();
        fix_transforms = true)
    turing_packed_prior = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogprior_internal, DynamicPPL.LinkAll();
        fix_transforms = true)
    turing_packed_likelihood = DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getloglikelihood, DynamicPPL.LinkAll();
        fix_transforms = true)
    turing_vector = _turing_vector(turing_packed_joint, parameters)
    isapprox(turing_vector, unconstrained; rtol = 1e-12, atol = 1e-12) ||
        error("Turing's linked parameter vector does not match the packed RK boundary")

    definitions = (
        (boundary = "packed_unconstrained", outcome = "joint",
         description = "full joint over the flattened sampler vector",
         rk = (rk_packed_joint, (unconstrained, X, y, NUM_CLASSES)),
         manual = (_manual_packed_joint, (unconstrained, X, y, nonreference, features)),
         turing = (_turing_density, (turing_packed_joint, turing_vector))),
        (boundary = "packed_unconstrained", outcome = "prior",
         description = "standard-normal coefficient log prior",
         rk = (rk_packed_prior, (unconstrained,)),
         manual = (_manual_packed_prior, (unconstrained, nonreference, features)),
         turing = (_turing_density, (turing_packed_prior, turing_vector))),
        (boundary = "packed_unconstrained", outcome = "likelihood",
         description = "summed softmax categorical log likelihood",
         rk = (rk_packed_likelihood, (unconstrained, X, y, NUM_CLASSES)),
         manual = (_manual_packed_likelihood, (unconstrained, X, y, nonreference, features)),
         turing = (_turing_density, (turing_packed_likelihood, turing_vector))),
        (boundary = "packed_unconstrained", outcome = "pointwise",
         description = "per-observation softmax categorical log likelihoods",
         rk = (rk_packed_pointwise, (unconstrained, X, y, NUM_CLASSES)),
         manual = (_manual_packed_pointwise, (unconstrained, X, y, nonreference, features)),
         turing = nothing),
        (boundary = "structured_parameters", outcome = "joint",
         description = "full joint over the (W, b) coefficients",
         rk = (rk_structured_joint, (W, b, X, y, NUM_CLASSES)),
         manual = (_manual_structured_joint, (W, b, X, y)),
         turing = (_turing_joint, (model, parameters))),
        (boundary = "structured_parameters", outcome = "prior",
         description = "standard-normal coefficient log prior",
         rk = (rk_structured_prior, (W, b)),
         manual = (_manual_structured_prior, (W, b)),
         turing = (_turing_prior, (model, parameters))),
        (boundary = "structured_parameters", outcome = "likelihood",
         description = "summed softmax categorical log likelihood",
         rk = (rk_structured_likelihood, (W, b, X, y, NUM_CLASSES)),
         manual = (_manual_structured_likelihood, (W, b, X, y)),
         turing = (_turing_likelihood, (model, parameters))),
        (boundary = "structured_parameters", outcome = "pointwise",
         description = "per-observation softmax categorical log likelihoods",
         rk = (rk_structured_pointwise, (W, b, X, y, NUM_CLASSES)),
         manual = (_manual_structured_pointwise, (W, b, X, y)),
         turing = nothing),
    )

    measurements = Dict{String,Any}[]
    for definition in definitions
        row = Dict{String,Any}(
            "boundary" => definition.boundary,
            "outcome" => definition.outcome,
            "description" => definition.description)
        reference = _comparable(definition.outcome, _evaluate(definition.manual))
        for (key, spec) in (
            "rk_native" => definition.rk,
            "manual_julia" => definition.manual,
            "turing_native" => definition.turing)
            spec === nothing && continue
            actual = _comparable(definition.outcome, _evaluate(spec))
            isapprox(actual, reference; rtol = 1e-9, atol = 1e-9) || error(
                "parity failed for $(definition.boundary) / $(definition.outcome) / $key")
            function_, arguments = spec
            row[key] = _measurement(function_, arguments...; rounds)
        end
        push!(measurements, row)
        println("boundary=$(definition.boundary) outcome=$(definition.outcome) complete")
    end

    receipt = Dict{String,Any}(
        "schema" => "mnist-logistic-v1",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => get(
                ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown"),
            "reactivekernels_dirty" => false,
            (string(lowercase(name), "_version") => _package_version(name)
             for name in COMPARISON_PACKAGES)...,
            "julia_version" => string(VERSION)),
        "environment" => Dict(
            "os" => string(Sys.KERNEL), "arch" => string(Sys.ARCH),
            "cpu" => first(Sys.cpu_info()).model,
            "julia_threads" => Threads.nthreads()),
        "protocol" => merge(Dict(
            "model" => "multinomial-logistic MNIST classifier",
            "num_classes" => NUM_CLASSES,
            "input_boundaries" => ["packed_unconstrained", "structured_parameters"],
            "outcomes" => ["joint", "prior", "likelihood", "pointwise"],
            "rounds" => rounds,
            "estimator" => "minimum of per-round BenchmarkTools minimum times (uncontended cost; medians and raw rounds retained)",
            "samples_per_round" => 200,
            "seconds_per_round" => 0.2,
            "setup_in_timed_region" => false,
            "preparation_in_timed_region" => false,
            "turing_transform_strategy" => "fixed transforms",
            "turing_pointwise_supported" => false,
            "gradients_included" => false,
            "parity_rtol" => 1e-9,
            "parity_atol" => 1e-9), dataset_metadata),
        "measurements" => measurements)

    output = _output_path()
    if output === nothing
        TOML.print(stdout, receipt; sorted = true)
    else
        mkpath(dirname(abspath(output)))
        open(output, "w") do io
            TOML.print(io, receipt; sorted = true)
        end
        println("receipt=$(abspath(output))")
    end
    receipt
end

get(ENV, "RK_MNIST_DEFINITIONS_ONLY", "") == "1" || run_comparison()
