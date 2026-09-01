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
    build_mnist_logistic_graph, build_mnist_logistic_optimized_graph, NUM_CLASSES
using LogExpFunctions: logsumexp
import DynamicPPL
import Turing
using Turing: filldist
using Distributions: Normal
using NNlib: softmax
import MLDatasets

include(joinpath(@__DIR__, "mnist_logistic_matrix_spec.jl"))
using .MNISTLogisticMatrixSpec

const LDP = DynamicPPL.LogDensityProblems
const DEFAULT_MNIST_ROUNDS = 10
# The published receipt fits the full MNIST training split; RK_MNIST_N overrides
# it for a quicker local reproduction.
const DEFAULT_MNIST_N = 60000
const COMPARISON_PACKAGES = (
    "ReactiveKernels", "ReactiveKernelsPPLExamples",
    "ReactiveKernelsDistributionKernels", "Turing", "DynamicPPL",
    "Distributions", "NNlib", "MLDatasets", "BenchmarkTools",
    "MutatingFunctions",
)

# The non-allocating column needs the optional MutatingFunctions extension.
# A definitions-only consumer (the AD harness includes this file for the exact
# baseline definitions, with RK_MNIST_DEFINITIONS_ONLY=1) skips the
# requirement together with the terminal run call.
if get(ENV, "RK_MNIST_DEFINITIONS_ONLY", "") != "1"
    using MutatingFunctions
    Base.get_extension(ReactiveKernels, :ReactiveKernelsMutatingFunctionsExt) === nothing &&
        error("the non-allocating column requires ReactiveKernelsMutatingFunctionsExt to load")
end

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

# DOCS-BASELINE-BEGIN: turing-optimized
# The heavier-optimized Turing baseline: identical priors, but the likelihood
# never materializes the padded logits matrix, the softmax matrix, or the
# per-observation index vector — the same implicit-reference stable
# log-sum-exp loop the handwritten control uses, accumulated directly into one
# `@addlogprob!` term.
Turing.@model function turing_mnist_logistic_optimized(X, y, C)
    N, D = size(X)
    W ~ filldist(Normal(), C - 1, D)
    b ~ filldist(Normal(), C - 1)
    nonreference = W * X' .+ b
    total = 0.0
    @inbounds for j in eachindex(y)
        column = @view nonreference[:, j]
        maximum_logit = max(0.0, maximum(column))
        accumulated = exp(-maximum_logit)
        for logit in column
            accumulated += exp(logit - maximum_logit)
        end
        log_normalizer = maximum_logit + log(accumulated)
        true_logit = y[j] == 1 ? 0.0 : column[y[j] - 1]
        total += true_logit - log_normalizer
    end
    Turing.@addlogprob! total
end
# DOCS-BASELINE-END: turing-optimized

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

function _package_git_revision(name)
    for info in values(Pkg.dependencies())
        info.name == name && return string(something(info.git_revision, "unknown"))
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
_observations() = parse(Int, get(
    ENV, "RK_MNIST_N", string(DEFAULT_MNIST_N)))

function _load_mnist(n)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    train = MLDatasets.MNIST(split = :train)
    total = size(train.features, 3)
    n <= total || error("requested $n MNIST images but only $total are available")
    pixels = reshape(train.features[:, :, 1:n], 28 * 28, n)   # 784×n Float32 in [0,1]
    X = Matrix{Float64}(transpose(pixels))                    # n×784
    y = Int.(train.targets[1:n]) .+ 1                         # one-based classes
    X, y
end

_mnist_want(outcome) = outcome == "joint" ? :density : Symbol(outcome)

function _rk_primal_definition(model, configuration, boundary, outcome,
                               unconstrained, W, b, X, y)
    state, reason = matrix_support(configuration, boundary, outcome)
    state == "supported" || return (; state, reason, spec = nothing)
    configuration.differentiation == "primal" ||
        return (; state = "unsupported",
                reason = "not a primal execution configuration", spec = nothing)
    configuration.compiler == "native" ||
        return (; state = "unsupported",
                reason = "Reactant configurations are measured by the matched compiler receipt",
                spec = nothing)

    packed = boundary == "packed_unconstrained"
    data_dependent = outcome != "prior"
    have = if packed
        data_dependent ? (:unconstrained, :X, :y, :num_classes) :
                         (:unconstrained,)
    else
        data_dependent ? (:W, :b, :X, :y, :num_classes) : (:W, :b)
    end
    arguments = if packed
        data_dependent && configuration.data == "unbound" ?
            (unconstrained, X, y, NUM_CLASSES) : (unconstrained,)
    else
        data_dependent && configuration.data == "unbound" ?
            (W, b, X, y, NUM_CLASSES) : (W, b)
    end
    bound = configuration.data == "bound" ?
        (; X, y, num_classes = NUM_CLASSES) : NamedTuple()
    kernel = if configuration.allocation == "ordinary"
        prepare(model; have, want = _mnist_want(outcome), bound)
    else
        prepare_nonallocating(model; have, want = _mnist_want(outcome), bound)
    end
    (; state, reason, spec = (kernel, arguments))
end

function _manual_primal_definition(boundary, outcome, unconstrained, W, b,
                                   X, y, nonreference, features)
    packed = boundary == "packed_unconstrained"
    if packed
        outcome == "joint" && return (
            _manual_packed_joint,
            (unconstrained, X, y, nonreference, features))
        outcome == "prior" && return (
            _manual_packed_prior, (unconstrained, nonreference, features))
        outcome == "likelihood" && return (
            _manual_packed_likelihood,
            (unconstrained, X, y, nonreference, features))
        return (_manual_packed_pointwise,
                (unconstrained, X, y, nonreference, features))
    end
    outcome == "joint" && return (_manual_structured_joint, (W, b, X, y))
    outcome == "prior" && return (_manual_structured_prior, (W, b))
    outcome == "likelihood" &&
        return (_manual_structured_likelihood, (W, b, X, y))
    (_manual_structured_pointwise, (W, b, X, y))
end

function _turing_primal_definition(model, packed_views, vector, parameters,
                                   boundary, outcome)
    outcome == "pointwise" && return nothing
    if boundary == "packed_unconstrained"
        view = outcome == "joint" ? packed_views.joint :
               outcome == "prior" ? packed_views.prior : packed_views.likelihood
        return (_turing_density, (view, vector))
    end
    evaluator = outcome == "joint" ? _turing_joint :
                outcome == "prior" ? _turing_prior : _turing_likelihood
    (evaluator, (model, parameters))
end

function _measurement_row(; provider, model, configuration, boundary, outcome,
                          description, state = "supported", reason = "",
                          result = nothing)
    row = Dict{String,Any}(
        "provider" => provider,
        "model" => model,
        "configuration" => configuration,
        "boundary" => boundary,
        "outcome" => outcome,
        "description" => description,
        "state" => state,
    )
    isempty(reason) || (row["reason"] = reason)
    result === nothing || (row["result"] = result)
    row
end

function run_comparison()
    rounds = _rounds()
    n = _observations()
    X, y = _load_mnist(n)
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

    rk_models = (
        (name = "idiomatic", graph = build_mnist_logistic_graph()),
        (name = "vcat_free", graph = build_mnist_logistic_optimized_graph()),
    )
    native_primal_configurations = filter(
        configuration -> configuration.differentiation == "primal" &&
            configuration.compiler == "native",
        MNIST_RK_CONFIGURATIONS,
    )

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

    model_optimized = turing_mnist_logistic_optimized(X, y, NUM_CLASSES)
    turing_opt_packed_joint = DynamicPPL.LogDensityFunction(
        model_optimized, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll();
        fix_transforms = true)
    turing_opt_packed_prior = DynamicPPL.LogDensityFunction(
        model_optimized, DynamicPPL.getlogprior_internal, DynamicPPL.LinkAll();
        fix_transforms = true)
    turing_opt_packed_likelihood = DynamicPPL.LogDensityFunction(
        model_optimized, DynamicPPL.getloglikelihood, DynamicPPL.LinkAll();
        fix_transforms = true)
    turing_opt_vector = _turing_vector(turing_opt_packed_joint, parameters)
    isapprox(turing_opt_vector, unconstrained; rtol = 1e-12, atol = 1e-12) ||
        error("optimized Turing's linked parameter vector does not match the packed RK boundary")

    turing_controls = (
        (name = "idiomatic", model = model,
         views = (joint = turing_packed_joint, prior = turing_packed_prior,
                  likelihood = turing_packed_likelihood),
         vector = turing_vector),
        (name = "vcat_free", model = model_optimized,
         views = (joint = turing_opt_packed_joint, prior = turing_opt_packed_prior,
                  likelihood = turing_opt_packed_likelihood),
         vector = turing_opt_vector),
    )
    descriptions = Dict(
        "joint" => "full joint over the selected parameter boundary",
        "prior" => "standard-normal coefficient log prior",
        "likelihood" => "summed softmax categorical log likelihood",
        "pointwise" => "per-observation softmax categorical log likelihoods",
    )
    measurements = Dict{String,Any}[]
    for boundary in MNIST_BOUNDARIES, outcome in MNIST_OUTCOMES
        description = descriptions[outcome]
        manual = _manual_primal_definition(
            boundary, outcome, unconstrained, W, b, X, y,
            nonreference, features)
        reference = _comparable(outcome, _evaluate(manual))

        for model_definition in rk_models,
            configuration in native_primal_configurations
            definition = _rk_primal_definition(
                model_definition.graph, configuration, boundary, outcome,
                unconstrained, W, b, X, y)
            result = nothing
            if definition.state == "supported"
                actual = _comparable(outcome, _evaluate(definition.spec))
                isapprox(actual, reference; rtol = 1e-9, atol = 1e-9) || error(
                    "parity failed for $(model_definition.name) / " *
                    "$(configuration.id) / $boundary / $outcome")
                function_, arguments = definition.spec
                result = _measurement(function_, arguments...; rounds)
            end
            push!(measurements, _measurement_row(;
                provider = "rk", model = model_definition.name,
                configuration = configuration.id, boundary, outcome,
                description, state = definition.state,
                reason = definition.reason, result))
        end

        manual_function, manual_arguments = manual
        push!(measurements, _measurement_row(;
            provider = "manual_julia", model = "implicit_reference",
            configuration = "manual_primal", boundary, outcome, description,
            result = _measurement(manual_function, manual_arguments...; rounds)))

        for control in turing_controls
            spec = _turing_primal_definition(
                control.model, control.views, control.vector, parameters,
                boundary, outcome)
            if spec === nothing
                push!(measurements, _measurement_row(;
                    provider = "turing", model = control.name,
                    configuration = "turing_$(control.name)_primal",
                    boundary, outcome, description, state = "unsupported",
                    reason = "the @addlogprob! likelihood has no public pointwise view"))
                continue
            end
            actual = _comparable(outcome, _evaluate(spec))
            isapprox(actual, reference; rtol = 1e-9, atol = 1e-9) || error(
                "Turing parity failed for $(control.name) / $boundary / $outcome")
            function_, arguments = spec
            push!(measurements, _measurement_row(;
                provider = "turing", model = control.name,
                configuration = "turing_$(control.name)_primal",
                boundary, outcome, description,
                result = _measurement(function_, arguments...; rounds)))
        end
        println("boundary=$boundary outcome=$outcome complete")
    end

    receipt = Dict{String,Any}(
        "schema" => "mnist-logistic-primal-v3",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => get(
                ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown"),
            "reactivekernels_dirty" => false,
            (string(lowercase(name), "_version") => _package_version(name)
             for name in COMPARISON_PACKAGES)...,
            "mutatingfunctions_rev" => _package_git_revision("MutatingFunctions"),
            "julia_version" => string(VERSION)),
        "environment" => Dict(
            "os" => string(Sys.KERNEL), "arch" => string(Sys.ARCH),
            "cpu" => first(Sys.cpu_info()).model,
            "julia_threads" => Threads.nthreads()),
        "protocol" => Dict(
            "model" => "multinomial-logistic MNIST classifier",
            "data" => "MLDatasets MNIST train split, first N images",
            "num_observations" => n,
            "num_features" => features,
            "num_classes" => NUM_CLASSES,
            "input_boundaries" => ["packed_unconstrained", "structured_parameters"],
            "outcomes" => ["joint", "prior", "likelihood", "pointwise"],
            "models" => collect(MNIST_MODELS),
            "rounds" => rounds,
            "estimator" => "minimum of per-round BenchmarkTools minimum times (uncontended cost; medians and raw rounds retained)",
            "samples_per_round" => 200,
            "seconds_per_round" => 0.2,
            "setup_in_timed_region" => false,
            "preparation_in_timed_region" => false,
            "turing_transform_strategy" => "fixed transforms",
            "turing_pointwise_supported" => false,
            "matrix_layout" => "long-form provider/model/configuration/boundary/outcome rows",
            "rk_configurations" => [
                configuration.id for configuration in native_primal_configurations],
            "rk_optimized_model" =>
                "vcat-free: reference class inside categorical_logit_ref",
            "rk_nonallocating_pass" =>
                "prepare_nonallocating over both public graphs (MutatingFunctions extension; caches seeded before timing)",
            "bound_ports" => ["X", "y", "num_classes"],
            "bound_prior_state" => "not_applicable: minimal prior cut has no data ports",
            "turing_optimized_model" =>
                "implicit-reference stable log-sum-exp likelihood accumulated into one @addlogprob! term",
            "gradients_included" => false,
            "parity_rtol" => 1e-9,
            "parity_atol" => 1e-9),
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
