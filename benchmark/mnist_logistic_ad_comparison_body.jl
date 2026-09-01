# Inner body for the pinned MNIST multinomial-logistic AD-only comparison.

using Dates
using Pkg
using Random
using TOML
using ReactiveKernels
using ReactiveKernelsPPLExamples.MNISTLogisticExample:
    build_mnist_logistic_graph, build_mnist_logistic_optimized_graph, NUM_CLASSES
using DifferentiationInterface
import DynamicPPL
import Enzyme

include(joinpath(@__DIR__, "mnist_logistic_matrix_spec.jl"))
using .MNISTLogisticMatrixSpec

include(joinpath(@__DIR__, "_ad_comparison_support.jl"))
using .ADComparisonSupport

# Load, rather than copy, the exact Turing model and handwritten Julia control
# used by the published primal receipt. The environment variable suppresses only
# that file's terminal run call.
module PublishedMNISTLogisticPrimal
ENV["RK_MNIST_DEFINITIONS_ONLY"] = "1"
include(joinpath(@__DIR__, "mnist_logistic_comparison_body.jl"))
end
delete!(ENV, "RK_MNIST_DEFINITIONS_ONLY")

const Primal = PublishedMNISTLogisticPrimal
const RK_AD_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
const RK_BOUND_AD_BACKEND = AutoEnzyme(
    ; mode = Enzyme.Reverse, function_annotation = Enzyme.Const)
const TURING_AD_BACKEND = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const,
)
const DEFAULT_MNIST_AD_ROUNDS = 10
const DEFAULT_MNIST_AD_N = 60000
const MNIST_AD_PACKAGES = (
    "ReactiveKernels", "ReactiveKernelsPPLExamples",
    "ReactiveKernelsDistributionKernels", "Turing", "DynamicPPL",
    "Distributions", "NNlib", "MLDatasets", "DifferentiationInterface",
    "Enzyme", "BenchmarkTools",
)
const MNIST_AD_BOUNDARIES = ("packed_unconstrained", "structured_parameters")
const MNIST_AD_OUTCOMES = ("joint", "prior", "likelihood", "pointwise")
const MNIST_MODEL_PUBLISHED_SHA =
    "9dbaa1fbfbdb1bb2ff4238a3910754f36635f81e"
const MNIST_COMPARATOR_PUBLISHED_SHA =
    "9dbaa1fbfbdb1bb2ff4238a3910754f36635f81e"

_rounds() = parse(Int, get(
    ENV, "RK_MNIST_AD_ROUNDS", string(DEFAULT_MNIST_AD_ROUNDS)))
_observations() = parse(Int, get(
    ENV, "RK_MNIST_AD_N", string(DEFAULT_MNIST_AD_N)))

function _manual_definition(outcome, q, X, y, nonreference, features)
    if outcome == "joint"
        objective = Primal._manual_packed_joint
        contexts = (
            Constant(X), Constant(y), Constant(nonreference), Constant(features))
        raw = x -> objective(x, X, y, nonreference, features)
    elseif outcome == "prior"
        objective = Primal._manual_packed_prior
        contexts = (Constant(nonreference), Constant(features))
        raw = x -> objective(x, nonreference, features)
    elseif outcome == "likelihood"
        objective = Primal._manual_packed_likelihood
        contexts = (
            Constant(X), Constant(y), Constant(nonreference), Constant(features))
        raw = x -> objective(x, X, y, nonreference, features)
    else
        return nothing
    end
    (; objective, contexts, raw, point = q)
end

function _rk_definition(model, outcome, q, X, y, data_binding)
    want = outcome == "joint" ? :density : Symbol(outcome)
    if data_binding == "bound"
        kernel = prepare(
            model; have = (:unconstrained, :X, :y, :num_classes),
            want, bound = (; X, y, num_classes = NUM_CLASSES))
        return (;
            kernel, spec = nothing, want, bound = NamedTuple(),
            arguments = (q,), active = :unconstrained, data_binding,
        )
    end
    if outcome == "joint"
        kernel = prepare(model;
            have = (:unconstrained, :X, :y, :num_classes), want = :density)
        arguments = (q, X, y, NUM_CLASSES)
    elseif outcome == "prior"
        kernel = prepare(model; have = :unconstrained, want = :prior)
        arguments = (q,)
    elseif outcome == "likelihood"
        kernel = prepare(model;
            have = (:unconstrained, :X, :y, :num_classes), want = :likelihood)
        arguments = (q, X, y, NUM_CLASSES)
    else
        return nothing
    end
    (; kernel, spec = nothing, want, bound = NamedTuple(), arguments,
       active = :unconstrained, data_binding)
end

function _prepare_rk_ad(definition)
    backend = definition.data_binding == "bound" ?
        RK_BOUND_AD_BACKEND : RK_AD_BACKEND
    prepare_ad(
        definition.kernel, backend, definition.arguments...;
        active = definition.active)
end

function _turing_logdensity(model, outcome)
    evaluator = outcome == "joint" ? DynamicPPL.getlogjoint_internal :
                outcome == "prior" ? DynamicPPL.getlogprior_internal :
                outcome == "likelihood" ? DynamicPPL.getloglikelihood : nothing
    isnothing(evaluator) && return nothing
    DynamicPPL.LogDensityFunction(
        model, evaluator, DynamicPPL.LinkAll();
        fix_transforms = true, adtype = TURING_AD_BACKEND)
end

function _unsupported_reason(boundary, outcome)
    outcome == "pointwise" && return (
        "pointwise is vector-valued and the compared public surfaces do not " *
        "share a matched Jacobian/VJP contract")
    boundary == "structured_parameters" && return (
        "the structured (W, b) boundary has two active HAVE ports while the " *
        "public RK AD boundary deliberately accepts exactly one active port")
    "unsupported matrix cell"
end

# Independent analytic score for the sampler-relevant packed coefficient vector.
# The reference-class logit is zero; the active rows correspond to classes 2:C.
function _analytic_likelihood_gradient(q, X, y, nonreference, features)
    W, _ = Primal._unpack(q, nonreference, features)
    b = @view q[(nonreference * features + 1):end]
    logits = W * transpose(X) .+ b
    scores = similar(logits)
    @inbounds for observation in axes(logits, 2)
        column = @view logits[:, observation]
        maximum_logit = max(0.0, maximum(column))
        denominator = exp(-maximum_logit)
        for logit in column
            denominator += exp(logit - maximum_logit)
        end
        for class in axes(logits, 1)
            probability = exp(column[class] - maximum_logit) / denominator
            scores[class, observation] =
                (y[observation] == class + 1 ? 1.0 : 0.0) - probability
        end
    end
    dW = scores * X
    db = vec(sum(scores; dims = 2))
    vcat(vec(dW), db)
end

function _analytic_gradients(q, X, y, nonreference, features)
    likelihood = _analytic_likelihood_gradient(
        q, X, y, nonreference, features)
    prior = -q
    Dict(
        "joint" => prior + likelihood,
        "prior" => prior,
        "likelihood" => likelihood,
    )
end

function _verified_model_source_pin(root, relative_path)
    published, text = published_source_pin(
        root, MNIST_MODEL_PUBLISHED_SHA, relative_path)
    current = normalized_read(joinpath(root, relative_path))
    mnist_model_source_preserves_published_authority(current, text) || error(
        "MNIST model source no longer preserves published idiomatic authority " *
        MNIST_MODEL_PUBLISHED_SHA)
    merge(published, Dict(
        "current" => source_pin(root, relative_path),
        "current_delta" => MNIST_MODEL_SOURCE_CURRENT_DELTA,
    ))
end

function _verified_comparator_source_pin(root, relative_path)
    published, text = published_source_pin(
        root, MNIST_COMPARATOR_PUBLISHED_SHA, relative_path)
    guard = "get(ENV, \"RK_MNIST_DEFINITIONS_ONLY\", \"\") == \"1\" || run_comparison()\n"
    current = normalized_read(joinpath(root, relative_path))
    comparator_source_matches_current_delta(current, text, guard) || error(
        "MNIST comparator differs from its published authority by more than " *
        MNIST_COMPARATOR_SOURCE_CURRENT_DELTA)
    merge(published, Dict(
        "current" => source_pin(root, relative_path),
        "current_delta" => MNIST_COMPARATOR_SOURCE_CURRENT_DELTA,
    ))
end

function run_mnist_logistic_ad_comparison()
    rounds = _rounds()
    n = _observations()
    root = normpath(joinpath(@__DIR__, ".."))
    source_path = joinpath(
        "packages", "ReactiveKernelsPPLExamples", "src", "mnist_logistic.jl")
    comparator_path = joinpath("benchmark", "mnist_logistic_comparison_body.jl")
    primal_receipt_path =
        joinpath("benchmark", "receipts", "mnist-logistic-primal-v3.toml")
    primal_path = get(ENV, "RK_MNIST_PRIMAL_RECEIPT", primal_receipt_path)
    primal_absolute = isabspath(primal_path) ? primal_path : joinpath(root, primal_path)

    # Fail authority/prerequisite gates before loading MNIST or measuring.
    model_source_pin = _verified_model_source_pin(root, source_path)
    comparator_source_pin = _verified_comparator_source_pin(root, comparator_path)
    primal_receipt = TOML.parsefile(primal_absolute)

    data_setup = @timed Primal._load_mnist(n)
    X, y = data_setup.value
    features = size(X, 2)
    nonreference = NUM_CLASSES - 1
    Random.seed!(20260901)
    W = 0.01 .* randn(nonreference, features)
    b = 0.01 .* randn(nonreference)
    q = vcat(vec(W), b)
    parameters = (; W, b)

    model_setup = @timed (
        (name = "idiomatic", graph = build_mnist_logistic_graph()),
        (name = "vcat_free", graph = build_mnist_logistic_optimized_graph()),
    )
    rk_models = model_setup.value
    turing_setup = @timed (
        (name = "idiomatic",
         model = Primal.turing_mnist_logistic(X, y, NUM_CLASSES)),
        (name = "vcat_free",
         model = Primal.turing_mnist_logistic_optimized(X, y, NUM_CLASSES)),
    )
    turing_models = map(turing_setup.value) do definition
        joint = DynamicPPL.LogDensityFunction(
            definition.model, DynamicPPL.getlogjoint_internal,
            DynamicPPL.LinkAll(); fix_transforms = true)
        vector = Primal._turing_vector(joint, parameters)
        isapprox(vector, q; rtol = 1e-12, atol = 1e-12) || error(
            "Turing $(definition.name) linked parameter vector does not " *
            "match the RK boundary")
        (; definition..., vector)
    end

    oracle_setup = @timed _analytic_gradients(q, X, y, nonreference, features)
    analytic_gradients = oracle_setup.value
    native_ad_configurations = filter(
        configuration -> configuration.differentiation == "value_and_gradient" &&
            configuration.compiler == "native",
        MNIST_RK_CONFIGURATIONS,
    )
    measurements = Dict{String,Any}[]
    for boundary in MNIST_AD_BOUNDARIES, outcome in MNIST_AD_OUTCOMES
        matrix_state, matrix_reason = matrix_support(
            first(native_ad_configurations), boundary, outcome)
        manual_definition = matrix_state == "supported" ? _manual_definition(
            outcome, q, X, y, nonreference, features) : nothing
        reference_value = matrix_state == "supported" ?
            Float64(manual_definition.raw(q)) : nothing
        reference_gradient = matrix_state == "supported" ?
            analytic_gradients[outcome] : nothing

        for model_definition in rk_models,
            configuration in native_ad_configurations
            state, reason = matrix_support(configuration, boundary, outcome)
            row = Dict{String,Any}(
                "provider" => "rk", "model" => model_definition.name,
                "configuration" => configuration.id,
                "boundary" => boundary, "outcome" => outcome,
                "state" => state,
            )
            if state == "supported"
                definition = _rk_definition(
                    model_definition.graph, outcome, q, X, y,
                    configuration.data)
                rk_call, rk_result, rk_setup = build_and_first_call() do
                    prepared = _prepare_rk_ad(definition)
                    RKValueGradientCall(
                        prepared, similar(q), definition.arguments)
                end
                row["active_port"] = String(definition.active)
                row["analytic_gradient_length"] = length(reference_gradient)
                row["result"] = record_implementation(
                    rk_call, rk_result, rk_setup,
                    reference_value, reference_gradient;
                    rounds, caller_owned = true)
            else
                row["reason"] = reason
            end
            push!(measurements, row)
        end

        manual_row = Dict{String,Any}(
            "provider" => "manual_julia", "model" => "implicit_reference",
            "configuration" => "manual_ad", "boundary" => boundary,
            "outcome" => outcome, "state" => matrix_state,
        )
        if matrix_state == "supported"
            manual_call, manual_result, manual_setup = build_and_first_call() do
                preparation = DifferentiationInterface.prepare_gradient(
                    manual_definition.objective, RK_AD_BACKEND,
                    q, manual_definition.contexts...)
                DIValueGradientCall(
                    manual_definition.objective, similar(q), preparation,
                    RK_AD_BACKEND, q, manual_definition.contexts)
            end
            manual_row["active_port"] = "unconstrained"
            manual_row["analytic_gradient_length"] = length(reference_gradient)
            manual_row["result"] = record_implementation(
                manual_call, manual_result, manual_setup,
                reference_value, reference_gradient;
                rounds, caller_owned = true)
        else
            manual_row["reason"] = matrix_reason
        end
        push!(measurements, manual_row)

        for definition in turing_models
            row = Dict{String,Any}(
                "provider" => "turing", "model" => definition.name,
                "configuration" => "turing_$(definition.name)_ad",
                "boundary" => boundary, "outcome" => outcome,
                "state" => matrix_state,
            )
            if matrix_state == "supported"
                turing_call, turing_result, turing_ad_setup =
                    build_and_first_call() do
                        TuringValueGradientCall(
                            _turing_logdensity(definition.model, outcome),
                            definition.vector)
                    end
                row["active_port"] = "unconstrained"
                row["analytic_gradient_length"] = length(reference_gradient)
                row["result"] = record_implementation(
                    turing_call, turing_result, turing_ad_setup,
                    reference_value, reference_gradient;
                    rounds, caller_owned = false)
            else
                row["reason"] = matrix_reason
            end
            push!(measurements, row)
        end
        println("boundary=$boundary outcome=$outcome complete")
    end

    receipt = Dict{String,Any}(
        "schema" => "mnist-logistic-ad-v2",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => get(
                ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown"),
            "reactivekernels_dirty" => false,
            "julia_version" => string(VERSION),
            "model_source" => model_source_pin,
            "primal_comparator_source" => comparator_source_pin,
            "primal_receipt_path" => primal_receipt_path,
            "primal_receipt_sha256" => text_sha256(primal_absolute),
            "primal_receipt_reactivekernels_sha" =>
                primal_receipt["pins"]["reactivekernels_sha"],
            (string(lowercase(name), "_version") => package_version(name)
             for name in MNIST_AD_PACKAGES)...,
        ),
        "environment" => Dict(
            "os" => string(Sys.KERNEL),
            "arch" => string(Sys.ARCH),
            "cpu" => first(Sys.cpu_info()).model,
            "julia_threads" => Threads.nthreads(),
        ),
        "setup" => Dict(
            "data_load_seconds" => data_setup.time,
            "data_load_bytes" => data_setup.bytes,
            "rk_model_build_seconds" => model_setup.time,
            "rk_model_build_bytes" => model_setup.bytes,
            "turing_model_build_seconds" => turing_setup.time,
            "turing_model_build_bytes" => turing_setup.bytes,
            "analytic_oracle_seconds" => oracle_setup.time,
            "analytic_oracle_bytes" => oracle_setup.bytes,
        ),
        "protocol" => Dict(
            "model" => "multinomial-logistic MNIST classifier",
            "data" => "MLDatasets MNIST train split, first N images",
            "num_observations" => n,
            "num_features" => features,
            "num_classes" => NUM_CLASSES,
            "active_parameter_count" => length(q),
            "input_boundaries" => collect(MNIST_AD_BOUNDARIES),
            "outcomes" => collect(MNIST_AD_OUTCOMES),
            "models" => collect(MNIST_MODELS),
            "matrix_layout" =>
                "long-form provider/model/configuration/boundary/outcome rows",
            "rk_configurations" => [
                configuration.id for configuration in native_ad_configurations],
            "bound_ports" => ["X", "y", "num_classes"],
            "source_reused" => true,
            "matrix_source" => primal_receipt_path,
            "gradient_operation" => "value and gradient",
            "rk_surface" => "prepare_ad + ad_value_and_gradient!",
            "rk_backend" => "AutoEnzyme(mode = Enzyme.Reverse)",
            "rk_bound_backend" =>
                "AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)",
            "manual_control" =>
                "the primal receipt's manual Julia density differentiated through the same prepared DI+Enzyme boundary",
            "turing_surface" =>
                "LogDensityProblems.logdensity_and_gradient on DynamicPPL.LogDensityFunction",
            "turing_backend" =>
                "AutoEnzyme(runtime activity, Const function annotation)",
            "parity_oracle" =>
                "independent analytic reference-class softmax and standard-normal score",
            "parity_relative_tolerance" => 1e-7,
            "parity_absolute_tolerance" => 1e-7,
            "pointwise_jacobian_or_vjp_invented" => false,
            "structured_multi_active_boundary_invented" => false,
            "rounds" => rounds,
            "samples_per_round" => 200,
            "seconds_per_round" => 0.2,
            "estimator" =>
                "minimum of per-round BenchmarkTools minimum times (uncontended cost; medians and raw rounds retained)",
            "setup_in_timed_region" => false,
            "preparation_in_timed_region" => false,
            "first_execution_in_steady_state_region" => false,
        ),
        "measurements" => measurements,
    )
    missing_paths = nothing_paths(receipt)
    isempty(missing_paths) || error(
        "receipt contains TOML-incompatible nothing values: " *
        join(missing_paths, ", "))

    output = output_path()
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

get(ENV, "RK_MNIST_AD_DEFINITIONS_ONLY", "") == "1" ||
    run_mnist_logistic_ad_comparison()
