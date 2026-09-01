# Inner body for the pinned MNIST multinomial-logistic AD-only comparison.

using Dates
using Pkg
using Random
using TOML
using ReactiveKernels
using ReactiveKernelsPPLExamples.MNISTLogisticExample:
    build_mnist_logistic_graph, NUM_CLASSES
using DifferentiationInterface
import DynamicPPL
import Enzyme

include(joinpath(@__DIR__, "_ad_comparison_support.jl"))
using .ADComparisonSupport

# Load, rather than copy, the exact Turing model and handwritten Julia control
# used by the published primal receipt. The environment variable suppresses only
# that file's terminal run call.
module PublishedMNISTLogisticPrimal
ENV["RK_MNIST_DEFINITIONS_ONLY"] = "1"
include(joinpath(@__DIR__, "mnist_logistic_wren_pca40_comparison_body.jl"))
end
delete!(ENV, "RK_MNIST_DEFINITIONS_ONLY")

const Primal = PublishedMNISTLogisticPrimal
const RK_AD_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
const TURING_AD_BACKEND = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const,
)
const DEFAULT_MNIST_AD_ROUNDS = 10
const MNIST_AD_PACKAGES = (
    "ReactiveKernels", "ReactiveKernelsPPLExamples",
    "ReactiveKernelsDistributionKernels", "Turing", "DynamicPPL",
    "Distributions", "NNlib", "MLDatasets", "DifferentiationInterface",
    "Enzyme", "BenchmarkTools",
)
const MNIST_AD_BOUNDARIES = ("packed_unconstrained", "structured_parameters")
const MNIST_AD_OUTCOMES = ("joint", "prior", "likelihood", "pointwise")
const MNIST_AD_SUPPORTED = Set((
    ("packed_unconstrained", "joint"),
    ("packed_unconstrained", "prior"),
    ("packed_unconstrained", "likelihood"),
))
const MNIST_MODEL_PUBLISHED_SHA =
    "9dbaa1fbfbdb1bb2ff4238a3910754f36635f81e"

_rounds() = parse(Int, get(
    ENV, "RK_MNIST_AD_ROUNDS", string(DEFAULT_MNIST_AD_ROUNDS)))
_observations(profile) = parse(Int, get(
    ENV, "RK_MNIST_AD_N", string(Primal._mnist_default_observations(profile))))

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

function _rk_definition(model, outcome, q, X, y)
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
    (; kernel, arguments, active = :unconstrained)
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
    normalized_read(joinpath(root, relative_path)) == text || error(
        "MNIST model source drifted from published authority " *
        MNIST_MODEL_PUBLISHED_SHA)
    merge(published, Dict("current" => source_pin(root, relative_path)))
end

function _verified_comparator_source_pin(root, relative_path)
    candidate_sha = get(ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown")
    occursin(r"^[0-9a-f]{40}$", candidate_sha) || error(
        "MNIST comparator requires an exact candidate SHA")
    published, text = published_source_pin(
        root, candidate_sha, relative_path)
    current = normalized_read(joinpath(root, relative_path))
    current == text || error(
        "MNIST comparator differs from its clean detached candidate authority")
    merge(published, Dict(
        "current" => source_pin(root, relative_path),
        "current_delta" => "none",
    ))
end

function run_mnist_logistic_ad_comparison()
    rounds = _rounds()
    dataset_profile = Primal._mnist_dataset_profile()
    n = _observations(dataset_profile)
    root = normpath(joinpath(@__DIR__, ".."))

    data_setup = @timed Primal._load_mnist_dataset(
        dataset_profile, n;
        wren_reference = Primal._mnist_wren_reference_path())
    X, y, dataset_metadata = data_setup.value
    GC.gc()
    features = size(X, 2)
    nonreference = NUM_CLASSES - 1
    Random.seed!(20260901)
    W = 0.01 .* randn(nonreference, features)
    b = 0.01 .* randn(nonreference)
    q = vcat(vec(W), b)
    parameters = (; W, b)

    model_setup = @timed build_mnist_logistic_graph()
    model = model_setup.value
    turing_setup = @timed Primal.turing_mnist_logistic(X, y, NUM_CLASSES)
    turing_model = turing_setup.value
    turing_joint = DynamicPPL.LogDensityFunction(
        turing_model, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll();
        fix_transforms = true)
    turing_q = Primal._turing_vector(turing_joint, parameters)
    isapprox(turing_q, q; rtol = 1e-12, atol = 1e-12) ||
        error("Turing's linked parameter vector does not match the RK boundary")

    oracle_setup = @timed _analytic_gradients(q, X, y, nonreference, features)
    analytic_gradients = oracle_setup.value
    measurements = Dict{String,Any}[]
    for boundary in MNIST_AD_BOUNDARIES, outcome in MNIST_AD_OUTCOMES
        row = Dict{String,Any}("boundary" => boundary, "outcome" => outcome)
        if !((boundary, outcome) in MNIST_AD_SUPPORTED)
            row["supported"] = false
            row["unsupported_reason"] = _unsupported_reason(boundary, outcome)
            push!(measurements, row)
            println("boundary=$boundary outcome=$outcome unsupported")
            continue
        end

        manual_definition = _manual_definition(
            outcome, q, X, y, nonreference, features)
        rk_definition = _rk_definition(model, outcome, q, X, y)
        row["supported"] = true
        row["active_port"] = String(rk_definition.active)
        reference_value = Float64(manual_definition.raw(q))
        reference_gradient = analytic_gradients[outcome]
        row["analytic_gradient_length"] = length(reference_gradient)

        rk_call, rk_result, rk_setup = build_and_first_call() do
            prepared = prepare_ad(
                rk_definition.kernel, RK_AD_BACKEND,
                rk_definition.arguments...; active = rk_definition.active)
            RKValueGradientCall(
                prepared, similar(q), rk_definition.arguments)
        end
        row["rk_native"] = record_implementation(
            rk_call, rk_result, rk_setup, reference_value, reference_gradient;
            rounds, caller_owned = true)

        manual_call, manual_result, manual_setup = build_and_first_call() do
            preparation = DifferentiationInterface.prepare_gradient(
                manual_definition.objective, RK_AD_BACKEND,
                q, manual_definition.contexts...)
            DIValueGradientCall(
                manual_definition.objective, similar(q), preparation,
                RK_AD_BACKEND, q, manual_definition.contexts)
        end
        row["manual_enzyme"] = record_implementation(
            manual_call, manual_result, manual_setup,
            reference_value, reference_gradient;
            rounds, caller_owned = true)

        turing_call, turing_result, turing_ad_setup = build_and_first_call() do
            TuringValueGradientCall(
                _turing_logdensity(turing_model, outcome), turing_q)
        end
        row["turing_enzyme"] = record_implementation(
            turing_call, turing_result, turing_ad_setup,
            reference_value, reference_gradient;
            rounds, caller_owned = false)

        push!(measurements, row)
        println("boundary=$boundary outcome=$outcome complete")
    end

    source_path = joinpath(
        "packages", "ReactiveKernelsPPLExamples", "src", "mnist_logistic.jl")
    primal_path = joinpath(
        "benchmark", "receipts", Primal._mnist_primal_receipt_name(dataset_profile))
    comparator_path = joinpath(
        "benchmark", "mnist_logistic_wren_pca40_comparison_body.jl")
    primal_receipt = TOML.parsefile(joinpath(root, primal_path))
    receipt = Dict{String,Any}(
        "schema" => "mnist-logistic-ad-v1",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => get(
                ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown"),
            "reactivekernels_dirty" => false,
            "julia_version" => string(VERSION),
            "model_source" => _verified_model_source_pin(root, source_path),
            "primal_comparator_source" =>
                _verified_comparator_source_pin(root, comparator_path),
            "primal_receipt_path" => primal_path,
            "primal_receipt_sha256" => text_sha256(joinpath(root, primal_path)),
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
        "protocol" => merge(Dict(
            "model" => "multinomial-logistic MNIST classifier",
            "num_classes" => NUM_CLASSES,
            "active_parameter_count" => length(q),
            "input_boundaries" => collect(MNIST_AD_BOUNDARIES),
            "outcomes" => collect(MNIST_AD_OUTCOMES),
            "source_reused" => true,
            "matrix_source" => primal_path,
            "gradient_operation" => "value and gradient",
            "rk_surface" => "prepare_ad + ad_value_and_gradient!",
            "rk_backend" => "AutoEnzyme(mode = Enzyme.Reverse)",
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
        ), dataset_metadata),
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

run_mnist_logistic_ad_comparison()
