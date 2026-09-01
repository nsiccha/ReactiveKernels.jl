# Inner body for the partial-evaluation MNIST comparison. The model variant
# below authors the same multinomial-logistic density as the landed docs model,
# but derives the reference-logits row from the data alone
# (`zeros(1, size(X, 1))` instead of `zeros(1, size(nonreference_logits, 2))`),
# so the row — and every other value in the data-only closure — is hoistable by
# the opt-in `bound` pre-pass. The unbound preparation of the same variant is
# the baseline; the landed docs model anchors value parity. Dropping the vcat
# from the models entirely is the separate model-level optimization tracked by
# the :ppl lane.

using BenchmarkTools
using Dates
using Pkg
using Random
using Statistics
using TOML
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, categorical_logit
using ReactiveKernelsPPLExamples.MNISTLogisticExample:
    build_mnist_logistic_graph, NUM_CLASSES
import MLDatasets

const DEFAULT_ROUNDS = 10
# The published receipt fits the full MNIST training split; RK_MNIST_N
# overrides it for a quicker local reproduction.
const DEFAULT_N = 60000
const COMPARISON_PACKAGES = (
    "ReactiveKernels", "ReactiveKernelsPPLExamples",
    "ReactiveKernelsDistributionKernels", "MLDatasets", "BenchmarkTools",
)

function build_partial_evaluation_variant()
    @kernel pe_mnist_model(unconstrained::Vector{Float64},
                           X::Matrix{Float64}, y::Vector{Int},
                           num_classes::Int) = begin
        feature_count::Int = size(X, 2)
        observation_count::Int = size(X, 1)
        nonreference::Int = num_classes - 1
        W::Matrix{Float64} =
            reshape(view(unconstrained, 1:(nonreference * feature_count)),
                    nonreference, feature_count)
        b::Vector{Float64} =
            unconstrained[(nonreference * feature_count + 1):length(unconstrained)]

        prior_terms = plate(unconstrained) do coefficient
            normal(0.0, 1.0).logpdf(coefficient)
        end
        prior::Float64 = sum(prior_terms)

        # Data-only: shape depends on X alone, so the `bound` pre-pass runs it
        # once per preparation instead of on every call.
        reference_row::Matrix{Float64} = zeros(1, observation_count)
        nonreference_logits = W * transpose(X) .+ b
        logits = vcat(reference_row, nonreference_logits)
        pointwise = plate(eachcol(logits), y) do observation_logits, observed_class
            categorical_logit(observation_logits).logpdf(observed_class)
        end
        likelihood::Float64 = sum(pointwise)
        density::Float64 = prior + likelihood
    end
end

function _output_path()
    for arg in ARGS
        startswith(arg, "--output=") && return split(arg, '='; limit = 2)[2]
    end
    nothing
end

_rounds() = parse(Int, get(ENV, "RK_PE_ROUNDS", string(DEFAULT_ROUNDS)))
_observations() = parse(Int, get(ENV, "RK_MNIST_N", string(DEFAULT_N)))

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

function _measurement(f, args...; rounds::Int)
    invocation = let f = f, args = args
        () -> f(args...)
    end
    benchmark = @benchmarkable $invocation()
    times_ns = Float64[]; bytes = Int[]; allocs = Int[]
    for _ in 1:rounds
        # This receipt's timed cells are all ms-scale (one dense matmul
        # dominates), so the sibling harnesses' 0.2 s round budget would admit
        # only ~2 evaluations per round; a 1 s budget keeps the per-round
        # minimum meaningful on a shared host.
        estimate = minimum(run(benchmark; samples = 200, seconds = 1.0))
        push!(times_ns, estimate.time)
        push!(bytes, estimate.memory)
        push!(allocs, estimate.allocs)
    end
    Dict(
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

function run_comparison()
    rounds = _rounds()
    n = _observations()
    X, y = _load_mnist(n)
    features = size(X, 2)
    nonreference = NUM_CLASSES - 1

    Random.seed!(20260901)
    W = 0.01 .* randn(nonreference, features)
    b = 0.01 .* randn(nonreference)
    unconstrained = vcat(vec(W), b)

    variant = build_partial_evaluation_variant()
    have = (:unconstrained, :X, :y, :num_classes)
    docs_model = build_mnist_logistic_graph()

    definitions = Dict{String,Any}[]
    for outcome in (:density, :likelihood, :prior)
        unbound = prepare(variant; have = have, want = outcome)
        bound = prepare(variant; have = have, want = outcome,
                        bound = (; X, y, num_classes = NUM_CLASSES))
        reference = prepare(docs_model; have = have, want = outcome)(
            unconstrained, X, y, NUM_CLASSES)

        unbound_value = unbound(unconstrained, X, y, NUM_CLASSES)
        bound_value = bound(unconstrained)
        for (label, value) in ("rk_unbound" => unbound_value,
                               "rk_bound" => bound_value)
            isapprox(value, reference; rtol = 1e-9, atol = 1e-9) || error(
                "parity failed for $outcome / $label against the landed docs model")
        end

        row = Dict{String,Any}(
            "outcome" => string(outcome),
            "reference_value" => reference,
            "rk_unbound" => _measurement(
                unbound, unconstrained, X, y, NUM_CLASSES; rounds),
            "rk_bound" => _measurement(bound, unconstrained; rounds))
        push!(definitions, row)
        println("outcome=$outcome complete")
    end

    # Pass introspection for the receipt: how much of the plan became
    # bind-time work (internal `_BoundConstant` naming is deliberate here —
    # a benchmark body may report compiler internals).
    p = plan(variant; have = have, want = :density)
    bound_ports = [v for v in p.have if v.name in (:X, :y, :num_classes)]
    residual_plan = partial_evaluation(
        p, Tuple(bound_ports), (X, y, NUM_CLASSES))
    constant_count = count(
        r -> r.op isa ReactiveKernels._BoundConstant, residual_plan.recipes)
    residual_count = length(residual_plan.recipes) - constant_count

    receipt = Dict{String,Any}(
        "schema" => "partial-evaluation-mnist-v1",
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
        "protocol" => Dict(
            "model" => "multinomial-logistic MNIST classifier, " *
                       "data-only reference-logits row variant",
            "data" => "MLDatasets MNIST train split, first N images",
            "num_observations" => n,
            "num_features" => features,
            "num_classes" => NUM_CLASSES,
            "input_boundary" => "packed_unconstrained",
            "outcomes" => ["density", "likelihood", "prior"],
            "bound_ports" => ["X", "y", "num_classes"],
            "plan_recipes_total" => length(p.recipes),
            "residual_recipes" => residual_count,
            "hoisted_constants" => constant_count,
            "rounds" => rounds,
            "estimator" => "minimum of per-round BenchmarkTools minimum " *
                           "times (uncontended cost; medians and raw rounds retained)",
            "samples_per_round" => 200,
            "seconds_per_round" => 1.0,
            "setup_in_timed_region" => false,
            "preparation_in_timed_region" => false,
            "parity_reference" => "landed docs model (build_mnist_logistic_graph)",
            "parity_rtol" => 1e-9,
            "parity_atol" => 1e-9),
        "measurements" => definitions)

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

run_comparison()
