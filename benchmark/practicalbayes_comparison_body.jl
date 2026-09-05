# Inner body for the exact-pin PracticalBayes PPL comparator. The benchmark
# exercises only documented public APIs: @model, build_layout,
# LogDensityFunction, fixed-parameter density views, and returned. No source
# from the unlicensed upstream repository is copied here.

using ADTypes: AutoEnzyme
using BenchmarkTools
using Dates
using Distributions: Categorical, Cauchy, MvNormal, Normal, logpdf, truncated
using LinearAlgebra: I
using NNlib: softmax
using Random
using SHA
using Statistics
using TOML
using ReactiveKernelsPPLExamples.EightSchoolsExample:
    EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA
import Enzyme
import LogDensityProblems as LDP
import Pkg
import PracticalBayes
using PracticalBayes: @addlogprob!

include(joinpath(@__DIR__, "_mnist_dataset_profiles.jl"))

const PB_BACKEND = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
const PB_DEFAULT_ROUNDS = 10
const PB_DEFAULT_SAMPLES_PER_ROUND = 20
const PB_DEFAULT_MNIST_N = 60000
const PB_SEED = 20260901

const PB_PACKAGES = (
    "PracticalBayes", "ADTypes", "BenchmarkTools", "Distributions", "Enzyme",
    "LogDensityProblems", "MLDatasets", "NNlib",
)

# DOCS-BASELINE-BEGIN: practicalbayes-eight-schools
PracticalBayes.@model function practicalbayes_eight_schools(
        observations, observation_scales)
    μ ~ Normal(0, 5)
    τ ~ truncated(Cauchy(0, 5); lower = 0)
    θ ~ PracticalBayes.filldist(Normal(μ, τ), length(observations))
    observations .~ Normal.(θ, observation_scales)
    return nothing
end
# DOCS-BASELINE-END: practicalbayes-eight-schools

# DOCS-BASELINE-BEGIN: practicalbayes-mnist-idiomatic
PracticalBayes.@model function practicalbayes_mnist_idiomatic(X, y, C)
    N, D = size(X)
    W ~ PracticalBayes.filldist(Normal(), C - 1, D)
    b ~ PracticalBayes.filldist(Normal(), C - 1)
    nonreference_logits = W * transpose(X) .+ b
    logits = vcat(zeros(eltype(nonreference_logits), 1, N), nonreference_logits)
    probabilities = softmax(logits; dims = 1)
    # Categorical currently requires an owned probability vector rather than a
    # column view. This allocation is part of the public model evaluation and
    # therefore remains inside every timed call.
    y .~ Categorical.(collect.(eachcol(probabilities)))
    return nothing
end
# DOCS-BASELINE-END: practicalbayes-mnist-idiomatic

# DOCS-BASELINE-BEGIN: practicalbayes-mnist-optimized
PracticalBayes.@model function practicalbayes_mnist_optimized(X, y, C)
    _, D = size(X)
    W ~ PracticalBayes.filldist(Normal(), C - 1, D)
    b ~ PracticalBayes.filldist(Normal(), C - 1)
    nonreference = W * transpose(X) .+ b
    total = zero(eltype(nonreference))
    @inbounds for j in eachindex(y)
        column = @view nonreference[:, j]
        maximum_logit = max(zero(eltype(nonreference)), maximum(column))
        accumulated = exp(-maximum_logit)
        for logit in column
            accumulated += exp(logit - maximum_logit)
        end
        log_normalizer = maximum_logit + log(accumulated)
        true_logit = y[j] == 1 ? zero(eltype(nonreference)) : column[y[j] - 1]
        total += true_logit - log_normalizer
    end
    @addlogprob! total
    return nothing
end
# DOCS-BASELINE-END: practicalbayes-mnist-optimized

# DOCS-BASELINE-BEGIN: practicalbayes-eval-throughput
PracticalBayes.@model function practicalbayes_normal_evaluation(μ, logσ, n)
    x ~ MvNormal(fill(μ, n), exp(logσ)^2 * I)
    return -0.5 .* ((x .- μ) ./ exp(logσ)) .^ 2 .-
        (logσ + 0.5 * log(2π))
end
# DOCS-BASELINE-END: practicalbayes-eval-throughput

_rounds() = parse(Int, get(ENV, "RK_PRACTICALBAYES_ROUNDS",
                                string(PB_DEFAULT_ROUNDS)))
_samples_per_round() = parse(Int, get(
    ENV, "RK_PRACTICALBAYES_SAMPLES_PER_ROUND",
    string(PB_DEFAULT_SAMPLES_PER_ROUND)))
_mnist_observations() = parse(Int, get(
    ENV, "RK_PRACTICALBAYES_MNIST_N", string(PB_DEFAULT_MNIST_N)))
_sections() = Tuple(filter(!isempty, split(get(
    ENV, "RK_PRACTICALBAYES_SECTIONS", "models,eval"), ',')))

function _output_path()
    matches = [split(arg, '='; limit = 2)[2] for arg in ARGS
               if startswith(arg, "--output=")]
    length(matches) <= 1 || error("--output may be supplied only once")
    isempty(matches) ? nothing : only(matches)
end

function _package_version(name)
    for info in values(Pkg.dependencies())
        info.name == name && return string(info.version)
    end
    error("package $name absent from the PracticalBayes environment")
end

function _package_git_revision(name)
    for info in values(Pkg.dependencies())
        info.name == name &&
            return string(something(info.git_revision, "unknown"))
    end
    error("package $name absent from the PracticalBayes environment")
end

function _source_sha256()
    bytes2hex(sha256(read(@__FILE__)))
end

function _measurement(thunk; rounds, samples_per_round)
    benchmark = @benchmarkable $thunk()
    times_ns = Float64[]
    bytes = Int[]
    allocs = Int[]
    for _ in 1:rounds
        estimate = minimum(run(
            benchmark; samples = samples_per_round, seconds = 0.2))
        push!(times_ns, estimate.time)
        push!(bytes, estimate.memory)
        push!(allocs, estimate.allocs)
    end
    Dict(
        "times_ns" => times_ns,
        "min_ns" => minimum(times_ns),
        "median_ns" => median(times_ns),
        "bytes" => bytes,
        "median_bytes" => Int(median(bytes)),
        "allocs" => allocs,
        "median_allocs" => Int(median(allocs)),
    )
end

_comparable(outcome, value) = outcome == "pointwise" ?
    collect(Float64, value) : Float64(value)

_float_vector(value::Number) = [Float64(value)]
_float_vector(value::AbstractArray) = vec(collect(Float64, value))
_float_vector(value::Tuple) = vcat((_float_vector(part) for part in value)...)

function _first_line(err)
    first(split(sprint(showerror, err), '\n'))
end

function _prepared_model(workload, model_name, model, init)
    layout = q = store = primal = ad = nothing
    layout_seconds = @elapsed begin
        layout, q, store = PracticalBayes.build_layout(model; init)
    end
    primal_prepare_seconds = @elapsed begin
        primal = PracticalBayes.LogDensityFunction(model, layout, store)
    end
    primal_first_call_seconds = @elapsed LDP.logdensity(primal, q)
    ad_prepare_seconds = @elapsed begin
        ad = PracticalBayes.LogDensityFunction(
            model, layout, store, PB_BACKEND; θ0 = q)
    end
    ad_first_call_seconds = @elapsed LDP.logdensity_and_gradient(ad, q)
    setup = Dict{String,Any}(
        "workload" => workload,
        "model" => model_name,
        "layout_seconds" => layout_seconds,
        "primal_prepare_seconds" => primal_prepare_seconds,
        "primal_first_call_seconds" => primal_first_call_seconds,
        "ad_prepare_seconds" => ad_prepare_seconds,
        "ad_first_call_seconds" => ad_first_call_seconds,
    )
    (; model, layout, q, store, primal, ad, setup)
end

function _normal_logpdf(x, location, scale)
    -0.5 * log(2π) - log(scale) - 0.5 * ((x - location) / scale)^2
end

function _eight_pointwise(θ, observations, observation_scales)
    [_normal_logpdf(observations[j], θ[j], observation_scales[j])
     for j in eachindex(observations)]
end

function _eight_prior(μ, τ, θ)
    total = _normal_logpdf(μ, 0.0, 5.0)
    total += log(2.0) - log(π) - log(5.0) - log1p((τ / 5.0)^2)
    total + sum(_normal_logpdf(value, μ, τ) for value in θ)
end

function _eight_joint_unconstrained(q, observations, observation_scales)
    μ = q[1]
    τ = exp(q[2])
    θ = @view q[3:end]
    _eight_prior(μ, τ, θ) + sum(_eight_pointwise(
        θ, observations, observation_scales)) + q[2]
end

function _eight_gradient(q, observations, observation_scales)
    μ = q[1]
    logτ = q[2]
    τ = exp(logτ)
    θ = @view q[3:end]
    gradient = similar(q)
    gradient[1] = -μ / 25 + sum((θ .- μ) ./ τ^2)
    gradient[2] = -2τ^2 / (25 + τ^2) - length(θ) +
        sum(abs2, θ .- μ) / τ^2 + 1
    @views gradient[3:end] .= -(θ .- μ) ./ τ^2 .+
        (observations .- θ) ./ observation_scales.^2
    gradient
end

function _model_row(; workload, model, configuration, boundary, outcome,
                    differentiation, state = "supported", reason = "",
                    diagnostic = "", parity_max_abs = nothing, result = nothing)
    row = Dict{String,Any}(
        "workload" => workload,
        "provider" => "practical_bayes",
        "model" => model,
        "configuration" => configuration,
        "boundary" => boundary,
        "outcome" => outcome,
        "differentiation" => differentiation,
        "state" => state,
    )
    isempty(reason) || (row["reason"] = reason)
    isempty(diagnostic) || (row["diagnostic"] = diagnostic)
    parity_max_abs === nothing || (row["parity_max_abs"] = parity_max_abs)
    result === nothing || (row["result"] = result)
    row
end

function _eight_schools_rows!(measurements, preparations; rounds, samples_per_round)
    observations = Float64.(EIGHT_SCHOOLS_Y)
    observation_scales = Float64.(EIGHT_SCHOOLS_SIGMA)
    parameters = (;
        μ = 1.5,
        τ = 2.0,
        θ = 0.25 .* collect(1.0:length(observations)),
    )
    model = practicalbayes_eight_schools(observations, observation_scales)
    prepared = _prepared_model("eight_schools", "centered", model, parameters)
    push!(preparations, prepared.setup)
    reference_joint = _eight_joint_unconstrained(
        prepared.q, observations, observation_scales)
    reference_gradient = _eight_gradient(
        prepared.q, observations, observation_scales)
    expected_q = [parameters.μ, log(parameters.τ), parameters.θ...]
    isapprox(prepared.q, expected_q; rtol = 1e-12, atol = 1e-12) ||
        error("PracticalBayes Eight Schools linked vector order changed")

    for boundary in ("packed_unconstrained", "constrained_parameters",
                     "minimal_likelihood"),
        outcome in ("joint", "prior", "likelihood", "pointwise"),
        differentiation in ("primal", "value_and_gradient")
        configuration = differentiation == "primal" ?
            "practicalbayes_primal" : "practicalbayes_ad"
        supported = (boundary == "packed_unconstrained" && outcome == "joint") ||
            (boundary == "constrained_parameters" && differentiation == "primal")
        if !supported
            reason = if boundary == "minimal_likelihood"
                "the public model evaluator requires the complete named parameter set; no theta-only likelihood boundary is exposed"
            elseif boundary == "packed_unconstrained"
                "LogDensityFunction exposes the transformed full joint only; no public linked prior, likelihood, or pointwise view is exposed"
            else
                "the public prepared AD surface accepts the linked vector only; no constrained NamedTuple AD boundary is exposed"
            end
            push!(measurements, _model_row(;
                workload = "eight_schools", model = "centered",
                configuration, boundary, outcome, differentiation,
                state = "unsupported", reason))
            continue
        end

        thunk, reference = if boundary == "packed_unconstrained"
            differentiation == "primal" ?
                (() -> LDP.logdensity(prepared.primal, prepared.q), reference_joint) :
                (() -> LDP.logdensity_and_gradient(prepared.ad, prepared.q),
                 (reference_joint, reference_gradient))
        else
            fixed = outcome == "joint" ?
                (() -> PracticalBayes.logjoint(model, parameters)) :
                outcome == "prior" ?
                (() -> PracticalBayes.logprior(model, parameters)) :
                outcome == "likelihood" ?
                (() -> PracticalBayes.loglikelihood_at(model, parameters)) :
                (() -> PracticalBayes.pointwise_loglikelihoods(
                    model, parameters; flatten = true))
            ref = outcome == "prior" ?
                _eight_prior(parameters.μ, parameters.τ, parameters.θ) :
                outcome == "likelihood" ?
                sum(_eight_pointwise(parameters.θ, observations, observation_scales)) :
                outcome == "pointwise" ?
                _eight_pointwise(parameters.θ, observations, observation_scales) :
                _eight_prior(parameters.μ, parameters.τ, parameters.θ) +
                    sum(_eight_pointwise(parameters.θ, observations, observation_scales))
            (fixed, ref)
        end
        actual = thunk()
        actual_values = _float_vector(actual)
        reference_values = _float_vector(reference)
        gap = maximum(abs.(actual_values .- reference_values))
        isapprox(actual_values, reference_values; rtol = 1e-9, atol = 1e-9) ||
            error("PracticalBayes Eight Schools parity failed for $boundary / $outcome / $differentiation")
        push!(measurements, _model_row(;
            workload = "eight_schools", model = "centered",
            configuration, boundary, outcome, differentiation,
            parity_max_abs = gap,
            result = _measurement(thunk; rounds, samples_per_round)))
    end
end

function _mnist_unpack(q, nonreference, features)
    stop = nonreference * features
    reshape(view(q, 1:stop), nonreference, features), view(q, (stop + 1):length(q))
end

function _mnist_pointwise(q, X, y, C)
    W, b = _mnist_unpack(q, C - 1, size(X, 2))
    nonreference = W * transpose(X) .+ b
    output = similar(y, Float64)
    @inbounds for j in eachindex(y)
        column = @view nonreference[:, j]
        maximum_logit = max(0.0, maximum(column))
        accumulated = exp(-maximum_logit)
        for logit in column
            accumulated += exp(logit - maximum_logit)
        end
        log_normalizer = maximum_logit + log(accumulated)
        output[j] = (y[j] == 1 ? 0.0 : column[y[j] - 1]) - log_normalizer
    end
    output
end

_mnist_prior(q) = -0.5 * length(q) * log(2π) - 0.5 * sum(abs2, q)
_mnist_joint(q, X, y, C) = _mnist_prior(q) + sum(_mnist_pointwise(q, X, y, C))

function _mnist_gradient(q, X, y, C)
    nonreference = C - 1
    W, b = _mnist_unpack(q, nonreference, size(X, 2))
    gradient_W = -copy(W)
    gradient_b = -copy(b)
    logits = W * transpose(X) .+ b
    @inbounds for j in eachindex(y)
        column = @view logits[:, j]
        maximum_logit = max(0.0, maximum(column))
        denominator = exp(-maximum_logit)
        for logit in column
            denominator += exp(logit - maximum_logit)
        end
        for c in 1:nonreference
            probability = exp(column[c] - maximum_logit) / denominator
            residual = (y[j] == c + 1 ? 1.0 : 0.0) - probability
            @views gradient_W[c, :] .+= residual .* X[j, :]
            gradient_b[c] += residual
        end
    end
    vcat(vec(gradient_W), gradient_b)
end

function _mnist_rows!(measurements, preparations, datasets;
                      rounds, samples_per_round)
    n = _mnist_observations()
    X, y, metadata = _load_mnist_dataset(MNIST_DATASET_FULL_RAW, n)
    datasets["mnist_logistic_full_raw"] = metadata
    nonreference = 9
    features = size(X, 2)
    rng = Random.Xoshiro(PB_SEED)
    W = 0.01 .* randn(rng, nonreference, features)
    b = 0.01 .* randn(rng, nonreference)
    parameters = (; W, b)
    reference_q = vcat(vec(W), b)
    reference_joint = _mnist_joint(reference_q, X, y, 10)
    reference_gradient = _mnist_gradient(reference_q, X, y, 10)
    models = (
        (name = "idiomatic", model = practicalbayes_mnist_idiomatic(X, y, 10)),
        (name = "vcat_free", model = practicalbayes_mnist_optimized(X, y, 10)),
    )

    for definition in models
        prepared = _prepared_model(
            "mnist_logistic", definition.name, definition.model, parameters)
        push!(preparations, prepared.setup)
        isapprox(prepared.q, reference_q; rtol = 1e-12, atol = 1e-12) ||
            error("PracticalBayes MNIST linked vector order changed")
        pointwise_diagnostic = ""
        if definition.name == "vcat_free"
            pointwise_diagnostic = try
                PracticalBayes.pointwise_loglikelihoods(
                    definition.model, parameters; flatten = true)
                error("optimized PracticalBayes pointwise unexpectedly succeeded")
            catch err
                _first_line(err)
            end
        end

        for boundary in ("packed_unconstrained", "structured_parameters"),
            outcome in ("joint", "prior", "likelihood", "pointwise"),
            differentiation in ("primal", "value_and_gradient")
            configuration = differentiation == "primal" ?
                "practicalbayes_$(definition.name)_primal" :
                "practicalbayes_$(definition.name)_ad"
            supported = (boundary == "packed_unconstrained" && outcome == "joint") ||
                (boundary == "structured_parameters" &&
                 differentiation == "primal" &&
                 !(outcome == "pointwise" && definition.name == "vcat_free"))
            if !supported
                reason, diagnostic = if boundary == "packed_unconstrained"
                    ("LogDensityFunction exposes the full joint only; no public packed prior, likelihood, or pointwise view is exposed", "")
                elseif differentiation == "value_and_gradient"
                    ("the public prepared AD surface accepts the packed vector only; no structured (W, b) AD boundary is exposed", "")
                else
                    ("the optimized model accumulates one scalar @addlogprob! likelihood, so the public pointwise accessor has no observation-site vector", pointwise_diagnostic)
                end
                push!(measurements, _model_row(;
                    workload = "mnist_logistic", model = definition.name,
                    configuration, boundary, outcome, differentiation,
                    state = "unsupported", reason, diagnostic))
                continue
            end

            thunk, reference = if boundary == "packed_unconstrained"
                differentiation == "primal" ?
                    (() -> LDP.logdensity(prepared.primal, prepared.q), reference_joint) :
                    (() -> LDP.logdensity_and_gradient(prepared.ad, prepared.q),
                     (reference_joint, reference_gradient))
            else
                fixed = outcome == "joint" ?
                    (() -> PracticalBayes.logjoint(definition.model, parameters)) :
                    outcome == "prior" ?
                    (() -> PracticalBayes.logprior(definition.model, parameters)) :
                    outcome == "likelihood" ?
                    (() -> PracticalBayes.loglikelihood_at(
                        definition.model, parameters)) :
                    (() -> PracticalBayes.pointwise_loglikelihoods(
                        definition.model, parameters; flatten = true))
                ref = outcome == "joint" ? reference_joint :
                      outcome == "prior" ? _mnist_prior(reference_q) :
                      outcome == "likelihood" ?
                        sum(_mnist_pointwise(reference_q, X, y, 10)) :
                        _mnist_pointwise(reference_q, X, y, 10)
                (fixed, ref)
            end
            actual = thunk()
            actual_values = _float_vector(actual)
            reference_values = _float_vector(reference)
            gap = maximum(abs.(actual_values .- reference_values))
            isapprox(actual_values, reference_values; rtol = 1e-8, atol = 1e-8) ||
                error("PracticalBayes MNIST parity failed for $(definition.name) / $boundary / $outcome / $differentiation")
            push!(measurements, _model_row(;
                workload = "mnist_logistic", model = definition.name,
                configuration, boundary, outcome, differentiation,
                parity_max_abs = gap,
                result = _measurement(thunk; rounds, samples_per_round)))
        end
    end
end

_eval_inputs(n) = (;
    μ = 0.3,
    logσ = log(1.3),
    x = [0.7 * sin(0.31i) - 0.2 * cos(0.17i) for i in 1:n],
)
_eval_pointwise(x, μ, logσ) =
    -0.5 .* ((x .- μ) ./ exp(logσ)) .^ 2 .- (logσ + 0.5 * log(2π))
_eval_gradient(x, μ, logσ) = -(x .- μ) ./ exp(logσ)^2

function _eval_rows!(measurements, preparations; rounds, samples_per_round)
    for n in (16, 256, 4096)
        (; μ, logσ, x) = _eval_inputs(n)
        model = practicalbayes_normal_evaluation(μ, logσ, n)
        prepared = _prepared_model(
            "eval_throughput", "iid_normal_n_$n", model, (; x))
        push!(preparations, prepared.setup)
        expected = Dict(
            "primal" => sum(_eval_pointwise(x, μ, logσ)),
            "gradient" => _eval_gradient(x, μ, logσ),
            "gq" => _eval_pointwise(x, μ, logσ),
        )
        thunks = Dict(
            "primal" => (() -> LDP.logdensity(prepared.primal, prepared.q)),
            "gradient" => (() -> last(
                LDP.logdensity_and_gradient(prepared.ad, prepared.q))),
            "gq" => (() -> PracticalBayes.returned(model, (; x))),
        )
        for mode in ("primal", "gradient", "gq")
            actual = thunks[mode]()
            gap = maximum(abs.(_float_vector(actual) .-
                               _float_vector(expected[mode])))
            isapprox(actual, expected[mode]; rtol = 1e-9, atol = 1e-9) ||
                error("PracticalBayes evaluation parity failed for n=$n / $mode")
            push!(measurements, Dict{String,Any}(
                "size" => n,
                "mode" => mode,
                "implementation" => "practical_bayes",
                "variant" => "native",
                "state" => "supported",
                "parity_max_abs" => gap,
                "result" => _measurement(
                    thunks[mode]; rounds, samples_per_round),
            ))
            push!(measurements, Dict{String,Any}(
                "size" => n,
                "mode" => mode,
                "implementation" => "practical_bayes",
                "variant" => "reactant",
                "state" => "unsupported",
                "reason" => "the pinned public API exposes no Reactant compilation boundary; no compiled comparator is invented",
            ))
        end
    end
end

function run_comparison()
    rounds = _rounds()
    samples_per_round = _samples_per_round()
    sections = _sections()
    all(section -> section in ("models", "eval"), sections) ||
        error("RK_PRACTICALBAYES_SECTIONS accepts models,eval")

    model_measurements = Dict{String,Any}[]
    eval_measurements = Dict{String,Any}[]
    preparations = Dict{String,Any}[]
    datasets = Dict{String,Any}()
    "models" in sections && begin
        _eight_schools_rows!(model_measurements, preparations;
            rounds, samples_per_round)
        _mnist_rows!(model_measurements, preparations, datasets;
            rounds, samples_per_round)
    end
    "eval" in sections && _eval_rows!(
        eval_measurements, preparations; rounds, samples_per_round)

    receipt = Dict{String,Any}(
        "schema" => "practicalbayes-comparison-v1",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => get(
                ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown"),
            "reactivekernels_dirty" => false,
            "practicalbayes_repository" => get(
                ENV, "RK_PRACTICALBAYES_REPOSITORY", "unknown"),
            "practicalbayes_revision" => get(
                ENV, "RK_PRACTICALBAYES_REVISION", "unknown"),
            "practicalbayes_git_revision" =>
                _package_git_revision("PracticalBayes"),
            "benchmark_body_sha256" => _source_sha256(),
            "julia_version" => string(VERSION),
            (string(lowercase(name), "_version") => _package_version(name)
             for name in PB_PACKAGES)...,
        ),
        "environment" => Dict(
            "os" => string(Sys.KERNEL),
            "arch" => string(Sys.ARCH),
            "cpu" => first(Sys.cpu_info()).model,
            "julia_threads" => Threads.nthreads(),
            "env_setup_seconds" => parse(Float64, get(
                ENV, "RK_PRACTICALBAYES_ENV_SETUP_SECONDS", "0")),
            "precompile_seconds" => parse(Float64, get(
                ENV, "RK_PRACTICALBAYES_PRECOMPILE_SECONDS", "0")),
        ),
        "compatibility" => Dict(
            "separate_environment" => true,
            "coexists_with_pinned_turing_environment" => false,
            "solver_diagnostic" => "DynamicPPL 0.42.6 is incompatible with PracticalBayes' Bijectors 0.15.24 pin",
            "reproduction" => "add PracticalBayes c6b340ba, then add Turing 0.47.1 and DynamicPPL 0.42.6 in the same Julia 1.10 environment",
            "upstream_license_file_at_pin" => "absent",
            "source_policy" => "public API calls only; no PracticalBayes source copied or vendored",
        ),
        "protocol" => Dict(
            "sections" => collect(sections),
            "model_matrix" => ["eight_schools", "mnist_logistic"],
            "model_boundaries" => Dict(
                "eight_schools" => ["packed_unconstrained", "constrained_parameters", "minimal_likelihood"],
                "mnist_logistic" => ["packed_unconstrained", "structured_parameters"],
            ),
            "model_outcomes" => ["joint", "prior", "likelihood", "pointwise"],
            "differentiation" => ["primal", "value_and_gradient"],
            "gradient_backend" => "ADTypes.AutoEnzyme with Enzyme reverse runtime activity",
            "eval_sizes" => [16, 256, 4096],
            "eval_modes" => ["primal", "gradient", "gq"],
            "mnist_full_observations" => _mnist_observations(),
            "rounds" => rounds,
            "samples_per_round" => samples_per_round,
            "estimator" => "minimum of each BenchmarkTools round; min, median, raw times, bytes, and allocations retained",
            "setup_in_timed_region" => false,
            "preparation_in_timed_region" => false,
            "warmup_in_timed_region" => false,
            "precision" => "Float64 parameters and data",
            "julia_threads" => Threads.nthreads(),
            "deterministic_seeds" => true,
            "unsupported_cells_recorded" => true,
            "parity_gate" => "closed-form/manual value and analytic gradient before timing; rtol/atol 1e-8 or tighter",
        ),
        "preparations" => preparations,
        "datasets" => datasets,
        "model_measurements" => model_measurements,
        "eval_measurements" => eval_measurements,
    )

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
