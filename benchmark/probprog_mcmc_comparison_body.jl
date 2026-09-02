# Inner body for the ProbProg NUTS sampling comparison. One RK-authored model
# density per case (Eight Schools packed posterior; MNIST multinomial logistic
# packed joint on the Wren PCA-40 workload) is sampled by three NUTS harnesses:
# Reactant's ProbProg compiled `mcmc_logpdf` consuming the RK density, native
# AdvancedHMC consuming the RK density and its prepared DI+Enzyme gradient, and
# Turing's NUTS on the source-attested Turing twin of the same model. Density
# parity between the harnesses is gated deterministically before any sampling
# statistic is recorded; sampler trajectories are compared statistically, never
# exactly (different adaptation schedules and tree implementations).

using Dates
using LinearAlgebra
using Pkg
using Random
using SHA
using Statistics
using TOML
using ReactiveKernels
using ReactiveKernelsPPLExamples.EightSchoolsExample:
    build_eight_schools_graph, EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA,
    EIGHT_SCHOOLS_SOURCE
using ReactiveKernelsPPLExamples.MNISTLogisticExample:
    build_mnist_logistic_graph, MNIST_LOGISTIC_SOURCE, NUM_CLASSES
import AdvancedHMC
import DynamicPPL
import Enzyme
import MCMCDiagnosticTools
import MLDatasets
import Turing
using ADTypes: AutoEnzyme
using Distributions: Cauchy, Normal, truncated
using NNlib: softmax
using Turing: filldist
using Reactant
using Reactant: ProbProg, ReactantRNG

include(joinpath(@__DIR__, "_mnist_dataset_profiles.jl"))

const RK_BOUND_AD_BACKEND = AutoEnzyme(
    ; mode = Enzyme.Reverse, function_annotation = Enzyme.Const)
const TURING_AD_BACKEND = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const,
)

const DEFAULT_WARMUP = 1000
const DEFAULT_SAMPLES = 1000
const DEFAULT_SEED = 20260901
const TARGET_ACCEPT = 0.8
const MAX_TREE_DEPTH = 10

# ---- the three sampling harnesses --------------------------------------------

# DOCS-BASELINE-BEGIN: probprog
# The whole warmup + sampling loop is one compiled Reactant program: NUTS with
# dual-averaging step-size and Welford diagonal mass-matrix adaptation runs as
# a single MLIR MCMC operation whose gradients Enzyme takes inside the compiled
# program. `logdensity` wraps the unmodified RK prepared kernel (data bound at
# preparation); ProbProg treats it as an opaque traced log density.
function probprog_nuts_program(rng, logdensity, position, step_size, inv_mass,
                               num_warmup::Int, num_samples::Int)
    samples, diagnostics, log_densities, _, _ = ProbProg.mcmc_logpdf(
        rng, logdensity, position;
        algorithm = :NUTS,
        step_size,
        inverse_mass_matrix = inv_mass,
        max_tree_depth = MAX_TREE_DEPTH,
        num_warmup, num_samples,
        adapt_step_size = true,
        adapt_mass_matrix = true)
    return samples, diagnostics, log_densities
end

function probprog_nuts(logdensity, initial_position; seed, num_warmup, num_samples)
    dimension = length(initial_position)
    rng = ReactantRNG(Reactant.to_rarray(UInt64[seed, seed + 1]))
    position = Reactant.to_rarray(reshape(copy(initial_position), 1, dimension))
    step_size = Reactant.ConcreteRNumber(0.1)
    inv_mass = Reactant.to_rarray(ones(dimension))
    compile_seconds = @elapsed compiled =
        Reactant.@compile optimize = :probprog sync = true probprog_nuts_program(
            rng, logdensity, position, step_size, inv_mass,
            num_warmup, num_samples)
    sampling_seconds = @elapsed raw = compiled(
        rng, logdensity, position, step_size, inv_mass, num_warmup, num_samples)
    (; samples = Array(raw[1]),
       diagnostics = Array(raw[2]),
       log_densities = vec(Array(raw[3])),
       compile_seconds, sampling_seconds)
end
# DOCS-BASELINE-END: probprog

# DOCS-BASELINE-BEGIN: advancedhmc
# Standard adaptive AdvancedHMC NUTS (multinomial, generalised no-U-turn,
# Stan-style joint step-size/diagonal-metric adaptation) over the same RK
# density. The value closure is the RK prepared kernel; the gradient closure is
# the RK prepared DI+Enzyme reverse gradient — no hand-written density and no
# hand-written derivative anywhere.
function advancedhmc_nuts(value_f, value_and_gradient_f, initial_position;
                          seed, num_warmup, num_samples)
    rng = Random.Xoshiro(seed)
    metric = AdvancedHMC.DiagEuclideanMetric(length(initial_position))
    hamiltonian = AdvancedHMC.Hamiltonian(metric, value_f, value_and_gradient_f)
    initial_step = AdvancedHMC.find_good_stepsize(
        rng, hamiltonian, copy(initial_position))
    integrator = AdvancedHMC.Leapfrog(initial_step)
    kernel = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
        integrator, AdvancedHMC.GeneralisedNoUTurn(; max_depth = MAX_TREE_DEPTH)))
    adaptor = AdvancedHMC.StanHMCAdaptor(
        AdvancedHMC.MassMatrixAdaptor(metric),
        AdvancedHMC.StepSizeAdaptor(TARGET_ACCEPT, integrator))
    samples, stats = AdvancedHMC.sample(
        rng, hamiltonian, kernel, copy(initial_position),
        num_warmup + num_samples, adaptor, num_warmup;
        drop_warmup = true, progress = false, verbose = false)
    (; samples = permutedims(reduce(hcat, samples)),
       divergences = count(stat -> stat.numerical_error, stats))
end
# DOCS-BASELINE-END: advancedhmc

# DOCS-BASELINE-BEGIN: turing-sampling
# Turing's public NUTS interface on the source-attested Turing twin of the same
# model, with the suite's established Enzyme reverse-mode configuration.
function turing_nuts(model, initial_params; seed, num_warmup, num_samples)
    rng = Random.Xoshiro(seed)
    sampler = Turing.NUTS(num_warmup, TARGET_ACCEPT;
        max_depth = MAX_TREE_DEPTH, adtype = TURING_AD_BACKEND)
    Turing.sample(rng, model, sampler, num_samples;
        initial_params, progress = false)
end
# DOCS-BASELINE-END: turing-sampling

# ---- source-attested Turing model twins ---------------------------------------

# DOCS-BASELINE-BEGIN: turing-eight-schools
Turing.@model function turing_eight_schools(observations, observation_scales)
    μ ~ Normal(0, 5)
    τ ~ truncated(Cauchy(0, 5); lower = 0)
    θ ~ filldist(Normal(μ, τ), length(observations))
    for j in eachindex(observations)
        observations[j] ~ Normal(θ[j], observation_scales[j])
    end
    return nothing
end
# DOCS-BASELINE-END: turing-eight-schools

# DOCS-BASELINE-BEGIN: turing-mnist
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
# DOCS-BASELINE-END: turing-mnist

# ---- shared helpers -----------------------------------------------------------

const LDP = DynamicPPL.LogDensityProblems

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

_warmup() = parse(Int, get(ENV, "RK_PROBPROG_MCMC_WARMUP", string(DEFAULT_WARMUP)))
_samples() = parse(Int, get(ENV, "RK_PROBPROG_MCMC_SAMPLES", string(DEFAULT_SAMPLES)))
_seed() = parse(Int, get(ENV, "RK_PROBPROG_MCMC_SEED", string(DEFAULT_SEED)))
# Hard wall-clock budget for one model's ProbProg compile+sampling attempt.
# Exceeding it is a recorded failure, never a longer wait: the pathological
# probprog tracing mode grinds a core for tens of minutes without a verdict,
# and the tracing pass has no interruption points, so the attempt runs in a
# subprocess the parent can kill (user ruling on decision
# 2026-09-02T01-44-18-996-0r85kze: a 40-minute compile is a failure).
_probprog_budget_seconds() =
    parse(Int, get(ENV, "RK_PROBPROG_MCMC_BUDGET_SECONDS", "600"))

function _turing_packed_density(model)
    DynamicPPL.LogDensityFunction(
        model, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll();
        fix_transforms = true)
end

# Turing returns a FlexiChains chain: scalar parameters read back as Float64
# columns, while vector/matrix parameters read back as per-draw arrays whose
# column-major `vec` matches the packed coefficient order.
function _chain_flat(chain, name)
    values = vec(Array(chain[name]))
    first(values) isa AbstractArray ?
        permutedims(reduce(hcat, (vec(Float64.(value)) for value in values))) :
        reshape(Float64.(values), :, 1)
end

_chain_divergences(chain) = Int(sum(Array(chain[:numerical_error])))

function _ess_summary(packed_draws)
    ess = MCMCDiagnosticTools.ess(reshape(
        packed_draws, size(packed_draws, 1), 1, size(packed_draws, 2)))
    (; min_ess = Float64(minimum(ess)), median_ess = Float64(median(ess)))
end

function _density_at_draws(kernel, packed_draws)
    [Float64(kernel(collect(view(packed_draws, index, :))))
     for index in axes(packed_draws, 1)]
end

# The deterministic pre-sampling gate: the RK density and the linked Turing
# density must agree at exact points, or no sampler comparison is meaningful.
function _gate_density_parity(rk_kernel, turing_ldf, points; label)
    for point in points
        rk_value = Float64(rk_kernel(point))
        turing_value = Float64(LDP.logdensity(turing_ldf, point))
        isfinite(rk_value) || error("$label RK density is not finite at a gate point")
        isapprox(rk_value, turing_value; rtol = 1e-8, atol = 1e-8) || error(
            "$label RK/Turing packed density parity failed: " *
            "$rk_value vs $turing_value")
    end
end

function _moment_gate(model_label, packed_means, tolerance, num_samples)
    # The per-model tolerance is calibrated for the publication protocol's
    # 1000 retained draws; Monte Carlo error grows like 1/√draws, so shorter
    # smoke runs widen the gate proportionally instead of flaking.
    tolerance *= sqrt(max(1.0, DEFAULT_SAMPLES / num_samples))
    harnesses = sort!(collect(keys(packed_means)))
    for i in eachindex(harnesses), j in (i + 1):length(harnesses)
        difference = maximum(abs.(
            packed_means[harnesses[i]] .- packed_means[harnesses[j]]))
        difference <= tolerance || error(
            "$model_label packed posterior means diverge between " *
            "$(harnesses[i]) and $(harnesses[j]): max |Δ| = $difference " *
            "(tolerance $tolerance)")
    end
end

function _ess_and_moments(packed_draws, constrained_summary)
    ess = _ess_summary(packed_draws)
    entries = Pair{String,Any}[
        "min_ess" => ess.min_ess,
        "median_ess" => ess.median_ess,
    ]
    for (name, value) in Base.pairs(constrained_summary(packed_draws))
        push!(entries, "posterior_" * String(name) => Float64(value))
    end
    entries
end

_case_builders() = (
    ("eight_schools", _eight_schools_case),
    ("mnist_logistic_wren_pca40", _mnist_case))

# Parent side of the budgeted ProbProg attempt: spawn this same file in the
# probprog-probe role, wait out the wall-clock budget, and classify.
function _probprog_row_via_subprocess(case; seed, num_warmup, num_samples)
    budget = _probprog_budget_seconds()
    probe_out = tempname()
    probe_log = tempname()
    command = addenv(
        `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $(abspath(@__FILE__))`,
        "RK_PROBPROG_MCMC_ROLE" => "probprog-probe",
        "RK_PROBPROG_MCMC_MODELS" => case.model_label,
        "RK_PROBPROG_MCMC_PROBE_OUT" => probe_out,
        "RK_PROBPROG_MCMC_WARMUP" => string(num_warmup),
        "RK_PROBPROG_MCMC_SAMPLES" => string(num_samples),
        "RK_PROBPROG_MCMC_SEED" => string(seed))
    process = run(pipeline(command; stdout = probe_log, stderr = probe_log);
                  wait = false)
    verdict = timedwait(() -> process_exited(process), Float64(budget);
                        pollint = 1.0)
    if verdict === :timed_out
        # The tracing pass has no interruption points; SIGKILL is the only
        # reliable stop for a non-yielding compile.
        kill(process, Base.SIGKILL)
        wait(process)
        return (Dict{String,Any}(
            "model" => case.model_label, "harness" => "probprog_nuts",
            "state" => "unsupported_runtime",
            "reason" => "ProbProg compile/sampling did not complete within " *
                "the $(budget)s wall-clock budget"), nothing)
    end
    wait(process)
    if isfile(probe_out)
        probe = TOML.parsefile(probe_out)
        haskey(probe, "gate_failure") && error(String(probe["gate_failure"]))
        row = Dict{String,Any}(probe["row"])
        packed_mean = haskey(probe, "packed_mean") ?
            Float64.(probe["packed_mean"]) : nothing
        return (row, packed_mean)
    end
    log_text = isfile(probe_log) ? read(probe_log, String) : ""
    error_line = ""
    for line in eachsplit(log_text, '\n')
        if startswith(line, "ERROR")
            error_line = String(line)
            break
        end
    end
    isempty(error_line) && (error_line = "ProbProg probe exited with status " *
        "$(process.exitcode) and produced no result or ERROR line")
    length(error_line) > 800 && (error_line = first(error_line, 797) * "...")
    (Dict{String,Any}(
        "model" => case.model_label, "harness" => "probprog_nuts",
        "state" => "unsupported_runtime", "reason" => error_line), nothing)
end

# Child role: run one model's ProbProg attempt and its verification gates,
# writing either the finished measured row (plus the packed mean the parent's
# moment gate needs), an unsupported row for a compile/execute failure, or a
# `gate_failure` the parent rethrows fatally.
function _probprog_probe()
    out = ENV["RK_PROBPROG_MCMC_PROBE_OUT"]
    label = only(split(get(ENV, "RK_PROBPROG_MCMC_MODELS", ""), ","))
    builder = only(b for (l, b) in _case_builders() if l == label)
    case = builder()
    seed = _seed()
    num_warmup = _warmup()
    num_samples = _samples()
    dimension = length(case.initial_position)
    probprog = try
        probprog_nuts(case.rk_logdensity, case.initial_position;
            seed, num_warmup, num_samples)
    catch err
        line = first(split(sprint(showerror, err), '\n'))
        length(line) > 800 && (line = first(line, 797) * "...")
        open(out, "w") do io
            TOML.print(io, Dict("row" => Dict(
                "model" => case.model_label, "harness" => "probprog_nuts",
                "state" => "unsupported_runtime", "reason" => line)))
        end
        return nothing
    end
    try
        size(probprog.samples) == (num_samples, dimension) || error(
            "$(case.model_label) ProbProg returned unexpected sample shape")
        probprog_native = _density_at_draws(case.rk_kernel, probprog.samples)
        all(isfinite, probprog_native) || error(
            "$(case.model_label) ProbProg draws leave the finite-density region")
        density_gap = maximum(abs.(probprog.log_densities .- probprog_native))
        density_gap <= 1e-8 || error(
            "$(case.model_label) ProbProg per-draw log densities diverge from " *
            "the native RK kernel: max |Δ| = $density_gap")
        row = Dict{String,Any}(
            "model" => case.model_label, "harness" => "probprog_nuts",
            "state" => "measured",
            "compile_or_warm_start_seconds" => probprog.compile_seconds,
            "sampling_seconds" => probprog.sampling_seconds,
            "divergences" => Int(sum(@view probprog.diagnostics[:, 2])),
            "density_parity_max_abs" => density_gap,
            _ess_and_moments(probprog.samples, case.constrained_summary)...)
        open(out, "w") do io
            TOML.print(io, Dict(
                "row" => row,
                "packed_mean" => vec(mean(probprog.samples; dims = 1))))
        end
    catch err
        open(out, "w") do io
            TOML.print(io, Dict("gate_failure" => sprint(showerror, err)))
        end
        rethrow()
    end
    nothing
end

# ---- per-model orchestration --------------------------------------------------

function _run_model(case; seed, num_warmup, num_samples)
    dimension = length(case.initial_position)
    turing_ldf = _turing_packed_density(case.turing_model)

    # Deterministic gates before any sampling statistic is trusted.
    gate_rng = Random.Xoshiro(seed + 17)
    gate_points = [copy(case.initial_position),
                   case.initial_position .+ 0.1 .* randn(gate_rng, dimension),
                   case.initial_position .+ 0.1 .* randn(gate_rng, dimension)]
    _gate_density_parity(case.rk_kernel, turing_ldf, gate_points;
        label = case.model_label)

    rows = Dict{String,Any}[]
    packed_means = Dict{String,Vector{Float64}}()

    # ProbProg compiled NUTS, in a budgeted subprocess (see
    # `_probprog_budget_seconds`). Only a compile/execute failure or a blown
    # budget degrades to an explicit unsupported cell; every verification gate
    # a successful execution runs stays fatal here in the parent — a parity or
    # shape violation must never be recorded as "unsupported".
    probprog_row, probprog_mean = _probprog_row_via_subprocess(case;
        seed, num_warmup, num_samples)
    push!(rows, probprog_row)
    probprog_mean === nothing ||
        (packed_means["probprog_nuts"] = probprog_mean)

    # AdvancedHMC over the RK density + prepared gradient.
    warm_start_seconds = @elapsed advancedhmc_nuts(
        case.rk_logdensity, case.rk_value_and_gradient, case.initial_position;
        seed = seed + 1, num_warmup = 5, num_samples = 5)
    sampling_seconds = @elapsed ahmc = advancedhmc_nuts(
        case.rk_logdensity, case.rk_value_and_gradient, case.initial_position;
        seed, num_warmup, num_samples)
    size(ahmc.samples) == (num_samples, dimension) || error(
        "$(case.model_label) AdvancedHMC returned unexpected sample shape")
    all(isfinite, _density_at_draws(case.rk_kernel, ahmc.samples)) || error(
        "$(case.model_label) AdvancedHMC draws leave the finite-density region")
    packed_means["advancedhmc_nuts"] = vec(mean(ahmc.samples; dims = 1))
    push!(rows, Dict{String,Any}(
        "model" => case.model_label, "harness" => "advancedhmc_nuts",
        "state" => "measured",
        "compile_or_warm_start_seconds" => warm_start_seconds,
        "sampling_seconds" => sampling_seconds,
        "divergences" => ahmc.divergences,
        _ess_and_moments(ahmc.samples, case.constrained_summary)...))

    # Turing NUTS on the attested twin model.
    turing_warm_seconds = @elapsed turing_nuts(
        case.turing_model, case.turing_initial_params;
        seed = seed + 1, num_warmup = 5, num_samples = 5)
    turing_seconds = @elapsed chain = turing_nuts(
        case.turing_model, case.turing_initial_params;
        seed, num_warmup, num_samples)
    turing_draws = case.packed_from_chain(chain)
    size(turing_draws) == (num_samples, dimension) || error(
        "$(case.model_label) Turing returned unexpected sample shape " *
        "$(size(turing_draws))")
    all(isfinite, _density_at_draws(case.rk_kernel, turing_draws)) || error(
        "$(case.model_label) Turing draws leave the finite-density region of " *
        "the RK density")
    packed_means["turing_nuts"] = vec(mean(turing_draws; dims = 1))
    push!(rows, Dict{String,Any}(
        "model" => case.model_label, "harness" => "turing_nuts",
        "state" => "measured",
        "compile_or_warm_start_seconds" => turing_warm_seconds,
        "sampling_seconds" => turing_seconds,
        "divergences" => _chain_divergences(chain),
        _ess_and_moments(turing_draws, case.constrained_summary)...))

    _moment_gate(case.model_label, packed_means, case.moment_tolerance,
        num_samples)
    rows
end

# ---- the two model cases -------------------------------------------------------

function _eight_schools_case()
    observations = Float64.(EIGHT_SCHOOLS_Y)
    observation_scales = Float64.(EIGHT_SCHOOLS_SIGMA)
    model = build_eight_schools_graph()
    bound = (; observations, observation_scales)
    have = (:unconstrained, :observations, :observation_scales)
    rk_kernel = prepare(model; have, want = :posterior, bound)
    initial_position = zeros(2 + length(observations))
    prepared_ad = prepare_ad(rk_kernel, RK_BOUND_AD_BACKEND, initial_position;
        active = :unconstrained)
    (; model_label = "eight_schools",
       rk_kernel,
       # ProbProg's traced position is shaped (1, D); the RK packed kernels
       # take the flat vector, so the wrapper reshapes before delegating.
       rk_logdensity = q -> rk_kernel(reshape(q, :)),
       rk_value_and_gradient = q -> ad_value_and_gradient(prepared_ad, q),
       turing_model = turing_eight_schools(observations, observation_scales),
       # Constrained-space twin of the packed zero start: τ = exp(0) = 1.
       turing_initial_params = DynamicPPL.InitFromParams((;
           μ = 0.0, τ = 1.0, θ = zeros(length(observations)))),
       # Turing chains report constrained τ; the packed space carries log τ.
       packed_from_chain = chain -> hcat(
           _chain_flat(chain, :μ),
           log.(_chain_flat(chain, :τ)),
           _chain_flat(chain, :θ)),
       initial_position,
       constrained_summary = packed_draws -> (;
           mean_mu = mean(@view packed_draws[:, 1]),
           sd_mu = std(@view packed_draws[:, 1]),
           mean_tau = mean(exp.(@view packed_draws[:, 2]))),
       moment_tolerance = 1.0,
       source_sha256 = bytes2hex(sha256(EIGHT_SCHOOLS_SOURCE)),
       dataset_metadata = Dict{String,Any}(
           "data" => "canonical Eight Schools effects and standard errors",
           "num_observations" => length(observations)))
end

function _mnist_case()
    X, y, dataset_metadata = _load_mnist_dataset(
        MNIST_DATASET_WREN_PCA40, MNIST_WREN_OBSERVATIONS)
    features = size(X, 2)
    nonreference = NUM_CLASSES - 1
    model = build_mnist_logistic_graph()
    bound = (; X, y, num_classes = NUM_CLASSES)
    have = (:unconstrained, :X, :y, :num_classes)
    rk_kernel = prepare(model; have, want = :density, bound)
    initial_position = zeros(nonreference * features + nonreference)
    prepared_ad = prepare_ad(rk_kernel, RK_BOUND_AD_BACKEND, initial_position;
        active = :unconstrained)
    (; model_label = "mnist_logistic_wren_pca40",
       rk_kernel,
       # ProbProg's traced position is shaped (1, D); the RK packed kernels
       # take the flat vector, so the wrapper reshapes before delegating.
       rk_logdensity = q -> rk_kernel(reshape(q, :)),
       rk_value_and_gradient = q -> ad_value_and_gradient(prepared_ad, q),
       turing_model = turing_mnist_logistic(X, y, NUM_CLASSES),
       # No transformed parameter: the constrained and packed spaces coincide,
       # and the all-zero coefficient point keeps every logit at zero.
       turing_initial_params = DynamicPPL.InitFromParams((;
           W = zeros(nonreference, features), b = zeros(nonreference))),
       packed_from_chain = chain -> hcat(
           _chain_flat(chain, :W), _chain_flat(chain, :b)),
       initial_position,
       constrained_summary = packed_draws -> (;
           mean_first_coefficient = mean(@view packed_draws[:, 1]),
           sd_first_coefficient = std(@view packed_draws[:, 1])),
       moment_tolerance = 0.5,
       source_sha256 = bytes2hex(sha256(MNIST_LOGISTIC_SOURCE)),
       dataset_metadata)
end

# ---- receipt ------------------------------------------------------------------

function run_comparison()
    seed = _seed()
    num_warmup = _warmup()
    num_samples = _samples()
    candidate_sha = get(ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown")

    # Publication receipts cover both models; the filter exists for smoke and
    # diagnosis runs only (a filtered receipt fails the validator's inventory).
    selected_models = split(get(
        ENV, "RK_PROBPROG_MCMC_MODELS",
        "eight_schools,mnist_logistic_wren_pca40"), ",")
    measurements = Dict{String,Any}[]
    dataset_records = Dict{String,Any}()
    source_records = Dict{String,Any}()
    for (label, case_builder) in _case_builders()
        label in selected_models || continue
        case = case_builder()
        println("model=$(case.model_label) starting")
        rows = _run_model(case; seed, num_warmup, num_samples)
        append!(measurements, rows)
        dataset_records[case.model_label] = case.dataset_metadata
        source_records[case.model_label] = case.source_sha256
        for row in rows
            if get(row, "state", "") == "measured"
                println("model=$(case.model_label) harness=$(row["harness"]) " *
                        "sampling_seconds=$(row["sampling_seconds"]) " *
                        "divergences=$(row["divergences"]) " *
                        "min_ess=$(row["min_ess"])")
            else
                println("model=$(case.model_label) harness=$(row["harness"]) " *
                        "state=$(row["state"]) reason=$(row["reason"])")
            end
        end
    end

    receipt = Dict{String,Any}(
        "schema" => "probprog-mcmc-v1",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => candidate_sha,
            "reactivekernels_dirty" => false,
            "reactivekernels_version" => _package_version("ReactiveKernels"),
            "reactivekernelspplexamples_version" =>
                _package_version("ReactiveKernelsPPLExamples"),
            "reactant_version" => _package_version("Reactant"),
            "advancedhmc_version" => _package_version("AdvancedHMC"),
            "turing_version" => _package_version("Turing"),
            "dynamicppl_version" => _package_version("DynamicPPL"),
            "mcmcdiagnostictools_version" =>
                _package_version("MCMCDiagnosticTools"),
            "enzyme_version" => _package_version("Enzyme"),
            "julia_version" => string(VERSION),
            "source_text_sha256" => source_records,
        ),
        "environment" => Dict(
            "os" => string(Sys.KERNEL),
            "arch" => string(Sys.ARCH),
            "cpu" => first(Sys.cpu_info()).model,
            "julia_threads" => Threads.nthreads(),
            "reactant_backend" => "default CPU",
        ),
        "protocol" => Dict(
            "models" => ["eight_schools", "mnist_logistic_wren_pca40"],
            "harnesses" => ["probprog_nuts", "advancedhmc_nuts", "turing_nuts"],
            "num_warmup" => num_warmup,
            "num_samples" => num_samples,
            "seed" => seed,
            "probprog_budget_seconds" => _probprog_budget_seconds(),
            "max_tree_depth" => MAX_TREE_DEPTH,
            "target_accept" => "0.8 for AdvancedHMC and Turing; ProbProg " *
                "uses its library-default dual-averaging target",
            "metric" => "diagonal (adapted) for all three harnesses",
            "divergence_threshold" =>
                "energy error 1000 for all three harnesses",
            "matched_initial_position" => true,
            "timing" => "one wall-clock run per harness covering warmup plus " *
                "retained draws; compilation/JIT warm-start is a separate " *
                "recorded column",
            "density_parity_gate" => "RK vs linked Turing joint at " *
                "deterministic points (rtol 1e-8); ProbProg per-draw log " *
                "densities vs the native RK kernel (atol 1e-8)",
            "moment_gate" => "pairwise max |Δ| of packed posterior means " *
                "within a coarse per-model tolerance; sampler trajectories " *
                "are never compared exactly",
            "ess" => "MCMCDiagnosticTools.ess (bulk) on the packed draws, " *
                "uniformly across harnesses",
        ),
        "datasets" => dataset_records,
        "measurements" => measurements,
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

if get(ENV, "RK_PROBPROG_MCMC_ROLE", "") == "probprog-probe"
    _probprog_probe()
elseif get(ENV, "RK_PROBPROG_MCMC_DEFINITIONS_ONLY", "") != "1"
    run_comparison()
end
