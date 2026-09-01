# Inner body of the ReactiveKernels-vs-Turing evaluation-throughput comparison.
# Runs inside the pinned environment provisioned by eval_throughput_comparison.jl.
#
# Model: iid Normal log density of a position vector x given fixed (mu, log-sigma).
# Three evaluation modes are timed, each WITH and WITHOUT Reactant:
#   primal   — the scalar log density
#   gradient — d(log density)/dx, reverse-mode Enzyme via DifferentiationInterface
#   gq       — a generated-quantity vector (the pointwise log densities)
#
# Honesty invariants:
#   * Every implementation computes the SAME quantity, parity-checked against the
#     closed form before any timing.
#   * Reactant device transfers and compilation happen OUTSIDE the timed region;
#     only the compiled call is timed (sync = true).
#   * Turing's DynamicPPL evaluations are not Reactant-traceable, so those cells
#     are reported unsupported — never faked.
#   * Timing is a hand-rolled minimum of batched runs (BenchmarkTools pins JSON
#     below the version Reactant needs, so it cannot be used here).
#   * Replicated rows are a separate throughput protocol: one compiled call
#     evaluates independent columns and reports both whole-batch and normalized
#     per-evaluation time. They do not replace the single-call latency rows.

using Dates
using LinearAlgebra
using Pkg
using Printf
using Random
using Statistics
using TOML
using ReactiveKernels
import Reactant
import Reactant: @compile
import Enzyme
import DifferentiationInterface as DI
import Turing
import DynamicPPL
using Distributions: MvNormal

const LDP = DynamicPPL.LogDensityProblems

# Reverse-mode Enzyme via DifferentiationInterface. This is the ONLY AD backend
# used; there is no forward-mode contrast. This comparator deliberately shares
# one backend configuration with DynamicPPL, whose evaluation path still needs
# runtime activity; it is not the configuration required by RK's prepared
# densities (the PPL/distribution tests exercise those with plain Reverse).
const BACKEND_ENZYME = DI.AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const,
)

const COMPARISON_PACKAGES = (
    "ReactiveKernels", "Reactant", "Turing", "DynamicPPL", "Distributions",
    "DifferentiationInterface", "Enzyme",
)

# One ReactiveKernels kernel assigns BOTH the pointwise vector (a generated
# quantity) and its sum (the primal); two prepared views select each WANT.
const NORMAL_MODEL = @kernel normal_model(
        x::Vector{Float64}, μ::Float64, logσ::Float64) = begin
    σ::Float64 = exp(logσ)
    pointwise::Vector{Float64} =
        -0.5 .* ((x .- μ) ./ σ) .^ 2 .- (logσ + 0.5 * log(2π))
    logdensity::Float64 = sum(pointwise)
end

_rk_primal() = prepare(normal_model; have = (:x, :μ, :logσ), want = :logdensity)
_rk_gq() = prepare(normal_model; have = (:x, :μ, :logσ), want = :pointwise)

# DOCS-BASELINE-BEGIN: turing
# The same model in Turing; its `return` is the generated-quantity vector so
# DynamicPPL.generated_quantities measures the comparable computation.
Turing.@model function turing_model(μ, logσ, n)
    x ~ MvNormal(fill(μ, n), exp(logσ)^2 * I)
    return -0.5 .* ((x .- μ) ./ exp(logσ)) .^ 2 .- (logσ + 0.5 * log(2π))
end
# DOCS-BASELINE-END: turing

# Reactant differentiates x (active); mu and log-sigma are Const. Wrapping the
# Enzyme.gradient call lets @compile return a plain callable.
_rk_reactant_grad(kernel, x, μ, logσ) =
    Enzyme.gradient(Enzyme.Reverse, kernel, x, Enzyme.Const(μ), Enzyme.Const(logσ))

_rk_replicated_sum(kernel, x, μ, logσ) = sum(kernel(x, μ, logσ))
function _rk_reactant_replicated_grad(kernel, x, μ, logσ)
    gradient = Enzyme.gradient(
        Enzyme.Reverse,
        Enzyme.Const(_rk_replicated_sum),
        Enzyme.Const(kernel),
        x,
        Enzyme.Const(μ),
        Enzyme.Const(logσ),
    )
    gradient[2]
end

_reference_pointwise(x, μ, logσ) =
    -0.5 .* ((x .- μ) ./ exp(logσ)) .^ 2 .- (logσ + 0.5 * log(2π))
_reference_primal(x, μ, logσ) = sum(_reference_pointwise(x, μ, logσ))
_reference_gradient(x, μ, logσ) = -(x .- μ) ./ exp(logσ)^2

# Hand-rolled timing: warm once, auto-size an inner batch so each timed region is
# at least ~100 microseconds, then take the minimum per-call time (ns) over
# `rounds` batches. No BenchmarkTools dependency.
function _median_ns(thunk; rounds::Int)
    thunk()
    inner = 1
    while true
        elapsed = @elapsed for _ in 1:inner
            thunk()
        end
        (elapsed > 1.0e-4 || inner >= 1 << 22) && break
        inner *= 4
    end
    best = Inf
    for _ in 1:rounds
        elapsed = @elapsed for _ in 1:inner
            thunk()
        end
        best = min(best, elapsed / inner)
    end
    best * 1.0e9
end

_inputs(n) = (
    μ = 0.3,
    logσ = log(1.3),
    x = [0.7 * sin(0.31i) - 0.2 * cos(0.17i) for i in 1:n],
)

function _sizes()
    raw = get(ENV, "RK_EVAL_SIZES", "")
    isempty(raw) && return (16, 256, 4096)
    Tuple(parse(Int, strip(item)) for item in split(raw, ','))
end

_rounds() = parse(Int, get(ENV, "RK_EVAL_ROUNDS", "50"))
_replicas() = parse(Int, get(ENV, "RK_EVAL_REPLICAS", "256"))

function _measure_size(n, rounds, replicas)
    replicas > 0 || throw(ArgumentError("RK_EVAL_REPLICAS must be positive"))
    (; μ, logσ, x) = _inputs(n)
    reference_primal = _reference_primal(x, μ, logσ)
    reference_gradient = _reference_gradient(x, μ, logσ)
    reference_pointwise = _reference_pointwise(x, μ, logσ)

    rk_primal = _rk_primal()
    rk_gq = _rk_gq()
    rk_primal_replicated = replica(rk_primal; batched = :x)
    rk_gq_replicated = replica(rk_gq; batched = :x)
    rk_primal_of_x(position) = rk_primal(position, μ, logσ)
    enzyme_preparation =
        DI.prepare_gradient(rk_primal_of_x, BACKEND_ENZYME, copy(x))

    model = turing_model(μ, logσ, n)
    turing_primal =
        DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal,
                                      DynamicPPL.LinkAll())
    turing_gradient =
        DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal,
                                      DynamicPPL.LinkAll(); adtype = BACKEND_ENZYME)

    device_x = Reactant.to_rarray(x)
    device_μ = Reactant.to_rarray(μ; track_numbers = true)
    device_logσ = Reactant.to_rarray(logσ; track_numbers = true)
    x_replicated = [x[i] + 0.001 * (j - 1) for i in eachindex(x), j in 1:replicas]
    device_x_replicated = Reactant.to_rarray(x_replicated)
    rk_primal_compiled = @compile sync = true rk_primal(device_x, device_μ, device_logσ)
    rk_gq_compiled = @compile sync = true rk_gq(device_x, device_μ, device_logσ)
    rk_grad_compiled =
        @compile sync = true _rk_reactant_grad(rk_primal, device_x, device_μ, device_logσ)
    rk_primal_replicated_compiled = @compile sync = true rk_primal_replicated(
        device_x_replicated, device_μ, device_logσ)
    rk_gq_replicated_compiled = @compile sync = true rk_gq_replicated(
        device_x_replicated, device_μ, device_logσ)
    rk_grad_replicated_compiled = @compile sync = true _rk_reactant_replicated_grad(
        rk_primal_replicated, device_x_replicated, device_μ, device_logσ)

    # Prove the requested backend is ACTUALLY exercised — correct output alone
    # does not distinguish Enzyme from any other AD backend (the exact trap that
    # invalidated the earlier sampler comparison, where an unused adtype produced
    # a fake axis). The DI preparation must be an Enzyme prep, and the Turing
    # gradient LogDensityFunction must carry the Enzyme adtype in its type.
    @assert occursin("Enzyme", string(typeof(enzyme_preparation)))
    @assert occursin("AutoEnzyme", string(typeof(turing_gradient)))

    # Parity — every value equals the reference before any timing.
    approx(a, b) = isapprox(collect(Float64.(a)), collect(Float64.(b));
                            rtol = 1.0e-8, atol = 1.0e-9)
    @assert approx(rk_primal(x, μ, logσ), reference_primal)
    @assert approx(LDP.logdensity(turing_primal, x), reference_primal)
    @assert approx(rk_primal_compiled(device_x, device_μ, device_logσ), reference_primal)
    @assert approx(DI.gradient(rk_primal_of_x, enzyme_preparation, BACKEND_ENZYME, x),
                   reference_gradient)
    @assert approx(last(LDP.logdensity_and_gradient(turing_gradient, x)),
                   reference_gradient)
    @assert approx(Array(rk_grad_compiled(rk_primal, device_x, device_μ, device_logσ)[1]),
                   reference_gradient)
    @assert approx(rk_gq(x, μ, logσ), reference_pointwise)
    @assert approx(Array(rk_gq_compiled(device_x, device_μ, device_logσ)), reference_pointwise)
    @assert approx(DynamicPPL.generated_quantities(model, (x = x,)), reference_pointwise)

    reference_pointwise_replicated = _reference_pointwise(x_replicated, μ, logσ)
    reference_primal_replicated = vec(sum(reference_pointwise_replicated; dims = 1))
    reference_gradient_replicated = _reference_gradient(x_replicated, μ, logσ)
    @assert approx(rk_primal_replicated(x_replicated, μ, logσ),
                   reference_primal_replicated)
    @assert approx(rk_gq_replicated(x_replicated, μ, logσ),
                   reference_pointwise_replicated)
    @assert approx(Array(rk_primal_replicated_compiled(
                       device_x_replicated, device_μ, device_logσ)),
                   reference_primal_replicated)
    @assert approx(Array(rk_gq_replicated_compiled(
                       device_x_replicated, device_μ, device_logσ)),
                   reference_pointwise_replicated)
    @assert approx(Array(rk_grad_replicated_compiled(
                       rk_primal_replicated, device_x_replicated,
                       device_μ, device_logσ)),
                   reference_gradient_replicated)

    ns(thunk) = _median_ns(thunk; rounds)
    cells = [
        ("primal", "reactivekernels", "native",
         ns(() -> rk_primal(x, μ, logσ))),
        ("primal", "reactivekernels", "reactant",
         ns(() -> rk_primal_compiled(device_x, device_μ, device_logσ))),
        ("primal", "turing", "native",
         ns(() -> LDP.logdensity(turing_primal, x))),
        ("gradient", "reactivekernels", "native",
         ns(() -> DI.gradient(rk_primal_of_x, enzyme_preparation, BACKEND_ENZYME, x))),
        ("gradient", "reactivekernels", "reactant",
         ns(() -> rk_grad_compiled(rk_primal, device_x, device_μ, device_logσ))),
        ("gradient", "turing", "native",
         ns(() -> LDP.logdensity_and_gradient(turing_gradient, x))),
        ("gq", "reactivekernels", "native",
         ns(() -> rk_gq(x, μ, logσ))),
        ("gq", "reactivekernels", "reactant",
         ns(() -> rk_gq_compiled(device_x, device_μ, device_logσ))),
        ("gq", "turing", "native",
         ns(() -> DynamicPPL.generated_quantities(model, (x = x,)))),
    ]
    rows = map(cells) do (mode, implementation, variant, median_ns)
        Dict{String,Any}(
            "size" => n, "mode" => mode, "implementation" => implementation,
            "variant" => variant, "median_ns" => median_ns,
        )
    end
    replicated_cells = [
        ("primal", ns(() -> rk_primal_replicated_compiled(
            device_x_replicated, device_μ, device_logσ))),
        ("gradient", ns(() -> rk_grad_replicated_compiled(
            rk_primal_replicated, device_x_replicated, device_μ, device_logσ))),
        ("gq", ns(() -> rk_gq_replicated_compiled(
            device_x_replicated, device_μ, device_logσ))),
    ]
    append!(rows, map(replicated_cells) do (mode, median_batch_ns)
        Dict{String,Any}(
            "size" => n,
            "mode" => mode,
            "implementation" => "reactivekernels",
            "variant" => "reactant_replicated",
            "batch_size" => replicas,
            "median_batch_ns" => median_batch_ns,
            "median_ns" => median_batch_ns / replicas,
        )
    end)
    rows
end

function _output_path()
    for arg in ARGS
        startswith(arg, "--output=") && return split(arg, '='; limit = 2)[2]
    end
    nothing
end

function _package_version(name)
    for info in values(Pkg.dependencies())
        info.name == name && return string(info.version)
    end
    error("package $name absent from the benchmark environment")
end

function run_comparison()
    Random.seed!(0x51a7)
    rounds = _rounds()
    replicas = _replicas()
    measurements = Dict{String,Any}[]
    for n in _sizes()
        append!(measurements, _measure_size(n, rounds, replicas))
        @printf("size=%d complete\n", n)
    end

    receipt = Dict{String,Any}(
        "schema" => "eval-throughput-v1",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => get(ENV, "REACTIVEKERNELS_CANDIDATE_SHA", "unknown"),
            "julia_version" => string(VERSION),
            (string(lowercase(name), "_version") => _package_version(name)
             for name in COMPARISON_PACKAGES)...,
        ),
        "environment" => Dict(
            "os" => string(Sys.KERNEL), "arch" => string(Sys.ARCH),
            "cpu" => first(Sys.cpu_info()).model,
            "julia_threads" => Threads.nthreads(),
            "reactant_backend" => "default CPU",
        ),
        "protocol" => Dict(
            "model" => "iid Normal log density of a position vector x given fixed (mu, log-sigma)",
            "modes" => ["primal", "gradient", "gq"],
            "gradient_backend" => "reverse-mode Enzyme via DifferentiationInterface",
            "gradient_wrt" => "x",
            "gq_definition" => "pointwise log densities (vector)",
            "element_type" => "Float64",
            "rounds" => rounds,
            "estimator" => "minimum per-call time over rounds of an auto-sized inner batch",
            "reactant_sync" => true,
            "reactant_transfers_included" => false,
            "reactant_compile_time_in_timed_region" => false,
            "replicas" => replicas,
            "replica_axis" => "independent position vectors on the trailing x axis",
            "replicated_gradient" =>
                "gradient of the sum of independent replicated log densities",
            "replicated_median_ns" =>
                "whole-batch median divided by replicas; median_batch_ns is also retained",
        ),
        "support" => Dict(
            "turing_reactant" => false,
            "turing_reactant_reason" =>
                "Turing's DynamicPPL evaluation re-runs the model and is not a " *
                "straight-line array program, so it is not Reactant-traceable.",
        ),
        "measurements" => measurements,
    )

    output = _output_path()
    if output === nothing
        TOML.print(stdout, receipt; sorted = true)
    else
        mkpath(dirname(abspath(output)))
        open(io -> TOML.print(io, receipt; sorted = true), output, "w")
        println("receipt=", abspath(output))
    end
    receipt
end

run_comparison()
