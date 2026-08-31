#!/usr/bin/env julia

module DistributionComparison

using BenchmarkTools
using Dates
using Distributions
using Pkg
using Printf
using ProbabilityMeasures
using Random
using Reactant
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using Statistics
using TOML

using Reactant: @compile

const PROBABILITY_MEASURES_SHA = "7cf3a6e112aaae2097b8d401b256d1bce635e03e"
const DEFAULT_SIZES = (1, 1_000, 10_000, 30_000, 100_000, 1_000_000)
const DEFAULT_ROUNDS = 5

@kernel benchmark_normal_logpdf(x::Float64, μ::Float64, σ::Float64) = begin
    logσ::Float64 = log(σ)
    z::Float64 = (x - μ) / σ
    logdensity::Float64 = -0.5 * log(2π) - logσ - 0.5 * z^2
end

const RK_DIRECT_REDUCE = plate(
    benchmark_normal_logpdf;
    have = (:x, :μ, :σ), want = :logdensity, batched = (:x,),
)
const RK_REDUCE = plate(
    normal.logpdf;
    have = (:x, :location, :scale), want = :logpdf, batched = (:x,),
)
const RK_SCALAR = prepare(extract(
    normal; have = (:x, :location, :scale), want = :logpdf,
))
const RK_REPLICATED = replica(RK_SCALAR; batched = :x)

distributions_reduce(μ, σ, xs) =
    sum(Distributions.logpdf.(Distributions.Normal(μ, σ), xs))

probability_measures_reduce(μ, σ, xs) =
    sum(ProbabilityMeasures.logdensityof.(ProbabilityMeasures.Normal(μ, σ), xs))

function distributions_loop(μ, σ, xs)
    d = Distributions.Normal(μ, σ)
    mapreduce(x -> Distributions.logpdf(d, x), +, xs; init = 0.0)
end

function probability_measures_loop(μ, σ, xs)
    d = ProbabilityMeasures.Normal(μ, σ)
    mapreduce(x -> ProbabilityMeasures.logdensityof(d, x), +, xs; init = 0.0)
end

function hand_hoisted_reduce(μ, σ, xs)
    constant = -0.5 * log(2π) - log(σ)
    acc = 0.0
    @inbounds for x in xs
        z = (x - μ) / σ
        acc += constant - 0.5 * z^2
    end
    acc
end

compile_rk(xs, μ, σ) = @compile sync = true RK_REDUCE(xs, μ, σ)
compile_rk_replicated(xs, μ, σ) =
    @compile sync = true RK_REPLICATED(xs, μ, σ)
compile_pm(μ, σ, xs) = @compile sync = true probability_measures_reduce(μ, σ, xs)
compile_distributions(μ, σ, xs) =
    @compile sync = true distributions_reduce(μ, σ, xs)

function _measurement(f, a, b, c; rounds::Int)
    benchmark = @benchmarkable $f($a, $b, $c)
    times_ns = Float64[]
    bytes = Int[]
    allocs = Int[]
    for _ in 1:rounds
        estimate = minimum(run(benchmark; samples = 200, seconds = 0.25))
        push!(times_ns, estimate.time)
        push!(bytes, estimate.memory)
        push!(allocs, estimate.allocs)
    end
    Dict(
        "times_ns" => times_ns,
        "median_ns" => median(times_ns),
        "bytes" => bytes,
        "median_bytes" => Int(median(bytes)),
        "allocs" => allocs,
        "median_allocs" => Int(median(allocs)),
    )
end

_git(repo, args...) = readchomp(Cmd(["git", "-C", repo, string.(args)...]))

function _package_version(name)
    for info in values(Pkg.dependencies())
        info.name == name && return string(info.version)
    end
    error("package $name is absent from the benchmark environment")
end

function _sizes()
    raw = get(ENV, "RK_DENSITY_BENCH_SIZES", "")
    isempty(raw) && return DEFAULT_SIZES
    Tuple(parse(Int, strip(item)) for item in split(raw, ','))
end

function _replica_counts()
    raw = get(ENV, "RK_DENSITY_REPLICAS", "")
    isempty(raw) && return (1, 16, 256)
    Tuple(parse(Int, strip(item)) for item in split(raw, ','))
end

function _output_path()
    for arg in ARGS
        startswith(arg, "--output=") && return split(arg, '='; limit = 2)[2]
    end
    nothing
end

function _reactant_amortization(μ, σ, rμ, rσ, rounds)
    rows = Dict{String,Any}[]
    for replicas in _replica_counts()
        replicas > 0 || throw(ArgumentError("RK_DENSITY_REPLICAS must be positive"))
        xs = [0.17 + 0.001 * (j - 1) for j in 1:replicas]
        references = [RK_SCALAR(x, μ, σ) for x in xs]
        native = RK_REPLICATED(xs, μ, σ)
        isapprox(native, references; rtol = 1e-11) ||
            error("native replicated value mismatch at replicas=$replicas")

        rxs = Reactant.to_rarray(xs)
        compile_s = @elapsed compiled = compile_rk_replicated(rxs, rμ, rσ)
        observed = Array(compiled(rxs, rμ, rσ))
        isapprox(observed, references; rtol = 1e-11) ||
            error("Reactant replicated value mismatch at replicas=$replicas")
        measurement = _measurement(compiled, rxs, rμ, rσ; rounds)
        push!(rows, Dict{String,Any}(
            "n" => 1,
            "replicas" => replicas,
            "max_abs_error" => maximum(abs.(observed .- references)),
            "rk_reactant_replicated" => measurement,
            "median_ns_per_evaluation" => measurement["median_ns"] / replicas,
            "compile_seconds" => compile_s,
        ))
    end
    rows
end

function run_benchmark()
    repo = normpath(joinpath(@__DIR__, ".."))
    dirty = !isempty(_git(repo, "status", "--porcelain"))
    allow_dirty = "--allow-dirty" in ARGS
    dirty && !allow_dirty && error(
        "refusing to write a receipt from a dirty tree; commit first or pass --allow-dirty for an exploratory run",
    )

    Random.seed!(0x5eed)
    μ, σ = 0.3, 1.2
    rμ = Reactant.ConcreteRNumber(μ)
    rσ = Reactant.ConcreteRNumber(σ)
    rounds = parse(Int, get(ENV, "RK_DENSITY_BENCH_ROUNDS", string(DEFAULT_ROUNDS)))

    rows = Dict{String,Any}[]
    distributions_reactant_supported = true
    distributions_reactant_error = ""

    for n in _sizes()
        xs = randn(n)
        rxs = Reactant.to_rarray(xs)
        reference = hand_hoisted_reduce(μ, σ, xs)

        native_values = (
            RK_REDUCE(xs, μ, σ),
            RK_DIRECT_REDUCE(xs, μ, σ),
            distributions_reduce(μ, σ, xs),
            probability_measures_reduce(μ, σ, xs),
            distributions_loop(μ, σ, xs),
            probability_measures_loop(μ, σ, xs),
        )
        all(value -> isapprox(value, reference; rtol = 1e-11), native_values) ||
            error("native value mismatch at N=$n")

        rk_compile_s = @elapsed rk_compiled = compile_rk(rxs, rμ, rσ)
        pm_compile_s = @elapsed pm_compiled = compile_pm(rμ, rσ, rxs)
        rk_value = Float64(rk_compiled(rxs, rμ, rσ))
        pm_value = Float64(pm_compiled(rμ, rσ, rxs))
        isapprox(rk_value, reference; rtol = 1e-11) ||
            error("Reactant RK value mismatch at N=$n")
        isapprox(pm_value, reference; rtol = 1e-11) ||
            error("Reactant ProbabilityMeasures value mismatch at N=$n")

        dist_compiled = nothing
        dist_compile_s = nothing
        dist_value = nothing
        if distributions_reactant_supported
            try
                dist_compile_s = @elapsed dist_compiled =
                    compile_distributions(rμ, rσ, rxs)
                dist_value = Float64(dist_compiled(rμ, rσ, rxs))
                isapprox(dist_value, reference; rtol = 1e-11) ||
                    error("Reactant Distributions value mismatch at N=$n")
            catch err
                distributions_reactant_supported = false
                distributions_reactant_error = first(split(sprint(showerror, err), '\n'))
            end
        end

        observed_values = Any[native_values..., rk_value, pm_value]
        dist_value === nothing || push!(observed_values, dist_value)

        row = Dict{String,Any}(
            "n" => n,
            "max_abs_error" =>
                maximum(abs(value - reference) for value in observed_values),
            "max_relative_error" => maximum(
                abs(value - reference) / abs(reference) for value in observed_values
            ),
            "rk_native" => _measurement(RK_REDUCE, xs, μ, σ; rounds),
            "rk_direct_native" =>
                _measurement(RK_DIRECT_REDUCE, xs, μ, σ; rounds),
            "distributions_native" =>
                _measurement(distributions_reduce, μ, σ, xs; rounds),
            "probability_measures_native" =>
                _measurement(probability_measures_reduce, μ, σ, xs; rounds),
            "distributions_loop" =>
                _measurement(distributions_loop, μ, σ, xs; rounds),
            "probability_measures_loop" =>
                _measurement(probability_measures_loop, μ, σ, xs; rounds),
            "hand_hoisted" =>
                _measurement(hand_hoisted_reduce, μ, σ, xs; rounds),
            "rk_reactant" => _measurement(rk_compiled, rxs, rμ, rσ; rounds),
            "probability_measures_reactant" =>
                _measurement(pm_compiled, rμ, rσ, rxs; rounds),
            "rk_reactant_compile_seconds" => rk_compile_s,
            "probability_measures_reactant_compile_seconds" => pm_compile_s,
        )
        if dist_compiled !== nothing
            row["distributions_reactant"] =
                _measurement(dist_compiled, rμ, rσ, rxs; rounds)
            row["distributions_reactant_compile_seconds"] = dist_compile_s
        end
        push!(rows, row)
        @printf("N=%d complete\n", n)
    end

    amortization = _reactant_amortization(μ, σ, rμ, rσ, rounds)

    receipt = Dict{String,Any}(
        "schema" => "distribution-logdensity-v1",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => _git(repo, "rev-parse", "HEAD"),
            "reactivekernels_dirty" => dirty,
            "probability_measures_sha" => PROBABILITY_MEASURES_SHA,
            "probability_measures_version" => string(Base.pkgversion(ProbabilityMeasures)),
            "distributions_version" => string(Base.pkgversion(Distributions)),
            "reactant_version" => string(Base.pkgversion(Reactant)),
            "reactant_jll_version" => _package_version("Reactant_jll"),
            "julia_version" => string(VERSION),
        ),
        "environment" => Dict(
            "os" => string(Sys.KERNEL),
            "arch" => string(Sys.ARCH),
            "cpu" => first(Sys.cpu_info()).model,
            "julia_threads" => Threads.nthreads(),
            "reactant_backend" => "default CPU",
        ),
        "protocol" => Dict(
            "formula" => "Normal log density with shared μ and σ",
            "element_type" => "Float64",
            "rounds" => rounds,
            "estimator" => "median of per-round BenchmarkTools minimum times",
            "samples_per_round" => 200,
            "seconds_per_round" => 0.25,
            "reactant_sync" => true,
            "reactant_transfers_included" => false,
            "reactant_parameters_traced" => true,
            "reactant_compile_time_in_timed_region" => false,
            "reactant_compile_order" => ["ReactiveKernels", "ProbabilityMeasures", "Distributions"],
            "reactant_compile_times_include_first_service_startup" => true,
            "reactant_replica_counts" => collect(_replica_counts()),
            "reactant_replica_axis" =>
                "independent one-observation data sets on the trailing x axis",
            "reactant_replica_normalization" =>
                "whole-batch median divided by replica count",
            "native_distributions_spelling" =>
                "sum(Distributions.logpdf.(Distributions.Normal(μ, σ), xs))",
            "native_probability_measures_spelling" =>
                "sum(ProbabilityMeasures.logdensityof.(ProbabilityMeasures.Normal(μ, σ), xs))",
            "rk_shared_spelling" =>
                "plate(normal.logpdf; have=(:x,:location,:scale), want=:logpdf, batched=(:x,))",
            "rk_direct_control_spelling" =>
                "one-off benchmark_normal_logpdf lifted with plate",
        ),
        "support" => Dict(
            "rk_reactant" => true,
            "probability_measures_reactant" => true,
            "distributions_reactant" => distributions_reactant_supported,
            "distributions_reactant_error" => distributions_reactant_error,
        ),
        "measurements" => rows,
        "reactant_amortization" => amortization,
    )

    output = _output_path()
    if output === nothing
        TOML.print(stdout, receipt; sorted = true)
    else
        mkpath(dirname(abspath(output)))
        open(output, "w") do io
            TOML.print(io, receipt; sorted = true)
        end
        println("receipt=", abspath(output))
    end
    receipt
end

end # module DistributionComparison

if abspath(PROGRAM_FILE) == @__FILE__
    using .DistributionComparison
    DistributionComparison.run_benchmark()
end
