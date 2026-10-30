#!/usr/bin/env julia

# Native Normal log-density reduction benchmark. Writes a
# `distribution-logdensity-v2` receipt; the checked-in v1 receipts also carry
# historical Reactant cells, measured with ReactiveKernels' removed Reactant path.

module DistributionComparison

include("distribution_benchmark_cases.jl")

using BenchmarkTools
using Dates
using Distributions
using Printf
using ProbabilityMeasures
using Random
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using Statistics
using TOML

using .DistributionBenchmarkCases:
    NORMAL_SIZES, normal_observations, normal_parameters

const PROBABILITY_MEASURES_SHA = "7cf3a6e112aaae2097b8d401b256d1bce635e03e"
const DEFAULT_SIZES = NORMAL_SIZES
const DEFAULT_ROUNDS = 5

@kernel benchmark_normal_logpdf(
        x::Float64, location::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)
    standardized::Float64 = (x - location) / scale
    logpdf::Float64 = -0.5 * log(2π) - log_scale - 0.5 * standardized^2
end

const RK_DIRECT_REDUCE = plate(
    benchmark_normal_logpdf;
    have = (:x, :location, :scale), want = :logpdf, batched = (:x,),
)
const RK_REDUCE = plate(
    normal.logpdf;
    have = (:x, :location, :scale), want = :logpdf, batched = (:x,),
)

@kernel benchmark_authored_normal_loglik(
        x, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        normal(li, si).logpdf(xi)
    end
    return sum(pointwise)
end

const RK_AUTHORED_RETURN = prepare(benchmark_authored_normal_loglik)

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

function _sizes()
    raw = get(ENV, "RK_DENSITY_BENCH_SIZES", "")
    isempty(raw) && return DEFAULT_SIZES
    Tuple(parse(Int, strip(item)) for item in split(raw, ','))
end

function _output_path()
    for arg in ARGS
        startswith(arg, "--output=") && return split(arg, '='; limit = 2)[2]
    end
    nothing
end

function run_benchmark()
    repo = normpath(joinpath(@__DIR__, ".."))
    dirty = !isempty(_git(repo, "status", "--porcelain"))
    allow_dirty = "--allow-dirty" in ARGS
    dirty && !allow_dirty && error(
        "refusing to write a receipt from a dirty tree; commit first or pass --allow-dirty for an exploratory run",
    )

    authored_native_ast = string(code_expr(RK_AUTHORED_RETURN))
    occursin("similar", authored_native_ast) &&
        error("authored return-only lowering materializes a pointwise output")

    Random.seed!(0x5eed)
    parameters = normal_parameters()
    μ, σ = parameters.location, parameters.scale
    rounds = parse(Int, get(ENV, "RK_DENSITY_BENCH_ROUNDS", string(DEFAULT_ROUNDS)))

    rows = Dict{String,Any}[]

    for n in _sizes()
        xs = normal_observations(n)
        reference = hand_hoisted_reduce(μ, σ, xs)

        native_values = (
            RK_REDUCE(xs, μ, σ),
            RK_AUTHORED_RETURN(xs, μ, σ),
            RK_DIRECT_REDUCE(xs, μ, σ),
            distributions_reduce(μ, σ, xs),
            probability_measures_reduce(μ, σ, xs),
            distributions_loop(μ, σ, xs),
            probability_measures_loop(μ, σ, xs),
        )
        all(value -> isapprox(value, reference; rtol = 1e-11), native_values) ||
            error("native value mismatch at N=$n")

        row = Dict{String,Any}(
            "n" => n,
            "max_abs_error" =>
                maximum(abs(value - reference) for value in native_values),
            "max_relative_error" => maximum(
                abs(value - reference) / abs(reference) for value in native_values
            ),
            "rk_native" => _measurement(RK_REDUCE, xs, μ, σ; rounds),
            "rk_authored_native" =>
                _measurement(RK_AUTHORED_RETURN, xs, μ, σ; rounds),
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
        )
        push!(rows, row)
        @printf("N=%d complete\n", n)
    end

    receipt = Dict{String,Any}(
        "schema" => "distribution-logdensity-v2",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => _git(repo, "rev-parse", "HEAD"),
            "reactivekernels_dirty" => dirty,
            "probability_measures_sha" => PROBABILITY_MEASURES_SHA,
            "probability_measures_version" => string(Base.pkgversion(ProbabilityMeasures)),
            "distributions_version" => string(Base.pkgversion(Distributions)),
            "julia_version" => string(VERSION),
        ),
        "environment" => Dict(
            "os" => string(Sys.KERNEL),
            "arch" => string(Sys.ARCH),
            "cpu" => first(Sys.cpu_info()).model,
            "julia_threads" => Threads.nthreads(),
        ),
        "protocol" => Dict(
            "formula" => "Normal log density with shared μ and σ",
            "element_type" => "Float64",
            "rounds" => rounds,
            "estimator" => "median of per-round BenchmarkTools minimum times",
            "samples_per_round" => 200,
            "seconds_per_round" => 0.25,
            "native_distributions_spelling" =>
                "sum(Distributions.logpdf.(Distributions.Normal(μ, σ), xs))",
            "native_probability_measures_spelling" =>
                "sum(ProbabilityMeasures.logdensityof.(ProbabilityMeasures.Normal(μ, σ), xs))",
            "rk_shared_spelling" =>
                "plate(normal.logpdf; have=(:x,:location,:scale), want=:logpdf, batched=(:x,))",
            "rk_authored_return_spelling" =>
                "@kernel normal_loglik(x, location, scale) = begin; pointwise = plate(x, location, scale) do xi, li, si; normal(li, si).logpdf(xi); end; return sum(pointwise); end",
            "rk_authored_native_lowering" =>
                "one reduction traversal with no similar/pointwise output",
            "rk_direct_control_spelling" =>
                "one-off benchmark_normal_logpdf lifted with plate",
        ),
        "measurements" => rows,
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
