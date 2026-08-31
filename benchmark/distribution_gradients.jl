#!/usr/bin/env julia

module DistributionGradientBenchmark

include("distribution_benchmark_cases.jl")

using BenchmarkTools
using Dates
using DifferentiationInterface
import Enzyme
using LinearAlgebra
using Pkg
using Printf
using Random
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, cauchy, laplace, bernoulli, lognormal, exponential, geometric,
    uniform, mvnormal
using Statistics
using TOML

using .DistributionBenchmarkCases:
    NORMAL_SIZES, SCALAR_GALLERY_FAMILIES, SCALAR_GALLERY_SIZES,
    STRUCTURED_SIZES, mvn_inputs, normal_observations, normal_parameters,
    scalar_family_inputs

const DEFAULT_ROUNDS = 5
const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

const NORMAL_KERNEL = plate(normal.logpdf;
    have = (:x, :location, :scale), want = :logpdf, batched = (:x,))
const CAUCHY_KERNEL = plate(cauchy.logpdf;
    have = (:x, :location, :scale), want = :logpdf, batched = (:x,))
const LAPLACE_KERNEL = plate(laplace.logpdf;
    have = (:x, :location, :scale), want = :logpdf, batched = (:x,))
const BERNOULLI_KERNEL = plate(bernoulli.logpdf;
    have = (:observed, :logit), want = :logpdf, batched = (:observed,))
const LOGNORMAL_KERNEL = plate(lognormal.logpdf;
    have = (:x, :location, :log_scale), want = :logpdf, batched = (:x,))
const EXPONENTIAL_KERNEL = plate(exponential.logpdf;
    have = (:x, :log_scale), want = :logpdf, batched = (:x,))
const GEOMETRIC_KERNEL = plate(geometric.logpdf;
    have = (:observed, :logitp), want = :logpdf, batched = (:observed,))
const UNIFORM_KERNEL = plate(uniform.logpdf;
    have = (:x, :lower, :upper), want = :logpdf, batched = (:x,))
const MVNORMAL_KERNEL = prepare(mvnormal.logpdf;
    have = (:x, :μ, :chol), want = :logpdf)

_family_kernel(::Val{:cauchy_location_scale}) = CAUCHY_KERNEL
_family_kernel(::Val{:laplace_location_scale}) = LAPLACE_KERNEL
_family_kernel(::Val{:bernoulli_logit}) = BERNOULLI_KERNEL
_family_kernel(::Val{:lognormal_logscale}) = LOGNORMAL_KERNEL
_family_kernel(::Val{:exponential_logscale}) = EXPONENTIAL_KERNEL
_family_kernel(::Val{:geometric_logit}) = GEOMETRIC_KERNEL
_family_kernel(::Val{:uniform_bounded}) = UNIFORM_KERNEL

_active_port(::Val{:cauchy_location_scale}) = :x
_active_port(::Val{:laplace_location_scale}) = :x
_active_port(::Val{:bernoulli_logit}) = :logit
_active_port(::Val{:lognormal_logscale}) = :x
_active_port(::Val{:exponential_logscale}) = :x
_active_port(::Val{:geometric_logit}) = :logitp
_active_port(::Val{:uniform_bounded}) = :x

function _expected_gradient(::Val{:cauchy_location_scale}, inputs)
    (; x, location, scale) = inputs
    z = @. (x - location) / scale
    @. -2z / (scale * (1 + z^2))
end

function _expected_gradient(::Val{:laplace_location_scale}, inputs)
    (; x, location, scale) = inputs
    @. -sign(x - location) / scale
end

function _expected_gradient(::Val{:bernoulli_logit}, inputs)
    (; observed, logit) = inputs
    probability = inv(1 + exp(-logit))
    sum(observed) - length(observed) * probability
end

function _expected_gradient(::Val{:lognormal_logscale}, inputs)
    (; x, location, log_scale) = inputs
    scale = exp(log_scale)
    @. -(1 + (log(x) - location) / scale^2) / x
end

function _expected_gradient(::Val{:exponential_logscale}, inputs)
    (; x, log_scale) = inputs
    fill(-exp(-log_scale), length(x))
end

function _expected_gradient(::Val{:geometric_logit}, inputs)
    (; observed, logitp) = inputs
    probability = inv(1 + exp(-logitp))
    sum((1 - probability) - value * probability for value in observed)
end

function _expected_gradient(::Val{:uniform_bounded}, inputs)
    zeros(length(inputs.x))
end

function _normal_expected_gradient(inputs)
    (; x, location, scale) = inputs
    @. -(x - location) / scale^2
end

function _mvn_expected_gradient(inputs)
    (; x, μ, chol) = inputs
    centered = x .- μ
    whitened = LowerTriangular(chol) \ centered
    -(transpose(LowerTriangular(chol)) \ whitened)
end

function _measurement(f; rounds::Int)
    benchmark = @benchmarkable $f()
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

_gradient_error(observed::Number, expected::Number) = abs(observed - expected)
_gradient_error(observed, expected) = maximum(abs.(observed .- expected); init = 0.0)

function _row(group, family, n, kernel, inputs::NamedTuple, active,
              expected_gradient; rounds)
    args = Tuple(inputs)
    prepared = prepare_ad(kernel, BACKEND, args...; active)
    returned_call = let prepared = prepared, args = args
        () -> ad_gradient(prepared, args...)
    end
    returned_gradient = returned_call()
    row = Dict{String,Any}(
        "group" => group,
        "family" => family,
        "n" => n,
        "active" => string(active),
        "active_kind" => returned_gradient isa Number ? "scalar" : "vector",
        "max_abs_error" => _gradient_error(returned_gradient, expected_gradient),
        "returned_gradient" => _measurement(returned_call; rounds),
    )

    if returned_gradient isa AbstractArray
        destination = similar(returned_gradient)
        caller_owned_call = let prepared = prepared, destination = destination,
                                args = args
            () -> ad_value_and_gradient!(prepared, destination, args...)
        end
        value, caller_owned_gradient = caller_owned_call()
        caller_owned_gradient === destination ||
            error("$family N=$n did not return the caller-owned destination")
        row["max_abs_error"] = max(
            row["max_abs_error"],
            _gradient_error(caller_owned_gradient, expected_gradient),
        )
        row["max_value_error"] = abs(value - kernel(args...))
        row["caller_owned_gradient"] = _measurement(caller_owned_call; rounds)
    end
    row
end

function _sizes(name, defaults)
    raw = get(ENV, name, "")
    isempty(raw) && return defaults
    Tuple(parse(Int, strip(item)) for item in split(raw, ','))
end

function _output_path()
    for arg in ARGS
        startswith(arg, "--output=") && return split(arg, '='; limit = 2)[2]
    end
    nothing
end

_git(repo, args...) = readchomp(Cmd(["git", "-C", repo, string.(args)...]))

function _source_receipts()
    receipt_dir = joinpath(@__DIR__, "receipts")
    normal_receipt = TOML.parsefile(
        joinpath(receipt_dir, "distribution-logdensity-v1.toml"))
    gallery_receipt = TOML.parsefile(
        joinpath(receipt_dir, "scalar-distribution-gallery-v1.toml"))
    structured_receipt = TOML.parsefile(
        joinpath(receipt_dir, "structured-distribution-logdensity-v1.toml"))

    Tuple(Int(row["n"]) for row in normal_receipt["measurements"]) == NORMAL_SIZES ||
        error("Normal gradient sizes drifted from the distribution receipt")
    Tuple(gallery_receipt["protocol"]["families"]) == SCALAR_GALLERY_FAMILIES ||
        error("scalar gradient families drifted from the distribution receipt")
    Tuple(Int.(gallery_receipt["protocol"]["sizes"])) == SCALAR_GALLERY_SIZES ||
        error("scalar gradient sizes drifted from the distribution receipt")
    Tuple(Int(row["n"]) for row in structured_receipt["measurements"]) ==
        STRUCTURED_SIZES ||
        error("structured gradient sizes drifted from the distribution receipt")

    Dict(
        "normal" => Dict(
            "schema" => normal_receipt["schema"],
            "generated_at" => normal_receipt["generated_at"],
            "reactivekernels_sha" => normal_receipt["pins"]["reactivekernels_sha"],
        ),
        "scalar_gallery" => Dict(
            "schema" => gallery_receipt["schema"],
            "generated_at" => gallery_receipt["generated_at"],
            "reactivekernels_sha" => gallery_receipt["pins"]["reactivekernels_sha"],
        ),
        "structured" => Dict(
            "schema" => structured_receipt["schema"],
            "generated_at" => structured_receipt["generated_at"],
            "reactivekernels_sha" =>
                structured_receipt["pins"]["reactivekernels_sha"],
        ),
    )
end

function run_benchmark()
    repo = normpath(joinpath(@__DIR__, ".."))
    dirty = !isempty(_git(repo, "status", "--porcelain"))
    dirty && !("--allow-dirty" in ARGS) && error(
        "refusing to write a receipt from a dirty tree; commit first or pass --allow-dirty",
    )
    rounds = parse(Int, get(
        ENV, "RK_DISTRIBUTION_GRADIENT_ROUNDS", string(DEFAULT_ROUNDS)))
    source_receipts = _source_receipts()
    rows = Dict{String,Any}[]

    Random.seed!(0x5eed)
    parameters = normal_parameters()
    for n in _sizes("RK_DISTRIBUTION_GRADIENT_NORMAL_SIZES", NORMAL_SIZES)
        inputs = (;
            x = normal_observations(n),
            location = parameters.location,
            scale = parameters.scale,
        )
        push!(rows, _row(
            "normal_plate", "normal", n, NORMAL_KERNEL, inputs, :x,
            _normal_expected_gradient(inputs); rounds,
        ))
        @printf("group=normal_plate family=normal N=%d complete\n", n)
    end

    gallery_sizes = _sizes(
        "RK_DISTRIBUTION_GRADIENT_GALLERY_SIZES", SCALAR_GALLERY_SIZES)
    for family in SCALAR_GALLERY_FAMILIES, n in gallery_sizes
        tag = Val(Symbol(family))
        inputs = scalar_family_inputs(tag, n)
        push!(rows, _row(
            "scalar_gallery", family, n, _family_kernel(tag), inputs,
            _active_port(tag), _expected_gradient(tag, inputs); rounds,
        ))
        @printf("group=scalar_gallery family=%s N=%d complete\n", family, n)
    end

    for n in _sizes("RK_DISTRIBUTION_GRADIENT_STRUCTURED_SIZES", STRUCTURED_SIZES)
        inputs = mvn_inputs(n)
        push!(rows, _row(
            "structured", "mvnormal_cholesky", n, MVNORMAL_KERNEL, inputs, :x,
            _mvn_expected_gradient(inputs); rounds,
        ))
        @printf("group=structured family=mvnormal_cholesky N=%d complete\n", n)
    end

    receipt = Dict{String,Any}(
        "schema" => "distribution-gradient-v1",
        "generated_at" => string(now(UTC), "Z"),
        "pins" => Dict(
            "reactivekernels_sha" => _git(repo, "rev-parse", "HEAD"),
            "reactivekernels_dirty" => dirty,
            "differentiationinterface_version" =>
                string(Base.pkgversion(DifferentiationInterface)),
            "enzyme_version" => string(Base.pkgversion(Enzyme)),
            "julia_version" => string(VERSION),
        ),
        "environment" => Dict(
            "os" => string(Sys.KERNEL),
            "arch" => string(Sys.ARCH),
            "cpu" => first(Sys.cpu_info()).model,
            "julia_threads" => Threads.nthreads(),
        ),
        "protocol" => Dict(
            "rounds" => rounds,
            "estimator" => "median of per-round BenchmarkTools minimum times",
            "samples_per_round" => 200,
            "seconds_per_round" => 0.25,
            "preparation_in_timed_region" => false,
            "backend" => "AutoEnzyme(mode = Enzyme.Reverse)",
            "returned_surface" => "ad_gradient",
            "caller_owned_surface" => "ad_value_and_gradient!",
            "scalar_destination_policy" =>
                "scalar gradients use ad_gradient and return an isbits Float64",
            "normal_sizes" => collect(NORMAL_SIZES),
            "scalar_gallery_families" => collect(SCALAR_GALLERY_FAMILIES),
            "scalar_gallery_sizes" => collect(SCALAR_GALLERY_SIZES),
            "structured_sizes" => collect(STRUCTURED_SIZES),
        ),
        "source_receipts" => source_receipts,
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

end # module DistributionGradientBenchmark

if abspath(PROGRAM_FILE) == @__FILE__
    using .DistributionGradientBenchmark
    DistributionGradientBenchmark.run_benchmark()
end
