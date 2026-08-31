#!/usr/bin/env julia

module StructuredDistributionComparison

include("distribution_benchmark_cases.jl")

using BenchmarkTools
using Dates
using Distributions
using LinearAlgebra
using Pkg
using Printf
using ProbabilityMeasures
using Random
using Reactant
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    MVNORMAL_SOURCE
using Statistics
using TOML

using .DistributionBenchmarkCases: STRUCTURED_SIZES, mvn_inputs

using Reactant: @compile

const PROBABILITY_MEASURES_SHA = "7cf3a6e112aaae2097b8d401b256d1bce635e03e"
const DEFAULT_SIZES = STRUCTURED_SIZES
const DEFAULT_ROUNDS = 5

function _evaluate_source(source)
    sandbox = Module(gensym(:StructuredDistributionKernel))
    Core.eval(sandbox, :(using ReactiveKernels))
    Core.eval(sandbox, Meta.parse("begin\n" * source * "\nend";
                                  filename = "structured-distribution-kernel.jl"))
    getfield(sandbox, :docs_example)
end

const MVN_EXAMPLE = _evaluate_source(MVNORMAL_SOURCE)

_pm_logdensity(d, x) = ProbabilityMeasures.logdensityof(d, x)
_distributions_logpdf(d, x) = Distributions.logpdf(d, x)

_compile_rk_mvn(k, x, μ, L) = @compile sync = true k(x, μ, L)
_compile_pm(d, x) = @compile sync = true _pm_logdensity(d, x)
_compile_distributions(d, x) =
    @compile sync = true _distributions_logpdf(d, x)

function _measurement(f, args...; rounds::Int)
    benchmark = @benchmarkable $f($args...)
    times_ns = Float64[]
    bytes = Int[]
    allocs = Int[]
    for _ in 1:rounds
        estimate = minimum(run(benchmark; samples = 100, seconds = 0.15))
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

function _package_version(name)
    for info in values(Pkg.dependencies())
        info.name == name && return string(info.version)
    end
    error("package $name is absent from the benchmark environment")
end

_git(repo, args...) = readchomp(Cmd(["git", "-C", repo, string.(args)...]))

function _sizes()
    raw = get(ENV, "RK_STRUCTURED_BENCH_SIZES", "")
    isempty(raw) && return DEFAULT_SIZES
    Tuple(parse(Int, strip(item)) for item in split(raw, ','))
end

function _output_path()
    for arg in ARGS
        startswith(arg, "--output=") && return split(arg, '='; limit = 2)[2]
    end
    nothing
end

_diagnostic(err) = first(split(sprint(showerror, err), '\n'))

function _baseline_reactant!(support, errors, family, pm, dist, rx, reference)
    pm_compiled = dist_compiled = nothing
    pm_compile_s = dist_compile_s = nothing
    if support[family]["probability_measures_reactant"]
        try
            pm_compile_s = @elapsed pm_compiled = _compile_pm(pm, rx)
            isapprox(Float64(pm_compiled(rx)), reference; rtol = 1e-10) ||
                error("ProbabilityMeasures + Reactant value mismatch")
        catch err
            support[family]["probability_measures_reactant"] = false
            errors[family]["probability_measures_reactant"] = _diagnostic(err)
        end
    end
    if support[family]["distributions_reactant"]
        try
            dist_compile_s = @elapsed dist_compiled = _compile_distributions(dist, rx)
            isapprox(Float64(dist_compiled(rx)), reference; rtol = 1e-10) ||
                error("Distributions + Reactant value mismatch")
        catch err
            support[family]["distributions_reactant"] = false
            errors[family]["distributions_reactant"] = _diagnostic(err)
        end
    end
    (; pm_compiled, dist_compiled, pm_compile_s, dist_compile_s)
end

function _row(family, n, reference, native_values, rk_native, rk_native_args,
              rk_compiled, rk_args,
              pm, dist, x, rx, baselines, rounds)
    observed = Any[native_values..., Float64(rk_compiled(rk_args...))]
    baselines.pm_compiled === nothing ||
        push!(observed, Float64(baselines.pm_compiled(rx)))
    baselines.dist_compiled === nothing ||
        push!(observed, Float64(baselines.dist_compiled(rx)))
    row = Dict{String,Any}(
        "family" => family,
        "n" => n,
        "max_abs_error" => maximum(abs(value - reference) for value in observed),
        "rk_native" => _measurement(rk_native, rk_native_args...; rounds),
        "distributions_native" =>
            _measurement(_distributions_logpdf, dist, x; rounds),
        "probability_measures_native" =>
            _measurement(_pm_logdensity, pm, x; rounds),
        "rk_reactant" => _measurement(rk_compiled, rk_args...; rounds),
    )
    baselines.pm_compiled === nothing || (row["probability_measures_reactant"] =
        _measurement(baselines.pm_compiled, rx; rounds))
    baselines.dist_compiled === nothing || (row["distributions_reactant"] =
        _measurement(baselines.dist_compiled, rx; rounds))
    row
end

function run_benchmark()
    repo = normpath(joinpath(@__DIR__, ".."))
    dirty = !isempty(_git(repo, "status", "--porcelain"))
    dirty && !("--allow-dirty" in ARGS) && error(
        "refusing to write a receipt from a dirty tree; commit first or pass --allow-dirty",
    )
    Random.seed!(0x51a7)
    rounds = parse(Int, get(ENV, "RK_STRUCTURED_BENCH_ROUNDS", string(DEFAULT_ROUNDS)))
    support = Dict(family => Dict(
        "rk_reactant" => true,
        "probability_measures_reactant" => true,
        "distributions_reactant" => true,
    ) for family in ("mvnormal_cholesky",))
    errors = Dict(family => Dict{String,String}() for family in keys(support))
    rows = Dict{String,Any}[]

    mvn_kernel = MVN_EXAMPLE.kernel
    for n in _sizes()
        mvn = mvn_inputs(n)
        pm_mvn = ProbabilityMeasures.MvNormal(mvn.μ, mvn.chol)
        dist_mvn = Distributions.MvNormal(
            mvn.μ, Symmetric(mvn.chol * mvn.chol'))
        mvn_reference = mvn_kernel(mvn.x, mvn.μ, mvn.chol)
        mvn_values = (
            mvn_reference, _distributions_logpdf(dist_mvn, mvn.x),
            _pm_logdensity(pm_mvn, mvn.x),
        )
        all(value -> isapprox(value, mvn_reference; rtol = 1e-10),
            (mvn_values[2], mvn_values[3])) || error("MVN parity mismatch at n=$n")
        rmx, rmμ, rmL = Reactant.to_rarray.((mvn.x, mvn.μ, mvn.chol))
        mvn_compile_s = @elapsed mvn_compiled =
            _compile_rk_mvn(mvn_kernel, rmx, rmμ, rmL)
        baselines = _baseline_reactant!(support, errors, "mvnormal_cholesky",
                                        pm_mvn, dist_mvn, rmx, mvn_reference)
        row = _row("mvnormal_cholesky", n, mvn_reference, mvn_values,
                   mvn_kernel, (mvn.x, mvn.μ, mvn.chol),
                   mvn_compiled, (rmx, rmμ, rmL),
                   pm_mvn, dist_mvn, mvn.x, rmx, baselines, rounds)
        row["rk_reactant_compile_seconds"] = mvn_compile_s
        push!(rows, row)
        @printf("n=%d MVN complete\n", n)
    end

    receipt = Dict{String,Any}(
        "schema" => "structured-distribution-logdensity-v1",
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
            "os" => string(Sys.KERNEL), "arch" => string(Sys.ARCH),
            "cpu" => first(Sys.cpu_info()).model,
            "julia_threads" => Threads.nthreads(), "reactant_backend" => "default CPU",
        ),
        "protocol" => Dict(
            "families" => ["MVN (Cholesky HAVE benchmark)"],
            "mvn_have_boundaries" => [
                "covariance", "cholesky", "precision", "precision_cholesky",
            ],
            "mvn_all_boundaries_native_and_reactant_accepted" => true,
            "construction_and_factorization_timed" => false,
            "element_type" => "Float64", "rounds" => rounds,
            "estimator" => "median of per-round BenchmarkTools minimum times",
            "samples_per_round" => 100, "seconds_per_round" => 0.15,
            "reactant_sync" => true, "reactant_transfers_included" => false,
            "reactant_compile_time_in_timed_region" => false,
        ),
        "support" => support,
        "support_errors" => errors,
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

end # module StructuredDistributionComparison

if abspath(PROGRAM_FILE) == @__FILE__
    using .StructuredDistributionComparison
    StructuredDistributionComparison.run_benchmark()
end
