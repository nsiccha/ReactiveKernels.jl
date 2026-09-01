#!/usr/bin/env julia

module ScalarDistributionGalleryComparison

include("distribution_benchmark_cases.jl")

using BenchmarkTools
using Dates
using Distributions
using LogExpFunctions: logistic
using Pkg
using Printf
using ProbabilityMeasures
using Random
using Reactant
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    cauchy, laplace, bernoulli, lognormal, exponential, geometric, uniform
using Statistics
using TOML

using .DistributionBenchmarkCases:
    SCALAR_GALLERY_FAMILIES, SCALAR_GALLERY_SIZES, scalar_family_inputs

using Reactant: @compile

const PROBABILITY_MEASURES_SHA = "7cf3a6e112aaae2097b8d401b256d1bce635e03e"
const DEFAULT_SIZES = SCALAR_GALLERY_SIZES
const DEFAULT_ROUNDS = 5
const FAMILIES = SCALAR_GALLERY_FAMILIES

const RK_CAUCHY = plate(cauchy.logpdf;
    have = (:x, :location, :scale), want = :logpdf, batched = (:x,))
const RK_LAPLACE = plate(laplace.logpdf;
    have = (:x, :location, :scale), want = :logpdf, batched = (:x,))
const RK_BERNOULLI = plate(bernoulli.logpdf;
    have = (:observed, :logit), want = :logpdf, batched = (:observed,))
const RK_LOGNORMAL = plate(lognormal.logpdf;
    have = (:x, :location, :log_scale), want = :logpdf, batched = (:x,))
const RK_EXPONENTIAL = plate(exponential.logpdf;
    have = (:x, :log_scale), want = :logpdf, batched = (:x,))
const RK_GEOMETRIC = plate(geometric.logpdf;
    have = (:observed, :logitp), want = :logpdf, batched = (:observed,))
const RK_UNIFORM = plate(uniform.logpdf;
    have = (:x, :lower, :upper), want = :logpdf, batched = (:x,))

rk_cauchy(xs, location, scale) = RK_CAUCHY(xs, location, scale)
rk_laplace(xs, location, scale) = RK_LAPLACE(xs, location, scale)
rk_bernoulli(observed, logit) = RK_BERNOULLI(observed, logit)
rk_lognormal(xs, location, log_scale) = RK_LOGNORMAL(xs, location, log_scale)
rk_exponential(xs, log_scale) = RK_EXPONENTIAL(xs, log_scale)
rk_geometric(observed, logitp) = RK_GEOMETRIC(observed, logitp)
rk_uniform(xs, lower, upper) = RK_UNIFORM(xs, lower, upper)

distributions_cauchy(xs, location, scale) =
    sum(Distributions.logpdf.(Distributions.Cauchy(location, scale), xs))
distributions_laplace(xs, location, scale) =
    sum(Distributions.logpdf.(Distributions.Laplace(location, scale), xs))
distributions_bernoulli(observed, logit) =
    sum(Distributions.logpdf.(Distributions.Bernoulli(logistic(logit)), observed))
distributions_lognormal(xs, location, log_scale) =
    sum(Distributions.logpdf.(Distributions.LogNormal(location, exp(log_scale)), xs))

distributions_exponential(xs, log_scale) =
    sum(Distributions.logpdf.(Distributions.Exponential(exp(log_scale)), xs))
distributions_geometric(observed, logitp) =
    sum(Distributions.logpdf.(Distributions.Geometric(logistic(logitp)), observed))
distributions_uniform(xs, lower, upper) =
    sum(Distributions.logpdf.(Distributions.Uniform(lower, upper), xs))

probability_measures_cauchy(xs, location, scale) = sum(
    ProbabilityMeasures.logdensityof.(
        ProbabilityMeasures.Cauchy(location, scale), xs))
probability_measures_laplace(xs, location, scale) = sum(
    ProbabilityMeasures.logdensityof.(
        ProbabilityMeasures.Laplace(location, scale), xs))
probability_measures_bernoulli(observed, logit) = sum(
    ProbabilityMeasures.logdensityof.(
        ProbabilityMeasures.Bernoulli(logistic(logit)), observed))
probability_measures_lognormal(xs, location, log_scale) = sum(
    ProbabilityMeasures.logdensityof.(
        ProbabilityMeasures.LogNormal(location, exp(log_scale)), xs))
probability_measures_exponential(xs, log_scale) = sum(
    ProbabilityMeasures.logdensityof.(
        ProbabilityMeasures.Exponential(exp(log_scale)), xs))
probability_measures_geometric(observed, logitp) = sum(
    ProbabilityMeasures.logdensityof.(
        ProbabilityMeasures.Geometric(logistic(logitp)), observed))
probability_measures_uniform(xs, lower, upper) = sum(
    ProbabilityMeasures.logdensityof.(
        ProbabilityMeasures.Uniform(lower, upper), xs))

_compile2(f, a, b) = @compile sync = true f(a, b)
_compile3(f, a, b, c) = @compile sync = true f(a, b, c)

function _compile(f, args)
    length(args) == 2 && return _compile2(f, args...)
    length(args) == 3 && return _compile3(f, args...)
    error("unsupported gallery benchmark arity $(length(args))")
end

function _measurement(f, args...; rounds::Int)
    benchmark = @benchmarkable $f($args...)
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

_family_functions(::Val{:cauchy_location_scale}) = (;
    rk = rk_cauchy, distributions = distributions_cauchy,
    probability_measures = probability_measures_cauchy)
_family_functions(::Val{:laplace_location_scale}) = (;
    rk = rk_laplace, distributions = distributions_laplace,
    probability_measures = probability_measures_laplace)
_family_functions(::Val{:bernoulli_logit}) = (;
    rk = rk_bernoulli, distributions = distributions_bernoulli,
    probability_measures = probability_measures_bernoulli)
_family_functions(::Val{:lognormal_logscale}) = (;
    rk = rk_lognormal, distributions = distributions_lognormal,
    probability_measures = probability_measures_lognormal)
_family_functions(::Val{:exponential_logscale}) = (;
    rk = rk_exponential, distributions = distributions_exponential,
    probability_measures = probability_measures_exponential)
_family_functions(::Val{:geometric_logit}) = (;
    rk = rk_geometric, distributions = distributions_geometric,
    probability_measures = probability_measures_geometric)
_family_functions(::Val{:uniform_bounded}) = (;
    rk = rk_uniform, distributions = distributions_uniform,
    probability_measures = probability_measures_uniform)

function _family_case(family, n)
    tag = Val(Symbol(family))
    native = Tuple(scalar_family_inputs(tag, n))
    traced = map(native) do value
        value isa Number ? Reactant.to_rarray(value; track_numbers = true) :
            Reactant.to_rarray(value)
    end
    (; native, traced, _family_functions(tag)...)
end

_git(repo, args...) = readchomp(Cmd(["git", "-C", repo, string.(args)...]))

function _package_version(name)
    for info in values(Pkg.dependencies())
        info.name == name && return string(info.version)
    end
    error("package $name is absent from the benchmark environment")
end

function _sizes()
    raw = get(ENV, "RK_SCALAR_GALLERY_BENCH_SIZES", "")
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

function run_benchmark()
    repo = normpath(joinpath(@__DIR__, ".."))
    dirty = !isempty(_git(repo, "status", "--porcelain"))
    dirty && !("--allow-dirty" in ARGS) && error(
        "refusing to write a receipt from a dirty tree; commit first or pass --allow-dirty",
    )
    Random.seed!(0x5ca1a7)
    rounds = parse(Int, get(
        ENV, "RK_SCALAR_GALLERY_BENCH_ROUNDS", string(DEFAULT_ROUNDS)))
    support = Dict(family => Dict(
        "rk_reactant" => true,
        "probability_measures_reactant" => true,
        "distributions_reactant" => true,
    ) for family in FAMILIES)
    support_errors = Dict(family => Dict{String,String}() for family in FAMILIES)
    rows = Dict{String,Any}[]

    for family in FAMILIES, n in _sizes()
        case = _family_case(family, n)
        reference = case.rk(case.native...)
        native_values = (
            reference,
            case.distributions(case.native...),
            case.probability_measures(case.native...),
        )
        all(value -> isapprox(value, reference; rtol = 1e-11), native_values) ||
            error("native value mismatch for $family at N=$n")

        rk_compile_s = @elapsed rk_compiled = _compile(case.rk, case.traced)
        rk_value = Float64(rk_compiled(case.traced...))
        isapprox(rk_value, reference; rtol = 1e-11) ||
            error("RK + Reactant value mismatch for $family at N=$n")

        pm_compile_s = @elapsed pm_compiled =
            _compile(case.probability_measures, case.traced)
        pm_value = Float64(pm_compiled(case.traced...))
        isapprox(pm_value, reference; rtol = 1e-11) ||
            error("ProbabilityMeasures + Reactant value mismatch for $family at N=$n")

        dist_compiled = nothing
        dist_compile_s = nothing
        if support[family]["distributions_reactant"]
            try
                dist_compile_s = @elapsed dist_compiled =
                    _compile(case.distributions, case.traced)
                dist_value = Float64(dist_compiled(case.traced...))
                isapprox(dist_value, reference; rtol = 1e-11) ||
                    error("Distributions + Reactant value mismatch")
            catch err
                support[family]["distributions_reactant"] = false
                support_errors[family]["distributions_reactant"] = _diagnostic(err)
            end
        end

        observed = Any[native_values..., rk_value, pm_value]
        dist_compiled === nothing || push!(observed,
            Float64(dist_compiled(case.traced...)))
        denominator = max(abs(reference), eps(Float64))
        row = Dict{String,Any}(
            "family" => family,
            "n" => n,
            "max_abs_error" => maximum(abs(value - reference) for value in observed),
            "max_relative_error" => maximum(
                abs(value - reference) / denominator for value in observed),
            "rk_native" => _measurement(case.rk, case.native...; rounds),
            "distributions_native" =>
                _measurement(case.distributions, case.native...; rounds),
            "probability_measures_native" =>
                _measurement(case.probability_measures, case.native...; rounds),
            "rk_reactant" => _measurement(rk_compiled, case.traced...; rounds),
            "probability_measures_reactant" =>
                _measurement(pm_compiled, case.traced...; rounds),
            "rk_reactant_compile_seconds" => rk_compile_s,
            "probability_measures_reactant_compile_seconds" => pm_compile_s,
        )
        if dist_compiled !== nothing
            row["distributions_reactant"] =
                _measurement(dist_compiled, case.traced...; rounds)
            row["distributions_reactant_compile_seconds"] = dist_compile_s
        end
        push!(rows, row)
        @printf("family=%s N=%d complete\n", family, n)
    end

    receipt = Dict{String,Any}(
        "schema" => "scalar-distribution-gallery-v1",
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
            "families" => collect(FAMILIES),
            "sizes" => collect(_sizes()),
            "formulas" => Dict(
                "cauchy_location_scale" => "Cauchy with location and scale",
                "laplace_location_scale" => "Laplace with location and scale",
                "bernoulli_logit" => "Bernoulli from logit probability",
                "lognormal_logscale" => "LogNormal from location and log scale",
                "exponential_logscale" => "Exponential from log scale",
                "geometric_logit" => "Geometric failures before success from logit p",
                "uniform_bounded" => "Uniform with dynamic lower and upper endpoints",
            ),
            "element_types" => Dict(
                "continuous_observations" => "Float64",
                "discrete_observations" => "Int",
                "parameters" => "Float64",
            ),
            "rounds" => rounds,
            "estimator" => "median of per-round BenchmarkTools minimum times",
            "samples_per_round" => 200,
            "seconds_per_round" => 0.25,
            "reactant_sync" => true,
            "reactant_transfers_included" => false,
            "reactant_parameters_traced" => true,
            "reactant_compile_time_in_timed_region" => false,
        ),
        "support" => support,
        "support_errors" => support_errors,
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

end # module ScalarDistributionGalleryComparison

if abspath(PROGRAM_FILE) == @__FILE__
    using .ScalarDistributionGalleryComparison
    ScalarDistributionGalleryComparison.run_benchmark()
end
