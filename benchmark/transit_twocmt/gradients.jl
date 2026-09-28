# Run in a scratch consumer environment containing RK PPL, Enzyme, and DI:
# julia --threads=1 --project=<env> gradients.jl <unit_bdf_model.so> <results.tsv>
using DifferentiationInterface: AutoEnzyme, Constant, prepare_gradient, gradient!
import Enzyme
using ReactiveKernelsPPL
using LinearAlgebra: BLAS
using Libdl
using Random
using Statistics: median
using Printf

BLAS.set_num_threads(1)
const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
const SETTINGS = ((1e-15, 24), (1e-15, 8), (1e-9, 4),
    (1e-7, 4), (1e-6, 4), (1e-5, 4))
const RULES = map(SETTINGS) do (rtol, terms)
    prepare_transit_twocmt_rule(; series_rtol = rtol, watson_terms = terms)
end
const CASES = (
    ("P1", [0.08, 0.15, 0.05, 0.20, 1.20]),
    ("P2", [0.50, 0.30, 0.40, 0.05, 1.05]),
    ("P3-shape8", [0.08, 0.15, 0.05, 0.30, 8.00]),
)
plain_sum(p, ts, rtol, terms) = sum(transit_twocmt_unit_response(ts,
    p[1], p[2], p[3], p[4], p[5];
    series_rtol = rtol, watson_terms = terms))
ruled_sum(p, ts, setting_index) = sum(RULES[setting_index](ts, p))

function lag_grid()
    rng = MersenneTwister(7)
    doses = 0.0:24.0:672.0
    writes = sort!(vcat(rand(rng, 120) .* 672,
        [d + rand(rng) * 24 for d in doses[1:20] for _ in 1:10]))
    lags = sort!(unique!(filter(>=(0), vec([w - d for w in writes, d in doses]))))
    return lags[unique!(round.(Int, range(1, length(lags); length = 1401)))]
end

struct StanUnit
    library::Ptr{Cvoid}
    model::Ptr{Cvoid}
    gradient_call::Ptr{Cvoid}
    free_error::Ptr{Cvoid}
    destroy::Ptr{Cvoid}
end
function stan_error(lib, error_ptr)
    message = unsafe_string(error_ptr)
    ccall(Libdl.dlsym(lib, :bs_free_error_msg), Cvoid, (Ptr{Cchar},), error_ptr)
    error(message)
end
function StanUnit(library, ts, tol)
    lib = Libdl.dlopen(library)
    data = "{\"n\":$(length(ts)),\"ts\":[$(join(ts, ','))],\"rtol\":$tol,\"atol\":$tol}"
    err = Ref{Ptr{Cchar}}(C_NULL)
    ptr = ccall(Libdl.dlsym(lib, :bs_model_construct), Ptr{Cvoid},
        (Cstring, Cuint, Ref{Ptr{Cchar}}), data, 7, err)
    ptr == C_NULL && stan_error(lib, err[])
    return StanUnit(lib, ptr, Libdl.dlsym(lib, :bs_log_density_gradient),
        Libdl.dlsym(lib, :bs_free_error_msg), Libdl.dlsym(lib, :bs_model_destruct))
end
function stan_gradient!(g, model, p, value, err)
    status = ccall(model.gradient_call, Cint,
        (Ptr{Cvoid}, Cuchar, Cuchar, Ptr{Cdouble}, Ref{Cdouble},
         Ptr{Cdouble}, Ref{Ptr{Cchar}}), model.model, 0, 0, p, value, g, err)
    status == 0 || stan_error(model.library, err[])
    return g
end

function enzyme_work(f, p, contexts...)
    prep = prepare_gradient(f, BACKEND, p, contexts...)
    g = zeros(5)
    work = () -> gradient!(f, g, prep, BACKEND, p, contexts...)
    work()
    return work, g
end
function stan_work(model, p)
    g = zeros(5)
    value = Ref(0.0)
    err = Ref{Ptr{Cchar}}(C_NULL)
    work = () -> stan_gradient!(g, model, p, value, err)
    work()
    return work, g
end
relative_error(g, reference) = maximum(abs.(g .- reference)) / maximum(abs.(reference))

function main(library, output)
    ts = lag_grid()
    println("Julia ", VERSION, "; Enzyme ", pkgversion(Enzyme),
        "; threads=", Threads.nthreads(), "; grid=", length(ts),
        "; lag range=", extrema(ts))
    models = [StanUnit(library, ts, tol) for tol in (1e-6, 1e-10, 1e-12)]
    info = ccall(Libdl.dlsym(models[1].library, :bs_model_info), Cstring,
        (Ptr{Cvoid},), models[1].model)
    println("CPU ", Sys.CPU_NAME, "; ", unsafe_string(info))
    prepared = []
    for (label, p) in CASES
        _, gtight10 = stan_work(models[2], p)
        _, gtight12 = stan_work(models[3], p)
        reference = copy(gtight12)
        stanrun, gstan = stan_work(models[1], p)
        stanerr = relative_error(gstan, reference)
        reference_err = relative_error(gtight10, reference)
        println(label, " Stan production relative gradient error=", stanerr,
            "; reference convergence=", reference_err)
        reference_err < stanerr / 10 || error("tight Stan gradients have not converged")
        runs = Tuple{String,Float64,Int,Function,Vector{Float64}}[
            ("Stan-BDF", 1e-6, 0, stanrun, gstan)]
        for (setting_index, (rtol, terms)) in enumerate(SETTINGS)
            for (name, f, contexts) in (
                    ("RK-Enzyme", plain_sum,
                        (Constant(ts), Constant(rtol), Constant(terms))),
                    ("RK-rule-Enzyme", ruled_sum,
                        (Constant(ts), Constant(setting_index))))
                run, g = enzyme_work(f, p, contexts...)
                println(label, " ", name, " rtol=", rtol, " terms=", terms,
                    " relative error=", relative_error(g, reference))
                push!(runs, (name, rtol, terms, run, g))
            end
        end
        samples = [Float64[] for _ in runs]
        for _ in 1:10, (_, _, _, run, _) in runs
            run()
        end
        GC.gc()
        # Deterministically shuffle each round so shared-host drift affects all
        # paths. Each sample is a batch of five complete reverse-gradient calls.
        rng = MersenneTwister(9)
        for round in 1:60
            for j in randperm(rng, length(runs))
                run = runs[j][4]
                start = time_ns()
                for _ in 1:5
                    run()
                end
                push!(samples[j], (time_ns() - start) / 5e9)
            end
        end
        for (j, (name, rtol, terms, run, g)) in enumerate(runs)
            push!(prepared, (case = label, path = name, series_rtol = rtol,
                watson_terms = terms, median_us = 1e6 * median(samples[j]),
                gradient_error = relative_error(g, reference),
                max_absolute_error = maximum(abs.(g .- reference)),
                stan_error = stanerr, reference_error = reference_err,
                matched = relative_error(g, reference) <= stanerr,
                allocated = @allocated(run()), gradient = copy(g)))
        end
    end
    open(output, "w") do io
        println(io, "case\tpath\tseries_rtol\twatson_terms\tmedian_us\tgradient_rel_error\tgradient_abs_error\tstan_rel_error\treference_rel_error\tmatched\tJulia_allocated_bytes\tgradient")
        for r in prepared
            println(io, join((r.case, r.path, r.series_rtol, r.watson_terms,
                r.median_us, r.gradient_error, r.max_absolute_error, r.stan_error,
                r.reference_error, r.matched, r.allocated, join(r.gradient, ',')), '\t'))
            @printf("%s %-16s rtol=%g terms=%d median=%.2f μs relerr=%.3g matched=%s\n",
                r.case, r.path, r.series_rtol, r.watson_terms, r.median_us,
                r.gradient_error, r.matched)
        end
    end
    for model in models
        ccall(model.destroy, Cvoid, (Ptr{Cvoid},), model.model)
        Libdl.dlclose(model.library)
    end
end

main(ARGS...)
