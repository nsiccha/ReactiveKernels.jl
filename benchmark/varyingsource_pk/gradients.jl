# julia --threads=1 --project=<consumer-env> gradients.jl <pk_bdf_model.so> <results.tsv>
using DifferentiationInterface: AutoEnzyme
import Enzyme
using LinearAlgebra: BLAS
using Libdl
using Printf
using Random
using Statistics: median
include("model.jl")

BLAS.set_num_threads(1)
const PK_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

struct StanPK
    library::Ptr{Cvoid}
    model::Ptr{Cvoid}
    gradient_call::Ptr{Cvoid}
    destroy::Ptr{Cvoid}
end
function stan_error(library, ptr)
    message = unsafe_string(ptr)
    ccall(Libdl.dlsym(library, :bs_free_error_msg), Cvoid, (Ptr{Cchar},), ptr)
    error(message)
end
json_number(x::Real) = isfinite(x) ? string(x) : error("nonfinite fixture data")
json_number(x::AbstractVector) = "[" * join(json_number.(x), ',') * "]"
function stan_data(cols, sched, names, tol)
    data = Dict{String,Any}("n_subjects" => sched.n_subjects,
        "n_reference" => sched.reference_ends[end],
        "n_dose" => length(sched.dose_amount), "n_lag" => length(sched.unique_dts),
        "n_concentration" => length(sched.concentration_idxs),
        "n_obs" => length(cols[:dv]), "rtol" => tol, "atol" => tol,
        "param_index" => [findfirst(==(port), names) for port in PK_PORTS])
    for key in (:reference_ends, :dose_ends, :lag_ends, :concentration_ends,
            :dose_amount, :dose_index, :treatment_map, :unique_dts,
            :concentration_idxs, :dosing_time_idxs, :obs_map)
        data[string(key)] = getproperty(sched, key)
    end
    for key in (:dose_x, :age_s, :dv)
        data[string(key)] = cols[key]
    end
    return "{" * join(["\"$key\":" * json_number(data[key])
        for key in sort!(collect(keys(data)))], ',') * "}"
end
function StanPK(library, json)
    lib = Libdl.dlopen(library)
    err = Ref{Ptr{Cchar}}(C_NULL)
    ptr = ccall(Libdl.dlsym(lib, :bs_model_construct), Ptr{Cvoid},
        (Cstring, Cuint, Ref{Ptr{Cchar}}), json, 7, err)
    ptr == C_NULL && stan_error(lib, err[])
    return StanPK(lib, ptr, Libdl.dlsym(lib, :bs_log_density_gradient),
        Libdl.dlsym(lib, :bs_model_destruct))
end
function stan_work(model, p)
    g, value, err = zeros(length(p)), Ref(0.0), Ref{Ptr{Cchar}}(C_NULL)
    work = () -> begin
        # The Stan model takes the same unconstrained coordinates and includes
        # RK's sigma Jacobian itself. Retain all density constants on both sides.
        status = ccall(model.gradient_call, Cint,
            (Ptr{Cvoid}, Cuchar, Cuchar, Ptr{Cdouble}, Ref{Cdouble},
             Ptr{Cdouble}, Ref{Ptr{Cchar}}), model.model, 0, 0, p, value, g, err)
        status == 0 || stan_error(model.library, err[])
        return nothing
    end
    work()
    return work, g, value
end
relative_error(g, reference) = maximum(abs.(g .- reference)) / maximum(abs.(reference))

function main(library, output)
    println("Julia ", VERSION, "; Enzyme ", pkgversion(Enzyme),
        "; CPU ", Sys.CPU_NAME, "; Julia/BLAS threads=1")
    println("RK bound controls: series_rtol=1e-15, watson_terms=8; complete reverse only")
    rows = []
    plan = pk_plan()
    for replicas in (1, 10)
        cols = pk_columns(replicas)
        sched = build_varyingsource_pk_schedule(cols[:subj], cols[:time],
            cols[:dsubj], cols[:dtime], cols[:damt], cols[:treatment])
        bound = bind_data(plan, cols; dims = Dict(:kernel_nsub_conc => 3replicas))
        built = build_kernel(bound)
        names = coordinate_names(built.layout)
        println("subjects=", sched.n_subjects, "; observations=", length(cols[:dv]),
            "; doses=", length(cols[:damt]), "; parameters=", join(names, ','))
        models = [StanPK(library, stan_data(cols, sched, names, tol))
            for tol in (1e-6, 1e-10, 1e-12)]
        info = ccall(Libdl.dlsym(models[1].library, :bs_model_info), Cstring,
            (Ptr{Cvoid},), models[1].model)
        println(unsafe_string(info))
        for label in ("P1", "P2", "P3-shape8")
            p = pk_point(names, label)
            _, gtight10, vtight10 = stan_work(models[2], p)
            _, gtight12, vtight12 = stan_work(models[3], p)
            reference, reference_value = copy(gtight12), vtight12[]
            stanrun, gstan, vstan = stan_work(models[1], p)
            q = prepare_sampler(built, bound, p; backend = PK_BACKEND)
            grk, vrk = zeros(length(p)), Ref(0.0)
            rkrun = () -> begin
                value, _ = sampler_value_and_gradient!(q, grk, p)
                vrk[] = value
                return nothing
            end
            rkrun()
            stanerr = relative_error(gstan, reference)
            reference_err = relative_error(gtight10, reference)
            rkerr = relative_error(grk, reference)
            println(label, " nsub=", sched.n_subjects,
                " reference gradient convergence=", reference_err,
                "; Stan production error=", stanerr, "; RK error=", rkerr,
                "; RK/reference density difference=", abs(vrk[] - reference_value))
            reference_err < max(stanerr / 10, 1e-11) ||
                error("tight Stan reverse gradients have not converged")
            rkerr < 1e-5 || error("PK slice reverse parity failed")
            abs(vrk[] - reference_value) / max(1, abs(reference_value)) < 1e-7 ||
                error("PK slice density parity failed")
            runs = (("RK-emitted-Enzyme", rkrun, grk, vrk),
                ("Stan-BDF", stanrun, gstan, vstan))
            samples = [Float64[] for _ in runs]
            for _ in 1:10, (_, run, _, _) in runs
                run()
            end
            GC.gc()
            rng = MersenneTwister(9)
            for _ in 1:60, j in randperm(rng, length(runs))
                run = runs[j][2]
                start = time_ns()
                for _ in 1:5
                    run()
                end
                push!(samples[j], (time_ns() - start) / 5e9)
            end
            for (j, (path, run, g, value)) in enumerate(runs)
                push!(rows, (case = label, subjects = sched.n_subjects, path = path,
                    median_us = 1e6 * median(samples[j]),
                    gradient_error = relative_error(g, reference),
                    max_absolute_error = maximum(abs.(g .- reference)),
                    value_error = abs(value[] - reference_value), stan_error = stanerr,
                    reference_error = reference_err,
                    reference_value_error = abs(vtight10[] - reference_value),
                    matched = relative_error(g, reference) <= stanerr,
                    allocated = @allocated(run()), value = value[],
                    coordinates = copy(p), gradient = copy(g), reference = reference))
                @printf("%s nsub=%d %-19s median=%.2f μs relerr=%.3g matched=%s\n",
                    label, sched.n_subjects, path, rows[end].median_us,
                    rows[end].gradient_error, rows[end].matched)
            end
        end
        for model in models
            ccall(model.destroy, Cvoid, (Ptr{Cvoid},), model.model)
            Libdl.dlclose(model.library)
        end
    end
    open(output, "w") do io
        println(io, "case\tsubjects\tpath\tmedian_us\tgradient_rel_error\tgradient_abs_error\tvalue_abs_error\tstan_rel_error\treference_rel_error\treference_value_abs_error\tmatched\tJulia_allocated_bytes\tvalue\tcoordinates\tgradient\treference_gradient")
        for r in rows
            println(io, join((r.case, r.subjects, r.path, r.median_us,
                r.gradient_error, r.max_absolute_error, r.value_error, r.stan_error,
                r.reference_error, r.reference_value_error, r.matched, r.allocated,
                r.value, join(r.coordinates, ','), join(r.gradient, ','),
                join(r.reference, ',')), '\t'))
        end
    end
end

main(ARGS...)
