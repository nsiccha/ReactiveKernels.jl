using LinearAlgebra: BLAS
using Printf, Statistics, Test
length(ARGS) in (3,4) && (length(ARGS)==3 || ARGS[4]=="counts-only") ||
    error("Usage: shared_gp.jl <repo-root> <results.tsv> <counts.tsv> [counts-only]")
const REPO_ROOT, OUTPUT, COUNTS = ARGS[1:3]
const MEASURE = length(ARGS) == 3
include(joinpath(REPO_ROOT, "benchmark/varyingsource_pkpd/model.jl"))
BLAS.set_num_threads(1)

function samples(f, n; batch=1)
    for _ in 1:3
        f()
    end
    GC.gc()
    times = Float64[]
    for _ in 1:n
        start = time_ns()
        for _ in 1:batch
            f()
        end
        push!(times, (time_ns() - start) / (1e3batch))
    end
    return (median(times), quantile(times, .25), quantile(times, .75), @allocated(f()))
end

function measure(columns, built, query, replicas, io)
    for case in 1:3
        u = pkpd_point(built.layout; case)
        value = query(u)
        steady = () -> query(u)
        med, q25, q75, bytes = samples(steady, 31; batch=5)
        println(io, join((replicas, length(columns[:sid]), case, "steady",
            med, q25, q75, bytes, value, join(u, ',')), '\t'))
        @printf("subjects=%d case=%d steady median=%.3f us bytes=%d\n",
            length(columns[:sid]), case, med, bytes)
        flush(io)
        flush(stdout)
    end
end

# Test instrumentation on RK-owned mathematical functions. Run after timing,
# in a primal-only process: these counters are deliberately not AD recipes.
function install_counters()
    @eval ReactiveKernelsPPL begin
        const _SHARED_GP_COUNTS = zeros(Int, 3)
        function varyingsource_gp_weights(w::AbstractVector{Float64}, d, c, sd)
            _SHARED_GP_COUNTS[1] += 1
            invoke(varyingsource_gp_weights, Tuple{AbstractVector,Any,Any,Any}, w, d, c, sd)
        end
        function varyingsource_effectiveness(w::Matrix{Float64}, d, c)
            _SHARED_GP_COUNTS[2] += 1
            invoke(varyingsource_effectiveness, Tuple{AbstractMatrix,Any,Any}, w, d, c)
        end
        if isdefined(@__MODULE__, :_vs_gp_normalizer)
            function _vs_gp_normalizer(has_doses::Bool, w::Matrix{Float64}, d, c)
                has_doses && (_SHARED_GP_COUNTS[2] += 1)
                invoke(_vs_gp_normalizer, Tuple{Any,Any,Any,Any}, has_doses, w, d, c)
            end
        end
        function varyingsource_log_placebo(t::AbstractVector, w::AbstractVector{Float64}, r, sd, lo, hi)
            isempty(t) || (_SHARED_GP_COUNTS[3] += 1)
            invoke(varyingsource_log_placebo,
                Tuple{AbstractVector,AbstractVector,Any,Any,Any,Any}, t, w, r, sd, lo, hi)
        end
    end
end

function count_case(plan, replicas, io)
    columns = pkpd_columns(replicas)
    bound = bind_data(plan, pkpd_bound_columns(columns);
        dims=Dict(:kernel_nsub_loc => length(columns[:sid])))
    built = build_kernel(bound)
    counts = ReactiveKernelsPPL._SHARED_GP_COUNTS
    fill!(counts, 0)
    query = prepare_query(built, bound, :sampler)
    preparation = copy(counts)
    fill!(counts, 0)
    u = pkpd_point(built.layout)
    value = Base.invokelatest(query, u)
    evaluation = copy(counts)
    fill!(counts, 0)
    changed = copy(u)
    changed[1] += .1
    Base.invokelatest(query, changed)
    @test counts == evaluation
    println(io, join((length(columns[:sid]), preparation..., evaluation..., value), '\t'))
    println("subjects=", length(columns[:sid]), " GP weights/normalizer/placebo courses: prepare=",
        preparation, " evaluation=", evaluation)
    flush(io)
end

function main()
    println("Julia ", VERSION, "; CPU ", Sys.CPU_NAME, "; Julia/BLAS threads=1")
    MEASURE && println("One initial prepare-and-first-evaluation per subject size; 31 steady samples of 5 calls, 3 warmups.")
    @test all(p.name != "Enzyme" for p in keys(Base.loaded_modules))
    plan = pkpd_plan()
    if MEASURE
        open(OUTPUT, "w") do io
            println(io, "replicas\tsubjects\tcase\tpath\tmedian_us\tq25_us\tq75_us\tJulia_bytes\tdensity\tRK_coordinates")
            for replicas in (1, 10)
                columns = pkpd_columns(replicas)
                bound = bind_data(plan, pkpd_bound_columns(columns);
                    dims=Dict(:kernel_nsub_loc => length(columns[:sid])))
                built = build_kernel(bound)
                u = pkpd_point(built.layout)
                # Measure the actual one-time query lifecycle, including bound
                # producers, lowering, native query creation and first-call
                # compilation. The built model and bound schedule are outside.
                setup = @timed begin
                    query = prepare_query(built, bound, :sampler)
                    query, Base.invokelatest(query, u)
                end
                query, value = setup.value
                println(io, join((replicas, length(columns[:sid]), 1,
                    "prepare-and-evaluate", 1e6setup.time, "", "", setup.bytes,
                    value, join(u, ',')), '\t'))
                @printf("subjects=%d prepare-and-first-evaluation=%.3f us bytes=%d\n",
                    length(columns[:sid]), 1e6setup.time, setup.bytes)
                flush(io)
                flush(stdout)
                Base.invokelatest(measure, columns, built, query, replicas, io)
            end
        end
    end
    install_counters()
    open(COUNTS, "w") do io
        println(io, "subjects\tprepare_weights\tprepare_normalizers\tprepare_placebos\tevaluate_weights\tevaluate_normalizers\tevaluate_placebos\tdensity")
        for replicas in (1, 10)
            Base.invokelatest(count_case, plan, replicas, io)
        end
    end
    @test all(p.name != "Enzyme" for p in keys(Base.loaded_modules))
end
main()
