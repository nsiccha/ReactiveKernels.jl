using LinearAlgebra: BLAS
using Printf, Random, Statistics, Test
length(ARGS)==3 || error("Usage: primal.jl <repo-root> <build-dir> <results.tsv>")
const REPO_ROOT, BUILD_ROOT, OUTPUT = ARGS
include(joinpath(REPO_ROOT,"benchmark/varyingsource_pkpd/model.jl"))
include(joinpath(REPO_ROOT,"benchmark/varyingsource_pkpd/stan.jl"))
BLAS.set_num_threads(1)

function timings(runs)
    for _ in 1:10, run in runs
        run()
    end
    GC.gc()
    samples = [Float64[] for _ in runs]
    rng = MersenneTwister(9)
    for _ in 1:60, j in randperm(rng,length(runs))
        start = time_ns()
        for _ in 1:5
            runs[j]()
        end
        push!(samples[j],(time_ns()-start)/5e3)
    end
    return [(median(s),quantile(s,.25),quantile(s,.75),minimum(s),@allocated(runs[j]()))
        for (j,s) in enumerate(samples)]
end

function measure_case(query, llquery, built, columns, models, mapping, case, replicas, offset)
    u = pkpd_point(built.layout;case)
    shape_range = extrema(pkpd_gamma_shapes(built.layout,u,columns))
    actual = pkpd_stan_unconstrain(models[1],
        pkpd_stan_constrained(built.layout,u,columns,models[1].names))
    stanu = pkpd_stan_coordinates(u,mapping)
    @test actual ≈ stanu rtol=3e-13 atol=3e-13
    rkvalue = query(u)
    stanvalues = [pkpd_stan_density(model,stanu) for model in models]
    rkerror = abs(rkvalue-stanvalues[3]-offset)
    reference_change = abs(stanvalues[2]-stanvalues[3])
    @test rkerror < 2e-5replicas
    tp = pkpd_stan_parameters(models[3],stanu;include_gq=true)
    llstan = sum(tp["obs_likelihood.$i"] for i in eachindex(columns[:dv]))
    @test abs(llquery(u)-llstan) < 2e-5replicas
    sinks = [Ref(0.) for _ in 1:4]
    rkrun = () -> begin sinks[1][] = query(u); nothing end
    stanruns = map(1:3) do j
        model, sink = models[j], sinks[j+1]
        () -> begin sink[] = pkpd_stan_density(model,stanu); nothing end
    end
    measured = timings((rkrun,stanruns...))
    names = ("RK-primal","Stan-BDF-1e-6","Stan-BDF-1e-10","Stan-BDF-1e-12")
    errors = (rkerror,abs(stanvalues[1]-stanvalues[3]),reference_change,0.)
    rows = []
    for j in 1:4
        med,q25,q75,minval,allocation = measured[j]
        row = (case,length(columns[:sid]),length(columns[:dv]),length(columns[:damt]),
            built.layout.total,names[j],med,q25,q75,minval,allocation,sinks[j][],
            errors[j],reference_change,offset,shape_range...,join(u,','))
        push!(rows,row)
        @printf("case=%d subjects=%d %s median=%.3f us IQR=[%.3f,%.3f] Julia_bytes=%d abs_density_error=%.5g\n",
            case,row[2],names[j],med,q25,q75,allocation,errors[j])
    end
    println("production Stan / RK primal ratio=",measured[2][1]/measured[1][1],
        "; tight/reference density change=",reference_change)
    flush(stdout)
    return rows
end

function unit_diagnostic()
    ts = [0.,.01,.1,.5,1.,2.,5.,10.,24.,72.,168.]
    for shape in (1.2,8.)
        p = [.2,.15,.3,2.,shape]
        rule = ReactiveKernelsPPL.transit_twocmt_rule
        direct = ReactiveKernelsPPL.transit_twocmt_unit_response
        rulevalue, directvalue = rule(ts,p), direct(ts,p...)
        @test rulevalue ≈ directvalue rtol=3e-14 atol=1e-15
        sink1,sink2=Ref(rulevalue),Ref(directvalue)
        runs = (() -> begin sink1[]=rule(ts,p); nothing end,
            () -> begin sink2[]=direct(ts,p...); nothing end)
        measured = timings(runs)
        println("unit diagnostic shape=",shape," lags=",length(ts),
            " rule-primal (median_us,q25,q75,min,Julia_bytes)=",measured[1],
            " direct-primal=",measured[2])
    end
end

function main()
    println("Julia ",VERSION,"; CPU ",Sys.CPU_NAME,"; Julia/BLAS threads=1")
    println("Full unconstrained log density, constants and Jacobian included; no gradient calls.")
    println("10 warmups; 60 shuffled rounds of 5 calls; compilation and coordinate transport excluded.")
    println("Julia allocation counts exclude native C++ allocations.")
    @test all(p.name != "Enzyme" for p in keys(Base.loaded_modules))
    plan=pkpd_plan()
    offset=ReactiveKernelsPPL.lkj_logconst(13,2.)-sum(log,PKPD_SCALE0)-6log(.1)-1.5log(3.)
    open(OUTPUT,"w") do io
        println(io,"case\tsubjects\tobservations\tdoses\tcoordinates\tpath\tmedian_us\tq25_us\tq75_us\tmin_us\tJulia_allocated_bytes\tdensity\tabs_density_error_vs_reference\ttight_reference_density_change\texpected_density_offset\tshape_min\tshape_max\tRK_coordinates")
        for replicas in (1,10)
            columns=pkpd_columns(replicas)
            bound=bind_data(plan,pkpd_bound_columns(columns);dims=Dict(:kernel_nsub_loc=>length(columns[:sid])))
            built=build_kernel(bound)
            query=prepare_query(built,bound,:sampler)
            llquery=prepare_query(built,bound,:likelihood)
            models=[StanPKPD(joinpath(BUILD_ROOT,name*"_model.so"),pkpd_stan_data(columns))
                for name in ("varyingsource3","varyingsource3_tight","varyingsource3_reference")]
            try
                @test built.layout.total==models[1].n_unc
                mapping=pkpd_coordinate_map(built.layout,models[1])
                for case in 1:3
                    rows=Base.invokelatest(measure_case,query,llquery,built,columns,models,mapping,case,replicas,offset)
                    for row in rows
                        println(io,join(row,'\t'))
                    end
                    flush(io)
                end
            finally
                foreach(close,models)
            end
        end
    end
    unit_diagnostic()
    @test all(p.name != "Enzyme" for p in keys(Base.loaded_modules))
    println("PASS: all density/coordinate checks; Enzyme was not loaded.")
end
main()
