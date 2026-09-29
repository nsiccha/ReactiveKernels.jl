using DifferentiationInterface: AutoEnzyme, Constant, gradient
using LinearAlgebra: BLAS
using Printf, Random, Statistics, Test
import Enzyme
include("model.jl")
include("stan.jl")
BLAS.set_num_threads(1)

relative_error(g,reference) = maximum(abs.(g-reference))/maximum(abs,reference)
ast_nodes(ex) = ex isa Expr ? 1+sum(ast_nodes,ex.args;init=0) : 1
function code_nodes(ex)
    ex isa Expr || return 1
    # The integer level table is data, folded with its bound grouping column
    # during preparation. Count it once; keep every executable subtree counted.
    if ex.head === :call && length(ex.args)==3 && ex.args[1]===:_declared_codes
        levels = ex.args[3]
        if levels isa Expr && levels.head === :vect && all(x -> x isa Integer,levels.args)
            return 2+sum(code_nodes,ex.args[1:2];init=0)
        end
    end
    return 1+sum(code_nodes,ex.args;init=0)
end
ast_calls(ex,name) = ex isa Expr ?
    Int(ex.head===:call && !isempty(ex.args) && ex.args[1]===name)+
        sum(a -> ast_calls(a,name),ex.args;init=0) : 0

function main(build_root,output)
    println("Julia ",VERSION,"; Enzyme ",pkgversion(Enzyme),"; CPU ",Sys.CPU_NAME,
        "; Julia/BLAS threads=1; series_rtol=1e-15, watson_terms=8")
    println("Times include the full density and reverse gradient in each engine's coordinates.")
    println("Stan coordinate transport is checked and excluded from its timings.")
    plan = pkpd_plan()
    backend = AutoEnzyme(;mode=Enzyme.Reverse)
    rows = []
    structural_size = nothing
    expected_offset = ReactiveKernelsPPL.lkj_logconst(13,2.)-
        sum(log,PKPD_SCALE0)-6log(.1)-1.5log(3.)
    for replicas in (1,10)
        columns = pkpd_columns(replicas)
        nsubjects = length(columns[:sid])
        bound = bind_data(plan,pkpd_bound_columns(columns);dims=Dict(:kernel_nsub_loc=>nsubjects))
        built = build_kernel(bound)
        expression = kernel_expr(bound,built.layout)
        @test ast_calls(expression,:varyingsource_pkpd_read_locs_over_subjects)==1
        @test ast_calls(expression,:_centered_correlated_logpdf)==1
        println("expression nodes=",ast_nodes(expression),
            "; code nodes excluding bound grouping-level data=",code_nodes(expression))
        if structural_size === nothing
            structural_size = code_nodes(expression)
        else
            @test code_nodes(expression)==structural_size
        end
        query = prepare_query(built,bound,:sampler)
        llquery = prepare_query(built,bound,:likelihood)
        models = [StanPKPD(joinpath(build_root,name*"_model.so"),pkpd_stan_data(columns))
            for name in ("varyingsource3","varyingsource3_tight","varyingsource3_reference")]
        try
            info = ccall(Libdl.dlsym(models[1].library,:bs_model_info),Cstring,
                (Ptr{Cvoid},),models[1].model)
            println(unsafe_string(info))
            @test built.layout.total == models[1].n_unc
            mapping = pkpd_coordinate_map(built.layout,models[1])
            println("subjects=",nsubjects," observations=",length(columns[:dv]),
                " doses=",length(columns[:damt])," coordinates=",built.layout.total)
            sampler = nothing
            for case in 1:3
                u = pkpd_point(built.layout;case)
                shape_range = extrema(pkpd_gamma_shapes(built.layout,u,columns))
                println("case=",case," selected-source Gamma shape range=",shape_range)
                case==3 && @test 5 < shape_range[1] < 8 < shape_range[2] < 12
                actual = pkpd_stan_unconstrain(models[1],
                    pkpd_stan_constrained(built.layout,u,columns,models[1].names))
                stanu = pkpd_stan_coordinates(u,mapping)
                @test actual ≈ stanu rtol=3e-13 atol=3e-13
                values,transported = Float64[],Vector{Float64}[]
                for model in models
                    value,g = pkpd_stan_gradient(model,stanu)
                    push!(values,value)
                    push!(transported,gradient(pkpd_map_pullback_objective,backend,u,
                        Constant(mapping),Constant(g)))
                end
                if sampler === nothing
                    sampler = prepare_sampler(built,bound,u;backend)
                end
                grk,vrk = zeros(length(u)),Ref(0.)
                rkrun = () -> begin
                    value,_ = sampler_value_and_gradient!(sampler,grk,u)
                    vrk[] = value
                    nothing
                end
                rkrun()
                reference = transported[3]
                stan_error = relative_error(transported[1],reference)
                convergence = relative_error(transported[2],reference)
                rk_error = relative_error(grk,reference)
                @test convergence < max(stan_error/10,1e-10)
                @test grk ≈ reference rtol=1e-5 atol=2e-4
                @test abs(vrk[]-values[3]-expected_offset) < 2e-5replicas
                @test vrk[] ≈ Base.invokelatest(query,u) rtol=2e-13
                tp = pkpd_stan_parameters(models[3],stanu;include_gq=true)
                llstan = sum(tp["obs_likelihood.$i"] for i in eachindex(columns[:dv]))
                @test abs(Base.invokelatest(llquery,u)-llstan) < 2e-5replicas
                println("case=",case," tight convergence=",convergence," RK/reference=",rk_error,
                    " production Stan/reference=",stan_error," density delta after constant=",
                    vrk[]-values[3]-expected_offset)
                stanvalue,stangrad = Ref(0.),zeros(models[1].n_unc)
                err = Ref{Ptr{Cchar}}(C_NULL)
                stanrun = () -> begin
                    pkpd_stan_gradient!(stanvalue,stangrad,models[1],stanu,err)
                    nothing
                end
                runs = (("RK-emitted-Enzyme",rkrun),("Stan-BDF",stanrun))
                for _ in 1:10, (_,run) in runs
                    run()
                end
                GC.gc()
                samples = [Float64[],Float64[]]
                rng = MersenneTwister(9)
                for _ in 1:60, j in randperm(rng,2)
                    start = time_ns()
                    for _ in 1:5
                        runs[j][2]()
                    end
                    push!(samples[j],(time_ns()-start)/5e3)
                end
                for j in 1:2
                    push!(rows,(case,nsubjects,length(columns[:dv]),length(columns[:damt]),
                        built.layout.total,runs[j][1],median(samples[j]),
                        j==1 ? rk_error : stan_error,convergence,
                        j==1 ? rk_error<=stan_error : true,@allocated(runs[j][2]()),
                        j==1 ? vrk[] : stanvalue[],expected_offset,
                        join(u,','),join(j==1 ? grk : transported[1],','),join(reference,',')))
                    @printf("case=%d subjects=%d %s median=%.2f μs matched=%s\n",
                        case,nsubjects,runs[j][1],rows[end][7],rows[end][10])
                end
            end
        finally
            foreach(close,models)
        end
    end
    open(output,"w") do io
        println(io,"case\tsubjects\tobservations\tdoses\tcoordinates\tpath\tmedian_us\tgradient_rel_error\treference_convergence\taccuracy_matched\tJulia_allocated_bytes\tdensity\texpected_density_offset\tRK_coordinates\tRK_gradient\treference_RK_gradient")
        for row in rows
            println(io,join(row,'\t'))
        end
    end
end

length(ARGS)==2 || error("Usage: gradients.jl <build-dir> <results.tsv>")
main(ARGS...)
