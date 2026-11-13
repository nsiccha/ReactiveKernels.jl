# Unchanged natural models from the supplied private evidence archive.
# ARGS: extracted archive root, extract_natural.py output, result directory.
using ReactiveKernels, ReactiveKernelsPPL, ReactiveKernelsDistributionKernels
using MutatingFunctions, Enzyme, DifferentiationInterface, BenchmarkTools, Statistics, TOML, SHA
include("prototypes.jl")
const RK = ReactiveKernels
const CONTEXT, INPUTS, OUT = abspath.(ARGS[1:3])
mkpath(OUT)
arg(args,name)=args[Symbol(name)]
function port_spellings end
function port_point end
decode(x)=x
include(joinpath(CONTEXT,"ports","classical_shrinkage_ports.jl"))
const MODES=(:control,:early,:early_snapshot,:late,:late_snapshot)
const RESULTS=Dict{String,Any}("julia"=>string(VERSION),"enzyme"=>string(pkgversion(Enzyme)),
    "enzyme_source"=>pathof(Enzyme),"libenzyme"=>Enzyme.Compiler.Enzyme_jll.libEnzyme_path,
    "input_sha256"=>strip(read(joinpath(INPUTS,"source.sha256"),String)),
    "model_source_sha256"=>bytes2hex(sha256(read(joinpath(CONTEXT,"ports","classical_shrinkage_ports.jl")))),
    "source_sha256"=>Dict(basename(f)=>bytes2hex(sha256(read(f))) for f in
        [@__FILE__,joinpath(@__DIR__,"prototypes.jl")]),
    "measurements"=>Any[],"failures"=>Any[])
readfloats(path,n)=open(io->read!(io,Vector{Float64}(undef,n)),path)
function save_results()
    open(joinpath(OUT,"results.toml"),"w") do io
        TOML.print(io,RESULTS;sorted=true)
    end
end
function run_model(name)
    dir=joinpath(INPUTS,name)
    n,d,np=parse.(Int,readlines(joinpath(dir,"shape.txt")))
    settings=readfloats(joinpath(dir,"settings.bin"),6)
    data=(;x=reshape(readfloats(joinpath(dir,"x.bin"),n*d),n,d),
        y=Int.(readfloats(joinpath(dir,"y.bin"),n)),
        scale_icept=settings[1],scale_global=settings[2],nu_global=settings[3],
        nu_local=settings[4],slab_scale=settings[5],slab_df=settings[6])
    _,factory=only(port_spellings(Val(Symbol(name)),data))
    modelplan=factory()
    build=@timed build_kernel(modelplan)
    built=build.value
    names=string.(coordinate_names(built.layout))
    positions=Dict(v=>i for (i,v) in enumerate(names))
    perm=[positions[v] for v in readlines(joinpath(dir,"coordinates.txt"))]
    length(unique(perm))==length(names) || error("coordinate map is not bijective")
    points=[let v=readfloats(joinpath(dir,"point-$i.bin"),3+2d)
        Float64.(unconstrain(built.layout,(;beta0=v[1],tau=v[2],caux=exp(v[3]),z=v[4:3+d],lambda=v[4+d:end])))
    end for i in 1:np]
    refs=[readfloats(joinpath(dir,"reference-$i.bin"),length(names)+1) for i in 1:np]
    ordinary=RK._kernel_latest_prepared(prepare_query(built,modelplan,:sampler;exemplars=(first(points),)))
    p=ordinary.plan; u=first(points)
    arms=Dict{Symbol,Any}()
    for mode in MODES
        println((;name,mode,stage="prepare"));flush(stdout)
        try
            prep=@timed if mode===:control
                prepare_nonallocating(ordinary,u)
            elseif mode in (:early,:early_snapshot)
                StorageAlternatives.early(p,u;snapshot=mode===:early_snapshot)
            else
                StorageAlternatives.late(p,u;snapshot=mode===:late_snapshot)
            end
            k=prep.value
            adprep=@timed prepare_ad(k,AutoEnzyme(;mode=Enzyme.Reverse),u;active=:unconstrained)
            ad=adprep.value;g=similar(u)
            firstcall=@timed ad_value_and_gradient!(ad,g,u)
            offsets=Float64[];errors=Float64[]
            for (point,ref) in zip(points,refs)
                before=copy(point)
                value,_=ad_value_and_gradient!(ad,g,point)
                isapprox(only(k(point)),value;rtol=1e-10,atol=1e-10) || error("primal/AD disagreement")
                point==before || error("input mutation")
                push!(offsets,value-ref[1])
                push!(errors,maximum(abs.(g[perm].-ref[2:end])./max.(1.0,abs.(g[perm]),abs.(ref[2:end]))))
            end
            maximum(errors)<=1e-8 || error("frozen-reference gradient mismatch: $(maximum(errors))")
            spread=maximum(abs.(offsets.-first(offsets)))
            spread<=1e-8*max(1.0,abs(first(offsets))) || error("density offset is not constant: $spread")
            point,contexts=RK._ad_prepared_arguments(ad,(u,),NamedTuple())
            audit=RK._code_findings(ad.call,RK._ad_call_argument_types(point,contexts);
                recurse=RK._code_recursion(Val(:all)))
            isempty(audit) || error("type audit findings: $audit")
            write(joinpath(OUT,"$name-$mode.jl"),sprint(show,code_expr(k)))
            write(joinpath(OUT,"$name-$mode-structure.jl"),sprint(show,RK._inline_plate_programs(k.ast)))
            row=Dict{String,Any}("model"=>name,"mode"=>string(mode),"dimension"=>length(u),
                "build_s"=>build.time,"prepare_s"=>prep.time,"ad_prepare_s"=>adprep.time,
                "build_bytes"=>build.bytes,"prepare_bytes"=>prep.bytes,
                "ad_prepare_bytes"=>adprep.bytes,"first_gradient_bytes"=>firstcall.bytes,
                "audit_findings"=>length(audit),
                "first_gradient_s"=>firstcall.time,"max_gradient_error"=>maximum(errors),
                "density_offset_spread"=>spread,"rounds"=>Any[])
            push!(RESULTS["measurements"],row)
            arms[mode]=(;k,ad,g,row)
            println((;name,mode,stage="validated",gradient_error=maximum(errors)));flush(stdout)
        catch e
            message=sprint(showerror,e,catch_backtrace())
            push!(RESULTS["failures"],Dict("model"=>name,"mode"=>string(mode),"error"=>message))
            println((;name,mode,stage="failed",error=message));flush(stdout)
        finally
            save_results()
        end
    end
    for round in 1:3,mode in circshift(collect(MODES),round-1)
        haskey(arms,mode) || continue
        (;k,ad,g,row)=arms[mode]
        k(u);ad_value_and_gradient!(ad,g,u)
        a=median(@benchmark $k($u) samples=50 evals=1 seconds=0.2)
        b=median(@benchmark ad_value_and_gradient!($ad,$g,$u) samples=50 evals=1 seconds=0.2)
        push!(row["rounds"],Dict("round"=>round,"primal_ns"=>a.time,"primal_bytes"=>a.memory,
            "gradient_ns"=>b.time,"gradient_bytes"=>b.memory,"gradient_allocs"=>b.allocs))
        println((;name,mode,round,gradient_ns=b.time,bytes=b.memory));flush(stdout)
    end
    save_results()
end
with_plate_reverse(false) do
    for name in ("LogisticRHSOvarian","LogisticRHSProstate")
        run_model(name)
    end
end
println("NATURAL_STORAGE_COMPARISON_COMPLETE failures=",length(RESULTS["failures"]))
