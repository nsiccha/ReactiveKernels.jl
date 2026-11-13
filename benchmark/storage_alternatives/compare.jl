# Run in a resolved environment with ordinary Enzyme and MutatingFunctions.
# Each arm owns its own caches. No production lowering or derivative rule is
# replaced. Raw early/late arms are diagnostics, not production candidates.
using ReactiveKernels, MutatingFunctions, Enzyme, DifferentiationInterface
using ReactiveKernelsDistributionKernels, BenchmarkTools, Statistics, TOML, SHA
include("prototypes.jl")
include("workloads.jl")
const RK = ReactiveKernels
const W = StorageWorkloads
const MODES = (:control, :early, :early_snapshot, :late, :late_snapshot)
const OUT = get(ENV,"STORAGE_RESULTS",joinpath(@__DIR__,"results"))
mkpath(OUT)

function prepare_arm(mode, p, args...)
    mode === :control && return prepare_nonallocating(prepare(p;bound_statistics=false),args...)
    mode in (:early,:early_snapshot) && return StorageAlternatives.early(p,args...;snapshot=mode===:early_snapshot)
    StorageAlternatives.late(p,args...;snapshot=mode===:late_snapshot)
end
function structure(ex)
    ex isa Expr || return (0,0)
    ex.head in (:quote,:inert) && return (0,0)
    n,l = 1, Int(ex.head in (:for,:while))
    for x in ex.args
        a,b = structure(x); n += a; l += b
    end
    n,l
end
const RESULTS = Dict{String,Any}(
    "julia"=>string(VERSION), "enzyme"=>string(pkgversion(Enzyme)),
    "enzyme_source"=>pathof(Enzyme),
    "libenzyme"=>Enzyme.Compiler.Enzyme_jll.libEnzyme_path,
    "source_sha256"=>Dict(basename(f)=>bytes2hex(sha256(read(f))) for f in
        [@__FILE__,joinpath(@__DIR__,"prototypes.jl"),joinpath(@__DIR__,"workloads.jl")]),
    "measurements"=>Any[], "failures"=>Any[], "snapshot_checks"=>Any[])
function save_results()
    open(joinpath(OUT,"results.toml"),"w") do io
        TOML.print(io,RESULTS;sorted=true)
    end
end
function attempt(f, label)
    println((;label,stage="start")); flush(stdout)
    try
        f()
        println((;label,stage="complete")); flush(stdout)
    catch e
        message=sprint(showerror,e,catch_backtrace())
        push!(RESULTS["failures"],Dict("label"=>label,"error"=>message))
        println((;label,stage="failed",error=message)); flush(stdout)
    finally
        save_results()
    end
end

function one_workload(kind, n)
    spec=getfield(W,kind); x=[0.2sin(i) for i in 1:n]; nu=3.0
    p=plan(spec); expected,slope=W.reference(kind,x,nu)
    arms=Dict{Symbol,Any}()
    for mode in MODES
        attempt("$kind/$n/$mode") do
            prep=@timed prepare_arm(mode,p,x,nu)
            k=prep.value
            original=copy(x)
            value=only(k(x,nu))
            isapprox(value,expected;rtol=2e-12,atol=2e-12) || error("primal mismatch")
            x==original || error("input mutation")
            adprep=@timed prepare_ad(k,AutoEnzyme(;mode=Enzyme.Reverse),x,nu;active=:u)
            ad=adprep.value; dx=similar(x)
            firstcall=@timed ad_value_and_gradient!(ad,dx,x,nu)
            isapprox(firstcall.value[1],expected;rtol=2e-12,atol=2e-12) || error("AD value mismatch")
            isapprox(dx,slope;rtol=2e-12,atol=2e-12) || error("gradient mismatch")
            point,contexts=RK._ad_prepared_arguments(ad,(x,nu),NamedTuple())
            audit=RK._code_findings(ad.call,RK._ad_call_argument_types(point,contexts);
                recurse=RK._code_recursion(Val(:all)))
            ast=RK._inline_plate_programs(k.ast)
            nodes,loops=structure(ast)
            write(joinpath(OUT,"$kind-$n-$mode.jl"),sprint(show,ast))
            row=Dict{String,Any}("workload"=>string(kind),"n"=>n,"mode"=>string(mode),
                "prepare_s"=>prep.time,"prepare_bytes"=>prep.bytes,
                "ad_prepare_s"=>adprep.time,"first_gradient_s"=>firstcall.time,
                "nodes"=>nodes,"loops"=>loops,"audit_findings"=>length(audit),
                "audit"=>string.(audit),"semantic_pass"=>true,"rounds"=>Any[])
            push!(RESULTS["measurements"],row)
            arms[mode]=(;k,ad,dx,row)
            # Same shape, changed runtime values; nothing is frozen to the exemplar.
            x2=x .+ 0.07; nu2=4.2; y2,d2=W.reference(kind,x2,nu2)
            v2,_=ad_value_and_gradient!(ad,dx,x2,nu2)
            isapprox(v2,y2;rtol=2e-12,atol=2e-12) || error("changed-value primal mismatch")
            isapprox(dx,d2;rtol=2e-12,atol=2e-12) || error("changed-value gradient mismatch")
        end
    end
    # Rotate warm timing order. Cold timings above share a process/compiler and
    # must not be interpreted as unbiased process-start compilation comparisons.
    for round in 1:3, mode in circshift(collect(MODES),round-1)
        haskey(arms,mode) || continue
        (;k,ad,dx,row)=arms[mode]
        k(x,nu); ad_value_and_gradient!(ad,dx,x,nu)
        primal=@benchmark $k($x,$nu) samples=100 evals=1 seconds=0.1
        grad=@benchmark ad_value_and_gradient!($ad,$dx,$x,$nu) samples=100 evals=1 seconds=0.1
        a,b=median(primal),median(grad)
        push!(row["rounds"],Dict("round"=>round,"primal_ns"=>a.time,
            "primal_bytes"=>a.memory,"primal_allocs"=>a.allocs,
            "gradient_ns"=>b.time,"gradient_bytes"=>b.memory,"gradient_allocs"=>b.allocs))
        println((;kind,n,mode,round,gradient_ns=b.time,bytes=b.memory)); flush(stdout)
    end
    save_results()
end

function snapshot_check(kind,mode)
    x=[0.1,0.2,0.3]; data=[0.3,0.4,0.5]; nu=3.0
    original=copy(x)
    peer=kind===:readonly ? W.ReadOnlyPeer(data) : kind===:global ? W.GlobalMutatingPeer(data) : W.MutatingPeer(data,x)
    spec=kind===:owned ? W.owned_peer : W.peer
    args=kind===:owned ? (x,data,nu) : (x,peer,nu)
    W.peer_victim[]=x
    p=plan(spec)
    k=prepare_arm(mode,p,args...)
    copyto!(x,original) # preparation may execute an exemplar in future variants
    value=only(k(args...))
    expected=first(W.reference(:stored,original,nu))+sum(data)
    preserved=isapprox(value,expected;rtol=2e-12,atol=2e-12)
    effects=StorageAlternatives.effects_gate(p,map(typeof,args))
    push!(RESULTS["snapshot_checks"],Dict("kind"=>string(kind),"mode"=>string(mode),
        "value"=>value,"expected"=>expected,"preserved"=>preserved,
        "effects_gate"=>effects.accepted,"effect_reports"=>string.(effects.reports)))
    println((;kind,mode,preserved,effects_gate=effects.accepted)); flush(stdout)
end

with_plate_reverse(false) do
    kinds=Symbol.(split(get(ENV,"STORAGE_CASES","stored,explicit,retained,logarithmic"),','))
    sizes=parse.(Int,split(get(ENV,"STORAGE_SIZES","0,1,17,1000"),','))
    for kind in kinds, n in sizes
        attempt("$kind/$n") do
            one_workload(kind,n)
        end
    end
    if get(ENV,"STORAGE_SNAPSHOTS","1")=="1"
        for kind in (:readonly,:mutating,:global,:owned), mode in MODES
            attempt("snapshot/$kind/$mode") do
                snapshot_check(kind,mode)
            end
        end
    end
end
println("STORAGE_COMPARISON_COMPLETE failures=",length(RESULTS["failures"]))
