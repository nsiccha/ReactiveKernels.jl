using ReactiveKernels, MutatingFunctions, Enzyme, DifferentiationInterface
using ReactiveKernelsDistributionKernels, TOML, SHA
include("prototypes.jl")
include("workloads.jl")
const RK=ReactiveKernels
const W=StorageWorkloads
const MODES=(:control,:early,:early_snapshot,:late,:late_snapshot)
const OUT=abspath(only(ARGS))
mkpath(OUT)
const RESULTS=Dict{String,Any}("cases"=>Any[],"failures"=>Any[],
    "julia"=>string(VERSION),"enzyme"=>string(pkgversion(Enzyme)),
    "libenzyme"=>Enzyme.Compiler.Enzyme_jll.libEnzyme_path,
    "prototype_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"prototypes.jl")))))

@kernel guarded(u::Vector{Float64}, enabled::Bool, index::Int) = begin
    w=exp.(u)
    terms=log.(w)
    extra=enabled ? log(u[index]) : 0.0
    total=sum(terms)+extra
    return total
end
@kernel ragged(u::Vector{Float64}, groups::Vector{Vector{Int}}) = begin
    w=exp.(u)
    pieces=plate(groups) do ids
        sum(log.(w[ids]))
    end
    total=sum(pieces)
    return total
end
function prepare_arm(mode,p,args...)
    mode===:control && return prepare_nonallocating(prepare(p;bound_statistics=false),args...)
    mode in (:early,:early_snapshot) && return StorageAlternatives.early(p,args...;snapshot=mode===:early_snapshot)
    StorageAlternatives.late(p,args...;snapshot=mode===:late_snapshot)
end
function structure(ex)
    ex isa Expr || return (0,0)
    ex.head in (:quote,:inert) && return (0,0)
    n,l=1,Int(ex.head in (:for,:while))
    for a in ex.args
        nn,ll=structure(a);n+=nn;l+=ll
    end
    n,l
end
function attempt(f,label)
    println((;label,stage="start"));flush(stdout)
    try
        row=f()
        row["label"]=label
        push!(RESULTS["cases"],row)
        println((;label,stage="complete"));flush(stdout)
    catch e
        message=sprint(showerror,e,catch_backtrace())
        push!(RESULTS["failures"],Dict("label"=>label,"error"=>message))
        println((;label,stage="failed",error=message));flush(stdout)
    finally
        open(joinpath(OUT,"results.toml"),"w") do io
            TOML.print(io,RESULTS;sorted=true)
        end
    end
end
function check_gradient(k,args,expected,slope)
    ad=prepare_ad(k,AutoEnzyme(;mode=Enzyme.Reverse),args...;active=:u)
    d=similar(first(args));before=deepcopy(args)
    value,_=ad_value_and_gradient!(ad,d,args...)
    isapprox(only(k(args...)),expected;rtol=3e-12,atol=3e-12) || error("primal mismatch")
    isapprox(value,expected;rtol=3e-12,atol=3e-12) || error("AD value mismatch")
    isapprox(d,slope;rtol=3e-12,atol=3e-12) || error("gradient mismatch")
    isequal(args,before) || error("input changed")
    point,contexts=RK._ad_prepared_arguments(ad,args,NamedTuple())
    audit=RK._code_findings(ad.call,RK._ad_call_argument_types(point,contexts);
        recurse=RK._code_recursion(Val(:all)))
    isempty(audit) || error("type audit findings: $audit")
    nodes,loops=structure(RK._inline_plate_programs(k.ast))
    Dict{String,Any}("value"=>value,"nodes"=>nodes,"loops"=>loops,"audit_findings"=>0)
end

with_plate_reverse(false) do
    for mode in MODES
        attempt("tails/$mode") do
            x=[-700.0,-350.0,-20.0,0.0,20.0,350.0,700.0];nu=3.0
            # High precision evaluates the same mathematics without Float64
            # overflow; it is an oracle, never a derivative implementation.
            expected=Float64(sum(z-(big(nu)+1)/2*log1p(exp(2z)/big(nu)) for z in BigFloat.(x)))
            slope=[Float64(1-(big(nu)+1)*exp(2z)/(big(nu)+exp(2z))) for z in BigFloat.(x)]
            k=prepare_arm(mode,plan(W.logarithmic),x,nu)
            check_gradient(k,(x,nu),expected,slope)
        end
        for n in (0,17,1000)
            attempt("returned/$n/$mode") do
                x=[0.1sin(i) for i in 1:n];nu=3.0;before=copy(x)
                k=prepare_arm(mode,plan(W.returned),x,nu)
                w,total=k(x,nu)
                isapprox(w,exp.(x);rtol=1e-12,atol=1e-12) || error("returned array mismatch")
                expected=sum(@. -(nu+1)/2*log1p(exp(2x)/nu))
                isapprox(total,expected;rtol=1e-12,atol=1e-12) || error("total mismatch")
                # An exposed output must not borrow its caller's array. This
                # does not assert persistence across calls of a reusable kernel.
                fill!(w,8.0)
                x==before || error("returned output aliases input")
                nodes,loops=structure(RK._inline_plate_programs(k.ast))
                Dict{String,Any}("nodes"=>nodes,"loops"=>loops)
            end
            attempt("inactive-guard/$n/$mode") do
                x=fill(0.4,n);index=n+5
                k=prepare_arm(mode,plan(guarded),x,false,index)
                check_gradient(k,(x,false,index),sum(x),ones(n))
            end
            attempt("ragged/$n/$mode") do
                x=[0.1sin(i) for i in 1:n]
                groups=n==0 ? Vector{Int}[] : [isodd(i) ? Int[] : [1,mod1(i,n),1] for i in 1:n]
                slope=zeros(n)
                for ids in groups, index in ids
                    slope[index]+=1
                end
                expected=sum(x.*slope)
                k=prepare_arm(mode,plan(ragged),x,groups)
                check_gradient(k,(x,groups),expected,slope)
            end
        end
        attempt("active-guard/$mode") do
            x=[0.4,0.7];index=2
            k=prepare_arm(mode,plan(guarded),x,true,index)
            row=check_gradient(k,(x,true,index),sum(x)+log(x[index]),[1.0,1+1/x[index]])
            for (input,i,exception) in ((x,20,BoundsError),([-0.4,0.7],1,DomainError))
                thrown=try
                    k(input,true,i);nothing
                catch e
                    e
                end
                thrown isa exception || error("required exception was not preserved: $thrown")
            end
            row
        end
    end
end
println("STORAGE_EDGES_COMPLETE failures=",length(RESULTS["failures"]))
