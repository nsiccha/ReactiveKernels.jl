# One arm per fresh Julia process. Package loading is outside the measured
# stages; planning, preparation, first primal, AD preparation and first reverse
# call are separate observations. Run modes in a rotated order across rounds.
using ReactiveKernels, MutatingFunctions, Enzyme, DifferentiationInterface
using ReactiveKernelsDistributionKernels, TOML, SHA
include("prototypes.jl")
include("workloads.jl")
const MODE=Symbol(ARGS[1])
const OUT=abspath(ARGS[2])
MODE in (:control,:early,:early_snapshot,:late,:late_snapshot) || error("unknown mode")
const row=Dict{String,Any}("mode"=>string(MODE),"workload"=>"stored","n"=>1000,
    "julia"=>string(VERSION),"enzyme"=>string(pkgversion(Enzyme)),
    "enzyme_source"=>pathof(Enzyme),"libenzyme"=>Enzyme.Compiler.Enzyme_jll.libEnzyme_path,
    "prototype_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"prototypes.jl")))))
function stage(f,name)
    t=@timed f()
    row[name*"_s"]=t.time
    row[name*"_bytes"]=t.bytes
    row[name*"_gc_s"]=t.gctime
    t.value
end
with_plate_reverse(false) do
    x=[0.2sin(i) for i in 1:1000];nu=3.0
    p=stage("plan") do
        plan(StorageWorkloads.stored)
    end
    k=stage("prepare") do
        if MODE===:control
            prepare_nonallocating(prepare(p;bound_statistics=false),x,nu)
        elseif MODE in (:early,:early_snapshot)
            StorageAlternatives.early(p,x,nu;snapshot=MODE===:early_snapshot)
        else
            StorageAlternatives.late(p,x,nu;snapshot=MODE===:late_snapshot)
        end
    end
    primal=stage(() -> only(k(x,nu)),"first_primal")
    ad=stage("ad_prepare") do
        prepare_ad(k,AutoEnzyme(;mode=Enzyme.Reverse),x,nu;active=:u)
    end
    dx=similar(x)
    value,_=stage(() -> ad_value_and_gradient!(ad,dx,x,nu),"first_gradient")
    expected,slope=StorageWorkloads.reference(:stored,x,nu)
    isapprox(primal,expected;rtol=2e-12,atol=2e-12) || error("primal mismatch")
    isapprox(value,expected;rtol=2e-12,atol=2e-12) || error("AD value mismatch")
    isapprox(dx,slope;rtol=2e-12,atol=2e-12) || error("gradient mismatch")
    row["semantic_pass"]=true
end
mkpath(dirname(OUT))
open(io -> TOML.print(io,row;sorted=true),OUT,"w")
println("COLD_STORAGE_COMPLETE ",row)
