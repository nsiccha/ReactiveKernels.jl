using Test, ReactiveKernels, ReactiveKernelsPPL, Enzyme, DifferentiationInterface, Reactant
import Distributions as D
Reactant.set_default_backend("cpu")
function _restricted_executable_inventory(hlo)
    matches=collect(eachmatch(r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\(",hlo))
    @test !isempty(matches)
    @test length(matches)==length(collect(eachmatch(r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = ",hlo)))
    counts=Dict{String,Int}()
    for m in matches
        op=m.captures[1];counts[op]=get(counts,op,0)+1
    end
    counts
end
function _restricted_mlir_inventory(text)
    counts=Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|tensor|scf)\.[a-zA-Z_]+\b", text)
        counts[m.match]=get(counts,m.match,0)+1
    end
    @test !isempty(counts)
    counts
end
function _restricted_measure(expr,data,q,label)
    p=bind_data(lower_rkppl(expr,data;conditioned=haskey(data,:y) ? (:y,) : ()),data)
    b=build_kernel(p); u=unconstrain(b.layout,q)
    sampler=prepare_sampler(b,p,u;backend=AutoEnzyme(;mode=Enzyme.Reverse))
    native,ng=sampler_value_and_gradient!(sampler,similar(u),u)
    return Base.invokelatest(_restricted_measure_prepared,sampler,u,native,ng,label)
end
function _restricted_measure_prepared(sampler,u,native,ng,label)
    ru=Reactant.to_rarray(u);kernel=sampler.kernel
    primal=Reactant.@compile kernel(ru)
    reverse=compile_ad_value_and_gradient(sampler.ad,ru)
    @test Float64(primal(ru)) ≈ native atol=1e-11
    v,g=reverse(ru)
    @test Float64(v) ≈ native atol=1e-11
    @test Array(g) ≈ ng atol=1e-10
    # Match compile_ad_value_and_gradient's core selector and standard DI backend.
    # Capture the immutable callable, rather than the native DI preparation.
    both=let call=ReactiveKernels._ADKernelCall{1,typeof(kernel)}(kernel), backend=sampler.ad.backend
        x -> DifferentiationInterface.value_and_gradient(call,backend,x)
    end
    result=map((("primal",kernel,primal),("reverse",both,reverse))) do (mode,fn,compiled)
        mlir=repr(Reactant.@code_hlo fn(ru))
        hlo=repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
        if haskey(ENV,"RKPPL_RESTRICTED_IR_DIR")
            write(joinpath(ENV["RKPPL_RESTRICTED_IR_DIR"],"$label-$mode.mlir"),mlir)
            write(joinpath(ENV["RKPPL_RESTRICTED_IR_DIR"],"$label-$mode.hlo"),hlo)
        end
        (;mlir=_restricted_mlir_inventory(mlir),executable=_restricted_executable_inventory(hlo))
    end
    println("COMPILED ",label," ",result);flush(stdout)
    result
end
@testset "restricted scalar support: default compiled values, reverse and size structure" begin
    for (family,ctor) in (("normal",:(Normal(a,1+exp(a)))),
            ("exponential",:(Exponential(1+exp(a)))),("flat",:(Flat())))
        structures=[]
        for n in (3,9)
            data=(;limits=collect(range(-0.5,1.5;length=n)))
            expr=quote
                x ~ restricted($ctor, minimum(limits) + a, maximum(limits) + exp(a))
                a ~ Normal(0,1)
            end
            push!(structures,_restricted_measure(expr,data,(;x=0.5,a=0.1),"$family-$n"))
        end
        @test structures[1]==structures[2]
    end
end

@testset "restricted arrays: default compiled values, reverse and size structure" begin
    structures=[]
    for n in (3,9)
        data=(;rows=zeros(n))
        expr=quote
            a ~ Normal(0,1)
            hi=2+exp(a)
            x[axes(rows,1)] .~ restricted.(Normal.(0,1), a, hi)
        end
        push!(structures,_restricted_measure(expr,data,(;a=0.2,x=fill(0.4,n)),"array-$n"))
    end
    @test structures[1]==structures[2]
end
