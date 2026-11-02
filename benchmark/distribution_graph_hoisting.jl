#!/usr/bin/env julia

# Compare the owned scalar-family graph with a per-cell Distributions call.
# Preparation is excluded. The compiler may optimize either implementation;
# timings report the outcome rather than assuming every opaque call repeats work.
# Run in an environment containing RK, DistributionKernels and Distributions:
#   julia --project=<environment> benchmark/distribution_graph_hoisting.jl <output-dir> [family]
module DistributionGraphHoisting
using ReactiveKernels, ReactiveKernelsDistributionKernels, Statistics, TOML
using SHA
import Distributions
import ReactiveKernelsDistributionKernels.DistributionKernelSources as DKS

const FAMILIES = ((:Normal,:normal,[0.3,1.2]), (:LogNormal,:lognormal,[0.3,1.2]),
    (:Gamma,:gamma,[2.3,1.2]), (:Exponential,:exponential,[1.2,0.0]),
    (:Cauchy,:cauchy,[0.3,1.2]), (:Laplace,:laplace,[0.3,1.2]),
    (:Logistic,:logistic,[0.3,1.2]))

for (family,object,_) in FAMILIES
    owned = Symbol(family,:_owned)
    opaque = Symbol(family,:_opaque)
    callback = Symbol(family,:_density)
    constructor = GlobalRef(Distributions,family)
    graph = GlobalRef(DKS,object)
    law = family === :Gamma ? :($graph(;shape=q[1],scale=q[2])) :
        family === :Exponential ? :($graph(q[1])) : :($graph(q[1],q[2]))
    args = family === :Exponential ? [:(q[1])] : [:(q[1]),:(q[2])]
    @eval begin
        $callback(x,q) = Distributions.logpdf($constructor($(args...)),x)
        @kernel $owned(xs::Vector{Float64},q::Vector{Float64}) = begin
            scores = plate(xs,Ref(q)) do x,q
                $law.logpdf(x)
            end
            return sum(scores)
        end
        @kernel $opaque(xs::Vector{Float64},q::Vector{Float64}) = begin
            scores = plate(xs,Ref(q)) do x,q
                $callback(x,q)
            end
            return sum(scores)
        end
    end
end

function measure(f,q;repetitions=200)
    f(q); f(q)
    samples = Float64[]
    active = copy(q)
    checksum = 0.0
    for round in 1:9
        elapsed = @elapsed begin
            for i in 1:repetitions
                # Consume every result and change an input, so a pure evaluator
                # cannot be deleted or hoisted outside the repetition loop.
                active[1] = q[1] + 1e-6*(i + repetitions*(round-1))
                checksum += f(active)
            end
        end
        push!(samples,elapsed/repetitions)
    end
    @assert isfinite(checksum)
    (; seconds=median(samples),allocated_bytes=@allocated(f(q)),checksum)
end

function run(output_dir;families=FAMILIES)
    mkpath(output_dir)
    rows = Dict{String,Any}[]
    for (family,_,q) in families, n in (1,64,1024)
        xs = [0.1 + 0.03mod(i,31) for i in 1:n]
        saved = copy(xs)
        owned = prepare(getfield(@__MODULE__,Symbol(family,:_owned));bound=(;xs))
        opaque = prepare(getfield(@__MODULE__,Symbol(family,:_opaque));bound=(;xs))
        for active in (q,q.+0.1)
            @assert isapprox(owned(active),opaque(active);rtol=2e-12,atol=1e-12)
        end
        @assert xs == saved
        checksums = Float64[]
        for (name,f) in (("owned",owned),("opaque",opaque))
            code_file = joinpath(output_dir,"$(family)-$(n)-$(name).jl")
            open(code_file,"w") do io
                show(io,code_expr(f))
                println(io)
            end
            measured = measure(f,q)
            push!(checksums,measured.checksum)
            row = Dict{String,Any}("family"=>string(family),"observations"=>n,
                "implementation"=>name,"seconds"=>measured.seconds,
                "allocated_bytes"=>measured.allocated_bytes,"code"=>code_file,
                "checksum"=>measured.checksum)
            push!(rows,row)
            println("HOISTING ",family," n=",n," implementation=",name,
                " seconds=",measured.seconds," bytes=",measured.allocated_bytes)
        end
        @assert isapprox(checksums[1],checksums[2];rtol=2e-12,atol=1e-10)
    end
    open(joinpath(output_dir,"measurements.toml"),"w") do io
        TOML.print(io,Dict("rows"=>rows,"julia_version"=>string(VERSION),
            "rk_commit"=>strip(read(`git rev-parse HEAD`,String)),
            "benchmark_sha256"=>bytes2hex(sha256(read(@__FILE__))),
            "rk_version"=>string(Base.pkgversion(ReactiveKernels)),
            "distribution_kernels_version"=>string(Base.pkgversion(ReactiveKernelsDistributionKernels)),
            "distributions_version"=>string(Base.pkgversion(Distributions))))
    end
end
end

length(ARGS) <= 2 || error("expected output directory and optional family name")
selected = length(ARGS) == 2 ? filter(x -> string(first(x)) == ARGS[2],
    DistributionGraphHoisting.FAMILIES) : DistributionGraphHoisting.FAMILIES
isempty(selected) && error("unknown family: $(ARGS[2])")
DistributionGraphHoisting.run(isempty(ARGS) ? mktempdir() : first(ARGS);families=selected)
