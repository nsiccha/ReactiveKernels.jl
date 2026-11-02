#!/usr/bin/env julia

# Composite law reuse with changing active inputs and consumed outputs.
# DM compares with the public Distributions density. Moment baselines hide
# the same public RK arithmetic behind an ordinary scalar function: this
# isolates graph visibility without copying a numerical implementation.
# The native compiler remains free to optimize either implementation.
#   julia --project=<environment> benchmark/distribution_composite_hoisting.jl <output-dir>
module DistributionCompositeHoisting
using ReactiveKernels, ReactiveKernelsDistributionKernels, Statistics, TOML, SHA
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    dirichlet_multinomial, lower_truncated_lognormal, upper_truncated_lognormal
import Distributions

opaque_dm(y,n,alpha) = Distributions.logpdf(Distributions.DirichletMultinomial(n,alpha),y)
@kernel owned_dm(events::Vector{Vector{Int}}, n::Int, alpha::Vector{Float64}) = begin
    scores = plate(events,Ref(n),Ref(alpha)) do y,n,alpha
        dirichlet_multinomial(n,alpha).logpdf(y)
    end
    return sum(scores)
end
@kernel opaque_dm_graph(events::Vector{Vector{Int}}, n::Int, alpha::Vector{Float64}) = begin
    scores = plate(events,Ref(n),Ref(alpha)) do y,n,alpha
        opaque_dm(y,n,alpha)
    end
    return sum(scores)
end

const LOWER_VARIANCE = prepare(extract(lower_truncated_lognormal;
    have=(:location,:scale,:lower),want=:var))
const UPPER_VARIANCE = prepare(extract(upper_truncated_lognormal;
    have=(:location,:scale,:upper),want=:var))
opaque_lower_variance(q) = LOWER_VARIANCE(q[1],exp(q[2]),exp(q[3]))
opaque_upper_variance(q) = UPPER_VARIANCE(q[1],exp(q[2]),exp(q[3]))

@kernel owned_lower(data::Vector{Float64},q::Vector{Float64}) = begin
    scores = plate(data,Ref(q)) do y,q
        y * lower_truncated_lognormal(q[1],exp(q[2]),exp(q[3])).var()
    end
    return sum(scores)
end
@kernel opaque_lower(data::Vector{Float64},q::Vector{Float64}) = begin
    scores = plate(data,Ref(q)) do y,q
        y * opaque_lower_variance(q)
    end
    return sum(scores)
end
@kernel owned_upper(data::Vector{Float64},q::Vector{Float64}) = begin
    scores = plate(data,Ref(q)) do y,q
        y * upper_truncated_lognormal(q[1],exp(q[2]),exp(q[3])).var()
    end
    return sum(scores)
end
@kernel opaque_upper(data::Vector{Float64},q::Vector{Float64}) = begin
    scores = plate(data,Ref(q)) do y,q
        y * opaque_upper_variance(q)
    end
    return sum(scores)
end

function measure(f,q;repetitions=50)
    f(q); f(q)
    samples = Float64[]
    active = copy(q)
    checksum = 0.0
    for round in 1:9
        elapsed = @elapsed begin
            for i in 1:repetitions
                active[1] = q[1] + 1e-6*(i + repetitions*(round-1))
                checksum += f(active)
            end
        end
        push!(samples,elapsed/repetitions)
    end
    @assert isfinite(checksum)
    (;seconds=median(samples),allocated_bytes=@allocated(f(q)),checksum)
end

function record!(rows,output_dir,label,n,owned,opaque,q)
    for active in (q,q.+1e-4)
        @assert isapprox(owned(active),opaque(active);rtol=3e-12,atol=1e-300)
    end
    checksums = Float64[]
    for (name,f) in (("owned",owned),("opaque",opaque))
        code_file = joinpath(output_dir,"$(label)-$(n)-$(name).jl")
        open(code_file,"w") do io
            show(io,code_expr(f))
            println(io)
        end
        measured = measure(f,q)
        push!(checksums,measured.checksum)
        push!(rows,Dict{String,Any}("case"=>label,"observations"=>n,
            "implementation"=>name,"seconds"=>measured.seconds,
            "allocated_bytes"=>measured.allocated_bytes,"code"=>code_file,
            "checksum"=>measured.checksum))
        println("COMPOSITE_HOISTING ",label," n=",n," implementation=",name,
            " seconds=",measured.seconds," bytes=",measured.allocated_bytes)
    end
    @assert isapprox(checksums[1],checksums[2];rtol=3e-12,atol=1e-300)
end

function run(output_dir)
    mkpath(output_dir)
    rows = Dict{String,Any}[]
    for k in (3,64), n in (1,64,1024)
        alpha = [0.2 + 0.1mod(j,7) for j in 1:k]
        events = [Int[mod(i+j,4) for j in 1:k] for i in 1:n]
        # Fix the event total without fixing the category count structurally.
        for y in events
            y[end] += 3k-sum(y)
        end
        saved = deepcopy(events)
        owned = prepare(owned_dm;bound=(;events,n=3k))
        opaque = prepare(opaque_dm_graph;bound=(;events,n=3k))
        record!(rows,output_dir,"DirichletMultinomial-K$k",n,owned,opaque,alpha)
        @assert events == saved
    end
    for (side,owned_spec,opaque_spec) in (("Lower",owned_lower,opaque_lower),
                                         ("Upper",owned_upper,opaque_upper)),
        (region,q) in (("Central",[0.2,log(0.3),-0.1]),
                      ("Tail",[0.0,log(0.001),side == "Lower" ? 0.01 : -0.01])),
        n in (1,64,1024)
        data = [0.1+0.03mod(i,31) for i in 1:n]
        saved = copy(data)
        owned = prepare(owned_spec;bound=(;data))
        opaque = prepare(opaque_spec;bound=(;data))
        record!(rows,output_dir,"$(side)LogNormalVariance-$region",n,owned,opaque,q)
        @assert data == saved
    end
    open(joinpath(output_dir,"measurements.toml"),"w") do io
        TOML.print(io,Dict("rows"=>rows,"julia_version"=>string(VERSION),
            "rk_commit"=>strip(read(`git rev-parse HEAD`,String)),
            "benchmark_sha256"=>bytes2hex(sha256(read(@__FILE__))),
            "rk_version"=>string(Base.pkgversion(ReactiveKernels)),
            "distribution_kernels_version"=>string(Base.pkgversion(ReactiveKernelsDistributionKernels)),
            "distributions_version"=>string(Base.pkgversion(Distributions)),
            "moment_baseline"=>"same public RK variance arithmetic behind a scalar function",
            "dm_baseline"=>"Distributions.logpdf(DirichletMultinomial(n,alpha),counts)"))
    end
end
end

length(ARGS) <= 1 || error("expected at most one output directory")
DistributionCompositeHoisting.run(isempty(ARGS) ? mktempdir() : first(ARGS))
