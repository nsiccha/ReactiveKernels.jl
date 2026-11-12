#!/usr/bin/env julia
# Same-process comparison of the indexed-invariant pass on a natural RKPPL
# whole-event model. The baseline disables only this pass during preparation.
# Both measurements use the default nonallocating sampler and ordinary Reverse.
#
# julia --project=<env> benchmark/whole_event_indexed_invariants.jl <output-dir> [data.toml]
# Optional public capture data: shape=[events,positions], y=column-major Bool
# entries, u=unconstrained point (weight followed by position probabilities).
module WholeEventIndexedInvariants
using ReactiveKernels, ReactiveKernelsPPL, BenchmarkTools, TOML, Random, Statistics
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    bernoulli_product_mixture_logdensity
using Distributions: Uniform
import DifferentiationInterface as DI
import Enzyme

function prepare_capture(events, raw; reuse)
    T = length(first(events))
    data = (; histories=events,T,absent=zeros(T))
    bound = bind_data(lower_rkppl(quote
        weight ~ Uniform(0,1)
        probability[1:T] .~ Uniform.(0,1)
        @plate for i in eachindex(histories)
            histories[i] ~ LogDensity(bernoulli_product_mixture_logdensity,
                hcat(probability,absent),[weight,1-weight])
        end
    end,data;mod=@__MODULE__,conditioned=(:histories,)),data)
    # This is a compiler experiment, not a second model or a sampler option.
    task_local_storage(ReactiveKernels._INDEXED_INVARIANT_DEPTH, reuse ? 0 : 1) do
        prepare_sampler(build_kernel(bound),bound,raw;
            backend=DI.AutoEnzyme(;mode=Enzyme.Reverse),native_forward_batch=false)
    end
end

function run(output_dir, data_file=nothing)
    mkpath(output_dir)
    if data_file === nothing
        events = [Bool[isodd(i >> j) for j in 0:2] for i in 1:237]
        raw = [-0.3,-0.8,0.2,0.7]
        source = "synthetic 237-event, three-position fixture"
    else
        record = TOML.parsefile(data_file)
        observed = reshape(Bool.(record["y"]),Tuple(Int.(record["shape"])))
        events = [collect(view(observed,i,:)) for i in axes(observed,1)]
        raw = Float64.(record["u"])
        source = abspath(data_file)
    end
    replay = Dict("shape"=>[length(events),length(first(events))],
        "y"=>Int.(vec(reduce(vcat,permutedims.(events)))),"u"=>raw)
    open(io -> TOML.print(io,replay),joinpath(output_dir,"data.toml"),"w")
    baseline = prepare_capture(events,raw;reuse=false)
    candidate = prepare_capture(events,raw;reuse=true)
    @assert !occursin("_indexed_map",string(code_expr(baseline.kernel)))
    @assert occursin("_indexed_map",string(code_expr(candidate.kernel)))
    @assert ReactiveKernels._indexed_data_safe((candidate.kernel.constants,raw))
    for (label,sampler) in (("baseline",baseline),("candidate",candidate))
        findings = ReactiveKernels._level_code_findings(:all,sampler.kernel,(typeof(raw),))
        @assert isempty(findings) "$label has non-concrete prepared code"
    end
    for values in (raw,raw .+ 0.1)
        vb, gb = sampler_value_and_gradient!(baseline,similar(values),values)
        vc, gc = sampler_value_and_gradient!(candidate,similar(values),values)
        @assert isapprox(vb,vc;rtol=1e-13)
        @assert isapprox(gb,gc;rtol=1e-12)
    end
    for (label,sampler) in (("baseline",baseline),("candidate",candidate))
        open(joinpath(output_dir,label*"-program.jl"),"w") do io
            Base.show_unquoted(io,code_expr(sampler.kernel))
            println(io)
        end
    end
    rounds = Any[]
    for round in 1:35
        load_before = Sys.loadavg()
        order = isodd(round) ? (("baseline",baseline),("candidate",candidate)) :
                              (("candidate",candidate),("baseline",baseline))
        measurements = Dict{String,Any}()
        for (label,sampler) in order
            gradient = similar(raw)
            primal = @benchmark $sampler($raw) seconds=0.1 samples=2000 evals=20
            reverse = @benchmark sampler_value_and_gradient!($sampler,$gradient,$raw) seconds=0.1 samples=2000 evals=20
            p, r = minimum(primal), minimum(reverse)
            measurements[label] = (; primal_ns=p.time,reverse_ns=r.time,
                primal_allocations=p.allocs,reverse_allocations=r.allocs,
                primal_bytes=p.memory,reverse_bytes=r.memory)
        end
        measurements["load_before"] = load_before
        measurements["load_after"] = Sys.loadavg()
        push!(rounds,measurements)
        println("round=",round," ",measurements)
    end
    open(joinpath(output_dir,"rounds.tsv"),"w") do io
        println(io,"round\tvariant\tprimal_ns\treverse_ns\tprimal_allocations\treverse_allocations\tprimal_bytes\treverse_bytes")
        for (i,round) in enumerate(rounds), label in ("baseline","candidate")
            println(io,join((i,label,Base.values(round[label])...), '\t'))
        end
    end
    rng = MersenneTwister(20261112)
    summary = Dict{String,Any}("source"=>source,"events"=>length(events),
        "positions"=>length(first(events)),"julia"=>string(VERSION),
        "enzyme"=>string(pkgversion(Enzyme)),"ordinary_reverse"=>true,
        "machine"=>Sys.MACHINE,"cpu"=>Sys.CPU_NAME,"threads"=>Threads.nthreads(),
        "host_load"=>[Dict("round"=>i,"before"=>r["load_before"],"after"=>r["load_after"])
                      for (i,r) in enumerate(rounds)])
    for metric in (:primal_ns,:reverse_ns)
        ratios = [getproperty(r["candidate"],metric)/getproperty(r["baseline"],metric)
                  for r in rounds]
        bootstrap = [exp(mean(log.(rand(rng,ratios,length(ratios))))) for _ in 1:10000]
        summary[String(metric)] = Dict("candidate_over_baseline"=>exp(mean(log.(ratios))),
            "bootstrap_95pct"=>quantile(bootstrap,[0.025,0.975]))
        println(metric," ",summary[String(metric)])
    end
    open(io -> TOML.print(io,summary),joinpath(output_dir,"summary.toml"),"w")
end

abspath(PROGRAM_FILE) == (@__FILE__) && run(ARGS[1],length(ARGS)>1 ? ARGS[2] : nothing)
end
