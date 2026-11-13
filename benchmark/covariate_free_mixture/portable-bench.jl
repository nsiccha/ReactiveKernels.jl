# CaptureMtbh diagnostic: the previous-capture covariate term is removed.
# This is a different model, used to isolate nested-loop allocation behavior.
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, JSON, Chairmarks
using DifferentiationInterface: AutoEnzyme
using LogExpFunctions: logit
using LinearAlgebra: BLAS
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    bernoulli_logit_product_mixture_logdensity
BLAS.set_num_threads(1)

const model = @rkppl begin
    omega ~ Uniform(0, 1)
    mean_p[1:T] .~ Uniform.(0, 1)
    gamma ~ Normal(0, 10)
    sigma ~ Uniform(0, 3)
    eps_raw[1:N] .~ Normal.(0, 1)
    @plate for i in eachindex(histories)
        eta = logit.(mean_p) .+ sigma .* eps_raw[i]
        histories[i] ~ LogDensity(bernoulli_logit_product_mixture_logdensity,
            hcat(eta, never[i]), [omega, 1 - omega])
    end
end
decode(x) = x isa AbstractDict && haskey(x, "data") ? reshape(x["data"], Tuple(x["size"])) : x
function structure(ex)
    ex isa ReactiveKernels._PlateProgram && return structure(ex.ast)
    ex isa Expr || return (0, 0)
    foldl(.+, (structure(a) for a in ex.args);
        init=(1, Int(ex.head in (:for, :while))))
end
function native_counts(f)
    f(); f()
    ccall(:begin_count, Cvoid, ())
    f()
    packed = ccall(:end_count, UInt64, ())
    (; malloc=Int(packed >> 32), free=Int(packed & 0xffffffff))
end
median(v) = (s=sort(v); n=length(s); isodd(n) ? s[(n+1)÷2] : (s[n÷2]+s[n÷2+1])/2)
function make_case(N, fixture)
    y = Bool.(decode(fixture["y"]))
    T = size(y, 2)
    0 <= N <= size(y, 1) || error("N must fit the fixture")
    histories = [collect(view(y, i, :)) for i in 1:N]
    never = [fill(-Inf, T) for _ in histories]
    plan = model(; N, T, never) | (; histories)
    built = build_kernel(plan)
    point = (; (Symbol(k)=>decode(v) for (k,v) in fixture["constrained"])...)
    point = merge(point, (; eps_raw=point.eps_raw[1:N]))
    u = collect(Float64, unconstrain(built.layout, point))
    started = time()
    sampler = prepare_sampler(built, plan, u;
        backend=AutoEnzyme(; mode=Enzyme.Reverse), warm=true,
        native_forward_batch=false)
    g = similar(u)
    gradient!() = sampler_value_and_gradient!(sampler, g, u)
    value, _ = gradient!()
    (; N, sampler, g, u, value, prepare_seconds=time()-started)
end
function measure(case)
    (; N, sampler, g, u, value, prepare_seconds) = case
    gradient!() = sampler_value_and_gradient!(sampler, g, u)
    println((; N, prepare_seconds, value,
        nonallocating=sampler.kernel isa ReactiveKernels.NonAllocatingKernel,
        structure=structure(code_expr(sampler.kernel))))
    if get(ENV, "COUNT_NATIVE", "0") == "1"
        println((; N, native=native_counts(gradient!)))
    end
    println((; N, julia_gradient_bytes=@allocated(gradient!())))
    dt, gt = Float64[], Float64[]
    isfile("/proc/loadavg") && println((; N, timing_start_load=strip(read("/proc/loadavg", String))))
    for _ in 1:10
        push!(dt, minimum(Chairmarks.benchmark(nothing, ()->sampler(u), _->nothing; seconds=0.08)).time*1e6)
        push!(gt, minimum(Chairmarks.benchmark(nothing, gradient!, _->nothing; seconds=0.08)).time*1e6)
    end
    println((; N, density_median_us=median(dt), gradient_median_us=median(gt),
        density_min_us=minimum(dt), gradient_min_us=minimum(gt)))
    isfile("/proc/loadavg") && println((; N, timing_end_load=strip(read("/proc/loadavg", String))))
    if haskey(ENV, "DUMP_PREFIX")
        prefix = ENV["DUMP_PREFIX"] * "-N$N"
        open(prefix * "-value-gradient.txt", "w") do io
            show(io, (; value, gradient=g, point=u)); println(io)
        end
        open(prefix * "-program.txt", "w") do io
            show(io, Base.remove_linenums!(deepcopy(code_expr(sampler.kernel)))); println(io)
        end
    end
    flush(stdout)
end
println((; julia=VERSION, enzyme=pkgversion(Enzyme),
    library=Enzyme.Compiler.Enzyme_jll.libEnzyme_path,
    rk=pkgversion(ReactiveKernels), rk_path=pkgdir(ReactiveKernels),
    rkppl=pkgversion(ReactiveKernelsPPL)))
if abspath(PROGRAM_FILE) == @__FILE__
    fixture = JSON.parsefile(joinpath(@__DIR__, "fixture.json"))
    for N in parse.(Int, split(get(ENV, "EVENT_COUNTS", "1,11,146"), ','))
        measure(make_case(N, fixture))
    end
end
