using ReactiveKernels, ReactiveKernelsDistributionKernels, Chairmarks
using DifferentiationInterface: AutoEnzyme
import Enzyme
include(joinpath(@__DIR__, "full-counter-method.jl"))
const DK = ReactiveKernelsDistributionKernels.DistributionKernelSources
Base.include(DK, joinpath(@__DIR__, "baseline-law.jl"))
function law_case(spec, observed, logits, weights; bound_data=true)
    kernel = bound_data ?
        prepare(spec; bound=(; observed, weights), type_stability=:error) :
        prepare(spec; type_stability=:error)
    args = bound_data ? (logits,) : (observed, logits, weights)
    ad = prepare_ad(kernel, AutoEnzyme(; mode=Enzyme.Reverse), args...; active=:logits)
    gradient = zero(logits)
    case = (; kernel, ad, gradient, logits, args)
    call_gradient!(case)
    case
end
function call_gradient!(case)
    ad_value_and_gradient!(case.ad, case.gradient, case.args...)
    nothing
end
function stats(case)
    f() = call_gradient!(case)
    (; native=native_counts(f), julia_bytes=@allocated(f()))
end
function timed(case)
    f() = call_gradient!(case)
    minimum(Chairmarks.benchmark(nothing, f, _->nothing; seconds=0.05)).time*1e6
end
median(v) = (s=sort(v); (s[length(s)÷2]+s[length(s)÷2+1])/2)
function compare(a, b, n)
    @assert isapprox(a.kernel(a.args...), b.kernel(b.args...); rtol=2e-13, atol=2e-13)
    @assert isapprox(a.gradient, b.gradient; rtol=2e-13, atol=2e-13)
    ta, tb = Float64[], Float64[]
    before = strip(read("/proc/loadavg", String))
    for round in 1:12
        if isodd(round)
            push!(ta, timed(a)); push!(tb, timed(b))
        else
            push!(tb, timed(b)); push!(ta, timed(a))
        end
    end
    println((; positions=n, rows=size(a.logits, 1), bound_data=length(a.args)==1,
        baseline=stats(a), counts=stats(b),
        baseline_gradient_us=median(ta), counts_gradient_us=median(tb),
        ratio=median(tb)/median(ta),
        max_gradient_error=maximum(abs.(a.gradient .- b.gradient); init=0.0),
        load_before=before, load_after=strip(read("/proc/loadavg", String))))
    flush(stdout)
end
println((; enzyme=pkgversion(Enzyme), library=Enzyme.Compiler.Enzyme_jll.libEnzyme_path))
for n in (5, 51, 501)
    observed = Bool[isodd(i) for i in 1:n]
    logits = [0.7 -1.2]
    weights = [0.35, 0.65]
    compare(law_case(DK.baseline_product_mixture, observed, logits, weights),
        law_case(DK.bernoulli_logit_product_mixture_logdensity, observed, copy(logits), weights), n)
    full_logits = [logits[1,k] + 0.1sin(i+k) for i in 1:n, k in 1:2]
    compare(law_case(DK.baseline_product_mixture, observed, full_logits, weights),
        law_case(DK.bernoulli_logit_product_mixture_logdensity, observed, copy(full_logits), weights), n)
    compare(law_case(DK.baseline_product_mixture, observed, full_logits, weights; bound_data=false),
        law_case(DK.bernoulli_logit_product_mixture_logdensity, observed, copy(full_logits), weights; bound_data=false), n)
end
