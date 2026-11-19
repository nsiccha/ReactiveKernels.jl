module LoweringWorkload
using ReactiveKernels, ReactiveKernelsPPL

# A builder retains its own definitions separately from the lowering cache.
const DEFINITIONS = Module(gensym(:FittedModel))
Core.eval(DEFINITIONS, :(using ReactiveKernels, ReactiveKernelsPPL))
Core.eval(DEFINITIONS, :(ReactiveKernels.@kernel fitmean(v) = sum(v) / length(v)))
Core.eval(DEFINITIONS, :(ReactiveKernelsPPL.@rkppl regression(X, n) = begin
    beta[1:n] .~ Normal.(0, 1)
    return X * beta
end))
const SOURCE = quote
    center = fitmean(x_fit)
    z = x .- center
    X = hcat(ones(length(x)), z)
    mu ~ regression(X, 2)
    y .~ Normal.(mu, 1)
end
const CACHE = LoweringCache()
const IN_IMAGE = ccall(:jl_generating_output, Cint, ()) != 0

function fitted(xs)
    training = (; x_fit=xs, x=xs, y=zeros(length(xs)))
    plan = lower_rkppl(SOURCE, training; mod=DEFINITIONS, conditioned=(:y,))
    return frozen_values(bind_data(plan, training), :center)
end
lower(design, frozen; cache=CACHE) =
    lower_rkppl(SOURCE, design; mod=DEFINITIONS, conditioned=(:y,), frozen, cache)

# Only the cache and program definitions survive the workload.
if IN_IMAGE
    lower((; x=[1., 0.], y=zeros(2)), fitted([0., 1., 2.]))
end
end
