module LoweringWorkload
using ReactiveKernels, ReactiveKernelsPPL

# A builder retains its own definitions separately from the lowering cache.
const DEFINITIONS = Core.eval(@__MODULE__, Expr(:module, true, gensym(:FittedModel), Expr(:block)))
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

@kernel clipped(x::Float64) = begin
    z::Float64 = x + 1
    return z
end
const CHILD_REVISION = kernel_revision(clipped)
adjust(x) = 2x
center(xs) = sum(xs) / length(xs)
const BUILDS = KernelBuildCache()
const SUBMODEL_BUILDS = KernelBuildCache()
const CHILD_MODEL = quote
    avg = center(x_fit)
    a ~ Normal(avg, 1)
    s ~ Exponential(1)
    offset = clipped(a)
    shift = adjust(offset)
    mu = shift .+ x
    y .~ Normal.(mu, s)
end

function child_plan(xs, design)
    train = (; x=xs, x_fit=xs, y=copy(xs))
    lower(d; kwargs...) = lower_rkppl(CHILD_MODEL, d; mod=@__MODULE__, conditioned=(:y,), kwargs...)
    fit = frozen_values(bind_data(lower(train), train), :avg)
    bind_data(lower(design; frozen=fit), design)
end

# Only the cache and program definitions survive the workload.
if IN_IMAGE
    let design = (; x=[1., 0.], y=zeros(2))
        submodel = bind_data(lower(design, fitted([0., 1., 2.])), design)
        build_kernel(submodel; cache=SUBMODEL_BUILDS)
        build_kernel(child_plan([0., 1., 2.], design); cache=BUILDS)
    end
end
end
