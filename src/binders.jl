"Token-preserving partial application for registered kernel callables."
struct PartialFunction{F,L<:Tuple,R<:Tuple,K<:NamedTuple} <: Function
    func::F
    largs::L
    rargs::R
    kwargs::K
end

(f::PartialFunction)(args...; kwargs...) =
    f.func(f.largs..., args..., f.rargs...; f.kwargs..., kwargs...)

function Base.getproperty(f::PartialFunction, name::Symbol)
    name in fieldnames(typeof(f)) && return getfield(f, name)
    getproperty(getfield(f, :kwargs), name)
end

Base.propertynames(f::PartialFunction, private::Bool = false) =
    private ? (fieldnames(typeof(f))..., keys(getfield(f, :kwargs))...) :
              keys(getfield(f, :kwargs))

partial(f, args...; kwargs...) = PartialFunction(f, args, (), (; kwargs...))
partial(f, ::Colon, args...; kwargs...) =
    PartialFunction(f, (), args, (; kwargs...))
partial(f, left, ::Colon, args...; kwargs...) =
    PartialFunction(f, (left,), args, (; kwargs...))

# `PartialFunction` is the approved token-preserving binder: opt IN to the factory's
# binder trait (declared default-nothing in kernel_factory.jl) by exposing its wrapped
# target. This is the ONLY sanctioned extension — the factory never duck-types `.func`.
_kernel_binder_target(f::PartialFunction) = getfield(f, :func)
# The bound keywords of the binder (RK 04:41) — the ONLY sanctioned way the lowerer reads a
# PartialFunction's `stepsize` etc.; never duck-typed. Bound numeric values stay runtime-typed
# (a same-type binder with stepsize .1 vs .2 must not alias a compiled constant).
_kernel_binder_kwargs(f::PartialFunction) = getfield(f, :kwargs)
# Bound POSITIONAL actuals (left/right) — the factory validates a subject callable binds none (RK 09:36).
_kernel_binder_positionals(f::PartialFunction) = (getfield(f, :largs), getfield(f, :rargs))

