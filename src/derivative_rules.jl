# Derivative rules generated from one pure-math graph.
#
# Policy (docs/src/constraints.md, "Derivative rules come from one mathematical
# graph, never by hand"): a numerical primitive's derivatives are authored ONCE
# as an ordinary `@kernel` graph, and every AD-protocol adapter is generated
# from activity-selected HAVE→WANT cuts of that graph, attached to a callable
# this package owns. This file is the first slice: scalar graphs with a primal
# output and one named partial per input. The scalar chain rule turns the
# partials into the JVP (Σ ∂y/∂xᵢ · ẋᵢ) and the VJP (ȳ · ∂y/∂xᵢ) without any
# general AD or program transposition, so the adapters in the backend
# extensions (`ext/ReactiveKernelsEnzymeExt.jl`) contain no mathematics of
# their own: they select a cut by the call's activity pattern and combine its
# partials with the backend's directions or covectors.
#
# Representation. Each cut is lowered by the ordinary planner
# (`_lower_with_ops`) to a straight-line body of `name = __ops__[i](names...)`
# statements over the cut's operation table. That body is encoded in the rule's
# TYPE (tuples of symbols and integers) and emitted by a generated function,
# while the operation tables are the only fields. The callable therefore holds
# no runtime-generated code object: with singleton source operations it is an
# `isbits` value, so a closure capturing it stays a plain value for every AD
# backend, and its call is generic over the scalar type (a Reactant-traced
# number takes the same cut, because the source operations select a native or
# tensorized body by argument style).

"""
    ScalarDerivativeRule{Name,N,Inputs}

Callable produced by [`scalar_derivative_rule`](@ref). Calling it with `N`
scalar arguments runs the primal cut of its graph (`want = primal`), so as a
function it is exactly the authored primal formula, generic over the argument
types. The activity-selected cuts are reached through
[`derivative_cut`](@ref); backend extensions attach their rule protocols to
`Const{<:ScalarDerivativeRule}` and read the partials from those cuts. The
callable is an immutable value whose only fields are the cuts' operation
tables (an `isbits` value when those operations are singletons), so a closure
capturing it needs no activity annotation.
"""
struct ScalarDerivativeRule{Name,N,Inputs,Primal,Cuts,PO,CO} <: Function
    primal_ops::PO
    cut_ops::CO
end

function Base.show(io::IO, ::ScalarDerivativeRule{Name,N,Inputs}) where {Name,N,Inputs}
    print(io, "ScalarDerivativeRule(:", Name, "; inputs = ", Inputs, ")")
end

# A cut body is `(signature_names, ((output, op_index, (arguments...)), ...),
# return_spec)`; every element is a Symbol, an Int, or a numeric literal, so
# the whole body is a valid type parameter.
function _rule_body_expr(spec, ops_expr)
    signature, body, ret = spec
    statements = Any[:(__ops__ = $ops_expr)]
    for (index, name) in enumerate(signature)
        push!(statements, :($name = getfield(args, $index)))
    end
    for (output, op, arguments) in body
        push!(statements, :($output = __ops__[$op]($(arguments...))))
    end
    push!(statements, ret isa Symbol ? :(return $ret) : :(return ($(ret...),)))
    Expr(:block, statements...)
end

@generated function (rule::ScalarDerivativeRule{Name,N,Inputs,Primal})(
        args::Vararg{Any,N}) where {Name,N,Inputs,Primal}
    _rule_body_expr(Primal, :(getfield(rule, :primal_ops)))
end

@generated function _derivative_cut_call(
        rule::ScalarDerivativeRule{Name,N,Inputs,Primal,Cuts}, ::Val{M},
        args::Vararg{Any,N}) where {Name,N,Inputs,Primal,Cuts,M}
    _rule_body_expr(Cuts[M], :(getfield(getfield(rule, :cut_ops), $M)))
end

@inline _activity_mask(::Tuple{}, bit::Int) = 0
@inline _activity_mask(active::Tuple, bit::Int) =
    (first(active) ? bit : 0) + _activity_mask(Base.tail(active), bit << 1)

"""
    derivative_cut(rule::ScalarDerivativeRule, active::NTuple{N,Bool}, args...)
        -> (y, ∂y/∂xᵢ for every active i, in input order)

Run the cut of the rule's graph that computes the primal together with the
partials of the inputs flagged `active`. The cut is selected from the
activity pattern alone, so a backend adapter whose activity pattern is a
compile-time property of the call signature reaches one concretely typed
cut; inactive inputs cost nothing beyond the primal. At least one input must
be active.
"""
@inline function derivative_cut(rule::ScalarDerivativeRule{Name,N},
        active::NTuple{N,Bool}, args::Vararg{Any,N}) where {Name,N}
    mask = _activity_mask(active, 1)
    mask == 0 && throw(ArgumentError(
        "derivative_cut of $(Name) needs at least one active input"))
    _derivative_cut_call(rule, Val(mask), args...)
end

"""
    scalar_derivative_rule(graph::KernelSpec; primal, partials, name) -> ScalarDerivativeRule

Generate an RK-owned callable whose value and derivatives all come from
`graph`, an ordinary `@kernel` whose HAVE ports are the function's real scalar
arguments in signature order and whose WANT ports include the primal output
`primal::Symbol` and one partial derivative per input, named in
`partials::NamedTuple` (input name => WANT port name, in signature order).

```julia
@kernel loggamma_graph(x::Float64) = begin
    y::Float64 = SpecialFunctions.loggamma(x)
    dy_dx::Float64 = SpecialFunctions.digamma(x)
    return y, dy_dx
end
const loggamma = scalar_derivative_rule(
    loggamma_graph; primal = :y, partials = (x = :dy_dx,), name = :loggamma)
```

The result computes `loggamma(x)` through the graph's primal cut (the partial
is pruned), and every AD backend with a ReactiveKernels extension loaded
treats it as a primitive whose reverse and forward rules are generated from
the activity-selected cuts (see [`derivative_cut`](@ref)): no backend-specific
derivative code is authored, and no rule attaches to a function this
repository does not own. Every input, the primal, and every partial must be
declared with a `Real` scalar type; the graph must not embed a plate.
"""
function scalar_derivative_rule(spec::KernelSpec; primal::Symbol,
        partials::NamedTuple, name::Symbol = :scalar_derivative_rule)
    inputs = Tuple(spec.have_names)
    N = length(inputs)
    N >= 1 || throw(ArgumentError(
        "scalar_derivative_rule($(name)) needs at least one input port; the graph has none"))
    keys(partials) == inputs || throw(ArgumentError(
        "scalar_derivative_rule($(name)): partials must name every input in " *
        "signature order $(inputs); got $(keys(partials))"))
    wants = Tuple(spec.want_names)
    primal in wants || throw(ArgumentError(
        "scalar_derivative_rule($(name)): primal port :$(primal) is not a WANT " *
        "port of the graph; WANT ports are $(wants)"))
    for (input, partial) in pairs(partials)
        partial isa Symbol || throw(ArgumentError(
            "scalar_derivative_rule($(name)): the partial for :$(input) must be a " *
            "WANT port name, got $(repr(partial))"))
        partial in wants || throw(ArgumentError(
            "scalar_derivative_rule($(name)): partial port :$(partial) (for " *
            ":$(input)) is not a WANT port of the graph; WANT ports are $(wants)"))
        partial === primal && throw(ArgumentError(
            "scalar_derivative_rule($(name)): partial port :$(partial) (for " *
            ":$(input)) is the primal port"))
    end
    for port in (inputs..., primal, values(partials)...)
        T = valtype(spec.ports[port])
        T <: Real || throw(ArgumentError(
            "scalar_derivative_rule($(name)) needs real scalar ports; :$(port) is " *
            "declared $(T)"))
    end
    primal_spec, primal_ops = _derivative_cut_lowering(spec, name, inputs, primal)
    cuts = ntuple(2^N - 1) do mask
        selected = Tuple(partials[i] for i in 1:N if (mask >> (i - 1)) & 1 == 1)
        _derivative_cut_lowering(spec, name, inputs, (primal, selected...))
    end
    cut_specs = map(first, cuts)
    cut_ops = map(last, cuts)
    ScalarDerivativeRule{
        name,N,inputs,primal_spec,cut_specs,typeof(primal_ops),typeof(cut_ops)}(
        primal_ops, cut_ops)
end

function _derivative_cut_lowering(spec::KernelSpec, name::Symbol, have, want)
    p = plan(spec; have = have, want = want)
    Tuple(v.name for v in p.have) == have || throw(ArgumentError(
        "scalar_derivative_rule($(name)): the planned HAVE order " *
        "$(Tuple(v.name for v in p.have)) differs from the signature order $(have)"))
    p = _partial_apply(p, ())
    _needs_embedded_tensorization(p) && throw(ArgumentError(
        "scalar_derivative_rule($(name)): a scalar derivative rule cannot embed a plate"))
    ast, ops, _ = _lower_with_ops(_fuse_authored_plate_chains(p))
    (_encode_cut_body(ast, name), ops)
end

# Encode a lowered `function (__ops__, x::T...) name = __ops__[i](names...);
# ...; return ... end` as type-level data. Anything outside that grammar is a
# lowering this slice does not cover and is refused explicitly.
function _encode_cut_body(ast::Expr, name::Symbol)
    unsupported(what) = throw(ArgumentError(
        "scalar_derivative_rule($(name)): the lowered cut contains $(what), " *
        "which the scalar rule generator does not encode: $(ast)"))
    ast.head === :function || unsupported("a non-function expression")
    signature = ast.args[1]
    (signature isa Expr && signature.head === :tuple &&
     !isempty(signature.args) && signature.args[1] === _OPS_ARG) ||
        unsupported("an unexpected signature")
    names = map(signature.args[2:end]) do arg
        arg isa Expr && arg.head === :(::) && arg.args[1] isa Symbol && return arg.args[1]
        arg isa Symbol && return arg
        unsupported("the argument $(arg)")
    end
    body = Any[]
    ret = nothing
    for statement in ast.args[2].args
        statement isa LineNumberNode && continue
        if statement isa Expr && statement.head === :return
            ret === nothing || unsupported("a second return")
            value = statement.args[1]
            if value isa Symbol
                ret = value
            elseif value isa Expr && value.head === :tuple && all(x -> x isa Symbol, value.args)
                ret = Tuple(value.args)
            else
                unsupported("the return value $(value)")
            end
            continue
        end
        ret === nothing || unsupported("a statement after the return")
        (statement isa Expr && statement.head === :(=) && statement.args[1] isa Symbol) ||
            unsupported("the statement $(statement)")
        call = statement.args[2]
        (call isa Expr && call.head === :call && call.args[1] isa Expr &&
         call.args[1].head === :ref && call.args[1].args[1] === _OPS_ARG &&
         call.args[1].args[2] isa Int) || unsupported("the statement $(statement)")
        arguments = map(call.args[2:end]) do arg
            (arg isa Symbol || arg isa Number) && return arg
            unsupported("the call argument $(arg)")
        end
        push!(body, (statement.args[1], call.args[1].args[2], Tuple(arguments)))
    end
    ret === nothing && unsupported("no return statement")
    (Tuple(names), Tuple(body), ret)
end
