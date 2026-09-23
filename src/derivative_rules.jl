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
function _rule_body_expr(spec, ops_expr; tuple_return::Bool = false)
    signature, body, ret = spec[1], spec[2], spec[3]
    statements = Any[:(__ops__ = $ops_expr)]
    for (index, name) in enumerate(signature)
        push!(statements, :($name = getfield(args, $index)))
    end
    for (output, op, arguments) in body
        push!(statements, :($output = __ops__[$op]($(arguments...))))
    end
    if ret isa Symbol
        push!(statements, tuple_return ? :(return ($ret,)) : :(return $ret))
    else
        push!(statements, :(return ($(ret...),)))
    end
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

# ------------------------------------------------------------------------
# Vector rules: one graph with authored forward and reverse branches.
#
# The scalar slice above derives both AD directions from named partials. For
# array-valued primitives the graph authors the directions itself, as in
# `examples/manual_derivative_rule.jl`: direction ports feed a tangent output
# (the JVP branch) and a covector port feeds cotangent outputs (the VJP
# branch). The generator prepares the primal cut, the forward cut over every
# direction, and one reverse cut per activity pattern that wants only the
# active cotangents; the residual environment a reverse cut needs is read off
# its lowered body (the inputs it actually references), so an adapter retains
# exactly those values across the primal/cotangent staging boundary. Nothing
# derives one direction from the other: both are authored mathematics.

"""
    DerivativeRule{Name,N,Inputs}

Callable produced by [`derivative_rule`](@ref) from one pure-math graph with
authored forward and/or reverse branches. Calling it with the `N` inputs runs
the primal cut. [`forward_cut`](@ref) runs the JVP branch over every direction,
[`reverse_cut`](@ref) runs the VJP branch of one activity pattern, and
[`reverse_residuals`](@ref) says which inputs that cut reads. Backend
extensions stage those cuts into their rule protocols; the callable's only
fields are the cuts' operation tables.
"""
struct DerivativeRule{Name,N,Inputs,Primal,Forward,Reverse,PO,FO,RO} <: Function
    primal_ops::PO
    forward_ops::FO
    reverse_ops::RO
end

function Base.show(io::IO,
        ::DerivativeRule{Name,N,Inputs,Primal,Forward,Reverse}) where
        {Name,N,Inputs,Primal,Forward,Reverse}
    branches = String[]
    Forward === nothing || push!(branches, "forward")
    Reverse === nothing || push!(branches, "reverse")
    print(io, "DerivativeRule(:", Name, "; inputs = ", Inputs, ", branches = ",
          join(branches, "+"), ")")
end

@generated function (rule::DerivativeRule{Name,N,Inputs,Primal})(
        args::Vararg{Any,N}) where {Name,N,Inputs,Primal}
    _rule_body_expr(Primal, :(getfield(rule, :primal_ops)))
end

"""
    has_forward_branch(rule::DerivativeRule) -> Bool
    has_reverse_branch(rule::DerivativeRule) -> Bool
"""
has_forward_branch(::DerivativeRule{Name,N,Inputs,Primal,Forward}) where
    {Name,N,Inputs,Primal,Forward} = Forward !== nothing
has_reverse_branch(::DerivativeRule{Name,N,Inputs,Primal,Forward,Reverse}) where
    {Name,N,Inputs,Primal,Forward,Reverse} = Reverse !== nothing

"""
    forward_cut(rule::DerivativeRule, inputs..., directions...) -> (y, ẏ)

Run the graph's JVP branch: the primal together with the tangent for the
given directions (one per input, in input order; an inactive input takes a
zero direction). Requires the rule to have a forward branch.
"""
@generated function forward_cut(
        rule::DerivativeRule{Name,N,Inputs,Primal,Forward},
        args::Vararg{Any,M}) where {Name,N,Inputs,Primal,Forward,M}
    Forward === nothing && return :(throw(ArgumentError(
        "forward_cut: rule " * $(string(Name)) * " has no forward branch")))
    M == 2N || return :(throw(ArgumentError(
        "forward_cut: expected " * $(string(N)) * " inputs followed by " *
        $(string(N)) * " directions, got " * $(string(M)) * " arguments")))
    _rule_body_expr(Forward, :(getfield(rule, :forward_ops)))
end

"""
    reverse_residuals(rule::DerivativeRule, ::Val{mask}) -> NTuple{N,Bool}

Which inputs the reverse cut of activity `mask` (bit `i` set when input `i` is
active) reads; an adapter retains exactly those across the primal/cotangent
staging boundary and may pass `nothing` for the others.
"""
@generated function reverse_residuals(
        ::DerivativeRule{Name,N,Inputs,Primal,Forward,Reverse},
        ::Val{M}) where {Name,N,Inputs,Primal,Forward,Reverse,M}
    Reverse === nothing && return :(throw(ArgumentError(
        "reverse_residuals: rule " * $(string(Name)) * " has no reverse branch")))
    Reverse[M][4]
end

"""
    reverse_cut(rule::DerivativeRule, ::Val{mask}, inputs..., covector)
        -> (cotangents of the active inputs, in input order)

Run the graph's VJP branch for one activity pattern. Inputs the cut does not
read (see [`reverse_residuals`](@ref)) may be passed as `nothing`. Always
returns a tuple. Requires the rule to have a reverse branch.
"""
@generated function reverse_cut(
        rule::DerivativeRule{Name,N,Inputs,Primal,Forward,Reverse}, ::Val{M},
        args::Vararg{Any,K}) where {Name,N,Inputs,Primal,Forward,Reverse,M,K}
    Reverse === nothing && return :(throw(ArgumentError(
        "reverse_cut: rule " * $(string(Name)) * " has no reverse branch")))
    K == N + 1 || return :(throw(ArgumentError(
        "reverse_cut: expected " * $(string(N)) * " inputs followed by the " *
        "covector, got " * $(string(K)) * " arguments")))
    (1 <= M <= length(Reverse)) || return :(throw(ArgumentError(
        "reverse_cut: activity mask " * $(string(M)) * " is out of range")))
    _rule_body_expr(Reverse[M], :(getfield(getfield(rule, :reverse_ops), $M));
                    tuple_return = true)
end

"""
    derivative_rule(graph::KernelSpec; primal, directions, tangent, covector,
                    cotangents, name) -> DerivativeRule

Generate an RK-owned callable from one pure-math graph that authors its own
derivative branches. The graph's HAVE ports are the function's inputs plus,
optionally, one direction port per input and one covector port; its WANT
ports include the primal output and, matching the supplied branches, the
tangent output and one cotangent output per input.

```julia
@kernel matvec_rule(A::Matrix{Float64}, x::Vector{Float64},
        A_dot::Matrix{Float64}, x_dot::Vector{Float64},
        y_bar::Vector{Float64}) = begin
    y::Vector{Float64} = A * x
    y_dot::Vector{Float64} = A_dot * x + A * x_dot
    A_bar::Matrix{Float64} = y_bar * transpose(x)
    x_bar::Vector{Float64} = transpose(A) * y_bar
    return y, y_dot, A_bar, x_bar
end
const matvec = derivative_rule(matvec_rule; primal = :y,
    directions = (A = :A_dot, x = :x_dot), tangent = :y_dot,
    covector = :y_bar, cotangents = (A = :A_bar, x = :x_bar), name = :matvec)
```

`matvec(A, x)` is the primal cut. The forward branch (`directions` +
`tangent`) and the reverse branch (`covector` + `cotangents`) are each
optional, but at least one must be given; every AD adapter is generated from
the cuts of this one graph, attached to the returned callable. Inputs are the
HAVE ports that are neither directions nor the covector, in HAVE order.
"""
function derivative_rule(spec::KernelSpec; primal::Symbol,
        directions = nothing, tangent = nothing, covector = nothing,
        cotangents = nothing, name::Symbol = :derivative_rule)
    haves = Tuple(spec.have_names)
    wants = Tuple(spec.want_names)
    fail(msg) = throw(ArgumentError("derivative_rule($(name)): " * msg))
    primal in wants || fail("primal port :$(primal) is not a WANT port; WANT ports are $(wants)")
    has_forward = directions !== nothing || tangent !== nothing
    has_reverse = covector !== nothing || cotangents !== nothing
    has_forward || has_reverse || fail(
        "give a forward branch (directions + tangent), a reverse branch " *
        "(covector + cotangents), or both")
    if has_forward
        (directions isa NamedTuple && tangent isa Symbol) || fail(
            "a forward branch needs both `directions::NamedTuple` and `tangent::Symbol`")
        tangent in wants || fail("tangent port :$(tangent) is not a WANT port; WANT ports are $(wants)")
        tangent === primal && fail("tangent port :$(tangent) is the primal port")
    end
    if has_reverse
        (covector isa Symbol && cotangents isa NamedTuple) || fail(
            "a reverse branch needs both `covector::Symbol` and `cotangents::NamedTuple`")
        covector in haves || fail("covector port :$(covector) is not a HAVE port; HAVE ports are $(haves)")
    end
    direction_names = has_forward ? Tuple(values(directions)) : ()
    inputs = Tuple(h for h in haves if !(h in direction_names) && h !== covector)
    isempty(inputs) && fail("no input port remains once directions and the covector are removed from $(haves)")
    N = length(inputs)
    if has_forward
        keys(directions) == inputs || fail(
            "directions must name every input in signature order $(inputs); got $(keys(directions))")
        for (input, direction) in pairs(directions)
            direction isa Symbol || fail("the direction for :$(input) must be a HAVE port name, got $(repr(direction))")
            direction in haves || fail("direction port :$(direction) (for :$(input)) is not a HAVE port; HAVE ports are $(haves)")
        end
        length(unique(direction_names)) == N || fail("direction ports must be distinct; got $(direction_names)")
    end
    if has_reverse
        keys(cotangents) == inputs || fail(
            "cotangents must name every input in signature order $(inputs); got $(keys(cotangents))")
        for (input, cotangent) in pairs(cotangents)
            cotangent isa Symbol || fail("the cotangent for :$(input) must be a WANT port name, got $(repr(cotangent))")
            cotangent in wants || fail("cotangent port :$(cotangent) (for :$(input)) is not a WANT port; WANT ports are $(wants)")
            cotangent === primal && fail("cotangent port :$(cotangent) (for :$(input)) is the primal port")
        end
        length(unique(Tuple(values(cotangents)))) == N || fail("cotangent ports must be distinct; got $(Tuple(values(cotangents)))")
    end
    primal_spec, primal_ops = _derivative_cut_lowering(spec, name, inputs, primal)
    forward_spec, forward_ops = has_forward ?
        _derivative_cut_lowering(spec, name, (inputs..., direction_names...), (primal, tangent)) :
        (nothing, nothing)
    reverse = if has_reverse
        ntuple(2^N - 1) do mask
            active = Tuple(cotangents[i] for i in 1:N if (mask >> (i - 1)) & 1 == 1)
            want = length(active) == 1 ? active[1] : active
            cut_spec, ops = _derivative_cut_lowering(spec, name, (inputs..., covector), want)
            used = _cut_referenced_names(cut_spec)
            residuals = Tuple(input in used for input in inputs)
            ((cut_spec[1], cut_spec[2], cut_spec[3], residuals), ops)
        end
    else
        nothing
    end
    reverse_specs = reverse === nothing ? nothing : map(first, reverse)
    reverse_ops = reverse === nothing ? nothing : map(last, reverse)
    DerivativeRule{name,N,inputs,primal_spec,forward_spec,reverse_specs,
                   typeof(primal_ops),typeof(forward_ops),typeof(reverse_ops)}(
        primal_ops, forward_ops, reverse_ops)
end

# Every name a lowered cut reads: call arguments and returned values.
function _cut_referenced_names(spec)
    _, body, ret = spec[1], spec[2], spec[3]
    used = Set{Symbol}()
    for (_, _, arguments) in body, argument in arguments
        argument isa Symbol && push!(used, argument)
    end
    if ret isa Symbol
        push!(used, ret)
    else
        for name in ret; push!(used, name); end
    end
    used
end
