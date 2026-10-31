# Cold analysis of compiler-visible Julia calls. OutputSignatures owns all
# numerical transfer rules; this layer only resolves source bindings and
# translates Julia's call/index/broadcast syntax. No user function is run.

function _metadata_callable(callee, mod::Module, shadowed)
    callee isa GlobalRef && return isdefined(callee.mod, callee.name) ?
        getglobal(callee.mod, callee.name) : nothing
    callee isa Symbol && callee in shadowed && return nothing
    if Meta.isexpr(callee, :., 2) && callee.args[1] isa Symbol
        callee.args[1] in shadowed && return nothing
    end
    return _kernel_resolve_binding(mod, callee)
end

"""
    _metadata_function(f)

The function whose OutputSignatures rules describe the calls of `f`: `f`
itself, unless `f` is a package's own formulation of a function those rules
already describe. A package states that for its own scalar function with a
method of this function, as it adds a JAX translation with
`_jax_elementwise`; OutputSignatures still owns every rule.
"""
_metadata_function(f) = f

# The OutputSignatures `CallInfo` for a source call whose operand descriptors
# have already been computed: `.status` says whether a rule applied, and an
# `:unknown` result may still carry sound partial facts (a broadcast's axes).
function _call_metadata(ex::Expr, args::Vector{OutputSignatures.ValueInfo},
                        mod::Module, shadowed;
                        kwargs = Pair{Symbol,OutputSignatures.ValueInfo}[])
    unknown(reason) = OutputSignatures.CallInfo(OutputSignatures.ValueInfo(), :unknown, reason)
    f, broadcasted = if ex.head === :ref
        getindex, false
    elseif ex.head === :call && !isempty(ex.args)
        callee = ex.args[1]
        if callee isa Symbol && startswith(String(callee), ".") && callee !== :.
            _metadata_callable(Symbol(String(callee)[2:end]), mod, shadowed), true
        else
            _metadata_callable(callee, mod, shadowed), false
        end
    elseif Meta.isexpr(ex, :., 2) && Meta.isexpr(ex.args[2], :tuple)
        _metadata_callable(ex.args[1], mod, shadowed), true
    else
        return unknown("no call syntax")
    end
    f === nothing && return unknown("callee is not a resolved binding")
    f = _metadata_function(f)
    # Julia lowers a literal exponent in x^n / x.^n to literal_pow. In
    # particular integer bases with negative literal powers produce floats;
    # the ordinary ^ call can throw instead. Normalize syntax, not a guessed
    # numerical rule, and keep alias calls (pow(x, n)) as ordinary calls.
    callee = first(ex.args)
    power_syntax = callee in (:^, Symbol(".^")) ||
        (callee isa GlobalRef && callee.name === :^)
    operands = ex.head === :call ? ex.args[2:end] :
        Meta.isexpr(ex, :., 2) && Meta.isexpr(ex.args[2], :tuple) ? ex.args[2].args : ()
    if f === Base.:^ && power_syntax && length(operands) == 2 && operands[2] isa Integer
        f = Base.literal_pow
        args = OutputSignatures.ValueInfo[OutputSignatures.known_info(Base.:^),
            args[1], OutputSignatures.known_info(Val(operands[2]))]
    end
    broadcasted || return _metadata_call(f, args; kwargs)
    isempty(kwargs) || return unknown("a broadcast call takes no keywords")
    return OutputSignatures.infer_call(broadcast,
        OutputSignatures.ValueInfo[OutputSignatures.known_info(f); args])
end

_metadata_call(f, args; kwargs) = OutputSignatures.infer_call(f, args; kwargs)

# A KernelSpec is already a visible graph, not an opaque Julia helper. Carry
# facts through its selected recipes without executing any of them. Partial
# facts deliberately remain :unknown: axes alone do not certify a numerical
# rule or authorize replacing a call with an elementwise operation.
function _metadata_call(spec::KernelSpec, args; kwargs)
    OS = OutputSignatures
    unknown = OS.CallInfo(OS.ValueInfo(), :unknown, "kernel boundary is unavailable")
    isempty(kwargs) && _kernel_inline_signature_supported(spec.call_signature) &&
        length(args) == length(inputs(spec)) &&
        length(outputs(spec)) == 1 || return unknown
    p = plan(spec)
    cid(v) = canon_id(p.graph, v.id)
    values = Dict{ValueId,OS.ValueInfo}(cid(v) => a for (v, a) in zip(p.have, args))
    for r in p.recipes
        operands = OS.ValueInfo[get(values, cid(v), OS.ValueInfo()) for v in r.inputs]
        result = _recipe_value_metadata(r, operands)
        for (v, fact) in zip(r.outputs, result)
            # A declared local can convert its assigned value. A conversion
            # whose input type is not already within that declaration may
            # change shape; retain only the declared type in that case.
            T = valtype(v)
            if !(r.op isa _BoundConstant) && !(OS.type_bound(fact) <: T)
                fact = OS.ValueInfo(; type_bound = T)
            end
            values[cid(v)] = fact
        end
    end
    OS.CallInfo(get(values, cid(only(p.want)), OS.ValueInfo()), :unknown,
        "facts from a visible kernel graph")
end

function _recipe_value_metadata(r::Recipe, args)
    OS = OutputSignatures
    unknown() = fill(OS.ValueInfo(), length(r.outputs))
    length(r.outputs) == 1 && !r.effectful || return unknown()
    if r.op isa _BoundConstant
        return [OS.known_info(r.op.value)]
    elseif r.op isa _AuthoredPlateOp
        op = r.op
        atomic = typeof(op).parameters[2]
        batched = OS.ValueInfo[a for (i, a) in enumerate(args) if !(i in atomic)]
        domain = OS.infer_call(broadcast, OS.ValueInfo[OS.known_info(op.kernel); batched])
        domain.status === :incompatible && return unknown()
        axes = OS.value_axes(domain.value)
        axes === nothing && return unknown()
        types = map(OS.type_bound, args)
        T = _cell_plate_output_type(op, types)
        T <: AbstractArray || return [OS.ValueInfo(; axes)]
        E = eltype(T)
        buffer = _plate_array_type(E, Val(atomic), types...)
        # Only a proved Base Array allocation carries standard numerical
        # array semantics. A custom marker's similar result keeps just facts.
        buffer !== Union{} && buffer <: Array ||
            return [OS.ValueInfo(; type_bound = T, element_type = E, axes)]
        dims = map(a -> a isa OS.SymbolicAxis ? a : OS.axis_length(a), axes)
        return [OS.array_info(E, dims...; container_type = buffer)]
    elseif r.op === identity && length(args) == 1
        return args
    elseif r.op isa _KernelSourceOp
        closure = _lane_source_closure(r)
        closure === nothing && return unknown()
        _, params, mod = closure
        bindings = _lane_source_arguments(params, args)
        return [_expression_value_info(_lane_strip_lines(r.source), bindings, mod;
            shadowed = params, partial = true)]
    end
    result = _metadata_call(r.op, args; kwargs = Pair{Symbol,OS.ValueInfo}[])
    result.status === :incompatible ? unknown() : [result.value]
end

# Return the result descriptor for a source call whose operand descriptors
# have already been computed. Unsupported or incompatible analysis leaves the
# authored operation intact at the caller; it is not a runtime error policy.
# An unsupported broadcast can still know its axes. Those partial facts
# alone do not authorize replacing its materialization with a plate.
function _call_value_info(ex::Expr, args::Vector{OutputSignatures.ValueInfo},
                          mod::Module, shadowed;
                          kwargs = Pair{Symbol,OutputSignatures.ValueInfo}[])
    result = _call_metadata(ex, args, mod, shadowed; kwargs)
    return result.status === :supported ? result.value : OutputSignatures.ValueInfo()
end

"""
    _expression_value_info(ex, values, mod; shadowed=keys(values), partial=false)

Propagate numerical metadata through a visible scalar/vector expression.
`values` maps lexical names to OutputSignatures descriptors, including unknown
intermediates. `shadowed` includes local names even when their facts are absent.
Opaque calls are queried by identity, never executed or recursively inferred.
`partial=true` also retains sound partial facts, such as axes, from unknown
calls and visible kernel graphs; these facts do not certify a numerical rule.
"""
function _expression_value_info(ex, values, mod::Module;
                                 shadowed = keys(values), partial::Bool = false)
    ex isa Symbol && return get(values, ex, OutputSignatures.ValueInfo())
    ex isa Number && return OutputSignatures.known_info(ex)
    ex isa QuoteNode && return OutputSignatures.known_info(ex.value)
    ex isa Expr || return OutputSignatures.ValueInfo()
    operands = if ex.head === :ref
        ex.args
    elseif ex.head === :call && !isempty(ex.args)
        ex.args[2:end]
    elseif Meta.isexpr(ex, :., 2) && Meta.isexpr(ex.args[2], :tuple)
        ex.args[2].args
    else
        return OutputSignatures.ValueInfo()
    end
    args = OutputSignatures.ValueInfo[]
    kwargs = Pair{Symbol,OutputSignatures.ValueInfo}[]
    for operand in operands
        if Meta.isexpr(operand, :parameters)
            for kw in operand.args
                Meta.isexpr(kw, :kw, 2) && kw.args[1] isa Symbol ||
                    return OutputSignatures.ValueInfo()
                push!(kwargs, kw.args[1] =>
                    _expression_value_info(kw.args[2], values, mod; shadowed, partial))
            end
        elseif Meta.isexpr(operand, :kw, 2) && operand.args[1] isa Symbol
            push!(kwargs, operand.args[1] =>
                _expression_value_info(operand.args[2], values, mod; shadowed, partial))
        else
            push!(args, _expression_value_info(operand, values, mod; shadowed, partial))
        end
    end
    if partial
        result = _call_metadata(ex, args, mod, shadowed; kwargs)
        return result.status === :incompatible ? OutputSignatures.ValueInfo() : result.value
    end
    return _call_value_info(ex, args, mod, shadowed; kwargs)
end
