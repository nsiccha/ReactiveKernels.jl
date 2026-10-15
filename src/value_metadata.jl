# Cold analysis of compiler-visible Julia calls. OutputSignatures owns all
# numerical transfer rules; this layer only resolves source bindings and
# translates Julia's call/index/broadcast syntax. No user function is run.

function _metadata_callable(callee, mod::Module, shadowed)
    callee isa Symbol && callee in shadowed && return nothing
    if Meta.isexpr(callee, :., 2) && callee.args[1] isa Symbol
        callee.args[1] in shadowed && return nothing
    end
    return _kernel_resolve_binding(mod, callee)
end

# Return the result descriptor for a source call whose operand descriptors
# have already been computed. Unsupported or incompatible analysis leaves the
# authored operation intact at the caller; it is not a runtime error policy.
function _call_value_info(ex::Expr, args::Vector{OutputSignatures.ValueInfo},
                          mod::Module, shadowed;
                          kwargs = Pair{Symbol,OutputSignatures.ValueInfo}[])
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
        return OutputSignatures.ValueInfo()
    end
    f === nothing && return OutputSignatures.ValueInfo()
    # Julia lowers a literal exponent in x^n / x.^n to literal_pow. In
    # particular integer bases with negative literal powers produce floats;
    # the ordinary ^ call can throw instead. Normalize syntax, not a guessed
    # numerical rule, and keep alias calls (pow(x, n)) as ordinary calls.
    if f === Base.:^ && ex.head === :call && length(ex.args) == 3 &&
            ex.args[1] in (:^, Symbol(".^")) && ex.args[3] isa Integer
        f = Base.literal_pow
        args = OutputSignatures.ValueInfo[OutputSignatures.known_info(Base.:^),
            args[1], OutputSignatures.known_info(Val(ex.args[3]))]
    end
    result = if broadcasted
        isempty(kwargs) || return OutputSignatures.ValueInfo()
        OutputSignatures.infer_call(broadcast,
            OutputSignatures.ValueInfo[OutputSignatures.known_info(f); args])
    else
        OutputSignatures.infer_call(f, args; kwargs)
    end
    # An unsupported broadcast can still know its axes. Those partial facts
    # alone do not authorize replacing its materialization with a plate.
    return result.status === :supported ? result.value : OutputSignatures.ValueInfo()
end

"""
    _expression_value_info(ex, values, mod; shadowed=keys(values))

Propagate numerical metadata through a visible scalar/vector expression.
`values` maps lexical names to OutputSignatures descriptors, including unknown
intermediates. `shadowed` includes local names even when their facts are absent.
Opaque calls are queried by identity, never executed or recursively inferred.
"""
function _expression_value_info(ex, values, mod::Module;
                                 shadowed = keys(values))
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
                    _expression_value_info(kw.args[2], values, mod; shadowed))
            end
        elseif Meta.isexpr(operand, :kw, 2) && operand.args[1] isa Symbol
            push!(kwargs, operand.args[1] =>
                _expression_value_info(operand.args[2], values, mod; shadowed))
        else
            push!(args, _expression_value_info(operand, values, mod; shadowed))
        end
    end
    return _call_value_info(ex, args, mod, shadowed; kwargs)
end
