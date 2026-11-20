# A generator may repeatedly build the same program with fresh helper graphs.
# Retain the lowered top-level instructions for a few programs in its own
# evaluation module. Never retain the arguments, graph, or recipe closures.
struct _KernelConstructors
    mod::Module
    thunks::Dict{Expr,Expr}
    lock::ReentrantLock
end
_KernelConstructors(mod::Module) = _KernelConstructors(mod, Dict{Expr,Expr}(), ReentrantLock())

const _CONSTRUCTOR_INPUTS = gensym(:constructor_inputs)
const _CONSTRUCTOR_ARGUMENT = gensym(:constructor_argument)
_constructor_inputs() = task_local_storage(_CONSTRUCTOR_INPUTS)::Vector{Any}

function _constructor_input!(inputs, @nospecialize(value))
    push!(inputs, value)
    Expr(:call, GlobalRef(Base, :getindex), _CONSTRUCTOR_ARGUMENT, length(inputs))
end

# _kernel_eval_construction evaluates recipe closures at top level, so they
# have no lexical captures. An ordinary source record takes its implementation
# from the source itself. Only a native Julia scope needs the closure value.
function _constructor_source(source::Expr, mod::Module)
    f = _kernel_source_native_scope(source.args[2]) ? Core.eval(mod, source) : nothing
    _kernel_source_function(f, source, mod)
end

struct _UnsupportedConstructor <: Exception end

# Julia gives some surface calls special lowering based on their syntax
# (e.g. x^-1 selects literal_pow). Only lift arguments out of calls emitted
# by the graph builder itself. Authored recipe bodies travel intact as inputs.
function _constructor_call(callee)
    Meta.isexpr(callee, :curly) && return true
    callee === GlobalRef(Base, :Val) && return true
    callee isa GlobalRef && callee.mod === (@__MODULE__) || return false
    callee.name in (:Graph, :KernelSpec) && return true
    name = String(callee.name)
    startswith(name, "_kernel_") || startswith(name, "_Kernel")
end

function _constructor_form(@nospecialize(x), inputs, locals, mod)
    x isa LineNumberNode && return x
    x === _CONSTRUCTOR_ARGUMENT && throw(_UnsupportedConstructor())
    x isa Symbol && return x
    x isa QuoteNode && return _constructor_input!(inputs, x.value)
    if x isa GlobalRef
        x.mod in (Core, Base, (@__MODULE__), mod) && return x
        # Read the binding when the instruction executes. An earlier call
        # can change it, and a read in an untaken branch must stay untaken.
        return Expr(:call, GlobalRef(Core, :getglobal),
                    _constructor_input!(inputs, x.mod), _constructor_input!(inputs, x.name))
    end
    x isa Expr || return _constructor_input!(inputs, x)
    _kernel_is_group_call(x) && return _constructor_form(x.args[1].args[2], inputs, locals, mod)
    if Meta.isexpr(x, :call, 4) && x.args[1] === GlobalRef(@__MODULE__, :_kernel_source_function)
        f, source, scope = x.args[2], x.args[3], x.args[4]
        if Meta.isexpr(f, :->, 2) && source isa QuoteNode && isequal(f, source.value) &&
                scope isa QuoteNode && scope.value === mod
            _kernel_mentions_any(f, locals) && throw(_UnsupportedConstructor())
            return Expr(:call, GlobalRef(@__MODULE__, :_constructor_source),
                        _constructor_input!(inputs, f), mod)
        end
    end
    if x.head === :->
        _kernel_mentions_any(x, locals) && throw(_UnsupportedConstructor())
        return Expr(:call, GlobalRef(Core, :eval), mod, _constructor_input!(inputs, x))
    elseif Meta.isexpr(x, :., 2) && x.args[2] isa QuoteNode && x.args[2].value isa Symbol
        return Expr(:., _constructor_form(x.args[1], inputs, locals, mod), x.args[2])
    elseif x.head === :call
        _constructor_call(first(x.args)) || throw(_UnsupportedConstructor())
    elseif x.head === :(=)
        # A short-form function definition lowers a fresh closure type. Its
        # definition cannot run twice from the same lowered thunk.
        lhs = first(x.args)
        lhs isa Symbol || Meta.isexpr(lhs, :ref) || throw(_UnsupportedConstructor())
    end
    # Only descend through syntax in RK's constructor vocabulary. In
    # particular quote interpolation and new Julia scopes need Julia's own
    # lowering. Decide eligibility before executing any instructions.
    x.head in (:let, :block, :(=), :ref, :call, :curly, :vect, :tuple, :parameters,
               :kw, :(...), :if, :&&, :||) || throw(_UnsupportedConstructor())
    Expr(x.head, Any[_constructor_form(a, inputs, locals, mod) for a in x.args]...)
end

function _kernel_eval_definition(cache::_KernelConstructors, definition::Expr)
    mod = cache.mod
    construction = _kernel_definition_construction(mod, definition)
    _image_target_module() === nothing || return _kernel_eval_construction(mod, construction)
    locals = _kernel_construction_locals!(Set{Symbol}(), construction)
    inputs = Any[]
    body = try
        _constructor_form(construction, inputs, locals, mod)
    catch err
        err isa _UnsupportedConstructor || rethrow()
        return _kernel_eval_construction(mod, construction)
    end
    # RK names its graph-building locals by definition and position. Keep
    # those names verbatim: a symbol can also name a keyword or a global.
    binding = Expr(:(=), _CONSTRUCTOR_ARGUMENT, Expr(:call, GlobalRef(@__MODULE__, :_constructor_inputs)))
    form = Expr(:let, binding, body)
    thunk = lock(cache.lock) do
        existing = get(cache.thunks, form, nothing)
        existing === nothing || return existing
        lowered = Meta.lower(mod, form)
        Meta.isexpr(lowered, :thunk) || return nothing
        # A backend can construct several related programs per fit. Keep a
        # bounded working set even in a service that sees unrelated models.
        length(cache.thunks) >= 8 && empty!(cache.thunks)
        cache.thunks[form] = lowered
    end
    thunk === nothing && return _kernel_eval_construction(mod, construction)
    # Julia's top-level interpreter takes a thunk with no arguments. Dynamic
    # scope restores the caller's vector on nested builds and on exceptions.
    task_local_storage(() -> Core.eval(mod, thunk), _CONSTRUCTOR_INPUTS, inputs)
end
