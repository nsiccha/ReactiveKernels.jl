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

function _constructor_form(@nospecialize(x), inputs, locals, mod, stage_sources=false)
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
    _kernel_is_group_call(x) && return _constructor_form(x.args[1].args[2], inputs, locals, mod, stage_sources)
    if Meta.isexpr(x, :call, 4) && x.args[1] === GlobalRef(@__MODULE__, :_kernel_source_function)
        f, source, quoted_scope = x.args[2], x.args[3], x.args[4]
        if Meta.isexpr(f, :->, 2) && source isa QuoteNode && isequal(f, source.value) &&
                quoted_scope isa QuoteNode && quoted_scope.value === mod
            _kernel_mentions_any(f, locals) && throw(_UnsupportedConstructor())
            if stage_sources
                _kernel_source_native_scope(f.args[2]) && throw(_UnsupportedConstructor())
                # Compile the closed leaf once. Each replay creates its own recipe.
                return _constructor_input!(inputs, _kernel_source_function(nothing, f, mod))
            end
            return Expr(:call, GlobalRef(@__MODULE__, :_constructor_source),
                        _constructor_input!(inputs, f), mod)
        end
    end
    if x.head === :->
        _kernel_mentions_any(x, locals) && throw(_UnsupportedConstructor())
        stage_sources && throw(_UnsupportedConstructor())
        return Expr(:call, GlobalRef(Core, :eval), mod, _constructor_input!(inputs, x))
    elseif Meta.isexpr(x, :., 2) && x.args[2] isa QuoteNode && x.args[2].value isa Symbol
        return Expr(:., _constructor_form(x.args[1], inputs, locals, mod, stage_sources), x.args[2])
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
    Expr(x.head, Any[_constructor_form(a, inputs, locals, mod, stage_sources) for a in x.args]...)
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

# The generator can stage a direct positional call by its port types alone.
# Expanding this expression sees a child with the same boundary as the actual
# call. The emitted splice reads this build's operation and source at replay.
struct _KernelBuildCall
    index::Int
    arguments::Tuple
    inputs::Tuple
    outputs::Tuple
end

function _kernel_build_call end
function _kernel_resolve_graph_constructor(::typeof(_kernel_build_call), args, ::Module)
    length(args) == 1 && only(args) isa QuoteNode || return nothing
    call = only(args).value
    call isa _KernelBuildCall || return nothing
    g = Graph()
    ports = Dict{Symbol,Value}()
    function boundary(prefix, types)
        map(eachindex(types)) do i
            name = Symbol(prefix, i)
            ports[name] = value!(g, name, types[i])
            name
        end
    end
    have, want = boundary(:in, call.inputs), boundary(:out, call.outputs)
    KernelSpec(g, ports, [have; want], have, want)
end

_build_call_expr(call) = Expr(:call, GlobalRef(@__MODULE__, :_kernel_build_call), QuoteNode(call))
_build_call_slot(x) = Meta.isexpr(x, :call, 2) &&
    first(x.args) === GlobalRef(@__MODULE__, :_kernel_build_call) &&
    last(x.args) isa QuoteNode && last(x.args).value isa _KernelBuildCall ?
    last(x.args).value : nothing

struct _BuildRecipeOperand
    index::Int
    source::Bool
end

# Require a genuine top-level splice for each staged call. Refuse a transformed
# call (an inlined branch, plate cell or computed argument) before publishing.
function _bind_build_calls(x, seen)
    x isa Expr || return x
    if x.head === :call && length(x.args) >= 9 &&
            x.args[1] === GlobalRef(@__MODULE__, :_kernel_add!)
        slot = _build_call_slot(x.args[5])
        if slot !== nothing && x.args[9] isa QuoteNode
            rhs = x.args[9].value
            if Meta.isexpr(rhs, :call) && _build_call_slot(first(rhs.args)) === slot &&
                    Tuple(rhs.args[2:end]) == slot.arguments
                slot.index in seen && throw(_UnsupportedConstructor())
                push!(seen, slot.index)
                args = copy(x.args)
                args[5] = QuoteNode(_BuildRecipeOperand(slot.index, false))
                args[9] = QuoteNode(_BuildRecipeOperand(slot.index, true))
                return Expr(:call, args...)
            end
        end
    end
    _build_call_slot(x) === nothing || throw(_UnsupportedConstructor())
    Expr(x.head, Any[_bind_build_calls(a, seen) for a in x.args]...)
end

struct _BuildStep
    run::Any
    indices::Vector{Int}
end

struct _BuildConstructor
    steps::Vector{_BuildStep}
    local_count::Int
    operands::Vector{Any}
    dynamic::Vector{Pair{Int,_BuildRecipeOperand}}
end

function _build_callable(mod, form)
    source = Expr(:->, _CONSTRUCTOR_ARGUMENT, form)
    qualified = macroexpand(mod, Expr(Symbol("hygienic-scope"), source, mod))
    compile(qualified)
end

# Compile one assembly statement at a time. Different definitions then share
# Julia methods for e.g. declaring a typed port or splicing a two-input child,
# regardless of the port names, positions, or other recipes in either model.
const _BUILD_FRAME = gensym(:build_frame)

function _build_step_form(x, locals, indices)
    x isa Symbol && haskey(locals, x) && return Expr(:ref, _BUILD_FRAME, locals[x])
    x isa Expr || return x
    if Meta.isexpr(x, :call, 3) && x.args[1] === GlobalRef(Base, :getindex) &&
            x.args[2] === _CONSTRUCTOR_ARGUMENT && x.args[3] isa Int
        push!(indices, x.args[3])
        return Expr(:ref, _CONSTRUCTOR_ARGUMENT, length(indices))
    end
    Expr(x.head, Any[_build_step_form(a, locals, indices) for a in x.args]...)
end

function _build_steps(mod, form)
    while Meta.isexpr(form, :block)
        body = filter(x -> !(x isa LineNumberNode), form.args)
        length(body) == 1 || throw(_UnsupportedConstructor())
        form = only(body)
    end
    Meta.isexpr(form, :let, 2) && Meta.isexpr(form.args[1], :block, 0) &&
        Meta.isexpr(form.args[2], :block) || throw(_UnsupportedConstructor())
    steps = _BuildStep[]
    locals = Dict{Symbol,Int}()
    functions = Dict{Expr,Any}()
    for statement in form.args[2].args
        statement isa LineNumberNode && continue
        if Meta.isexpr(statement, :(=), 2) && statement.args[1] isa Symbol
            get!(locals, first(statement.args), length(locals)+1)
        end
        indices = Int[]
        body = _build_step_form(statement, locals, indices)
        source = Expr(:->, Expr(:tuple, _BUILD_FRAME, _CONSTRUCTOR_ARGUMENT), body)
        run = get!(functions, source) do
            compile(macroexpand(mod, Expr(Symbol("hygienic-scope"), source, mod)))
        end
        push!(steps, _BuildStep(run, indices))
    end
    steps, length(locals)
end

function _stage_build(mod, definition, count)
    construction = _kernel_definition_construction(mod, definition)
    seen = BitSet()
    construction = _bind_build_calls(construction, seen)
    length(seen) == count || throw(_UnsupportedConstructor())
    locals = _kernel_construction_locals!(Set{Symbol}(), construction)
    operands = Any[]
    form = _constructor_form(construction, operands, locals, mod, true)
    dynamic = Pair{Int,_BuildRecipeOperand}[i => x for (i,x) in pairs(operands)
                                            if x isa _BuildRecipeOperand]
    steps, local_count = _build_steps(mod, form)
    _BuildConstructor(steps, local_count, operands, dynamic)
end

function _current_build_operation(mod, rhs, @nospecialize(f))
    f isa KernelSpec && return f
    deps = Symbol[rhs.args[2:end]...]
    op = _kernel_operation(rhs, deps, Set(deps); mod)
    args = Any[]
    form = _constructor_form(op, args, Set{Symbol}(), mod, true)
    Base.invokelatest(_build_callable(mod, form), args)
end

function _replay_build(entry::_BuildConstructor, mod, calls)
    # Recipe sources and embedded endpoint fragments can be edited through a
    # returned graph. The owner keeps its own operands, including during a
    # concurrent replay. Each returned spec owns its prepared graph.
    memo = IdDict()
    args = Any[_deepcopy_code(x, memo) for x in entry.operands]
    for (i, slot) in entry.dynamic
        rhs, f = calls[slot.index]
        args[i] = slot.source ? rhs : _current_build_operation(mod, rhs, f)
    end
    frame = Vector{Any}(undef, entry.local_count)
    result = nothing
    for step in entry.steps
        result = Base.invokelatest(step.run, frame, view(args, step.indices))
    end
    result::KernelSpec
end
