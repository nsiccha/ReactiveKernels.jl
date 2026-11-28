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

function _constructor_form(@nospecialize(x), inputs, locals, mod, stage_sources=false, scope=nothing)
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
    _kernel_is_group_call(x) &&
        return _constructor_form(x.args[1].args[2], inputs, locals, mod, stage_sources, scope)
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
            source_input = _constructor_input!(inputs, f)
            eval_mod = scope === nothing ? mod : _constructor_input!(inputs, scope)
            return Expr(:call, GlobalRef(@__MODULE__, :_constructor_source), source_input, eval_mod)
        end
    end
    if x.head === :->
        _kernel_mentions_any(x, locals) && throw(_UnsupportedConstructor())
        stage_sources && throw(_UnsupportedConstructor())
        eval_mod = scope === nothing ? mod : _constructor_input!(inputs, scope)
        return Expr(:call, GlobalRef(Core, :eval), eval_mod, _constructor_input!(inputs, x))
    elseif Meta.isexpr(x, :., 2) && x.args[2] isa QuoteNode && x.args[2].value isa Symbol
        return Expr(:., _constructor_form(x.args[1], inputs, locals, mod, stage_sources, scope), x.args[2])
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
    Expr(x.head, Any[_constructor_form(a, inputs, locals, mod, stage_sources, scope) for a in x.args]...)
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
    name::Symbol
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
end

# Metadata can contain the entire authored plate, including calls inside its
# cell. Keep their callees as slots, even when lowering lifted the arguments
# into intermediate ports. The operation itself must have a real splice.
struct _BuildRecipeSource
    source::Expr
end

function _has_build_call(x)
    x isa QuoteNode && return _has_build_call(x.value)
    x isa Expr || return false
    _build_call_slot(x) !== nothing || any(_has_build_call, x.args)
end

function _build_recipe_source(x, calls)
    x isa QuoteNode && return QuoteNode(_build_recipe_source(x.value, calls))
    x isa Expr || return x
    slot = _build_call_slot(x)
    slot === nothing || return first(calls[slot.index][1].args)
    Expr(x.head, Any[_build_recipe_source(a, calls) for a in x.args]...)
end

function _bind_build_calls(x, seen)
    x isa QuoteNode && _has_build_call(x) && throw(_UnsupportedConstructor())
    x isa Expr || return x
    if x.head === :call && length(x.args) >= 9 &&
            x.args[1] === GlobalRef(@__MODULE__, :_kernel_add!)
        args = copy(x.args)
        slot = _build_call_slot(x.args[5])
        if slot !== nothing
            push!(seen, slot.index)
            args[5] = QuoteNode(_BuildRecipeOperand(slot.index))
        end
        if args[9] isa QuoteNode && args[9].value isa Expr && _has_build_call(args[9])
            args[9] = QuoteNode(_BuildRecipeSource(args[9].value))
        end
        return Expr(:call, Any[_bind_build_calls(a, seen) for a in args]...)
    end
    _build_call_slot(x) === nothing || throw(_UnsupportedConstructor())
    Expr(x.head, Any[_bind_build_calls(a, seen) for a in x.args]...)
end

# One assembly instruction. `run(frame, operands, indices, slots)` reads build
# local `j` as `frame[slots[j]]` and operand `j` as `operands[indices[j]]`.
struct _BuildStep
    run::Any
    indices::Vector{Int}
    slots::Vector{Int}
end

struct _BuildConstructor
    steps::Vector{_BuildStep}
    local_count::Int
    operands::Vector{Any}
    dynamic::Vector{Int}
end

function _build_callable(mod, form)
    source = Expr(:->, _CONSTRUCTOR_ARGUMENT, form)
    qualified = macroexpand(mod, Expr(Symbol("hygienic-scope"), source, mod))
    compile(qualified)
end

# Compile one assembly call at a time. Each instruction reads its locals and
# operands by position, so its Julia method depends only on the call's shape.
# Different definitions then share the methods for e.g. declaring a typed port,
# building a recipe or splicing a two-input child, regardless of the port
# names, positions, recipes or the number of statements in either model.
# A recipe group, or the plate cell graph built inside a call's argument, is
# unnested into such calls too. Calls keep Julia's left-to-right evaluation
# order. Syntax with its own evaluation rules (a branch, keyword arguments,
# splatting, `end` in an index) stays in one instruction.
const _BUILD_FRAME = gensym(:build_frame)
const _BUILD_INDICES = gensym(:build_indices)
const _BUILD_SLOTS = gensym(:build_slots)

_build_operand(@nospecialize(x)) = Meta.isexpr(x, :call, 3) &&
    x.args[1] === GlobalRef(Base, :getindex) && x.args[2] === _CONSTRUCTOR_ARGUMENT &&
    x.args[3] isa Int

# Read in place by the instruction that uses it: a name, a literal, an operand
# or a type expression of constant parameters, e.g. `Dict{Symbol,Value}`.
function _build_inline(@nospecialize(x))
    x isa Expr || return true
    _build_operand(x) && return true
    x.head === :curly && all(a -> !(a isa Symbol) && !_build_operand(a) && _build_inline(a), x.args)
end

_build_mentions_end(@nospecialize(x)) = x === :end || x === :begin ||
    x isa Expr && any(_build_mentions_end, x.args)

# A call, tuple, array, index or field read whose parts evaluate left to right.
function _build_node(@nospecialize(x))
    x isa Expr || return false
    x.head === :. && return length(x.args) == 2 && x.args[2] isa QuoteNode
    x.head in (:call, :tuple, :vect, :ref, :curly) || return false
    args = x.head === :call ? x.args[2:end] : x.args
    any(a -> Meta.isexpr(a, (:parameters, :kw, :..., :(=))), args) && return false
    !(x.head === :ref && any(_build_mentions_end, x.args[2:end]))
end
_build_parts(x::Expr) = x.head === :. ? Any[x.args[1]] : x.args
_build_rebuild(x::Expr, leaves) =
    x.head === :. ? Expr(:., only(leaves), x.args[2]) : Expr(x.head, leaves...)

_build_sequence(@nospecialize(x)) = Meta.isexpr(x, :block) ||
    Meta.isexpr(x, :let, 2) && Meta.isexpr(x.args[1], :block, 0)
_build_statements(x::Expr) =
    filter(a -> !(a isa LineNumberNode), (x.head === :let ? x.args[2] : x).args)

function _build_assigned!(names, @nospecialize(x))
    x isa Expr || return names
    Meta.isexpr(x, :(=), 2) && x.args[1] isa Symbol && push!(names, x.args[1])
    for a in x.args
        _build_assigned!(names, a)
    end
    names
end

# The instructions of one assembly, in evaluation order: `name => value`
# assigns a build local, `nothing => value` only evaluates. `locals` holds
# every name the construction assigns; any other name is a global. A
# temporary is a fresh local that only its own instruction assigns.
struct _BuildFlat
    out::Vector{Pair{Union{Nothing,Symbol},Any}}
    locals::Set{Symbol}
    temps::Set{Symbol}
end

# Can a read of `x` change before a later part of the same call executes?
function _build_mutable_read(@nospecialize(x), assigned, flat)
    x isa Symbol && return !(x in flat.temps) && (x in assigned || !(x in flat.locals))
    x isa GlobalRef && return !_global_isconst(x.mod, x.name)
    false
end

function _build_temp!(flat, @nospecialize(x))
    t = gensym(:build_value)
    push!(flat.temps, t)
    push!(flat.out, t => x)
    t
end

# Evaluate `parts` in order. A part that is a call gets its own instructions;
# a read of a name that a later such part can assign takes its value first.
function _build_parts!(flat, parts)
    n = length(parts)
    hoist = Bool[!_build_inline(p) for p in parts]
    stop = something(findlast(hoist), 0)
    after = [Set{Symbol}() for _ in 1:n]
    for i in stop-1:-1:1
        union!(after[i], after[i+1])
        _build_assigned!(after[i], parts[i+1])
    end
    leaves = Vector{Any}(undef, n)
    for i in 1:n
        leaf = hoist[i] ? _build_value!(flat, parts[i]) : parts[i]
        i < stop && _build_mutable_read(leaf, after[i], flat) && (leaf = _build_temp!(flat, leaf))
        leaves[i] = leaf
    end
    leaves
end
_build_node!(flat, x::Expr) = _build_rebuild(x, _build_parts!(flat, _build_parts(x)))

# `a[i...] = v` evaluates `v` before `a` and `i`.
_build_index_assignment(@nospecialize(x)) = Meta.isexpr(x, :(=), 2) &&
    Meta.isexpr(x.args[1], :ref) && length(x.args[1].args) >= 1 &&
    !any(a -> Meta.isexpr(a, (:parameters, :kw, :..., :(=))), x.args[1].args) &&
    !any(_build_mentions_end, x.args[1].args[2:end])

function _build_index_assign!(flat, x)
    leaves = _build_parts!(flat, Any[x.args[2]; x.args[1].args])
    Expr(:(=), Expr(:ref, leaves[2:end]...), leaves[1]), leaves[1]
end

# The value of `a = v` or `a[i] = v` is `v`, even once something reassigns `a`.
_build_kept!(flat, @nospecialize(value)) =
    value isa GlobalRef || value isa Symbol && !(value in flat.temps) ?
        _build_temp!(flat, value) : value

# Emit `x`, whose value is unused, as instructions.
function _build_statement!(flat, @nospecialize(x))
    x isa LineNumberNode && return flat
    if _build_sequence(x)
        for s in _build_statements(x)
            _build_statement!(flat, s)
        end
    elseif Meta.isexpr(x, :(=), 2) && x.args[1] isa Symbol
        lhs, rhs = x.args
        value = _build_node(rhs) ? _build_node!(flat, rhs) :
            _build_inline(rhs) ? rhs : _build_value!(flat, rhs)
        push!(flat.out, lhs => value)
    elseif _build_index_assignment(x)
        push!(flat.out, nothing => first(_build_index_assign!(flat, x)))
    elseif _build_node(x)
        push!(flat.out, nothing => _build_node!(flat, x))
    else
        push!(flat.out, nothing => x)
    end
    flat
end

# Emit `x` as instructions. Return what reads its value afterwards: a name,
# a literal or an operand.
function _build_value!(flat, @nospecialize(x))
    _build_inline(x) && return x
    if _build_sequence(x)
        body = _build_statements(x)
        isempty(body) && return nothing
        for s in body[1:end-1]
            _build_statement!(flat, s)
        end
        return _build_value!(flat, body[end])
    elseif Meta.isexpr(x, :(=), 2) && x.args[1] isa Symbol
        value = _build_kept!(flat, _build_value!(flat, x.args[2]))
        push!(flat.out, x.args[1] => value)
        return value
    elseif _build_index_assignment(x)
        assignment, value = _build_index_assign!(flat, x)
        value = _build_kept!(flat, value)
        push!(flat.out, nothing => assignment)
        return value
    end
    _build_temp!(flat, _build_node(x) ? _build_node!(flat, x) : x)
end

function _build_step_form(x, slot, indices, slots)
    if x isa Symbol && haskey(slot, x)
        push!(slots, slot[x])
        return Expr(:ref, _BUILD_FRAME, Expr(:ref, _BUILD_SLOTS, length(slots)))
    end
    x isa Expr || return x
    if _build_operand(x)
        push!(indices, x.args[3])
        return Expr(:ref, _CONSTRUCTOR_ARGUMENT, Expr(:ref, _BUILD_INDICES, length(indices)))
    end
    # A keyword's name is not a read; `f(; a)` passes the value of `a` as `a`.
    Meta.isexpr(x, :kw, 2) && x.args[1] isa Symbol &&
        return Expr(:kw, x.args[1], _build_step_form(x.args[2], slot, indices, slots))
    if x.head === :parameters
        return Expr(:parameters, Any[a isa Symbol && haskey(slot, a) ?
            Expr(:kw, a, _build_step_form(a, slot, indices, slots)) :
            _build_step_form(a, slot, indices, slots) for a in x.args]...)
    end
    Expr(x.head, Any[_build_step_form(a, slot, indices, slots) for a in x.args]...)
end

function _build_steps(mod, form)
    while Meta.isexpr(form, :block)
        body = filter(x -> !(x isa LineNumberNode), form.args)
        length(body) == 1 || throw(_UnsupportedConstructor())
        form = only(body)
    end
    Meta.isexpr(form, :let, 2) && Meta.isexpr(form.args[1], :block, 0) &&
        Meta.isexpr(form.args[2], :block) || throw(_UnsupportedConstructor())
    flat = _BuildFlat(Pair{Union{Nothing,Symbol},Any}[], _build_assigned!(Set{Symbol}(), form), Set{Symbol}())
    result = _build_value!(flat, form)
    out = flat.out
    (isempty(out) || first(out[end]) !== result) && push!(out, nothing => result)
    slot = Dict{Symbol,Int}()
    for (target, _) in out
        target === nothing || get!(slot, target, length(slot)+1)
    end
    steps = _BuildStep[]
    functions = Dict{Expr,Any}()
    for (target, value) in out
        indices, slots = Int[], Int[]
        body = if target === nothing
            _build_step_form(value, slot, indices, slots)
        else
            push!(slots, slot[target])
            lhs = Expr(:ref, _BUILD_FRAME, Expr(:ref, _BUILD_SLOTS, 1))
            Expr(:(=), lhs, _build_step_form(value, slot, indices, slots))
        end
        source = Expr(:->, Expr(:tuple, _BUILD_FRAME, _CONSTRUCTOR_ARGUMENT,
                                _BUILD_INDICES, _BUILD_SLOTS), body)
        run = get!(functions, source) do
            compile(macroexpand(mod, Expr(Symbol("hygienic-scope"), source, mod)))
        end
        push!(steps, _BuildStep(run, indices, slots))
    end
    steps, length(slot)
end

function _stage_build(mod, definition, count)
    construction = try
        _kernel_definition_construction(mod, definition)
    catch err
        err isa LoadError && err.error isa _UnsupportedConstructor && throw(err.error)
        rethrow()
    end
    seen = BitSet()
    construction = _bind_build_calls(construction, seen)
    length(seen) == count || throw(_UnsupportedConstructor())
    locals = _kernel_construction_locals!(Set{Symbol}(), construction)
    operands = Any[]
    form = _constructor_form(construction, operands, locals, mod, true)
    dynamic = findall(x -> x isa Union{_BuildRecipeOperand,_BuildRecipeSource}, operands)
    steps, local_count = _build_steps(mod, form)
    _BuildConstructor(steps, local_count, operands, dynamic)
end

# Expansion of a nested child (e.g. a call inside a plate) reads the child's
# actual graph. Repeat that expansion, but share the instructions for
# assembling its result. All source records, modules, values and types from
# this invocation travel in operands. Only the instruction form is reusable.
function _current_build_form(mod, definition)
    construction = _kernel_definition_construction(mod, definition)
    locals = _kernel_construction_locals!(Set{Symbol}(), construction)
    operands = Any[]
    scope = _kernel_eval_module(mod, construction)
    form = _constructor_form(construction, operands, locals, mod, false, scope)
    form, operands
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
    for i in entry.dynamic
        slot = entry.operands[i]
        if slot isa _BuildRecipeSource
            args[i] = _build_recipe_source(slot.source, calls)
        else
            rhs, f = calls[slot.index]
            args[i] = _current_build_operation(mod, rhs, f)
        end
    end
    _run_build_steps(entry.steps, entry.local_count, args)
end

function _run_build_steps(steps, local_count, args)
    frame = Vector{Any}(undef, local_count)
    result = nothing
    for step in steps
        result = Base.invokelatest(step.run, frame, args, step.indices, step.slots)
    end
    result::KernelSpec
end
