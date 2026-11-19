# The cache sits after submodel expansion and all inspection of the current
# inputs. Its key describes the actual arguments to the structural pass.
# In particular a computed row partition is evaluated even on a hit.
struct _LoweringEntry
    plan::StructuralPlan
    # Locations of references to the caller's module in the private template.
    # The template names this file's sentinel module instead of retaining a
    # builder's otherwise unreachable namespace (and everything bound in it).
    caller_refs::Vector{Tuple{Expr,Int}}
    next_name::Int
end
module _LoweringCaller end

"""
    LoweringCache()

Own the reusable structural work of calls to `lower_rkppl(ast, data; cache)`.
A package may keep the cache in its own image. Every call supplies its current
data and optional frozen record. Input classifications, including the results
of computed index definitions, are evaluated at each call. Unrecognized source
or dependencies take the ordinary lowering path. Each result owns its mutable
plan storage. Drop the cache to release its entries.
"""
struct LoweringCache
    lock::ReentrantLock
    entries::Dict{Any,_LoweringEntry}
end
LoweringCache() = LoweringCache(ReentrantLock(), Dict{Any,_LoweringEntry}())

# Only persist module references whose owner already has a durable binding.
# The caller itself may be a fresh, unbound Module: its references use slots.
function _lowering_named_module(m::Module)
    m in (Base, Core, Main) && return true
    # Builders can bind generated modules under their package during image
    # construction. Such a module still owns the builder's private globals.
    Base.isgensym(nameof(m)) && return false
    p = parentmodule(m)
    if p === m
        id = Base.PkgId(m)
        return Base.root_module_exists(id) && Base.root_module(id) === m
    end
    name = nameof(m)
    return ReactiveKernels._global_isconst(p, name) &&
        ReactiveKernels._global_value(p, name) === Some(m) &&
        _lowering_named_module(p)
end

# A kernel can hold numeric constants and preparations. Retain only its port
# identities and graph revision. Definitions constructed anew can miss.
function _lowering_binding(value::KernelSpec)
    # Default providers can close over private modules or fitted values.
    sig = value.call_signature
    sig === nothing || (sig isa ReactiveKernels._KernelCallSignature &&
        all(p -> p === ReactiveKernels._KERNEL_REQUIRED_ARGUMENT, sig.defaults)) || return nothing
    all(n -> haskey(value.ports, n), value.port_order) || return nothing
    return Some((; signature = typeof(sig), ports = [(n, value.ports[n].id)
        for n in value.port_order], have = copy(value.have_names),
        want = copy(value.want_names), version = value.graph.version))
end
function _lowering_binding(@nospecialize(value))
    value === Colon() && return Some(value)
    if value isa Function && Base.issingletontype(typeof(value)) &&
            isempty(typeof(value).parameters) && _lowering_named_module(parentmodule(typeof(value))) &&
            ReactiveKernels._global_value(parentmodule(typeof(value)), nameof(value)) === Some(value)
        return Some(value)
    elseif value isa Module && _lowering_named_module(value)
        return Some(value)
    elseif value isa Type && (value isa DataType || value isa UnionAll) &&
            isdefined(parentmodule(value), nameof(value)) &&
            getglobal(parentmodule(value), nameof(value)) === value &&
            _lowering_named_module(parentmodule(value))
        return Some(value)
    end
    return nothing
end

struct _UncacheableLowering end
const _UNCACHEABLE_LOWERING = _UncacheableLowering()

# Read (including the absence of) the binding without forcing an import into
# the caller's namespace. Julia's lookup must still see later definitions.
function _lowering_global(mod::Module, name::Symbol)
    binding = ReactiveKernels._global_value(mod, name)
    binding === nothing && return nothing
    value = something(binding)
    if haskey(_BASE_NUMBERS, name) && ReactiveKernels._binding_owner(mod, name) in
            (Base, Base.MathConstants)
        return (GlobalRef, Base, name)
    end
    guard = _lowering_binding(value)
    guard === nothing && return _UNCACHEABLE_LOWERING
    return (ReactiveKernels._global_isconst(mod, name), something(guard))
end

# Julia compares numeric literals of different types by value. Tag them in
# the key: a model helper may dispatch on the literal's actual type.
_lowering_syntax(x::Number, mod, bindings) = isprimitivetype(typeof(x)) &&
    parentmodule(typeof(x)) in (Base, Core) ?
    (typeof(x), x) : _UNCACHEABLE_LOWERING
_lowering_syntax(x::Union{LineNumberNode,String,Char,Nothing}, mod, bindings) = x
_lowering_syntax(x, mod, bindings) = _UNCACHEABLE_LOWERING
function _lowering_syntax(x::Symbol, mod, bindings)
    value = get!(bindings, x) do
        _lowering_global(mod, x)
    end
    return value === _UNCACHEABLE_LOWERING ? value : x
end
function _lowering_syntax(x::GlobalRef, mod, bindings)
    x.mod === mod || _lowering_named_module(x.mod) || return _UNCACHEABLE_LOWERING
    value = _lowering_global(x.mod, x.name)
    value === _UNCACHEABLE_LOWERING && return value
    scope = x.mod === mod ? _LoweringCaller : x.mod
    bindings[(scope, x.name)] = value
    return (GlobalRef, scope, x.name)
end
function _lowering_syntax(x::QuoteNode, mod, bindings)
    # Expansion quotes values pinned inside submodels, including scalars.
    return x.value isa Symbol ? (QuoteNode, x.value) : _UNCACHEABLE_LOWERING
end
# Inspect a possibly qualified module path without binding an import. A name
# shadowed by a model local can conservatively add an extra dependency.
function _lowering_module_path(x, mod)
    if x isa Symbol
        binding = ReactiveKernels._global_value(mod, x)
    elseif Meta.isexpr(x, :., 2) && x.args[2] isa QuoteNode && x.args[2].value isa Symbol
        parent = _lowering_module_path(x.args[1], mod)
        parent === nothing && return nothing
        binding = ReactiveKernels._global_value(parent, x.args[2].value)
    else
        return nothing
    end
    binding === nothing && return nothing
    value = something(binding)
    return value isa Module ? value : nothing
end

function _lowering_syntax(x::Expr, mod, bindings)
    # Caller-defined sampling geometry can run arbitrary code during the
    # structural pass. Likewise a mixture can contain an external law.
    if (_is_sample(x) || _is_broadcast_sample(x)) &&
            (_external_rhs(x.args[3]) || _mentions_symbol(x.args[3], :MixtureModel))
        return _UNCACHEABLE_LOWERING
    end
    # Qualified reads of modules must check the terminal binding too. A
    # property of a model value is already described by the model's syntax.
    if Meta.isexpr(x, :., 2) && x.args[2] isa QuoteNode && x.args[2].value isa Symbol
        parent = _lowering_module_path(x.args[1], mod)
        if parent !== nothing
            _lowering_named_module(parent) || return _UNCACHEABLE_LOWERING
            name = x.args[2].value
            value = _lowering_global(parent, name)
            value === _UNCACHEABLE_LOWERING && return value
            bindings[(parent, name)] = value
        end
    end
    args = Any[]
    for arg in x.args
        item = _lowering_syntax(arg, mod, bindings)
        item === _UNCACHEABLE_LOWERING && return item
        push!(args, item)
    end
    return (Expr, x.head, args)
end

# Find the lexical call heads to relocate in a private copy of the plan.
# A reference outside an expression slot, or a descriptor supplied by a
# client, has no relocation here: leave that program to fresh lowering.
function _lowering_slots!(slots, x, mod, bindings)
    if x isa Expr
        for (i, arg) in pairs(x.args)
            if arg isa GlobalRef && arg.mod === mod
                # A lexical dependency introduced by a rewrite needs its own
                # guard. Leave it to ordinary lowering until it has one.
                (haskey(bindings, arg.name) || haskey(bindings, (_LoweringCaller, arg.name))) ||
                    return false
                push!(slots, (x, i))
            else
                _lowering_slots!(slots, arg, mod, bindings) || return false
            end
        end
    elseif x isa GlobalRef
        x.mod !== mod && _lowering_named_module(x.mod) || return false
        haskey(bindings, (x.mod, x.name)) && return true
        # Compiler-inserted references can also name the package's fixed
        # helpers. Mutable kernel graphs need a source guard for their revision.
        binding = ReactiveKernels._global_value(x.mod, x.name)
        return binding !== nothing && ReactiveKernels._global_isconst(x.mod, x.name) &&
            something(binding) isa Union{Function,Type} &&
            _lowering_binding(something(binding)) !== nothing
    elseif x isa Union{Nothing,Symbol,String,LineNumberNode,Char,Bool} ||
            (x isa Number && isprimitivetype(typeof(x)) && parentmodule(typeof(x)) in (Base, Core))
        return true
    elseif x isa Dict
        return all(p -> _lowering_slots!(slots, p, mod, bindings), x)
    elseif x isa Union{Array,Set,Tuple,NamedTuple}
        return all(a -> _lowering_slots!(slots, a, mod, bindings), x)
    elseif x isa Union{QuoteNode,Pair} || parentmodule(typeof(x)) === @__MODULE__
        return all(i -> _lowering_slots!(slots, getfield(x, i), mod, bindings), 1:nfields(x))
    else
        return false
    end
    return true
end

function _replay_lowering(entry::_LoweringEntry, mod)
    plan, refs = deepcopy((entry.plan, entry.caller_refs))
    for (ex, i) in refs
        ex.args[i] = GlobalRef(mod, ex.args[i].name)
    end
    return plan
end

function _cached_lowering(cache::LoweringCache, ast, data, mod;
        submodel_scopes, conditioned, array_entries, observing)
    lower() = _lower_rkppl_once(ast, data, mod; submodel_scopes, conditioned,
        array_entries, observing)
    bindings = Dict{Union{Tuple{Module,Symbol},Symbol},Any}()
    syntax = _lowering_syntax(ast, mod, bindings)
    syntax === _UNCACHEABLE_LOWERING && return lower()
    names = task_local_storage(ReactiveKernels._KERNEL_EXPANSION_NAMES)
    key = (syntax, bindings, copy(data), copy(conditioned), copy(array_entries), copy(observing),
        names.fingerprint, names.count,
        _lowering_named_module(mod) ? Base.fullname(mod) : nothing)
    entry = lock(() -> get(cache.entries, key, nothing), cache.lock)
    if entry !== nothing
        names.count = entry.next_name
        return _with(_replay_lowering(entry, mod); submodel_scopes)
    end

    plan = lower()
    # No lexical submodel record, documentation, frozen record or data store
    # belongs to this template. The calling lowering attaches its fresh ones.
    template = _with(plan; submodel_scopes = SubmodelScope[])
    slots = Tuple{Expr,Int}[]
    if _lowering_slots!(slots, template, mod, bindings)
        template, slots = deepcopy((template, slots))
        for (ex, i) in slots
            ex.args[i] = GlobalRef(_LoweringCaller, ex.args[i].name)
        end
        lock(cache.lock) do
            get!(cache.entries, key, _LoweringEntry(template, slots, names.count))
        end
    end
    return plan
end
