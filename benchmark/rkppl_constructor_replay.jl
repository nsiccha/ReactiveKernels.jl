"""
An isolated Julia 1.10 experiment for repeated RKPPL kernel construction.
Include this file and call `KernelConstructionReplay.install!()` in a benchmark
process before building models. Read `KernelConstructionReplay.counts()` to
distinguish executions of cached constructors from newly lowered constructors.
"""
module KernelConstructionReplay

import ReactiveKernels as RK
import ReactiveKernelsPPL as PPL

const constructors = Dict{Expr,Expr}()
const cache_lock = ReentrantLock()
const hits = Ref(0)
const misses = Ref(0)
const input_key = gensym(:kernel_construction_inputs)
const input_name = gensym(:inputs)
inputs() = task_local_storage(input_key)

function input!(args, value)
    push!(args, value)
    Expr(:call, GlobalRef(Base, :getindex), input_name, length(args))
end

# Preserve when the original construction would read a backend's binding.
readref(ref::GlobalRef) = getglobal(ref.mod, ref.name)

function source_record(source, mod)
    # Separately evaluated top-level recipe closures have zero capture fields.
    # The source builder needs the function itself only for native Julia scopes.
    f = RK._kernel_source_native_scope(source.args[2]) ? Core.eval(mod, source) : nothing
    RK._kernel_source_function(f, source, mod)
end

struct UnsupportedConstruction <: Exception end

function skeleton(x, args, locals, mod)
    if x isa Symbol
        # _canonical_locals alpha-renames generated symbols. A generated global
        # outside the constructor's binders must retain its original identity.
        RK._lowering_gensym(x) && !(x in locals) && throw(UnsupportedConstruction())
        return x
    end
    x isa LineNumberNode && return LineNumberNode(0)
    x isa QuoteNode && return input!(args, x.value)
    if x isa GlobalRef
        RK._anonymous_module(x.mod) || return x
        return Expr(:call, GlobalRef(@__MODULE__, :readref), input!(args, x))
    end
    x isa Expr || return input!(args, x)
    if RK._kernel_is_group_call(x)
        return skeleton(x.args[1].args[2], args, locals, mod)
    elseif x.head in (:quote, :inert)
        # Interpolation can see lexical locals. Leave such constructions with
        # the ordinary evaluator, before executing any of their instructions.
        throw(UnsupportedConstruction())
    elseif x.head === :->
        RK._kernel_mentions_any(x, locals) && throw(UnsupportedConstruction())
        return Expr(:call, GlobalRef(Core, :eval), mod, input!(args, x))
    elseif Meta.isexpr(x, :call, 4) && x.args[1] === GlobalRef(RK, :_kernel_source_function)
        f, source, scope = x.args[2], x.args[3], x.args[4]
        if Meta.isexpr(f, :->, 2) && source isa QuoteNode && isequal(f, source.value) &&
                scope isa QuoteNode && scope.value === mod
            RK._kernel_mentions_any(f, locals) && throw(UnsupportedConstruction())
            return Expr(:call, GlobalRef(@__MODULE__, :source_record), input!(args, source.value), mod)
        end
    elseif Meta.isexpr(x, :., 2) && x.args[2] isa QuoteNode && x.args[2].value isa Symbol
        return Expr(:., skeleton(x.args[1], args, locals, mod), x.args[2])
    end
    Expr(x.head, Any[skeleton(arg, args, locals, mod) for arg in x.args]...)
end

function construct(def::Expr)
    mod = PPL.PPLGeneratedModels
    RK._image_target_module() === nothing || return RK._kernel_eval_definition(mod, def)
    expanded = macroexpand(mod, Expr(:macrocall, Symbol("@kernel"), LineNumberNode(1), def))
    Meta.isexpr(expanded, :(=), 2) && expanded.args[1] isa Symbol ||
        throw(ArgumentError("expected a stateless RKPPL kernel definition"))
    construction = expanded.args[2]
    locals = Set{Symbol}()
    RK._kernel_construction_locals!(locals, construction)
    args = Any[]
    body = try
        skeleton(construction, args, locals, mod)
    catch err
        err isa UnsupportedConstruction || rethrow()
        return RK._kernel_eval_construction(mod, construction)
    end
    binding = Expr(:(=), input_name, Expr(:call, GlobalRef(@__MODULE__, :inputs)))
    form = RK._canonical_locals(Expr(:let, binding, body))
    thunk = lock(cache_lock) do
        existing = get(constructors, form, nothing)
        if existing !== nothing
            hits[] += 1
            return existing
        end
        code = Meta.lower(mod, form)
        Meta.isexpr(code, :thunk) || return nothing
        misses[] += 1
        constructors[form] = code
    end
    thunk === nothing && return RK._kernel_eval_construction(mod, construction)
    task_local_storage(() -> Core.eval(mod, thunk), input_key, args)
end

counts() = lock(cache_lock) do
    (; hits=hits[], misses=misses[])
end

function install!()
    replay = GlobalRef(@__MODULE__, :construct)
    @eval PPL _eval_kernel_def(def::Expr) = $replay(def)
    nothing
end

end
