# Diagnostic prototypes, deliberately outside the package's preparation path.
# These reuse the existing algebra, cell-array and buffer decomposition passes.
# No derivative or production method is replaced.
module StorageAlternatives
using ReactiveKernels
const RK = ReactiveKernels

function expose_sources(ast, recipes)
    RK._index_run_replace(ast) do ex
        if ex.head === :(=) && length(ex.args) == 2 && ex.args[1] isa Symbol
            rhs = RK._cell_array_expand_call(ex.args[2], recipes)
            rhs == ex.args[2] || return Expr(:(=), ex.args[1], rhs)
        end
        ex
    end
end

function algebra_then_arrays(ast, recipes; snapshot=false, types=Dict{Symbol,Any}())
    ast = RK._stable_rewrites(ast, recipes)
    body = RK._lower_cell_arrays_native(ast.args[2], recipes)
    snapshot && (body = snapshot_new_broadcasts(ast.args[2],body,x -> RK._nonalloc_expr_type(types,x)))
    RK._stable_rewrites(Expr(ast.head, ast.args[1], body), recipes)
end

# Preserve the point-in-time values of every dense floating operand of a newly
# deferred broadcast. Unsupported producer shapes keep their original statement.
# This is a narrow structural prototype, not a function-effect whitelist.
function snapshot_new_broadcasts(before, after, type_of;
        copy_operand=(x,T) -> Expr(:call,GlobalRef(Base,:copy),x))
    original = RK._nonalloc_table_definitions(before)
    function snapshot_operand(x, statements)
        if Meta.isexpr(x,:call) && RK._cell_array_function(x.args[1]) in
                (RK._native_broadcasted, Base.Broadcast.broadcasted)
            operands = Any[snapshot_operand(a,statements) for a in x.args[3:end]]
            any(isnothing,operands) && return nothing
            return Expr(:call,x.args[1:2]...,operands...)
        end
        T = type_of(x)
        if T isa Type && T <: Union{Array{Float32},Array{Float64}}
            name = gensym(:snapshot)
            push!(statements,Expr(:(=),name,copy_operand(x,T)))
            return name
        elseif T isa Type && isconcretetype(T) && isbitstype(T) && T <: Number
            return x
        end
        nothing
    end
    function walk(ex)
        ex isa Expr || return ex
        ex.head in (:quote,:inert) && return ex
        if ex.head === :block
            out = Any[]
            for stmt in ex.args
                if Meta.isexpr(stmt,:(=),2) && stmt.args[1] isa Symbol
                    lhs,rhs = stmt.args
                    old = get(original,lhs,rhs)
                    if rhs != old && Meta.isexpr(rhs,:call,2) &&
                            rhs.args[1] == GlobalRef(RK,:_cell_array_broadcast)
                        copies = Any[]
                        bc = snapshot_operand(rhs.args[2],copies)
                        if bc === nothing
                            push!(out,Expr(:(=),lhs,old))
                        else
                            append!(out,copies)
                            push!(out,Expr(:(=),lhs,Expr(:call,rhs.args[1],bc)))
                        end
                        continue
                    end
                end
                push!(out,walk(stmt))
            end
            return Expr(:block,out...)
        end
        Expr(ex.head,map(walk,ex.args)...)
    end
    walk(after)
end

# Source exposure leaves ordinary GlobalRef calls where the old storage pass
# expects recipe-table calls. Feed these expressions through the same existing
# decomposition grammar, retaining an expression when that grammar declines it.
function source_statement!(prog, recipes, types, stmt)
    if Meta.isexpr(stmt, :(=), 2) && stmt.args[1] isa Symbol &&
            Meta.isexpr(stmt.args[2], :call) &&
            RK._callable_slot(stmt.args[2].args[1]) === nothing
        lhs, rhs = stmt.args
        if rhs.args[1] == GlobalRef(RK, :_cell_array_broadcast)
            return RK._nonalloc_statement!(prog, recipes, types, stmt)
        end
        # Stable rewriting can introduce callable-value constructors (e.g.
        # _RwFn(exp), Val(2)) outside the fused-source decomposition grammar.
        # The emitted expression still identifies this allocation explicitly.
        # Use the existing materialization step and its planned concrete array
        # type; do not leave an allocation solely because its scalar callable
        # could not be decomposed. Its operands remain evaluated here, once.
        if rhs.args[1] == GlobalRef(RK, :_native_broadcast_materialize) && length(rhs.args)==2
            inferred = RK._nonalloc_expr_type(types, rhs)
            T = inferred isa Type && isconcretetype(inferred) ? inferred : get(types,lhs,Any)
            if T isa DataType && isconcretetype(T) && T <: Array
                j = RK._step!(prog,RK._MaterializeStep(),RK._nonalloc_slot(T))
                types[lhs] = T
                return Any[Expr(:(=),lhs,RK._step_call(j,rhs.args[2]))]
            end
        end
        ctx = RK._FusedDecomposition(RK, Dict{Symbol,Any}(s => s for s in keys(types)),
            types, Any[], Any[], Any[], length(prog.ops), prog.cache_apply)
        dec = RK._decompose(ctx, rhs, false)
        if dec !== nothing
            append!(prog.ops, ctx.ops)
            append!(prog.caches, ctx.caches)
            types[lhs] = dec[2]
            return Any[ctx.stmts..., Expr(:(=), lhs, dec[1])]
        end
    end
    RK._nonalloc_statement!(prog, recipes, types, stmt)
end

# A benchmark-only fork of _nonallocating_program at 0a6b45626c23c20425de836b99215de97d317a9f.
# The differences are source exposure + algebra/array reconsideration before
# step creation and source_statement! accepting the exposed expressions. The
# ownership, table-lane typing and post-step stable pass are unchanged.
function early_program(p; have_types, cache_apply, snapshot=false)
    lowered = RK._lowering_plan(p)
    values = RK._nonalloc_value_types(lowered, have_types)
    lowered = RK._nonalloc_nested_plates(lowered, values)
    recycle = RK._nonalloc_lanes(lowered, values)
    table_lanes = Pair{Symbol,Symbol}[]
    value_ids = Dict{Symbol,RK.ValueId}()
    ast, constants, callables, recipes = RK._lower_with_ops(lowered;
        exemplar_types=have_types, value_ids, declare=false, recycle,
        pushdowns=true, outline=false, plate_element_types=true,
        plate_programs=true, inline_sources=false, indexed_lanes=table_lanes)
    types = Dict{Symbol,Any}(name => values[cid] for (name,cid) in value_ids if haskey(values,cid))
    ast = algebra_then_arrays(expose_sources(ast, recipes), recipes;snapshot,types)
    prog = RK._StepProgram(Any[callables...], Any[nothing for _ in callables], cache_apply)
    body = Expr(:block)
    for stmt in ast.args[2].args
        append!(body.args, source_statement!(prog, recipes, types, stmt))
    end
    body = RK._stable_rewrites(Expr(:function,ast.args[1],body),
        (recipes...,prog.ops[length(recipes)+1:end]...)).args[2]
    read = RK._nonalloc_cache_reads!(Set{Int}(),body)
    for j in eachindex(prog.caches)
        j in read || (prog.caches[j] = nothing)
    end
    mentioned = RK._nonalloc_mentions!(Set{Symbol}(),body)
    lanes = Any[]
    for (cid,lane) in recycle
        lane in mentioned || continue
        A = get(values,cid,Any)
        slot = A isa Type && A <: Array && isconcretetype(A) ?
            Base.RefValue{Union{Nothing,A}}(nothing) : Ref{Any}(nothing)
        j = RK._step!(prog,RK._NonallocLane(),slot)
        push!(lanes,Expr(:(=),lane,Expr(:ref,RK._CACHES_ARG,j)))
    end
    definitions = RK._nonalloc_table_definitions(body)
    for (lane,root) in table_lanes
        lane in mentioned || continue
        R = RK._nonalloc_static_type(root,types,definitions,constants,prog.ops)
        A = R === nothing ? Any : Base.promote_op(RK._indexed_table,R)
        slot = A isa Type && A <: Array && isconcretetype(A) ?
            Base.RefValue{Union{Nothing,A}}(nothing) : Ref{Any}(nothing)
        j = RK._step!(prog,RK._NonallocLane(),slot)
        push!(lanes,Expr(:(=),lane,Expr(:ref,RK._CACHES_ARG,j)))
    end
    prepend!(body.args,lanes)
    ports = ast.args[1].args[3:2+length(lowered.have)]
    (RK._extract_plate_programs(Expr(:function,
        Expr(:tuple,RK._CONSTANTS_ARG,RK._CALLABLES_ARG,RK._CACHES_ARG,
            RK._CACHE_APPLY_ARG,ports...),body)), constants,Tuple(prog.ops),Tuple(prog.caches))
end

function early(p, args...; snapshot=false)
    cache_apply = Base.get_extension(RK,:ReactiveKernelsMutatingFunctionsExt).cache_apply!
    types = map(typeof,args)
    ast, constants, callables, caches = early_program(p;have_types=types,cache_apply,snapshot)
    ast = RK._nonalloc_exact_signature(ast,types)
    RK.NonAllocatingKernel{Tuple{types...}}(RK.compile(RK._nonallocating_body(ast)),
        constants,callables,caches,cache_apply,Tuple(p.have),Tuple(p.want),p,ast)
end

# A separately prepared kernel owns each arm's caches. Only owned broadcast
# materialization steps are exposed; every survivor is restored exactly.
function late(p, args...; snapshot=false)
    cache_apply = Base.get_extension(RK,:ReactiveKernelsMutatingFunctionsExt).cache_apply!
    k = RK._prepare_nonallocating(p,cache_apply;have_types=map(typeof,args),exact_signature=true)
    ast = RK._inline_plate_programs(k.ast)
    recipes = RK.prepare(p;bound_statistics=false).lowered_recipes
    restore = Dict{Any,Any}()
    exposed = RK._index_run_replace(ast) do ex
        if ex.head === :call && length(ex.args)==4 && ex.args[1] === RK._CACHE_APPLY_ARG
            slot = RK._callable_slot(ex.args[3])
            if slot !== nothing && k.callables[slot] isa RK._MaterializeStep
                view = Expr(:call,GlobalRef(RK,:_native_broadcast_materialize),ex.args[4])
                restore[view] = ex
                return view
            end
        end
        ex
    end
    body = RK._lower_cell_arrays_native(exposed.args[2],recipes)
    prog = RK._StepProgram(Any[k.callables...],Any[k.caches...],k.cache_apply)
    if snapshot
        types = Dict{Symbol,Any}()
        for (port,T) in zip(ast.args[1].args[5:end],map(typeof,args))
            types[port isa Symbol ? port : port.args[1]] = T
        end
        definitions = RK._nonalloc_table_definitions(exposed.args[2])
        type_of(x) = RK._nonalloc_static_type(x,types,definitions,k.constants,k.callables)
        # Snapshot at the original production point, with the same generic
        # destination-passing copy used by normal storage planning. Each arm
        # owns these slots; repeated calls can reuse their dense buffers.
        function copy_operand(x,T)
            j = RK._step!(prog,Base.copy,RK._nonalloc_slot(T))
            RK._step_call(j,x)
        end
        body = snapshot_new_broadcasts(exposed.args[2],body,type_of;copy_operand)
    end
    restored = RK._index_run_replace(ex -> get(restore,ex,ex),Expr(ast.head,ast.args[1],body))
    rewritten = RK._stable_rewrites(restored,(recipes...,prog.ops[length(recipes)+1:end]...))
    S = typeof(k).parameters[end]
    RK.NonAllocatingKernel{S}(RK.compile(RK._nonallocating_body(rewritten)),
        k.constants,Tuple(prog.ops),Tuple(prog.caches),k.cache_apply,k.inputs,k.outputs,k.plan,rewritten)
end

# Diagnostic compiler-effects gate. An unconditional effect-free result for
# EACH recipe protects incoming references and global state. A whole-kernel
# effect-free result alone would permit mutations of private intermediates.
# A refusal keeps ordinary storage; it never rejects a client program.
function effects_gate(p, types)
    p = RK._lowering_plan(p)
    values = RK._nonalloc_value_types(p,types)
    reports = NamedTuple[]
    for r in p.recipes
        argtypes = Tuple(RK._input_list(v -> get(values,RK.canon_id(p.graph,v.id),Any),r.inputs))
        if r.effectful || !all(isconcretetype,argtypes)
            push!(reports,(; recipe=r.id, accepted=false, effects="effectful or abstract"))
            continue
        end
        q = RK._partial_subplan(p,RK.Value[r.inputs...],RK.Value[r.outputs...],RK.Recipe[r])
        ast, constants, callables, _ = RK._lower_with_ops(q;exemplar_types=argtypes,plate_programs=true)
        f = RK.compile(ast)
        table = RK._native_table(ast,callables)
        sig = Tuple{typeof(constants),typeof(table),argtypes...}
        effects = Base.infer_effects(f,sig)
        accepted = Core.Compiler.is_effect_free(effects)
        push!(reports,(;recipe=r.id,accepted,effects=string(effects)))
    end
    (;accepted=all(r -> r.accepted,reports),reports)
end
end
