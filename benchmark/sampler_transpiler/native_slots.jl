# Experimental lowering over captured MethodIR and prepared recipe handles.
# No algorithm/field-name recognition. Fixed structured children are flattened
# to their existing typed canonical stores; captured free methods are inlined.
struct SlotContext
    pf
    owned
    shared
    prefix::Symbol
    producer
end
slotplan(c) = RK.kernel_prepared_plan(c.pf)
slotfields(c) = RK._exec_canon_map(slotplan(c))
function scope_slots(x, c)
    x isa Symbol && return x === :owned ? Symbol(c.prefix, :_owned) :
        x === :shared ? Symbol(c.prefix, :_shared) :
        x === :handles ? Symbol(c.prefix, :_handles) : x
    x isa Expr || return x
    Expr(x.head, (scope_slots(a, c) for a in x.args)...)
end
function slot_read(c, name; ensure=true)
    canon = slotfields(c)[name]
    read = scope_slots(RK._pp_read(slotplan(c), canon), c)
    ensure || return read
    statements = Any[]
    handles = RK.kernel_prepared_handles(c.pf)
    hidx = Dict(RK.kernel_plan_recipes(slotplan(c))[i] => (h, i)
                for (i,h) in enumerate(handles))
    # These stores are compiler-owned and sources are current at construction.
    # Every emitted source write completes before it is read; shared slots are
    # fixed for this program. Only derived owned fields need runtime currentness.
    known = Set(canon for canon in values(slotfields(c)) if
        !haskey(c.producer,canon) || first(RK.kernel_plan_field(slotplan(c),canon)) === :shared)
    RK._exec_ensure!(statements, canon, known, Set{Int}(),
        slotplan(c), c.producer, hidx, typeof(c.owned), typeof(c.shared))
    Expr(:block, (scope_slots(s, c) for s in statements)..., read)
end
function slot_write(c, name, value, dot, result)
    plan, fields = slotplan(c), slotfields(c)
    canon = fields[name]
    role, slot = RK.kernel_plan_field(plan, canon)
    role === :owned || error("native slots: write to shared field $name")
    T = RK._pp_fieldtype(plan, canon, typeof(c.owned), typeof(c.shared))
    statements = Any[:(local $result = $value)]
    RK._exec_mask!(statements, plan, canon, :kill)
    for dependent in sort!(collect(RK._exec_kill_closure(plan, canon, c.producer)))
        RK._exec_mask!(statements, plan, dependent, :kill)
    end
    if dot
        T <: AbstractArray || error("native slots: broadcast requires array")
        push!(statements, :(Base.materialize!($(RK._pp_read(plan,canon)),
            Base.Broadcast.broadcasted(identity, $result))))
    else
        push!(statements, :(RK._canon_set!(owned, Val($slot), $result)))
    end
    RK._exec_mask!(statements, plan, canon, :bless)
    Expr(:block, (scope_slots(s,c) for s in statements)...)
end

function native_endpoint(spec, method, arguments)
    reg = RK.kernel_registration(method)
    pf = RK._prepare_factory(spec, reg)
    sources = RK._transition_sources(spec, pf, arguments, NamedTuple())
    handles = RK.kernel_prepared_handles(pf)
    values = RK._bootstrap_canon_values(RK.kernel_prepared_plan(pf), handles, sources)
    owned, shared = RK._construct_endpoint_from_values(RK.kernel_prepared_plan(pf), handles, values)
    (; pf, owned, shared)
end

function compile_native_slots(kernel, state, name; endpoints, effects, hoist=true, bufferize=true)
    pf = getfield(kernel,:prepared)
    skel = getfield(kernel,:skeleton)
    irs = RK.method_irs(skel)
    fields = RK._stateful_field_regs(getfield(kernel,:bindings))
    written = RK._sm_global_written(RK.kernel_prepared_plan(pf), irs, fields)
    mainctx = SlotContext(pf, getfield(state,:owned), getfield(state,:shared),
        :__slot_root, RK._sm_active_producer(RK.kernel_prepared_plan(pf),written))
    children = Dict(root => SlotContext(ep.pf, ep.owned, ep.shared, Symbol(:__slot_child_,i),
        Dict(RK.kernel_plan_producer(RK.kernel_prepared_plan(ep.pf))))
        for (i,(root,ep)) in enumerate(pairs(endpoints)))
    contexts = (mainctx, (children[root] for root in keys(endpoints))...)
    declared = Dict(ir.id.name => ir for ir in irs)
    local_id = Ref(0)
    fresh(name) = Symbol(:_ns_,name,:_,(local_id[]+=1))
    constants = Any[]
    borrowed = Set{Tuple{Symbol,Int}}()
    function place(x, context)
        x isa RK._SelfField || error("native slots: unsupported place $(typeof(x))")
        if context === mainctx && haskey(children,first(x.path))
            length(x.path)==2 || error("native slots: expected one nested field")
            return children[first(x.path)],last(x.path)
        end
        length(x.path)==1 || error("native slots: deeper nested path")
        context,only(x.path)
    end
    rhs = nothing
    emit = nothing
    function static_value(x, context)
        if x isa RK._Lit
            return true,x.value
        elseif x isa RK._SelfField
            c,n = place(x,context)
            role,slot = RK.kernel_plan_field(slotplan(c),slotfields(c)[n])
            role === :shared || return false,nothing
            return true,RK._canon_slot(c.shared,Val(slot))
        elseif x isa RK._RegisteredCall
            effect = x.registration.primitive_effect
            (effect === nothing || effect.kind === :pure) || return false,nothing
            isempty(x.kw) || return false,nothing
            arguments = map(a->static_value(a,context),x.args)
            all(first,arguments) || return false,nothing
            f = RK._sm_exact_callee(x)
            return true,f(map(last,arguments)...)
        end
        false,nothing
    end
    rhs = function(x, context, locals, formals, dot=false)
        if hoist && x isa RK._RegisteredCall && !dot
            known,value = static_value(x,context)
            if known
                push!(constants,value)
                return :(getfield(constant_values,$(length(constants))))
            end
        end
        if x isa RK._SelfField
            c,n = place(x,context)
            return slot_read(c,n)
        elseif x isa RK._Lit
            return x.value
        elseif x isa RK._FormalRef
            return formals[x.arg]
        elseif x isa RK._LocalRef
            return locals[x.name]
        elseif x isa RK._RegisteredCall
            f = RK._exec_captured_callee(x)
            args = Any[rhs(a,context,locals,formals,dot) for a in x.args]
            isempty(x.kw) || error("native slots: primitive keyword")
            return dot ? Expr(:call,GlobalRef(Base,:broadcasted),f,args...) : Expr(:call,f,args...)
        elseif x isa RK._IfExpr
            return Expr(:if,rhs(x.cond,context,locals,formals),
                rhs(x.thenv,context,locals,formals),rhs(x.elsev,context,locals,formals))
        elseif x isa RK._Short
            return Expr(x.op,rhs(x.lhs,context,locals,formals),rhs(x.rhs,context,locals,formals))
        elseif x isa RK._CallExpr
            x.target isa RK._SelfRef || error("native slots: foreign sibling subject")
            callee = declared[x.name]
            args = Dict(f.name=>rhs(a,context,locals,formals) for (f,a) in zip(callee.formals,x.pos))
            return Expr(:call,Expr(:->,Expr(:tuple),emit(callee.body,context,Dict{Symbol,Any}(),args)))
        else
            error("native slots: unsupported expression $(typeof(x))")
        end
    end
    emit = function(body,context,locals,formals)
        out = Any[]
        for s in body
            if s isa RK._PlaceWrite
                if s.dot && context === mainctx && s.target isa RK._SelfField &&
                   length(s.target.path)==1 && haskey(children,only(s.target.path))
                    s.rhs isa RK._SelfField && length(s.rhs.path)==1 || error("native slots: structural source")
                    target, source = children[only(s.target.path)], children[only(s.rhs.path)]
                    push!(out, :(RK._canon_copy_endpoint!($(Symbol(target.prefix,:_owned)),
                                                          $(Symbol(source.prefix,:_owned)))))
                else
                    c,n = place(s.target,context)
                    value = rhs(s.rhs,context,locals,formals,s.dot)
                    # A diagonal product can reuse a dying owned vector. An
                    # earlier local observation conservatively prevents reuse.
                    if bufferize && !s.dot && s.rhs isa RK._RegisteredCall &&
                       RK._exec_captured_callee(s.rhs) === (*) && length(s.rhs.args)==2 &&
                       s.rhs.args[2] isa RK._SelfField &&
                       place(s.rhs.args[2],context)==(c,n) &&
                       !((c.prefix,slotfields(c)[n]) in borrowed)
                        known,factor=static_value(s.rhs.args[1],context)
                        if known && factor isa Diagonal
                            push!(constants,factor)
                            factor_value=:(getfield(constant_values,$(length(constants))))
                            value=:(LinearAlgebra.lmul!($factor_value,$(slot_read(c,n))))
                        end
                    end
                    if !s.dot
                        known,constant = static_value(s.rhs,context)
                        if known && constant isa AbstractArray
                            # Hoisting an allocating expression does not grant
                            # mutable state ownership of the shared constant.
                            value = :(RK._sm_structural_copy($value))
                        end
                    end
                    push!(out,slot_write(c,n,value,s.dot,fresh(:write)))
                end
            elseif s isa RK._LocalAssign
                s.style===:single || error("native slots: multiple assignment")
                RK._kmir_walk(s.rhs) do node
                    if node isa RK._SelfField
                        c,n=place(node,context)
                        canon=slotfields(c)[n]
                        T=RK._pp_fieldtype(slotplan(c),canon,typeof(c.owned),typeof(c.shared))
                        T <: AbstractArray && push!(borrowed,(c.prefix,canon))
                    end
                end
                value = rhs(s.rhs,context,locals,formals)
                sym = get!(locals,only(s.lhs)) do; fresh(only(s.lhs)); end
                push!(out,:($sym = $value))
            elseif s isa RK._For
                var = fresh(only(s.var))
                nested = copy(locals); nested[only(s.var)] = var
                push!(out,Expr(:for,Expr(:(=),var,rhs(s.iter,context,locals,formals)),
                    emit(s.body,context,nested,formals)))
            elseif s isa RK._If
                push!(out,Expr(:if,rhs(s.cond,context,locals,formals),
                    emit(s.thenb,context,copy(locals),formals),
                    emit(s.elseb,context,copy(locals),formals)))
            elseif s isa RK._Guard
                condition = rhs(s.cond,context,locals,formals)
                s.op === :|| && (condition = :(!$condition))
                push!(out,Expr(:if,condition,
                    emit(s.body,context,copy(locals),formals)))
            elseif s isa RK._Return
                push!(out,Expr(:return,s.value===nothing ? :nothing : rhs(s.value,context,locals,formals)))
            elseif s isa RK._ExprStmt && s.expr isa RK._FieldCall
                call = s.expr; field = only(call.path)
                if haskey(effects,field)
                    binding = getproperty(effects,field)
                    length(call.pos)==1 && only(call.pos) isa RK._SelfField || error("native slots: effect subject")
                    child = children[only(only(call.pos).path)]
                    ir = only(RK.method_irs(binding.source))
                    push!(out,emit(ir.body,child,Dict{Symbol,Any}(),Dict{Symbol,Any}(pairs(binding.controls))))
                    push!(out,:(counts[1] += 1))
                elseif get(fields,field,:unknown) === nothing
                    # Authored optional observer has the captured no-effect binding.
                    nothing
                else
                    error("native slots: effect has no captured method binding $field")
                end
            elseif s isa RK._ExprStmt
                push!(out,rhs(s.expr,context,locals,formals))
            else
                error("native slots: unsupported statement $(typeof(s))")
            end
        end
        Expr(:block,out...)
    end
    ir = declared[name]
    length(ir.formals)==1 || error("native slots: expected one runtime formal")
    formals = Dict{Symbol,Any}(only(ir.formals).name=>:argument)
    body = emit(ir.body,mainctx,Dict{Symbol,Any}(),formals)
    setup = Any[]
    for (i,c) in enumerate(contexts)
        push!(setup,:(local $(Symbol(c.prefix,:_owned)) = getfield(getfield(stores,$i),1)))
        push!(setup,:(local $(Symbol(c.prefix,:_shared)) = getfield(getfield(stores,$i),2)))
        push!(setup,:(local $(Symbol(c.prefix,:_handles)) = getfield(resources,$i)))
    end
    expression = :((stores,resources,constants,counts,argument)->begin
        $(setup...)
        local constant_values = constants[]
        $body
        nothing
    end)
    # The emitted source uses the package's primitive helpers directly.
    function qualify(x)
        x isa Expr || return x
        if x.head === :. && x.args[1] === :RK
            return GlobalRef(RK,x.args[2].value)
        elseif x.head===:call && x.args[1] isa Symbol && isdefined(RK,x.args[1])
            return Expr(:call,GlobalRef(RK,x.args[1]),(qualify(a) for a in x.args[2:end])...)
        end
        Expr(x.head,(qualify(a) for a in x.args)...)
    end
    expression=qualify(expression)
    (; f=RK.compile(expression), stores=Tuple((c.owned,c.shared) for c in contexts),
       resources=Tuple(RK.kernel_prepared_handles(c.pf) for c in contexts),
       constants=Ref(Tuple(constants)), counts=[0], expression,contexts)
end

function slot_chain(program,rng,n)
    for _ in 1:n
        Base.@inline RK.RuntimeGeneratedFunctions.generated_callfunc(
            program.f,program.stores,program.resources,program.constants,program.counts,rng)
    end
    nothing
end
