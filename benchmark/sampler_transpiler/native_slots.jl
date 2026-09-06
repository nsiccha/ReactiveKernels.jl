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
function slot_read(c, name; ensure=true, count_gradients=false)
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
        slotplan(c), c.producer, hidx, typeof(c.owned), typeof(c.shared);
        recipe_hook=count_gradients ? ((_,h)->RK.recipe_handle_mode(h)===:destination ?
            :(counts[2]+=1) : nothing) : nothing)
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

function coalesced_slot_transfer(body,start,context,mainctx,children,borrowed)
    context===mainctx || return nothing
    first_write=body[start]
    first_write isa RK._PlaceWrite || return nothing
    target,source=first_write.target,first_write.rhs
    target isa RK._SelfField && source isa RK._SelfField &&
        length(target.path)==2 && length(source.path)==2 || return nothing
    haskey(children,first(target.path)) && haskey(children,first(source.path)) || return nothing
    destination,origin=children[first(target.path)],children[first(source.path)]
    typeof(destination.owned)===typeof(origin.owned) &&
        RK.kernel_plan_key(slotplan(destination))===RK.kernel_plan_key(slotplan(origin)) &&
        destination.shared===origin.shared || return nothing
    fields=slotfields(destination)
    owned=Set(c for c in values(fields) if first(RK.kernel_plan_field(slotplan(destination),c))===:owned)
    any(c->(destination.prefix,c) in borrowed,owned) && return nothing
    sources=setdiff(owned,keys(destination.producer))
    isempty(sources) && return nothing
    transferred=Set{Int}()
    for index in start:length(body)
        write=body[index]
        write isa RK._PlaceWrite || return nothing
        lhs,rhs=write.target,write.rhs
        lhs isa RK._SelfField && rhs isa RK._SelfField &&
            length(lhs.path)==2 && length(rhs.path)==2 &&
            first(lhs.path)===first(target.path) && first(rhs.path)===first(source.path) || return nothing
        canon=fields[last(lhs.path)]
        canon==fields[last(rhs.path)] && canon in sources || return nothing
        T=RK._pp_fieldtype(slotplan(destination),canon,typeof(destination.owned),typeof(destination.shared))
        write.dot==(T <: AbstractArray) || return nothing
        push!(transferred,canon)
        if transferred==sources
            # Equal authoritative inputs and shared recipe authorities imply
            # equal derived values. Transfer the cache and its validity bits,
            # preserving every destination buffer, instead of invalidating and
            # recomputing an equivalent endpoint. Only consecutive plain field
            # transfers qualify; a retained local cache observation prevents it.
            value=:(RK._canon_copy_endpoint!($(Symbol(destination.prefix,:_owned)),
                                             $(Symbol(origin.prefix,:_owned))))
            return index-start,value
        end
    end
    nothing
end

function compile_native_slots(kernel, state, name; endpoints, effects, hoist=true,
        bufferize=true, coalesce_transfers=true, count_gradients=false)
    pf = getfield(kernel,:prepared)
    skel = getfield(kernel,:skeleton)
    irs = RK.method_irs(skel)
    fields = RK._stateful_field_regs(getfield(kernel,:bindings))
    written = RK._sm_global_written(RK.kernel_prepared_plan(pf), irs, fields)
    for ir in irs
        RK._kmir_walk_calls(ir.body) do call
            if call isa RK._FieldCall && length(call.path)==1 && haskey(effects,only(call.path))
                source=getproperty(effects,only(call.path)).source
                if !isempty(RK.kernel_write_roots(source))
                    length(call.pos)==1 && only(call.pos) isa RK._SelfField ||
                        error("native slots: a captured effect must expose its state root")
                    root=first(only(call.pos).path)
                    push!(written,RK._exec_canon_map(RK.kernel_prepared_plan(pf))[root])
                end
            end
        end
    end
    mainctx = SlotContext(pf, getfield(state,:owned), getfield(state,:shared),
        :__slot_root, RK._sm_active_producer(RK.kernel_prepared_plan(pf),written))
    children = Dict(root => SlotContext(ep.pf, ep.owned, ep.shared, Symbol(:__slot_child_,i),
        Dict(RK.kernel_plan_producer(RK.kernel_prepared_plan(ep.pf))))
        for (i,(root,ep)) in enumerate(pairs(endpoints)))
    contexts = (mainctx, (children[root] for root in keys(endpoints))...)
    child_canons=Set(slotfields(mainctx)[root] for root in keys(endpoints))
    for (recipe,inputs) in RK.kernel_plan_recipe_inputs(slotplan(mainctx))
        if any(in(child_canons),inputs) && any(==(recipe),values(mainctx.producer))
            error("native slots: a live parent recipe reading a child needs flattened recipe dependencies")
        end
    end
    declared = Dict(ir.id.name => ir for ir in irs)
    length(declared)==length(irs) || error("native slots: overloaded methods require dispatch lowering")
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
            return slot_read(c,n;count_gradients)
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
            isempty(x.kw) && length(callee.formals)==length(x.pos) ||
                error("native slots: sibling argument binding is unsupported")
            args = Dict{Symbol,Any}(); evaluations = Any[]
            for (formal,actual) in zip(callee.formals,x.pos)
                symbol=fresh(:argument)
                push!(evaluations,:(local $symbol = $(rhs(actual,context,locals,formals))))
                args[formal.name]=symbol
            end
            call=Expr(:call,Expr(:->,Expr(:tuple),emit(callee.body,context,Dict{Symbol,Any}(),args)))
            return Expr(:block,evaluations...,call)
        else
            error("native slots: unsupported expression $(typeof(x))")
        end
    end
    emit = function(body,context,locals,formals)
        out = Any[]
        skip=0
        for (index,s) in enumerate(body)
            if skip>0
                skip-=1
                continue
            end
            if coalesce_transfers
                transfer=coalesced_slot_transfer(body,index,context,mainctx,children,borrowed)
                if transfer!==nothing
                    skip,value=transfer
                    push!(out,value)
                    continue
                end
            end
            if s isa RK._PlaceWrite
                if s.dot && context === mainctx && s.target isa RK._SelfField &&
                   length(s.target.path)==1 && haskey(children,only(s.target.path))
                    s.rhs isa RK._SelfField && length(s.rhs.path)==1 || error("native slots: structural source")
                    target, source = children[only(s.target.path)], children[only(s.rhs.path)]
                    typeof(target.owned)===typeof(source.owned) &&
                        RK.kernel_plan_key(slotplan(target))===RK.kernel_plan_key(slotplan(source)) &&
                        target.shared===source.shared || error("native slots: structural copy requires one shared endpoint layout")
                    push!(out, :(RK._canon_copy_endpoint!($(Symbol(target.prefix,:_owned)),
                                                          $(Symbol(source.prefix,:_owned)))))
                else
                    c,n = place(s.target,context)
                    value = rhs(s.rhs,context,locals,formals,s.dot)
                    reuses_destination=false
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
                            reuses_destination=true
                        end
                    end
                    T=RK._pp_fieldtype(slotplan(c),slotfields(c)[n],typeof(c.owned),typeof(c.shared))
                    if !s.dot && T <: AbstractArray && !reuses_destination
                        effect=s.rhs isa RK._RegisteredCall ? s.rhs.registration.primitive_effect : nothing
                        result_alias=effect===nothing ? nothing : effect.result_alias
                        preserves_destination=result_alias!==nothing &&
                            s.rhs.args[result_alias] isa RK._SelfField &&
                            s.rhs.args[result_alias].path==s.target.path
                        if !preserves_destination
                            # Value writes need owned storage. In particular a
                            # hoisted constant or another field cannot become
                            # an observable mutable alias of this destination.
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
                    effect_body=ir.body
                    if !isempty(effect_body) && last(effect_body) isa RK._Return && last(effect_body).value===nothing
                        effect_body=Base.front(effect_body)
                    end
                    for statement in effect_body
                        RK._kmir_walk(statement) do node
                            node isa RK._Return && error("native slots: nonterminal free-method return needs call-local control")
                        end
                    end
                    push!(out,emit(effect_body,child,Dict{Symbol,Any}(),Dict{Symbol,Any}(pairs(binding.controls))))
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
       constants=Ref(Tuple(constants)), counts=[0,0], expression,contexts)
end

function slot_chain(program,rng,n)
    for _ in 1:n
        Base.@inline RK.RuntimeGeneratedFunctions.generated_callfunc(
            program.f,program.stores,program.resources,program.constants,program.counts,rng)
    end
    nothing
end
