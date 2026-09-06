# Internal backend pass over the emitted native slot program. Numerical
# slots become explicit locals; fixed recipe/callback authorities stay outside
# the loop carry. The mathematical kernel is not reconstructed here.
struct SlotStatic{T}
    values::T
end
Reactant.make_tracer(seen, previous::SlotStatic, path, mode; kwargs...) = previous
Reactant.traced_type_inner(::Type{T}, seen, mode::Reactant.TraceMode,
    track_numbers::Type, ndevices, runtime) where {T<:SlotStatic} = T

@inline function slot_diagonal_solve!(destination, factor, rhs)
    diagonal = factor.factors.diag
    @. destination = (rhs / diagonal) / diagonal
    destination
end
@inline function slot_lmul!(factor::Diagonal, destination::AbstractVector)
    diagonal=factor.diag
    @. destination=diagonal*destination
    destination
end
slot_lmul!(factor,destination)=lmul!(factor,destination)
@inline slot_scalar_getindex(array,indices...) =
    Reactant.@allowscalar @inbounds getindex(array,indices...)

function compile_traced_slots(program; static_currentness=true, unroll_limit=0, reuse_code=true)
    contextmap = Dict(Symbol(c.prefix, :_owned)=>c for c in program.contexts)
    sharedmap = Dict(Symbol(c.prefix, :_shared)=>c.shared for c in program.contexts)
    handlemap = Dict(Symbol(c.prefix, :_handles)=>program.resources[i]
        for (i,c) in enumerate(program.contexts))
    slots = Dict{Tuple{Symbol,Int},Symbol}()
    validity = Dict{Tuple{Symbol,Int},Symbol}()
    statics = Any[]
    fresh_index = Ref(0)
    fresh(label) = Symbol(:_ts_,label,:_,fresh_index[]+=1)
    function fixed(value)
        # Repeated uses share the same fixed authority. Keep distinct mutable
        # objects distinct; equality of their current contents is insufficient.
        i=findfirst(v->v===value,statics)
        if i===nothing
            push!(statics,value)
            i=length(statics)
        end
        :(getfield(metadata.values,$i))
    end
    function slot(object,index)
        haskey(sharedmap,object) && return fixed(RK._canon_slot(sharedmap[object],Val(index)))
        get!(slots,(object,index)) do
            value=RK._canon_slot(contextmap[object].owned,Val(index))
            (value isa Number || value isa AbstractArray{<:Number}) ||
                error("traced slots: nonnumeric owned slot")
            fresh(:value)
        end
    end
    current(object,index)=get!(validity,(object,index)) do; fresh(:current); end
    callee(x,name) = x isa GlobalRef ? x.name===name :
        x isa Expr && x.head===:. ? x.args[2]===QuoteNode(name) : x===name
    index(x) = only(x.args[2:end])
    function fixed_sample(x)
        x isa Expr || return x
        if x.head===:block
            parts=filter(a->a!==nothing && !(a isa LineNumberNode),x.args)
            length(parts)==1 && return fixed_sample(only(parts))
        elseif x.head===:call
            if callee(x.args[1],:getfield) && x.args[2]===:constant_values
                return getfield(program.constants[],x.args[3])
            elseif callee(x.args[1],:_canon_slot) && haskey(sharedmap,x.args[2])
                return RK._canon_slot(sharedmap[x.args[2]],Val(index(x.args[3])))
            end
        end
        nothing
    end
    function lower(x)
        x isa LineNumberNode && return nothing
        x isa Expr || return x
        if x.head===:local
            return lower(only(x.args))
        elseif x.head===:ref && x.args[1]===:counts
            return Symbol(:_ts_counter_,x.args[2])
        elseif x.head===:call
            f=x.args[1]
            if f===Random.rand && length(x.args)==3
                range=fixed_sample(x.args[3])
                if range isa UnitRange
                    return Expr(:call,slot_rand_range,lower(x.args[2]),fixed(SlotRangeSampler(range)))
                end
            end
            if callee(f,:_canon_slot)
                return slot(x.args[2],index(x.args[3]))
            elseif callee(f,:_canon_current)
                return current(x.args[2],index(x.args[3]))
            elseif callee(f,:_canon_set!)
                return :($(slot(x.args[2],index(x.args[3]))) = $(lower(x.args[4])))
            elseif callee(f,:_canon_kill!) || callee(f,:_canon_bless!)
                return :($(current(x.args[2],index(x.args[3]))) = $(callee(f,:_canon_bless!)))
            elseif callee(f,:_canon_bless2!)
                return Expr(:block,(:($(current(x.args[2],index(i))) = true) for i in x.args[3:4])...)
            elseif callee(f,:_canon_copy_slot!)
                dest,source=x.args[2:3]
                i=index(x.args[4])
                value=RK._canon_slot(contextmap[dest].owned,Val(i))
                d,s=slot(dest,i),slot(source,i)
                return value isa AbstractArray ? :(copyto!($d,$s)) : :($d=$s)
            elseif callee(f,:_canon_copy_endpoint!)
                dest,source=x.args[2:3]
                native=contextmap[dest].owned
                statements=Any[]
                for i in 1:fieldcount(typeof(native))-1
                    value=RK._canon_slot(native,Val(i))
                    value===nothing && continue
                    d,s=slot(dest,i),slot(source,i)
                    push!(statements,value isa AbstractArray ? :(copyto!($d,$s)) : :($d=$s))
                    push!(statements,:($(current(dest,i)) = $(current(source,i))))
                end
                return Expr(:block,statements...)
            elseif callee(f,:_pp_diag_cholesky_ldiv!)
                return Expr(:call,slot_diagonal_solve!,(lower(a) for a in x.args[2:end])...)
            elseif callee(f,:lmul!)
                return Expr(:call,slot_lmul!,(lower(a) for a in x.args[2:end])...)
            elseif f===RK.NativeSlotCompiler.slot_scalar_getindex
                return Expr(:call,slot_scalar_getindex,(lower(a) for a in x.args[2:end])...)
            elseif callee(f,:recipe_handle_op)
                ref=x.args[2]
                ref.head===:ref || error("traced slots: dynamic handle")
                return fixed(RK.recipe_handle_op(handlemap[ref.args[1]][ref.args[2]]))
            elseif callee(f,:getfield) && x.args[2]===:constant_values
                return fixed(getfield(program.constants[],x.args[3]))
            end
        end
        Expr(x.head,(lower(a) for a in x.args)...)
    end
    # The native emitter puts setup before one source-body block and a void
    # tail. Select the source block by its position, not by sampler contents.
    nativebody=filter(x->!(x isa LineNumberNode),program.expression.args[2].args)
    body=lower(nativebody[end-1])
    body isa Expr && body.head===:block || error("traced slots: expected emitted source body")

    # Put nested expression effects in evaluation order before tracing control.
    # Immediate zero-argument sibling lambdas have their own terminal return.
    value=nothing
    statement=nothing
    function blockstatements(x,out,live)
        pieces=x isa Expr && x.head===:block ? x.args : Any[x]
        may_return=false
        for piece in pieces
            piece===nothing && continue
            target=Any[]
            returns=statement(piece,target,live)
            append!(out,may_return ? Any[Expr(:if,live,Expr(:block,target...))] : target)
            may_return |= returns
        end
        may_return
    end
    value=function(x,out,live)
        x isa Expr || return x
        if x.head===:block
            pieces=filter(!isnothing,x.args)
            isempty(pieces) && return nothing
            for piece in pieces[1:end-1]; statement(piece,out,live); end
            return value(last(pieces),out,live)
        elseif x.head===:if
            condition=value(x.args[1],out,live)
            result=fresh(:branch)
            branches=map(x.args[2:end]) do branch
                statements=Any[]
                resultvalue=value(branch,statements,live)
                push!(statements,:($result=$resultvalue))
                Expr(:block,statements...)
            end
            push!(out,Expr(:if,condition,branches...))
            return result
        elseif x.head===:return
            return isempty(x.args) ? nothing : value(only(x.args),out,live)
        elseif x.head===:call && x.args[1] isa Expr && x.args[1].head===:->
            length(x.args)==1 || error("traced slots: sibling lambda arguments")
            return value(x.args[1].args[2],out,live)
        elseif x.head===:call
            prefixes=Vector{Any}[]; arguments=Any[]
            for a in x.args
                prefix=Any[]
                push!(arguments,value(a,prefix,live));push!(prefixes,prefix)
            end
            for i in eachindex(arguments)
                append!(out,prefixes[i])
                if any(!isempty,prefixes[i+1:end])
                    saved=fresh(:operand)
                    push!(out,:($saved=$(arguments[i])))
                    arguments[i]=saved
                end
            end
            return Expr(:call,arguments...)
        elseif x.head in (:&&,:||)
            condition=value(x.args[1],out,live)
            conditional=x.head===:&& ? Expr(:if,condition,x.args[2],false) :
                Expr(:if,condition,true,x.args[2])
            return value(conditional,out,live)
        end
        Expr(x.head,(value(a,out,live) for a in x.args)...)
    end
    statement=function(x,out,live)
        x isa Expr || (push!(out,x);return false)
        if x.head===:block
            return blockstatements(x,out,live)
        elseif x.head===:return
            all(a->a===nothing || a===:nothing,x.args) || error("traced slots: nonvoid outer return")
            push!(out,:($live=false))
            return true
        elseif x.head===:if
            condition=value(x.args[1],out,live)
            branches=Any[];returns=false
            for branch in x.args[2:end]
                statements=Any[]
                returns |= blockstatements(branch,statements,live)
                push!(branches,Expr(:block,statements...))
            end
            push!(out,Expr(:if,condition,branches...))
            return returns
        elseif x.head===:for
            binding=x.args[1]
            iterator=value(binding.args[2],out,live)
            static_range = iterator isa Expr && iterator.head===:call &&
                iterator.args[1]===:getfield && iterator.args[2]==:(metadata.values) ?
                statics[iterator.args[3]] : nothing
            static_range isa AbstractRange{<:Integer} ||
                error("traced slots: expected a preparation-fixed integer range")
            if length(static_range)<=unroll_limit
                returns=false
                for item in static_range
                    unrolled=Any[:($(binding.args[1])=$item)]
                    returns |= blockstatements(x.args[2],unrolled,live)
                    push!(out,Expr(:if,live,Expr(:block,unrolled...)))
                end
                return returns
            end
            nested=Any[]
            returns=blockstatements(x.args[2],nested,live)
            push!(out,Expr(:for,Expr(:(=),binding.args[1],QuoteNode(static_range)),
                Expr(:block,nested...),returns ? live : nothing))
            return returns
        elseif x.head===:(=)
            rhs=value(x.args[2],out,live)
            push!(out,Expr(:(=),x.args[1],rhs))
        else
            push!(out,value(x,out,live))
        end
        false
    end
    live=fresh(:live)
    compiled_statements=Any[:($live=true)]
    blockstatements(body,compiled_statements,live)

    # Prove entry facts for every transition of this privately owned program.
    # Joining construction with every possible exit loses a fact whenever an
    # authored branch can invalidate it. No numerical comparison is involved.
    initial_facts=Dict(name=>RK._canon_current(contextmap[obj].owned,Val(i))
        for ((obj,i),name) in validity)
    function boolvalue(x,facts)
        x isa Bool && return x
        x isa Symbol && return get(facts,x,nothing)
        if x isa Expr && x.head===:call && callee(x.args[1],:!)
            operand=boolvalue(x.args[2],facts)
            operand===nothing || return !operand
        end
        nothing
    end
    function join_facts(a,b)
        Dict(k=>v for (k,v) in a if haskey(b,k) && b[k]===v)
    end
    function assume_bool!(facts,condition,truth)
        if condition isa Symbol
            facts[condition]=truth
        elseif condition isa Expr && condition.head===:call && callee(condition.args[1],:!)
            assume_bool!(facts,condition.args[2],!truth)
        end
        facts
    end
    function specialize_bools(x,facts)
        x isa Symbol && return get(facts,x,x)
        x isa Expr || return x
        if x.head===:block
            return Expr(:block,(specialize_bools(a,facts) for a in x.args)...)
        elseif x.head===:if
            condition=boolvalue(x.args[1],facts)
            if condition!==nothing
                selected=condition ? x.args[2] : length(x.args)==3 ? x.args[3] : nothing
                return specialize_bools(selected,facts)
            end
            yes_facts,no_facts=copy(facts),copy(facts)
            assume_bool!(yes_facts,x.args[1],true)
            assume_bool!(no_facts,x.args[1],false)
            yes=specialize_bools(x.args[2],yes_facts)
            no=specialize_bools(length(x.args)==3 ? x.args[3] : Expr(:block),no_facts)
            merged=join_facts(yes_facts,no_facts)
            empty!(facts);merge!(facts,merged)
            return Expr(:if,x.args[1],yes,no)
        elseif x.head===:for
            incoming=copy(facts)
            invariant=copy(incoming)
            while true
                after=copy(invariant)
                delete!(after,x.args[1].args[1])
                specialize_bools(x.args[2],after)
                next=join_facts(incoming,after)
                next==invariant && break
                invariant=next
            end
            after=copy(invariant)
            loop_body=specialize_bools(x.args[2],after)
            # Include zero-trip and early-exit paths. Losing a fact keeps its
            # runtime validity bit; it never authorizes a stale cache read.
            merged=join_facts(incoming,after)
            empty!(facts);merge!(facts,merged)
            return Expr(:for,x.args[1],loop_body,x.args[3])
        elseif x.head===:(=) && x.args[1] isa Symbol
            original=x.args[2]
            known=boolvalue(original,facts)
            rhs=specialize_bools(original,facts)
            if known===nothing
                delete!(facts,x.args[1])
            else
                facts[x.args[1]]=known
            end
            return Expr(:(=),x.args[1],rhs)
        elseif x.head===:+=
            delete!(facts,x.args[1])
            return Expr(:+=,x.args[1],specialize_bools(x.args[2],facts))
        end
        Expr(x.head,(specialize_bools(a,facts) for a in x.args)...)
    end
    entry_facts=static_currentness ? copy(initial_facts) : Dict{Symbol,Bool}()
    if static_currentness
        while true
            exit_facts=copy(entry_facts)
            specialize_bools(Expr(:block,compiled_statements...),exit_facts)
            next=join_facts(initial_facts,exit_facts)
            next==entry_facts && break
            entry_facts=next
        end
        compiled_statements=specialize_bools(
            Expr(:block,compiled_statements...),copy(entry_facts)).args
    end
    # Backward liveness keeps branch-local temporaries out of the control ABI.
    # Inputs/outputs are explicit tuples; mutable numerical arguments retain
    # their identities through Reactant's ordinary branch mutation handling.
    runtime_validity=Dict(k=>v for (k,v) in validity if !haskey(entry_facts,v))
    persistent=Set{Symbol}([collect(Base.values(slots))...,
        collect(Base.values(runtime_validity))...,:_ts_counter_1,:_ts_counter_2,live])
    variables=union(persistent,Set([:argument,:metadata]))
    function assignments(x)
        found=Set{Symbol}()
        x isa Expr || return found
        if x.head in (:(=),:+=) && x.args[1] isa Symbol
            push!(found,x.args[1])
        end
        for a in x.args;union!(found,assignments(a));end
        found
    end
    union!(variables,assignments(Expr(:block,compiled_statements...)))
    function references(x)
        x isa Symbol && return x in variables ? Set([x]) : Set{Symbol}()
        found=Set{Symbol}()
        x isa Expr || return found
        for a in x.args;union!(found,references(a));end
        found
    end
    lower_control=nothing
    function control_block(x,live_after)
        pieces=x isa Expr && x.head===:block ? x.args : Any[x]
        needed=copy(live_after)
        lowered_reversed=Any[]
        for piece in reverse(pieces)
            emitted,needed=lower_control(piece,needed)
            push!(lowered_reversed,emitted)
        end
        Expr(:block,reverse(lowered_reversed)...),needed
    end
    lower_control=function(x,live_after)
        x isa Expr || return x,union(live_after,references(x))
        if x.head===:block
            return control_block(x,live_after)
        elseif x.head===:if
            t=x.args[2]
            f=length(x.args)==3 ? x.args[3] : Expr(:block)
            outputs=sort!(collect(intersect(union(assignments(t),assignments(f)),live_after)))
            true_code,true_inputs=control_block(t,Set(outputs))
            false_code,false_inputs=control_block(f,Set(outputs))
            inputs=sort!(collect(union(true_inputs,false_inputs)))
            function branch(code)
                branch_args=fresh(:args)
                locals=sort!(collect(union(Set(inputs),assignments(code))))
                bindings=Any[Expr(:local,name) for name in locals]
                append!(bindings,[:($name=getfield($branch_args,$i)) for (i,name) in enumerate(inputs)])
                Expr(:->,Expr(:tuple,branch_args),Expr(:block,
                    bindings...,code,Expr(:tuple,outputs...)))
            end
            true_function,false_function=branch(true_code),branch(false_code)
            call=Expr(:call,GlobalRef(Reactant.ReactantCore,:traced_if),
                Expr(:parameters,Expr(:kw,:track_numbers,Number)),
                x.args[1],true_function,false_function,Expr(:tuple,inputs...))
            emitted=isempty(outputs) ? call : Expr(:(=),Expr(:tuple,outputs...),call)
            needed=union(setdiff(live_after,Set(outputs)),Set(inputs),references(x.args[1]))
            return emitted,needed
        elseif x.head===:(=) && x.args[1] isa Symbol
            needed=union(setdiff(live_after,Set([x.args[1]])),references(x.args[2]))
            return x,needed
        elseif x.head===:for
            induction=x.args[1].args[1]
            range=x.args[1].args[2].value
            stop=x.args[3]
            carried=copy(live_after)
            stop===nothing || push!(carried,stop)
            loop_code=nothing
            while true
                loop_code,read_before=control_block(x.args[2],carried)
                next=union(carried,setdiff(read_before,Set([induction])))
                next==carried && break
                carried=next
            end
            names=sort!(collect(carried))
            tuple_argument=fresh(:loop_args)
            index_argument=fresh(:loop_index)
            locals=sort!(collect(union(carried,assignments(loop_code),Set([induction]))))
            bindings=Any[Expr(:local,name) for name in locals]
            append!(bindings,[:($name=getfield($tuple_argument,$i)) for (i,name) in enumerate(names)])
            push!(bindings,:($induction=$index_argument))
            fn=Expr(:->,Expr(:tuple,tuple_argument,index_argument),
                Expr(:block,bindings...,loop_code,Expr(:tuple,names...)))
            stop_index=stop===nothing ? 0 : findfirst(==(stop),names)
            descriptor=Expr(:call,SlotLoop,fn,QuoteNode(range),:(Val($stop_index)))
            invocation=Expr(:call,run_slot_loop,descriptor,Expr(:tuple,names...))
            return Expr(:(=),Expr(:tuple,names...),invocation),carried
        end
        x,union(live_after,references(x))
    end
    ordered=sort!(collect(slots);by=p->string(first(p)))
    # Construction and every transition exit agree on these removed flags.
    # They need no traced input or persistent transition output. Inner control
    # still derives any temporarily varying values through ordinary liveness.
    valid_ordered=sort!(collect(runtime_validity);by=p->string(first(p)))
    setup=Any[:($(last(p))=getfield(state.values,$i)) for (i,p) in enumerate(ordered)]
    append!(setup,[:($name=$known) for (name,known) in sort!(collect(entry_facts);by=first)])
    append!(setup,[:($(last(p))=getfield(state.current,$i)) for (i,p) in enumerate(valid_ordered)])
    values=Tuple(RK._canon_slot(contextmap[obj].owned,Val(i)) for ((obj,i),_) in ordered)
    masks=Tuple(RK._canon_current(contextmap[obj].owned,Val(i)) for ((obj,i),_) in valid_ordered)
    result=:( (values=$(Expr(:tuple,last.(ordered)...)),current=$(Expr(:tuple,last.(valid_ordered)...))) )
    loop,_=control_block(Expr(:block,compiled_statements...),persistent)
    expression=:((state,argument,counts,metadata)->begin
        $(setup...)
        _ts_counter_1=counts[1]
        _ts_counter_2=counts[2]
        $loop
        (; state=$result,argument,counts=(_ts_counter_1,_ts_counter_2))
    end)
    expanded=macroexpand(@__MODULE__,slot_code_copy(expression))
    (;f=compile_slot_code(expanded;reuse_code),expression,expanded,metadata=SlotStatic(Tuple(statics)),
      state=(;values,current=masks),ordered,entry_facts,valid_ordered)
end

struct SlotLoop{F,R,S}
    body::F
    range::R
end
SlotLoop(body,range,::Val{S}) where {S}=SlotLoop{typeof(body),typeof(range),S}(body,range)
Reactant.make_tracer(seen, previous::SlotLoop, path, mode; kwargs...) = previous
Reactant.traced_type_inner(::Type{T}, seen, mode::Reactant.TraceMode,
    track_numbers::Type, ndevices, runtime) where {T<:SlotLoop} = T
slot_loop_live(::SlotLoop{F,R,0},state) where {F,R}=true
slot_loop_live(::SlotLoop{F,R,S},state) where {F,R,S}=getfield(state,S)
slot_loop_scalar(value)=value
# Julia scalar assignment copies a value. Reactant numbers are mutable trace
# wrappers, so separate logical slots need independent wrappers at a while
# boundary even when their current SSA value is identical. copy emits no
# arithmetic or array copy; mutable array aliases remain unchanged.
slot_loop_scalar(value::Reactant.TracedRNumber)=copy(value)
slot_loop_scalar(value::T) where {T<:Number}=
    Reactant.promote_to(Reactant.TracedRNumber{T},value)
function run_slot_loop(loop,state)
    state=map(slot_loop_scalar,state)
    counter=0
    Reactant.@trace while (counter<length(loop.range)) & slot_loop_live(loop,state)
        state=map(slot_loop_scalar,loop.body(state,first(loop.range)+counter*step(loop.range)))
        counter+=1
    end
    state
end

function traced_slot_batch(program,state,seed,counts,n)
    Reactant.@trace for _ in 1:n
        result=program(state,seed,counts)
        state=result.state
        seed=result.seed
        counts=result.counts
    end
    (;state,seed,counts)
end

struct TracedSlotCall{F,M}
    f::F
    metadata::M
end
Reactant.make_tracer(seen, previous::TracedSlotCall, path, mode; kwargs...) = previous
Reactant.traced_type_inner(::Type{T}, seen, mode::Reactant.TraceMode,
    track_numbers::Type, ndevices, runtime) where {T<:TracedSlotCall} = T
function (program::TracedSlotCall)(state,seed,counts)
    # RNG packing belongs to this caller adapter. The emitted finite MethodIR
    # function accepts and returns its ordinary runtime argument unchanged in
    # representation, including any source-visible effects on that argument.
    result=program.f(state,Reactant.ReactantRNG(seed),counts,program.metadata)
    (;state=result.state,seed=result.argument.seed,counts=result.counts)
end
