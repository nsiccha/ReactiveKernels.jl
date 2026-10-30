using ReactiveKernels
using ReactiveKernels.NativeSlotCompiler: NativePoint,native_endpoint,
    fast_native_factory,compile_native_slots,slot_chain,slotplan,slotfields
const RK=ReactiveKernels

@kernel coupled_point(x,w)=begin
    y=2.0 .* x
    energy=sum(y)^2
    z=energy+w
end
@kernel point_effect!(point;delta)=begin
    point.x .= point.x .+ delta
    point.w=point.w+delta
    point.z
    return
end
@kernel partial_copy(init)=begin
    other=deepcopy(init)
    value=0.0
    advance!(invalidate)=begin
        other.w=10.0
        if invalidate
            init.x .= init.x .+ 1.0
        end
        other.x .= init.x
        return
    end
    observe!(unused)=begin
        value=other.z
        return
    end
    check!(invalidate)=begin
        other.w=10.0
        if invalidate
            init.x .= init.x .+ 1.0
        end
        other.x .= init.x
        value=other.z
        return
    end
end
@kernel borrowed_partial_copy(init;value=zeros(2))=begin
    other=deepcopy(init)
    advance!(unused)=begin
        init.x .= init.x .+ 1.0
        init.y
        retained=other.y
        other.x .= init.x
        value .= retained
        return
    end
end
function make_parent(source=partial_copy)
    point=native_endpoint(coupled_point,point_effect!,([1.0,2.0],3.0))
    fast_native_factory(source,NativePoint(point))
end
prepare(parent,name)=compile_native_slots(parent.kernel,parent.state,name;
    endpoints=parent.endpoints,effects=parent.effects)
function slot_index(context,name)
    _,i=RK.kernel_plan_field(slotplan(context),slotfields(context)[name])
    i
end
current(context,name)=RK._canon_current(context.owned,Val(slot_index(context,name)))
read_slot(context,name)=RK._canon_slot(context.owned,Val(slot_index(context,name)))
for invalid in (false,true)
    parent=make_parent()
    advance=prepare(parent,:advance!)
    slot_chain(advance,invalid,1)
    other=advance.contexts[3]
    current(other,:y)==!invalid || error("source cache validity was not transferred")
    current(other,:energy)==!invalid || error("scalar source cache validity was not transferred")
    !current(other,:z) || error("cache depending on an uncopied owned input was reused")
    observe=prepare(parent,:observe!)
    slot_chain(observe,0,1)
    read_slot(observe.contexts[1],:value)==(invalid ? 110.0 : 46.0) || error("partial-copy result")
end
borrowed=prepare(make_parent(borrowed_partial_copy),:advance!)
slot_chain(borrowed,0,1)
read_slot(borrowed.contexts[1],:value)==[2.0,4.0] || error("hidden cache transfer changed a borrowed array")
!current(borrowed.contexts[3],:y) || error("borrowed destination cache was overwritten")
println("native_partial_cache_validity=true uncopied_input_invalidates=true borrowed_cache_preserved=true")
