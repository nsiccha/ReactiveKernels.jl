using ReactiveKernels, Random
using ReactiveKernels.NativeSlotCompiler: NativePoint,native_endpoint,
    fast_native_factory,compile_native_slots,slot_chain,slotplan,slotfields
const RK=ReactiveKernels

@kernel scalar_point(x)=begin
end
@kernel add_twice!(point;amount)=begin
    point.x=point.x+amount
    point.x=point.x+amount
end
@kernel with_runtime_keyword(point;update=nothing)=begin
    advance!(unused)=begin
        update(point;amount=point.x)
        return
    end
end
@kernel missing_second_keyword(point;update=nothing)=begin
    advance!(unused)=begin
        update(point;amount=point.x)
        update(point)
        return
    end
end
@kernel extra_keyword(point;update=nothing)=begin
    advance!(unused)=begin
        update(point;amount=point.x,unknown=1.0)
        return
    end
end
function keyword_parent(source;update=add_twice!)
    endpoint=native_endpoint(scalar_point,add_twice!,(1.0,))
    fast_native_factory(source,NativePoint(endpoint);update)
end
function keyword_program()
    parent=keyword_parent(with_runtime_keyword)
    compile_native_slots(parent.kernel,parent.state,:advance!;
        endpoints=parent.endpoints,effects=parent.effects)
end
function read_x(program)
    context=program.contexts[2]
    _,slot=RK.kernel_plan_field(slotplan(context),slotfields(context)[:x])
    RK._canon_slot(context.owned,Val(slot))
end
program=keyword_program()
slot_chain(program,0,1)
read_x(program)==3.0 || error("keyword expression was not evaluated once before the helper body")
for (source,update,reason) in (
        (missing_second_keyword,add_twice!,"REQUIRED keyword"),
        (extra_keyword,add_twice!,"not accepted"),
        (with_runtime_keyword,partial(add_twice!;amount=2.0),"both bound and runtime"))
    rejected=false
    try
        keyword_parent(source;update)
    catch err
        occursin(reason,sprint(showerror,err)) || rethrow()
        rejected=true
    end
    rejected || error("invalid per-callsite keyword contract admitted")
end
println("native_runtime_keyword_evaluated_once=true per_callsite_rejections=true")
