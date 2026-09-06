# Internal construction-only endpoint over canonical native storage.
# Copies isolate owned fields and preserve the fixed shared authorities.
struct NativePoint{Key,P,O,S}
    pf::P
    owned::O
    shared::S
end
function NativePoint(ep)
    Key=RK.kernel_plan_key(RK.kernel_prepared_plan(ep.pf))
    NativePoint{Key,typeof(ep.pf),typeof(ep.owned),typeof(ep.shared)}(ep.pf,ep.owned,ep.shared)
end
@generated function point_read(p::NativePoint{Key},::Val{Name}) where {Key,Name}
    matches=filter(slot->last(slot[1])===Name,Key[2])
    isempty(matches) && return :(throw(ArgumentError("unknown endpoint property")))
    _,_,role,slot=first(matches)
    object=role===:owned ? :owned : :shared
    :(RK._canon_slot(getfield(p,$(QuoteNode(object))),Val($slot)))
end
Base.getproperty(p::NativePoint,name::Symbol)=point_read(p,Val(name))
function Base.deepcopy_internal(p::NativePoint,dict::IdDict)
    NativePoint((pf=getfield(p,:pf),
        owned=Base.deepcopy_internal(getfield(p,:owned),dict),shared=getfield(p,:shared)))
end
RK._recipe_dom_raw_deepcopy(::Type{<:NativePoint})=true
RK._recipe_dom_deepcopy(::Type{<:NativePoint})=true
RK._sm_structural_copy(p::NativePoint)=deepcopy(p)

function fast_native_factory(skeleton,args...;kwargs...)
    signature=getfield(getfield(skeleton,:spec_snapshot),:call_signature)
    P,K=typeof(signature).parameters[1:2]
    values=RK._kernel_signature_invoke(RK._KernelSignatureCallable(tuple,signature),
        args,NamedTuple(kwargs))
    sources=NamedTuple{(P...,K...)}(values)
    descriptors=Pair{Symbol,Any}[];effects=Pair{Symbol,Any}[]
    for name in sort!(collect(RK._kernel_factory_called_fields(skeleton)))
        source=getproperty(sources,name)
        if source === nothing
            push!(descriptors,name=>nothing)
        else
            callable=RK._prepare_callable(name,source)
            push!(descriptors,name=>RK.prepared_callable_registration(callable))
            controls=RK.prepared_callable_kwargs(callable)
            push!(effects,name=>(source=RK.prepared_callable_source(callable),
                                controls=controls===nothing ? NamedTuple() : controls))
        end
    end
    bindings=RK.stateful_compiler_bindings(;descriptors...)
    pf=RK._prepare_stateful(skeleton;field_regs=Dict(descriptors))
    owned,shared=RK._construct_bound_stateful(skeleton,pf,bindings,args...;kwargs...)
    children=Pair{Symbol,Any}[]
    for slot in RK.kernel_plan_slots(RK.kernel_prepared_plan(pf))
        role,i=RK.kernel_plan_field(RK.kernel_prepared_plan(pf),slot.canon)
        object=role===:owned ? owned : shared
        value=RK._canon_slot(object,Val(i))
        if value isa NativePoint
            push!(children,last(slot.path)=>(pf=getfield(value,:pf),
                owned=getfield(value,:owned),shared=getfield(value,:shared)))
        end
    end
    (; kernel=(skeleton=skeleton,prepared=pf,bindings),state=(;owned,shared),
       endpoints=NamedTuple(children),effects=NamedTuple(effects))
end
