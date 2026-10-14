# Bound-data contiguous-index runs for native scalar reductions.
# The derived metadata is an ordinary data-only graph dependency. It therefore
# occupies the existing leading constant slots and is recomputed on rebinding;
# no bound values enter a cached template's operation tail or emitted syntax.

const _IndexRunData = Tuple{Vector{Int},Vector{Int},Vector{Bool},Int,Bool}
const _INDEX_RUN_MINIMUM = 16

# Unknown index representations retain the original gather and loop. A concrete
# return type also keeps concretely typed client programs concrete.
_index_run_metadata(indices) = (Int[],Int[],Bool[],0,false)
function _index_run_metadata(indices::Vector{Int})
    starts,stops,grouped = Int[],Int[],Bool[]
    n=length(indices)
    n==0 && return (starts,stops,grouped,n,false)
    # Short adjacent runs share one flat span. They keep the original scalar
    # body and scatter rather than paying a new outer-loop iteration per run.
    flat=1; lo=1
    @inbounds for i in 2:(n+1)
        if i==n+1 || indices[i]!=indices[i-1]
            hi=i-1
            if hi-lo+1 >= _INDEX_RUN_MINIMUM
                if flat<lo
                    push!(starts,flat);push!(stops,lo-1);push!(grouped,false)
                end
                push!(starts,lo);push!(stops,hi);push!(grouped,true)
                flat=i
            end
            lo=i
        end
    end
    if flat<=n
        push!(starts,flat);push!(stops,n);push!(grouped,false)
    end
    (starts,stops,grouped,n,any(grouped))
end

# The extra dependency is compiler metadata only. The ordinary and tensorized
# semantic fallback return exactly the original borrowed gather representation.
@inline _readonly_gather_runs(parent,indices,runs) = _readonly_gather(parent,indices)

_index_run_gather(op) = op === _readonly_gather ||
    (op isa _KernelSourceOp && op.call_identity !== nothing &&
     first(op.call_identity) === _readonly_gather)

function _partial_index_runs(p::Plan,bound_values::Vector{Value})
    isempty(bound_values) && return p
    g=p.graph
    # Limit metadata construction to ancestors of a visible total-only plate.
    wanted=Set(canon_id(g,w.id) for w in p.want)
    needed=Set{Int}()
    for r in p.recipes
        r.op isa _AuthoredPlateOp || continue
        total=_authored_plate_sum_recipe(p,r)
        total===nothing && continue
        id=canon_id(g,only(r.outputs).id)
        id in wanted && continue
        any(q -> q!==total && any(v->canon_id(g,v.id)==id,q.inputs),p.recipes) && continue
        union!(needed,(canon_id(g,v.id) for v in r.inputs))
    end
    isempty(needed) && return p
    for r in Iterators.reverse(p.recipes)
        any(v->canon_id(g,v.id) in needed,r.outputs) || continue
        union!(needed,(canon_id(g,v.id) for v in r.inputs))
    end
    bound=Set(canon_id(g,v.id) for v in bound_values)
    _,_,dataonly=_partial_split(p,bound)
    eligible=Set(r.id for r in p.recipes if !r.effectful &&
        length(r.inputs)==2 && length(r.outputs)==1 && _index_run_gather(r.op) &&
        canon_id(g,only(r.outputs).id) in needed &&
        canon_id(g,r.inputs[2].id) in dataonly)
    isempty(eligible) && return p
    copied=Graph(copy(g.values),copy(g.recipes),copy(g.producers),copy(g.aliases),g.version)
    recipes=Recipe[]
    metadata=Dict{Int,Value}()
    fresh=minimum((r.id for r in p.recipes);init=0)-1
    for r in p.recipes
        if !(r.id in eligible)
            push!(recipes,r);continue
        end
        index=r.inputs[2];cid=canon_id(g,index.id)
        if !haskey(metadata,cid)
            value=value!(copied,gensym(:index_runs),_IndexRunData)
            metadata[cid]=value
            push!(recipes,Recipe(fresh,(index,),(value,),_index_run_metadata,
                0.0,nothing,false))
            fresh-=1
        end
        push!(recipes,Recipe(r.id,(r.inputs...,metadata[cid]),r.outputs,
            _readonly_gather_runs,r.cost,r.cse_key,false))
    end
    template=Plan(copied,p.have,p.want,p.recipes,p.producer,p.cost,p.candidates)
    _partial_subplan(template,p.have,p.want,recipes)
end

_index_run_call(x,f,n=nothing)=x isa Expr && x.head===:call &&
    x.args[1]==f && (n===nothing || length(x.args)==n+1)
function _index_run_visit(f,x)
    x isa Expr || return
    x.head in (:quote,:inert) && return
    f(x)
    foreach(a->_index_run_visit(f,a),x.args)
end
function _index_run_replace(f,x)
    x isa Expr || return x
    x.head in (:quote,:inert) && return x
    y=f(x)
    y===x || return y
    Expr(x.head,(_index_run_replace(f,a) for a in x.args)...)
end

# This local emitter transformation reads only compiler-owned operation identity
# and generated data flow. It adds no operation slots, so native and tensorized
# products retain identical tables. Tensorized execution keeps the original loop.
function _lower_index_runs_native(body::Expr,ops)
    any(op->op===_readonly_gather_runs,ops) || return body
    gathers=Dict{Symbol,Symbol}();raw=Dict{Symbol,Symbol}();projections=Dict{Symbol,Symbol}()
    _index_run_visit(body) do ex
        ex.head===:(=) && length(ex.args)==2 && ex.args[1] isa Symbol || return
        lhs,rhs=ex.args
        rhs isa Expr && rhs.head===:call || return
        fn=rhs.args[1]
        if fn isa Expr && fn.head===:ref && length(fn.args)==2 &&
                fn.args[1]===_OPS_ARG && fn.args[2] isa Int
            ops[fn.args[2]]===_readonly_gather_runs && length(rhs.args)==4 &&
                rhs.args[4] isa Symbol && (gathers[lhs]=rhs.args[4])
        elseif _index_run_call(rhs,GlobalRef(Base,:broadcastable),1)
            m=get(gathers,rhs.args[2],nothing);m===nothing || (raw[lhs]=m)
        elseif _index_run_call(rhs,GlobalRef(Base.Broadcast,:preprocess),2)
            m=get(raw,rhs.args[3],nothing);m===nothing || (projections[lhs]=m)
        end
    end
    isempty(projections) && return body
    _index_run_replace(body) do ex
        ex.head===:if && length(ex.args)==3 || return ex
        # A previously grouped reduction has an inner conditional here. Only
        # the original macro-expanded SIMD block is eligible, making this
        # pass idempotent when an already lowered body is embedded again.
        native=ex.args[2]
        native isa Expr && native.head===:block || return ex
        fallback=ex.args[3]
        fallback isa Expr && fallback.head===:for || return ex
        binding,cellbody=fallback.args
        binding isa Expr && binding.head===:(=) || return ex
        index,iteration=binding.args
        index isa Symbol || return ex
        _index_run_call(iteration,GlobalRef(@__MODULE__,:_plate_cells),1) || return ex
        domain=iteration.args[2]
        _index_run_call(domain,GlobalRef(Base,:CartesianIndices),1) || return ex
        axes=domain.args[2];condition=ex.args[1]
        condition==:(length($axes)==1) || return ex
        selected=Symbol[];additions=Expr[]
        _index_run_visit(cellbody) do cell
            if _index_run_call(cell,GlobalRef(Base.Broadcast,:_broadcast_getindex),2) &&
                    cell.args[3]==index && haskey(projections,cell.args[2])
                cell.args[2] in selected || push!(selected,cell.args[2])
            end
            cell.head===:(=) && length(cell.args)==2 &&
                _index_run_call(cell.args[2],GlobalRef(@__MODULE__,:_plate_total_add),2) &&
                cell.args[2].args[2]==cell.args[1] && push!(additions,cell)
        end
        isempty(selected) && return ex
        length(additions)==1 || return ex
        accumulator=only(additions).args[1]
        accumulator isa Symbol || return ex
        metadata=projections[first(selected)]
        # Distinct index maps keep their original scalar projection. Only inputs
        # sharing this exact data-only metadata are invariant over its runs.
        filter!(s->projections[s]===metadata,selected)
        shared=Dict(s=>gensym(:index_run_value) for s in selected)
        part,run,lo,hi,linear,isgrouped=(gensym(s) for s in (:run_partial,:run,:run_lo,:run_hi,:run_linear,:run_grouped))
        grouped_body=_index_run_replace(cellbody) do cell
            if _index_run_call(cell,GlobalRef(Base.Broadcast,:_broadcast_getindex),2) &&
                    cell.args[3]==index && haskey(shared,cell.args[2])
                return shared[cell.args[2]]
            end
            cell
        end
        grouped_body=_index_run_replace(grouped_body) do cell
            cell.head===:(=) && cell.args[1]==accumulator &&
                _index_run_call(cell.args[2],GlobalRef(@__MODULE__,:_plate_total_add),2) ?
                :($part=$(GlobalRef(@__MODULE__,:_plate_total_add))($part,$(cell.args[2].args[3]))) : cell
        end
        simd(block)=macroexpand(@__MODULE__,Expr(:macrocall,GlobalRef(Base,Symbol("@simd")),
            LineNumberNode(0),Expr(:for,Expr(:(=),linear,:($lo:$hi)),
                Expr(:block,:($index=Base.CartesianIndex($linear)),block))))
        grouped=Expr(:block)
        for s in selected
            push!(grouped.args,:($(shared[s])=$(GlobalRef(Base.Broadcast,:_broadcast_getindex))($s,Base.CartesianIndex($lo))))
        end
        push!(grouped.args,:($part=zero($accumulator)),simd(grouped_body),
            :($accumulator=$(GlobalRef(@__MODULE__,:_plate_total_add))($accumulator,$part)))
        reads=_inbounds_expr(Expr(:block,
            :($lo=getfield($metadata,1)[$run]),:($hi=getfield($metadata,2)[$run]),
            :($isgrouped=getfield($metadata,3)[$run])))
        # All three vectors are created together, indexed by eachindex(starts).
        # Authored array accesses remain under their original checks.
        runbody=Expr(:block,reads,Expr(:if,isgrouped,grouped,simd(deepcopy(cellbody))))
        runs=Expr(:for,Expr(:(=),run,:(eachindex(getfield($metadata,1)))),runbody)
        guard=:(getfield($metadata,5) && Base.only($axes)==Base.OneTo(getfield($metadata,4)) && $accumulator isa Number)
        Expr(:if,condition,Expr(:if,guard,runs,ex.args[2]),fallback)
    end
end
