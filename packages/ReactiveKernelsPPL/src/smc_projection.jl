# A portable subset of native parameter geometry. Historical entries and
# unused bound columns/definitions never enter the returned record.
_smc_output_name(e::LayoutEntry) = e.kind === :coefficient ? e.predictor : e.name
_smc_output_path(layout,e) = get(layout.name_paths,_smc_output_name(e),(_smc_output_name(e),))

function _smc_parameter_projection(layout::LayoutTable,parameters)
    names=Tuple(Symbol.(parameters))
    !isempty(names) && length(unique(names))==length(names) ||
        throw(ArgumentError("selected parameter names must be nonempty and unique"))
    wanted=Set(names)
    public_name(e)=Symbol(join(_smc_output_path(layout,e),"."))
    selected=filter(e -> public_name(e) in wanted,layout.entries)
    Set(public_name.(selected))==wanted ||
        throw(ArgumentError("selected parameters must name complete authored parameter blocks"))
    for e in selected
        e.kind===:scan && throw(ArgumentError("cannot carry historical scan coordinates as static parameters"))
        (e.sampling===nothing && !haskey(layout.cell_geometries,e.name)) ||
            throw(ArgumentError("selected parameter $(public_name(e)) has custom geometry without a portable native layout; use a static calibration and an explicit mapping"))
        isempty(e.scan_blocks) || throw(ArgumentError("selected parameter retains historical transform blocks"))
    end
    sampled=Set(_smc_output_name(e) for e in layout.entries)
    kept=Set(_smc_output_name(e) for e in selected)
    expressions=Dict{Symbol,Any}()
    constants=Dict{Symbol,Any}()
    active=Set{Symbol}()
    freezing=Set{Symbol}()
    dependency_cache=Dict{Symbol,Bool}()
    function dynamic(name::Symbol)
        name in sampled && return true
        haskey(dependency_cache,name) && return dependency_cache[name]
        haskey(layout.bound_values,name) && return false
        haskey(layout.bound_exprs,name) || return false
        name in active && throw(ArgumentError("cyclic parameter bound at $name"))
        push!(active,name)
        value=any(dynamic,_value_symbols(layout.bound_exprs[name]))
        delete!(active,name)
        dependency_cache[name]=value
        value
    end
    lookup=_layout_lookup(layout,Dict{Symbol,Any}())
    function freeze(ex)
        ex isa Real && return ex
        # One bound per element, already this binding's numbers.
        ex isa AbstractArray{<:Real} && return copy(ex)
        if !any(dynamic,_value_symbols(ex))
            value=_eval_value_expr(ex,lookup,:selected_parameter_bound)
            value isa Real || throw(ArgumentError(
                "a selected transform still needs nonscalar calibration data; freeze a scalar summary in the authored model"))
            return value
        elseif ex isa Symbol
            ex in kept && return ex
            ex in sampled && throw(ArgumentError("selected transform depends on discarded parameter $ex"))
            if !haskey(expressions,ex)
                haskey(layout.bound_exprs,ex) || throw(ArgumentError("unavailable transform dependency $ex"))
                ex in freezing && throw(ArgumentError("cyclic parameter bound at $ex"))
                push!(freezing,ex)
                expressions[ex]=freeze(layout.bound_exprs[ex])
                delete!(freezing,ex)
            end
            return ex
        elseif ex isa Expr
            # Function names are syntax, not bound data. Only their arguments
            # are folded; ordinary host constrain keeps the transform math.
            if ex.head===:call
                return Expr(:call,ex.args[1],map(freeze,ex.args[2:end])...)
            end
            return Expr(ex.head,map(freeze,ex.args)...)
        end
        throw(ArgumentError("unsupported portable parameter bound $ex"))
    end
    entries=NamedTuple[]
    indices=Int[]
    paths=Dict{Symbol,Tuple{Vararg{Symbol}}}()
    cells=Dict{Symbol,Tuple{Symbol,Symbol}}()
    offset=1
    for e in selected
        append!(indices,e.offset:e.offset+e.size-1)
        push!(entries,(;kind=e.kind,predictor=e.predictor,name=e.name,labels=copy(e.labels),
            offset,size=e.size,transform=e.transform,lo=freeze(e.lo),hi=freeze(e.hi),dims=copy(e.dims),
            elementwise=e.elementwise))
        output=_smc_output_name(e)
        haskey(layout.name_paths,output) && (paths[output]=layout.name_paths[output])
        if haskey(layout.cell_latents,e.name)
            cell=layout.cell_latents[e.name]
            cells[e.name]=cell
            constants[last(cell)]=deepcopy(_cell_latent_dims(layout,e.name))
        end
        offset+=e.size
    end
    path_by_name=Dict(public_name(e)=>_smc_output_path(layout,e) for e in selected)
    output_paths=Tuple(path_by_name[name] for name in names)
    record=deepcopy((;entries,total=offset-1,name_paths=paths,bound_values=constants,
        bound_exprs=expressions,cell_latents=cells))
    return (;record,indices,names,output_paths)
end

function _smc_projection_layout(record)
    entries=LayoutEntry[LayoutEntry(e.kind,e.predictor,e.name,e.labels,e.offset,e.size,
        e.transform,e.lo,e.hi,e.dims,LayoutEntry[],nothing,nothing,nothing,
        (nothing,nothing),get(e,:elementwise,(false,false))) for e in record.entries]
    LayoutTable(entries,record.total,record.name_paths,record.bound_values,
        record.bound_exprs,record.cell_latents)
end

struct _SMCParameterMap{L,N,P}
    layout::L
    names::N
    paths::P
end
function (map::_SMCParameterMap)(u)
    values=constrain(map.layout,u)
    NamedTuple{map.names}(Tuple(_scope_lookup(values,path) for path in map.paths))
end
function _smc_parameter_map(carried::SMCCarriedReference)
    hasproperty(carried.frame,:parameters) || throw(ArgumentError(
        "carry_smc(completed; parameters=...) is required for the automatic parameter mapping"))
    projection=carried.frame.parameters
    _SMCParameterMap(_smc_projection_layout(deepcopy(projection.layout)),
        projection.names,projection.paths)
end

"""
    smc_parameters(carried::SMCCarriedReference, u)

Restore the complete selected constrained parameter tuple from a compact
unconstrained position. Construct `carried` with
`carry_smc(completed; parameters=authored_names, calibration_id)` first.
Uses the model's native `constrain` transforms, including dependent static
bounds, nested author paths, vector and array geometry. It retains no old
latent coordinates or observations. `prepare_filter_smc` uses this mapping
automatically when `parameters` is omitted.
"""
smc_parameters(carried::SMCCarriedReference,u::AbstractVector) = _smc_parameter_map(carried)(u)
