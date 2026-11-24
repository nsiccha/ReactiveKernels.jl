# In a sum of group totals the derivative of the outer addition with respect
# to each completed inner total is one. Establish that use from the program
# before seeding the inner sweep, and keep its scope inside the group loop.
function _pr_additive_groups(body, totals, terminal, assignments)
    groups = IdDict{Expr,_PRAdditiveGroup}()
    intermediate = setdiff(totals, terminal)
    declared = Dict{Symbol,Any}()
    isempty(intermediate) && return groups, declared
    reads = _rw_reads!(Dict{Symbol,Int}(), body)
    _index_run_visit(body) do node
        node.head === :local || return
        for arg in node.args
            Meta.isexpr(arg, :(::), 2) && arg.args[1] isa Symbol &&
                (declared[arg.args[1]] = arg.args[2])
        end
    end
    covered = Set{Symbol}()
    _index_run_visit(body) do loop
        Meta.isexpr(loop, :for, 2) || return
        plate = _rw_plate_loop(loop)
        plate === nothing && return
        cell = only(plate.bodies)
        xs = Any[a for a in cell.args if !(a isa LineNumberNode)]
        isempty(xs) && return
        add = last(xs)
        Meta.isexpr(add, :(=), 2) || return
        outer, rhs = add.args
        outer in terminal && _index_run_call(rhs, GlobalRef(@__MODULE__, :_plate_total_add), 2) &&
            rhs.args[2] === outer || return
        result = rhs.args[3]
        k = length(xs) - 1
        while k > 0
            s = xs[k]
            Meta.isexpr(s, :(=), 2) && s.args[1] === result && result isa Symbol &&
                get(reads, result, 0) == 1 || return
            rhs = s.args[2]
            if haskey(declared, result)
                # Typed assignment can call convert. Only a raw value already
                # asserted to that type is an identity on this path.
                Meta.isexpr(rhs, :(::), 2) && rhs.args[2] === declared[result] || return
                rhs = rhs.args[1]
            end
            result = rhs
            result isa Symbol || break
            k -= 1
        end
        _index_run_call(result, GlobalRef(@__MODULE__, :_plate_total_value), 1) || return
        inner = result.args[2]
        inner in intermediate && !(inner in covered) || return
        prefix = xs[1:k-1]
        # One reset per group, no external uses or updates, and only the
        # accumulator operand of each addition and this final value read.
        seed = findall(prefix) do s
            Meta.isexpr(s, :(=), 2) && s.args[1] === inner &&
                _index_run_call(s.args[2], GlobalRef(@__MODULE__, :_plate_total_seed), 2)
        end
        length(seed) == 1 || return
        updates = [v for (name, v) in assignments if name === inner]
        additions = [v for v in updates if
            _index_run_call(v, GlobalRef(@__MODULE__, :_plate_total_add), 2) && v.args[2] === inner]
        isempty(additions) && return
        length(updates) == length(additions) + 1 && get(reads, inner, 0) == length(additions) + 1 || return
        after = _pr_assignments!(Pair{Symbol,Any}[], Expr(:block, prefix[only(seed)+1:end]...))
        count(p -> first(p) === inner, after) == length(additions) || return
        scope = Expr(:block, prefix...)
        haskey(_rw_reads!(Dict{Symbol,Int}(), scope), outer) && return
        locals = sort!(collect(_program_bound!(Set{Symbol}(), scope)))
        groups[loop] = _PRAdditiveGroup(prefix, xs[k:end], plate.index, locals)
        push!(covered, inner)
    end
    covered == intermediate || _pr_fail("an intermediate plate total needs a propagated adjoint")
    groups, declared
end

@inline _pr_group_view(::Nothing, indices) = nothing
@inline _pr_group_view(shadow::Vector{<:Vector}, row::Int) = shadow[row]
@inline _pr_group_view(shadow, indices) = view(shadow, indices)
@inline _pr_group_view(shadow::_PRMappedShadow, indices) =
    _PRMappedShadow(shadow.f, _pr_group_view(shadow.storage, indices))

# One logical cotangent per row entry, including separate slots for two
# logical rows that alias. The reverse accumulates both into the shared row.
# The prefix offsets occupy the head of this operand's leased tape slice.
struct _PRRaggedBuffer{O,V}
    offsets::O
    values::V
end
function _pr_ragged_buffer(slice, rows)
    n = length(rows)
    offsets = reinterpret(Int64, view(slice, 1:n+1))
    values = view(slice, n+2:length(slice))
    offsets[1] = 0
    @inbounds for i in eachindex(rows)
        offsets[i+1] = offsets[i] + length(rows[i])
    end
    _PRRaggedBuffer(offsets, values)
end
@inline function _pr_group_view(buffer::_PRRaggedBuffer, i::Int)
    view(buffer.values, buffer.offsets[i]+1:buffer.offsets[i+1])
end
function _pr_check_shadow(s::_PRRaggedBuffer, rows::Vector{Vector{Float64}})
    length(s.offsets) == length(rows) + 1 &&
        all(i -> s.offsets[i+1] - s.offsets[i] == length(rows[i]), eachindex(rows)) ||
        throw(DimensionMismatch("ragged plate shadow has different row lengths"))
    nothing
end

struct _PRGroupProjection{Declared}
    parent::Int
    rows::Int
    grouped::Bool
    batched::Bool
    only_group_reads::Bool
end

# A gathered group operand uses its own indices in both directions. A view
# of the parent's cotangent storage scatters repeated positions on addition.
function _pr_group_shadows!(reads, groups, params, active, prepared, mod, values, declared)
    sources = Set{Symbol}()
    bindings = IdDict{Expr,Expr}()
    scatters = IdDict{Expr,Expr}()
    projections = _PRGroupProjection[]
    for group in Base.values(groups), s in group.prefix
        Meta.isexpr(s, :(=), 2) || continue
        name, rhs = s.args
        name in active && rhs isa Expr || continue
        source = Meta.isexpr(rhs, :let, 2) ? _rw_peel_let(rhs, x -> x isa Symbol) : rhs
        isgather = Meta.isexpr(source, :call, 3) &&
            _pr_call_function(source.args[1], mod, values) === _cell_array_gather
        isgather || Meta.isexpr(source, :ref, 2) || continue
        parent, indices = source.args[end-1:end]
        parent isa Symbol && parent in params && indices isa Symbol && !(indices in active) ||
            _pr_fail("a group gather needs a parameter and data indices")
        declaration = get(declared, name, Any)
        declaration isa Type || _pr_fail("a group read has a dynamic type declaration")
        # Select the projection rule at dispatch for the actual source and
        # index types. A user-defined indexed operation can have a different
        # pullback even if it returns an ordinary vector.
        rows = indices
        if !(rows in params)
            definition = findfirst(group.prefix) do stmt
                Meta.isexpr(stmt, :(=), 2) && stmt.args[1] === indices
            end
            definition === nothing && _pr_fail("unresolved group indices")
            expr = group.prefix[definition].args[2]
            parts = _rw_read_parts(expr)
            read = parts === nothing ? expr : parts[2]
            _index_run_read(read) && read.args[3] === group.index ||
                _pr_fail("unresolved group index projection")
            rows = get(prepared, read.args[2], nothing)
            rows in params || _pr_fail("unknown group index operand")
        end
        batched = name in Base.values(prepared)
        push!(projections, _PRGroupProjection{declaration}(findfirst(==(parent), params),
            findfirst(==(rows), params), rows !== indices, batched, !(parent in Base.values(prepared))))
        push!(sources, parent)
        reads[rhs] = nothing
        shadow = Symbol(parent, :_shadow)
        if batched
            bindings[s] = :($(Symbol(name, :_shadow)) =
                $(GlobalRef(@__MODULE__, :_pr_group_view))($shadow, $indices))
        else
            # All hours in this group contributed to the same scalar.
            scatters[rhs] = :($(GlobalRef(@__MODULE__, :_pr_shadow_add!))(
                $shadow, $indices, $(_pr_bar(name))))
        end
    end
    sources, bindings, scatters, Tuple(unique(projections))
end

_pr_group_projections_cover(projections, types) =
    all(p -> _pr_group_projection_covers(p, types), projections)

function _pr_group_projection_covers(p::_PRGroupProjection{D}, types) where {D}
    source = types[p.parent]
    source <: Union{Vector,SubArray{<:Any,1,<:Vector}} || return false
    indices = p.grouped ? eltype(types[p.rows]) : types[p.rows]
    # A typed assignment may convert the indexed result. An integer index
    # returns the element. A vector of indices can return a vector or a view.
    if D !== Any
        result = indices === Int ? eltype(source) : AbstractVector{eltype(source)}
        result <: D || return false
    end
    if indices === Int
        return p.batched ? source <: Vector{<:Vector{<:Union{Float32,Float64}}} :
                           eltype(source) <: AbstractFloat
    end
    p.batched && indices <: Union{Vector{Int},UnitRange{Int},Base.OneTo{Int}}
end

# Expose the two leaves of a lazy scalar-plus-vector broadcast at the rule
# boundary. The group still loads its entry once. Its ordinary scalar graph
# now returns the shared cotangent and scatters the indexed cotangent.
struct _PRGroupBroadcastRule{Positions,F,R} <: Function
    original::F
    expanded::R
end

const _PRScalarPlusVector = _CellBroadcast{T,1,B} where
    {T<:Union{Float32,Float64},B<:Base.Broadcast.Broadcasted{
        Base.Broadcast.DefaultArrayStyle{1},<:Any,typeof(+),Tuple{T,Vector{T}}}}

@inline _pr_broadcast_type(::Val{T}) where {T} = T

function _pr_expand_group_reads(ast, positions)
    params = ast.args[1].args
    shared = Dict(params[i] => gensym(:broadcast_scalar) for i in positions)
    original_types = Dict(params[i] => gensym(:broadcast_type) for i in positions)
    function expand(ex)
        ex isa Symbol && haskey(shared, ex) && _pr_fail(
            "a group broadcast has a use other than a scalar index")
        ex isa Expr && ex.head !== :quote || return ex
        if Meta.isexpr(ex, :call, 2) &&
                ex.args[1] in (typeof, GlobalRef(Base, :typeof), GlobalRef(Core, :typeof)) &&
                haskey(original_types, ex.args[2])
            return Expr(:call, GlobalRef(@__MODULE__, :_pr_broadcast_type), original_types[ex.args[2]])
        end
        if ex.head === :block
            stmts = Any[]
            for stmt in ex.args
                if Meta.isexpr(stmt, :(=), 2) && Meta.isexpr(stmt.args[2], :ref, 2)
                    rhs = stmt.args[2]
                    parent, index = rhs.args
                    if haskey(shared, parent) && index isa Symbol
                        entry = gensym(:group_entry)
                        push!(stmts, :($entry = $parent[$index]),
                              Expr(:(=), stmt.args[1], Expr(:call, GlobalRef(Base, :+), shared[parent], entry)))
                        continue
                    end
                end
                push!(stmts, expand(stmt))
            end
            return Expr(:block, stmts...)
        end
        Expr(ex.head, map(expand, ex.args)...)
    end
    scalars = [shared[params[i]] for i in positions]
    types = [original_types[params[i]] for i in positions]
    Expr(:function, Expr(:tuple, params..., types..., scalars...), expand(ast.args[2])), scalars
end

function _pr_group_broadcast_expr(positions, projections, types, dispatch)
    sited = _pr_sited(types)
    count = length(types) - sited
    extra = dispatch === :_pr_rule_applies ? (:check,) : ()
    original = :($dispatch(rule.original, args, $(extra...)))
    all(i -> types[i] <: _PRScalarPlusVector, positions) || return original
    for p in projections
        p.parent in positions || continue
        indices = p.grouped ? eltype(types[p.rows]) : types[p.rows]
        !p.batched && indices === Int || return original
    end
    leaves = Any[:(getfield(args, $i)) for i in 1:count]
    checks = Any[]
    for i in positions
        cells = leaves[i]
        vector = :(getfield($cells.broadcast.args, 2))
        push!(checks, :(axes($cells) == axes($vector)))
        leaves[i] = vector
        push!(leaves, :(Val(typeof($cells))))
    end
    append!(leaves, (:(getfield(getfield(args, $i).broadcast.args, 1)) for i in positions))
    # The expanded call keeps its site record, or the original call is made.
    if sited
        length(leaves) < _PLATE_CALL_MAX_ARGUMENTS || return original
        push!(leaves, :(getfield(args, $(count + 1))))
    end
    condition = foldl((a, b) -> :($a && $b), checks)
    :($condition ? $dispatch(rule.expanded, ($(leaves...),), $(extra...)) : $original)
end
