# In a sum of group totals the derivative of the outer addition with respect
# to each completed inner total is one. Establish that use from the program
# before seeding the inner sweep, and keep its scope inside the group loop.
function _pr_additive_groups(body, totals, terminal, assignments)
    groups = IdDict{Expr,_PRAdditiveGroup}()
    intermediate = setdiff(totals, terminal)
    isempty(intermediate) && return groups
    reads = _rw_reads!(Dict{Symbol,Int}(), body)
    declared = Dict{Symbol,Any}()
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
    groups
end

@inline _pr_group_view(::Nothing, indices) = nothing
@inline _pr_group_view(shadow, indices) = view(shadow, indices)
@inline _pr_group_view(shadow::_PRMappedShadow, indices) =
    _PRMappedShadow(shadow.f, _pr_group_view(shadow.storage, indices))

# A gathered group operand uses its own indices in both directions. A view
# of the parent's cotangent storage scatters repeated positions on addition.
function _pr_group_shadows!(reads, groups, params, active, prepared, mod, values)
    sources = Set{Symbol}()
    bindings = IdDict{Expr,Expr}()
    projections = Tuple{Int,Int,Bool}[]
    for group in Base.values(groups), s in group.prefix
        Meta.isexpr(s, :(=), 2) || continue
        name, rhs = s.args
        name in Base.values(prepared) && name in active || continue
        source = Meta.isexpr(rhs, :let, 2) ? _rw_peel_let(rhs, x -> x isa Symbol) : rhs
        isgather = Meta.isexpr(source, :call, 3) &&
            _pr_call_function(source.args[1], mod, values) === _cell_array_gather
        isgather || Meta.isexpr(source, :ref, 2) || continue
        parent, indices = source.args[end-1:end]
        parent isa Symbol && parent in params && indices isa Symbol && !(indices in active) ||
            _pr_fail("a group gather needs a parameter and data indices")
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
        push!(projections, (findfirst(==(parent), params), findfirst(==(rows), params), rows !== indices))
        push!(sources, parent)
        reads[rhs] = nothing
        bindings[s] = :($(Symbol(name, :_shadow)) = $(GlobalRef(@__MODULE__, :_pr_group_view))(
            $(Symbol(parent, :_shadow)), $indices))
    end
    sources, bindings, Tuple(unique(projections))
end

function _pr_group_projections_cover(projections, types)
    all(projections) do (parent, rows, grouped)
        source = types[parent]
        source <: Union{Vector,SubArray{<:Any,1,<:Vector}} || return false
        indices = grouped ? eltype(types[rows]) : types[rows]
        indices <: Union{Vector{Int},UnitRange{Int},Base.OneTo{Int}}
    end
end
