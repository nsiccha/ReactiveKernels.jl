# Construction code fills a table with a loop, not with `Dict(itr)`,
# `append!(out, itr)`, `collect(T, itr)` or `Set(itr)`, when `itr` is a
# generator, `zip`, `enumerate` or `Iterators.flatten` over a value inference
# cannot type. Those calls size their result with `IteratorSize(itr)`, which
# then reaches Base's `IteratorSize(::Type)` fallback, so any package that
# defines `IteratorSize` for its own iterator (LibGit2's `GitBranchIter`,
# loaded with Pkg or Enzyme) invalidates the caller, and a fresh process
# recompiles lowering and `build_kernel` from RK's and RKPPL's images (snag
# downstream-workl-75f8cba4). A loop asks no `IteratorSize`. Over concretely
# typed collections the Base calls are fine.

"""`D(itr)` for a dictionary type `D`: the pairs of `itr` in order, a later
key replacing an earlier one."""
function _dict_from(::Type{D}, itr) where {D<:AbstractDict}
    d = D()
    for (k, v) in itr
        d[k] = v
    end
    return d
end

"""`append!(out, itr)`, one `push!` per element (also `union!` of a set)."""
function _push_each!(out, itr)
    for x in itr
        push!(out, x)
    end
    return out
end

"""The type of each input of `p` by canonical id, `types` in `p.have` order."""
function _have_types(p::Plan, types)
    known = Dict{ValueId,Any}()
    for (v, T) in zip(p.have, types)
        known[canon_id(p.graph, v.id)] = T
    end
    return known
end
