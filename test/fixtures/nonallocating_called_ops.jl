# The operations a non-allocating program calls: plain calls `__ops__[i](...)`
# and cache steps `__cache_apply__(__caches__[i], __ops__[i], ...)`. The
# program's operation table is the native lowering's, so a decomposed source
# keeps its slot there without being called.
function called_ops(k)
    slots = Set{Int}()
    function walk(node)
        node isa Expr || return
        if node.head === :call
            callee = node.args[1] === :__cache_apply__ ? node.args[3] : node.args[1]
            slot = ReactiveKernels._operation_slot(callee)
            slot === nothing || push!(slots, slot)
        end
        foreach(walk, node.args)
    end
    walk(code_expr(k))
    [k.ops[slot] for slot in sort!(collect(slots))]
end
