# The operations a non-allocating program calls: plain calls
# `(__callables__[i])(...)` and cache steps
# `__cache_apply__(__caches__[i], __callables__[i], ...)`. The program's
# callables table is the native lowering's, so a decomposed source keeps its
# position there without being called.
function called_ops(k)
    slots = Set{Int}()
    function walk(node)
        node isa Expr || return
        if node.head === :call
            callee = node.args[1] === :__cache_apply__ ? node.args[3] : node.args[1]
            slot = ReactiveKernels._callable_slot(callee)
            slot === nothing || push!(slots, slot)
        end
        foreach(walk, node.args)
    end
    walk(code_expr(k))
    [k.callables[slot] for slot in sort!(collect(slots))]
end

# A source recipe's table entry: its record, or the metadata around one (a
# branch, a split, a generator sum). The recipe's operation wrapper stays with
# the plan.
source_entry(op) = op isa Union{ReactiveKernels._KernelSource, ReactiveKernels._KernelBranch,
                                ReactiveKernels._KernelSplit, ReactiveKernels._KernelReduction}
