# Count emitted syntax at each call site, including the plate program it runs.
# Alternative optimized and fallback loops contribute to size independently.
prepared_structure(x) = (; nodes = 0, depth = 0)
prepared_structure(p::ReactiveKernels._PlateProgram) = prepared_structure(p.ast)
function prepared_structure(ex::Expr)
    nodes, depth = 1, 0
    ex.head in (:quote, :inert) && return (; nodes, depth)
    for arg in ex.args
        child = prepared_structure(arg)
        nodes += child.nodes
        depth = max(depth, child.depth)
    end
    (; nodes, depth = depth + (ex.head in (:for, :while)))
end
