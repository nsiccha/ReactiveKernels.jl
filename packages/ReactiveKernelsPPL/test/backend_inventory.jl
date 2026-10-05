import Reactant

# Constants and redundant vector identity broadcasts may specialize by shape.
# Retain the complete inventory for diagnostics; the body inventory still
# counts arithmetic, indexing, reductions, batching and control flow.
function _ppl_backend_operation_inventory(hlo)
    all_ops, body_ops = Dict{String,Int}(), Dict{String,Int}()
    for line in split(hlo, '\n')
        m = match(r"""^\s*(?:%[^=]+ = )?"?((?:stablehlo|enzyme)\.[a-z_]+)""", line)
        m === nothing && continue
        op = m.captures[1]
        all_ops[op] = get(all_ops, op, 0) + 1
        identity = op == "stablehlo.broadcast_in_dim" &&
            occursin(r"dims = \[0\] : \(tensor<([^>]+)>\) -> tensor<\1>", line)
        (op == "stablehlo.constant" || identity) && continue
        body_ops[op] = get(body_ops, op, 0) + 1
    end
    return (; all_ops, body_ops)
end

# Parse every dialect rather than matching only printed StableHLO spellings.
# Keep batch callees separately so retained calls must still reach fixed cell
# bodies; complete inventories remain available for optimizer diagnostics.
function _ppl_mlir_structure(module_text)
    operations = String[]
    functions = Dict{String,Vector{String}}()
    callees = String[]
    function walk(op, names)
        name = Reactant.MLIR.IR.name(op)
        push!(names, name)
        if name == "enzyme.batch"
            push!(callees, Reactant.MLIR.IR.rootref(
                Reactant.MLIR.IR.getattr(op, "fn")))
        end
        children = name == "func.func" ? String[] : names
        for region in op, block in region, child in block
            walk(child, children)
        end
        if name == "func.func"
            functions[String(Reactant.MLIR.IR.getattr(op, "sym_name"))] = children
            append!(names, children)
        end
    end
    Reactant.MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
        mod = parse(Reactant.MLIR.IR.Module, String(module_text); context=ctx)
        try
            walk(Reactant.MLIR.IR.Operation(mod), operations)
        finally
            Reactant.MLIR.IR.dispose(mod)
        end
    end
    return (; operations, cells=[functions[callee] for callee in callees])
end
