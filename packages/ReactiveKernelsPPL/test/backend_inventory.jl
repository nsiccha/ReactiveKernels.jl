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
