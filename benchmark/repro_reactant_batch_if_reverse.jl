# Standalone CPU reproducer: only Reactant and Enzyme are required.
#
# Enzyme reverse through a lazy `@trace if` inside an `Ops.batch` cell works
# for a small batch (4 lanes, which the batching pass unrolls per lane) and
# fails for a larger one (6 lanes, which it realizes as a loop) with
#   error: had set op which was not a direct descendant
# in the MLIR pass pipeline. The primal compiles at every size. This is the
# shape of every per-observation lazy branch of a plated likelihood
# (`plate(...) do` cells with `?:` guards) under Reactant; RK keeps the
# authored branch lazy per docs/src/constraints.md, so reverse through such
# plates is unavailable until this lowers upstream. Recorded on strato2,
# Reactant 0.2.284 / Enzyme 0.13, 2026-09-22.
using Reactant, Enzyme

function cell(x0)
    x = Reactant.@allowscalar x0[]
    y = zero(x)
    Reactant.@trace if x > 0
        y = log(x)
    end
    y
end
batched(v) = only(Reactant.Ops.batch(cell, [v], Int64[length(v)]))
loss(v) = sum(batched(v))
gradient(v) = only(Enzyme.gradient(Enzyme.Reverse, loss, v))

small = Reactant.to_rarray([2.0, -1.0, 0.5, -3.0])
large = Reactant.to_rarray([2.0, -1.0, 0.5, -3.0, 1.5, -0.5])
@assert Float64((Reactant.@compile loss(small))(small)) ≈ log(2.0) + log(0.5)
@assert Float64((Reactant.@compile loss(large))(large)) ≈ log(2.0) + log(0.5) + log(1.5)
@assert Array((Reactant.@compile gradient(small))(small)) == [0.5, 0.0, 2.0, 0.0]   # 4 lanes: fine
compiled_large = Reactant.@compile gradient(large)                                   # 6 lanes: fails here
@assert Array(compiled_large(large)) == [0.5, 0.0, 2.0, 0.0, 1 / 1.5, 0.0]
