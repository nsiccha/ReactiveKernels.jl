# Standalone CPU reproducer: only Reactant and Enzyme are required.
#
# `Ops.batch` over a cell that computes an Enzyme reverse gradient fails in
# Reactant's MLIR pipeline: the batching pass hands a 2-D permutation to a
# `stablehlo.transpose` whose operand is 1-D inside the batched reverse
# region. The same batch of the cell WITHOUT the gradient lowers, and the
# same gradient inside a `@trace for` over the replicas lowers, so the
# gap is specifically batching an `enzyme.autodiff` region. Recorded on
# strato2, Reactant 0.2.284 / Enzyme 0.13, 2026-09-22:
#   CompilationError: MLIR pass pipeline "all" failed
#   error: TransposeOp operand rank 1 does not match permutation size 2
#   error: 'stablehlo.transpose' op failed to infer returned types
#     (tensor<2xf64>) -> tensor<2x6xf64>, permutation = array<i64: 1, 0>
using Reactant, Enzyme

f(x, c) = sum(abs2, x .- c) + sum(x)^2 / 3
scalar(c0) = Reactant.@allowscalar c0[]
cell_primal(x, c0) = x .* scalar(c0)
cell_gradient(x, c0) = only(Enzyme.gradient(Enzyme.Reverse, xx -> f(xx, scalar(c0)), x))
# Batch axis LEADING: X is (replicas, n), C is (replicas,).
batched(cell, X, C) = only(Reactant.Ops.batch(cell, Reactant.TracedRArray[X, C], Int64[size(X, 1)]))

R = 6
X = Reactant.to_rarray(Matrix(reshape(collect(1.0:(2R)) ./ 7, R, 2)))
C = Reactant.to_rarray(collect(0.1:0.1:(0.1R)))
primal(X, C) = batched(cell_primal, X, C)
@assert Array((Reactant.@compile primal(X, C))(X, C)) ≈ Array(X) .* Array(C)   # batching itself is fine
gradient(X, C) = batched(cell_gradient, X, C)
compiled = Reactant.@compile gradient(X, C)                                     # fails here
expected = hcat([2 .* (Array(X)[i, :] .- Array(C)[i]) .+ 2 * sum(Array(X)[i, :]) / 3 for i in 1:R]...)'
@assert Array(compiled(X, C)) ≈ expected
