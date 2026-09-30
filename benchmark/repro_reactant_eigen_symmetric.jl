# Standalone CPU reproducer: only Reactant is required.
#
# Eigendecomposition of a traced matrix does not lower — at several stacked
# layers. `eigen(Symmetric(A))` for a traced `A` dies during tracing in
# `LinearAlgebra.isdiag(::Symmetric)` → Reactant `isbanded`/`_istril`, which
# calls `overloaded_triu` on the `UpperTriangular` wrapper: Reactant's
# `src/stdlibs/LinearAlgebra.jl:366` defines that method for `TracedRArray{T,
# 2}` only. The same missing method already has an upstream issue via the
# symmetric-solve path (EnzymeAD/Reactant.jl#3369, open). Behind it the gap
# is deeper: Reactant ships no traced `eigen`/`eigvals` primal at all — the
# nonsymmetric `eigen(A)` dies branching on a traced `Bool` in the generic
# Schur path, and `eigvals(Symmetric(A))` has no `eigvals!` method for
# traced wrappers. Recorded failures (strato2, Reactant 0.2.289, Julia
# 1.10.11, 2026-09-30):
#   eigen(Symmetric(A)): MethodError: no method matching
#     overloaded_triu(::UpperTriangular{TracedRNumber{Float64},
#     TracedRArray{Float64, 2}}, ::Int64)
#   eigen(A): TypeError: non-boolean (TracedRNumber{Bool}) used in boolean context
#   eigvals(Symmetric(A)): MethodError: no method matching
#     eigvals!(::Symmetric{TracedRNumber{Float64}, TracedRArray{Float64, 2}})
# This is the shape of the posteriordb `kronecker_gp` example's exact
# Kronecker-eigenspace marginal likelihood (`packages/
# ReactiveKernelsPPLExamples/src/kronecker_gp.jl`); RK keeps the authored
# `eigen(Symmetric(·))` calls per docs/src/constraints.md. Tracked on
# nsiccha/ReactiveKernels.jl (relay upstream from there).
using Reactant, LinearAlgebra

S2 = [2.0 0.5; 0.5 1.0]
rS2 = Reactant.to_rarray(S2)
compiled = Reactant.@compile eigen(Symmetric(rS2))   # fails here
@assert Array(compiled(rS2).values) ≈ eigvals(Symmetric(S2))
