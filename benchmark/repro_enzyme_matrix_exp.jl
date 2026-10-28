# Backend-only reproducer; run in an environment providing Enzyme.
# Julia 1.10.12 / Enzyme 0.13.206 and 0.13.209 cannot differentiate the
# LAPACK.gebal! call inside dense matrix exp (EnzymeNoDerivativeError).
# https://github.com/EnzymeAD/Enzyme.jl/issues/1222 records this boundary.
# ReactiveKernels' authored native exp calls use its existing owned generated
# rule; this deliberately tests the raw Julia builtin without ReactiveKernels.
using Enzyme, LinearAlgebra

println("Julia ", VERSION, ", Enzyme ", pkgversion(Enzyme))
A = [0.13 -0.27; 0.31 0.08]
println("primal = ", sum(exp(A)))
Enzyme.gradient(Enzyme.Reverse, M -> sum(exp(M)), A)
