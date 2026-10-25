# Standalone CPU reproducer: only Enzyme is required.
#
# Native Enzyme forward mode over forward mode (the second directional
# derivative D²f(u)[v, v]) of a function that multiplies a constant matrix by
# a VIEW of the active vector, `sum(abs2, A * view(u, 2:4))`, returns NaN on
# some calls on Julia 1.12 and 1.13. Identical calls in one process disagree,
# and the NaN count varies between processes. The single forward pass is
# exact on every call, and so is the same nested call with a slice
# (`A * u[2:4]`) or on Julia 1.10.
#
# ReactiveKernels' third-order directional derivatives (`prepare_ad_third`,
# reverse over forward over forward) inherit this: an RKPPL program reads its
# coefficients as views of the unconstrained vector, so a design-matrix
# linear predictor meets this shape (`packages/ReactiveKernelsPPL/test/
# test_query_derivatives.jl`). Enzyme snag `nested-forward-o-9434176c`.
# Recorded on gordito, Enzyme 0.13.216, Enzyme_jll 0.0.303, 300 calls each,
# 2026-10-25:
#   Julia 1.10.12: view 0 NaN, slice 0 NaN
#   Julia 1.12.7:  view 14 NaN, slice 0 NaN
#   Julia 1.13.1:  view 19 NaN, slice 0 NaN
using Enzyme

const A = hcat(ones(20), [0.3 * sin(i) for i in 1:20],
               [0.2 * cos(2i) for i in 1:20])

blas_view(u, A) = sum(abs2, A * view(u, 2:4))
blas_copy(u, A) = sum(abs2, A * u[2:4])

directional(u, v, f, A) =
    only(autodiff(Forward, Const(f), Duplicated(u, v), Const(A)))
second_directional(u, v, f, A) =
    only(autodiff(Forward, Const(directional), Duplicated(u, v), Const(v),
                  Const(f), Const(A)))

u = [0.15 * cos(i + 7) for i in 1:8]
v = [sin(i) for i in 1:8]
for (label, f) in (("A * view(u, 2:4)", blas_view), ("A * u[2:4]", blas_copy))
    first_order = [directional(u, v, f, A) for _ in 1:300]
    second_order = [second_directional(u, v, f, A) for _ in 1:300]
    println(rpad(label, 18), "Df[v] distinct ", length(unique(first_order)),
            "; D²f[v, v] NaN in ", count(isnan, second_order), " of 300, values ",
            unique(filter(!isnan, second_order)))
end
