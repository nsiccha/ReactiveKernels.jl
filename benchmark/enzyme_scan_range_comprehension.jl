# Backend-only companion to test_scan_range_argument.jl. No RK code is loaded.
# Julia 1.13.1 / registered Enzyme 0.13.219 reports runtime activity in this
# native comprehension + concatenation reference, including its empty case.
# Keep the expression here so testing the lowering against the full native
# hand loop does not hide that separate backend limitation.
# Run with an environment containing Enzyme:
# julia --project=<env> benchmark/enzyme_scan_range_comprehension.jl
using Enzyme

reference(q, xs, selected) = sum(abs2, vcat(q[2],
    [x > 0 ? sqrt(q[1] * x) : q[1] * x * x for x in xs])[selected])

println((julia = VERSION, enzyme = pkgversion(Enzyme)))
for (xs, selected) in ((Float64[], 1:1), ([-0.2], 2:2),
                       ([-0.2, 0.1, 0.3, -0.4], 2:4),
                       ([-0.2, 0.1, 0.3, -0.4], 3:2))
    q = [0.3, 0.6]
    gradient = zero(q)
    Enzyme.autodiff(Enzyme.Reverse, reference, Enzyme.Active,
                   Enzyme.Duplicated(q, gradient),
                   Enzyme.Const(xs), Enzyme.Const(selected))
    println((; xs, selected, value = reference(q, xs, selected), gradient))
end
