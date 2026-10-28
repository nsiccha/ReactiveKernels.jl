# Standalone CPU reproducer: only Enzyme is required.
#
# Native Enzyme reverse mode (static activity analysis, the default) fails
# when Base's dense concatenation methods meet constant and active arrays of
# one element type:
#   EnzymeRuntimeActivityError: Detected potential need for runtime activity.
#   Constant memory is stored (or returned) to a differentiable variable
# The activity of every argument is fixed (`Const` data, `Active` scalar), so
# static analysis should succeed.  `vcat(::Vector{T}...)`,
# `hcat(::Vector{T}...)` and the `typed_hcat`/`typed_vcat`/`typed_hvcat`
# loops behind `hcat`, `vcat` and `hvcat` of `Vector{T}`/`Matrix{T}` read
# each operand from their vararg tuple at a runtime index, which joins the
# constant and active operands' activities; `stack` fails the same way.
# The fold `reduce(vcat, A; init)` reaches `vcat(acc, a)` once per element,
# so constant per-element arrays fail even when only the fold's result meets
# an active value; Base's `reduce(vcat, A)` without `init` allocates once and
# differentiates.  Operands of different element types, all-active operands,
# and a plain sequence of one inlined copy per operand differentiate.  The
# last is how ReactiveKernels' native kernel bodies concatenate
# (`_native_hcat`, `_native_vcat` and `_native_hvcat` in `src/core.jl`).
# Base's generic `cat(...; dims)` differentiates on Julia 1.10.12 / Enzyme
# 0.13.209 but fails the same way on Julia 1.12.7 / Enzyme 0.13.210.  An
# `@rkppl` design matrix `hcat(ones(n), x, exp.(a .* x))` hits the Base
# methods (snag `rkppl-hcat-desig-3016966b`).  Recorded on gordito,
# 2026-10-05, with both version pairs above.  The local Enzyme `dogfood` line
# overrides Base's concatenation inside Enzyme: at `82c07be` every case here
# differentiates on Julia 1.10.12, while Enzyme 0.13.210 fails all eight
# (gordito, 2026-10-08, snag `inner-plate-pe-d-5575dbdb`).  No released
# Enzyme carries that repair.
using Enzyme

const x = [-1.0, -0.3, 0.2, 0.8, 1.5]
const C = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0; 9.0 10.0]
const xi = [1, 2, 3, 4, 5]
weigh(M) = sum(M .* reshape(collect(1.0:length(M)), size(M)))
live(a) = exp.(a .* x)

# One inlined copy per operand, each tied to its tuple position.
@inline copy_operands!(out, offset, ::Tuple{}) = out
@inline function copy_operands!(out, offset, arrays::Tuple)
    a = first(arrays)
    Base.unsafe_copyto!(out, offset, a, 1, length(a))
    copy_operands!(out, offset + length(a), Base.tail(arrays))
end
per_operand_hcat(vs::Vector{T}...) where {T} =
    copy_operands!(Matrix{T}(undef, length(first(vs)), length(vs)), 1, vs)

# Constant per-element arrays, as bound ragged data or a plate result hoisted
# at preparation are.
const R = [x[1:2], x[3:3], Float64[], x[4:5]]

cases = (
    ("vcat(Vector c, Vector a)", a -> weigh(vcat(x, live(a)))),
    ("hcat(Vector c, Vector a)", a -> weigh(hcat(x, live(a)))),
    ("[c v] bracket syntax", a -> weigh([x live(a)])),
    ("hcat(Matrix c, Vector a)", a -> weigh(hcat(C, live(a)))),
    ("vcat(Matrix c, Matrix a)", a -> weigh(vcat(C, a .* C))),
    ("hvcat [c v; v c]", a -> weigh([x live(a); live(a) x])),
    ("stack((c, a))", a -> weigh(stack((x, live(a))))),
    ("reduce(vcat, c; init) .* a", a -> weigh(reduce(vcat, R; init = Float64[]) .* live(a))),
    ("control: cat(c, a; dims=2)", a -> weigh(cat(x, live(a); dims = 2))),
    ("control: hcat(Int c, Float a)", a -> weigh(hcat(xi, live(a)))),
    ("control: hcat all active", a -> weigh(hcat(a .* x, live(a)))),
    ("control: per-operand copies", a -> weigh(per_operand_hcat(x, live(a)))),
    ("control: reduce(vcat, c) .* a", a -> weigh(reduce(vcat, R) .* live(a))),
)

central(f, a; h = 1e-6) = (f(a + h) - f(a - h)) / 2h
println("Enzyme ", pkgversion(Enzyme), ", Julia ", VERSION)
for (label, f) in cases
    snapshot = (copy(x), copy(C))
    result = try
        g = autodiff(Reverse, f, Active, Active(0.1))[1][1]
        isapprox(g, central(f, 0.1); rtol = 1e-6) ? "OK  d/da = $g" :
            "WRONG d/da = $g"
    catch err
        "$(nameof(typeof(err)))"
    end
    intact = (x, C) == snapshot ? "data intact" : "DATA MUTATED"
    println(rpad(label, 34), rpad(result, 44), intact)
end
