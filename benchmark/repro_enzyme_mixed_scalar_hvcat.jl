# Standalone CPU reproducer: only Enzyme is required.
#
# Native Enzyme reverse mode on Julia 1.12 cannot compile Base's `hvcat` of
# block literals that mix scalars with arrays, such as `[A v; 0 0 1]`:
#   IllegalTypeAnalysisException: Enzyme compilation failed due to illegal
#   type analysis. ... Failure within method: hvncat_fill!(...)
# Julia 1.12 builds these literals through `hvncat_fill!`.  Compilation fails
# with scalar operands of either type, and even when the literal sits on a
# branch that never runs, so a function that merely falls back to Base's
# `hvcat` for layouts it rejects fails the same way.  Scalar-only and
# array-only literals differentiate.  Julia 1.10 differentiates every case.
# ReactiveKernels' native kernel bodies build these literals in
# `_native_hvcat` (`src/core.jl`) without calling Base's methods, so `@kernel`
# bodies differentiate on both versions (snag `native-bracket-f-09be44ed`);
# the limitation stays with opaque helpers a kernel calls.  Recorded on
# gordito, 2026-10-07, Enzyme 0.13.210.
using Enzyme

const A = [1.0 2.0; 3.0 4.0]

# One write per element of `[s .* A [exp(s), 0.0]; 0.0 0.0 1.0]`.
function per_operand(s)
    out = Matrix{Float64}(undef, 3, 3)
    B = s .* A
    for j in 1:2, i in 1:2
        out[i, j] = B[i, j]
    end
    out[1, 3] = exp(s)
    out[2, 3] = 0.0
    out[3, 1] = 0.0
    out[3, 2] = 0.0
    out[3, 3] = 1.0
    out
end
# The literal on a branch the calls below never take; a `Ref` keeps the
# compiler from removing it.
const TAKE = Ref(false)
unused_literal(s) = TAKE[] ? [s .* A [exp(s), 0.0]; 0 0 1] : per_operand(s)

cases = (
    ("[A v; 0 0 1] (Int scalars)", s -> sum([s .* A [exp(s), 0.0]; 0 0 1])),
    ("[A v; 0.0 0.0 1.0]", s -> sum([s .* A [exp(s), 0.0]; 0.0 0.0 1.0])),
    ("same literal on an unused branch", s -> sum(unused_literal(s))),
    ("control: [s 0; s 2s] scalars only", s -> sum([s 0; s 2s])),
    ("control: [A v; A v] active arrays", s -> sum([s .* A [exp(s), 0.0]; s .* A [s, 1.0]])),
    ("control: per-operand writes", s -> sum(per_operand(s))),
)

central(f, s; h = 1e-6) = (f(s + h) - f(s - h)) / 2h
println("Enzyme ", pkgversion(Enzyme), ", Julia ", VERSION)
for (label, f) in cases
    result = try
        g = autodiff(Reverse, f, Active, Active(0.3))[1][1]
        isapprox(g, central(f, 0.3); rtol = 1e-6) ? "OK  d/ds = $g" :
            "WRONG d/ds = $g"
    catch err
        "$(nameof(typeof(err)))"
    end
    println(rpad(label, 36), result)
end
