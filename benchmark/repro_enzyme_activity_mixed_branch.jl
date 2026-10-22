# Standalone CPU reproducer: only Enzyme is required.
#
# Native Enzyme reverse mode (static activity analysis, the default) fails
# when a lazy branch returns a fresh constant array in one arm and active
# memory in the other, and the constant arm is taken:
#   EnzymeRuntimeActivityError: Detected potential need for runtime activity.
#   Constant memory is stored (or returned) to a differentiable variable
# The shape is `!isempty(v) ? v[idx] : zeros(length(idx))` with `v` computed
# from the active input, as in a per-group read of a plate output where some
# groups produce no values. Every value is valid, and the derivative through
# the constant arm is zero. The gradient is fine whenever the active arm is
# taken, and the branch-free read differentiates. With the branch inside a
# `@noinline` helper this reproducer differentiates. This is the shape
# ReactiveKernels keeps when a kernel authors such a branch (the ternary is
# one operation and `zeros(...)` stays in its arm, with bound data or not),
# and the scan-carry form of the same boundary is
# `repro_enzyme_branch_kept_seed_carry.jl`.
# `set_runtime_activity(Reverse)` differentiates every case. So does the
# Enzyme dogfood line from `d2892e7`, whose mixed-activity handler gives a
# fresh allocation that no active data reaches a zero shadow.
# Recorded on gordito, Julia 1.10.12, 2026-10-22: Enzyme 0.13.216 fails the
# inline branch; dogfood `23b72c9` passes every case.
using Enzyme

function produce(x, n)
    v = Vector{Float64}(undef, n)
    for t in 1:n
        v[t] = x[1] * t + x[2]
    end
    v
end
squares(v) = (s = 0.0; for a in v; s += a * a; end; s)

# The branch as authored.
function branch(x, idx, n)
    v = produce(x, n)
    picked = !isempty(v) ? v[idx] : zeros(length(idx))
    squares(picked)
end
# The same branch inside a non-inlined helper.
@noinline pick(v, idx) = !isempty(v) ? v[idx] : zeros(length(idx))
branch_helper(x, idx, n) = squares(pick(produce(x, n), idx))
# No constant arm, valid only when `v` is long enough.
branch_free(x, idx, n) = squares(produce(x, n)[idx])

expected(x, idx, n) = n == 0 ? [0.0, 0.0] :
    [sum(2 * (x[1] * t + x[2]) * t for t in idx), sum(2 * (x[1] * t + x[2]) for t in idx)]

function run(label, f, x, idx, n; mode = Reverse)
    result = try
        g = Enzyme.gradient(mode, f, x, Const(idx), Const(n))[1]
        g ≈ expected(x, idx, n) ? "gradient = $g" : "WRONG gradient $g"
    catch err
        "$(nameof(typeof(err)))"
    end
    println(rpad(label, 46), result)
end

x = [0.3, -0.2]
println("Enzyme ", pkgversion(Enzyme), ", Julia ", VERSION)
for (label, idx, n) in (("active arm taken", [1, 3, 2], 3),
                        ("constant arm taken", [1, 2], 0))
    println("-- ", label, ": expected gradient ", expected(x, idx, n))
    run("branch", branch, x, idx, n)
    run("branch in a non-inlined helper", branch_helper, x, idx, n)
    n > 0 && run("branch-free read", branch_free, x, idx, n)
    run("branch, runtime activity", branch, x, idx, n;
        mode = set_runtime_activity(Reverse))
end
