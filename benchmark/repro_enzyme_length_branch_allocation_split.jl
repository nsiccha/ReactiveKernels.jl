# Standalone CPU reproducer: only Enzyme is required.
#
# On Julia 1.10, native Enzyme reverse mode (static activity analysis, the
# default) fails when a function branches on an array's length and then
# computes Base's `s * t` (or `t * s`, `-t`, `a * s * t`): a broadcast into a
# fresh array of that length, which Enzyme materializes with its own loop
# (`OverrideBCMaterialize`): allocate the output, then branch on its length.
#   EnzymeRuntimeActivityError: Detected potential need for runtime activity.
#   Constant memory is stored (or returned) to a differentiable variable
#   Mismatched activity for: phi ... const val: ... @ijl_alloc_array_1d(..., 0)
# The optimizer threads that loop guard through the earlier branch and
# splits the allocation into an empty arm. That arm's array is never written,
# so static activity analysis calls it constant, and it meets the written
# array in one value. It fails only when `t` is empty; every value is valid
# and the derivative is defined. The earlier branch can be authored
# (`isempty(t)`, a loop over `t`) or come from how `t` was built: a copy whose
# dimension check compares `t`'s length with the broadcast's (`copyto!` of a
# broadcast, as ReactiveKernels' native dotted calls materialize) fails too.
# Which spellings the optimizer splits depends on inlining: the last row of
# each block shows a checked copy of `s .* t` passing here, yet the same copy
# fails inside larger plate cells.
# `set_runtime_activity(Reverse)` differentiates every case, Julia 1.12 with
# the same Enzyme passes every case, and so does an Enzyme whose
# mixed-activity handler gives a fresh allocation that no active data reaches
# a zero shadow. The same split underlies
# `repro_enzyme_branch_allocation_phi.jl`.
# Recorded on gordito, Enzyme 0.13.210, Enzyme_jll 0.0.301, Julia 1.10.12 and
# 1.12.7, 2026-10-07.
using Enzyme

# `copyto!` is Enzyme's broadcast copy here; it checks the fresh output's
# dimensions against the broadcast's before its loop.
@inline function copied(bc)
    src = Base.Broadcast.preprocess(nothing, Base.Broadcast.instantiate(bc))
    copyto!(similar(src, Float64), src)
end

copied_then_scale(x, idx, r, s) =
    sum(s * copied(Base.broadcasted(/, x[idx], r)))
isempty_then_scale(x, idx, r, s) =
    (t = x[idx] ./ r; (isempty(t) ? 0.0 : t[1]) + sum(s * t))
loop_then_scale(x, idx, r, s) =
    (t = x[idx] ./ r; a = 0.0; for v in t; a += v; end; a + sum(s * t))
copied_then_negate(x, idx, r, s) =
    sum(-copied(Base.broadcasted(/, x[idx], r)))
copied_then_chain(x, idx, r, s) =
    sum(0.5 * s * copied(Base.broadcasted(/, x[idx], r)))
base_then_scale(x, idx, r, s) = sum(s * (x[idx] ./ r))
copied_then_copied_scale(x, idx, r, s) =
    sum(copied(Base.broadcasted(*, s, copied(Base.broadcasted(/, x[idx], r)))))

# Active data and scales for one group; `idx` selects observations.
objective(f, q, idx) = f([0.5, 1.0, 1.5, 2.0] .* q[1], idx, exp(q[2]), q[3])

function run(label, f, idx; mode = Reverse)
    result = try
        autodiff(mode, objective, Active, Const(f), Duplicated([0.1, 0.5, 0.9], zeros(3)), Const(idx))
        "ok"
    catch err
        "$(nameof(typeof(err)))"
    end
    println(rpad(label, 50), rpad("n = $(length(idx))", 8), result)
end

println("Julia $VERSION; expected ok everywhere, failures are the boundary")
for idx in ([1, 3], Int[])
    run("checked copy, then s * t", copied_then_scale, idx)
    run("authored isempty(t), then s * t", isempty_then_scale, idx)
    run("authored loop over t, then s * t", loop_then_scale, idx)
    run("checked copy, then -t", copied_then_negate, idx)
    run("checked copy, then 0.5 * s * t", copied_then_chain, idx)
    run("checked copy, then s * t, runtime activity", copied_then_scale, idx;
        mode = set_runtime_activity(Reverse))
    run("Base broadcast, then s * t", base_then_scale, idx)
    run("checked copy, then checked copy of s .* t", copied_then_copied_scale, idx)
end
