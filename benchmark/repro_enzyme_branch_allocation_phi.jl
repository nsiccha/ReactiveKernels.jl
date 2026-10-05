# Standalone CPU reproducer: only Enzyme is required.
#
# Native Enzyme reverse mode (static activity analysis, the default) fails
# when two buffers allocated in the arms of a branch meet in one value, the
# empty arm's buffer is never written with active data, and Base `sum`
# reads the merged value:
#   EnzymeRuntimeActivityError: Detected potential need for runtime activity.
#   Constant memory is stored (or returned) to a differentiable variable
#   Mismatched activity for: phi ... const val: ... @ijl_alloc_array_1d(..., 0)
# Every value is valid and the derivative is zero for the empty input. This
# is the shape of a recurrence that returns an empty vector of its output
# type for an empty sequence and otherwise fills a buffer typed by its first
# output. A single allocation dominating the branch differentiates, as does
# a manual sum loop. One allocation after a branch that peels the first step
# to choose the element type still fails (`single_after_typing`): in the
# generated kernel of that shape the merged value was again a phi with an
# `ijl_alloc_array_1d(..., 0)` on the empty path, because the optimizer
# split the allocation back into the two arms.
# `set_runtime_activity(Reverse)` differentiates every case. ReactiveKernels'
# native scan lowering therefore allocates its buffers once, before the
# emptiness branch, whenever the step's inferred output type is concrete.
# Recorded on gordito, Enzyme 0.13.209, Julia 1.10.12, 2026-10-05.
using Enzyme

function fill_steps!(u, xs, g)
    c = 0.0
    for i in eachindex(xs)
        c = c + xs[i] * g
        u[i] = c
    end
    u
end

function branch_allocations(xs, g)
    if isempty(xs)
        u = similar(xs, Float64)
    else
        u = fill_steps!(similar(xs, Float64), xs, g)
    end
    sum(u)
end

function branch_allocations_loop_sum(xs, g)
    if isempty(xs)
        u = similar(xs, Float64)
    else
        u = fill_steps!(similar(xs, Float64), xs, g)
    end
    s = 0.0
    for v in u
        s += v
    end
    s
end

# Both single-allocation variants peel the first step, as a recurrence does
# to type its buffer by the first output.
function single_after_typing(xs, g)
    idx = eachindex(xs)
    if isempty(idx)
        T = Base.promote_op(*, eltype(xs), typeof(g))
    else
        i1 = first(idx)
        c = xs[i1] * g
        T = typeof(c)
    end
    u = similar(xs, T)
    if !isempty(idx)
        u[i1] = c
        for i in Iterators.drop(idx, 1)
            c = c + xs[i] * g
            u[i] = c
        end
    end
    sum(u)
end

function single_before_branch(xs, g)
    idx = eachindex(xs)
    u = similar(xs, Base.promote_op(*, eltype(xs), typeof(g)))
    if !isempty(idx)
        i1 = first(idx)
        c = xs[i1] * g
        u[i1] = c
        for i in Iterators.drop(idx, 1)
            c = c + xs[i] * g
            u[i] = c
        end
    end
    sum(u)
end

function run(label, f, xs; mode = Reverse)
    result = try
        "d/dg = $(autodiff(mode, f, Active, Const(xs), Active(0.3))[1][2])"
    catch err
        "$(nameof(typeof(err)))"
    end
    println(rpad(label, 44), rpad("n = $(length(xs))", 8), result)
end

println("expected d/dg = 0.0 at n = 0 and 2.0 at n = 2")
for xs in (Float64[], [0.5, 1.0])
    run("allocations in both arms, Base sum", branch_allocations, xs)
    run("allocations in both arms, loop sum", branch_allocations_loop_sum, xs)
    run("one allocation after a typing branch", single_after_typing, xs)
    run("one allocation before the branch", single_before_branch, xs)
    run("allocations in both arms, runtime activity", branch_allocations, xs;
        mode = set_runtime_activity(Reverse))
end
