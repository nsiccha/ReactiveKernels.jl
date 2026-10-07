# Standalone CPU reproducer: only Enzyme is required.
#
# Native Enzyme reverse mode (static activity analysis, the default) fails
# when a recurrence's carry starts as a constant allocation and a lazy branch
# can keep it, while other paths replace it with active memory:
#   EnzymeRuntimeActivityError: Detected potential need for runtime activity.
#   Constant memory is stored (or returned) to a differentiable variable
#   ... @ijl_alloc_array_1d(..., 2)
# This is the shape ReactiveKernels emits for an outer `scan` whose step runs
# an inner `scan` seeded by the carry and then keeps the carry when the inner
# sequence is empty, `isempty(us) ? previous : inner[end]`. Both scans peel
# their first step and allocate their result once before an emptiness branch.
# The derivative is well defined on every path; only the activity of the
# carried array depends on the data. The same branch differentiates when the
# kept arm is unreachable (`unreachable_keep`: `first` has already proved the
# inner sequence nonempty) and without the branch (`last_output`). Writing the
# seed into the inner result, as `scan(...; include_init = true)` does, and
# reading its last element also differentiates (`seeded_result`), with the
# same value: an empty inner sequence leaves the seed there.
# `set_runtime_activity(Reverse)` differentiates every case.
# Recorded on gordito, Enzyme 0.13.209, Julia 1.10.12, 2026-10-07.
using Enzyme

# An inner scan over the weights, seeded by the carry. Its empty arm returns
# the buffer allocated before the branch, unwritten, as the native lowering
# does.
@inline function inner(carry, x, w)
    indices = eachindex(w)
    outputs = similar(indices, Vector{Float64})
    if !isempty(indices)
        i = first(indices)
        state = carry .+ x * w[i]
        outputs[i] = state
        for i in Iterators.drop(indices, 1)
            state = state .+ x * w[i]
            outputs[i] = state
        end
    end
    outputs
end

# The same inner scan without its empty arm: `first` throws for no weights.
@inline function inner_nonempty(carry, x, w)
    indices = eachindex(w)
    outputs = similar(indices, Vector{Float64})
    i = first(indices)
    state = carry .+ x * w[i]
    outputs[i] = state
    for i in Iterators.drop(indices, 1)
        state = state .+ x * w[i]
        outputs[i] = state
    end
    outputs
end

# `include_init = true`: the seed is the first element, then each output.
@inline function inner_seeded(carry, x, w)
    indices = eachindex(w)
    outputs = similar(indices, Vector{Float64}, length(indices) + 1)
    outputs[1] = carry
    state = carry
    for (position, i) in enumerate(indices)
        state = state .+ x * w[i]
        outputs[position + 1] = state
    end
    outputs
end

@inline keep_carry(w, previous, outputs) = isempty(w) ? previous : outputs[end]
@inline last_element(w, previous, outputs) = outputs[end]

# The outer scan: a constant seed, its first step peeled.
@inline function outer(scan_inner, select, xs, w)
    seed = zeros(2)
    totals = similar(xs)
    indices = eachindex(xs)
    if !isempty(indices)
        j = first(indices)
        carry = select(w, seed, scan_inner(seed, xs[j], w))
        totals[j] = sum(carry)
        for j in Iterators.drop(indices, 1)
            carry = select(w, carry, scan_inner(carry, xs[j], w))
            totals[j] = sum(carry)
        end
    end
    sum(totals)
end

kept_seed(xs, w) = outer(inner, keep_carry, xs, w)
unreachable_keep(xs, w) = outer(inner_nonempty, keep_carry, xs, w)
last_output(xs, w) = outer(inner, last_element, xs, w)
seeded_result(xs, w) = outer(inner_seeded, last_element, xs, w)

function run(label, f, xs, w; mode = Reverse)
    result = try
        "gradient = $(Enzyme.gradient(mode, f, Const(xs), w)[2])"
    catch err
        "$(nameof(typeof(err)))"
    end
    println(rpad(label, 46), result)
end

xs, w = [0.3, -0.2, 0.5], [1.0, 2.0, 0.5]
println("expected value 7.0 and gradient [2.0, 2.0, 2.0] for every case")
@assert all(f -> f(xs, w) ≈ 7.0,
            (kept_seed, unreachable_keep, last_output, seeded_result))
run("branch may keep the constant seed", kept_seed, xs, w)
run("branch whose keep arm is unreachable", unreachable_keep, xs, w)
run("no branch, last inner output", last_output, xs, w)
run("seed written into the inner result", seeded_result, xs, w)
run("branch may keep the seed, runtime activity", kept_seed, xs, w;
    mode = set_runtime_activity(Reverse))
