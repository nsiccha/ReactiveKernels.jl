# Standalone CPU reproducer: only Enzyme is required.
#
# Native Enzyme reverse mode raises `OutOfMemoryError()` for a loop nested in
# another loop when the inner loop sits behind an emptiness test, its sequence
# is the same at every outer iteration, and that sequence is empty. The primal
# is correct; the outer loop must run at least twice. This is the shape
# ReactiveKernels emitted for a `scan` inside a `scan` step over an
# outer-invariant sequence: each scan peels its first step behind
# `isempty(indices)` and runs the remaining steps as
# `for i in Iterators.drop(indices, 1)`, with both loops in one generated body.
# Iterating `Iterators.drop`, or a generator over `2:length(indices)`, fails.
# The unit range `(first(indices) + 1):last(indices)`, which ReactiveKernels
# now iterates, differentiates with the same values. So does a loop over all
# indices without the peel. A sequence read anew at each outer step
# (`seqs[j]`), a single outer step, a nonempty sequence, or a step function
# that is not inlined into the outer loop also differentiates. Runtime activity
# does not change any outcome. The failing allocation inside Enzyme's
# derivative has not been localized.
# Recorded on gordito, Julia 1.10.12, Enzyme 0.13.209 and 0.13.210 (Enzyme_jll
# 0.0.301), 2026-10-07.
using Enzyme

# The inner scan, ordinary or `include_init = true` (the seed is the first
# element of the result). `rest` iterates the indices after the peeled first.
@inline function inner(rest, seed, x, us; include_init)
    indices = eachindex(us)
    offset = include_init ? 1 : 0
    outputs = similar(us, Vector{Float64}, length(indices) + offset)
    include_init && (outputs[1] = seed)
    if !isempty(indices)
        i = first(indices)
        state = seed .+ x .* us[i]
        outputs[1 + offset] = state
        position = 1 + offset
        for i in rest(indices)
            state = state .+ x .* us[i]
            position += 1
            @inbounds outputs[position] = state
        end
    end
    outputs
end
@inline rest_drop(indices) = Iterators.drop(indices, 1)
@inline rest_generator(indices) = (indices[k] for k in 2:length(indices))
@inline rest_range(indices) = (first(indices) + 1):last(indices)

# One outer step: the next carry is the inner result's last element, or the
# seed when the inner sequence is empty.
@inline function step(rest, carry, x, us, w; include_init)
    seed = carry .* w[1]
    outputs = inner(rest, seed, x, us; include_init)
    next = include_init ? outputs[end] : (isempty(us) ? seed : outputs[end])
    (next, sum(next))
end

# The outer scan, first step peeled. `sequence(us, j)` is the inner sequence
# of step `j`: the same `us` at every step, or one element of a vector of
# sequences.
@inline function outer(rest, xs, us, w; include_init, sequence)
    carry, total = step(rest, zeros(2), xs[1], sequence(us, 1), w; include_init)
    for j in 2:length(xs)
        carry, out = step(rest, carry, xs[j], sequence(us, j), w; include_init)
        total += out
    end
    total + sum(w)
end
same(us, j) = us
per_step(seqs, j) = seqs[j]

function run(label, f, xs, us, w; mode = Reverse)
    h = 1e-6
    central = [(f(xs, us, w .+ h .* (1:3 .== k)) -
                f(xs, us, w .- h .* (1:3 .== k))) / 2h for k in 1:3]
    result = try
        gradient = Enzyme.gradient(mode, Const(f), Const(xs), Const(us), w)[3]
        "gradient = $gradient, central differences agree: $(isapprox(gradient, central; atol = 1e-6))"
    catch err
        "$(nameof(typeof(err)))"
    end
    println(rpad(label, 58), result)
end

xs, w = [0.3, -0.2, 0.5], [1.0, 2.0, 0.5]
empty = Vector{Float64}[]
println("expected value 3.5 and gradient [1.0, 1.0, 1.0] for every empty case")
for include_init in (true, false)
    kind = include_init ? "include_init" : "ordinary"
    for (name, rest) in (("drop", rest_drop), ("generator", rest_generator),
                         ("unit range", rest_range))
        f = (xs, us, w) -> outer(rest, xs, us, w; include_init, sequence = same)
        @assert f(xs, empty, w) ≈ 3.5
        run("$kind, remaining steps over $name", f, xs, empty, w)
    end
    f = (xs, us, w) -> outer(rest_drop, xs, us, w; include_init, sequence = same)
    run("$kind, drop, runtime activity", f, xs, empty, w;
        mode = set_runtime_activity(Reverse))
    run("$kind, drop, one outer step", f, xs[1:1], empty, w)
    run("$kind, drop, nonempty sequence", f, xs, [[1.0, 2.0], [0.5, -1.0]], w)
    g = (xs, seqs, w) -> outer(rest_drop, xs, seqs, w; include_init,
                               sequence = per_step)
    run("$kind, drop, sequence read per outer step", g, xs, [empty, empty, empty], w)
end
