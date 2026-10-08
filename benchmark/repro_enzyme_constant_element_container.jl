
# Standalone CPU reproducer: only Enzyme is required.
#
# Native Enzyme reverse mode (static activity analysis, the default) rejects a
# fresh container whose elements are arrays loaded from constant data when the
# container's contents then reach an active result:
#   EnzymeRuntimeActivityError: Detected potential need for runtime activity.
#   Constant memory is stored (or returned) to a differentiable variable
#   Mismatched activity for: store ... const val: %arrayref = load ...
# The flagged store writes an element pointer of the constant argument into
# the fresh container. Every element is constant and the derivative is
# defined; static analysis cannot prove the container inactive. `map(identity,
# groups)` fails the same way, as does a field read of constant records. The
# consumer does not matter: Base's `reduce(vcat, A)` and a one-allocation copy
# fold both fail. Copying each element into fresh memory passes, and so does
# reading the constant argument without the intermediate container.
# `set_runtime_activity(Reverse)` differentiates every case. This is the shape
# of a ReactiveKernels plate whose cell returns its unbound argument's element
# (`plate(groups) do g; g end`, `identity(g)`, `s.xs`).
# Recorded on gordito, Julia 1.10.12 and 1.12.7, Enzyme 0.13.210 / Enzyme_jll
# 0.0.301 and Enzyme `dogfood` 07ebfe22, 2026-10-08.
using Enzyme

struct Record
    xs::Vector{Float64}
end

# One allocation, one contiguous copy per element.
function onealloc(per)
    n = 0
    for p in per
        n += length(p)
    end
    out = Vector{Float64}(undef, n)
    k = 1
    for p in per
        unsafe_copyto!(out, k, p, 1, length(p))
        k += length(p)
    end
    out
end
function aliased(groups)
    per = similar(groups)
    for i in eachindex(groups)
        @inbounds per[i] = groups[i]
    end
    per
end
function copied(groups)
    per = similar(groups)
    for i in eachindex(groups)
        @inbounds per[i] = copy(groups[i])
    end
    per
end

cases = [
    "element pointers, Base reduce(vcat)" => (r, g) -> sum(reduce(vcat, aliased(g)) .* r),
    "element pointers, one-allocation fold" => (r, g) -> sum(onealloc(aliased(g)) .* r),
    "map(identity), one-allocation fold" => (r, g) -> sum(onealloc(map(identity, g)) .* r),
    "record fields, one-allocation fold" =>
        (r, g) -> sum(onealloc(map(s -> s.xs, Record.(g))) .* r),
    "element pointers, runtime activity" => (r, g) -> sum(onealloc(aliased(g)) .* r),
    "fresh element copies, Base reduce(vcat)" => (r, g) -> sum(reduce(vcat, copied(g)) .* r),
    "no container, one-allocation fold" => (r, g) -> sum(onealloc(g) .* r),
]

groups = [[0.5, 1.0, 1.0], [0.5], [1.0, 1.0]]
rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
println("Julia $VERSION; expected ok everywhere, failures are the boundary")
for (label, f) in cases
    mode = occursin("runtime activity", label) ? set_runtime_activity(Reverse) : Reverse
    shadow = zero(rates)
    result = try
        autodiff(mode, Const(f), Active, Duplicated(copy(rates), shadow), Const(groups))
        shadow == reduce(vcat, groups) ? "ok" : "wrong gradient $shadow"
    catch err
        "$(nameof(typeof(err)))"
    end
    println(rpad(label, 44), result)
end
