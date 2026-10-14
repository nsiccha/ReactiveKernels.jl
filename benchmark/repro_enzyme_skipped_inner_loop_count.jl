# Standalone CPU reproducer: only Enzyme is required.
#
# Native Enzyme reverse mode sizes the cache of an inner loop outside the loop
# around it when it can evaluate the inner trip count there. For a loop over
# `2:m` (a scan that runs its first step before its loop) that count is
# `m - 1`, which is -1 when the sequence is empty and the inner loop does not
# run. The combined cache is then sized `(m - 1) * outer steps`. Since Enzyme
# 0.13.206 a negative count reaches `jl_svec_fill`: -2 or less throws
# `OutOfMemoryError` from the gradient, and exactly -1 writes past a
# zero-byte allocation (a segmentation fault, or silent heap corruption). The
# primal and its derivative are well defined: the skipped loop contributes
# nothing. EnzymeAD/Enzyme.jl#3787 (open); the Enzyme `dogfood` line carries
# the fix (`d37d3d9`, "Size caches of skipped inner loops as empty").
#
# ReactiveKernels' native `scan` peels its first step and loops over the rest
# (`repro_enzyme_guarded_inner_loop_cache.jl` records why it iterates a unit
# range), so a scan inside a scan step over an empty sequence is this shape.
# Until RK's source callable asked `Core._hasmethod` (snag
# `guarded-kernelso-e7baca4c`), the per-step `hasmethod` query left an opaque
# call in the step on Julia 1.12+, and Enzyme did not move that cache;
# `test_scan_plate_ad.jl` now pins the affected cases on Julia 1.12+.
#
# The shape needs the array-valued carry and the inlined helper; with a scalar
# carry, or with `scan_inner` not inlined, the gradient is correct. Runtime
# activity does not change the outcome.
#
# Recorded on gordito, 2026-10-14, for `length(us) = 0` and three outer steps
# (expected gradient [1.0, 1.0, 1.0]):
#   Julia 1.13.1 / 1.12.7, Enzyme 0.13.216 (Enzyme_jll 0.0.303):
#     counted: OutOfMemoryError; iterated: OutOfMemoryError
#   Julia 1.13.1, Enzyme dogfood de43d9d (0.13.215, Enzyme_jll 0.0.301): both correct
#   Julia 1.10.12, Enzyme 0.13.216: every row, nonempty `us` included, raises
#     `EnzymeRuntimeActivityError` instead, so this file does not isolate the
#     count there. RK's own nested-scan testset differentiates on Julia 1.10.
# Pass `one-step` to also run one outer step, which overruns the heap on an
# affected Enzyme; it is opt-in so the default run cannot corrupt the process.
using Enzyme

# A scan that runs its first step before its loop: the loop is `for i in 2:m`
# and runs `m - 1` times, so not at all when `us` is empty.
@inline function scan_inner(init, us, x)
    m = length(us)
    c = init
    if m >= 1
        c = c .+ x .* us[1]
        for i in 2:m
            c = c .+ x .* us[i]
        end
    end
    return c
end

# Outer loop with a trip count known on entry.
function counted(xs, us, w)
    carry = zeros(2)
    total = 0.0
    for k in eachindex(xs)
        carry = scan_inner(carry .* w[1], us, xs[k])
        total += sum(carry)
    end
    return total + sum(w)
end

# The same computation with the outer loop written over the values.
function iterated(xs, us, w)
    carry = zeros(2)
    total = 0.0
    for x in xs
        carry = scan_inner(carry .* w[1], us, x)
        total += sum(carry)
    end
    return total + sum(w)
end

w = [1.3, 2.0, 0.5]
outer = "one-step" in ARGS ? ([0.3, -0.2, 0.5], [0.3]) : ([0.3, -0.2, 0.5],)
println("Julia ", VERSION, ", Enzyme ", pkgversion(Enzyme))
for f in (counted, iterated), us in ([[1.0, 2.0], [0.5, -1.0]], Vector{Float64}[]),
        xs in outer
    dw = zero(w)
    result = try
        autodiff(Reverse, f, Active, Const(xs), Const(us), Duplicated(w, dw))
        dw
    catch err
        nameof(typeof(err))
    end
    println(f, ", length(us) = ", length(us), ", length(xs) = ", length(xs),
            ": ", result)
end
