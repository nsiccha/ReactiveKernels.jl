# Standalone CPU reproducer: only Enzyme is required. Run it with
# `--check-bounds=yes` to cover the `broadcast!` rows (Base's broadcast loop
# elides its bounds checks otherwise).
#
# Native Enzyme reverse mode allocates GC memory in proportion to a loop's
# length when the loop stores into a buffer held in a `Ref` slot, the stores
# keep their bounds checks, and the buffer is stored back into the slot after
# the loop. Julia 1.12's `setindex!` loads the array's `Memory` pointer in
# every iteration for its bounds check. Enzyme computes the shadow of that load
# and tapes it once per iteration, a GC-tracked tape plus a same-length
# type-tag svec (`ijl_svec_fill`): 16 bytes per element, although the loaded
# address does not change in the loop. The rows show which shapes trip it: a
# result selected as reuse-or-`similar`, and `broadcast!` into the slot's
# buffer, allocate; always reusing the buffer in a hand-written loop, or not
# storing the buffer back, does not. The broadcast tape is larger on Julia
# 1.13 than 1.12. Julia 1.10 allocates nothing in every row. Gradients are
# correct in every row.
#
# ReactiveKernels' non-allocating reverse plate loop (`_ad_na_plate_loop!`)
# selects its result buffer as reuse-or-`similar`, and a non-allocating cache
# step applies `broadcast!` into its slot, so under `--check-bounds=yes` both
# gradients allocate in proportion to element count on Julia 1.12 and later
# (`test_nonallocating_ad.jl`).
# Recorded on gordito, Enzyme 0.13.216 (Enzyme_jll 0.0.303), 2026-10-13, with
# `--check-bounds=yes`, as warm Julia 1.12.7 gradient bytes at
# n = (8, 64, 512, 4096):
#   loop, reuse-or-similar: (80, 544, 8384, 65728)
#   loop, reuse:           (0, 0, 0, 0)
#   broadcast!, writeback: (80, 544, 8384, 65728)
#   broadcast!, no write:  (0, 0, 0, 0)
# On Julia 1.13.1 the writeback broadcast row is (80, 544, 12608, 98624);
# the reuse-or-similar row remains (80, 544, 8384, 65728). All four rows are
# zero on Julia 1.10.12.
using Enzyme

cell(xi, b, yi) = -0.5 * (xi * b - yi)^2

# Reuse the buffer when its length matches, otherwise allocate: the phi.
select_reuse_or_similar(cache, n) = length(cache) == n ? cache : similar(cache, n)
# Always reuse the buffer.
select_reuse(cache, n) = cache

struct HandLoop{S,X,Y}
    select::S
    x::X
    y::Y
end

function (p::HandLoop)(q, slot)
    b = q[1]
    x, y = p.x, p.y
    result = p.select(slot[], length(x))
    for i in eachindex(x)
        result[i] = cell(x[i], b, y[i])
    end
    slot[] = result
    return sum(result)
end

struct Broadcast{W,T}
    c::T
end

function (p::Broadcast{W})(x, slot) where {W}
    result = slot[]
    broadcast!(*, result, x, p.c)
    W && (slot[] = result)
    return sum(result)
end

function gradient_bytes(program, n)
    x = collect(range(-1.0, 1.0; length = n))
    y = fill(0.3, n)
    q, dq = [0.7], zeros(1)
    slot, dslot = Ref(zeros(n)), Ref(zeros(n))
    callable = program(x, y)
    gradient = () -> autodiff(Reverse, Const(callable), Active,
                              Duplicated(q, dq), Duplicated(slot, dslot))
    measure(gradient, dq, dslot, [-sum((0.7 .* x .- 0.3) .* x)])
end

function broadcast_gradient_bytes(::Val{W}, n) where {W}
    x, dx = collect(range(-1.0, 1.0; length = n)), zeros(n)
    slot, dslot = Ref(zeros(n)), Ref(zeros(n))
    callable = Broadcast{W,Float64}(2.5)
    gradient = () -> autodiff(Reverse, Const(callable), Active,
                              Duplicated(x, dx), Duplicated(slot, dslot))
    measure(gradient, dx, dslot, fill(2.5, n))
end

function measure(gradient::F, dq, dslot, expected) where {F}
    gradient()
    gradient()
    dq .= 0
    dslot[] .= 0
    gradient()
    correct = isapprox(dq, expected; rtol = 1e-10)
    (@allocated gradient()), correct
end

println("Julia ", VERSION, ", Enzyme ", pkgversion(Enzyme), ", check-bounds ",
        Base.JLOptions().check_bounds)
for (label, probe) in (
        ("loop, reuse-or-similar", n -> gradient_bytes((x, y) -> HandLoop(select_reuse_or_similar, x, y), n)),
        ("loop, reuse", n -> gradient_bytes((x, y) -> HandLoop(select_reuse, x, y), n)),
        ("broadcast!, write back", n -> broadcast_gradient_bytes(Val(true), n)),
        ("broadcast!, no write back", n -> broadcast_gradient_bytes(Val(false), n)))
    results = map(probe, (8, 64, 512, 4096))
    println(rpad(label, 27), "bytes at n = (8, 64, 512, 4096): ",
            map(first, results), ", gradients correct: ", all(last, results))
end
