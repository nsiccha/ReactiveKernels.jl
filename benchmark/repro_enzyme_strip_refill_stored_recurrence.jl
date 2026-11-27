# Standalone CPU reproducer: only Enzyme is required.
#
# Native Enzyme reverse mode crashes the process (a segmentation fault, or
# glibc's `malloc(): unaligned tcache chunk detected` / `free(): invalid size`,
# varying from run to run) at the first gradient of this in-bounds function.
# The primal value and central differences are right.
#
# `strip_walks` is a hand transcription of the loops ReactiveKernels generates
# for a plate cell whose gather plate feeds only a scan. Such a plate runs
# strip by strip with the scan (`_PLATE_STRIP` = 128 cells): one buffer is
# filled with the first 128 gathered values, the scan steps over them, and a
# `while` loop refills the same buffer for each later strip. The kernel is
#
#     @kernel grouped_walks(theta::Vector{Float64}, index_by, groups) = begin
#         cells = plate(groups) do g
#             amount = plate(index_by[g]) do q
#                 theta[q]
#             end
#             walk = scan(amount; init = 0.0) do carry, amt
#                 next = 0.5 * carry + amt
#                 (next, next)
#             end
#             sum(abs2, walk; init = 0.0)
#         end
#         total::Float64 = sum(cells)
#         return total
#     end
#
# and `prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), theta; active = :theta)`
# crashes the same way from 257 steps in one group, with default bounds
# checking and with `--check-bounds=yes`, and passes at 256.
#
# What this function needs to crash (each line one fresh process):
#
#   groups 387   one group of 387 steps (strips of 128, 128, 128, 3)   crash
#   groups 257   strips of 128, 128, 1                                 crash
#   groups 256   strips of 128, 128                                    passes
#   single 387   the same body, no loop over groups                    passes
#
# Variants that pass at 387 steps, not kept here: every strip (the first
# included) inside the `while` loop; nothing stored (the squares summed in the
# recurrence); a new buffer per strip; strips counted by a `for` loop. So the
# loop over groups, the first strip ahead of the `while` loop, at least two
# passes of that loop and the stored steps are all needed. Turning off the
# dogfood line's packed-cache grouping (`EnzymeGroupPackedCaches`) or its tape
# arena (`EnzymeCallerOwnedTapes`) does not avoid it.
#
# It is an upstream Enzyme defect: `groups 387` crashes on each of
#
#   - Enzyme.jl dogfood `b77d0b96c` with libEnzyme `eb5570f7`;
#   - the same Enzyme.jl with the library `Enzyme_jll` 0.0.301 ships;
#   - registered Enzyme 0.13.221 with `Enzyme_jll` 0.0.310.
#
# `groups 256` passes on the first and the third; it was not run on the
# second.
#
# It is specific to Julia 1.10. Julia 1.12.7 and 1.13.1 with the same
# registered Enzyme 0.13.221 return the gradient at 387, 257 and 256 steps.
#
# ReactiveKernels keeps its strip lowering: the program is valid, and
# reshaping it so that Enzyme accepts it would hide the defect. The repair
# belongs in Enzyme.
#
# Recorded on a Linux fleet host, 2026-11-27, on Julia 1.10.12 (LLVM 15)
# unless another version is named.
#
#   julia --startup-file=no --project=<env> benchmark/repro_enzyme_strip_refill_stored_recurrence.jl groups 387
using Enzyme

const STRIP = 128

function strip_walks(theta::Vector{Float64}, index_by::Vector{Vector{Int}})
    total = 0.0
    for g in eachindex(index_by)
        total += strip_walk(theta, index_by[g])
    end
    return total
end

@inline function strip_walk(theta::Vector{Float64}, idx::Vector{Int})
    n = length(idx)
    lo = 1
    hi = min(lo + STRIP - 1, n)
    amount = Vector{Float64}(undef, hi - lo + 1)
    @inbounds for i in 1:(hi - lo + 1)
        amount[i] = theta[idx[lo + i - 1]]
    end
    walk = Vector{Float64}(undef, n)
    if n > 0
        carry = 0.0
        @inbounds for j in 1:hi
            next = 0.5 * carry + amount[(j - lo) + 1]
            carry = next
            walk[j] = next
        end
        lo = hi + 1
        while lo <= n
            hi = min(lo + STRIP - 1, n)
            @inbounds for i in 1:(hi - lo + 1)
                amount[i] = theta[idx[lo + i - 1]]
            end
            @inbounds for j in lo:hi
                next = 0.5 * carry + amount[(j - lo) + 1]
                carry = next
                walk[j] = next
            end
            lo = hi + 1
        end
    end
    s = 0.0
    @inbounds for j in 1:n
        s += abs2(walk[j])
    end
    return s
end

single_walk(theta::Vector{Float64}, index_by::Vector{Vector{Int}}) = strip_walk(theta, index_by[1])

function central(f, theta; h = 1e-6)
    map(eachindex(theta)) do i
        up = copy(theta)
        up[i] += h
        down = copy(theta)
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end
end

function main(shape, lengths)
    f = shape == "groups" ? strip_walks : shape == "single" ? single_walk :
        error("the first argument is `groups` or `single`")
    index_by = [[1 + mod(i, 3) for i in 1:n] for n in lengths]
    theta = [0.3, -0.2, 0.7]
    println("libEnzyme ", Enzyme.Compiler.Enzyme_jll.libEnzyme_path)
    println("primal ", f(theta, index_by))
    expected = central(t -> f(t, index_by), theta)
    println("central differences ", expected)
    flush(stdout)
    for round in 1:3
        gradient = zero(theta)
        Enzyme.autodiff(Reverse, f, Active, Duplicated(theta, gradient), Const(index_by))
        println("round ", round, " gradient ", gradient, " max error ",
            maximum(abs, gradient .- expected))
        flush(stdout)
    end
    println("DONE")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: <groups|single> <comma-separated group lengths>")
    main(ARGS[1], parse.(Int, split(ARGS[2], ",")))
end
