# Run with ReactiveKernels and its MutatingFunctions extension installed.
using ReactiveKernels, MutatingFunctions, Statistics, LinearAlgebra
BLAS.set_num_threads(1)

@kernel increment(x, h, g) = begin
    y = x .+ (h / 2) .* g
end

function swaps!(step!::F, a, b, g, count) where {F}
    for _ in 1:count
        step!(b, a, g)
        a, b = b, a
    end
    sum(a)
end

function sample(step!::F, a, b, g, count) where {F}
    fill!(a, 0)
    fill!(b, 0)
    @timed swaps!(step!, a, b, g, count)
end

function compare(k, width; count = 200_000)
    a, b, g = zeros(width), zeros(width), cos.(1.0:width)
    h = 0.125
    borrowed = prepare_nonallocating(k, a, h, g)
    writer = prepare_into(k, a, h, g)
    steps = ((dst, x, g) -> copyto!(dst, borrowed(x, h, g)),
             (dst, x, g) -> broadcast!(k, dst, x, h, g),
             (dst, x, g) -> writer(dst, x, h, g))
    times = [Float64[] for _ in steps]
    bytes = [Int[] for _ in steps]
    for round in 0:9
        reference = nothing
        for j in 1:3
            i = mod1(j + round, 3)
            run = sample(steps[i], a, b, g, count)
            if reference === nothing
                reference = (run.value, copy(a), copy(b))
            else
                @assert isequal((run.value, a, b), reference)
            end
            round == 0 && continue
            push!(times[i], 1e9 * run.time / count)
            push!(bytes[i], run.bytes)
        end
    end
    (; width, calls_per_batch = count,
       ns_per_call = map(median, times), max_batch_bytes = map(maximum, bytes),
       paired_copy_over_writer = median(times[1] ./ times[3]))
end

kernel = prepare(increment)
println("Julia ", VERSION, ", Julia threads ", Threads.nthreads(),
    ", BLAS threads ", BLAS.get_num_threads(), ". Columns: borrowed + copy, scalar broadcast, checked writer.")
foreach(n -> println(compare(kernel, n)), (5, 8, 16, 64))
