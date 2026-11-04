# julia --startup-file=no --project=. benchmark/shared_cell_keys.jl
# Isolate the lookup used by a data-only plate cell. Fresh equal arrays model
# gathered inputs; repeated objects and distinct values are the controls.
using ReactiveKernels
using Statistics

function shared_sweep(rows)
    table = nothing
    call = ReactiveKernels._SharedCall{(true,)}(sum)
    result = 0.0
    for row in rows
        table, value = ReactiveKernels._shared_step(table, call, row)
        result += value
    end
    result
end

function measure_keys(n, classes; same_objects = false, rounds = 15, repeats = 20)
    exemplars = [fill(Float64(i), n) for i in 1:classes]
    rows = [same_objects ? exemplars[mod1(i, classes)] : copy(exemplars[mod1(i, classes)])
            for i in 1:256]
    expected = sum(sum, rows)
    @assert shared_sweep(rows) == expected
    shared_sweep(rows)
    samples = Float64[]
    for _ in 1:rounds
        elapsed = @elapsed for _ in 1:repeats
            shared_sweep(rows)
        end
        push!(samples, elapsed / repeats)
    end
    bytes = @allocated shared_sweep(rows)
    println((length = n, classes, same_objects, median_seconds = median(samples),
             minimum_seconds = minimum(samples), bytes, value = expected))
end

println((julia = VERSION, threads = Threads.nthreads(), source = pathof(ReactiveKernels)))
for n in (16, 512, 4096), classes in (1, 2, 256)
    measure_keys(n, classes)
end
for classes in (1, 2, 256)
    measure_keys(512, classes; same_objects = true)
end
