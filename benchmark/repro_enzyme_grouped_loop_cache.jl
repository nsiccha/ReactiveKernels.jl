# Count native reverse-cache allocations for an event × component × index
# loop. The two primals compute the same polynomial on disjoint buffers.
# One writes each event total to caller-owned storage before summing it.
#
# On Linux with glibc, from the repository root:
#   counter_dir=$(mktemp -d)
#   cc -O2 -shared -fPIC benchmark/covariate_free_mixture/count-allocators.c -o "$counter_dir/count.so"
#   LD_PRELOAD="$counter_dir/count.so" julia --project=<enzyme-env> benchmark/repro_enzyme_grouped_loop_cache.jl
module GroupedLoopCache
using Enzyme
include("covariate_free_mixture/full-counter-method.jl")

function score(x, events, output)
    total = 0.0
    @inbounds for i in eachindex(events)
        event = events[i]
        event_score = 0.0
        for k in eachindex(event)
            indices = event[k]
            s = 0.0
            for j in eachindex(indices)
                s += x[indices[j]]^2
            end
            event_score += s^2
        end
        if output === nothing
            total += event_score
        else
            output[i] = event_score
        end
    end
    return output === nothing ? total : sum(output)
end

function reference(x, events)
    gradient = zero(x)
    total = 0.0
    for event in events, indices in event
        s = sum(abs2, x[indices]; init=0.0)
        total += s^2
        for index in indices
            gradient[index] += 4s * x[index]
        end
    end
    return total, gradient
end

function measure(x, events, output, annotation)
    dx = zero(x)
    function gradient!()
        fill!(dx, 0.0)
        if annotation isa Duplicated
            fill!(annotation.dval, 0.0)
        end
        last(autodiff(ReverseWithPrimal, score, Active,
            Duplicated(x, dx), Const(events), annotation))
    end
    value, gradient = reference(x, events)
    @assert score(x, events, output) ≈ value
    @assert gradient!() ≈ value
    @assert dx ≈ gradient
    primal = native_counts(() -> score(x, events, output))
    reverse = native_counts(gradient!)
    println((; events=length(events), components=sum(length, events; init=0),
        output=output === nothing ? "scalar" : "buffer",
        primal_malloc=Int(primal.malloc), reverse_malloc=Int(reverse.malloc),
        reverse_free=Int(reverse.free), julia_bytes=@allocated(gradient!())))
end

function compare(x, events)
    measure(x, events, nothing, Const(nothing))
    output = zeros(length(events))
    measure(x, events, output, Duplicated(output, zero(output)))
end

function run()
    println((; julia=VERSION, enzyme=pkgversion(Enzyme),
        library=Enzyme.Compiler.Enzyme_jll.libEnzyme_path))
    x = [0.04cos(i) for i in 1:50]
    compare(x, Vector{Vector{Int}}[])
    compare(x, [Vector{Int}[], [Int[], [3, 3, 5]]])
    for n in (5, 50, 100), k in (1, 2, 4)
        events = [[[mod1(i + j, length(x)) for j in 1:(1 + mod(i + c, 4))]
            for c in 1:k] for i in 1:n]
        compare(x, events)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && run()
end
