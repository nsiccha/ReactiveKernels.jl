# Run with Enzyme in the active environment. The second Reverse call raises
# EnzymeNoTypeError on Julia 1.10.12, Enzyme 3f556164, libEnzyme c423e860.
using Enzyme

const Record = Tuple{@NamedTuple{pointwise::@NamedTuple{y::Matrix{Float64}}},
                     @NamedTuple{prior::Float64}}

@inline function coordinate_changed(index, previous, states)
    @inbounds for d in eachindex(axes(states))
        size(states, d) == 1 && continue
        index[d] != previous[d] && return true
    end
    false
end

Base.@noinline function fill_records!(records, states, x, y)
    scaled = x .* y
    local record
    previous = nothing
    first_cell = true
    for k in CartesianIndices(states)
        if first_cell || coordinate_changed(k, previous, states)
            i, j = states[k]
            record = ((; pointwise = (; y = scaled)), (; prior = x * i + j))
        end
        @inbounds records[k] = record
        first_cell, previous = false, k
    end
    nothing
end

function scheduled_records(x, a, b, y)
    states = collect(Iterators.product(a, b))
    records = similar(states, Record)
    isempty(states) || fill_records!(records, states, x, y)
    sum(r -> sum(r[1].pointwise.y) + r[2].prior, records; init = 0.0) +
        length(records) + size(records, 2)
end

function comprehension(x, a, b, y)
    records = [((; pointwise = (; y = x .* y)), (; prior = x * i + j)) for i in a, j in b]
    sum(r -> sum(r[1].pointwise.y) + r[2].prior, records; init = 0.0) +
        length(records) + size(records, 2)
end

x, a, b, y = 0.4, [1, 2], [3, 4, 5], [0.2 0.3; 0.5 0.7]
for f in (comprehension, scheduled_records)
    value = f(x, a, b, y)
    println(f, " value = ", value)
    @assert value ≈ 40.68
    gradient = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active,
        Enzyme.Active(x), Enzyme.Const(a), Enzyme.Const(b), Enzyme.Const(y))[1][1]
    println(f, " gradient = ", gradient)
    @assert gradient ≈ 19.2
end
