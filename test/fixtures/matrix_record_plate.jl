module MatrixRecordPlateFixture

using ReactiveKernels

@kernel matrix_records(x::Float64, a::Vector{Int}, b::Vector{Int}, y::Matrix{Float64}) = begin
    states = collect(Iterators.product(a, b))
    records = plate(states, x, Ref(y)) do ij, s, ys
        ((; pointwise = (; y = ys .* s)), (; prior = s * ij[1] + ij[2]))
    end
    terms = plate(records) do r
        sum(r[1].pointwise.y) + r[2].prior
    end
    value = sum(terms) + length(records) + size(records, 2)
    return value
end

function reference(x, a, b, y)
    records = [((; pointwise = (; y = y .* x)), (; prior = x * i + j)) for i in a, j in b]
    terms = map(r -> sum(r[1].pointwise.y) + r[2].prior, records)
    sum(terms) + length(records) + size(records, 2)
end

end
