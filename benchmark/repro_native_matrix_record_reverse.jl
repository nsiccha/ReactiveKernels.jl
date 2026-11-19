# Native-only reproduction; no JAX emission or Python is involved.
# Julia 1.10.12-kb6, Enzyme.jl 9430736f8c9f694279002001f08675d52289637b,
# libEnzyme eb82b697068538ee9074e2751f160838a8154005 / LLVM 15, 2026-11-16.
# On that build both primals returned 40.68. Plain Julia reverse returned 19.2,
# and the prepared plate raised EnzymeNoTypeError at setindex!. Enzyme dogfood
# 22bb66a8 repairs the loop-carried record, giving 19.2 for both gradients.
using ReactiveKernels, Enzyme

@kernel matrix_record_kernel(x::Float64, a::Vector{Int}, b::Vector{Int}, y::Matrix{Float64}) = begin
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

function plain_records(x, a, b, y)
    records = [((; pointwise = (; y = y .* x)), (; prior = x * i + j)) for i in a, j in b]
    terms = map(r -> sum(r[1].pointwise.y) + r[2].prior, records)
    sum(terms) + length(records) + size(records, 2)
end

args = (0.4, [1, 2], [3, 4, 5], [0.2 0.3; 0.5 0.7])
native = prepare(matrix_record_kernel; have = (:x, :a, :b, :y), want = :value)
for (name, f) in (("plain Julia", plain_records), ("prepared kernel", native))
    println(name, " value=", f(args...))
    try
        derivative = Enzyme.autodiff(Enzyme.Reverse, f, Enzyme.Active,
            Enzyme.Active(first(args)), map(Enzyme.Const, Base.tail(args))...)[1][1]
        println(name, " gradient=", derivative)
    catch err
        println(name, " error: ", sprint(showerror, err))
    end
end
