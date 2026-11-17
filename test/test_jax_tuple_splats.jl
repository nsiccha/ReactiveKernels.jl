module JAXTupleSplatTests
# A tuple literal or call that splats a tuple (`(a, rest(...)...)`,
# `tuple(a, rest...)`, `f(args...)`) passes the tuple's fields, whose count is
# the tuple's type's: a recursion over a record's field names unrolls over
# those names, never over data (snag jax-tuple-splat-05c8497a). Each value and
# ordinary reverse gradient is compared with the native kernel's.
using ReactiveKernels, Test, Enzyme
using LogExpFunctions: logsumexp
include("jax_python.jl")

# The reported case: per-field column sums, by recursion over the names.
@traceable splat_column_total(column) = sum(sum, column)
@traceable splat_field_totals(values, ::Tuple{}) = ()
@traceable function splat_field_totals(values, names::Tuple)
    name = first(names)
    column = map(values) do value
        getproperty(value, name)
    end
    return (splat_column_total(column), splat_field_totals(values, Base.tail(names))...)
end
@traceable function splat_record_totals(values::AbstractArray{<:NamedTuple{names}}) where {names}
    return NamedTuple{names}(splat_field_totals(values, names))
end
@traceable function splat_plain(records)
    out = splat_record_totals(vec(records))
    return out.a + 2 * out.b
end

# The same recursion, where each field's step also advances weights shared
# across the fields, in field order: field `b` reads the weights field `a`
# left behind (a sequential factorization).
@traceable function splat_advance!(column, weights)
    total = 0.0
    for i in eachindex(first(column))
        before = logsumexp(weights)
        for (s, value) in enumerate(column)
            weights[s] += value[i]
        end
        total += logsumexp(weights) - before
    end
    return total
end
@traceable splat_field_steps(values, weights, ::Tuple{}) = ()
@traceable function splat_field_steps(values, weights, names::Tuple)
    name = first(names)
    column = map(values) do value
        getproperty(value, name)
    end
    step = splat_advance!(column, weights)
    return (step, splat_field_steps(values, weights, Base.tail(names))...)
end
@traceable function splat_record_steps(values::AbstractArray{<:NamedTuple{names}},
                                       weights) where {names}
    return NamedTuple{names}(splat_field_steps(values, weights, names))
end
@traceable function splat_sequential(records, w0)
    weights = copy(w0)
    out = splat_record_steps(vec(records), weights)
    return out.a + 2 * out.b + sum(weights)
end

# Other spellings: `tuple` called with a splat, a splat in the middle of a
# literal, a helper called with splatted arguments, a splatted named tuple,
# and a plain (lowered) helper that splats a record into a call.
@traceable splat_scaled(s, a, b, c) = s * (a + 2b + 3c)
plain_combine(a, b, c) = a * b - c
plain_splat_record(r) = plain_combine(r...)
@traceable function splat_spellings(records, x)
    out = splat_record_totals(vec(records))
    pair = (out.a, out.b)
    t = tuple(1.5, pair...)
    middle = (out.b, pair..., x)
    fields = (out..., x)
    return t[1] * t[2] + t[3] + splat_scaled(2.0, pair..., x) + middle[1] * middle[4] +
           middle[2] - middle[3] + fields[1] * fields[3] + fields[2] +
           plain_splat_record((; a = out.a, b = x, c = out.b))
end

for (name, value) in ((:jax_splat_plain, :(splat_plain(records))),
                      (:jax_splat_spellings, :(splat_spellings(records, x))))
    @eval @kernel $name(x::Float64, indices::Vector{Int}, y::Vector{Float64}) = begin
        records = plate(indices, x, Ref(y)) do i, s, ys
            (; a = ys .* (s + i), b = ys .* (s - i))
        end
        value = $value
        return value
    end
end
@kernel jax_splat_sequential(x::Float64, indices::Vector{Int}, y::Vector{Float64},
                             w0::Vector{Float64}) = begin
    records = plate(indices, x, Ref(y)) do i, s, ys
        (; a = ys .* (s + i), b = ys .* (s - i))
    end
    value = splat_sequential(records, w0)
    return value
end

# Plain-Julia references, written without the kernel.
function splat_plain_reference(x, indices, y)
    a = sum(sum(y .* (x + i)) for i in indices; init = 0.0)
    b = sum(sum(y .* (x - i)) for i in indices; init = 0.0)
    return a + 2b
end
function splat_sequential_reference(x, indices, y, w0)
    weights = copy(w0)
    out = map((+, -)) do op
        total = 0.0
        for k in eachindex(y)
            before = logsumexp(weights)
            for (s, i) in enumerate(indices)
                weights[s] += y[k] * op(x, i)
            end
            total += logsumexp(weights) - before
        end
        total
    end
    return out[1] + 2 * out[2] + sum(weights)
end

# A splat whose length is data (a vector's) has no fixed count to pass: it
# stays refused rather than unrolled over the data (CONTRIBUTING.md,
# "Data-dependent iteration").
@traceable splat_vector_sum(v) = sum((v...,))
@kernel jax_splat_vector(x::Float64, y::Vector{Float64}) = begin
    value = splat_vector_sum(x .* y)
    return value
end

function jax_splat_parity(spec, have, datasets)
    data(d) = NamedTuple{Base.tail(have)}(d)
    native = prepare(spec; have, want = :value)
    src = jax_source(spec; have, data = data(first(datasets)), functions = (value = :value,))
    # One module for every dataset: the splats unroll over field names, and
    # nothing is unrolled over the data.
    @test all(string(jax_source(spec; have, data = data(d), functions = (value = :value,))) ==
              string(src) for d in datasets)
    if rk_jax_available()
        for d in datasets, mode in (true, :arguments), x in (0.4, -1.3)
            got = rk_jax_evaluate(src, :value, (x, d...); grad = :joint, prepare = mode)
            @test got.value ≈ native(x, d...) atol = 1e-10
            @test got.joint_value ≈ got.value atol = 1e-10
            derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active,
                Enzyme.Active(x), map(Enzyme.Const, d)...)[1][1]
            @test got.grad ≈ derivative atol = 1e-10
        end
    else
        @test_skip false
    end
    src
end

@testset "jax_source: splatted tuples in literals and calls" begin
    datasets = [([1, 2, 3], [0.2, 0.7]), ([4, 1], [0.3, -0.1, 1.5, 2.0]), ([2, 5, 1], Float64[])]
    native = prepare(jax_splat_plain; have = (:x, :indices, :y), want = :value)
    for d in datasets
        @test native(0.4, d...) ≈ splat_plain_reference(0.4, d...)
    end
    jax_splat_parity(jax_splat_plain, (:x, :indices, :y), datasets)
    jax_splat_parity(jax_splat_spellings, (:x, :indices, :y), datasets)

    sequential = [([1, 2, 3], [0.2, 0.7], [0.0, -0.5, 0.3]),
                  ([4, 1], [0.3, -0.1, 1.5, 2.0], [0.1, 0.2]),
                  ([2, 5, 1], Float64[], [0.0, 1.0, -1.0])]
    native = prepare(jax_splat_sequential; have = (:x, :indices, :y, :w0), want = :value)
    for d in sequential
        @test native(0.4, d...) ≈ splat_sequential_reference(0.4, d...)
    end
    jax_splat_parity(jax_splat_sequential, (:x, :indices, :y, :w0), sequential)

    native = prepare(jax_splat_vector; have = (:x, :y), want = :value)
    @test native(0.5, [1.0, 2.0]) == 1.5
    err = try
        jax_source(jax_splat_vector; have = (:x, :y), want = :value)
        nothing
    catch e
        e
    end
    @test err isa JAXEmissionError
    @test occursin("splatting a `Vector{Float64}` has no JAX translation", sprint(showerror, err))
end

end
