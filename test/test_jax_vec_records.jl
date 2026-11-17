module JAXVecRecordTests
# `vec` and `reshape` of an array of plate records of rank two or more in a
# `@traceable` helper: a plate over a matrix (or a rank-3 array) of states
# returns records stored as the tuple of their fields' arrays, the array's
# axes first, and `vec` of them is the vector of those records in Julia's
# column-major order, each field flattened alike (snag
# jax-vec-of-a-mat-a50cdf28); `reshape` reorders them alike. Each record is
# weighted by its position in the vector, so an order other than Julia's
# changes the value. Values and ordinary reverse gradients are compared with
# the native kernel's.
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

# The records in Julia's linear order, three ways: `vec`, `reshape` to a
# vector, and `vec` of the transposed shape. Julia's reshape keeps the linear
# order, so all three weigh the records alike; a reshape in row-major order
# would not.
@traceable flat_vec(records) = vec(records)
@traceable flat_reshape(records) = reshape(records, :)
@traceable flat_transposed(records) = vec(reshape(records, size(records, 2), :))

# A vector-valued field, read per record by position.
@traceable function weighted_total(flat)
    pointwise = map(flat) do record
        record.pointwise
    end
    total = 0.0
    for (k, value) in enumerate(pointwise)
        total += k * sum(value)
    end
    return total + length(pointwise)
end
# Two fields, one a nested record, read in one map.
@traceable function nested_total(flat)
    totals = map(flat) do record
        record.a * sum(record.nested.y) + record.nested.b
    end
    total = 0.0
    for (k, value) in enumerate(totals)
        total += k * value
    end
    return total + length(flat)
end

const FLATTENS = (:flat_vec, :flat_reshape, :flat_transposed)
for f in FLATTENS
    @eval @kernel $(Symbol(:jax_vector_field_, f))(x::Float64, m::Int, y::Vector{Float64}) = begin
        states = collect(Iterators.product(0:1, 0:m))
        records = plate(states, x, Ref(y)) do state, s, ys
            (; pointwise = ys .* (s + state[1] + 2 * state[2]))
        end
        value = weighted_total($f(records))
        return value
    end
    @eval @kernel $(Symbol(:jax_scalar_field_, f))(x::Float64, m::Int, y::Vector{Float64}) = begin
        states = collect(Iterators.product(0:1, 0:m))
        records = plate(states, x, Ref(y)) do state, s, ys
            (; pointwise = sum(ys) * (s + state[1] + 2 * state[2]))
        end
        value = weighted_total($f(records))
        return value
    end
    @eval @kernel $(Symbol(:jax_nested_fields_, f))(x::Float64, m::Int, y::Vector{Float64}) = begin
        states = collect(Iterators.product(0:2, 0:m))
        records = plate(states, x, Ref(y)) do state, s, ys
            (; a = s + state[1], nested = (; y = ys .* (s - state[2]), b = s * state[2] + state[1]))
        end
        value = nested_total($f(records))
        return value
    end
    # Each record's field is a ragged row of the data, padded on the device:
    # the records keep their own lengths.
    @eval @kernel $(Symbol(:jax_padded_field_, f))(x::Float64, m::Int, rows::Vector{Vector{Float64}}) = begin
        states = collect(Iterators.product(1:length(rows), 0:m))
        records = plate(states, x, Ref(rows)) do state, s, rs
            (; pointwise = rs[state[1]] .* (s + state[2]))
        end
        value = weighted_total($f(records))
        return value
    end
end
@kernel jax_vec_rank3(x::Float64, m::Int, y::Vector{Float64}) = begin
    states = collect(Iterators.product(0:1, 0:m, 1:2))
    records = plate(states, x, Ref(y)) do state, s, ys
        (; pointwise = ys .* (s + state[1] + 2 * state[2] + 3 * state[3]))
    end
    value = weighted_total(vec(records))
    return value
end
# A plate over a padded matrix (both axes padded by a ragged row's length)
# inside a cell: its records' valid entries come first through `vec`, in
# column-major order, and the padding is not counted.
@traceable function padded_cells_total(cells)
    values = map(vec(cells)) do cell
        cell.v
    end
    total = 0.0
    for (k, v) in enumerate(values)
        total += k * v
    end
    return total + length(values)
end
@kernel jax_vec_padded_matrix(x::Float64, rows::Vector{Vector{Float64}}) = begin
    totals = plate(rows, x) do row, s
        products = row .* (row .+ 1.0)'
        cells = plate(products, s) do p, t
            (; v = p * t)
        end
        padded_cells_total(cells)
    end
    value = sum(totals)
    return value
end
# `reshape` of such a padded matrix of records is refused, as `reshape` of a
# padded numerical array is.
@traceable function reshaped_cells_total(cells)
    values = map(reshape(cells, :)) do cell
        cell.v
    end
    return sum(values)
end
@kernel jax_reshape_padded_matrix(x::Float64, rows::Vector{Vector{Float64}}) = begin
    totals = plate(rows, x) do row, s
        products = row .* (row .+ 1.0)'
        cells = plate(products, s) do p, t
            (; v = p * t)
        end
        reshaped_cells_total(cells)
    end
    value = sum(totals)
    return value
end

function jax_vec_parity(spec, have, datasets)
    data(d) = NamedTuple{Base.tail(have)}(d)
    native = prepare(spec; have, want = :value)
    src = jax_source(spec; have, data = data(first(datasets)), functions = (value = :value,))
    @test occursin("_rk_reshape_records", string(src))
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

@testset "jax_source: vec and reshape of arrays of plate records" begin
    dense = [(2, [0.2, 0.7]), (0, [1.5]), (3, [0.3, -0.1, 1.5])]
    rows = [(1, [[0.2, 0.7], [0.3], [0.1, -0.3, 0.5]]), (0, [[1.0], [2.0, 3.0]])]
    for f in FLATTENS
        for kind in (:vector_field, :scalar_field, :nested_fields)
            jax_vec_parity(getfield(@__MODULE__, Symbol(:jax_, kind, :_, f)), (:x, :m, :y), dense)
        end
        jax_vec_parity(getfield(@__MODULE__, Symbol(:jax_padded_field_, f)), (:x, :m, :rows), rows)
    end
    jax_vec_parity(jax_vec_rank3, (:x, :m, :y), dense)
    # The plain-Julia value of the reported kernel: records in column-major
    # order over (i in 0:1, j in 0:m), each weighted by its position.
    reference(x, m, y) = sum(k * sum(y) * (x + i + 2j)
                             for (k, (i, j)) in enumerate(Iterators.product(0:1, 0:m))) + 2 * (m + 1)
    native = prepare(jax_vector_field_flat_vec; have = (:x, :m, :y), want = :value)
    @test native(0.4, 2, [0.2, 0.7]) ≈ reference(0.4, 2, [0.2, 0.7])
    padded = jax_vec_parity(jax_vec_padded_matrix, (:x, :rows),
        [([[0.2, 0.7], [0.3], [0.1, -0.3, 0.5]],), ([[1.0], [2.0, 3.0]],)])
    @test occursin("stable=True", string(padded))
    refused = try
        jax_source(jax_reshape_padded_matrix; have = (:x, :rows),
                   data = (; rows = [[0.2, 0.7], [0.3]]), functions = (value = :value,))
    catch e
        e
    end
    @test refused isa JAXEmissionError
    @test occursin("`reshape` of a padded", sprint(showerror, refused))
end

end
