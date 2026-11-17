module JAXMappedRecordTests
# Plate records mapped through `@traceable` helpers keep their field arrays'
# shapes and lengths in `jax_source`: `map` do-blocks over plate records,
# ragged rows and earlier maps, `first`/`last`/indexing/`similar` of the
# mapped rows, `for x in xs` over them, generators and `sum` of rows. Each
# value and ordinary reverse gradient is compared with the native kernel's.
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

@traceable function mapped_sum_columns(values)
    result = similar(first(values), Float64)
    for i in eachindex(result)
        total = 0.0
        for value in values
            total += value[i]
        end
        result[i] = total
    end
    return sum(result)
end
@traceable function mapped_select_column(values::AbstractArray{<:NamedTuple{names}}) where {names}
    name = first(names)
    column = map(values) do value
        getproperty(value, name)
    end
    return mapped_sum_columns(column)
end
# The record itself, then its field by a structural name.
@traceable function mapped_nested(records)
    values = map(vec(records)) do record
        record.nested
    end
    return mapped_select_column(values)
end
# The field read directly.
@traceable function mapped_direct(records)
    column = map(vec(records)) do record
        record.nested.y
    end
    return mapped_sum_columns(column)
end
# Rows read whole, from either end, by generators and summed as arrays.
@traceable function mapped_row_reads(records)
    rows = map(vec(records)) do record
        record.nested.y
    end
    totals = sum(rows)
    return totals[1] + 2 * totals[end] + sum(last(rows)) + sum(rows[1]) +
           sum(sum(r) for r in rows)
end
# A loop over the rows: `continue`, and an outer binding the loop's own
# variable leaves unchanged, as Julia's loop variable is local.
@traceable function mapped_row_loop(records)
    rows = map(vec(records)) do record
        record.nested.y
    end
    total = 0.0
    value = 10.0
    for value in rows
        isempty(value) && continue
        total += value[1] * value[end]
    end
    return total + value
end
# A loop over the rows that exits early.
@traceable function mapped_row_break(records)
    rows = map(vec(records)) do record
        record.nested.y
    end
    total = 0.0
    for value in rows
        total > 1.0 && break
        total += sum(value)
    end
    return total
end
# Per-record sums of a padded field: each record keeps its own length.
@traceable function mapped_field_sums(records)
    totals = map(vec(records)) do record
        sum(record.nested.y)
    end
    return sum(totals)
end
# Records with a padded field, mapped twice.
@traceable function mapped_record_sums(records)
    values = map(vec(records)) do record
        record.nested
    end
    totals = map(values) do value
        sum(value.y) + length(value.y)
    end
    return sum(totals)
end
# Padded rows, then reduced by a second map.
@traceable function mapped_row_sums(records)
    rows = map(vec(records)) do record
        record.nested.y
    end
    totals = map(rows) do row
        sum(row)
    end
    return sum(totals)
end
# A map over a ragged vector of vectors computed on the device.
@traceable function mapped_ragged(rows)
    totals = map(rows) do row
        sum(abs2, row)
    end
    return sum(totals)
end

for (name, helper) in ((:jax_mapped_nested, :mapped_nested), (:jax_mapped_direct, :mapped_direct),
                       (:jax_mapped_row_reads, :mapped_row_reads), (:jax_mapped_row_loop, :mapped_row_loop),
                       (:jax_mapped_row_break, :mapped_row_break))
    @eval @kernel $name(x::Float64, indices::Vector{Int}, y::Vector{Float64}) = begin
        records = plate(indices, x, Ref(y)) do i, s, ys
            (; nested = (; y = ys .* (s + i)))
        end
        value = $helper(records)
        return value
    end
end
for (name, helper) in ((:jax_mapped_padded_nested, :mapped_nested),
                       (:jax_mapped_field_sums, :mapped_field_sums),
                       (:jax_mapped_record_sums, :mapped_record_sums),
                       (:jax_mapped_row_sums, :mapped_row_sums))
    @eval @kernel $name(x::Float64, rows::Vector{Vector{Float64}}) = begin
        records = plate(rows, x) do row, s
            (; nested = (; y = row .* s))
        end
        value = $helper(records)
        return value
    end
end
@kernel jax_mapped_ragged(x::Float64, rows::Vector{Vector{Float64}}) = begin
    scaled = plate(rows, x) do row, s
        row .* s
    end
    value = mapped_ragged(scaled)
    return value
end
# A plate's vector cells summed as arrays.
@kernel jax_mapped_cell_sum(x::Float64, indices::Vector{Int}, y::Vector{Float64}) = begin
    cells = plate(indices, x, Ref(y)) do i, s, ys
        ys .* (s + i)
    end
    totals = sum(cells)
    value = totals[1] + 2 * totals[end]
    return value
end

function jax_mapped_parity(spec, have, datasets)
    data(d) = NamedTuple{Base.tail(have)}(d)
    native = prepare(spec; have, want = :value)
    src = jax_source(spec; have, data = data(first(datasets)), functions = (value = :value,))
    # One module for every dataset: lengths are runtime shapes, and nothing
    # is unrolled over the data.
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

@testset "jax_source: mapped plate records" begin
    dense = [([1, 2, 3], [0.2, 0.7]), ([4, 1], [0.3, -0.1, 1.5, 2.0]), ([2, 5, 1], Float64[])]
    for spec in (jax_mapped_nested, jax_mapped_direct)
        jax_mapped_parity(spec, (:x, :indices, :y), dense)
    end
    nonempty = [([1, 2, 3], [0.2, 0.7]), ([4, 1], [0.3, -0.1, 1.5, 2.0]), ([2, 2], [0.5])]
    jax_mapped_parity(jax_mapped_row_reads, (:x, :indices, :y), nonempty)
    jax_mapped_parity(jax_mapped_row_loop, (:x, :indices, :y), [nonempty; ([1, 2], Float64[])])
    jax_mapped_parity(jax_mapped_row_break, (:x, :indices, :y), [nonempty; ([1, 2], Float64[])])
    jax_mapped_parity(jax_mapped_cell_sum, (:x, :indices, :y), nonempty)
    # Rows of one length per dataset, padded as ragged data.
    jax_mapped_parity(jax_mapped_padded_nested, (:x, :rows),
        [([[0.2, 0.7], [0.1, -0.3]],), ([[0.5], [0.25], [2.0]],), ([[1.0, 2.0, 3.0]],)])
    ragged = [([[0.2, 0.7], Float64[], [0.3]],), ([[0.1], [0.2, 0.5, 0.9]],),
              ([Float64[], Float64[]],), (Vector{Float64}[],)]
    for spec in (jax_mapped_field_sums, jax_mapped_record_sums, jax_mapped_row_sums, jax_mapped_ragged)
        jax_mapped_parity(spec, (:x, :rows), ragged)
    end
end

end
