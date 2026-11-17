# Consumer reproduction for snag jax-mapped-recor-9ae50e32: plate records
# with a numerical vector field, mapped out by `@traceable` helpers that call
# `first`, `similar` and loops over the mapped rows. Run in an environment
# containing this RK checkout and Enzyme, with RK_JAX_PYTHON pointing to
# Python/JAX. RK 0.1.295 raised `JAXEmissionError: a ragged array without its
# lengths` for both spellings; the native value and ordinary reverse gradient
# are the reference. The permanent compiler regression is
# test/test_jax_mapped_records.jl.
using ReactiveKernels, Enzyme, Test
include(joinpath(@__DIR__, "..", "test", "jax_python.jl"))

@traceable function sum_columns(values)
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

@traceable function select_column(values::AbstractArray{<:NamedTuple{names}}) where {names}
    name = first(names)
    column = map(values) do value
        getproperty(value, name)
    end
    return sum_columns(column)
end

@traceable function mapped_nested(records)
    values = map(vec(records)) do record
        record.nested
    end
    return select_column(values)
end

@traceable function mapped_direct(records)
    column = map(vec(records)) do record
        record.nested.y
    end
    return sum_columns(column)
end

@kernel nested(x::Float64, indices::Vector{Int}, y::Vector{Float64}) = begin
    records = plate(indices, x, Ref(y)) do i, s, ys
        (; nested = (; y = ys .* (s + i)))
    end
    value = mapped_nested(records)
    return value
end

@kernel direct(x::Float64, indices::Vector{Int}, y::Vector{Float64}) = begin
    records = plate(indices, x, Ref(y)) do i, s, ys
        (; nested = (; y = ys .* (s + i)))
    end
    value = mapped_direct(records)
    return value
end

@testset "mapped record field lengths: native and JAX" begin
    @test rk_jax_available()
    datasets = [([1, 2, 3], [0.2, 0.7]), ([4, 1], [0.3, -0.1, 1.5, 2.0]), ([2, 5, 1], Float64[])]
    for (name, spec) in (("nested", nested), ("direct", direct))
        native = prepare(spec; have = (:x, :indices, :y), want = :value)
        @test native(0.4, datasets[1]...) ≈ 6.48
        src = jax_source(spec; types = (; x = Float64, indices = Vector{Int}, y = Vector{Float64}),
                         data = (; indices = datasets[1][1], y = datasets[1][2]),
                         functions = (; value = :value))
        lines = count(==('\n'), string(src))
        for (indices, y) in datasets, x in (0.4, -1.3)
            got = rk_jax_evaluate(src, :value, (x, indices, y); grad = :joint, prepare = true)
            derivative = Enzyme.autodiff(Enzyme.Reverse, native, Enzyme.Active, Enzyme.Active(x),
                                         Enzyme.Const(indices), Enzyme.Const(y))[1][1]
            @test got.value ≈ native(x, indices, y) atol = 1e-10
            @test got.grad ≈ derivative atol = 1e-10
            println("MAPPED_LENGTHS ", name, " n=", length(indices), " m=", length(y), " x=", x,
                    " native=", native(x, indices, y), " reverse=", derivative,
                    " jax=", got.value, " jax_grad=", got.grad, " module_lines=", lines)
        end
    end
end
