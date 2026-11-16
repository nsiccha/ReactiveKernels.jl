module JAXValuePortTests
using ReactiveKernels, Test, Enzyme
include("jax_python.jl")

# A branch may return a ragged row alone, or inside an authored tuple. The
# logical length remains a value even though JAX uses padded physical arrays.
@traceable function row_choice(a, row)
    selected, scale = if a > 0
        (a .* row, a)
    else
        (2a .* row, -a)
    end
    sum(selected) + scale * length(selected)
end
@kernel chosen_rows(a::Float64, rows::Vector{Vector{Float64}}) = begin
    totals = plate(rows, Ref(a)) do row, a
        row_choice(a, row)
    end
    total = sum(totals)
    return total
end
function native_rows(a, rows)
    total = 0.0
    for row in rows
        total += row_choice(a, row)
    end
    total
end

@traceable function bounded_range(a, row, n)
    indices = 0:min(n, length(row) - 1)
    a * (sum(indices) + length(indices))
end
@kernel bounded_rows(a::Float64, rows::Vector{Vector{Float64}}, n::Int) = begin
    totals = plate(rows, Ref(a), Ref(n)) do row, a, n
        bounded_range(a, row, n)
    end
    total = sum(totals)
    return total
end

@traceable function matrix_row(a, row)
    matrix = Matrix{Float64}(undef, length(row), 2)
    for i in eachindex(row)
        matrix[i, 1] = a * row[i]
        matrix[i, 2] = a * row[i]^2
    end
    sum(matrix) + sum(ones(length(row), 2)) + sum(fill(a, length(row), 2))
end
@kernel matrix_rows(a::Float64, rows::Vector{Vector{Float64}}) = begin
    totals = plate(rows, Ref(a)) do row, a
        matrix_row(a, row)
    end
    total = sum(totals)
    return total
end

@testset "conditional value ports retain ragged lengths" begin
    src = jax_source(chosen_rows; have=(:a, :rows), functions=(total=:total,))
    @test string(jax_source(chosen_rows; have=(:a, :rows), want=:total,
        data=(rows=[ones(3)],))) == string(jax_source(chosen_rows;
        have=(:a, :rows), want=:total, data=(rows=[ones(129)],)))
    if rk_jax_available()
        for a in (-0.7, 0.7), rows in (Vector{Float64}[], [Float64[]],
                [Float64[], [0.2], [0.3, -0.7, 0.5]], [ones(129)])
            derivative = Enzyme.autodiff(Enzyme.Reverse, native_rows, Enzyme.Active,
                Enzyme.Active(a), Enzyme.Const(rows))[1][1]
            got = rk_jax_evaluate(src, :total, (a, rows); grad=:joint)
            @test got.value ≈ native_rows(a, rows)
            @test got.joint_value ≈ got.value
            @test got.grad ≈ derivative
        end
        ranges = jax_source(bounded_rows; have=(:a, :rows, :n), functions=(total=:total,))
        rows = [Float64[], [0.2], [0.3, -0.7, 0.5], ones(129)]
        for n in (-3, 0, 2, 128)
            derivative = sum(bounded_range(1.0, row, n) for row in rows)
            got = rk_jax_evaluate(ranges, :total, (0.7, rows, n); grad=:joint)
            @test got.value ≈ 0.7 * derivative
            @test got.joint_value ≈ got.value
            @test got.grad ≈ derivative
        end
        matrices = jax_source(matrix_rows; have=(:a, :rows), functions=(total=:total,))
        for rows in (Vector{Float64}[], [Float64[]], [Float64[], [0.2], [0.3, -0.7, 0.5]])
            derivative = sum(sum(row) + sum(abs2, row) + 2length(row) for row in rows; init=0.0)
            value = sum(matrix_row(0.7, row) for row in rows; init=0.0)
            got = rk_jax_evaluate(matrices, :total, (0.7, rows); grad=:joint)
            @test got.value ≈ value
            @test got.joint_value ≈ value
            @test got.grad ≈ derivative
        end
    end
end
end
