module CellFillTests
using Test, ReactiveKernels, DifferentiationInterface, Enzyme

@kernel filled_columns(n::Int, m::Int, scale::Float64) = begin
    table = fill(scale, n, m)
    columns = plate(1:m) do j
        cells = plate(1:n) do i
            abs2(table[i, j])
        end
        sum(cells)
    end
    return sum(columns)
end
@kernel filled_groups(groups::Vector{Vector{Float64}}, scale::Float64) = begin
    values = plate(groups) do group
        table = hcat(fill(scale, length(group)), group)
        cells = plate(eachindex(group)) do i
            abs2(table[i, 1]) + abs2(table[i, 2])
        end
        sum(cells)
    end
    return sum(values)
end
@kernel filled_index(n::Int, scale::Float64, index) = begin
    table = fill(scale, n)
    return table[index]
end
@kernel filled_empty_reader(n::Int, scale::Float64, data::Vector{Float64}) = begin
    table = fill(scale, n)
    cells = plate(data) do x
        table[1] + x
    end
    return sum(cells)
end

function structure(ex)
    ex isa Expr || return (0, 0)
    result = (1, Int(ex.head in (:for, :while)))
    for child in ex.args
        result = result .+ structure(child)
    end
    result
end

@testset "nonescaping fills preserve shapes, indexing and ownership" begin
    RK = ReactiveKernels
    for T in (Float32, Float64), dims in ((), (0,), (3,), (2, 3), (1, 2, 1))
        dense = fill(T(0.25), dims...)
        filled = RK._cell_array_fill(T(0.25), dims...)
        @test size(filled) == size(dense)
        @test eltype(filled) === T
        @test collect(filled) == dense
        for i in eachindex(dense)
            @test filled[i] === dense[i]
        end
        @test filled[:] == dense[:]
    end
    for dims in ((-1,), (0, -1), (-1, 0), (typemax(Int), 2))
        original = try fill(0.25, dims...); nothing catch err; err end
        composed = try RK._cell_array_fill(0.25, dims...); nothing catch err; err end
        @test original !== nothing
        @test typeof(composed) === typeof(original)
    end
    index = prepare(filled_index)
    @test index(4, 0.25, [4, 1, 4]) == fill(0.25, 3)
    @test index(4, 0.25, 2:3) == fill(0.25, 2)
    @test_throws BoundsError index(4, 0.25, 0)
    @test_throws BoundsError index(4, 0.25, 5)
    @test_throws BoundsError index(0, 0.25, 1)
    returned = prepare(filled_index; have=(:n, :scale), want=:table)
    a, b = returned(4, 0.25), returned(4, 0.25)
    @test a isa Vector{Float64}
    a[1] = 9.0
    @test b == fill(0.25, 4)
    @test returned(4, 0.25) == b
    # Fill's reference-valued elements retain Julia's shared-element identity.
    ref = Ref(1.0)
    refs = RK._cell_array_fill(ref, 3)
    @test refs isa Vector{typeof(ref)}
    @test all(x -> x === ref, refs)
    empty_reader = prepare(filled_empty_reader)
    @test empty_reader(0, 0.25, Float64[]) == 0.0
    @test_throws ArgumentError empty_reader(-1, 0.25, Float64[])
end

@testset "filled cells retain loops and ordinary Reverse" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    shapes = Tuple{Int,Int}[]
    for (n, m) in ((2, 3), (29, 5), (0, 3), (2, 0))
        k = prepare(filled_columns; bound=(; n, m), type_stability=:error)
        @test k(0.25) ≈ n * m * 0.25^2
        ad = prepare_ad(k, backend, 0.25; active=:scale)
        value, gradient = ad_value_and_gradient(ad, 0.25)
        @test value ≈ n * m * 0.25^2
        @test gradient ≈ 2n * m * 0.25
        n > 0 && m > 0 && push!(shapes, structure(code_expr(k)))
    end
    @test shapes[1] == shapes[2]
    @test last(first(shapes)) >= 2
    groups = [[0.2, -0.4], Float64[], [0.7], [1.0, 0.5, -0.3]]
    before = deepcopy(groups)
    k = prepare(filled_groups; bound=(; groups), type_stability=:error)
    expected(s) = sum(length(g) * s^2 + sum(abs2, g; init=0.0) for g in groups)
    ad = prepare_ad(k, backend, 0.25; active=:scale)
    for scale in (0.25, -0.5)
        value, gradient = ad_value_and_gradient(ad, scale)
        @test value ≈ expected(scale)
        @test gradient ≈ 2scale * sum(length, groups)
    end
    @test groups == before
    @test occursin("_cell_array_fill", string(code_expr(k)))
    bytes = map((3, 60)) do n
        prepared = prepare(filled_groups; bound=(; groups=[ones(4) for _ in 1:n]))
        prepared(0.25)
        @allocated prepared(0.25)
    end
    @test bytes[1] == bytes[2]
end
end
