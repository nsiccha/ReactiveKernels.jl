module CellWindowRowTests
using ReactiveKernels, Enzyme, Test
using DifferentiationInterface

const Window = ReactiveKernels._CellWindow

row_sum(w, indices) = sum(i -> sum(abs2, ReactiveKernels._lane_gathered(w, i, :)),
    indices; init=0.0)
function row_objective(data, rows, indices)
    w = Window(data, rows, (Base.OneTo(10), axes(data, 2)), true)
    row_sum(w, indices)
end

@testset "cell window row reads preserve axes, bounds and derivatives" begin
    data = reshape(collect(0.1:0.1:1.2), 3, 4)
    rows = [2, 5, 9]
    w = Window(data, rows, (Base.OneTo(10), axes(data, 2)), true)
    for (k, row) in enumerate(rows)
        @test w[row, :] == data[k, :]
        @test w[row, :] isa Vector{Float64}
        @test !Base.mightalias(w[row, :], data)
        @test ReactiveKernels._lane_gathered(w, row, :) == data[k, :]
        @test axes(w[row, :]) == axes(data[k, :])
        @test w[row, 3] == data[k, 3]
    end
    @test_throws BoundsError w[4, :]
    @test_throws BoundsError ReactiveKernels._lane_gathered(w, 4, :)
    @test_throws BoundsError w[2, 5]
    one = Window(data[2:2, :], 5, axes(w), false)
    @test one[5, :] == data[2, :]
    @test_throws BoundsError one[2, :]
    cube = reshape(collect(1.0:24.0), 3, 4, 2)
    cw = Window(cube, rows, (Base.OneTo(10), axes(cube, 2), axes(cube, 3)), true)
    @test cw[5, :, :] == cube[2, :, :]
    @test cw[5, 2, :] == cube[2, 2, :]
    @test cw[5, :, 2] == cube[2, :, 2]
    @test cw[5, 2, 2] == cube[2, 2, 2]

    indices = [9, 2, 9, 5]
    gradient = zero(data)
    original = copy(data)
    Enzyme.autodiff(Enzyme.Reverse, row_objective, Enzyme.Active,
        Enzyme.Duplicated(data, gradient), Enzyme.Const(rows), Enzyme.Const(indices))
    @test gradient ≈ 2 .* data .* [1, 1, 2]
    @test data == original
    @test row_objective(data, rows, indices) ≈ sum(abs2, data[[3, 1, 3, 2], :])
end

@testset "repeated read-only cell rows do not allocate row copies" begin
    data = reshape(collect(0.1:0.1:1.2), 3, 4)
    w = Window(data, [2, 5, 9], (Base.OneTo(10), axes(data, 2)), true)
    short, long = fill(5, 2), fill(5, 2000)
    row_sum(w, short); row_sum(w, long)
    @test (@allocated row_sum(w, long)) <= (@allocated row_sum(w, short)) + 256
end

@kernel broadcast_rows(x::Matrix{Float64}, indices::Vector{Int}, scale::Float64) = begin
    scaled = scale .* x
    terms = plate(indices) do i
        row = scaled[i, :] .* scaled[i, :]
        row[1]
    end
    value = sum(terms)
    return value
end

@testset "prepared cell broadcasts borrow their row operands" begin
    data = reshape(collect(0.1:0.1:1.2), 3, 4)
    indices = [3, 1, 3, 2]
    spec = plate_cell(broadcast_rows, :terms; index=:cell, locality=:error)
    kernel = prepare(spec; have=(:x, :indices, :scale, :cell), want=:terms_cell,
        bound=(; x=data, indices), exemplars=(0.7, 1))
    @test occursin("_lane_gathered", string(code_expr(kernel)))
    ad = prepare_ad(kernel, AutoEnzyme(; mode=Enzyme.Reverse), 0.7, 1; active=:scale)
    for cell in eachindex(indices), scale in (-0.4, 0.7)
        expected = abs2(data[indices[cell], 1])
        value, gradient = ad_value_and_gradient(ad, scale, cell)
        @test value ≈ scale^2 * expected
        @test gradient ≈ 2scale * expected
    end
end

@kernel planned_columns(x::Vector{Float64}, group::Vector{Int},
                        w::Vector{Float64}, z::Vector{Float64},
                        rows::Vector{Int}, subject::Vector{Int}) = begin
    mu = x[group] .* w
    matrix = hcat(w, mu, z)
    terms = plate(rows) do i
        sum(abs2, matrix[i, :])
    end
end

@testset "planned row operands retain only the selected dense storage" begin
    RK = ReactiveKernels
    for data in (collect(0.1:0.1:1.2), reshape(collect(0.1:0.1:1.2), 4, 3)),
            rows in ([3, 1, 3], Int[], 2)
        borrowed = RK._cell_rows_view(data, rows)
        planned = RK._cell_plan_rows(data, rows)
        @test typeof(planned) === typeof(borrowed)
        @test axes(planned) == axes(borrowed)
        @test planned == borrowed
        @test length(parent(planned)) == length(planned)
        @test !Base.mightalias(planned, data)
    end
    allocations = Int[]
    for n in (6, 6000)
        data = (; group=repeat([1, 2], n ÷ 2), w=sin.(1:n), z=cos.(1:n),
            rows=collect(1:n), subject=[1, 1, fill(2, n-2)...])
        x = [0.2, -0.3]
        spec = plate_partition(planned_columns, :terms;
            by=:subject, index=:label, plan=:plan, locality=:error)
        kernel = prepare(spec; have=(:x, :group, :w, :z, :rows, :subject, :label, :plan),
            want=:partition_total, bound=data, exemplars=(x, 1, nothing))
        plan = partition_plan(kernel, x, 1, nothing)
        ad = prepare_ad(kernel, AutoEnzyme(; mode=Enzyme.Reverse), x, 1, plan; active=:x)
        gradient = zero(x)
        for scale in (1.0, -0.7)
            point = scale .* x
            value, _ = ad_value_and_gradient!(ad, gradient, point, 1, plan)
            expected = sum(abs2, data.w[1:2]) + sum(abs2, point .* data.w[1:2]) +
                sum(abs2, data.z[1:2])
            @test value ≈ expected ≈ kernel(point, 1, plan)
            @test gradient ≈ 2 .* point .* data.w[1:2].^2
        end
        push!(allocations, @allocated ad_value_and_gradient!(ad, gradient, x, 1, plan))
    end
    @test allocations[2] <= allocations[1] + 1024
    @test allocations[2] < 6000sizeof(Float64)
end
end
