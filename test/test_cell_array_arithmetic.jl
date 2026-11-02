module CellArrayArithmeticTests
using ReactiveKernels, Test

add_arrays(a, b) = a + b
reverse_subtract(a, b) = b - a
@kernel arrays(a::AbstractMatrix{Float64}, b::AbstractMatrix{Float64}) = begin
    c = add_arrays(a, b)
    density = plate(c) do x
        x^2
    end
end
@kernel subtract(a::Vector{Float64}, b::Vector{Float64}) = begin
    c = reverse_subtract(a, b)
    density = plate(c) do x
        x^3
    end
end
@kernel trailing_axis(a::Vector{Float64}, b::Matrix{Float64}) = begin
    c = a + b
    density = plate(c) do x
        x^2
    end
end

@testset "ordinary array arithmetic retains its full shape contract" begin
    spec = plate_cell(arrays, :density; index=:cell, locality=:error)
    k = prepare(spec; have=(:a, :b, :cell), want=:density_cell)
    a, b = reshape(collect(1.0:6.0), 2, 3), fill(0.25, 2, 3)
    @test [k(a, b, i) for i in eachindex(a)] == vec((a + b).^2)
    @test isempty(ReactiveKernels._code_findings(k, (typeof(a), typeof(b), Int);
        recurse=ReactiveKernels._rk_or_generated(@__MODULE__)))
    # Broadcasting would expand this singleton axis. Ordinary + must throw,
    # even though the selected row itself is in bounds in both operands.
    @test_throws DimensionMismatch a + b[:, 1:1]
    @test_throws DimensionMismatch k(a, b[:, 1:1], 1)
    @test_throws BoundsError k(a, b, 7)
    minus = prepare(plate_cell(subtract, :density; index=:cell, locality=:error);
        have=(:a, :b, :cell), want=:density_cell)
    @test minus([1.0, 2.0], [4.0, 6.0], 2) == 64.0
    trailing = prepare(plate_cell(trailing_axis, :density; index=:cell, locality=:error);
        have=(:a, :b, :cell), want=:density_cell)
    @test trailing([1.0, 2.0], reshape([3.0, 4.0], 2, 1), 2) == 36.0
end

# A concrete array can choose different arithmetic from Base's row law.
struct ShiftedMatrix <: AbstractMatrix{Float64}
    values::Matrix{Float64}
end
Base.size(a::ShiftedMatrix) = size(a.values)
Base.getindex(a::ShiftedMatrix, i::Int, j::Int) = a.values[i,j]
Base.:+(a::ShiftedMatrix, b::ShiftedMatrix) = a.values + b.values .+ 10

dispatched(a, b) = a + b
dispatched(a::Matrix, b::Matrix) = a + b .+ 10
@kernel custom_helper(a::AbstractMatrix{Float64}, b::AbstractMatrix{Float64}) = begin
    c = dispatched(a, b)
    density = plate(c) do x
        x^2
    end
end
mutable_add = +
mutable_forward(a, b) = mutable_add(a, b)
@kernel mutable_helper(a::Matrix{Float64}, b::Matrix{Float64}) = begin
    c = mutable_forward(a, b)
    density = plate(c) do x
        x^2
    end
end

@testset "local array arithmetic does not change custom dispatch" begin
    a, b = ShiftedMatrix(fill(1.0, 2, 2)), ShiftedMatrix(fill(2.0, 2, 2))
    whole = prepare(arrays; want=:density)
    @test whole(a, b) == fill(169.0, 2, 2)
    localcell = prepare(plate_cell(arrays, :density; index=:cell, locality=:error);
        have=(:a, :b, :cell), want=:density_cell)
    @test_throws ArgumentError localcell(a, b, 1)
    spec = plate_cell(custom_helper, :density; index=:cell, locality=:error)
    @test_throws PartitionLocalityError prepare(spec;
        have=(:a, :b, :cell), want=:density_cell)
    mutable_spec = plate_cell(mutable_helper, :density; index=:cell, locality=:error)
    @test_throws PartitionLocalityError prepare(mutable_spec;
        have=(:a, :b, :cell), want=:density_cell)
end
end
