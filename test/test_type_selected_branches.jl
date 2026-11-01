module TypeSelectedBranchTests
using Test, ReactiveKernels
const RK = ReactiveKernels
const OS = RK.OutputSignatures

const calls = Ref(0)
fallback(cells) = (calls[] += 1; length.(cells))
effectful_condition(x) = (calls[] += 1; true)

@kernel collect_cells(cells) = if eltype(cells) <: Number
    cells
elseif eltype(cells) <: AbstractVector
    isempty(cells) ? eltype(eltype(cells))[] : reduce(vcat, cells)
else
    fallback(cells)
end
@kernel dynamic_branch(x, flag) = flag ? x : reverse(x)
@kernel effectful_branch(x) = effectful_condition(x) ? x : reverse(x)
@kernel invalid_arm(x) = eltype(x) <: Number ? x : x[999]
@kernel nested_branch(x) = eltype(x) <: AbstractVector ? x[1] :
    eltype(x) <: Number ? x : reverse(x)
@kernel selected_cells(x::Vector{Float64}) = begin
    values = invalid_arm(x)
    cells = plate(values) do value
        value^2
    end
    return cells
end
@kernel dynamic_cells(x::Vector{Float64}, flag::Bool) = begin
    values = dynamic_branch(x, flag)
    cells = plate(values) do value
        value^2
    end
    return cells
end
@kernel effectful_cells(x::Vector{Float64}) = begin
    values = effectful_branch(x)
    cells = plate(values) do value
        value^2
    end
    return cells
end

metadata(kernel, args...) = RK._metadata_call(kernel, OS.ValueInfo[args...];
    kwargs = Pair{Symbol,OS.ValueInfo}[]).value

@testset "type-selected branches carry only the selected arm's metadata" begin
    for n in (0, 1, 16, OS.SymbolicAxis(:rows))
        input = OS.vector_info(Float64, n)
        for kernel in (collect_cells, invalid_arm, nested_branch)
            info = metadata(kernel, input)
            @test OS.value_axes(info) == OS.value_axes(input)
            @test OS.type_bound(info) === Vector{Float64}
        end
    end
    @test OS.value_axes(metadata(collect_cells, OS.ValueInfo())) === nothing
    @test OS.value_axes(metadata(dynamic_branch, OS.vector_info(Float64, 4),
        OS.scalar_info(Bool))) === nothing
    before = calls[]
    @test OS.value_axes(metadata(effectful_branch, OS.vector_info(Float64, 4))) === nothing
    @test calls[] == before
end

@testset "type selection preserves vector, empty and fallback execution" begin
    f = prepare(collect_cells)
    for x in (Float64[], [0.2], [0.2, -0.7], Vector{Float64}[],
            [Float64[], [0.3, 0.7], Float64[], [-0.2]])
        saved = deepcopy(x)
        expected = eltype(x) <: Number ? x : isempty(x) ? Float64[] : reduce(vcat, x)
        @test f(x) == expected
        @test x == saved
    end
    before = calls[]
    @test f(["abc", "d"]) == [3, 1]
    @test calls[] == before + 1
    dynamic = prepare(dynamic_branch)
    @test dynamic([1., 2.], true) == [1., 2.]
    @test dynamic([1., 2.], false) == [2., 1.]
    @test prepare(invalid_arm)([1., 2.]) == [1., 2.]
    @test_throws BoundsError prepare(invalid_arm)([[1.], [2.]])
end

@testset "cell lowering preserves lazy and effectful conditions" begin
    selected = prepare(plate_cell(selected_cells, :cells; index=:i, locality=:error))
    @test selected([1., 2.], 2) == 4.
    @test_throws BoundsError selected(Float64[], 1)
    dynamic = prepare(plate_cell(dynamic_cells, :cells; index=:i, locality=:ignore))
    @test dynamic([1., 2.], true, 1) == 1.
    @test dynamic([1., 2.], false, 1) == 4.
    before = calls[]
    effectful = prepare(plate_cell(effectful_cells, :cells; index=:i, locality=:ignore))
    @test calls[] == before
    @test effectful([1., 2.], 2) == 4.
    @test calls[] == before + 1
end
end
