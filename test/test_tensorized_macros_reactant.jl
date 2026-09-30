module TensorizedMacroReactantTests

using ReactiveKernels, Reactant, Test

# A recipe written with `@.` compiles to the program of its explicit-dot
# spelling: scalar and position-batched, with and without an indexed read of
# the broadcast operand, and elementwise over a matrix.

@kernel _cfb_indexed(position::Vector{Float64}, times::Vector{Float64}) = begin
    trajectory::Vector{Float64} = position[1] .* exp.(-position[2] .* times) .+ 1.0
    cfb::Vector{Float64} = @.(100 * (trajectory - trajectory[1]) / trajectory[1])
    result = (; trajectory, cfb)
    return result
end

@kernel _cfb_scalar(position::Vector{Float64}, times::Vector{Float64}) = begin
    trajectory::Vector{Float64} = position[1] .* exp.(-position[2] .* times) .+ 1.0
    baseline::Float64 = position[1] + 1.0
    cfb::Vector{Float64} = @.(100 * (trajectory - baseline) / baseline)
    result = (; trajectory, cfb)
    return result
end

@kernel _cfb_dots(position::Vector{Float64}, times::Vector{Float64}) = begin
    trajectory::Vector{Float64} = position[1] .* exp.(-position[2] .* times) .+ 1.0
    cfb::Vector{Float64} = 100 .* (trajectory .- trajectory[1]) ./ trajectory[1]
    result = (; trajectory, cfb)
    return result
end

@kernel _elementwise_product(m::Matrix{Float64}) = begin
    out::Matrix{Float64} = @.(m * m)
    return out
end

@kernel _viewed_head(v::Vector{Float64}) = begin
    out::Float64 = sum(@view v[1:2])
    return out
end

const POSITIONS = [1.0 2.0 3.0; 0.5 0.25 0.125]
const LATER = [4.0 0.5 1.5; 0.1 0.9 0.3]
const TIMES = [0.0, 0.5, 1.0, 2.0]

function operation_names(spec, count)
    batch = prepare_batched(spec; batched = :position, want = :result)
    positions = Reactant.to_rarray(vcat(fill(2.0, 1, count), fill(0.25, 1, count)))
    hlo = repr(Reactant.@code_hlo batch(positions, TIMES))
    [m.match for m in eachmatch(r"(?:stablehlo|enzyme)\.[a-z_]+", hlo)]
end

@testset "scalar kernels with a broadcast macro" begin
    for spec in (_cfb_indexed, _cfb_scalar)
        kernel = prepare(spec; want = :result)
        position = POSITIONS[:, 1]
        traced = Reactant.to_rarray(position)
        compiled = Reactant.@compile kernel(traced, TIMES)
        result = compiled(traced, TIMES)
        expected = kernel(position, TIMES)
        @test Array(result.trajectory) ≈ expected.trajectory
        @test Array(result.cfb) ≈ expected.cfb
    end
end

@testset "position batches with a broadcast macro" begin
    for spec in (_cfb_indexed, _cfb_scalar)
        batch = prepare_batched(spec; batched = :position, want = :result)
        traced = Reactant.to_rarray(POSITIONS)
        compiled = Reactant.@compile batch(traced, TIMES)
        result = compiled(traced, TIMES)
        expected = batch(POSITIONS, TIMES)
        first_cfb = Array(result.cfb)
        @test size(first_cfb) == (length(TIMES), size(POSITIONS, 2))
        @test first_cfb ≈ expected.cfb
        @test Array(result.trajectory) ≈ expected.trajectory
        # The executable is reused with changed inputs; the first output is
        # retained and the inputs are unchanged.
        later = compiled(Reactant.to_rarray(LATER), TIMES)
        @test Array(later.cfb) ≈ batch(LATER, TIMES).cfb
        @test first_cfb ≈ expected.cfb
        @test Array(traced) == POSITIONS
    end
end

@testset "the macro spelling emits the explicit-dot program" begin
    macro_form = operation_names(_cfb_indexed, 3)
    @test !isempty(macro_form)
    @test macro_form == operation_names(_cfb_dots, 3)
    # Position count is data: one retained loop, the same operations.
    @test macro_form == operation_names(_cfb_indexed, 23)
    @test count(==("stablehlo.while"), macro_form) == 1
end

@testset "a broadcast macro stays elementwise over a matrix" begin
    m = [1.0 2.0; 3.0 4.0]
    kernel = prepare(_elementwise_product)
    traced = Reactant.to_rarray(m)
    compiled = Reactant.@compile kernel(traced)
    @test Array(compiled(traced)) ≈ m .* m
    @test !(Array(compiled(traced)) ≈ m * m)
    @test Array(traced) == m
end

@testset "a view macro in a recipe" begin
    v = [2.0, 3.0, 5.0]
    kernel = prepare(_viewed_head)
    traced = Reactant.to_rarray(v)
    compiled = Reactant.@compile kernel(traced)
    @test compiled(traced) ≈ kernel(v)
    @test Array(traced) == v
end

end # module
