module PlateDirectReadsTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test
using ReactiveKernels: code_expr

# A plate's cells read each batched array that has exactly the plate's axes
# directly, under a check made once per call; an array that broadcasting
# extrudes (a singleton dimension) is read through Base's projection as before
# (`_plate_direct_read_loop`, snag plate-reverse-lo-31fd6869). Both loops
# compute the same values and gradients; which one runs depends only on the
# call's shapes, never on a shape seen at preparation.
@kernel direct_normal_total(mu::Vector{Float64}, s::Float64, y::Vector{Float64}) = begin
    pointwise = plate(y, mu) do yi, mi
        -0.5 * ((yi - mi) / s)^2 - log(s)
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel direct_normal_pointwise(mu::Vector{Float64}, s::Float64, y::Vector{Float64}) = begin
    pointwise = plate(y, mu) do yi, mi
        -0.5 * ((yi - mi) / s)^2 - log(s)
    end
    return pointwise
end

@kernel direct_one_total(s::Float64, y::Vector{Float64}) = begin
    pointwise = plate(y) do yi
        -0.5 * (yi / s)^2 - log(s)
    end
    total::Float64 = sum(pointwise)
    return total
end

@kernel direct_grid_total(a::Matrix{Float64}, b::Matrix{Float64}) = begin
    cells = plate(a, b) do ai, bi
        ai * bi - 0.25 * bi^2
    end
    total::Float64 = sum(cells)
    return total
end

normal_cells(mu, s, y) = @. -0.5 * ((y - mu) / s)^2 - log(s)
function normal_gradients(mu, s, y)
    r = (y .- mu) ./ s^2
    dmu = length(mu) == length(y) ? r : [sum(r)]
    ds = sum(@. (y - mu)^2 / s^3 - 1 / s)
    dmu, ds
end

@testset "plate cells read arrays of the plate's axes directly" begin
    y = collect(range(-1.0, 2.0; length = 37))
    mu = 0.3 .* y .+ 0.05 .* sin.(1:37)
    s = 0.7
    total = prepare(direct_normal_total)
    pointwise = prepare(direct_normal_pointwise)
    # The summed loop has its direct-read copy beside the loop as it is.
    program = string(code_expr(total))
    @test occursin("ReactiveKernels._plate_spans_axis", program)
    @test occursin("ReactiveKernels._plate_read", program)
    @test occursin("Base.Broadcast.preprocess", program)
    # Spanning operands: the direct loop.
    @test total(mu, s, y) ≈ sum(normal_cells(mu, s, y))
    @test pointwise(mu, s, y) ≈ normal_cells(mu, s, y)
    # A singleton location stretches over the plate: the extruded loop.
    one = [0.4]
    @test total(one, s, y) ≈ sum(normal_cells(one, s, y))
    @test pointwise(one, s, y) ≈ normal_cells(one, s, y)
    # A singleton response, stretched over the locations.
    @test total(mu, s, [0.2]) ≈ sum(normal_cells(mu, s, [0.2]))
    @test pointwise(mu, s, [0.2]) ≈ normal_cells(mu, s, [0.2])
    # Mismatched extents still throw.
    @test_throws DimensionMismatch total(mu[1:5], s, y)
    # An empty plate.
    @test total(Float64[], s, Float64[]) == 0.0
    @test pointwise(Float64[], s, Float64[]) == Float64[]
end

@testset "a plate over one batched array reads it directly in its one loop" begin
    # Its axes are the plate's, so broadcasting never extrudes it: no check
    # and no copy, and each cell recipe stays emitted once (snag
    # prepared-kernel-ae180d1a).
    y = collect(range(-1.0, 2.0; length = 41))
    s = 0.6
    one = prepare(direct_one_total)
    program = string(code_expr(one))
    @test occursin("ReactiveKernels._plate_read", program)
    @test !occursin("ReactiveKernels._plate_spans_axis", program)
    @test one(s, y) ≈ sum(@. -0.5 * (y / s)^2 - log(s))
    @test one(s, Float64[]) == 0.0
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    prepared = prepare_ad(one, backend, s, y; active = :s)
    value, ds = ad_value_and_gradient(prepared, s, y)
    @test ds ≈ sum(@. y^2 / s^3 - 1 / s)
end

@testset "a multi-dimensional plate over full and extruded arrays" begin
    a = reshape(collect(1.0:12.0), 3, 4) ./ 7
    b = reshape(collect(12.0:-1.0:1.0), 3, 4) ./ 5
    grid = prepare(direct_grid_total)
    cells(a, b) = @. a * b - 0.25 * b^2
    @test grid(a, b) ≈ sum(cells(a, b))
    row, column = b[1:1, :], b[:, 1:1]
    @test grid(a, row) ≈ sum(cells(a, row))
    @test grid(a, column) ≈ sum(cells(a, column))
end

@testset "native Reverse through the direct and the extruded loop" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    y = collect(range(-1.0, 2.0; length = 29))
    mu = 0.3 .* y .+ 0.05 .* cos.(1:29)
    s = 0.9
    for bound in (false, true)
        @testset "bound data = $bound" begin
            for location in (mu, [0.4])
                kernel = bound ? prepare(direct_normal_total; bound = (; y)) :
                    prepare(direct_normal_total)
                args = bound ? (location, s) : (location, s, y)
                prepared = prepare_ad(kernel, backend, args...; active = (:mu, :s))
                value, (dmu, ds) = ad_value_and_gradient(prepared, args...)
                expected_mu, expected_s = normal_gradients(location, s, y)
                @test value ≈ sum(normal_cells(location, s, y))
                @test dmu ≈ expected_mu
                @test ds ≈ expected_s
                # One preparation serves either shape: a call that stretches
                # the location after one that spans the plate, and back.
                other = length(location) == 1 ? mu : [0.4]
                other_args = bound ? (other, s) : (other, s, y)
                prepared_other = prepare_ad(kernel, backend, args...; active = (:mu, :s))
                value, (dmu, ds) = ad_value_and_gradient(prepared_other, other_args...)
                expected_mu, expected_s = normal_gradients(other, s, y)
                @test value ≈ sum(normal_cells(other, s, y))
                @test dmu ≈ expected_mu
                @test ds ≈ expected_s
            end
        end
    end
end

end
