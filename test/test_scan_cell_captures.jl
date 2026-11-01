module ScanCellCaptureTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

@kernel recurrent(z::Vector{Float64}, gain::Float64, n::Int,
        group::Vector{Int}, rows::Vector{Vector{Int}},
        positions::Vector{Int}, steps::Vector{Int}) = begin
    values = reshape(z, n, 1)
    scale = exp.(gain .* values[group])
    cells = plate(positions) do p
        lookup = vec(rows)
        path = scan(steps; init = 0.0) do carry, t
            next = carry + (scale[lookup[p]])[t]
            (next, next)
        end
        sum(path)
    end
    return cells
end

# The scan's iterated element must not be mistaken for an unchanged outer
# cell index. This program actually reads a different subject at each step.
@kernel cross_subject(z::Vector{Float64}, n::Int,
        positions::Vector{Int}, steps::Vector{Int}) = begin
    values = reshape(z, n, 1)
    scale = exp.(values)
    cells = plate(positions) do p
        path = scan(steps; init = 0.0) do carry, t
            next = carry + scale[t] + p
            (next, next)
        end
        sum(path)
    end
    return cells
end

@kernel with_history(z::Vector{Float64}, n::Int, positions::Vector{Int},
        steps::Vector{Int}) = begin
    values = reshape(z, n, 1)
    scale = exp.(values)
    cells = plate(positions) do p
        path = scan(steps; init = 0.0, history = -1.0) do carry, t, earlier
            next = (t == 1 ? carry : earlier[t - 1]) + scale[p]
            (next, next)
        end
        sum(path)
    end
    return cells
end

astsize(x::Expr) = 1 + sum(astsize, x.args; init = 0)
astsize(x) = 0

# Registered Enzyme cannot always classify a cell window's returned integer
# rows beside its active data (Enzyme dogfood repair f9c2eee). Acceptance on
# that repair requires ordinary static Reverse, as the other cell tests do.
function static_reverse(f)
    try
        f()
    catch err
        err isa Enzyme.Compiler.EnzymeRuntimeActivityError || rethrow()
        get(ENV, "RK_TEST_REQUIRE_STATIC_ENZYME", "0") == "1" && rethrow()
        @test_broken false
    end
end

@testset "scan captures retain the outer cell's rows" begin
    allocated, sizes = Int[], Int[]
    for n in (3, 3000)
        data = (; n, group = repeat(1:n, 2),
            rows = [[i, n + i] for i in 1:n], positions = collect(1:n), steps = [1, 2])
        saved = deepcopy(data)
        z = sin.(1:n) ./ 10
        spec = plate_cell(recurrent, :cells; index = :i, locality = :error)
        cell = prepare(spec; have = (:z, :gain, keys(data)..., :i), want = :cells_cell, bound = data)
        full = prepare(recurrent; have = (:z, :gain, keys(data)...), bound = data)
        for gain in (0.0, 0.7), i in (1, 2, n)
            @test cell(z, gain, i) ≈ 3exp(gain * z[i])
            @test partition_footprint(cell, z, gain, i).z == [i]
        end
        @test [cell(z, 0.7, i) for i in 1:n] ≈ full(z, 0.7)
        @test all(e.status === :rows for e in partition_locality(cell)
            if e.value === :scale)
        cell(z, 0.7, 2)
        push!(allocated, @allocated cell(z, 0.7, 2))
        push!(sizes, astsize(code_expr(cell)))
        static_reverse() do
            ad = prepare_ad(cell, AutoEnzyme(; mode = Enzyme.Reverse), z, 0.7, 2;
                active = :z)
            for gain in (0.0, 0.7)
                value, gradient = ad_value_and_gradient(ad, z, gain, 2)
                expected = zeros(n)
                expected[2] = 3gain * exp(gain * z[2])
                @test value ≈ cell(z, gain, 2)
                @test gradient ≈ expected
            end
        end
        @test data == saved
    end
    @test allocated[2] <= allocated[1] + 256
    @test sizes[2] == sizes[1]
end

@testset "scan capture proofs preserve inactive invalid reads" begin
    z = [0.1, -0.2, 0.3]
    for steps in (Int[], [1])
        data = (; n = 3, group = [1, 2, 3], rows = [[99], [2], [3]],
            positions = [1, 2, 3], steps)
        cell = prepare(plate_cell(recurrent, :cells; index = :i, locality = :error);
            have = (:z, :gain, keys(data)..., :i), want = :cells_cell, bound = data)
        full = prepare(recurrent; have = (:z, :gain, keys(data)...), bound = data)
        if isempty(steps)
            @test cell(z, 0.7, 1) == 0.0
            @test full(z, 0.7) == zeros(3)
        else
            @test_throws BoundsError cell(z, 0.7, 1)
            @test_throws BoundsError full(z, 0.7)
        end
    end
    data = (; n = 3, positions = [1, 2, 3], steps = [1, 2, 3])
    cell = prepare(plate_cell(cross_subject, :cells; index = :i, locality = :error);
        have = (:z, keys(data)..., :i), want = :cells_cell, bound = data)
    footprint = partition_footprint(cell, z, 1).z
    @test footprint === :whole || footprint == [1, 2, 3]
    @test cell(z, 1) ≈ sum(cumsum(exp.(z) .+ 1))
    unbound = prepare(plate_cell(recurrent, :cells; index = :i, locality = :error);
        have = (:z, :gain, :n, :group, :rows, :positions, :steps, :i), want = :cells_cell)
    args = (z, 0.7, 3, [1, 2, 3], [[1], [2], [3]], [1, 2, 3], [1], 1)
    @test unbound(args...) ≈ exp(0.7z[1])
    @test partition_footprint(unbound, args...).z === :whole
end

@testset "scan history stays separate from shared outer indices" begin
    data = (; n = 3, positions = [1, 2, 3], steps = [1, 2])
    cell = prepare(plate_cell(with_history, :cells; index = :i, locality = :error);
        have = (:z, keys(data)..., :i), want = :cells_cell, bound = data)
    z = [0.1, -0.2, 0.3]
    for i in 1:3
        @test cell(z, i) ≈ 3exp(z[i])
        @test partition_footprint(cell, z, i).z == [i]
    end
end
end
