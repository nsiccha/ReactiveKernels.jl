using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# `plate_cell` makes ONE cell of an authored plate a graph value at a runtime
# index: the plate's own scalar body at that position, with a single-consumer
# plate upstream composed into it (snag `rkppl-per-subjec-a3d6e6d8`, user
# decision `06z2prg`, option 1a).

@kernel _cell_scaled(x, s::Float64) = begin
    pw = plate(x) do xi
        v::Float64 = xi * s + 1.0
        v
    end
    total = sum(pw)
    return total
end

# A location plate feeding a density plate, the shape RKPPL emits for an
# `@plate for` loop with a cell local and an observation.
@kernel _cell_chain(x, y, s::Float64) = begin
    loc = plate(x) do xi
        exp(xi * s)
    end
    dens = plate(loc, y) do li, yi
        -(yi - li)^2 / 2
    end
    total = sum(dens)
    return total
end

# The location reads whole values at the cell's own position, as RKPPL's
# `@plate for i in eachindex(y)` cells read `log_k[i]`.
@kernel _cell_indexed(positions, y, log_k, scale::Float64) = begin
    loc = plate(positions) do i
        exp(log_k[i])
    end
    dens = plate(loc, y) do li, yi
        -((yi - li) / scale)^2 / 2 - log(scale)
    end
    total = sum(dens)
    return total
end

# One array of observations per cell (ragged, empty allowed): nested plates.
@kernel _cell_grouped(groups, scale::Float64) = begin
    group_density = plate(groups) do observations
        pointwise = plate(observations) do x
            -0.5 * (x / scale)^2 - log(scale)
        end
        sum(pointwise)
    end
    total = sum(group_density)
    return total
end

@kernel _cell_matrix(M, v) = begin
    cells = plate(M, v) do m, vi
        m^2 + vi
    end
    return cells
end

_cell_central(f, x; h = 1e-6) = (f(x + h) - f(x - h)) / (2h)

@testset "plate_cell: one cell at a runtime index" begin
    x = [0.3, -1.2, 2.0, 0.7]
    s = 1.7
    before = length(kernel_graph(_cell_scaled).recipes)
    cellspec = plate_cell(_cell_scaled, :pw; index = :i)
    @test length(kernel_graph(_cell_scaled).recipes) == before
    # The cell's value has the plate body's declared result type.
    @test cellspec[:pw_cell] isa ReactiveKernels.Value{Float64}
    k = prepare(cellspec; have = (:x, :s, :i), want = :pw_cell)
    full = prepare(_cell_scaled; have = (:x, :s), want = :pw)(x, s)
    @test [k(x, s, i) for i in eachindex(x)] == full
    @test prepare(cellspec)(x, s, 2) == full[2]
    @test_throws BoundsError k(x, s, 0)
    @test_throws BoundsError k(x, s, length(x) + 1)
    bound = prepare(cellspec; have = (:x, :s, :i), want = :pw_cell, bound = (; x))
    @test [bound(s, i) for i in eachindex(x)] == full
    ignoring = prepare(cellspec; have = (:x, :s, :i), want = :pw_cell, on_error = :ignore)
    @test [ignoring(x, s, i) for i in eachindex(x)] == full
    @test_throws ArgumentError plate_cell(_cell_scaled, :total)
    @test_throws ArgumentError plate_cell(cellspec, :pw; index = :i, name = :pw_cell)
end

@testset "plate_cell: a single-consumer upstream plate runs at the cell only" begin
    k = prepare(plate_cell(_cell_chain, :dens; index = :i);
        have = (:x, :y, :s, :i), want = :dens_cell)
    full = prepare(_cell_chain; have = (:x, :y, :s), want = :dens)
    function data(n)
        x = collect(range(-1.0, 1.0; length = n))
        (x, 2 .* sin.(x), 0.8)
    end
    x, y, s = data(9)
    reference = full(x, y, s)
    @test [k(x, y, s, i) for i in eachindex(x)] ≈ reference
    # The cell does not materialize the location plate over the domain.
    allocated(x, y, s) = (k(x, y, s, 3); @allocated k(x, y, s, 3))
    small, large = allocated(data(10)...), allocated(data(100_000)...)
    @test large == small
    @test large < 8 * 1000
end

@testset "plate_cell: captured whole values read at the cell's position" begin
    n = 6
    positions = collect(1:n)
    log_k = collect(range(-0.5, 0.5; length = n))
    y = [1.1, 0.4, 0.9, 1.8, 0.2, 1.4]
    spec = plate_cell(_cell_indexed, :dens; index = :i)
    k = prepare(spec; have = (:positions, :y, :log_k, :scale, :i), want = :dens_cell)
    full = prepare(_cell_indexed; have = (:positions, :y, :log_k, :scale), want = :dens)
    @test [k(positions, y, log_k, 0.7, i) for i in 1:n] ≈ full(positions, y, log_k, 0.7)
    bound = prepare(spec; have = (:positions, :y, :log_k, :scale, :i), want = :dens_cell,
        bound = (; positions, y))
    @test [bound(log_k, 0.7, i) for i in 1:n] ≈ full(positions, y, log_k, 0.7)
    # An active captured vector beside bound constant data, under native Reverse.
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for i in (1, 4)
        ad = prepare_ad(bound, backend, log_k, 0.7, i; active = :log_k)
        value, dlog_k = ad_value_and_gradient(ad, log_k, 0.7, i)
        @test value ≈ bound(log_k, 0.7, i)
        @test count(!iszero, dlog_k) == 1
        e = zeros(n); e[i] = 1.0
        @test dlog_k[i] ≈ _cell_central(h -> bound(log_k .+ h .* e, 0.7, i), 0.0) rtol = 1e-6
    end
end

@testset "plate_cell: nested plates over one array per cell" begin
    groups = [[0.2, 0.7], Float64[], [-1.2], [0.1, 0.4, -0.3]]
    k = prepare(plate_cell(_cell_grouped, :group_density; index = :i);
        have = (:groups, :scale, :i), want = :group_density_cell)
    full = prepare(_cell_grouped; have = (:groups, :scale), want = :group_density)
    @test [k(groups, 1.3, i) for i in eachindex(groups)] ≈ full(groups, 1.3)
    @test k(groups, 1.3, 2) == 0.0
    # Reverse through a cell over one array per index leaves the caller's
    # arrays untouched: the slice must not be a fresh container of them.
    original = deepcopy(groups)
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for i in (1, 4)
        ad = prepare_ad(k, backend, groups, 1.3, i; active = :scale)
        value, dscale = ad_value_and_gradient(ad, groups, 1.3, i)
        @test value ≈ k(groups, 1.3, i)
        @test dscale ≈ _cell_central(s -> k(groups, s, i), 1.3) rtol = 1e-6
        @test groups == original
    end
    @test [k(groups, 1.3, i) for i in eachindex(groups)] ≈ full(groups, 1.3)
end

@testset "plate_cell: Cartesian and linear positions, extruded arguments" begin
    M = reshape(collect(1.0:12.0), 3, 4)
    v = [10.0 20.0 30.0 40.0]
    k = prepare(plate_cell(_cell_matrix, :cells; index = :i);
        have = (:M, :v, :i), want = :cells_cell)
    full = prepare(_cell_matrix; have = (:M, :v), want = :cells)(M, v)
    @test all(k(M, v, I) == full[I] for I in CartesianIndices(full))
    @test all(k(M, v, j) == full[j] for j in eachindex(full))
end

@testset "plate_cell: several plates at one index port" begin
    first = plate_cell(_cell_chain, :loc; index = :i)
    both = plate_cell(first, :dens; index = :i)
    k = prepare(both; have = (:x, :y, :s, :i), want = (:loc_cell, :dens_cell))
    x, y = [0.1, 0.5, -0.4], [1.0, 2.0, 0.5]
    loc = prepare(_cell_chain; have = (:x, :y, :s), want = :loc)(x, y, 0.9)
    dens = prepare(_cell_chain; have = (:x, :y, :s), want = :dens)(x, y, 0.9)
    @test all(k(x, y, 0.9, i) == (loc[i], dens[i]) for i in 1:3)
end

@testset "plate_cell: native Enzyme Reverse through a composed cell" begin
    k = prepare(plate_cell(_cell_chain, :dens; index = :i);
        have = (:x, :y, :s, :i), want = :dens_cell)
    x = collect(range(-1.0, 1.0; length = 7))
    y = 2 .* sin.(x)
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for i in (1, 4, 7)
        scalar = prepare_ad(k, backend, x, y, 0.8, i; active = :s)
        value, gradient = ad_value_and_gradient(scalar, x, y, 0.8, i)
        @test value ≈ k(x, y, 0.8, i)
        @test gradient ≈ _cell_central(s -> k(x, y, s, i), 0.8) rtol = 1e-6
        vector = prepare_ad(k, backend, x, y, 0.8, i; active = :x)
        _, dx = ad_value_and_gradient(vector, x, y, 0.8, i)
        @test count(!iszero, dx) == 1
        e = zeros(length(x)); e[i] = 1.0
        @test dx[i] ≈ _cell_central(h -> k(x .+ h .* e, y, 0.8, i), 0.0) rtol = 1e-6
    end
end
