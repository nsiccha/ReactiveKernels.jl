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

# A location plate read by two density plates, the shape RKPPL emits for an
# `@plate for` cell whose shared work feeds two observations.
const _CELL_FORK_CALLS = Ref(0)
_cell_fork_exp(v) = (_CELL_FORK_CALLS[] += 1; exp(v))
@kernel _cell_fork(x, y, z, s::Float64) = begin
    loc = plate(x) do xi
        _cell_fork_exp(xi * s)
    end
    dy = plate(loc, y) do li, yi
        -(yi - li)^2 / 2
    end
    dz = plate(loc, z) do li, zi
        -(zi - 2li)^2 / 2
    end
    total = sum(dy) + sum(dz)
    return total
end

@testset "plate_cell: a plate read only by cells at one position runs at that cell" begin
    cells = plate_cell(plate_cell(_cell_fork, :dy; index = :i), :dz; index = :i)
    k = prepare(cells; have = (:x, :y, :z, :s, :i), want = (:dy_cell, :dz_cell))
    full = prepare(_cell_fork; have = (:x, :y, :z, :s), want = (:dy, :dz))
    function data(n)
        xs = collect(range(-1.0, 1.0; length = n))
        (xs, 2 .* sin.(xs), cos.(xs), 0.8)
    end
    x, y, z, s = data(9)
    dy, dz = full(x, y, z, s)
    @test all(k(x, y, z, s, i) == (dy[i], dz[i]) for i in eachindex(x))
    # Each cell runs the location at its position: not the whole plate.
    _CELL_FORK_CALLS[] = 0
    k(x, y, z, s, 4)
    @test _CELL_FORK_CALLS[] == 2
    allocated(args...) = (k(args..., 3); @allocated k(args..., 3))
    small, large = allocated(data(10)...), allocated(data(100_000)...)
    @test large == small
    # The whole kernel still evaluates the location once per cell.
    total = prepare(_cell_fork; have = (:x, :y, :z, :s), want = :total)
    _CELL_FORK_CALLS[] = 0
    total(x, y, z, s)
    @test _CELL_FORK_CALLS[] == length(x)
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

# Upstream arrays the cell reads only at its own row (`log_k[i]`, `shift[i]`):
# a dotted predictor over a gather of a matrix product's rows, and gathers on
# their own. Each is computed at the cell's row only (increment 1b).
@kernel _cell_upstream(positions::Vector{Int}, y, w, codes::Vector{Int},
                       Z::Matrix{Float64}, F::Matrix{Float64}, a::Float64) = begin
    R = Z * F'
    eta = R[codes, 1]
    log_k = a .+ 0.5 .* w .+ eta
    shift = R[codes, 2]
    loc = plate(positions) do i
        exp(log_k[i]) + shift[i]
    end
    dens = plate(loc, y) do li, yi
        -(yi - li)^2 / 2
    end
    total = sum(dens)
    return total
end

function _cell_upstream_data(n; levels = max(1, n ÷ 2))
    (; positions = collect(1:n), y = sin.(1:n),
       w = collect(range(-1.0, 1.0; length = n)),
       codes = [mod1(3j, levels) for j in 1:n],
       Z = reshape(collect(range(-0.5, 0.7; length = 2levels)), levels, 2))
end

const _CELL_F = [0.8 0.0; 0.3 0.5]
const _CELL_HAVE = (:positions, :y, :w, :codes, :Z, :F, :a)

@testset "plate_cell: upstream arrays computed at the cell's row" begin
    spec = plate_cell(_cell_upstream, :dens; index = :i)
    k = prepare(spec; have = (_CELL_HAVE..., :i), want = :dens_cell)
    full = prepare(_cell_upstream; have = _CELL_HAVE, want = :dens)
    d = _cell_upstream_data(9)
    reference = full(d..., _CELL_F, 0.3)
    @test [k(d..., _CELL_F, 0.3, i) for i in d.positions] ≈ reference
    # Neither the predictors nor the product are computed for the domain.
    allocated(d) = (k(d..., _CELL_F, 0.3, 2); @allocated k(d..., _CELL_F, 0.3, 2))
    small, large = allocated(_cell_upstream_data(10)), allocated(_cell_upstream_data(100_000))
    @test large == small
    @test large < 8 * 1000
    bound = prepare(spec; have = (_CELL_HAVE..., :i), want = :dens_cell,
        bound = (; d.positions, d.y, d.w, d.codes))
    @test [bound(d.Z, _CELL_F, 0.3, i) for i in d.positions] ≈ reference
    # An operand extruded by broadcasting, and a value also wanted whole.
    one_w = (; d..., w = [0.2])
    @test [k(one_w..., _CELL_F, 0.3, i) for i in d.positions] ≈ full(one_w..., _CELL_F, 0.3)
    both = prepare(spec; have = (_CELL_HAVE..., :i), want = (:dens_cell, :log_k))
    log_k = prepare(_cell_upstream; have = _CELL_HAVE, want = :log_k)(d..., _CELL_F, 0.3)
    @test all(both(d..., _CELL_F, 0.3, i) == (k(d..., _CELL_F, 0.3, i), log_k)
              for i in d.positions)
    # The cell's own reads are checked, and operand shapes as a whole.
    @test_throws BoundsError k(d..., _CELL_F, 0.3, 10)
    bad_code = (; d..., codes = [d.codes[1], size(d.Z, 1) + 1, d.codes[3:end]...])
    @test_throws BoundsError k(bad_code..., _CELL_F, 0.3, 2)
    @test_throws DimensionMismatch k((; d..., w = [d.w; 0.0])..., _CELL_F, 0.3, 2)
end

@testset "plate_cell: native Enzyme Reverse through upstream rows" begin
    k = prepare(plate_cell(_cell_upstream, :dens; index = :i);
        have = (_CELL_HAVE..., :i), want = :dens_cell)
    d = _cell_upstream_data(9)
    original = deepcopy(d)
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for i in (1, 6)
        value = k(d..., _CELL_F, 0.3, i)
        matrix = prepare_ad(k, backend, d..., _CELL_F, 0.3, i; active = :Z)
        v, dZ = ad_value_and_gradient(matrix, d..., _CELL_F, 0.3, i)
        @test v ≈ value
        # Only the row of Z the cell gathers has a derivative.
        @test findall(vec(any(!iszero, dZ; dims = 2))) == [d.codes[i]]
        for column in 1:2
            e = zero(d.Z); e[d.codes[i], column] = 1.0
            @test dZ[d.codes[i], column] ≈ _cell_central(
                h -> k((; d..., Z = d.Z .+ h .* e)..., _CELL_F, 0.3, i), 0.0) rtol = 1e-6
        end
        scalar = prepare_ad(k, backend, d..., _CELL_F, 0.3, i; active = :a)
        _, da = ad_value_and_gradient(scalar, d..., _CELL_F, 0.3, i)
        @test da ≈ _cell_central(a -> k(d..., _CELL_F, a, i), 0.3) rtol = 1e-6
        factor = prepare_ad(k, backend, d..., _CELL_F, 0.3, i; active = :F)
        _, dF = ad_value_and_gradient(factor, d..., _CELL_F, 0.3, i)
        e = zero(_CELL_F); e[2, 1] = 1.0
        @test dF[2, 1] ≈ _cell_central(h -> k(d..., _CELL_F .+ h .* e, 0.3, i), 0.0) rtol = 1e-6
    end
    @test d == original
end

# A population predictor `X * beta` read as an operand of the dotted
# predictor: a matrix-vector product, computed at the cell's row as well.
@kernel _cell_population(positions::Vector{Int}, y, X::Matrix{Float64},
                         beta::Vector{Float64}, codes::Vector{Int}, Z) = begin
    pop = X * beta
    log_k = pop .+ Z[codes, 1]
    loc = plate(positions) do i
        exp(log_k[i])
    end
    dens = plate(loc, y) do li, yi
        -(yi - li)^2 / 2
    end
    total = sum(dens)
    return total
end

@testset "plate_cell: a matrix-vector product read by a dotted predictor" begin
    have = (:positions, :y, :X, :beta, :codes, :Z)
    k = prepare(plate_cell(_cell_population, :dens; index = :i);
        have = (have..., :i), want = :dens_cell)
    full = prepare(_cell_population; have, want = :dens)
    function data(n)
        x = collect(range(-1.0, 1.0; length = n))
        (; positions = collect(1:n), y = cos.(1:n), X = hcat(ones(n), x, x .^ 2),
           beta = [0.1, -0.4, 0.3], codes = [mod1(j, 3) for j in 1:n],
           Z = [0.2 -0.1; 0.5 0.3; -0.3 0.0])
    end
    d = data(7)
    @test [k(d..., i) for i in d.positions] ≈ full(d...)
    allocated(d) = (k(d..., 2); @allocated k(d..., 2))
    @test allocated(data(10)) == allocated(data(100_000))
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for i in (2, 5)
        ad = prepare_ad(k, backend, d..., i; active = :beta)
        value, dbeta = ad_value_and_gradient(ad, d..., i)
        @test value ≈ k(d..., i)
        for j in 1:3
            e = zeros(3); e[j] = 1.0
            @test dbeta[j] ≈ _cell_central(h -> k((; d..., beta = d.beta .+ h .* e)..., i),
                0.0) rtol = 1e-6
        end
        design = prepare_ad(k, backend, d..., i; active = :X)
        _, dX = ad_value_and_gradient(design, d..., i)
        @test findall(vec(any(!iszero, dX; dims = 2))) == [i]
    end
    @test d == data(7)
end
