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

# A composed reader can zip the predictor itself as a plate argument, instead
# of reading it through `log_k[i]` inside the cell. Its producer still needs
# only the selected row.
@kernel _cell_batched_population(y, X::Matrix{Float64}, beta::Vector{Float64},
                                 codes::Vector{Int}, Z) = begin
    pop = X * beta
    log_k = pop .+ Z[codes, 1]
    dens = plate(log_k, y) do li, yi
        -(yi - exp(li))^2 / 2
    end
    return sum(dens)
end

@testset "plate_cell: batched predictor inputs run at the selected row" begin
    have = (:y, :X, :beta, :codes, :Z)
    k = prepare(plate_cell(_cell_batched_population, :dens; index = :i);
        have = (have..., :i), want = :dens_cell)
    full = prepare(_cell_batched_population; have, want = :dens)
    function data(n)
        x = collect(range(-1.0, 1.0; length = max(n, 2)))[1:n]
        (; y = cos.(1:n), X = hcat(ones(n), x), beta = [0.1, -0.4],
           codes = [mod1(j, 3) for j in 1:n], Z = [0.2 -0.1; 0.5 0.3; -0.3 0.0])
    end
    d = data(7)
    @test [k(d..., i) for i in eachindex(d.y)] ≈ full(d...)
    singleton = (; data(1)..., y = d.y)
    @test [k(singleton..., i) for i in eachindex(singleton.y)] ≈ full(singleton...)
    allocated(d) = (k(d..., 2); @allocated k(d..., 2))
    @test allocated(data(10)) == allocated(data(100_000))
    @test_throws BoundsError k(d..., length(d.y) + 1)
    @test_throws DimensionMismatch k((; d..., y = [d.y; 0.0])..., 2)
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    ad = prepare_ad(k, backend, d..., 2; active = :Z)
    value, dZ = ad_value_and_gradient(ad, d..., 2)
    @test value ≈ k(d..., 2)
    @test findall(vec(any(!iszero, dZ; dims = 2))) == [d.codes[2]]
end

# A plate's output read whole by a later plate at that plate's own position
# (`pred[i]`), the shape RKPPL emits for a caller-owned law observing a kernel
# plate's per-row output (snag `brm-s-nlme-subje-f6f0efd3`). The producing
# plate, and the predictor it reads at its own row, run at the cell's row only.
@kernel _cell_plate_upstream(positions::Vector{Int}, y, rows::Vector{Int}, w,
                             codes::Vector{Int}, z, a::Float64) = begin
    log_k = a .+ 0.5 .* w .+ z[codes]
    pred = plate(rows) do r
        k = exp(log_k[r])
        (k, 2k)
    end
    second = plate(rows, pred) do r, p
        p[2]
    end
    dens = plate(positions, y) do i, yi
        -(yi - second[i])^2 / 2
    end
    total = sum(dens)
    return total
end

@testset "plate_cell: a plate's output read at the cell's own position" begin
    have = (:positions, :y, :rows, :w, :codes, :z, :a)
    k = prepare(plate_cell(_cell_plate_upstream, :dens; index = :i);
        have = (have..., :i), want = :dens_cell)
    full = prepare(_cell_plate_upstream; have, want = :dens)
    function data(n; levels = max(1, n ÷ 2))
        (; positions = collect(1:n), y = sin.(1:n), rows = collect(1:n),
           w = collect(range(-1.0, 1.0; length = n)),
           codes = [mod1(3j, levels) for j in 1:n],
           z = collect(range(-0.5, 0.7; length = levels)))
    end
    d = data(9)
    reference = full(d..., 0.3)
    @test [k(d..., 0.3, i) for i in d.positions] ≈ reference
    # Neither plate nor the predictor is computed for the domain.
    allocated(d) = (k(d..., 0.3, 2); @allocated k(d..., 0.3, 2))
    small, large = allocated(data(10)), allocated(data(100_000))
    @test large == small
    @test large < 8 * 1000
    bound = prepare(plate_cell(_cell_plate_upstream, :dens; index = :i);
        have = (have..., :i), want = :dens_cell,
        bound = (; d.positions, d.y, d.rows, d.w, d.codes))
    @test [bound(d.z, 0.3, i) for i in d.positions] ≈ reference
    # A value also wanted whole is computed whole and read in place.
    both = prepare(plate_cell(_cell_plate_upstream, :dens; index = :i);
        have = (have..., :i), want = (:dens_cell, :second))
    second = prepare(_cell_plate_upstream; have, want = :second)(d..., 0.3)
    @test all(both(d..., 0.3, i) == (k(d..., 0.3, i), second) for i in d.positions)
    # The read keeps its index semantics: a shorter producer is out of bounds
    # at the cells it lacks, as in the whole program, never extruded.
    short = (; d..., rows = [1, 2, 3])
    @test [k(short..., 0.3, i) for i in 1:3] ≈ [k(d..., 0.3, i) for i in 1:3]
    @test_throws BoundsError full(short..., 0.3)
    @test_throws BoundsError k(short..., 0.3, 4)
    @test_throws BoundsError k(d..., 0.3, 10)
    bad_code = (; d..., codes = [d.codes[1], length(d.z) + 1, d.codes[3:end]...])
    @test_throws BoundsError k(bad_code..., 0.3, 2)
    @test_throws DimensionMismatch k((; d..., w = [d.w; 0.0])..., 0.3, 2)
    # Native Enzyme Reverse: only the gathered entry of z has a derivative.
    original = deepcopy(d)
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for i in (1, 6)
        latent = prepare_ad(k, backend, d..., 0.3, i; active = :z)
        value, dz = ad_value_and_gradient(latent, d..., 0.3, i)
        @test value ≈ reference[i]
        @test findall(!iszero, dz) == [d.codes[i]]
        e = zero(d.z); e[d.codes[i]] = 1.0
        @test dz[d.codes[i]] ≈ _cell_central(
            h -> k((; d..., z = d.z .+ h .* e)..., 0.3, i), 0.0) rtol = 1e-6
        scalar = prepare_ad(k, backend, d..., 0.3, i; active = :a)
        _, da = ad_value_and_gradient(scalar, d..., 0.3, i)
        @test da ≈ _cell_central(a -> k(d..., a, i), 0.3) rtol = 1e-6
    end
    @test d == original
end

@kernel _cell_batched_matrix(x::Matrix{Float64}, y::Matrix{Float64}, a::Float64) = begin
    loc = x .+ a
    dens = plate(loc, y) do li, yi
        -(yi - li)^2 / 2
    end
    return dens
end

@testset "plate_cell: a batched matrix keeps its trailing cell coordinate" begin
    k = prepare(plate_cell(_cell_batched_matrix, :dens; index = :i);
        have = (:x, :y, :a, :i), want = :dens_cell)
    full = prepare(_cell_batched_matrix; have = (:x, :y, :a), want = :dens)
    x = reshape(collect(range(-0.5, 0.5; length = 12)), 4, 3)
    y = cos.(x)
    @test [k(x, y, 0.2, I) for I in CartesianIndices(x)] ≈ full(x, y, 0.2)
    @test k(x, y, 0.2, CartesianIndex(2, 3)) ≈ full(x, y, 0.2)[2, 3]
    @test_throws BoundsError k(x, y, 0.2, CartesianIndex(5, 1))
end

const _CELL_GUARD_CALLS = Ref(0)
_cell_guard_value(x, scale) = (_CELL_GUARD_CALLS[] += 1; x * scale)

@kernel _cell_guarded_chain(x, present, y, scale::Float64) = begin
    loc = plate(x, present) do xi, keep
        keep ? _cell_guard_value(xi, scale) : 0.0
    end
    density = plate(loc, y, present) do li, yi, keep
        keep ? -(yi - li)^2 / 2 : 0.0
    end
    return sum(density)
end

@testset "plate_cell: a bound presence guard stays inside the queried cell" begin
    x = collect(range(0.2, 1.0; length = 9))
    y = x .+ 0.1
    present = trues(length(x))
    present[3] = false
    spec = plate_cell(_cell_guarded_chain, :density; index = :i)
    p = plan(spec; have = (:x, :present, :y, :scale, :i), want = :density_cell)
    k = prepare(p; bound = (p.have[1] => x, p.have[2] => present, p.have[3] => y))
    for i in (1, 3, 5, 9)
        _CELL_GUARD_CALLS[] = 0
        @test k(0.7, i) ≈ (present[i] ? -(y[i] - 0.7x[i])^2 / 2 : 0.0)
        @test _CELL_GUARD_CALLS[] == (present[i] ? 1 : 0)
    end
end

# A cell reading a row-wise value at several rows: `log_w[r]`, `r` one integer
# vector per cell, as a total over a group of rows reads them (snag
# `compact-cell-sam-d85d96ae`). The predictor and the product rows it gathers
# run at those rows only.
@kernel _cell_ragged(rows::Vector{Vector{Int}}, totals, w, codes::Vector{Int},
                     Z::Matrix{Float64}, F::Matrix{Float64}, a::Float64) = begin
    R = Z * F'
    log_w = a .+ 0.5 .* w .+ R[codes, 1]
    dens = plate(rows, totals) do r, t
        -(t - sum(exp.(log_w[r])))^2 / 2
    end
    total = sum(dens)
    return total
end

# The same rows read through a shared index array, `log_w[rows[k]]`, the
# shape RKPPL and BRM emit for `rows = groups(g => cells)`.
@kernel _cell_ragged_shared(positions::Vector{Int}, rows::Vector{Vector{Int}}, totals,
                            w, codes::Vector{Int}, z, a::Float64) = begin
    log_w = a .+ 0.5 .* w .+ z[codes]
    dens = plate(positions, totals) do k, t
        -(t - sum(exp.(log_w[rows[k]])))^2 / 2
    end
    return dens
end

function _cell_ragged_data(n; size = 3, levels = max(1, n ÷ 4))
    rows = [collect(j:min(j + size - 1, n)) for j in 1:size:n]
    (; rows, totals = collect(range(1.0, 4.0; length = length(rows))),
       w = collect(range(-1.0, 1.0; length = n)),
       codes = [mod1(3j, levels) for j in 1:n],
       Z = reshape(collect(range(-0.5, 0.7; length = 2levels)), levels, 2))
end

const _RAGGED_HAVE = (:rows, :totals, :w, :codes, :Z, :F, :a)

@testset "plate_cell: a value read at several rows of a cell" begin
    spec = plate_cell(_cell_ragged, :dens; index = :i)
    k = prepare(spec; have = (_RAGGED_HAVE..., :i), want = :dens_cell)
    full = prepare(_cell_ragged; have = _RAGGED_HAVE, want = :dens)
    d = _cell_ragged_data(13)
    # A short group, an empty one, a repeated row and unsorted rows.
    d = (; d..., rows = [d.rows..., Int[], [2, 2, 9], [7, 1]],
           totals = [d.totals..., 0.5, 1.5, 2.5])
    reference = full(d..., _CELL_F, 0.3)
    @test [k(d..., _CELL_F, 0.3, i) for i in eachindex(d.rows)] ≈ reference
    # Neither the predictor nor the product is computed for the domain.
    allocated(d) = (k(d..., _CELL_F, 0.3, 2); @allocated k(d..., _CELL_F, 0.3, 2))
    @test allocated(_cell_ragged_data(12)) == allocated(_cell_ragged_data(120_000))
    bound = prepare(spec; have = (_RAGGED_HAVE..., :i), want = :dens_cell,
        bound = (; d.rows, d.totals, d.w, d.codes))
    @test [bound(d.Z, _CELL_F, 0.3, i) for i in eachindex(d.rows)] ≈ reference
    # An operand extruded by broadcasting repeats its row.
    one_w = (; d..., w = [0.2])
    @test [k(one_w..., _CELL_F, 0.3, i) for i in eachindex(d.rows)] ≈
        full(one_w..., _CELL_F, 0.3)
    # The cell's reads are checked, and operand shapes as a whole.
    beyond = (; d..., rows = [d.rows[1], [1, length(d.w) + 1]], totals = d.totals[1:2])
    @test_throws BoundsError full(beyond..., _CELL_F, 0.3)
    @test_throws BoundsError k(beyond..., _CELL_F, 0.3, 2)
    @test_throws DimensionMismatch k((; d..., w = [d.w; 0.0])..., _CELL_F, 0.3, 2)
    # Native Enzyme Reverse: only the gathered rows of Z have a derivative, and
    # a repeated row adds both reads.
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    original = deepcopy(d)
    for i in (1, length(d.rows) - 1)
        matrix = prepare_ad(k, backend, d..., _CELL_F, 0.3, i; active = :Z)
        value, dZ = ad_value_and_gradient(matrix, d..., _CELL_F, 0.3, i)
        @test value ≈ reference[i]
        gathered = sort!(unique(d.codes[d.rows[i]]))
        @test findall(vec(any(!iszero, dZ; dims = 2))) == gathered
        for j in gathered
            e = zero(d.Z); e[j, 1] = 1.0
            @test dZ[j, 1] ≈ _cell_central(
                h -> k((; d..., Z = d.Z .+ h .* e)..., _CELL_F, 0.3, i), 0.0) rtol = 1e-6
        end
        scalar = prepare_ad(k, backend, d..., _CELL_F, 0.3, i; active = :a)
        _, da = ad_value_and_gradient(scalar, d..., _CELL_F, 0.3, i)
        @test da ≈ _cell_central(a -> k(d..., _CELL_F, a, i), 0.3) rtol = 1e-6
    end
    @test d == original
end

@testset "plate_cell: several rows read through a shared index array" begin
    have = (:positions, :rows, :totals, :w, :codes, :z, :a)
    k = prepare(plate_cell(_cell_ragged_shared, :dens; index = :i);
        have = (have..., :i), want = :dens_cell)
    full = prepare(_cell_ragged_shared; have, want = :dens)
    function data(n; levels = max(1, n ÷ 4))
        base = _cell_ragged_data(n; levels)
        (; positions = collect(eachindex(base.rows)), base.rows, base.totals, base.w,
           base.codes, z = collect(range(-0.5, 0.7; length = levels)))
    end
    d = data(13)
    reference = full(d..., 0.3)
    @test [k(d..., 0.3, i) for i in d.positions] ≈ reference
    allocated(d) = (k(d..., 0.3, 2); @allocated k(d..., 0.3, 2))
    @test allocated(data(12)) == allocated(data(120_000))
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for i in (1, 3)
        ad = prepare_ad(k, backend, d..., 0.3, i; active = :z)
        value, dz = ad_value_and_gradient(ad, d..., 0.3, i)
        @test value ≈ reference[i]
        @test findall(!iszero, dz) == sort!(unique(d.codes[d.rows[i]]))
        j = first(d.codes[d.rows[i]])
        e = zero(d.z); e[j] = 1.0
        @test dz[j] ≈ _cell_central(h -> k((; d..., z = d.z .+ h .* e)..., 0.3, i), 0.0) rtol = 1e-6
    end
end

# Design matrices and group blocks as BRM and RKPPL emit them: an `hcat` design
# with a computed column read by a product, a level block scaled by a
# literal-position read (`z .* tau[1]`), a per-row plate written as one matrix
# column (`hcat`) and read back as `B[:, 1]`, and the same reads inline in a
# dotted predictor. Each runs at the cell's row only.
@kernel _cell_designs(positions::Vector{Int}, y, one, x, year, g::Vector{Int},
                      s::Vector{Int}, beta::Vector{Float64}, z::Matrix{Float64},
                      tau::Vector{Float64}, c0::Float64) = begin
    c = x .- c0
    X = hcat(one, c, year)
    pop = X * beta
    b = z .* tau[1]
    strat = plate(g, s) do gi, si
        z[gi, 1] * tau[si]
    end
    B = hcat(strat)
    col = B[:, 1]
    eta = pop .+ b[g, 1] .+ col
    eta2 = X * beta .+ B[:, 1]
    loc = plate(positions) do i
        exp(eta[i]) + eta2[i]
    end
    dens = plate(loc, y) do li, yi
        -(yi - li)^2 / 2
    end
    return dens
end

@testset "plate_cell: designs, column reads and level blocks read at the cell's row" begin
    have = (:positions, :y, :one, :x, :year, :g, :s, :beta, :z, :tau, :c0)
    spec = plate_cell(_cell_designs, :dens; index = :i)
    k = prepare(spec; have = (have..., :i), want = :dens_cell)
    full = prepare(_cell_designs; have, want = :dens)
    function data(n; levels = max(2, n ÷ 3))
        (; positions = collect(1:n), y = cos.(1:n), one = ones(n),
           x = collect(range(0.5, 1.5; length = n)), year = collect(range(-1.0, 1.0; length = n)),
           g = [mod1(2j, levels) for j in 1:n], s = [1 + j % 2 for j in 1:n])
    end
    params(levels) = (; beta = [0.1, -0.3, 0.2],
        z = reshape(collect(range(-0.4, 0.6; length = levels)), levels, 1),
        tau = [0.7, 1.3], c0 = 0.9)
    d, p = data(9), params(3)
    reference = full(d..., p...)
    @test [k(d..., p..., i) for i in d.positions] ≈ reference
    pushed = let lp = ReactiveKernels._lowering_plan(k.plan),
                 cells = ReactiveKernels._cell_pushdowns(lp)
        Set(String(cells.producer[c].outputs[1].name) for c in cells.pushed)
    end
    @test issubset(Set(["c", "X", "pop", "b", "strat", "B", "col", "eta", "eta2"]), pushed)
    allocated(d, p) = (k(d..., p..., 2); @allocated k(d..., p..., 2))
    @test allocated(data(12), params(4)) == allocated(data(120_000), params(40_000))
    # `hcat` operands of different heights fail as the whole program does.
    short = (; d..., year = d.year[1:end-1])
    @test_throws DimensionMismatch full(short..., p...)
    @test_throws DimensionMismatch k(short..., p..., 2)
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for i in (1, 6), name in (:beta, :z, :tau)
        ad = prepare_ad(k, backend, d..., p..., i; active = name)
        value, dp = ad_value_and_gradient(ad, d..., p..., i)
        @test value ≈ reference[i]
        for j in eachindex(getfield(p, name))
            e = zero(getfield(p, name)); e[j] = 1.0
            @test dp[j] ≈ _cell_central(h -> k(d..., merge(p, NamedTuple{(name,)}(
                (getfield(p, name) .+ h .* e,)))..., i), 0.0) atol = 1e-6
        end
    end
    _, dc = ad_value_and_gradient(prepare_ad(k, backend, d..., p..., 3; active = :c0),
        d..., p..., 3)
    @test dc ≈ _cell_central(c -> k(d..., merge(p, (; c0 = c))..., 3), p.c0) rtol = 1e-6
end

# A joint product's columns read through views, as RKPPL reads a
# per-observation value's columns (`view(P, :, j)`, `_observation_slice_views`;
# snag `column-read-lowe-24508269`): the view is the column read `P[:, j]`,
# alone in RKPPL's `Base.Colon()` spelling and as a dotted operand, so the cell
# computes the product and its `hcat` design at its row only. Read by two
# views, `P` is computed at the union of their rows, so its `hcat` design's
# data operands are row views beside a computed column, which native Enzyme
# Reverse rejects in Base's `_typed_hcat`, as for any `hcat` read at several
# rows (Enzyme snag `native-reverse-t-6d51cae6`; user principle
# `enzyme-fixed-in-enzyme-everywhere`): that gradient is pinned broken.
@kernel _cell_column_view(positions::Vector{Int}, y, one, x, W::Matrix{Float64},
                          c0::Float64) = begin
    c = x .- c0
    X = hcat(one, c)
    P = X * W
    p2 = Base.view(P, Base.Colon(), 2)
    eta = p2 .+ 0.5
    loc = plate(positions) do i
        exp(eta[i])
    end
    dens = plate(loc, y) do li, yi
        -(yi - li)^2 / 2
    end
    return dens
end

@kernel _cell_column_view_inline(positions::Vector{Int}, y, one, x, W::Matrix{Float64},
                                 c0::Float64) = begin
    c = x .- c0
    X = hcat(one, c)
    P = X * W
    eta = 0.5 .+ 2.0 .* view(P, :, 2)
    loc = plate(positions) do i
        exp(eta[i])
    end
    dens = plate(loc, y) do li, yi
        -(yi - li)^2 / 2
    end
    return dens
end

@kernel _cell_column_views_two(positions::Vector{Int}, y, one, x, W::Matrix{Float64},
                               c0::Float64) = begin
    c = x .- c0
    X = hcat(one, c)
    P = X * W
    p1 = Base.view(P, Base.Colon(), 1)
    eta = p1 .+ 2.0 .* view(P, :, 2)
    loc = plate(positions) do i
        exp(eta[i])
    end
    dens = plate(loc, y) do li, yi
        -(yi - li)^2 / 2
    end
    return dens
end

@testset "plate_cell: column views of a product read at the cell's row" begin
    have = (:positions, :y, :one, :x, :W, :c0)
    data(n) = (; positions = collect(1:n), y = cos.(1:n), one = ones(n),
        x = collect(range(0.5, 1.5; length = n)))
    W, c0 = [0.1 -0.2; -0.3 0.4], 0.9
    d = data(9)
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    @testset "$label" for (label, spec, names, broken) in (
            ("one view", _cell_column_view, ["c", "X", "P", "p2", "eta"], false),
            ("an inline view", _cell_column_view_inline, ["c", "X", "P", "eta"], false),
            ("two views", _cell_column_views_two, ["c", "X", "P", "p1", "eta"], true))
        k = prepare(plate_cell(spec, :dens; index = :i); have = (have..., :i),
            want = :dens_cell)
        reference = prepare(spec; have, want = :dens)(d..., W, c0)
        @test [k(d..., W, c0, i) for i in d.positions] ≈ reference
        pushed = let lp = ReactiveKernels._lowering_plan(k.plan),
                     cells = ReactiveKernels._cell_pushdowns(lp)
            cells === nothing ? Set{String}() :
                Set(String(cells.producer[c].outputs[1].name) for c in cells.pushed)
        end
        @test issubset(Set(names), pushed)
        allocated(d) = (k(d..., W, c0, 2); @allocated k(d..., W, c0, 2))
        @test allocated(data(12)) == allocated(data(120_000))
        for i in (1, 6)
            result = try
                ad_value_and_gradient(prepare_ad(k, backend, d..., W, c0, i; active = :W),
                    d..., W, c0, i)
            catch err
                broken || rethrow()
                err
            end
            if result isa Exception
                @test_broken !(result isa Exception)
                @test nameof(typeof(result)) === :IllegalTypeAnalysisException
                continue
            end
            broken && @test_broken !(result isa Exception)
            value, dW = result
            @test value ≈ reference[i]
            for j in eachindex(W)
                e = zero(W); e[j] = 1.0
                @test dW[j] ≈ _cell_central(h -> k(d..., W .+ h .* e, c0, i), 0.0) rtol = 1e-6
            end
            broken && continue
            _, dc = ad_value_and_gradient(
                prepare_ad(k, backend, d..., W, c0, i; active = :c0), d..., W, c0, i)
            @test dc ≈ _cell_central(c -> k(d..., W, c, i), c0) rtol = 1e-6
        end
    end
end

# An `hcat` design read at several rows of a cell: values run at those rows.
# Native Enzyme Reverse rejects Base `hcat` of views beside dense arrays
# (`IllegalTypeAnalysisException` in `_typed_hcat`), which those rows are. That
# is an Enzyme gap, fixed in Enzyme rather than worked around in RK (Enzyme
# snag `native-reverse-t-6d51cae6`; user principle
# `enzyme-fixed-in-enzyme-everywhere`), so the gradient is pinned broken: an
# "Unexpected Pass" signals the repair.
@kernel _cell_ragged_design(positions::Vector{Int}, rows::Vector{Vector{Int}}, totals,
                            one, x, beta::Vector{Float64}, c0::Float64) = begin
    c = x .- c0
    X = hcat(one, c)
    log_w = X * beta
    dens = plate(positions, totals) do k, t
        -(t - sum(exp.(log_w[rows[k]])))^2 / 2
    end
    return dens
end

@testset "plate_cell: an hcat design read at several rows of a cell" begin
    have = (:positions, :rows, :totals, :one, :x, :beta, :c0)
    k = prepare(plate_cell(_cell_ragged_design, :dens; index = :i);
        have = (have..., :i), want = :dens_cell)
    full = prepare(_cell_ragged_design; have, want = :dens)
    function data(n)
        base = _cell_ragged_data(n)
        (; positions = collect(eachindex(base.rows)), base.rows, base.totals,
           one = ones(n), x = collect(range(0.5, 1.5; length = n)))
    end
    d = data(13)
    reference = full(d..., [0.2, -0.4], 0.3)
    @test [k(d..., [0.2, -0.4], 0.3, i) for i in d.positions] ≈ reference
    allocated(d) = (k(d..., [0.2, -0.4], 0.3, 2); @allocated k(d..., [0.2, -0.4], 0.3, 2))
    @test allocated(data(12)) == allocated(data(120_000))
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    result = try
        ad = prepare_ad(k, backend, d..., [0.2, -0.4], 0.3, 2; active = :beta)
        ad_value_and_gradient(ad, d..., [0.2, -0.4], 0.3, 2)
    catch err
        err
    end
    @test_broken !(result isa Exception)
    if result isa Exception
        @test nameof(typeof(result)) === :IllegalTypeAnalysisException
    else
        value, dbeta = result
        @test value ≈ reference[2]
        @test dbeta[2] ≈ _cell_central(b -> k(d..., [0.2, b], 0.3, 2), -0.4) rtol = 1e-6
    end
end

# A completed covariate read through gathers, the shape of a missing-value
# completion: drawn entries `vcat(drawn_free, z)` and a literal-padded
# `vcat(z, 0.0)` gathered onto the covariate's rows, a typed alias of the
# completion, and that alias gathered by two index vectors in one expression.
# Each runs at the rows the cell reads (snag `on-arbeitsplatza-6065a1ef`).
@kernel _cell_completion(y, observed, mask, lookup::Vector{Int}, free::Vector{Float64},
                         scale::Float64, z::Vector{Float64}, r1::Vector{Int},
                         r2::Vector{Int}) = begin
    drawn_free = exp.(free .* scale)
    missing_values = vcat(drawn_free, z)
    padded = vcat(z, 0.0)
    completed = observed .+ mask .* missing_values[lookup] .+ padded[lookup]
    typed::AbstractVector = completed
    eta = log.(typed[r1]) .+ log.(typed[r2])
    dens = plate(eta, y) do ei, yi
        -(yi - ei)^2 / 2
    end
    return dens
end

# Group totals a plate computes through a shared index array (`log_w[rows[k]]`),
# gathered by group in the predictor the cell reads, as a mixture intensity
# sums a component's rows over each group: the plate runs at the groups read,
# and `log_w` at those groups' rows.
@kernel _cell_group_totals(y, positions::Vector{Int}, rows::Vector{Vector{Int}},
                           g::Vector{Int}, x, beta::Vector{Float64}) = begin
    log_w = x .* beta[1] .+ beta[2]
    totals = plate(positions) do k
        sum(exp.(log_w[rows[k]]))
    end
    mu = log.(totals[g])
    dens = plate(mu, y) do mi, yi
        -(yi - mi)^2 / 2
    end
    return dens
end

# One value read at two row sets by two definitions: computed once per cell,
# at both sets of rows, with everything upstream of it.
@kernel _cell_shared_reads(y, x, r1::Vector{Int}, r2::Vector{Int}, a::Float64) = begin
    base = exp.(x .* a)
    first = base[r1] .* 2.0
    second = base[r2] .+ 1.0
    total = first .+ second
    dens = plate(total, y) do ti, yi
        -(yi - ti)^2 / 2
    end
    return dens
end

# Matrices stacked by `vcat` and read by a row gather.
@kernel _cell_stacked(y, A::Matrix{Float64}, B::Matrix{Float64}, g::Vector{Int}) = begin
    M = vcat(A, B)
    v = M[g, 2]
    dens = plate(v, y) do vi, yi
        -(yi - vi)^2 / 2
    end
    return dens
end

# A declaration that converts elements (`Vector{Float64}` of integer rows)
# keeps the whole value: its rows alone could not show the conversion.
@kernel _cell_converted(y, x::Vector{Int}, r::Vector{Int}) = begin
    d = x .* 2
    t::Vector{Float64} = d
    e = t[r] ./ 3
    dens = plate(e, y) do ei, yi
        -(yi - ei)^2 / 2
    end
    return dens
end

# A number computed at a cell's rows and read by several readers has no rows to
# take: a dotted expression of numbers is a number, whichever reader reads it.
@kernel _cell_number_readers(y, x, z, a::Float64, b::Float64) = begin
    s = a .+ b
    l = x .* s
    m = z .+ s
    mu = l .+ m
    dens = plate(mu, y) do mi, yi
        -(yi - mi)^2 / 2
    end
    return dens
end

# The same number read by a dotted operand and by the cell itself.
@kernel _cell_number_lane(y, x, a::Float64) = begin
    s = exp.(a)
    l = x .* s
    dens = plate(l, s, y) do li, si, yi
        -(yi - li)^2 / 2 - si
    end
    return dens
end

# ... by a `vcat` and by a dotted operand; the `vcat` reads no row of it in
# most cells.
@kernel _cell_number_part(y, x, lookup::Vector{Int}, u::Vector{Float64}, a::Float64) = begin
    s = exp.(a)
    padded = vcat(s, u)
    mu = padded[lookup] .+ x .* s
    dens = plate(mu, y) do mi, yi
        -(yi - mi)^2 / 2
    end
    return dens
end

# A rank-0 array read through an alias has no rows either.
@kernel _cell_rank_zero(y, x, z, a) = begin
    s::AbstractArray = a
    l = x .* s
    m = z .+ s
    mu = l .+ m
    dens = plate(mu, y) do mi, yi
        -(yi - mi)^2 / 2
    end
    return dens
end

@testset "plate_cell: vcat operands, typed aliases, several gather indices and index-array plates at the cell's rows, each value once" begin
    have = (:y, :observed, :mask, :lookup, :free, :scale, :z, :r1, :r2)
    k = prepare(plate_cell(_cell_completion, :dens; index = :i);
        have = (have..., :i), want = :dens_cell)
    full = prepare(_cell_completion; have, want = :dens)
    # The rows the tested cells read fall in the same operands at every size:
    # a cell reads `completed` at rows 10 and 16 (`r1`, `r2`), which look up a
    # free draw and the padding literal.
    function data(n)
        nz = max(2, n ÷ 2)
        lookup = [j == 16 ? nz + 1 : mod1(3j, 7) for j in 1:n]
        mask = Float64[isodd(j) for j in 1:n]
        (; y = cos.(1:2n), observed = 1.0 .+ (1 .- mask) .* collect(range(0.1, 0.9; length = n)),
           mask, lookup)
    end
    params(n) = (; free = collect(range(-0.5, 0.4; length = max(1, n ÷ 3))), scale = 0.7,
        z = collect(range(0.1, 0.6; length = max(2, n ÷ 2))))
    rows(n) = (; r1 = [mod1(5j, n) for j in 1:2n], r2 = [mod1(7j + 2, n) for j in 1:2n])
    d, p, r = data(15), params(15), rows(15)
    reference = full(d..., p..., r...)
    @test [k(d..., p..., r..., i) for i in eachindex(reference)] ≈ reference
    pushed = let lp = ReactiveKernels._lowering_plan(k.plan),
                 cells = ReactiveKernels._cell_pushdowns(lp)
        Set(String(cells.producer[c].outputs[1].name) for c in cells.pushed)
    end
    @test issubset(Set(["drawn_free", "missing_values", "padded", "completed", "typed",
        "eta"]), pushed)
    allocated(n) = let d = data(n), p = params(n), r = rows(n)
        k(d..., p..., r..., 2); @allocated k(d..., p..., r..., 2)
    end
    @test allocated(30) == allocated(30_000)
    # A cell's own reads are checked, as the whole program checks them.
    @test_throws BoundsError k(d..., p..., r..., 2 * 15 + 1)
    beyond = (; d..., lookup = [d.lookup[1:end-1]; length(p.z) + 2])
    @test_throws BoundsError full(beyond..., p..., r...)
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for i in (1, 8), name in (:free, :scale, :z)
        ad = prepare_ad(k, backend, d..., p..., r..., i; active = name)
        value, dp = ad_value_and_gradient(ad, d..., p..., r..., i)
        @test value ≈ reference[i]
        for j in eachindex(getfield(p, name))
            e = getfield(p, name) isa Number ? 1.0 :
                setindex!(zero(getfield(p, name)), 1.0, j)
            @test dp[j] ≈ _cell_central(h -> k(d..., merge(p, NamedTuple{(name,)}(
                (getfield(p, name) .+ h .* e,)))..., r..., i), 0.0) atol = 1e-6
        end
    end
    # Stacked matrices: a row of `vcat(A, B)` is a row of `A` or of `B`, and
    # operands of different widths fail as the whole program does.
    sk = prepare(plate_cell(_cell_stacked, :dens; index = :i);
        have = (:y, :A, :B, :g, :i), want = :dens_cell)
    sfull = prepare(_cell_stacked; have = (:y, :A, :B, :g), want = :dens)
    stacked(n) = (; y = sin.(1:n), A = reshape(collect(range(0, 1; length = 3n)), n, 3),
        B = reshape(collect(range(-1, 0; length = 6)), 2, 3), g = [mod1(4j, n + 2) for j in 1:n])
    s = stacked(9)
    @test [sk(s..., i) for i in eachindex(s.y)] ≈ sfull(s...)
    salloc(n) = let s = stacked(n); sk(s..., 2); @allocated sk(s..., 2) end
    @test salloc(10) == salloc(10_000)
    narrow = (; s..., B = s.B[:, 1:2])
    @test_throws DimensionMismatch sfull(narrow...)
    @test_throws DimensionMismatch sk(narrow..., 2)
    # Group totals read through a shared index array, gathered by group.
    tk = prepare(plate_cell(_cell_group_totals, :dens; index = :i);
        have = (:y, :positions, :rows, :g, :x, :beta, :i), want = :dens_cell)
    tfull = prepare(_cell_group_totals; have = (:y, :positions, :rows, :g, :x, :beta),
        want = :dens)
    function totals_data(n)
        groups = max(2, n ÷ 4)
        rows = [Int[j for j in 1:n if mod1(j, groups) == k] for k in 1:groups]
        (; y = cos.(1:n), positions = collect(1:groups), rows,
           g = [mod1(3j, groups) for j in 1:n], x = collect(range(-1, 1; length = n)))
    end
    t, beta = totals_data(12), [0.4, -0.2]
    treference = tfull(t..., beta)
    @test [tk(t..., beta, i) for i in eachindex(t.y)] ≈ treference
    tpushed = let lp = ReactiveKernels._lowering_plan(tk.plan),
                  cells = ReactiveKernels._cell_pushdowns(lp)
        Set(String(cells.producer[c].outputs[1].name) for c in cells.pushed)
    end
    @test issubset(Set(["log_w", "totals", "mu"]), tpushed)
    talloc(n) = let t = totals_data(n); tk(t..., beta, 2); @allocated tk(t..., beta, 2) end
    @test talloc(16) == talloc(16_000)
    tad = prepare_ad(tk, backend, t..., beta, 5; active = :beta)
    tvalue, dbeta = ad_value_and_gradient(tad, t..., beta, 5)
    @test tvalue ≈ treference[5]
    for j in 1:2
        e = setindex!(zeros(2), 1.0, j)
        @test dbeta[j] ≈ _cell_central(h -> tk(t..., beta .+ h .* e, 5), 0.0) rtol = 1e-6
    end
    # A value read at two row sets is computed once, at both: one row check
    # per value computed at rows (`base`, `first`, `second`, `total`).
    rk = prepare(plate_cell(_cell_shared_reads, :dens; index = :i);
        have = (:y, :x, :r1, :r2, :a, :i), want = :dens_cell)
    rfull = prepare(_cell_shared_reads; have = (:y, :x, :r1, :r2, :a), want = :dens)
    shared(n) = (; y = sin.(1:n), x = collect(range(-1, 1; length = n)),
        r1 = [mod1(3j, n) for j in 1:n], r2 = [mod1(5j + 1, n) for j in 1:n])
    sr = shared(11)
    @test [rk(sr..., 0.4, i) for i in 1:11] ≈ rfull(sr..., 0.4)
    code = string(ReactiveKernels.code_expr(rk))
    @test length(findall("_cell_check_row", code)) == 4
    @test occursin("_cell_union_rows", code)
    ralloc(n) = let sr = shared(n); rk(sr..., 0.4, 2); @allocated rk(sr..., 0.4, 2) end
    @test ralloc(11) == ralloc(11_000)
    _, da = ad_value_and_gradient(prepare_ad(rk, backend, sr..., 0.4, 3; active = :a),
        sr..., 0.4, 3)
    @test da ≈ _cell_central(a -> rk(sr..., a, 3), 0.4) rtol = 1e-6
    # A converting declaration is computed whole, and still converts.
    ck = prepare(plate_cell(_cell_converted, :dens; index = :i);
        have = (:y, :x, :r, :i), want = :dens_cell)
    cfull = prepare(_cell_converted; have = (:y, :x, :r), want = :dens)
    c = (; y = cos.(1:6), x = collect(1:4), r = [1, 4, 2, 3, 3, 1])
    @test [ck(c..., i) for i in 1:6] ≈ cfull(c...)
    converted = let lp = ReactiveKernels._lowering_plan(ck.plan),
                    cells = ReactiveKernels._cell_pushdowns(lp)
        cells === nothing ? Set{String}() :
            Set(String(cells.producer[v].outputs[1].name) for v in cells.pushed)
    end
    @test !("t" in converted)
end

@testset "plate_cell: a number read by several readers of a cell" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    pushed(k) = let lp = ReactiveKernels._lowering_plan(k.plan),
                    cells = ReactiveKernels._cell_pushdowns(lp)
        Set(String(cells.producer[v].outputs[1].name) for v in cells.pushed)
    end
    function check(spec, have, data, p, names)
        k = prepare(plate_cell(spec, :dens; index = :i); have = (have..., :i), want = :dens_cell)
        full = prepare(spec; have, want = :dens)
        reference = full(data..., p...)
        @test [k(data..., p..., i) for i in eachindex(reference)] ≈ reference
        @test "s" in pushed(k)
        for i in (1, lastindex(reference)), name in names
            ad = prepare_ad(k, backend, data..., p..., i; active = name)
            value, dp = ad_value_and_gradient(ad, data..., p..., i)
            @test value ≈ reference[i]
            @test dp ≈ _cell_central(h -> k(data..., merge(p, NamedTuple{(name,)}(
                (getfield(p, name) + h,)))..., i), 0.0) rtol = 1e-6
        end
        k
    end
    grid(n) = (; y = cos.(1:n), x = sin.(1:n), z = collect(range(0.1, 1.0; length = n)))
    k = check(_cell_number_readers, (:y, :x, :z, :a, :b), grid(8), (; a = 0.3, b = 0.4), (:a, :b))
    allocated(n) = let d = grid(n)
        k(d..., 0.3, 0.4, 2); @allocated k(d..., 0.3, 0.4, 2)
    end
    @test allocated(30) == allocated(30_000)
    check(_cell_number_lane, (:y, :x, :a), (; y = cos.(1:8), x = sin.(1:8)), (; a = 0.3), (:a,))
    check(_cell_number_part, (:y, :x, :lookup, :u, :a),
        (; y = cos.(1:8), x = sin.(1:8), lookup = [1, 2, 3, 2, 1, 3, 3, 1], u = [0.5, -0.2]),
        (; a = 0.3), (:a,))
    d = grid(8)
    scalar = prepare(plate_cell(_cell_number_readers, :dens; index = :i);
        have = (:y, :x, :z, :a, :b, :i), want = :dens_cell)
    zero_rank = prepare(plate_cell(_cell_rank_zero, :dens; index = :i);
        have = (:y, :x, :z, :a, :i), want = :dens_cell)
    @test "s" in pushed(zero_rank)
    @test [zero_rank(d..., fill(0.7), i) for i in 1:8] ≈ [scalar(d..., 0.3, 0.4, i) for i in 1:8]
end

# A gather straight from an inline `vcat`, alone and inside a dotted
# completion, as one expression (snag `inline-vcat-gath-bb73ae09`): the cell
# lowering outlines the `vcat` into a value of its own, so the cell computes
# both at its rows, as their two-statement spelling (`stacked = vcat(...)`,
# `stacked[order]`) does. A completion also read whole stays as written.
@kernel _cell_inline_gather(y, observed, z, order::Vector{Int}, s::Float64) = begin
    drawn = exp.(z .* s)
    completed = (vcat(observed, drawn))[order]
    dens = plate(completed, y) do ci, yi
        -(yi - ci)^2 / 2
    end
    return dens
end

@kernel _cell_inline_masked(y, observed, mask, z, lookup::Vector{Int}, s::Float64) = begin
    drawn = exp.(z .* s)
    completed = observed .+ mask .* vcat(drawn, z)[lookup]
    dens = plate(completed, y) do ci, yi
        -(yi - ci)^2 / 2
    end
    return dens
end

@kernel _cell_inline_whole(y, observed, z, order::Vector{Int}, s::Float64) = begin
    drawn = exp.(z .* s)
    completed = (vcat(observed, drawn))[order]
    total = sum(completed)
    dens = plate(completed, y) do ci, yi
        -(yi - ci)^2 / 2 + total
    end
    return dens
end

@testset "plate_cell: a gather from an inline vcat runs at the cell's rows" begin
    gathered(n) = let nobs = n ÷ 2
        (; y = cos.(1:n), observed = collect(range(0.1, 0.9; length = nobs)),
           z = collect(range(-0.5, 0.4; length = n - nobs)), order = collect(n:-1:1))
    end
    masked(n) = (; y = cos.(1:n), observed = collect(range(0.1, 0.9; length = n)),
        mask = Float64[isodd(j) for j in 1:n], z = collect(range(-0.5, 0.4; length = n)),
        lookup = [mod1(3j, 2n) for j in 1:n])
    outlined(k) = ReactiveKernels._cell_outlined_pushdowns(
        ReactiveKernels._lowering_plan(k.plan))
    pushed(k) = let (_, cells) = outlined(k)
        cells === nothing ? Set{String}() :
            Set(String(cells.producer[v].outputs[1].name) for v in cells.pushed)
    end
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for (spec, have, data) in ((_cell_inline_gather, (:y, :observed, :z, :order), gathered),
                               (_cell_inline_masked, (:y, :observed, :mask, :z, :lookup), masked))
        k = prepare(plate_cell(spec, :dens; index = :i); have = (have..., :s, :i),
            want = :dens_cell)
        full = prepare(spec; have = (have..., :s), want = :dens)
        d = data(10)
        reference = full(d..., 0.7)
        @test [k(d..., 0.7, i) for i in eachindex(reference)] ≈ reference
        @test issubset(Set(["drawn", "completed_part1", "completed"]), pushed(k))
        allocated(n) = let d = data(n); k(d..., 0.7, 2); @allocated k(d..., 0.7, 2) end
        @test allocated(20) == allocated(20_000)
        @test_throws BoundsError k(d..., 0.7, 11)
        for i in (1, 7)
            ad = prepare_ad(k, backend, d..., 0.7, i; active = :s)
            value, ds = ad_value_and_gradient(ad, d..., 0.7, i)
            @test value ≈ reference[i]
            @test ds ≈ _cell_central(s -> k(d..., s, i), 0.7) atol = 1e-8
        end
    end
    # Read whole by `sum`, the completion is computed whole: nothing is outlined.
    k = prepare(plate_cell(_cell_inline_whole, :dens; index = :i);
        have = (:y, :observed, :z, :order, :s, :i), want = :dens_cell)
    full = prepare(_cell_inline_whole; have = (:y, :observed, :z, :order, :s), want = :dens)
    d = gathered(10)
    @test [k(d..., 0.7, i) for i in 1:10] ≈ full(d..., 0.7)
    lp = ReactiveKernels._lowering_plan(k.plan)
    @test first(ReactiveKernels._cell_outlined_pushdowns(lp)) === lp
end
