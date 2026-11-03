module PositionMapCellTests
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, Test
using DifferentiationInterface: AutoEnzyme, SecondOrder

const REVERSE = AutoEnzyme(; mode = Enzyme.Reverse)
const SECOND = SecondOrder(REVERSE, AutoEnzyme(; mode = Enzyme.Forward))

# Per-trajectory helpers in the shapes of an archive model: row `g` reads
# trajectory `g`'s values and shared arrays. `tick` counts the rows computed.
const ROWS = Ref(0)
tick() = (ROWS[] += 1; nothing)
cellpath(x::AbstractVector, a::Real) = hcat(a .* x, (a + 1) .* x)
enumerated(field, r, paths) = [(tick(); field[ix] .+ r[g]) for (g, ix) in enumerate(paths)]
indexed(field, r, paths) = [(tick(); field[paths[g]] .+ r[g]) for g in eachindex(paths)]
function pushed(paths, recovery)
    out = Vector{eltype(paths[1])}[]
    for g in eachindex(paths)
        tick()
        row = similar(paths[g], size(paths[g], 1))
        previous = zero(eltype(row))
        for t in eachindex(row)
            row[t] = recovery[g] * (previous + paths[g][t, 2]) / 2
            previous = paths[g][t, 2]
        end
        push!(out, row)
    end
    out
end
air_row(path, recovery) = recovery .* (vcat(zero(eltype(path)), path[1:end-1, 2]) .+ path[:, 2]) ./ 2
nested(paths, recovery) = [(tick(); air_row(paths[g], recovery[g])) for g in eachindex(paths)]
lagged(r) = [g > 1 ? r[g] - r[g - 1] : r[g] for g in eachindex(r)]

const MODELS = (
    enumerated = @rkppl(begin
        beta ~ Normal(0, 1); z[1:G] .~ Normal.(0, 1); field = exp.(beta .* s)
        background = enumerated(field, z, paths)
        @plate for h in eachindex(y); y[h] ~ Normal(background[traj[h]][hour[h]], 1.0); end
    end),
    indexed = @rkppl(begin
        beta ~ Normal(0, 1); z[1:G] .~ Normal.(0, 1); field = exp.(beta .* s)
        background = indexed(field, z, paths)
        @plate for h in eachindex(y); y[h] ~ Normal(background[traj[h]][hour[h]], 1.0); end
    end),
    pushed = @rkppl(begin
        beta ~ Normal(0, 1); z[1:G] .~ Normal.(0, 1); recovery = exp.(z)
        @plate for g in eachindex(rain_index); path[g] = cellpath(rain[rain_index[g]], z[g]); end
        air = pushed(path, recovery)
        @plate for h in eachindex(y); y[h] ~ Normal(beta + air[traj[h]][hour[h]], 1.0); end
    end),
    nested = @rkppl(begin
        beta ~ Normal(0, 1); z[1:G] .~ Normal.(0, 1); recovery = exp.(z)
        @plate for g in eachindex(rain_index); path[g] = cellpath(rain[rain_index[g]], z[g]); end
        air = nested(path, recovery)
        @plate for h in eachindex(y); y[h] ~ Normal(beta + air[traj[h]][hour[h]], 1.0); end
    end),
    archive = @rkppl(begin
        beta ~ Normal(0, 1); z[1:G] .~ Normal.(0, 1); field = exp.(beta .* s); recovery = exp.(z)
        @plate for g in eachindex(rain_index); path[g] = cellpath(rain[rain_index[g]], z[g]); end
        air = pushed(path, recovery)
        background = enumerated(field, air, paths)
        @plate for h in eachindex(y); y[h] ~ Normal(background[traj[h]][hour[h]], 1.0); end
    end))
const LAGGED = @rkppl begin
    beta ~ Normal(0, 1); z[1:G] .~ Normal.(0, 1); field = exp.(beta .* s)
    offsets = lagged(z)
    background = enumerated(field, offsets, paths)
    @plate for h in eachindex(y); y[h] ~ Normal(background[traj[h]][hour[h]], 1.0); end
end

function fixture(model; G = 4, T = 3)
    paths = [collect((g - 1) * T + 1:g * T) for g in 1:G]
    data = (; G, paths, rain_index = paths, s = collect(range(0.1, 1.2; length = G * T)),
            rain = collect(range(0.2, 2.0; length = G * T)),
            traj = repeat(1:G; inner = T), hour = repeat(1:T; outer = G))
    y = collect(range(0.5, 1.5; length = G * T))
    plan = model(; data...) | (; y)
    built = build_kernel(plan)
    names = coordinate_names(built.layout)
    beta = findfirst(==(:beta), names)
    z = [findfirst(==(Symbol("z.$g")), names) for g in 1:G]
    u = collect(range(-0.2, 0.3; length = length(names)))
    (; data, y, plan, built, beta, z, u, G)
end
finite(f, u, j; h = 1e-6) = (a = copy(u); b = copy(u); a[j] += h; b[j] -= h; (f(a) - f(b)) / 2h)

@testset "strict cell queries and outer factors agree on per-trajectory helpers" begin
    for (name, model) in pairs(MODELS)
        (; data, y, plan, built, z, u, G) = fixture(model)
        full = prepare_query(built, plan, :pointwise)(u)[:y]
        partitioned = (:z => collect(1:G),)
        q = prepare_cell_query(built, plan, :y; partitioned, locality = :error, u0 = u)
        f = prepare_cell_factors(built, plan, :y; partitioned, locality = :error, u0 = u)
        outer = prepare_cell_factor_query(f, :outer)
        @test isempty(cell_reductions(f))
        for query in (h -> q(u, h), h -> outer(u, Float64[], h))
            ROWS[] = 0
            @test [query(h) for h in eachindex(y)] ≈ full
            # Each cell computes its own trajectory's row of each helper.
            @test ROWS[] == (name === :archive ? 2 : 1) * length(y)
        end
        # Changing another trajectory's coordinate leaves a cell unchanged;
        # changing its own or a shared one does not.
        old = q(u, 1)
        u2 = copy(u); u2[z[G]] += 0.4
        @test q(u2, 1) == old
        u2[z[data.traj[1]]] += 0.3
        @test q(u2, 1) ≈ prepare_query(built, plan, :pointwise)(u2)[:y][1]
    end
end

@testset "per-trajectory helpers: footprints and native derivatives" begin
    for (name, model) in pairs(MODELS)
        (; data, y, plan, built, beta, z, u, G) = fixture(model)
        whole = prepare_cell_query(built, plan, :y; locality = :ignore)
        own(h) = sort([beta, z[data.traj[h]]])
        q = prepare_cell_query(built, plan, :y; locality = :error)
        footprint = partition_footprint_kernel(q.kernel)
        @test all(h -> Base.invokelatest(footprint, u, h).unconstrained == own(h), eachindex(y))
        ad = prepare_compact_cell_ad(prepare_ad, built, plan, :y, u, own(1), REVERSE;
                                    locality = :error)
        units = ntuple(k -> Float64[j == k for j in 1:2], 2)
        hvp = prepare_compact_cell_ad(prepare_ad_hvp, built, plan, :y, u, own(1), SECOND,
                                     units; locality = :error)
        partitioned = (:z => collect(1:G),)
        local_ad = prepare_compact_cell_ad(prepare_ad, built, plan, :y, u, [z[1]], REVERSE;
                                          partitioned, locality = :error)
        factors = prepare_cell_factors(built, plan, :y; partitioned, locality = :error, u0 = u)
        factor_ad = prepare_cell_factor_ad(prepare_ad, factors, :outer, u, [z[1]], REVERSE)
        for h in (1, 5, length(y))
            S = own(h)
            value, gradient = ad_value_and_gradient(ad, u, u[S], S, h)
            @test value ≈ whole(u, h)
            @test gradient ≈ [finite(p -> whole(p, h), u, j) for j in S] rtol = 1e-6 atol = 1e-8
            g, columns = ad_gradient_and_hvp(hvp, units, u, u[S], S, h)
            @test g ≈ gradient
            second(p) = ad_value_and_gradient(ad, p, p[S], S, h)[2]
            @test reduce(hcat, columns) ≈ reduce(hcat, [finite(second, u, j; h = 1e-5) for j in S]) rtol = 1e-5 atol = 1e-6
            L = [z[data.traj[h]]]
            value, gradient = ad_value_and_gradient(local_ad, u, u[L], L, h)
            @test value ≈ whole(u, h)
            @test gradient ≈ [finite(p -> whole(p, h), u, only(L))] rtol = 1e-6 atol = 1e-8
            value, (sums, gradient) = ad_value_and_gradient(factor_ad, u, Float64[], u[L], L, h)
            @test value ≈ whole(u, h)
            @test isempty(sums)
            @test gradient ≈ [finite(p -> whole(p, h), u, only(L))] rtol = 1e-6 atol = 1e-8
        end
        # Shared coordinates through one prefix pullback.
        pullback = prepare_cell_pullback(local_ad, REVERSE, u)
        accumulator = prepare_cell_accumulator(local_ad, REVERSE, u; order = 0)
        weights = collect(range(0.3, 1.1; length = length(y)))
        gradient = zeros(length(u))
        cell_pullback!(gradient, pullback, u) do batch
            for h in eachindex(y)
                L = [z[data.traj[h]]]
                accumulate_cell!(batch, accumulator, nothing, weights[h], u[L], L, h)
            end
        end
        total(p) = sum(weights[h] * whole(p, h) for h in eachindex(y))
        @test gradient ≈ [finite(total, u, j) for j in eachindex(u)] rtol = 1e-6 atol = 1e-7
    end
end

@testset "a cross-row argument keeps both strict paths refused" begin
    (; plan, built, u, G) = fixture(LAGGED)
    partitioned = (:z => collect(1:G),)
    @test_throws PartitionLocalityError prepare_cell_query(built, plan, :y; partitioned,
        locality = :error, u0 = u)
    @test_throws PartitionLocalityError prepare_cell_factor_query(
        prepare_cell_factors(built, plan, :y; partitioned, locality = :error, u0 = u), :outer)
end

# The paired random-Fourier-feature field evaluator of an archive model: a
# loop over the rows of a coordinate matrix that looks each row's block of
# coefficients up in a prologue product and accumulates a filled `n × k`
# result, read through `vec` (`src/cell_position_maps.jl`).
function rff(coordinates, block, z, frequency, lengths, sd, L)
    features = size(frequency, 2); channels = size(L, 1)
    coefficients = L * reshape(z, channels, :)
    out = zeros(promote_type(eltype(z), eltype(lengths), typeof(sd), eltype(L)),
                size(coordinates, 1), channels)
    scale = sd / sqrt(features)
    for i in axes(coordinates, 1)
        tick()
        offset = (block[i] - 1) * 2features
        for j in 1:features
            phase = zero(sd)
            for d in axes(coordinates, 2)
                phase += frequency[d, j] * coordinates[i, d] / lengths[d]
            end
            s, c = sincos(phase)
            for k in 1:channels
                out[i, k] += coefficients[k, offset + 2j - 1] * c + coefficients[k, offset + 2j] * s
            end
        end
        for k in 1:channels
            out[i, k] *= scale
        end
    end
    return out
end
# Each row adds the previous row's value: cross-row.
function rff_lagged(coordinates, block, z, frequency, lengths, sd, L)
    out = zeros(promote_type(eltype(z), typeof(sd)), size(coordinates, 1), size(L, 1))
    for i in axes(coordinates, 1)
        out[i, 1] = sd * coordinates[i, 1] + (i > 1 ? out[i - 1, 1] : zero(sd))
    end
    out
end

const RFF_MODELS = (
    one = @rkppl(begin
        beta ~ Normal(0, 1)
        w[1:G] .~ Normal.(0, 1)
        rain_z[1:ncoef] .~ Normal.(0.0, 1.0)
        rain_sd ~ Exponential(1.0)
        rain_length[1:D] .~ LogNormal.(0.0, 0.5)
        rain_matrix = rff(coords, block, rain_z, freq, rain_length, rain_sd, corr)
        rain_field = vec(rain_matrix)
        rain_mm = exp.(beta .+ rain_field)
        @plate for h in eachindex(y)
            y[h] ~ Normal(w[grp[h]] + rain_mm[site[h]], 1.0)
        end
    end),
    # Two channels: each cell reads both columns of its row through `vec`.
    two = @rkppl(begin
        w[1:G] .~ Normal.(0, 1)
        rain_z[1:ncoef] .~ Normal.(0.0, 1.0)
        rain_sd ~ Exponential(1.0)
        rain_length[1:D] .~ LogNormal.(0.0, 0.5)
        rain_matrix = rff(coords, block, rain_z, freq, rain_length, rain_sd, corr)
        rain_field = vec(rain_matrix)
        @plate for h in eachindex(y)
            y[h] ~ Normal(w[grp[h]] + rain_field[site[h]] + rain_field[site2[h]], 1.0)
        end
    end))
const RFF_LAGGED = @rkppl begin
    w[1:G] .~ Normal.(0, 1)
    rain_z[1:ncoef] .~ Normal.(0.0, 1.0)
    rain_sd ~ Exponential(1.0)
    rain_length[1:D] .~ LogNormal.(0.0, 0.5)
    rain_matrix = rff_lagged(coords, block, rain_z, freq, rain_length, rain_sd, corr)
    rain_field = vec(rain_matrix)
    @plate for h in eachindex(y)
        y[h] ~ Normal(w[grp[h]] + rain_field[site[h]], 1.0)
    end
end

function rff_fixture(model; channels = 1, N = 6, G = 3, D = 2, features = 4, blocks = 2)
    coords = [0.3 * i + 0.1 * d for i in 1:N, d in 1:D]
    block = [1 + (i - 1) * blocks ÷ N for i in 1:N]
    freq = [sin(1.0 + d + 2j) for d in 1:D, j in 1:features]
    corr = channels == 1 ? fill(1.0, 1, 1) : [1.0 0.0; 0.4 0.9]
    ncoef = channels * 2features * blocks
    site = [3, 1, 6, 2, 5, 4]
    grp = [1 + (i - 1) * G ÷ N for i in 1:N]
    y = collect(range(0.2, 1.4; length = N))
    data = (; G, ncoef, D, coords, block, freq, corr, grp, site)
    channels == 2 && (data = merge(data, (; site2 = site .+ N)))
    plan = model(; data...) | (; y)
    built = build_kernel(plan)
    names = coordinate_names(built.layout)
    u = collect(range(-0.3, 0.4; length = length(names)))
    zs = [k for (k, n) in enumerate(names) if startswith(string(n), "rain_z")]
    (; plan, built, names, u, zs, y, G)
end

@testset "a matrix-row field evaluator and its vec in strict cell queries" begin
    for (channels, model) in ((1, RFF_MODELS.one), (2, RFF_MODELS.two))
        (; plan, built, u, zs, y, G) = rff_fixture(model; channels)
        whole = prepare_cell_query(built, plan, :y; locality = :ignore)
        full = prepare_query(built, plan, :pointwise)(u)[:y]
        # Unpartitioned: the field is computed at each cell's rows only.
        q = prepare_cell_query(built, plan, :y; locality = :error)
        ROWS[] = 0
        @test [q(u, h) for h in eachindex(y)] ≈ full
        @test ROWS[] == length(y)
        entries = partition_locality(q.kernel)
        @test all(v -> any(e -> e.value === v && e.status === :rows, entries),
                  (:rain_matrix, :rain_field))
        # Partitioned coefficients: each cell's field row reads them whole.
        partitioned = (:rain_z => collect(1:length(zs)),)
        qz = prepare_cell_query(built, plan, :y; partitioned, locality = :error, u0 = u)
        ROWS[] = 0
        @test [qz(u, h) for h in eachindex(y)] ≈ full
        @test ROWS[] == length(y)
        # The coefficients are read whole, so every one is in a cell's support.
        fp = Base.invokelatest(partition_footprint_kernel(q.kernel), u, 1).unconstrained
        @test issubset(zs, fp)
        ad = prepare_compact_cell_ad(prepare_ad, built, plan, :y, u, zs, REVERSE;
                                     partitioned, locality = :error)
        for h in (1, length(y))
            value, gradient = ad_value_and_gradient(ad, u, u[zs], zs, h)
            @test value ≈ whole(u, h)
            @test gradient ≈ [finite(p -> whole(p, h), u, j) for j in zs] rtol = 1e-6 atol = 1e-8
        end
        factors = prepare_cell_factors(built, plan, :y; partitioned, locality = :error, u0 = u)
        outer = prepare_cell_factor_query(factors, :outer)
        @test [outer(u, Float64[], h) for h in eachindex(y)] ≈ full
        # Partitioned local effects: the field is a shared value.
        qw = prepare_cell_query(built, plan, :y; partitioned = (:w => collect(1:G),),
                                locality = :error, u0 = u)
        @test [qw(u, h) for h in eachindex(y)] ≈ full
    end
end

@testset "a cross-row matrix-row helper keeps strict locality checks" begin
    (; plan, built, u, zs) = rff_fixture(RFF_LAGGED)
    @test_throws PartitionLocalityError prepare_cell_query(built, plan, :y; locality = :error)
    @test_throws PartitionLocalityError prepare_cell_query(built, plan, :y;
        partitioned = (:rain_z => collect(1:length(zs)),), locality = :error, u0 = u)
end
end
