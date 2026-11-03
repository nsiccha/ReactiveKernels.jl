module PositionMapRowTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test
const RK = ReactiveKernels

# Helpers whose row `g` reads their per-position arguments only at `g`
# (`src/cell_position_maps.jl`). `field` is read through `paths[g]`, so it is
# passed whole; `r`, `paths` and `recovery` are read at the row.
enumerated(field, r, paths) = [field[ix] .+ r[g] for (g, ix) in enumerate(paths)]
indexed(field, r, paths) = [field[paths[g]] .+ r[g] for g in eachindex(paths)]
wrapped(field, r, paths) = enumerated(field, r, paths)
function pushed(paths, recovery)
    out = Vector{eltype(paths[1])}[]
    for g in eachindex(paths)
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
nested(paths, recovery) = [air_row(paths[g], recovery[g]) for g in eachindex(paths)]
function preallocated(x, w)
    T = promote_type(eltype(x), eltype(w))
    out = zeros(T, length(x))
    for i in 1:length(x)
        s = zero(T)
        if w[i] > 0
            s = x[i] * w[i]
        end
        out[i] = s
    end
    out
end
zipped(a, b, c) = [x * y + c[i] for (i, (x, y)) in enumerate(zip(a, b))]
mapped(x, s) = map(v -> exp(s * v), x)

# Cross-row or carried work: no row form.
crossrow(x) = [x[g] - x[g - 1] for g in 2:length(x)]
lagged(x) = [g > 1 ? x[g] - x[g - 1] : x[g] for g in eachindex(x)]
normalized(x) = [x[g] / length(x) for g in eachindex(x)]
function cumulative(x)
    out = Float64[]
    s = 0.0
    for g in eachindex(x)
        s += x[g]
        push!(out, s)
    end
    out
end
function counting(x)
    out = Float64[]
    for g in eachindex(x)
        push!(out, x[g] + length(out))
    end
    out
end
function skipping(x)
    out = Float64[]
    for g in eachindex(x)
        x[g] > 0 || continue
        push!(out, x[g])
    end
    out
end
positional(x) = [x[g] * g for g in eachindex(x)]
# An array argument reduced or broadcast whole in each row is cross-row work.
reduced(x, w) = [x[g] + sum(w) for g in eachindex(x)]
spread(x, w) = [x[g] .* w for g in eachindex(x)]
looked_up(x, w, ix) = [x[g] + w[ix[g]] for g in eachindex(x)]

@testset "position-map proofs: accepted shapes and refusals" begin
    V, S = Vector{Float64}, typeof(view(zeros(5), 2:4))
    M, VI = Vector{Matrix{Float64}}, Vector{Vector{Int}}
    proof(f, T...) = RK._cell_position_proof(typeof(f), Any[T...])
    p = proof(enumerated, V, S, VI)
    @test p.positions == [2, 3] && p.sources == [3] && isempty(p.checks)
    @test proof(indexed, V, S, VI).positions == [2, 3]
    @test proof(wrapped, Any, S, VI).positions == [2, 3]
    @test proof(pushed, M, V).positions == [1, 2]
    @test proof(nested, M, V).sources == [1]
    @test proof(preallocated, V, V).positions == [1, 2]
    @test proof(zipped, V, V, V).sources == [1, 2]
    @test proof(mapped, V, Float64).positions == [1]
    # An abstract argument type keeps the proof, with a run-time check of the
    # original producer's type.
    @test proof(pushed, AbstractVector, V).checks == [(1, :eltype), (1, :standard)]
    @test proof(looked_up, V, V, Vector{Int}).positions == [1, 3]
    for (f, T) in ((crossrow, (V,)), (lagged, (V,)), (normalized, (V,)),
                   (cumulative, (V,)), (counting, (V,)), (skipping, (V,)),
                   (positional, (V,)), (reduced, (V, V)), (spread, (V, V)))
        @test proof(f, T...) === nothing
    end
end

@kernel background_rows(z::AbstractVector{Float64}, beta::Float64, s::Vector{Float64},
                        paths::Vector{Vector{Int}}, traj::Vector{Int}, hour::Vector{Int},
                        y::Vector{Float64}) = begin
    field = exp.(beta .* s)
    background = enumerated(field, z, paths)
    density = plate(traj, hour, y) do t, h, v
        -(v - background[t][h])^2 / 2
    end
end
@kernel indexed_rows(z::AbstractVector{Float64}, beta::Float64, s::Vector{Float64},
                     paths::Vector{Vector{Int}}, traj::Vector{Int}, hour::Vector{Int},
                     y::Vector{Float64}) = begin
    field = exp.(beta .* s)
    background = indexed(field, z, paths)
    density = plate(traj, hour, y) do t, h, v
        -(v - background[t][h])^2 / 2
    end
end
@kernel archive_rows(z::AbstractVector{Float64}, beta::Float64, s::Vector{Float64},
                     rain::Vector{Float64}, paths::Vector{Vector{Int}}, traj::Vector{Int},
                     hour::Vector{Int}, y::Vector{Float64}) = begin
    field = exp.(beta .* s)
    recovery = exp.(z)
    path = plate(paths, z) do ix, a
        hcat(a .* rain[ix], (a + 1) .* rain[ix])
    end
    air = pushed(path, recovery)
    background = enumerated(field, air, paths)
    density = plate(traj, hour, y) do t, h, v
        -(v - background[t][h])^2 / 2
    end
end
@kernel nested_rows(z::AbstractVector{Float64}, beta::Float64, rain::Vector{Float64},
                    paths::Vector{Vector{Int}}, traj::Vector{Int}, hour::Vector{Int},
                    y::Vector{Float64}) = begin
    recovery = exp.(z)
    path = plate(paths, z) do ix, a
        hcat(a .* rain[ix], (a + 1) .* rain[ix])
    end
    air = nested(path, recovery)
    density = plate(traj, hour, y) do t, h, v
        -(v - beta - air[t][h])^2 / 2
    end
end

function fixture(G = 4, T = 3)
    paths = [collect((g - 1) * T + 1:g * T) for g in 1:G]
    (; G, paths, s = collect(range(0.1, 1.2; length = G * T)),
       rain = collect(range(0.2, 2.0; length = G * T)),
       traj = repeat(1:G; inner = T), hour = repeat(1:T; outer = G),
       y = collect(range(0.5, 1.5; length = G * T)))
end
finite(f, x, j; h = 1e-6) = (a = copy(x); b = copy(x); a[j] += h; b[j] -= h; (f(a) - f(b)) / 2h)

@testset "position maps run at a cell's rows with native Reverse derivatives" begin
    d = fixture()
    z, beta = collect(range(-0.3, 0.4; length = d.G)), 0.35
    cases = (
        (background_rows, (:z, :beta, :s, :paths, :traj, :hour, :y), (; d.s, d.paths, d.traj, d.hour, d.y)),
        (indexed_rows, (:z, :beta, :s, :paths, :traj, :hour, :y), (; d.s, d.paths, d.traj, d.hour, d.y)),
        (archive_rows, (:z, :beta, :s, :rain, :paths, :traj, :hour, :y),
            (; d.s, d.rain, d.paths, d.traj, d.hour, d.y)),
        (nested_rows, (:z, :beta, :rain, :paths, :traj, :hour, :y),
            (; d.rain, d.paths, d.traj, d.hour, d.y)))
    for (kernel, ports, bound) in cases
        spec = plate_cell(kernel, :density; index = :cell, locality = :error)
        k = prepare(spec; have = (ports..., :cell), want = :density_cell, bound)
        whole = prepare(kernel; have = ports, want = :density, bound)
        rows = [entry for entry in partition_locality(k) if entry.status === :rows]
        @test any(entry -> entry.value in (:background, :air), rows)
        ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), z, beta, 1; active = :z)
        for cell in eachindex(d.y)
            expected = whole(z, beta)[cell]
            @test k(z, beta, cell) ≈ expected
            value, gradient = ad_value_and_gradient(ad, z, beta, cell)
            @test value ≈ expected
            @test gradient ≈ [finite(x -> whole(x, beta)[cell], z, j) for j in eachindex(z)] rtol = 1e-6 atol = 1e-8
            @test all(iszero, gradient[setdiff(eachindex(z), d.traj[cell])])
        end
    end
end

@kernel partitioned_rows(z::AbstractVector{Float64}, beta::Float64, s::Vector{Float64},
                         rain::Vector{Float64}, paths::Vector{Vector{Int}}, traj::Vector{Int},
                         hour::Vector{Int}, y::Vector{Float64}, label::Vector{Int}) = begin
    field = exp.(beta .* s)
    recovery = exp.(z)
    path = plate(paths, z) do ix, a
        hcat(a .* rain[ix], (a + 1) .* rain[ix])
    end
    air = pushed(path, recovery)
    background = enumerated(field, air, paths)
    density = plate(traj, hour, y) do t, h, v
        -(v - background[t][h])^2 / 2
    end
end
@testset "position maps in partitions: several, repeated and reordered rows" begin
    d = fixture()
    z, beta = collect(range(-0.3, 0.4; length = d.G)), 0.35
    ports = (:z, :beta, :s, :rain, :paths, :traj, :hour, :y)
    label = [2, 1, 2, 1, 4, 4, 1, 2, 2, 4, 1, 1]
    bound = (; d.s, d.rain, d.paths, d.traj, d.hour, d.y, label)
    spec = plate_partition(partitioned_rows, :density; by = :label, index = :which,
                           locality = :error)
    k = prepare(spec; have = (ports..., :label, :which), want = :partition_total, bound)
    whole = prepare(partitioned_rows; have = (ports..., :label), want = :density, bound)
    ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), z, beta, 1; active = :z)
    for which in (1, 2, 4)
        members = findall(==(which), label)
        expected = sum(whole(z, beta)[members]; init = 0.0)
        @test k(z, beta, which) ≈ expected
        value, gradient = ad_value_and_gradient(ad, z, beta, which)
        @test value ≈ expected
        @test gradient ≈ [finite(x -> sum(whole(x, beta)[members]; init = 0.0), z, j)
                          for j in eachindex(z)] rtol = 1e-6 atol = 1e-8
    end
end

@kernel selected_rows(z::AbstractVector{Float64}, rain::Vector{Float64},
                      paths::Vector{Vector{Int}}, selections::Vector{Vector{Int}}) = begin
    recovery = exp.(z)
    path = plate(paths, z) do ix, a
        hcat(a .* rain[ix], (a + 1) .* rain[ix])
    end
    air = pushed(path, recovery)
    density = plate(selections) do selected
        sum(sum, air[selected]; init = 0.0)
    end
end
@testset "a cell reading no rows of a loop helper with a typed prologue" begin
    d = fixture()
    z = collect(range(-0.3, 0.4; length = d.G))
    # `pushed` reads `paths[1]` for its element type; a window of no rows
    # must not call it.
    selections = [Int[], [3, 1], [2, 2, 4]]
    bound = (; d.rain, d.paths, selections)
    spec = plate_cell(selected_rows, :density; index = :cell, locality = :error)
    k = prepare(spec; have = (:z, :rain, :paths, :selections, :cell),
                want = :density_cell, bound)
    whole = prepare(selected_rows; have = (:z, :rain, :paths, :selections),
                    want = :density, bound)
    ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), z, 1; active = :z)
    for cell in eachindex(selections)
        expected = whole(z)[cell]
        @test k(z, cell) ≈ expected
        value, gradient = ad_value_and_gradient(ad, z, cell)
        @test value ≈ expected
        @test gradient ≈ [finite(x -> whole(x)[cell], z, j) for j in eachindex(z)] rtol = 1e-6 atol = 1e-8
    end
    @test k(z, 1) == 0.0
end

@kernel crossrow_rows(z::AbstractVector{Float64}, traj::Vector{Int}) = begin
    field = lagged(z)
    density = plate(traj) do t
        field[t]^2
    end
end
@kernel counted_rows(z::AbstractVector{Float64}, traj::Vector{Int}) = begin
    field = counting(z)
    density = plate(traj) do t
        field[t]^2
    end
end
@testset "cross-row and carried helpers keep strict locality checks" begin
    for kernel in (crossrow_rows, counted_rows)
        spec = plate_cell(kernel, :density; index = :cell, locality = :error)
        @test_throws PartitionLocalityError prepare(spec; have = (:z, :traj, :cell),
            want = :density_cell, bound = (traj = [3, 1, 2],))
    end
end

# A recording vector whose iteration is Base's: proven at run time through an
# abstract port. One that iterates in another order is refused there.
struct Recorded <: AbstractVector{Int}
    values::Vector{Int}
end
Base.size(x::Recorded) = size(x.values)
Base.getindex(x::Recorded, i::Int) = x.values[i]
struct Reversed <: AbstractVector{Int}
    values::Vector{Int}
end
Base.size(x::Reversed) = size(x.values)
Base.getindex(x::Reversed, i::Int) = x.values[i]
Base.iterate(x::Reversed, state = length(x)) = state == 0 ? nothing : (x.values[state], state - 1)
shifted(z, ids) = [z[g] + id for (g, id) in enumerate(ids)]
@kernel abstract_rows(z::AbstractVector{Float64}, ids::AbstractVector{Int},
                      traj::Vector{Int}) = begin
    field = shifted(z, ids)
    density = plate(traj) do t
        field[t]^2
    end
end
@testset "abstract iterated arguments are checked at run time" begin
    z = [0.2, -0.4, 0.7]
    spec = plate_cell(abstract_rows, :density; index = :cell, locality = :error)
    k = prepare(spec; have = (:z, :ids, :traj, :cell), want = :density_cell,
                bound = (traj = [3, 1, 2],))
    @test k(z, Recorded([4, 5, 6]), 1) ≈ (z[3] + 6)^2
    @test_throws ArgumentError k(z, Reversed([4, 5, 6]), 1)
end
end
