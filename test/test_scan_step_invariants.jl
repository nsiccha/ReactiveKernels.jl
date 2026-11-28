module ScanStepInvariantTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# A scan step that reads its series through a gather of shared values,
# `mu[rows][t]`: the gather reads neither the carry nor the step's element,
# so the native scan computes it once instead of at every step, and one value
# costs time and memory linear in the scan's length.
@kernel gathered(z::Vector{Float64}, rows::Vector{Int}, steps::Vector{Int},
        phi::Float64) = begin
    mu = exp.(z)
    path = scan(steps; init = 0.0) do carry, t
        next = phi * carry + mu[rows][t]
        (next, next)
    end
    total = sum(path)
    return total
end

# The same gather per plate cell, the shape `@rkppl` emits for one observed
# `@scan` per `@plate` cell that reads `mu[rows[g]][t]`.
@kernel gathered_cells(z::Vector{Float64}, rows::Vector{Vector{Int}},
        steps::Vector{Vector{Int}}, positions::Vector{Int}, phi::Float64) = begin
    mu = exp.(z)
    cells = plate(positions) do g
        path = scan(steps[g]; init = 0.0) do carry, t
            next = phi * carry + mu[rows[g]][t] - mu[rows[g]][max(t - 1, 1)]
            (next, next)
        end
        sum(path)
    end
    total = sum(cells)
    return total
end

# A shared condition selects a lazy arm that reads the gather: the arm, moved
# out of the steps, still runs only when its branch is taken.
@kernel guarded(z::Vector{Float64}, rows::Vector{Int}, steps::Vector{Int},
        use::Bool) = begin
    mu = exp.(z)
    path = scan(steps; init = 0.0) do carry, t
        level = use ? sum(mu[rows]) : 1.0
        next = carry + level * t
        (next, next)
    end
    total = sum(path)
    return total
end

@kernel with_history(z::Vector{Float64}, rows::Vector{Int}, steps::Vector{Int}) = begin
    mu = exp.(z)
    path = scan(steps; init = 0.0, history = 0.0) do carry, t, earlier
        next = (t == 1 ? 0.0 : earlier[t - 1]) + mu[rows][t]
        (next, next)
    end
    total = sum(path)
    return total
end

function reference(z, rows, steps, phi)
    mu = exp.(z)
    carry, total = 0.0, 0.0
    for t in steps
        carry = phi * carry + mu[rows][t]
        total += carry
    end
    total
end

function reference_cells(z, rows, steps, phi)
    mu = exp.(z)
    total = 0.0
    for g in eachindex(rows)
        carry = 0.0
        for t in steps[g]
            carry = phi * carry + mu[rows[g]][t] - mu[rows[g]][max(t - 1, 1)]
            total += carry
        end
    end
    total
end

const AD = AutoEnzyme(; mode = Enzyme.Reverse)

@testset "a scan step computes a shared gather once" begin
    allocated, differentiated = Int[], Int[]
    for n in (500, 4000)
        z = sin.(1:n) ./ 10
        rows = collect(n:-1:1)
        steps = collect(1:n)
        k = prepare(gathered; have = (:z, :rows, :steps, :phi), want = :total)
        @test k(z, rows, steps, 0.3) ≈ reference(z, rows, steps, 0.3)
        k(z, rows, steps, 0.3)
        push!(allocated, @allocated k(z, rows, steps, 0.3))
        ad = prepare_ad(k, AD, z, rows, steps, 0.3; active = :z)
        value, gradient = ad_value_and_gradient(ad, z, rows, steps, 0.3)
        @test value ≈ reference(z, rows, steps, 0.3)
        @test gradient ≈ DifferentiationInterface.gradient(
            x -> reference(x, rows, steps, 0.3), AD, z)
        push!(differentiated, @allocated ad_value_and_gradient(ad, z, rows, steps, 0.3))
    end
    # A length-n gather per step would allocate 8n² bytes: 128 MB at
    # n = 4000. Once per scan, the eightfold longer scan allocates about
    # eight times as much, for the value and for its gradient.
    @test allocated[2] <= 10 * allocated[1] + 4096
    @test allocated[2] < 8 * 4000 * 64
    @test differentiated[2] <= 10 * differentiated[1] + 4096
    @test differentiated[2] < 8 * 4000 * 64
end

@testset "a plate cell's scan computes its gather once" begin
    allocated, differentiated = Int[], Int[]
    for n in (500, 4000)
        z = cos.(1:2n) ./ 10
        rows = [collect(1:n), collect(2n:-1:(n + 1))]
        steps = [collect(1:n), collect(1:n)]
        args = (z, rows, steps, [1, 2], 0.4)
        k = prepare(gathered_cells; have = (:z, :rows, :steps, :positions, :phi),
            want = :total)
        @test k(args...) ≈ reference_cells(z, rows, steps, 0.4)
        k(args...)
        push!(allocated, @allocated k(args...))
        ad = prepare_ad(k, AD, args...; active = :z)
        value, gradient = ad_value_and_gradient(ad, args...)
        @test value ≈ reference_cells(z, rows, steps, 0.4)
        @test gradient ≈ DifferentiationInterface.gradient(
            x -> reference_cells(x, rows, steps, 0.4), AD, z)
        push!(differentiated, @allocated ad_value_and_gradient(ad, args...))
    end
    @test allocated[2] <= 10 * allocated[1] + 4096
    @test allocated[2] < 2 * 8 * 4000 * 64
    @test differentiated[2] <= 10 * differentiated[1] + 4096
    @test differentiated[2] < 2 * 8 * 4000 * 64
end

@testset "an empty scan evaluates no moved statement" begin
    k = prepare(gathered; have = (:z, :rows, :steps, :phi), want = :total)
    # `mu[rows]` would throw: no step runs, so nothing reads it.
    @test k([0.1, 0.2], [99], Int[], 0.3) == 0.0
    @test_throws BoundsError k([0.1, 0.2], [99], [1], 0.3)
    h = prepare(with_history; have = (:z, :rows, :steps), want = :total)
    @test h([0.1, 0.2], [99], Int[]) == 0.0
    @test_throws BoundsError h([0.1, 0.2], [99], [1])
end

@testset "a moved lazy arm runs only when taken" begin
    k = prepare(guarded; have = (:z, :rows, :steps, :use), want = :total)
    steps = collect(1:5)
    @test k([0.1, 0.2], [99], steps, false) == sum(cumsum(steps))
    @test_throws BoundsError k([0.1, 0.2], [99], steps, true)
    @test k([0.1, 0.2], [2, 1], steps, true) ≈ (exp(0.1) + exp(0.2)) * sum(cumsum(steps))
end

@testset "a history scan computes its shared gather once" begin
    allocated = Int[]
    for n in (500, 4000)
        z = sin.(1:n) ./ 10
        rows = collect(n:-1:1)
        steps = collect(1:n)
        k = prepare(with_history; have = (:z, :rows, :steps), want = :total)
        @test k(z, rows, steps) ≈ sum(cumsum(exp.(z)[rows]))
        k(z, rows, steps)
        push!(allocated, @allocated k(z, rows, steps))
    end
    @test allocated[2] <= 10 * allocated[1] + 4096
end
end
