module PartitionPrefixBranchTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

# An opaque scalar helper reading whole parameter arrays. Its result type is
# inferred only when their containers are concrete; over the declared packed
# input (`AbstractVector{Float64}`) it is `Any`.
function wave(frequency, scales, amplitude, coefficients, coordinate)
    total = 0.0
    for j in axes(frequency, 2)
        phase = 0.0
        for d in axes(frequency, 1)
            phase += frequency[d, j] * coordinate[d] / scales[d]
        end
        total += coefficients[2j - 1] * cos(phase) + coefficients[2j] * sin(phase)
    end
    amplitude * total / sqrt(size(frequency, 2))
end

# A type guard around plate cells, as a model generator writes for an offset of
# a plate output: the live arm is decided by the cells' element type.
@kernel scalar_cells(cells) = begin
    values = eltype(cells) <: Real ? cells : throw(ArgumentError("cells must be scalars"))
    return values
end

function fixture(nstates; rows=6, F=4, D=2)
    unit = repeat(1:(nstates ÷ rows); inner=rows)
    data = (; block=ones(Int, nstates), coordinates=[cos.(s .* (1:D) ./ 5) for s in 1:nstates],
        frequency=reshape(sin.(1:(D * F)), D, F), unit, state=collect(1:nstates),
        y=sin.(1:nstates), blocks=1, width=2F, dimensions=D)
    model = quote
        z[1:width] .~ Normal.(0.0, 1.0)
        amplitude ~ Exponential(1.0)
        scales[1:dimensions] .~ LogNormal.(0.0, 1.0)
        level ~ Normal(0.0, 1.0)
        eta[levels(unit)] .~ Normal.(0.0, 1.0)
        coefficients = reshape(z, blocks, width)
        @plate for s in eachindex(block)
            field[s] = wave(frequency, scales, amplitude, coefficients[block[s], :], coordinates[s])
        end
        intensity = exp.(level .+ scalar_cells(field))
        @plate for i in eachindex(y)
            y[i] ~ Normal(intensity[state[i]] + eta[unit[i]], 1.0)
        end
    end
    bound = bind_data(lower_rkppl(model, data; conditioned=(:y,), mod=@__MODULE__), data)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.3; length=built.layout.total))
    positions(name) = let e = only(e for e in built.layout.entries if e.name === name)
        collect(e.offset:(e.offset + e.size - 1))
    end
    (; bound, built, u, positions, D)
end

# Snag state-axis-compo-353e325e: the guard's arm was not proved from the
# declared packed type, so the split lifted the whole per-state field into the
# shared prefix and every partition gradient pulled back through all states.
@testset "a type-guarded plate read at rows stays out of the shared prefix" begin
    for nstates in (24, 96)
        (; bound, built, u, positions, D) = fixture(nstates)
        query = prepare_partition(built, bound, :y; by=:unit, partitioned=:eta,
            locality=:error, u0=u)
        baseline = prepare_partition_sampler(built, bound, :y, u; by=:unit,
            backend=BACKEND, compact=false, locality=:ignore)
        sampler = prepare_partition_sampler(built, bound, :y, u; by=:unit,
            backend=BACKEND, partitioned=:eta, locality=:error)
        @test sampler.prefix !== nothing
        # The prefix caches only shared parameters, never a value per state,
        # and reads no coefficient: the field runs at each partition's rows.
        @test all(v -> length(v) <= D, sampler.prefix.values)
        @test isempty(intersect(sampler.prefix.positions, positions(:z)))
        for label in (1, nstates ÷ 12, nstates ÷ 6)
            fp = partition_footprint(sampler, label)
            @test positions(:z) ⊆ fp
            @test positions(:eta)[label] in fp
            reference = zero(u)
            expected, _ = partition_value_and_gradient!(baseline, reference, u, label)
            gradient = zeros(length(fp))
            value, _ = partition_value_and_gradient!(sampler, gradient, u, label)
            @test value ≈ expected ≈ query(u, label)
            @test gradient ≈ reference[fp]
            # Another point type re-prepares the same split and proves its own arm.
            @test query(view(copy(u), :), label) ≈ value
        end
        point = copy(u)
        point[positions(:z)] .+= 0.1
        fp = partition_footprint(sampler, 2)
        gradient = zeros(length(fp))
        partition_value_and_gradient!(sampler, gradient, point, 2)
        for (k, position) in enumerate(fp)
            plus, minus = copy(point), copy(point)
            plus[position] += 1e-6
            minus[position] -= 1e-6
            @test gradient[k] ≈ (query(plus, 2) - query(minus, 2)) / 2e-6 atol=1e-5 rtol=1e-5
        end
    end
end
end
