module StrictPartitionPrefixTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

groupmix(L, Z) = L * Z
localopaque(x) = x .^ 2
const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

function fixture(producer)
    model = quote
        a ~ Normal(0, 1)
        L ~ LKJCholesky(6, 2.0)
        Z[levels(group), 1:2] .~ Normal.(0, 1)
        eta[levels(unit)] .~ Normal.(0, 1)
        R = $producer
        # Isolate the shared-prefix defect from the pre-existing inline
        # factor/design ownership limitation (tracked by todo 0mtcwfj and
        # covered in test_partition_prefix.jl).
        local_eta = eta[unit]
        mu = a .+ local_eta .+ R[group, 1] .+ w .* R[group, 2]
        y .~ Normal.(mu, 1)
    end
    group = repeat(1:6; inner=3)
    data = (; group, unit=cld.(group, 2), w=sin.(eachindex(group)), y=cos.(eachindex(group)))
    plan = lower_rkppl(model, data; conditioned=(:y,), mod=@__MODULE__)
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.4; length=built.layout.total))
    (; bound, built, u)
end

@testset "strict partition splits shared producers before locality checks" begin
    for producer in (:(groupmix(L, Z)), :(L * Z))
        (; bound, built, u) = fixture(producer)
        # Without a local/shared declaration the existing strict guarantee
        # still applies to a producer with no row form. The split must not turn
        # into a blanket exemption. `L * Z` has one: `L` computes its rows and
        # `Z` is read whole, as the right operand of a product always is.
        query = prepare_partition(built, bound, :y; by=:unit, locality=:ignore)
        if producer == :(groupmix(L, Z))
            @test_throws ReactiveKernels.PartitionLocalityError prepare_partition(
                built, bound, :y; by=:unit, locality=:error)
        else
            strict = prepare_partition(built, bound, :y; by=:unit, locality=:error)
            @test all(l -> strict(u, l) ≈ query(u, l), (1, 2, 3))
        end
        baseline = prepare_partition_sampler(built, bound, :y, u; by=:unit,
            backend=BACKEND, compact=false, locality=:ignore)
        for compact in (true, false)
            sampler = prepare_partition_sampler(built, bound, :y, u; by=:unit,
                backend=BACKEND, compact, partitioned=:eta, locality=:error)
            @test sampler.prefix !== nothing
            cached = sampler.prefix.values
            local_positions = let e = only(e for e in built.layout.entries if e.name === :eta)
                collect(e.offset:(e.offset + e.size - 1))
            end
            @test length(local_positions) == 3
            for label in (1, 3, 2), change in (0.0, 0.2)
                point = copy(u)
                point[local_positions] .+= change
                fp = partition_footprint(sampler, label)
                local_fp = partition_footprint(sampler, label; shared=false)
                @test length(local_fp) == 1
                @test local_fp == [local_positions[label]]
                reference = zero(u)
                expected, _ = partition_value_and_gradient!(baseline, reference, point, label)
                gradient = zeros(compact ? length(fp) : length(u))
                value, _ = partition_value_and_gradient!(sampler, gradient, point, label)
                @test value ≈ expected ≈ query(point, label)
                @test gradient ≈ (compact ? reference[fp] : reference)
                @test sampler.prefix.values === cached
                local_gradient = zeros(compact ? length(local_fp) : length(u))
                local_value, _ = partition_value_and_gradient!(sampler, local_gradient,
                    point, label; shared=false)
                @test local_value ≈ value
                @test local_gradient ≈ (compact ? reference[local_fp] :
                    [i in local_fp ? reference[i] : 0.0 for i in eachindex(u)])
            end
            point = copy(u)
            point[sampler.prefix.positions] .+= 0.1
            fp = partition_footprint(sampler, 1)
            gradient = zeros(compact ? length(fp) : length(u))
            value, _ = partition_value_and_gradient!(sampler, gradient, point, 1)
            @test sampler.prefix.values !== cached
            @test value ≈ query(point, 1)
            for position in fp
                plus, minus = copy(point), copy(point)
                plus[position] += 1e-6
                minus[position] -= 1e-6
                finite = (query(plus, 1) - query(minus, 1)) / 2e-6
                slot = compact ? findfirst(==(position), fp) : position
                @test gradient[slot] ≈ finite atol=1e-5 rtol=1e-5
            end
        end
    end
end

@testset "opaque local work remains subject to strict locality" begin
    model = quote
        a ~ Normal(0, 1)
        eta[levels(unit)] .~ Normal.(0, 1)
        field = localopaque(eta)
        mu = a .+ field[unit]
        y .~ Normal.(mu, 1)
    end
    data = (; unit=[1, 1, 2, 2, 3, 3], y=collect(range(-0.2, 0.2; length=6)))
    bound = bind_data(lower_rkppl(model, data; conditioned=(:y,), mod=@__MODULE__), data)
    built = build_kernel(bound)
    u = zeros(built.layout.total)
    err = try
        prepare_partition_sampler(built, bound, :y, u; by=:unit, backend=BACKEND,
            partitioned=:eta, locality=:error)
        nothing
    catch e
        e
    end
    @test err isa Union{ReactiveKernels.PartitionLocalityError,
        ReactiveKernels.PartitionSeparabilityError}
end
end
