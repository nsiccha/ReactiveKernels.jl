module PartitionLocalStorageTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

# An ordinary shared computation with no row lowering. The local derivative
# must read its cached rows without allocating a cotangent for the whole field.
sharedfield(a, w) = a .* w
const MODEL = quote
    a ~ Normal(0, 1)
    z[levels(group)] .~ Normal.(0, 1)
    field = sharedfield(a, w)
    local_z = z[group]
    mu = field .+ local_z
    y .~ Normal.(mu, 1)
end

@testset "local gradients keep shared frontier storage inactive" begin
    allocations = Int[]
    for n in (6, 6000)
        # Unequal local widths exercise padding without activating a shared
        # coordinate in the shorter partition.
        data = (; group=[1, 1, repeat([2, 3], (n-2) ÷ 2)...],
            unit=[1, 1, fill(2, n-2)...], w=sin.(1:n), y=cos.(1:n))
        bound = bind_data(lower_rkppl(MODEL, data; conditioned=(:y,), mod=@__MODULE__), data)
        built = build_kernel(bound)
        u = collect(range(-0.2, 0.3; length=built.layout.total))
        declaration = (:z => [1, 2, 2],)
        query = prepare_partition(built, bound, :y; by=:unit, partitioned=declaration)
        for compact in (true, false)
            sampler = prepare_partition_sampler(built, bound, :y, u; by=:unit,
                partitioned=declaration, compact, backend=AutoEnzyme(; mode=Enzyme.Reverse))
            @test any(v -> v isa AbstractVector && length(v) == n, sampler.prefix.values)
            if compact && n == 6
                # The ordinary residual still requires active frontier ports:
                # a conditional local preparation does not relax that guard.
                ad = ReactiveKernelsPPL._partition_ad(sampler.ad, 1)
                err = try
                    prepare_ad(ad.kernel, AutoEnzyme(; mode=Enzyme.Reverse), u,
                        sampler.active, sampler.slots, 1, sampler.plans[1], sampler.prefix.values...;
                        active=:_ppl_cell_active)
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("transitively downstream", sprint(showerror, err))
            end
            for label in (1, 2, 1), change in (0.0, 0.1)
                point = copy(u)
                point[sampler.prefix.positions] .+= change
                fp = partition_footprint(sampler, label)
                local_fp = partition_footprint(sampler, label; shared=false)
                full = zeros(compact ? length(fp) : length(u))
                local_gradient = zeros(compact ? length(local_fp) : length(u))
                value, _ = partition_value_and_gradient!(sampler, full, point, label)
                local_value, _ = partition_value_and_gradient!(sampler, local_gradient,
                    point, label; shared=false)
                @test local_value ≈ value ≈ query(point, label)
                expected = compact ? full[indexin(local_fp, fp)] :
                    [i in local_fp ? full[i] : 0.0 for i in eachindex(u)]
                @test local_gradient ≈ expected
                for (slot, position) in enumerate(local_fp)
                    plus, minus = copy(point), copy(point)
                    plus[position] += 1e-6; minus[position] -= 1e-6
                    finite = (query(plus, label) - query(minus, label)) / 2e-6
                    @test local_gradient[compact ? slot : position] ≈ finite rtol=1e-6 atol=1e-6
                end
            end
            if compact
                gradient = zeros(length(partition_footprint(sampler, 1; shared=false)))
                partition_value_and_gradient!(sampler, gradient, u, 1; shared=false)
                push!(allocations, @allocated partition_value_and_gradient!(
                    sampler, gradient, u, 1; shared=false))
            end
        end
    end
    @test allocations[2] <= allocations[1] + 1024
    @test allocations[2] < 6000sizeof(Float64)
end

@testset "fixed local width stays compact as the packed panel grows" begin
    allocations = Int[]
    for n in (6, 600)
        units = repeat(1:(n ÷ 2); inner=2)
        data = (; group=collect(1:n), unit=units, w=sin.(1:n), y=cos.(1:n))
        bound = bind_data(lower_rkppl(MODEL, data; conditioned=(:y,), mod=@__MODULE__), data)
        built = build_kernel(bound)
        u = collect(range(-0.2, 0.3; length=built.layout.total))
        declaration = (:z => units,)
        query = prepare_partition(built, bound, :y; by=:unit, partitioned=declaration)
        sampler = prepare_partition_sampler(built, bound, :y, u; by=:unit,
            partitioned=declaration, compact=true, backend=AutoEnzyme(; mode=Enzyme.Reverse))
        @test length(sampler.prefix.local_state.active) == 2
        gradient = zeros(2)
        for label in (n ÷ 2, 1)
            value, _ = partition_value_and_gradient!(sampler, gradient, u, label; shared=false)
            @test value ≈ query(u, label)
            for (slot, position) in enumerate(partition_footprint(sampler, label; shared=false))
                plus, minus = copy(u), copy(u)
                plus[position] += 1e-6
                minus[position] -= 1e-6
                @test gradient[slot] ≈ (query(plus, label) - query(minus, label)) / 2e-6 atol=1e-6
            end
        end
        push!(allocations, @allocated partition_value_and_gradient!(sampler,
            gradient, u, 1; shared=false))
    end
    @test allocations[2] <= allocations[1] + 1024
end

@testset "local gradients read shared coordinates left in a mixed parameter block" begin
    data = (; group=[1, 2, 3, 2], unit=[1, 1, 2, 2], w=sin.(1:4), y=cos.(1:4))
    bound = bind_data(lower_rkppl(MODEL, data; conditioned=(:y,), mod=@__MODULE__), data)
    built = build_kernel(bound)
    u = collect(range(-0.2, 0.3; length=built.layout.total))
    declaration = (:z => Union{Missing,Int}[1, missing, 2],)
    for compact in (true, false)
        sampler = prepare_partition_sampler(built, bound, :y, u; by=:unit,
            partitioned=declaration, compact, backend=AutoEnzyme(; mode=Enzyme.Reverse))
        for label in (1, 2), change in (0.0, 0.15)
            point = copy(u)
            point[sampler.prefix.positions] .+= change
            fp = partition_footprint(sampler, label)
            local_fp = partition_footprint(sampler, label; shared=false)
            full = zeros(compact ? length(fp) : length(u))
            gradient = zeros(compact ? length(local_fp) : length(u))
            expected, _ = partition_value_and_gradient!(sampler, full, point, label)
            value, _ = partition_value_and_gradient!(sampler, gradient, point, label; shared=false)
            @test value ≈ expected
            @test gradient ≈ (compact ? full[indexin(local_fp, fp)] :
                [i in local_fp ? full[i] : 0.0 for i in eachindex(u)])
        end
    end
end

@testset "full and local gradients retain only the shared rows a unit reads" begin
    model = quote
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        eta[levels(unit)] .~ Normal.(0, 1)
        z[1:ncoeff] .~ Normal.(0, 1)
        local_eta = eta[unit]
        mu = a .+ local_eta .+ z[c1] .+ w .* z[c2]
        y .~ Normal.(mu, sigma)
    end
    allocations = Int[]
    local_allocations = Int[]
    for n in (6, 96)
        data = (; unit=collect(1:n), ncoeff=2n, c1=collect(1:2:2n),
            c2=collect(2:2:2n), w=sin.(1:n), y=cos.(1:n))
        bound = bind_data(lower_rkppl(model, data; conditioned=(:y,), mod=@__MODULE__), data)
        built = build_kernel(bound)
        u = collect(range(-0.2, 0.3; length=built.layout.total))
        query = prepare_partition(built, bound, :y; by=:unit, partitioned=:eta)
        for compact in (true, false)
            sampler = prepare_partition_sampler(built, bound, :y, u; by=:unit,
                partitioned=:eta, compact, backend=AutoEnzyme(; mode=Enzyme.Reverse))
            @test length(sampler.prefix.point) <= 2
            @test all(v -> !(v isa AbstractArray) || length(v) <= 2, sampler.prefix.values)
            for label in (n, 1, n), change in (0.0, 0.15)
                point = u .+ change
                fp = partition_footprint(sampler, label)
                @test length(fp) == 5
                gradient = zeros(compact ? length(fp) : length(u))
                value, _ = partition_value_and_gradient!(sampler, gradient, point, label)
                @test value ≈ query(point, label)
                for (slot, position) in enumerate(fp)
                    plus, minus = copy(point), copy(point)
                    plus[position] += 1e-6
                    minus[position] -= 1e-6
                    @test gradient[compact ? slot : position] ≈
                        (query(plus, label) - query(minus, label)) / 2e-6 atol=1e-6 rtol=1e-6
                end
                if !compact
                    @test all(iszero, gradient[setdiff(eachindex(u), fp)])
                end
                local_fp = partition_footprint(sampler, label; shared=false)
                local_gradient = zeros(compact ? length(local_fp) : length(u))
                local_value, _ = partition_value_and_gradient!(sampler,
                    local_gradient, point, label; shared=false)
                @test local_value ≈ value
                @test local_gradient ≈ (compact ? gradient[indexin(local_fp, fp)] :
                    [i in local_fp ? gradient[i] : 0.0 for i in eachindex(u)])
            end
            if compact
                gradient = zeros(length(partition_footprint(sampler, 1)))
                partition_value_and_gradient!(sampler, gradient, u, 1)
                push!(allocations, @allocated partition_value_and_gradient!(sampler, gradient, u, 1))
                local_gradient = zeros(length(partition_footprint(sampler, 1; shared=false)))
                partition_value_and_gradient!(sampler, local_gradient, u, 1; shared=false)
                push!(local_allocations, @allocated partition_value_and_gradient!(sampler,
                    local_gradient, u, 1; shared=false))
            end
        end
    end
    @test allocations[2] <= allocations[1] + 256
    @test local_allocations[2] <= local_allocations[1] + 256
end
end
