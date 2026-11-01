module SharedValueQueryTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Test

groupmix(L, Z) = L * Z
function fixture(n)
    model = quote
        a ~ Normal(0, 1)
        L ~ LKJCholesky(6, 2.0)
        Z[levels(group), 1:2] .~ Normal.(0, 1)
        eta[levels(unit)] .~ Normal.(0, 1)
        R = groupmix(L, Z)
        local_eta = eta[unit]
        mu = a .+ local_eta .+ R[group, 1] .+ w .* R[group, 2]
        y .~ Normal.(mu, 1)
    end
    group = repeat(1:6; inner=n)
    data = (; group, unit=cld.(group, 2), w=sin.(eachindex(group)), y=cos.(eachindex(group)))
    bound = bind_data(lower_rkppl(model, data; conditioned=(:y,), mod=@__MODULE__), data)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.4; length=built.layout.total))
    (; bound, built, u, data)
end

exprnodes(x) = x isa Expr ? 1 + sum(exprnodes, x.args; init=0) : 0
@testset "shared value queries preserve supports and row work" begin
    storage = Int[]
    structures = Int[]
    for n in (3, 30)
        (; bound, built, u, data) = fixture(n)
        cell = prepare_cell_query(built, bound, :y; partitioned=:eta, u0=u, locality=:error)
        partition = prepare_partition(built, bound, :y; by=:unit, partitioned=:eta, u0=u)
        cell_reference = prepare_cell_query(built, bound, :y; locality=:ignore)
        partition_reference = prepare_partition(built, bound, :y; by=:unit, locality=:ignore)
        for q in (cell, partition)
            @test q.prefix !== nothing
            @test !any(e -> e.status === :whole, ReactiveKernels.partition_locality(q))
            @test any(e -> e.value === :R && e.status === :input, ReactiveKernels.partition_locality(q))
            @test all(v -> !(v isa AbstractArray) || !Base.mightalias(v, u), q.prefix.cache.values)
            # Six groups and two modes; no observation-length field belongs
            # in the prefix when its producer can be evaluated at rows.
            @test all(v -> !(v isa AbstractArray) || length(v) <= 36, q.prefix.cache.values)
        end
        push!(storage, Base.summarysize(cell.prefix.cache.values))
        push!(structures, exprnodes(code_expr(cell.prefix.residual)))
        local_positions = findall(name -> startswith(string(name), "eta."),
            coordinate_names(built.layout))
        @test length(local_positions) == 3
        cached_cell, cached_partition = cell.prefix.cache.values, partition.prefix.cache.values
        for change in (0.0, 0.17), i in (1, 2n+1, 4n+1, 1)
            point = copy(u)
            point[local_positions] .+= change
            @test cell(point, i) ≈ cell_reference(point, i)
            @test partition(point, data.unit[i]) ≈ partition_reference(point, data.unit[i])
            @test cell.prefix.cache.values === cached_cell
            @test partition.prefix.cache.values === cached_partition
            fp = ReactiveKernels.partition_footprint(cell.kernel, point, i).unconstrained
            reference_fp = ReactiveKernels.partition_footprint(cell_reference.kernel, point, i).unconstrained
            @test fp == reference_fp
            @test intersect(fp, local_positions) == [local_positions[data.unit[i]]]
        end
        point = copy(u)
        point[cell.prefix.cache.positions] .+= 0.1
        @test cell(point, 1) ≈ cell_reference(point, 1)
        @test partition(point, 1) ≈ partition_reference(point, 1)
        @test cell(view(point, :), 1) ≈ cell_reference(point, 1)
        @test partition(view(point, :), 1) ≈ partition_reference(point, 1)
        @test cell(point, Int32(1)) ≈ cell_reference(point, Int32(1))
        @test cell(point, CartesianIndex(1)) ≈ cell_reference(point, CartesianIndex(1))
        @test cell.prefix.cache.values !== cached_cell
        @test partition.prefix.cache.values !== cached_partition
        @test_throws ContractValidationError cell(point[2:end], 1)
        @test_throws ContractValidationError partition(point[2:end], 1)
        @test_throws ContractValidationError prepare_cell_query(built, bound, :y;
            partitioned=:eta, u0=u[2:end])
        @test_throws ContractValidationError prepare_partition(built, bound, :y;
            by=:unit, partitioned=:missing_parameter)
        # Warmed diagnostics, without a timing threshold on a shared host.
        cell(u, 1)
        partition(u, 1)
        cell_times = [@elapsed cell(u, 1) for _ in 1:21]
        partition_times = [@elapsed partition(u, 1) for _ in 1:21]
        cell_bytes = @allocated cell(u, 1)
        partition_bytes = @allocated partition(u, 1)
        println((; observations=length(data.y), cached_bytes=last(storage),
            emitted_nodes=last(structures), cell_seconds=sort(cell_times)[11],
            partition_seconds=sort(partition_times)[11], cell_bytes, partition_bytes))
    end
    @test allequal(storage)
    @test allequal(structures)
end
end
