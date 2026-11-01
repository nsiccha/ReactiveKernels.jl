module SharedValueLifecycleTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Test

function build_model(model, data)
    bound = bind_data(lower_rkppl(model, data; conditioned=(:y,), mod=@__MODULE__), data)
    bound, build_kernel(bound)
end

@testset "value query cache failure and independent bindings" begin
    model = quote
        a ~ Normal(0, 1)
        z[levels(subject)] .~ Normal.(0, 1)
        scale = sqrt(a)
        eta = z[subject]
        mu = scale .+ eta
        y .~ Normal.(mu, 1)
    end
    data = (; subject=[1, 1, 2], y=[0.1, 0.2, 0.3])
    bound, built = build_model(model, data)
    u = ones(built.layout.total)
    a = only(findall(==(:a), coordinate_names(built.layout)))
    cell = prepare_cell_query(built, bound, :y; partitioned=:z, u0=u, locality=:error)
    part = prepare_partition(built, bound, :y; by=:subject, partitioned=:z, u0=u)
    other = prepare_cell_query(built, bound, :y; partitioned=:z, u0=u, locality=:error)
    cell_ref = prepare_cell_query(built, bound, :y)
    part_ref = prepare_partition(built, bound, :y; by=:subject)
    for (q, reference) in ((cell, cell_ref), (part, part_ref))
        point = copy(u)
        point[a] = -1
        for _ in 1:2
            @test_throws DomainError q(point, 1)
            @test !q.prefix.cache.valid
        end
        point[a] = 4
        @test q(point, 1) ≈ reference(point, 1)
        @test q.prefix.cache.valid
    end
    @test other(u, 1) ≈ cell_ref(u, 1)
    @test other.prefix.cache.point !== cell.prefix.cache.point
    fresh_data = merge(data, (; y=data.y .+ 0.8))
    fresh_bound, fresh_built = build_model(model, fresh_data)
    fresh = prepare_cell_query(fresh_built, fresh_bound, :y;
        partitioned=:z, u0=u, locality=:error)
    @test fresh(u, 1) != other(u, 1)
    @test fresh(u, 1) ≈ prepare_cell_query(fresh_built, fresh_bound, :y)(u, 1)
end

@testset "value query mixed labels and no shared prefix" begin
    model = quote
        theta[1:4] .~ Normal.(0, 1)
        z[levels(unit)] .~ Normal.(0, 1)
        eta = z[unit]
        mu = theta[g] .+ eta
        y .~ Normal.(mu, 1)
    end
    labels = Any[1, "two", :three]
    unit = [1, 1, 2, 2, 2, 3]
    data = (; unit, subject=labels[unit], g=[1, 1, 2, 3, 2, 4],
        y=[0.2, -0.1, 0.3, 0.1, -0.2, 0.4])
    bound, built = build_model(model, data)
    u = collect(range(-0.4, 0.3; length=built.layout.total))
    q = prepare_partition(built, bound, :y; by=:subject,
        partitioned=:z => labels, u0=u)
    reference = prepare_partition(built, bound, :y; by=:subject)
    @test partition_labels(q) == labels
    for change in (0.0, 0.3), label in labels[[1, 2, 3, 1]]
        point = u .+ change
        @test q(point, label) ≈ reference(point, label)
        @test ReactiveKernels.partition_footprint(q.kernel, point, label, nothing) ==
            ReactiveKernels.partition_footprint(reference.kernel, point, label, nothing)
    end
    @test_throws ArgumentError q(u, :absent)
    @test q(u, :three) ≈ reference(u, :three)
    model_local = quote
        z[levels(unit)] .~ Normal.(0, 1)
        eta = z[unit]
        # The explicit intermediate matches the existing no-prefix sampler
        # fixture; direct affine factor ownership is tracked by 0mtcwfj.
        mu = 0.0 .+ eta
        y .~ Normal.(mu, 1)
    end
    model_factor = quote
        z[levels(unit)] .~ Normal.(0, 1)
        eta = z[unit]
        y .~ Normal.(eta, 1)
    end
    factor_bound, factor_built = build_model(model_factor, data)
    @test_throws ReactiveKernels.PartitionSeparabilityError prepare_partition(
        factor_built, factor_bound, :y; by=:unit, partitioned=:z)
    local_bound, local_built = build_model(model_local, data)
    local_u = [0.2, -0.1, 0.3]
    local_cell = prepare_cell_query(local_built, local_bound, :y;
        partitioned=:z, u0=local_u)
    local_part = prepare_partition(local_built, local_bound, :y;
        by=:unit, partitioned=:z, u0=local_u)
    @test local_cell.prefix === nothing
    @test local_part.prefix === nothing
    @test local_part(local_u, 2) ≈ sum(local_cell(local_u, i) for i in findall(==(2), unit))
end
end
