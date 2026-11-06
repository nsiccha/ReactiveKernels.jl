module PartitionValueCallTests
# A value query's call costs its kernel's work and no copy of what the query
# holds: neither the prepared kernel nor the label's plan reaches the heap per
# call (snag partition-value-0a538ab4). Each bound compares a query's call with
# the direct call of its kernel, so it does not depend on what the body allocates.
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Test
const RKPPL = ReactiveKernelsPPL

allocated_call(q, u, label) = @allocated q(u, label)
allocated_kernel(kernel, args...) = @allocated kernel(args...)
function fewest_bytes(f, args...)
    for _ in 1:3
        Base.invokelatest(f, args...)
    end
    minimum(Base.invokelatest(f, args...) for _ in 1:5)
end
query_overhead(q, u, label, kernel, args...) =
    fewest_bytes(allocated_call, q, u, label) - fewest_bytes(allocated_kernel, kernel, args...)
inferred(f, args...) = only(Base.return_types(f, map(typeof, args)))

function fixture(model, data)
    bound = bind_data(lower_rkppl(model, data; conditioned=(:y,), mod=@__MODULE__), data)
    built = build_kernel(bound)
    (; bound, built, u=collect(range(-0.3, 0.4; length=built.layout.total)))
end

@testset "partition value calls copy neither the kernel nor the plan" begin
    model = quote
        sigma ~ Exponential(1)
        a ~ Normal(0, 1)
        z[levels(subject)] .~ Normal.(0, 1)
        mu = a .+ z[subject] .* w
        y .~ Normal.(mu, sigma)
    end
    subject = repeat(1:6; inner=3)
    (; bound, built, u) = fixture(model,
        (; subject, w=sin.(eachindex(subject)), y=cos.(eachindex(subject))))
    dense = prepare_partition(built, bound, :y; by=:subject)
    cached = prepare_partition(built, bound, :y; by=:subject, partitioned=:z)
    @test cached.prefix !== nothing
    label = 2
    @test inferred(RKPPL._partition_density, dense, u, label) === Float64
    @test inferred(RKPPL._partition_density, cached, u, label) === Float64

    plan = dense.plans[label]
    @test dense(u, label) == Base.invokelatest(dense.kernel, u, label, plan)
    @test query_overhead(dense, u, label, dense.kernel, u, label, plan) < sizeof(plan)

    plan = cached.plans[label]
    value = cached(u, label)  # the prefix now holds this point's values
    values = cached.prefix.cache.values
    residual = cached.prefix.residual
    @test value == Base.invokelatest(residual, u, label, plan, values...)
    @test value ≈ dense(u, label)
    @test query_overhead(cached, u, label, residual, u, label, plan, values...) < sizeof(plan)

    # A view has its own prepared residual, reached through a dictionary. That
    # lookup is a dynamic call, and it still copies neither.
    point = view(copy(u), :)
    @test cached(point, label) == value
    viewed = cached.prefix.residuals[typeof((point, label, nothing, values...))]
    @test query_overhead(cached, point, label, viewed, point, label, plan, values...) < sizeof(residual)

    # A label no observation has is still refused by the kernel's own check.
    @test_throws ArgumentError dense(u, 999)
    @test_throws ArgumentError cached(u, 999)
    @test_throws ArgumentError cached(point, 999)
end

# A helper with no row form stays in the shared prefix, and it reads the only
# shared coordinate.
value_field(grid, beta) = map(t -> exp(beta * t), grid)

@testset "cell value calls do not copy the residual" begin
    model = quote
        beta ~ Normal(0, 1)
        z[levels(id)] .~ Normal.(0, 1)
        field = value_field(grid, beta)
        zi = z[id]
        rate = exp.(zi) .* field
        y .~ Poisson.(rate)
    end
    n = 6
    (; bound, built, u) = fixture(model,
        (; id=collect(1:n), grid=collect(range(-0.7, 0.9; length=n)), y=mod.(1:n, 4)))
    q = prepare_cell_query(built, bound, :y; partitioned=:z, locality=:error)
    cell = 2
    @test inferred(RKPPL._cell_density, q, u, cell) === Float64
    value = q(u, cell)  # the prefix now holds this point's values
    values = q.prefix.cache.values
    residual = q.prefix.residual
    @test value == Base.invokelatest(residual, u, cell, values...)
    @test query_overhead(q, u, cell, residual, u, cell, values...) < sizeof(residual)
end

end
