module PartitionBatchTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

# A whole array with no row form: each entry reads every earlier one, so a strict
# partition cannot compute it at its own rows.
opaque_field(grid, beta) = beta .* cumsum(grid) ./ length(grid)

# No unit owns a coordinate: `beta` reaches every row through `field`, and units
# overlap on `w`, which is read at rows.
const SHARED = quote
    beta ~ Normal(0, 1)
    w[1:nw] .~ Normal.(0, 1)
    field = opaque_field(grid, beta)
    @plate for k in eachindex(y)
        y[k] ~ Poisson(exp(w[pos[k]] + field[k]))
    end
end
const OWNED = quote
    beta ~ Normal(0, 1)
    w[1:nw] .~ Normal.(0, 1)
    z[levels(unit)] .~ Normal.(0, 1)
    field = opaque_field(grid, beta)
    @plate for k in eachindex(y)
        y[k] ~ Poisson(exp(z[unit[k]] + w[pos[k]] + field[k]))
    end
end
const ROWS_ONLY = quote
    w[1:nw] .~ Normal.(0, 1)
    z[levels(unit)] .~ Normal.(0, 1)
    @plate for k in eachindex(y)
        y[k] ~ Poisson(exp(z[unit[k]] + w[pos[k]]))
    end
end
const NAMED_NONE = quote
    none[levels(unit)] .~ Normal.(0, 1)
    @plate for k in eachindex(y)
        y[k] ~ Poisson(exp(none[unit[k]]))
    end
end

function fixture(model; n=12, units=4)
    unit = repeat(1:units; inner=cld(n, units))[1:n]
    nw = max(2, n ÷ 3)
    data = (; unit, pos=mod1.(1:n, nw), nw, grid=collect(range(-0.7, 0.9; length=n)),
        y=mod.(1:n, 4))
    bound = bind_data(lower_rkppl(model, data; conditioned=(:y,), mod=@__MODULE__), data)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.4; length=built.layout.total))
    (; data, bound, built, u)
end

function positions(built, name)
    index = findfirst(e -> e.name === name, built.layout.entries)
    index === nothing && return nothing
    e = built.layout.entries[index]
    collect(e.offset:(e.offset + e.size - 1))
end

# An independent oracle: every parameter here is unconstrained, so the Poisson
# log density and its gradient follow directly from the packed point.
function oracle(fx, point, labels; weights=ones(length(labels)))
    (; data, built) = fx
    beta, w, z = positions(built, :beta), positions(built, :w), positions(built, :z)
    scale = cumsum(data.grid) ./ length(data.grid)
    value = 0.0
    gradient = zeros(length(point))
    for (label, weight) in zip(labels, weights), k in findall(==(label), data.unit)
        eta = point[w[data.pos[k]]]
        z === nothing || (eta += point[z[data.unit[k]]])
        beta === nothing || (eta += point[only(beta)] * scale[k])
        rate = exp(eta)
        value += weight * logpdf(Poisson(rate), data.y[k])
        residual = weight * (data.y[k] - rate)
        gradient[w[data.pos[k]]] += residual
        z === nothing || (gradient[z[data.unit[k]]] += residual)
        beta === nothing || (gradient[only(beta)] += scale[k] * residual)
    end
    value, gradient
end

@testset "partitioned = :none" begin
    fx = fixture(SHARED)
    (; bound, built, u, data) = fx
    labels = sort(unique(data.unit))
    # Nothing declared stays refused, and the refusal names the route.
    refusal = try
        prepare_partition(built, bound, :y; by=:unit, locality=:error)
        nothing
    catch err
        err
    end
    @test refusal isa ReactiveKernels.PartitionLocalityError
    @test occursin("partitioned = :none", sprint(showerror, refusal))
    @test_throws ReactiveKernels.PartitionLocalityError prepare_partition(
        built, bound, :y; by=:unit, locality=:error, partitioned=())

    query = prepare_partition(built, bound, :y; by=:unit, locality=:error, partitioned=:none, u0=u)
    @test any(e -> e.value === :field && e.status === :input, partition_locality(query))
    spelled = prepare_partition(built, bound, :y; by=:unit, locality=:error,
        partitioned=(:w => fill(missing, data.nw),), u0=u)
    for label in labels
        @test query(u, label) ≈ first(oracle(fx, u, [label]))
        @test query(u, label) == spelled(u, label)
    end

    for compact in (true, false)
        sampler = prepare_partition_sampler(built, bound, :y, u; by=:unit, backend=BACKEND,
            compact, locality=:error, partitioned=:none)
        @test sampler.prefix !== nothing
        for label in labels
            footprint = partition_footprint(sampler, label)
            gradient = zeros(compact ? length(footprint) : length(u))
            value, _ = partition_value_and_gradient!(sampler, gradient, u, label)
            expected_value, expected = oracle(fx, u, [label])
            @test value ≈ expected_value
            @test gradient ≈ (compact ? expected[footprint] : expected)
            @test isempty(partition_footprint(sampler, label; shared=false))
        end
    end

    # The same declaration serves the cell routes.
    cells = prepare_cell_query(built, bound, :y; locality=:error, partitioned=:none, u0=u)
    @test sum(cells(u, k) for k in eachindex(data.y)) ≈ first(oracle(fx, u, labels))
    @test prepare_cell_factors(built, bound, :y; locality=:error, partitioned=:none, u0=u) !== nothing
    compact_ad = prepare_compact_cell_ad(prepare_ad, built, bound, :y, u, Int[], BACKEND;
        locality=:error, partitioned=:none)
    accumulator = prepare_cell_accumulator(compact_ad, BACKEND, u; order=0)
    pullback = prepare_cell_pullback(compact_ad, BACKEND, u)
    gradient = zeros(length(u))
    cell_pullback!(gradient, pullback, u) do batch
        for k in eachindex(data.y)
            accumulate_cell!(batch, accumulator, nothing, 1.0, Float64[], Int[], k)
        end
    end
    @test gradient ≈ last(oracle(fx, u, labels))

    # A parameter named `none` keeps its name, and the bare symbol is not guessed.
    named = fixture(NAMED_NONE)
    message = try
        prepare_partition(named.built, named.bound, :y; by=:unit, partitioned=:none)
        ""
    catch err
        err isa ContractValidationError ? sprint(showerror, err) : rethrow()
    end
    @test occursin("ambiguous", message)
    @test occursin("(:none,)", message)
    by_name = prepare_partition(named.built, named.bound, :y; by=:unit, partitioned=(:none,),
        u0=named.u)
    @test by_name(named.u, 1) ≈ prepare_partition(named.built, named.bound, :y;
        by=:unit, locality=:ignore)(named.u, 1)
end

# Everything a batch must get right, whatever the sampler holds.
function check_batch(fx, sampler, point; prefix_reads=nothing)
    labels = partition_labels(sampler)
    g = fill(123.0, length(point))
    before = copy(point)
    subset = labels[1:2:end]

    value, returned = partition_batch_value_and_gradient!(sampler, g, point, subset)
    expected_value, expected = oracle(fx, point, subset)
    @test returned === g
    @test value ≈ expected_value
    @test g ≈ expected
    @test point == before

    value, _ = partition_batch_value_and_gradient!(sampler, g, point, labels)
    expected_value, expected = oracle(fx, point, labels)
    @test value ≈ expected_value
    @test g ≈ expected

    # A repeated label counts again, and a weight scales its term.
    value, _ = partition_batch_value_and_gradient!(sampler, g, point, labels[[1, 1]])
    expected_value, expected = oracle(fx, point, labels[[1, 1]])
    @test value ≈ expected_value
    @test g ≈ expected
    weights = [0.5, 2.0]
    value, _ = partition_batch_value_and_gradient!(sampler, g, point, labels[1:2]; weights)
    expected_value, expected = oracle(fx, point, labels[1:2]; weights)
    @test value ≈ expected_value
    @test g ≈ expected
    value, _ = partition_batch_value_and_gradient!(sampler, g, point, labels[1:2]; weights=[1, 3])
    expected_value, expected = oracle(fx, point, labels[1:2]; weights=[1.0, 3.0])
    @test value ≈ expected_value
    @test g ≈ expected

    # A batch agrees with the per-label call it replaces.
    total, summed = 0.0, zeros(length(point))
    compact = !isempty(sampler.slots)
    for label in labels
        footprint = partition_footprint(sampler, label)
        unit_gradient = zeros(compact ? length(footprint) : length(point))
        total += first(partition_value_and_gradient!(sampler, unit_gradient, point, label))
        if compact
            summed[footprint] .+= unit_gradient
        else
            summed .+= unit_gradient
        end
    end
    value, _ = partition_batch_value_and_gradient!(sampler, g, point, labels)
    @test value ≈ total
    @test g ≈ summed

    # An empty batch is zero, and a refused call leaves `g` as it was.
    value, _ = partition_batch_value_and_gradient!(sampler, g, point, empty(labels))
    @test value == 0.0
    @test all(iszero, g)
    fill!(g, 7.0)
    @test_throws ContractValidationError partition_batch_value_and_gradient!(
        sampler, g, point, [first(labels), :absent])
    @test_throws ContractValidationError partition_batch_value_and_gradient!(
        sampler, zeros(length(point) + 1), point, labels)
    @test_throws ContractValidationError partition_batch_value_and_gradient!(
        sampler, g, zeros(length(point) + 1), labels)
    @test_throws ContractValidationError partition_batch_value_and_gradient!(
        sampler, g, point, labels; weights=[1.0])
    @test all(==(7.0), g)
end

@testset "batches: $name, compact=$compact" for (name, model, partitioned, cached) in (
        ("nothing owned", SHARED, :none, true),
        ("owned effects", OWNED, :z, true),
        ("no shared values", ROWS_ONLY, (), false)), compact in (true, false)
    fx = fixture(model)
    (; bound, built, u) = fx
    sampler = prepare_partition_sampler(built, bound, :y, u; by=:unit, backend=BACKEND,
        compact, locality=:error, partitioned)
    @test (sampler.prefix !== nothing) == cached
    check_batch(fx, sampler, u)

    # A second batch at the same point reuses the cached values. A coordinate
    # they do not read leaves them, and one they do read refreshes them.
    g = zeros(length(u))
    labels = partition_labels(sampler)
    moved = copy(u)
    moved[positions(built, :w)] .+= 0.2
    if cached
        partition_batch_value_and_gradient!(sampler, g, u, labels)
        values = sampler.prefix.values
        partition_batch_value_and_gradient!(sampler, g, moved, labels)
        @test g ≈ last(oracle(fx, moved, labels))
        @test sampler.prefix.values === values
        moved[only(positions(built, :beta))] += 0.2
        partition_batch_value_and_gradient!(sampler, g, moved, labels)
        @test g ≈ last(oracle(fx, moved, labels))
        @test sampler.prefix.values !== values
    else
        partition_batch_value_and_gradient!(sampler, g, moved, labels)
        @test g ≈ last(oracle(fx, moved, labels))
    end
    check_batch(fx, sampler, moved)

    # A vector that is not a `Vector{Float64}` is accepted, as by the per-label call.
    value, _ = partition_batch_value_and_gradient!(sampler, g, view(u, eachindex(u)), labels)
    @test value ≈ first(oracle(fx, u, labels))
end

@testset "batches over mixed label types, compact=$compact" for compact in (true, false)
    names = Any[1, "two", :three]
    n, units = 12, 3
    unit = repeat(1:units; inner=4)
    nw = 4
    data = (; unit, subject=names[unit], pos=mod1.(1:n, nw), nw,
        grid=collect(range(-0.7, 0.9; length=n)), y=mod.(1:n, 4))
    bound = bind_data(lower_rkppl(OWNED, data; conditioned=(:y,), mod=@__MODULE__), data)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.4; length=built.layout.total))
    fx = (; data, built)
    sampler = prepare_partition_sampler(built, bound, :y, u; by=:subject, backend=BACKEND,
        compact, locality=:error, partitioned=(:z => names))
    g = zeros(length(u))
    value, _ = partition_batch_value_and_gradient!(sampler, g, u, names)
    expected_value, expected = oracle(fx, u, 1:units)
    @test value ≈ expected_value
    @test g ≈ expected
    value, _ = partition_batch_value_and_gradient!(sampler, g, u, Any["two", :three])
    expected_value, expected = oracle(fx, u, 2:3)
    @test value ≈ expected_value
    @test g ≈ expected
end

# The batch's reason to exist. A per-label gradient allocates a cotangent of the
# whole shared value on every call, so a sweep's memory grows with units times
# that value. A batch adds to one cotangent, so its extra memory per label must
# not grow with the shared value.
@testset "a batch pays for one shared cotangent" begin
    per_label_bytes(sampler, gradients, u, labels) = @allocated for label in labels
        partition_value_and_gradient!(sampler, gradients[label], u, label)
    end
    batch_bytes(sampler, g, u, labels) =
        @allocated partition_batch_value_and_gradient!(sampler, g, u, labels)
    bytes = map((80, 800)) do n
        fx = fixture(SHARED; n, units=8)
        sampler = prepare_partition_sampler(fx.built, fx.bound, :y, fx.u; by=:unit,
            backend=BACKEND, locality=:error, partitioned=:none)
        labels = partition_labels(sampler)
        @test any(v -> v isa AbstractVector && length(v) == n, sampler.prefix.values)
        g = zeros(length(fx.u))
        gradients = Dict(l => zeros(length(partition_footprint(sampler, l))) for l in labels)
        for _ in 1:2
            per_label_bytes(sampler, gradients, fx.u, labels)
            batch_bytes(sampler, g, fx.u, labels)
            batch_bytes(sampler, g, fx.u, labels[1:1])
        end
        whole = batch_bytes(sampler, g, fx.u, labels)
        one = batch_bytes(sampler, g, fx.u, labels[1:1])
        sweep = per_label_bytes(sampler, gradients, fx.u, labels)
        (; n, whole, one, sweep)
    end
    @info "batch and per-label sweep allocations, 8 units" bytes
    # With 800 shared entries the sweep's per-label cotangents dominate and the
    # batch keeps one. With 80, eight cotangents (5 KiB) weigh less than the
    # batch's own fixed storage, so the comparison would test that instead.
    @test bytes[2].whole < bytes[2].sweep / 2
    extra = map(b -> b.whole - b.one, bytes)
    # Each of the 8 units reads 10 and then 100 rows, so a label's own work grows
    # with its rows: 10 times the rows may not cost more than 10 times the bytes,
    # while a cotangent per label would add 8 times the 800-entry shared value.
    @test extra[2] <= 10 * extra[1] + 4096
    @test extra[2] < 8 * 800 * 8 * 4
end

end # module
