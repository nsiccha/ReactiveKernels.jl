module PartitionPrefixSamplerTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

const MODEL = quote
    sigma ~ Exponential(1)
    a ~ Normal(0, 1)
    sd[1:2] .~ Exponential.(1)
    L ~ LKJCholesky(2, 2.0)
    Z[levels(group), 1:2] .~ Normal.(0, 1)
    F = sd .* L
    R = Z * F'
    mu = a .+ R[group, 1] .+ w .* R[group, 2]
    y .~ Normal.(mu, sigma)
end

function build_model(model, data)
    plan = lower_rkppl(model, data; conditioned=(:y,), mod=@__MODULE__)
    bound = bind_data(plan, data)
    bound, build_kernel(bound)
end
const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

@testset "shared prefix reuse and compact/dense full/local gradients" begin
    for sizes in ((2, 3, 4, 5, 2, 3), (3, 2, 25, 40, 80, 60))
        group = Int[g for (g, n) in enumerate(sizes) for _ in 1:n]
        data = (; group, subject=cld.(group, 2), w=sin.(eachindex(group)), y=cos.(eachindex(group)))
        bound, built = build_model(MODEL, data)
        u = collect(range(-0.3, 0.4; length=built.layout.total))
        original = copy(u)
        query = prepare_partition(built, bound, :y; by=:subject)
        baseline = prepare_partition_sampler(built, bound, :y, u; by=:subject,
            backend=BACKEND, compact=false)
        @test baseline.prefix === nothing
        @test_throws ContractValidationError partition_footprint(baseline, 1; shared=false)
        @test_throws ContractValidationError partition_value_and_gradient!(baseline, zero(u), u, 1; shared=false)
        for compact in (true, false)
            sampler = prepare_partition_sampler(built, bound, :y, u; by=:subject,
                backend=BACKEND, compact, partitioned=(:Z => [1, 1, 2, 2, 3, 3],))
            @test sampler.prefix !== nothing
            # The residual gradient receives each label's plan behind a
            # reference, never as a by-value aggregate of all its fields.
            @test ReactiveKernelsPPL._partition_ad(sampler.ad, 1).resolver isa
                  ReactiveKernels._ADPlanBoxing
            shared_positions = sampler.prefix.positions
            local_positions = setdiff(eachindex(u), shared_positions)
            initial_values = sampler.prefix.values
            for value in initial_values
                value isa AbstractArray && @test !Base.mightalias(value, u)
            end
            for label in (1, 3, 2, 1), local_change in (0.0, 0.12)
                point = copy(u)
                point[local_positions] .+= local_change
                full_footprint = partition_footprint(sampler, label)
                local_footprint = partition_footprint(sampler, label; shared=false)
                @test local_footprint == intersect(full_footprint, local_positions)
                expected = zero(u)
                baseline_value, _ = partition_value_and_gradient!(baseline, expected, point, label)
                gradient = zeros(compact ? length(full_footprint) : length(u))
                value, returned = partition_value_and_gradient!(sampler, gradient, point, label)
                @test returned === gradient
                @test value ≈ baseline_value ≈ query(point, label)
                @test gradient ≈ (compact ? expected[full_footprint] : expected)
                @test sampler.prefix.values === initial_values
                local_gradient = zeros(compact ? length(local_footprint) : length(u))
                local_value, _ = partition_value_and_gradient!(sampler, local_gradient, point, label; shared=false)
                @test local_value ≈ value
                @test sampler.prefix.values === initial_values
                if compact
                    @test local_gradient ≈ expected[local_footprint]
                else
                    @test local_gradient[local_footprint] ≈ expected[local_footprint]
                    @test all(iszero, local_gradient[setdiff(eachindex(u), local_footprint)])
                end
            end
            # Mutate the same input object, including shared coordinates;
            # object identity must not hide the invalidation.
            point = copy(u)
            point[shared_positions] .+= 0.2
            for label in (2, 1)
                fp = partition_footprint(sampler, label)
                gradient = zeros(compact ? length(fp) : length(u))
                value, _ = partition_value_and_gradient!(sampler, gradient, point, label)
                @test sampler.prefix.values !== initial_values
                expected = zero(u)
                baseline_value, _ = partition_value_and_gradient!(baseline, expected, point, label)
                @test value ≈ baseline_value
                @test gradient ≈ (compact ? expected[fp] : expected)
                # Independent differences cover shared and local coordinates.
                for position in fp
                    plus, minus = copy(point), copy(point)
                    plus[position] += 1e-6
                    minus[position] -= 1e-6
                    finite = (query(plus, label) - query(minus, label)) / 2e-6
                    index = compact ? findfirst(==(position), fp) : position
                    @test gradient[index] ≈ finite atol=1e-5 rtol=1e-5
                end
            end
            @test u == original
            @test_throws ContractValidationError partition_value_and_gradient!(sampler, Float64[], point, 1)
            @test_throws ContractValidationError partition_value_and_gradient!(sampler, zero(point), point[2:end], 1)
            @test_throws ContractValidationError partition_footprint(sampler, 99; shared=false)
        end
    end
end

@testset "no dynamic shared frontier" begin
    model = quote
        z[levels(unit)] .~ Normal.(0, 1)
        eta = z[unit]
        mu = 0.0 .+ eta
        y .~ Normal.(mu, 1)
    end
    unit = [1, 1, 2, 2, 3]
    for labels in ([1, 2, 3], Any[1, "two", :three])
        data = (; unit, subject=labels[unit], y=[0.1, 0.3, -0.2, 0.5, -0.1])
        bound, built = build_model(model, data)
        by = eltype(labels) === Int ? :unit : :subject
        declaration = by === :unit ? :z : (:z => labels)
        u = [0.2, 0.1, -0.3]
        for compact in (true, false)
            sampler = prepare_partition_sampler(built, bound, :y, u; by,
                backend=BACKEND, compact, partitioned=declaration)
            @test sampler.prefix === nothing
            for (j, label) in enumerate(labels)
                fp = partition_footprint(sampler, label)
                @test fp == partition_footprint(sampler, label; shared=false) == [j]
                gradient = zeros(compact ? length(fp) : length(u))
                local_gradient = similar(gradient)
                full_value, _ = partition_value_and_gradient!(sampler, gradient, u, label)
                local_value, _ = partition_value_and_gradient!(sampler, local_gradient, u, label; shared=false)
                @test local_value == full_value
                @test local_gradient == gradient
                expected = sum(data.y[unit .== j] .- u[j])
                @test (compact ? only(gradient) : gradient[j]) ≈ expected
            end
        end
    end
end

@testset "shared cache does not bypass existing foreign-read rejection" begin
    # Inline factor terms currently become a dense one-hot design product,
    # which reads every level (tracked separately by todo 0mtcwfj).
    model = quote
        z[levels(subject)] .~ Normal.(0, 1)
        y .~ Normal.(z[subject], 1)
    end
    data = (; subject=[1, 1, 2, 2, 3], y=[0.1, 0.3, -0.2, 0.5, -0.1])
    bound, built = build_model(model, data)
    for compact in (true, false)
        @test_throws ReactiveKernels.PartitionSeparabilityError prepare_partition_sampler(
            built, bound, :y, [0.2, 0.1, -0.3]; by=:subject,
            backend=BACKEND, compact, partitioned=:z)
    end
end

@testset "a failed prefix refresh stays invalid until a successful retry" begin
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
    sampler = prepare_partition_sampler(built, bound, :y, u; by=:subject,
        backend=BACKEND, partitioned=:z)
    gradient = zeros(length(partition_footprint(sampler, 1; shared=false)))
    point = copy(u)
    point[a] = -1.0
    for _ in 1:2
        @test_throws DomainError partition_value_and_gradient!(sampler, gradient, point, 1; shared=false)
        @test !sampler.prefix.valid
    end
    point[a] = 4.0
    value, _ = partition_value_and_gradient!(sampler, gradient, point, 1; shared=false)
    @test sampler.prefix.valid
    query = prepare_partition(built, bound, :y; by=:subject)
    @test value ≈ query(point, 1)
end

@testset "shared array views and mixed partition labels" begin
    model = quote
        theta[1:4] .~ Normal.(0, 1)
        z[levels(unit)] .~ Normal.(0, 1)
        eta = z[unit]
        mu = theta[g] .+ eta
        y .~ Normal.(mu, 1)
    end
    labels = Any[1, "two", :three]
    unit = [1, 1, 2, 2, 2, 3]
    data = (; unit, subject=labels[unit], g=[1, 1, 2, 3, 2, 4], y=[0.2, -0.1, 0.3, 0.1, -0.2, 0.4])
    bound, built = build_model(model, data)
    u = collect(range(-0.4, 0.3; length=built.layout.total))
    baseline = prepare_partition_sampler(built, bound, :y, u; by=:subject,
        backend=BACKEND, compact=false)
    for compact in (true, false)
        point = copy(u)
        sampler = prepare_partition_sampler(built, bound, :y, point; by=:subject,
            backend=BACKEND, compact, partitioned=:z => labels)
        @test sampler.prefix !== nothing
        initial = sampler.prefix.values
        @test any(x -> x isa AbstractArray, initial)
        for x in initial
            x isa AbstractArray && @test !Base.mightalias(x, point)
        end
        for label in labels[[1, 2, 3, 1]]
            fp = partition_footprint(sampler, label)
            gradient = zeros(compact ? length(fp) : length(point))
            value, _ = partition_value_and_gradient!(sampler, gradient, point, label)
            expected = zero(point)
            expected_value, _ = partition_value_and_gradient!(baseline, expected, point, label)
            @test value ≈ expected_value
            @test gradient ≈ (compact ? expected[fp] : expected)
            @test all(iszero, expected[setdiff(eachindex(point), fp)])
            @test sampler.prefix.values === initial
        end
        saved = deepcopy(initial)
        point[sampler.prefix.positions] .+= 0.5
        @test all(isequal.(sampler.prefix.values, saved))
        fp = partition_footprint(sampler, "two"; shared=false)
        local_gradient = zeros(compact ? length(fp) : length(point))
        value, _ = partition_value_and_gradient!(sampler, local_gradient, point, "two"; shared=false)
        @test sampler.prefix.values !== initial
        expected = zero(point)
        expected_value, _ = partition_value_and_gradient!(baseline, expected, point, "two")
        @test value ≈ expected_value
        @test (compact ? local_gradient : local_gradient[fp]) ≈ expected[fp]
        compact || @test all(iszero, local_gradient[setdiff(eachindex(point), fp)])
    end
end

# A joint vector observation cell under a shared prefix: the program BRM
# emits for connected NLME units (snag `brm-joint-cell-s-2d923270`), whose
# cell mean vectors come from an argument plate and whose group scale is read
# as `tau[1]`. The residual is prepared from exemplars and its plan port is
# typed by the plan its exemplar-typed lowering computes; each label's plan
# must come from that same lowering. An untyped plan lowering made other
# type-dependent choices for this program (a scalar parameter `tau` forms no
# shared prefix here) and emitted other plan fields, so every prefixed
# call refused its own plan and the sampler failed at preparation.
@testset "joint vector cells under a shared prefix" begin
    model = quote
        tau[1:1] .~ LogNormal.(0, 1)
        z[1:3, 1:1] .~ Normal.(0, 1)
        b_subject = z .* tau[1]
        ranef = b_subject[subject_level, 1]
        intercept ~ Normal(0.0, 1.0)
        mu = intercept .+ ranef
        beta ~ Normal(0.0, 1.0)
        @plate for i in eachindex(y_cell)
            y_mean[i] = [mu[left[i]], mu[right[i]]]
        end
        shared_y .~ Normal.(beta, 1)
        @plate for i in eachindex(y)
            y[i] ~ MvNormal(y_mean[i], C)
        end
    end
    data = (; subject_level=[1, 2, 3], left=[1, 3], right=[2, 3], y_cell=[1, 2],
        y=[[0.1, 0.2], [-0.1, 0.4]], C=[1.0 0.2; 0.2 1.0], shared_y=[0.3])
    names = (:y, :shared_y)
    by = (; y=[1, 2], shared_y=[1])
    rule = (:z => [1, 1, 2],)
    plan = lower_rkppl(model, data; conditioned=names, mod=@__MODULE__)
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    position(name) = only(e.offset for e in built.layout.entries if e.name === name)
    function density(u, label)
        b, tau, beta = u[position(:intercept)], exp(u[position(:tau)]), u[position(:beta)]
        mu = b .+ tau .* u[position(:z) .+ (0:2)]
        total = label == 1 ? logpdf(Normal(beta, 1), only(data.shared_y)) : 0.0
        for i in eachindex(data.y)
            by.y[i] == label || continue
            total += logpdf(MvNormal([mu[data.left[i]], mu[data.right[i]]], data.C), data.y[i])
        end
        total
    end
    u = collect(range(-0.3, 0.4; length=built.layout.total))
    query = prepare_partition(built, bound, names; by, locality=:error, partitioned=rule, u0=u)
    @test query.prefix !== nothing
    for point in (u, u .+ 0.1), label in (2, 1)
        @test query(point, label) ≈ density(point, label)
    end
    plain = prepare_partition(built, bound, names; by, locality=:error)
    for compact in (true, false)
        sampler = prepare_partition_sampler(built, bound, names, u; by, backend=BACKEND,
            compact, partitioned=rule)
        @test sampler.prefix !== nothing
        baseline = prepare_partition_sampler(built, bound, names, u; by, backend=BACKEND,
            compact=false)
        for label in (2, 1)
            fp = partition_footprint(sampler, label)
            gradient = zeros(compact ? length(fp) : length(u))
            value, _ = partition_value_and_gradient!(sampler, gradient, u, label)
            expected = zero(u)
            expected_value, _ = partition_value_and_gradient!(baseline, expected, u, label)
            @test value ≈ density(u, label) ≈ plain(u, label) ≈ expected_value
            @test gradient ≈ (compact ? expected[fp] : expected)
        end
    end
end
end # module
