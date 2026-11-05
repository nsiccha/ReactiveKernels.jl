module ScalarPartitionReadTests
using Test, ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme

function build(source, data)
    bound = bind_data(lower_rkppl(source, data; conditioned = (:y,), mod = @__MODULE__), data)
    bound, build_kernel(bound)
end

# The registered Enzyme compact-overlay gap is also recorded by the existing
# cell-factor tests. Patched-Enzyme acceptance requires every native check.
function static_reverse(f, compact)
    try
        f()
    catch err
        compact && err isa Enzyme.Compiler.EnzymeRuntimeActivityError || rethrow()
        get(ENV, "RK_TEST_REQUIRE_STATIC_ENZYME", "0") == "1" && rethrow()
        @test occursin("Mismatched activity", sprint(showerror, err))
        @test_broken false
    end
end

@testset "strict partitions preserve scalar indexed expressions" begin
    data = (; X = hcat(ones(3), [0.0, 1.0, 2.0]), label = [1, 2, 3], y = [1.0, 2.0, 3.0])
    for expression in (:(exp.(a[1])), :(exp(a[1])))
        bound, built = build(quote
            beta[1:2] .~ Normal.(0.0, 1.0)
            a = X * beta
            w = $expression
            mu = w .+ a
            y .~ Normal.(mu, 1.0)
        end, data)
        query = prepare_partition(built, bound, :y; by = :label, locality = :error)
        full = prepare_query(built, bound, :likelihood)
        for beta in (zeros(2), [0.2, -0.3], [-0.1, 0.4])
            a = data.X * beta
            residual = exp(a[1]) .+ a .- data.y
            @test [query(beta, label) for label in partition_labels(query)] ≈
                -0.5 .* (log(2pi) .+ residual .^ 2)
            @test sum(query(beta, label) for label in partition_labels(query)) ≈ full(beta)
        end
        @test all(entry.status !== :whole for entry in partition_locality(query.kernel))
        beta = [0.2, -0.3]
        sampler = prepare_partition_sampler(built, bound, :y, beta; by = :label,
            backend = AutoEnzyme(; mode = Enzyme.Reverse), compact = false, locality = :error)
        gradient = similar(beta)
        a = data.X * beta
        for label in partition_labels(query)
            value, _ = partition_value_and_gradient!(sampler, gradient, beta, label)
            residual = exp(a[1]) + a[label] - data.y[label]
            expected = -residual .* (data.X[label, :] .+ exp(a[1]) .* data.X[1, :])
            @test value ≈ query(beta, label)
            @test gradient ≈ expected
            @test partition_footprint(query.kernel, beta, label, nothing).unconstrained == [1, 2]
        end
    end
end

@testset "shared scalar predictor with a separate row domain" begin
    data = (; X = [1.0 -1.0; 1.0 0.5; 1.0 2.0], label = [1, 2, 1, 2],
        other = [2, 1, 2, 1], y = [0.2, -0.1, 0.7, 0.4])
    for nonlinear in (false, true), index in (:label, :other)
        !nonlinear && index === :other && continue
        local_mean = nonlinear ? :(exp.(eta[$index])) : :(eta[$index])
        bound, built = build(quote
            beta[1:2] .~ Normal.(0.0, 1.0)
            eta[levels(label)] .~ Normal.(0.0, 1.0)
            shared = X * beta
            weight = exp.(shared[1])
            shift = shared[2] .- shared[1]
            mu = weight .+ shift .+ $local_mean
            y .~ Normal.(mu, 1.0)
        end, data)
        u = [0.2, -0.3, 0.1, -0.2]
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        if index === :other
            @test_throws PartitionSeparabilityError prepare_partition_sampler(
                built, bound, :y, u; by = :label, backend, partitioned = (:eta,))
            continue
        end
        query = prepare_partition(built, bound, :y; by = :label, locality = :error)
        full = prepare_query(built, bound, :likelihood)
        a = data.X * u[1:2]
        own = u[2 .+ data.label]
        residual = exp(a[1]) .+ (a[2] - a[1]) .+ (nonlinear ? exp.(own) : own) .- data.y
        @test sum(query(u, label) for label in (1, 2)) ≈ full(u)
        for compact in (false, true)
            static_reverse(compact) do
                sampler = prepare_partition_sampler(built, bound, :y, u;
                    by = :label, backend, compact,
                    partitioned = nonlinear ? (:eta,) : (), locality = :error)
                for label in (1, 2)
                    footprint = partition_footprint(sampler, label)
                    # The affine model uses a dense design and physically reads
                    # all coefficients. The nonlinear gather reads only its unit.
                    @test footprint == (nonlinear ? [1, 2, 2 + label] : collect(1:4))
                    gradient = zeros(compact ? length(footprint) : length(u))
                    value, _ = partition_value_and_gradient!(sampler, gradient, u, label)
                    rows = findall(==(label), data.label)
                    @test value ≈ -sum(log(2pi) .+ residual[rows] .^ 2) / 2
                    expected = zeros(length(u))
                    expected[1:2] .= -sum(residual[rows]) .* (
                        exp(a[1]) .* data.X[1, :] .+ data.X[2, :] .- data.X[1, :])
                    expected[2 + label] = -sum(residual[rows]) *
                        (nonlinear ? exp(u[2 + label]) : 1.0)
                    @test gradient ≈ (compact ? expected[footprint] : expected)
                end
            end
        end
    end
end

_scalar_covariance_add(a, b) = a + b

@testset "absent observation with a whole covariance factor" begin
    for captured in (false, true)
        data = (; y = [0.1, 0.2], soil = [0.3, 0.4], group = [1, 2])
        location = captured ? quote
            ra = R[group, 1]
            rb = R[group, 2]
            @plate for i in eachindex(soil)
                mu[i] = _scalar_covariance_add(ra[i], rb[i])
            end
        end : quote
            mu = R[group, 1] .+ R[group, 2]
        end
        source = quote
            a[1:2] .~ Normal.(0.0, 1.0)
            sd[1:2] .~ Exponential.(1.0)
            L ~ LKJCholesky(2, 2.0)
            Z[1:2, 1:2] .~ Normal.(0.0, 1.0)
            F = sd .* L
            R = Z * F'
            $(location.args...)
            y .~ Normal.(a, 1.0)
            soil .~ Normal.(mu, 1.0)
        end
        plan = bind_data(lower_rkppl(source, data;
            conditioned = (:y, :soil), mod = @__MODULE__), data)
        built = build_kernel(plan)
        u = collect(range(-0.3, 0.3; length = built.layout.total))
        by = (; y = [1, 2], soil = [1, 1])
        partitioned = (:a => [1, 2], :sd => [1, 1], :L => [1, 1], :Z => [1, 1])
        query = prepare_partition(built, plan, (:y, :soil); by, partitioned, u0 = u)
        full = prepare_query(built, plan, :likelihood)
        for point in (u, u .+ 0.2)
            @test partition_footprint(query.kernel, point, 1, nothing).unconstrained ==
                [1, 3, 4, 5, 6, 7, 8, 9]
            @test partition_footprint(query.kernel, point, 2, nothing).unconstrained == [2]
            @test sum(query(point, label) for label in (1, 2)) ≈ full(point)
            @test query(point, 2) ≈ -(log(2pi) + (point[2] - data.y[2])^2) / 2
        end
        # The active covariance still reads all three of its coordinates: moving
        # its ownership to another label must remain a separability violation.
        @test_throws PartitionSeparabilityError prepare_partition(built, plan, (:y, :soil);
            by, partitioned = (:a => [1, 2], :sd => [2, 2], :L => [2, 2], :Z => [1, 1]), u0 = u)
    end
end

@testset "absent scalar observation" begin
    data = (; y=[0.1, 0.2, 0.3], extra=0.4, second=0.8)
    for kind in (:direct, :indexed, :nonlinear), shared in (false, true)
        expression = kind === :direct ? :(z[1]) :
            kind === :indexed ? :(mu[1]) : :(exp(mu[1]) + mu[1])
        names = shared ? (:extra, :second, :y) : (:extra, :y)
        source = quote
            z[1:3] .~ Normal.(0.0, 1.0)
            mu = exp.(z)
            one = $expression
            extra .~ Normal.(one, 1.0)
            y .~ Normal.(mu, 1.0)
        end
        shared && push!(source.args, :(second .~ Normal.(one, 1.0)))
        plan = bind_data(lower_rkppl(source, data;
            conditioned=names, mod=@__MODULE__), data)
        built = build_kernel(plan)
        by = shared ? (; extra=[1], second=[1], y=[1,2,3]) : (; extra=[1], y=[1,2,3])
        query = prepare_partition(built, plan, names; by,
            u0=[0.2, 0.4, 0.6], partitioned=(:z => [1,2,3],))
        for compact in (false, true)
            sampler = prepare_partition_sampler(built, plan, names, [0.2,0.4,0.6];
                by, partitioned=(:z => [1,2,3],), compact,
                backend=AutoEnzyme(; mode=Enzyme.Reverse))
            for u in ([0.2,0.4,0.6], [-0.1,0.3,-0.2]), label in 1:3
                @test partition_footprint(sampler, label) == [label]
                gradient = zeros(compact ? 1 : 3)
                value, _ = partition_value_and_gradient!(sampler, gradient, u, label)
                @test value ≈ query(u, label)
                expected_value = -(log(2pi) + (exp(u[label]) - data.y[label])^2) / 2
                expected = zeros(3)
                mu = exp.(u)
                expected[label] = -(mu[label] - data.y[label]) * mu[label]
                if label == 1
                    one = kind === :direct ? u[1] : kind === :indexed ? mu[1] : exp(mu[1]) + mu[1]
                    derivative = kind === :direct ? 1.0 : kind === :indexed ? mu[1] : (exp(mu[1]) + 1) * mu[1]
                    for observed in (shared ? (data.extra, data.second) : (data.extra,))
                        expected[1] -= (one - observed) * derivative
                        expected_value -= (log(2pi) + (one - observed)^2) / 2
                    end
                end
                @test value ≈ expected_value
                @test gradient ≈ (compact ? expected[[label]] : expected)
            end
        end
        @test_throws PartitionSeparabilityError prepare_partition(built, plan, names;
            by=merge(by, (; extra=[2])), u0=[0.2,0.4,0.6], partitioned=(:z => [1,2,3],))
    end
end
end
