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
end
