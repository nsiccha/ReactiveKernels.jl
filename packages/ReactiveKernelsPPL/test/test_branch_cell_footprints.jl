using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

function _bcf_source(named)
    named ? quote
        a ~ Normal(0, 1)
        z[levels(group), 1:2] .~ Normal.(0, 1)
        left = z[group, 1]
        right = z[group, 2]
        @plate for i in eachindex(y)
            mu = a > 0 ? left[i] : right[i]
            y[i] ~ Normal(mu, 1)
        end
    end : quote
        a ~ Normal(0, 1)
        z[levels(group), 1:2] .~ Normal.(0, 1)
        @plate for i in eachindex(y)
            mu = a > 0 ? z[group[i], 1] : z[group[i], 2]
            y[i] ~ Normal(mu, 1)
        end
    end
end

function _bcf_build(source, data)
    bound = bind_data(lower_rkppl(source, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(bound)
    bound, built, prepare_cell_query(built, bound, :y; locality = :error)
end

@testset "branch capture unions preserve ragged and empty rows" begin
    source = quote
        a ~ Normal(0, 1)
        z[1:4] .~ Normal.(0, 1)
        @plate for i in eachindex(y)
            mu = a > 0 ? sum(z[left[i]]) : sum(z[right[i]])
            y[i] ~ Normal(mu, 1)
        end
    end
    data = (; left = [[1, 1, 3], Int[], [2]], right = [[2, 4], Int[], [2, 2]],
        y = [0.2, -0.1, 0.4])
    bound, built, q = _bcf_build(source, data)
    u = [0.3, 0.2, 0.5, -0.1, 0.7]
    sampler = prepare_cell_sampler(built, bound, :y, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    for sign in (-1, 1), i in eachindex(data.y)
        u[1] = sign * 0.3
        rows = sign > 0 ? data.left[i] : data.right[i]
        mu = sum(u[1 .+ rows])
        expected = logpdf(Normal(mu, 1), data.y[i])
        @test q(u, i) ≈ expected
        @test partition_footprint(q.kernel, u, i).unconstrained ==
            sort!(unique([1; 1 .+ data.left[i]; 1 .+ data.right[i]]))
        value, gradient = cell_value_and_gradient!(sampler, similar(u), u, i)
        expected_gradient = [0.0; [count(==(j), rows) * (data.y[i] - mu) for j in 1:4]]
        @test value ≈ expected
        @test gradient ≈ expected_gradient
    end
end

_bcf_nodes(ex) = ex isa Expr ? 1 + sum(_bcf_nodes, ex.args; init = 0) : 1
_bcf_heads(ex, head) = ex isa Expr ?
    Int(ex.head === head) + sum(x -> _bcf_heads(x, head), ex.args; init = 0) : 0
_bcf_support(fp, u, i) = Base.invokelatest(fp, u, i).unconstrained

@testset "retained branches union bound capture rows" begin
    for named in (false, true)
        costs = NamedTuple[]
        for n in (2, 2_000)
            data = (; group = collect(1:n), y = collect(range(-0.2, 0.4; length = n)))
            before = deepcopy(data)
            bound, built, q = _bcf_build(_bcf_source(named), data)
            fp = partition_footprint_kernel(q.kernel)
            u = collect(range(0.2, 0.8; length = built.layout.total))
            sampler = n == 2 ? prepare_cell_sampler(built, bound, :y, u;
                backend = AutoEnzyme(; mode = Enzyme.Reverse), locality = :error) : nothing
            for sign in (-1, 1)
                u[1] = sign * 0.3
                saved = copy(u)
                full = prepare_query(built, bound, :pointwise)(u).y
                for i in (1, n)
                    expected_support = [1, 1 + i, 1 + n + i]
                    selected = 1 + i + (sign < 0 ? n : 0)
                    expected = logpdf(Normal(u[selected], 1), data.y[i])
                    @test q(u, i) ≈ expected ≈ full[i]
                    @test _bcf_support(fp, u, i) == expected_support
                    if sampler !== nothing
                        gradient = zeros(length(u))
                        gradient[selected] = data.y[i] - u[selected]
                        value, actual = cell_value_and_gradient!(sampler, similar(u), u, i)
                        @test value ≈ expected
                        @test actual ≈ gradient
                    end
                end
                @test u == saved
                @test data == before
            end
            @test_throws BoundsError q(u, n + 1)
            @test_throws BoundsError _bcf_support(fp, u, n + 1)
            q(u, 1)
            _bcf_support(fp, u, 1)
            push!(costs, (; bytes = minimum(@allocated(q(u, 1)) for _ in 1:3),
                footprint_bytes = minimum(@allocated(_bcf_support(fp, u, 1)) for _ in 1:3),
                nodes = _bcf_nodes(q.kernel.ast), loops = _bcf_heads(q.kernel.ast, :for),
                branches = _bcf_heads(q.kernel.ast, :if)))
        end
        @test costs[1] == costs[2]
        @test costs[1].loops > 0
        @test costs[1].branches > 0
        @info "retained branch cell work" named costs
    end
end

@testset "branch row proofs preserve inactive errors" begin
    # The second arm's index is invalid. It must stay lazy; the known-safe
    # first arm is not permission to request the second arm's row eagerly.
    source = quote
        a ~ Normal(0, 1)
        z[1:3, 1:2] .~ Normal.(0, 1)
        @plate for i in eachindex(y)
            mu = a > 0 ? z[g[i], 1] : z[bad[i], 2]
            y[i] ~ Normal(mu, 1)
        end
    end
    data = (; g = [1, 2], bad = [4, 0], y = [0.1, 0.2])
    bound, built, q = _bcf_build(source, data)
    u = fill(0.5, built.layout.total)
    sampler = prepare_cell_sampler(built, bound, :y, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    for i in 1:2
        @test q(u, i) ≈ logpdf(Normal(u[1 + i], 1), data.y[i])
        @test partition_footprint(q.kernel, u, i).unconstrained == collect(eachindex(u))
        value, gradient = cell_value_and_gradient!(sampler, similar(u), u, i)
        @test value ≈ q(u, i)
        expected = zeros(length(u))
        expected[1 + i] = data.y[i] - u[1 + i]
        @test gradient ≈ expected
    end
    u[1] = -0.5
    @test_throws BoundsError q(u, 1)
    @test_throws BoundsError q(u, 2)

    # Both row indices are valid, but the inactive arithmetic is not. The
    # window may gather coordinates, never execute that arm's sqrt early.
    guarded = quote
        a ~ Normal(0, 1)
        z[levels(group), 1:2] .~ Normal.(0, 1)
        @plate for i in eachindex(y)
            mu = a > 0 ? z[group[i], 1] : sqrt(-z[group[i], 2])
            y[i] ~ Normal(mu, 1)
        end
    end
    bound, built, q = _bcf_build(guarded, (; group = [1, 2], y = [0.1, 0.2]))
    u = fill(0.5, built.layout.total)
    sampler = prepare_cell_sampler(built, bound, :y, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    @test q(u, 1) ≈ logpdf(Normal(0.5, 1), 0.1)
    @test partition_footprint(q.kernel, u, 1).unconstrained == [1, 2, 4]
    value, gradient = cell_value_and_gradient!(sampler, similar(u), u, 1)
    @test value ≈ q(u, 1)
    @test gradient ≈ [0.0, -0.4, 0.0, 0.0, 0.0]
    u[1] = -0.5
    @test_throws DomainError q(u, 1)
end
