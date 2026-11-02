using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions

const _CFV_MODEL = @rkppl begin
    b ~ Normal(0, 1)
    z[levels(g)] .~ Normal.(0, 1)
    sigma ~ Exponential(1)
    eta = b .* x .+ z[g]
    @plate for k in eachindex(y)
        y[k] ~ Normal(sum(exp.(eta[rows[k]])), sigma)
    end
end

_cfv_field(w) = map(exp, w)
const _CFV_SHARED = @rkppl begin
    w[levels(id)] .~ Normal.(0, 1)
    z[levels(id)] .~ Normal.(0, 1)
    field = _cfv_field(w)
    rate = exp.(z[id]) .* field
    @plate for k in eachindex(y)
        y[k] ~ Normal(log1p(sum(rate[rows[k]])) + w[1], 1)
    end
end

function _cfv_batch(f::F, args...) where {F}
    for _ in 1:100
        f(args...)
    end
end

function _cfv_bytes(f, args...)
    _cfv_batch(f, args...)
    minimum([@allocated(_cfv_batch(f, args...)) for _ in 1:3]) / 100
end
_cfv_kernel(q, u, a, b) = Base.invokelatest(q.kernel, u, a, b)::Float64
_cfv_query(q, u, a, b) = q(u, a, b)

@testset "Factor value wrappers preserve concrete calls" begin
    for n in (8, 128)
        data = (; g = collect(1:n), x = sin.(1:n),
                  rows = [[1, 1, 3], Int[], [n]], y = [0.3, -0.2, 0.5])
        bound = _CFV_MODEL(; data.g, data.x, data.rows) | (; data.y)
        built = build_kernel(bound)
        u = fill(0.01, built.layout.total)
        factors = prepare_cell_factors(built, bound, :y; locality = :error)
        term = prepare_cell_factor_query(factors, 1)
        outer = prepare_cell_factor_query(factors, :outer)
        full = prepare_cell_query(built, bound, :y; locality = :error)
        sums = [sum((term(u, 1, j) for j in 1:3); init = 0.0)]
        for (query, a, b) in ((term, 1, 1), (outer, sums, 1))
            @test query(u, a, b) == _cfv_kernel(query, u, a, b)
            # The public wrapper may validate its arguments, but must not
            # repeatedly box the prepared program to splat an empty cache.
            direct = _cfv_bytes(_cfv_kernel, query, u, a, b)
            wrapped = _cfv_bytes(_cfv_query, query, u, a, b)
            @test wrapped <= direct + 128
        end
        for cell in eachindex(data.y)
            total = sum((term(u, cell, j) for j in eachindex(data.rows[cell])); init = 0.0)
            @test outer(u, [total], cell) ≈ full(u, cell)
        end
        @test_throws BoundsError term(u, 2, 1)
        @test_throws ContractValidationError outer(u, Float64[], 1)
    end
end

@testset "Factor values refresh shared producers" begin
    data = (; id = collect(1:4), rows = [[1, 1, 3], Int[], [2, 4]],
              y = [0.3, -0.2, 0.5])
    bound = _CFV_SHARED(; data.id, data.rows) | (; data.y)
    built = build_kernel(bound)
    u = collect(range(-0.2, 0.3; length = built.layout.total))
    factors = prepare_cell_factors(built, bound, :y; partitioned = :z, u0 = u)
    term = prepare_cell_factor_query(factors, 1)
    outer = prepare_cell_factor_query(factors, :outer)
    full = prepare_cell_query(built, bound, :y; locality = :ignore)
    for shift in (0.0, 0.2, -0.1)
        point = u .+ shift
        for cell in eachindex(data.y)
            total = 0.0
            for j in eachindex(data.rows[cell])
                value = term(point, cell, j)
                @test value ≈ _cfv_kernel(term, point, cell, j)
                total += value
            end
            @test outer(point, [total], cell) ≈ _cfv_kernel(outer, point, [total], cell)
            @test outer(point, [total], cell) ≈ full(point, cell)
        end
    end
end
