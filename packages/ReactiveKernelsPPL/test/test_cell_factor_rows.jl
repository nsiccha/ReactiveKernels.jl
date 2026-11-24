module CellFactorRowTests

using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

function checked_rows(n, rows)
    all(j -> 1 <= j <= n, rows) || throw(ArgumentError("row outside sampled vector"))
    rows
end

const CHECKED = @rkppl begin
    z[1:n] .~ Normal.(0, 1)
    @plate for i in eachindex(y)
        rows = checked_rows(length(z), bins[i])
        y[i] ~ Normal(sum(z[rows]), 1)
    end
end

@testset "checked gather preserves the likelihood and bounds check" begin
    bins = [[1, 3, 1], [2], Int[]]
    y = [0.0, 0.2, -0.1]
    u = [0.1, -0.3, 0.2]
    indices = collect(eachindex(u))
    bound = CHECKED(; n=length(u), bins) | (; y)
    built = build_kernel(bound)
    for partitioned in ((), (:z => indices,))
        f = prepare_cell_factors(built, bound, :y; partitioned, u0=u, locality=:error)
        outer = prepare_cell_factor_query(f, :outer)
        terms = [prepare_cell_factor_query(f, j) for j in eachindex(cell_reductions(f))]
        grad = prepare_cell_factor_ad(prepare_ad, f, :outer, u, indices,
            AutoEnzyme(; mode=Enzyme.Reverse))
        for cell in eachindex(y)
            rows = bins[cell]
            residual = y[cell] - sum(u[rows])
            expected = logpdf(Normal(sum(u[rows]), 1), y[cell])
            sums = Float64[sum((term(u, cell, k) for k in eachindex(reduction_rows(f, j, cell)));
                               init=0.0) for (j, term) in enumerate(terms)]
            @test outer(u, sums, cell) ≈ expected
            value, (partials, direct) = ad_value_and_gradient(grad, u, sums, u, indices, cell)
            @test value ≈ expected
            counts = [count(==(j), rows) for j in indices]
            # If the graph proves the length from data, account for the
            # extracted sum's derivative. Otherwise all reads are direct.
            @test direct + sum(partials) * counts ≈ residual * counts
        end
        @test_throws "row outside sampled vector" begin
            invalid = CHECKED(; n=length(u), bins=[[4]]) | (; y=[0.0])
            bad = prepare_cell_factors(build_kernel(invalid), invalid, :y;
                partitioned, u0=u, locality=:error)
            for j in eachindex(cell_reductions(bad))
                reduction_rows(bad, j, 1)
            end
            prepare_cell_factor_query(bad, :outer)(u, zeros(length(cell_reductions(bad))), 1)
        end
    end
end

select_rows(x, rows) = x > 0 ? rows : Int[]
const NESTED = @rkppl begin
    z[1:n] .~ Normal.(0, 1)
    @plate for i in eachindex(y)
        rows = select_rows(z[1], bins[i])
        inner = sum(z[rows])
        total = sum(inner .* z[bins[i]])
        y[i] ~ Normal(total + sum(z[other[i]]), 1)
    end
end

@testset "a point-dependent inner sum leaves its enclosing sum in the outer" begin
    bins, other = [[1, 3], [2]], [[2, 2], [1]]
    y = [0.0, 0.2]
    bound = NESTED(; n=3, bins, other) | (; y)
    built = build_kernel(bound)
    for partitioned in ((), (:z => 1:3,))
        f = prepare_cell_factors(built, bound, :y; partitioned, u0=zeros(3), locality=:error)
        @test length(cell_reductions(f)) == 1
        term, outer = prepare_cell_factor_query(f, 1), prepare_cell_factor_query(f, :outer)
        for u in ([0.1, -0.3, 0.2], [-0.4, 0.2, 0.3]), i in eachindex(y)
            @test reduction_rows(f, 1, i) == other[i]
            terms = [term(u, i, t) for t in eachindex(other[i])]
            @test terms == u[other[i]]
            inner = sum(u[select_rows(u[1], bins[i])])
            mu = sum(inner .* u[bins[i]]) + sum(u[other[i]])
            @test outer(u, [sum(terms)], i) ≈ logpdf(Normal(mu, 1), y[i])
        end
    end
end

end
