using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

# Array-valued fields of one shared plate result feed a ragged observation.
# The other observation keeps the model's in-cell sharing in the authored graph.
const _CFR_MODEL = @rkppl begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    sigma ~ Exponential(1)
    z[levels(g)] .~ Normal.(0, 1)
    eta = a .* x .+ z[g]
    @plate for i in eachindex(rows)
        mean[i] = sin.(eta[rows[i]])
        auxiliary[i] .~ Normal.(a, 1)
        y[i] .~ censored.(Normal.(mean[i], sqrt.(sigma^2 .+ (b .* mean[i]).^2)),
                          lower[i], Inf)
    end
end

# Selecting input storage must not bless an opaque computed array. Its own
# elements still need a row form when an individual summand requests them.
_cfr_opaque(x) = reverse(x)
const _CFR_OPAQUE = @rkppl begin
    z[levels(g)] .~ Normal.(0, 1)
    eta = z[g]
    @plate for i in eachindex(rows)
        mean[i] = _cfr_opaque(eta[rows[i]]) .+ x[rows[i]]
        y[i] .~ Normal.(mean[i], 1)
    end
end

struct _CFRPoint <: AbstractVector{Float64}
    values::Vector{Float64}
    reads::Vector{Int}
end
Base.size(u::_CFRPoint) = size(u.values)
Base.IndexStyle(::Type{_CFRPoint}) = IndexLinear()
Base.getindex(u::_CFRPoint, i::Int) = (push!(u.reads, i); u.values[i])

_cfr_nodes(x) = x isa Expr ? 1 + sum(_cfr_nodes, x.args; init=0) : 1

function _cfr_data(n)
    rows = [vcat([1, 2, 2], collect(3:n)), Int[], [3, 1]]
    lower = [fill(-0.15, length(r)) for r in rows]
    y = [[isodd(j) ? -0.15 : 0.4 for j in eachindex(r)] for r in rows]
    (; g=collect(1:n), x=cos.(collect(1:n)), rows, lower, y,
       auxiliary=[[0.2, -0.1], Float64[], [0.3]])
end

function _cfr_oracle(built, data, u, cell, term)
    p = constrain(built.layout, u)
    row = data.rows[cell][term]
    mu = sin(p.a * data.x[row] + p.z[row])
    sd = sqrt(p.sigma^2 + (p.b * mu)^2)
    logpdf(censored(Normal(mu, sd), data.lower[cell][term], Inf), data.y[cell][term])
end

@testset "ragged factors retain shared array fields and input storage" begin
    emitted = Int[]
    for n in (6, 120)
        data = _cfr_data(n)
        bound = _CFR_MODEL(; data.g, data.x, data.rows, data.lower) |
                (; data.y, data.auxiliary)
        built = build_kernel(bound)
        u = unconstrain(built.layout, (; a=0.4, b=0.15, sigma=0.8,
                         z=collect(range(-0.2, 0.3; length=n))))
        factors = prepare_cell_factors(built, bound, :y; locality=:error)
        @test length(cell_reductions(factors)) == 1
        term = prepare_cell_factor_query(factors, 1)
        outer = prepare_cell_factor_query(factors, :outer)
        footprint = partition_footprint_kernel(term.kernel)
        push!(emitted, _cfr_nodes(code_expr(term.kernel)))
        whole = prepare_query(built, bound, :pointwise)(u).y
        for cell in eachindex(data.rows)
            @test reduction_rows(factors, 1, cell) == data.rows[cell]
            values = [term(u, cell, j) for j in eachindex(data.rows[cell])]
            expected = Float64[_cfr_oracle(built, data, u, cell, j)
                        for j in eachindex(data.rows[cell])]
            @test values ≈ expected
            @test outer(u, [sum(values; init=0.0)], cell) ≈ sum(whole[cell]; init=0.0)
        end
        # Both the censored atom and the density branch, repeated source rows,
        # and a second group. Actual reads and structural footprints stay local.
        for (cell, j) in ((1, 1), (1, 2), (1, 3), (3, 2))
            point = _CFRPoint(u, Int[])
            @test term(point, cell, j) ≈ _cfr_oracle(built, data, u, cell, j)
            indices = Base.invokelatest(footprint, u, cell, j).unconstrained
            @test indices isa Vector{Int}
            @test sort!(unique!(point.reads)) == indices
            @test length(indices) == 4
            ad = prepare_cell_factor_ad(prepare_ad, factors, 1, u, indices,
                AutoEnzyme(; mode=Enzyme.Reverse); cell0=cell, term0=j)
            value, gradient = ad_value_and_gradient(ad, u, u[indices], indices, cell, j)
            @test value ≈ _cfr_oracle(built, data, u, cell, j)
            for (k, index) in enumerate(indices)
                plus, minus = copy(u), copy(u)
                plus[index] += 1e-6
                minus[index] -= 1e-6
                expected = (_cfr_oracle(built, data, plus, cell, j) -
                            _cfr_oracle(built, data, minus, cell, j)) / 2e-6
                @test gradient[k] ≈ expected atol=2e-8 rtol=2e-6
            end
        end
    end
    # Growing the packed point and ragged term count does not expand RK code.
    @test emitted[1] == emitted[2]
end

@testset "ragged factor locality still rejects opaque computed arrays" begin
    data = _cfr_data(6)
    bound = _CFR_OPAQUE(; data.g, data.x, data.rows) | (; data.y)
    factors = prepare_cell_factors(build_kernel(bound), bound, :y; locality=:error)
    @test length(cell_reductions(factors)) == 1
    @test_throws PartitionLocalityError prepare_cell_factor_query(factors, 1)
end
