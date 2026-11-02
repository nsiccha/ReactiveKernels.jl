module CellFactorArrayAddTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme, SecondOrder

add_arrays(a, b) = a + b
const models = (
    direct = @rkppl(begin
        z[1:n] .~ Normal.(0, 1)
        w[1:n] .~ Normal.(0, 1)
        @plate for k in eachindex(rows)
            y[k] ~ Normal(sum(z[rows[k]] + w[rows[k]]), 1)
        end
    end),
    helper = @rkppl(begin
        z[1:n] .~ Normal.(0, 1)
        w[1:n] .~ Normal.(0, 1)
        @plate for k in eachindex(rows)
            y[k] ~ Normal(sum(add_arrays(z[rows[k]], w[rows[k]]) .^ 2), 1)
        end
    end))

struct RecordedPoint <: AbstractVector{Float64}
    values::Vector{Float64}
    reads::Vector{Int}
end
Base.size(u::RecordedPoint) = size(u.values)
Base.IndexStyle(::Type{RecordedPoint}) = IndexLinear()
Base.getindex(u::RecordedPoint, i::Int) = (push!(u.reads, i); u.values[i])
nodes(x) = x isa Expr ? 1 + sum(nodes, x.args; init=0) : 1
support(footprint, args...) = Base.invokelatest(footprint, args...).unconstrained

@testset "ordinary array sums expose strict local factors" begin
    for (name, model) in pairs(models), cached in (false, true)
        sizes = Int[]
        for n in (4, 80)
            rows = [vcat([1, 2, 2], collect(3:n)), Int[], [4, 4]]
            y = [0.2, -0.1, 0.7]
            bound = model(; n, rows) | (; y)
            built = build_kernel(bound)
            u = collect(range(-0.3, 0.4; length=built.layout.total))
            params = constrain(built.layout, u)
            names = string.(coordinate_names(built.layout))
            zi = [findfirst(==("z.$i"), names) for i in 1:n]
            wi = [findfirst(==("w.$i"), names) for i in 1:n]
            options = cached ? (; partitioned=(:z=>collect(1:n), :w=>collect(1:n)), u0=u) : (;)
            factors = prepare_cell_factors(built, bound, :y; locality=:error, options...)
            @test length(cell_reductions(factors)) == 1
            term = prepare_cell_factor_query(factors, 1)
            outer = prepare_cell_factor_query(factors, :outer)
            footprint = partition_footprint_kernel(term.kernel)
            push!(sizes, nodes(code_expr(term.kernel)))
            whole = prepare_query(built, bound, :pointwise)(u).y
            for cell in eachindex(rows)
                @test reduction_rows(factors, 1, cell) == rows[cell]
                values = Float64[]
                for (t, row) in enumerate(rows[cell])
                    expected = params.z[row] + params.w[row]
                    name === :helper && (expected = expected^2)
                    point = RecordedPoint(u, Int[])
                    @test term(point, cell, t) ≈ expected
                    @test sort!(unique!(point.reads)) == sort([zi[row], wi[row]])
                    @test support(footprint, u, cell, t) == sort([zi[row], wi[row]])
                    push!(values, term(u, cell, t))
                end
                total = sum(values; init=0.0)
                @test outer(u, [total], cell) ≈ whole[cell] ≈ logpdf(Normal(total, 1), y[cell])
            end
            @test_throws BoundsError term(u, 2, 1)
            @test isempty(ReactiveKernels._code_findings(term.kernel,
                (Vector{Float64}, Int, Int); recurse=ReactiveKernels._rk_or_generated(@__MODULE__)))
            if n == 4
                indices = support(footprint, u, 1, 1)
                reverse = AutoEnzyme(; mode=Enzyme.Reverse)
                second = SecondOrder(reverse, AutoEnzyme(; mode=Enzyme.Forward))
                g = prepare_cell_factor_ad(prepare_ad, factors, 1, u, indices, reverse)
                direction = [0.3, -0.1]
                h = prepare_cell_factor_ad(prepare_ad_hvp, factors, 1, u, indices,
                    second, (direction,))
                for (cell, t) in ((1, 1), (1, 3), (3, 2))
                    indices = support(footprint, u, cell, t)
                    point = (u, u[indices], indices, cell, t)
                    value, gradient = ad_value_and_gradient(g, point...)
                    _, (hv,) = ad_gradient_and_hvp(h, (direction,), point...)
                    s = sum(u[indices])
                    @test value ≈ (name === :direct ? s : s^2)
                    @test gradient ≈ fill(name === :direct ? 1.0 : 2s, 2)
                    @test hv ≈ fill(name === :direct ? 0.0 : 2sum(direction), 2)
                end
            end
        end
        @test sizes[1] == sizes[2]
    end
end
end
