using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme
using DifferentiationInterface: AutoEnzyme, SecondOrder

_cs_field(grid, beta) = [exp(beta * t) for t in grid]
const _CS_MODEL = @rkppl begin
    beta ~ Normal(0, 1)
    z[levels(id)] .~ Normal.(0, 1)
    field = _cs_field(grid, beta)
    zi = z[id]
    rate = exp.(zi) .* field
    y .~ Poisson.(rate)
end
const _CS_SUM = @rkppl begin
    beta ~ Normal(0, 1)
    z[levels(id)] .~ Normal.(0, 1)
    field = _cs_field(grid, beta)
    rate = exp.(z[id]) .* field
    @plate for k in eachindex(y)
        y[k] ~ Normal(log1p(sum(rate[rows[k]])) + beta, 1.0)
    end
end
const _CS_REVERSE = AutoEnzyme(; mode = Enzyme.Reverse)
const _CS_SECOND = SecondOrder(_CS_REVERSE, AutoEnzyme(; mode = Enzyme.Forward))
const _CS_THIRD = SecondOrder(_CS_SECOND, AutoEnzyme(; mode = Enzyme.Forward))

function _cs_fixture(n = 4)
    data = (; id = collect(1:n), grid = collect(range(-0.7,0.9; length = n)), y = mod.(1:n,4))
    bound = _CS_MODEL(; data.id, data.grid) | (; data.y)
    built = build_kernel(bound)
    u = collect(range(-0.2,0.3; length = built.layout.total))
    (; data, bound, built, u)
end

@testset "Shared compact cells and mixed derivatives" begin
    (; data, bound, built, u) = _cs_fixture()
    q = prepare_compact_cell_ad(prepare_ad_third, built, bound, :y, u, [2],
        _CS_THIRD, ([0.6],); partitioned = :z, locality = :error)
    @test_throws ContractValidationError prepare_compact_cell_ad(prepare_ad_third,
        built, bound, :y, u, [1,2], _CS_THIRD, ([0.0,0.6],);
        partitioned = :z, locality = :error)
    cached = q.shared.cache.values
    for i in 1:4
        g, (hv,), (tv,) = ad_gradient_hvp_and_third(q, ([0.6],), u, u[[i+1]], [i+1], i)
        rate = exp(u[i+1] + u[1]*data.grid[i])
        @test g ≈ [data.y[i]-rate]
        @test hv ≈ [-rate*0.6]
        @test tv ≈ [-rate*0.6^2]
    end
    @test q.shared.cache.values === cached
    p = prepare_cell_pullback(q, _CS_REVERSE, u)
    weights = [0.3,-0.4,1.2,0.8]
    for (order, backend) in ((0,_CS_REVERSE),(1,_CS_SECOND),(2,_CS_THIRD))
        a = prepare_cell_accumulator(q, backend, u; order)
        gradient = zeros(length(u))
        for beta in (-0.2,0.35)
            u[1] = beta
            cell_pullback!(gradient, p, u) do batch
                for i in 1:4
                    accumulate_cell!(batch, a, order == 0 ? nothing : [0.6],
                        weights[i], u[[i+1]], [i+1], i)
                end
            end
            rate = exp.(u[2:end] .+ beta .* data.grid)
            expected = weights .* (order == 0 ? data.y .- rate : -rate .* 0.6^order)
            @test gradient[2:end] ≈ expected
            @test gradient[1] ≈ sum(data.grid .* expected)
            @test !p.active
        end
    end
    gradient = fill(123.0,length(u))
    @test_throws ErrorException cell_pullback!(_ -> error("abort pass"), gradient, p, u)
    @test all(==(123.0), gradient)
    cell_pullback!(_ -> nothing, gradient, p, u)
    @test all(iszero, gradient)
    @test_throws ContractValidationError cell_pullback!(gradient, p, u) do batch
        changed = copy(u); changed[1] += 0.1
        ad_gradient_hvp_and_third(q, ([0.6],), changed, changed[[2]], [2], 1)
    end
    cell_pullback!(_ -> nothing, gradient, p, u)
    @test all(iszero, gradient)
end

@testset "Shared sum factors" begin
    data = (; id = collect(1:4), grid = [-0.7,-0.2,0.4,0.9],
        rows = [[1,2,2],[3,4],Int[]], y = [0.2,0.4,0.0])
    bound = _CS_SUM(; data.id, data.grid, data.rows) | (; data.y)
    built = build_kernel(bound)
    u = [0.2,-0.1,0.3,0.4,0.1]
    f = prepare_cell_factors(built, bound, :y; partitioned = :z, u0 = u, locality = :error)
    @test length(cell_reductions(f)) == 1
    @test reduction_rows(f,1,1) == data.rows[1]
    @test isempty(reduction_rows(f,1,3))
    term = prepare_cell_factor_query(f,1)
    outer = prepare_cell_factor_query(f,:outer)
    rate = exp.(u[2:end] .+ u[1].*data.grid)
    for i in 1:3
        s = sum(rate[data.rows[i]])
        @test outer(u,[s],i) ≈ logpdf(Normal(log1p(s)+u[1],1),data.y[i])
        for (j,r) in enumerate(data.rows[i])
            @test term(u,i,j) ≈ rate[r]
        end
    end
    # Inspection still has the ordinary public boundary and full shared reads.
    fp = partition_footprint_kernel(term.kernel)
    @test 1 in Base.invokelatest(fp,u,1,1).unconstrained
    q = prepare_cell_factor_ad(prepare_ad_third,f,1,u,[2],_CS_THIRD,([0.6],))
    a = prepare_cell_accumulator(q,_CS_THIRD,u;order=2)
    p = prepare_cell_pullback(f,_CS_REVERSE,u)
    gradient = zeros(length(u))
    cell_pullback!(gradient,p,u) do batch
        for (j,r) in enumerate(data.rows[1])
            accumulate_cell!(batch,a,[0.6],1.0,u[[r+1]],[r+1],1,j)
        end
    end
    counts = [1,2,0,0]
    expected = counts .* rate .* 0.6^2
    @test gradient[2:end] ≈ expected
    @test gradient[1] ≈ sum(expected.*data.grid)
    # Outer sum values are independent; only its direct/shared derivative is
    # included here. Its sum covector is returned for the caller's chain rule.
    qouter = prepare_cell_factor_ad(prepare_ad_third,f,:outer,u,Int[],_CS_THIRD,
        (([0.6],Float64[]),))
    aouter = prepare_cell_accumulator(qouter,_CS_THIRD,u;order=2)
    s = sum(rate[data.rows[1]])
    cell_pullback!(gradient,p,u) do batch
        value, local_gradient = accumulate_cell!(batch,aouter,([0.6],Float64[]),1.0,
            [s],Float64[],Int[],1)
        @test length(local_gradient[1]) == 1
        @test isempty(local_gradient[2])
    end
    @test gradient[1] ≈ 0.6^2/(1+s)^2
    @test all(iszero,gradient[2:end])
end
