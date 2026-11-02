using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme
using DifferentiationInterface: AutoEnzyme, SecondOrder

# map remains a whole-array helper; the independent helper-locality lowering
# can now evaluate a collect(Generator) directly at rows.
_cs_field(grid, beta) = map(t -> exp(beta * t), grid)
_cs_vector_field(grid, beta) = map((t, b) -> exp(b * t), grid, beta)
const _CS_WIDE = @rkppl begin
    beta[levels(id)] .~ Normal.(0, 1)
    z[levels(id)] .~ Normal.(0, 1)
    field = _cs_vector_field(grid, beta)
    @plate for k in eachindex(y)
        y[k] ~ Poisson(exp(z[id[k]]) * field[k])
    end
end
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
const _CS_LOCAL = @rkppl begin
    beta ~ Normal(0, 1)
    z[levels(id)] .~ Normal.(0, 1)
    matrix = reshape(_cs_field(grid, beta), :, 1)
    field = vec(matrix)
    @plate for k in eachindex(y)
        y[k] ~ Poisson(exp(z[id[k]]) * field[k])
    end
end
const _CS_NO_PREFIX = @rkppl begin
    z[levels(id)] .~ Normal.(0, 1)
    @plate for k in eachindex(y)
        y[k] ~ Poisson(exp(z[id[k]]))
    end
end
const _CS_VARYING = @rkppl begin
    beta ~ Normal(0, 1)
    z[levels(id)] .~ Normal.(0, 1)
    field = _cs_field(grid, beta + sum(z))
    y .~ Poisson.(field)
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
    reference = prepare_cell_query(built, bound, :y; locality = :ignore)
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
            function contraction(point)
                sum(eachindex(data.y)) do i
                    if order == 0
                        weights[i] * reference(point, i)
                    else
                        g, (hv,), _ = ad_gradient_hvp_and_third(q, ([0.6],),
                            point, point[[i+1]], [i+1], i)
                        weights[i] * 0.6 * only(order == 1 ? g : hv)
                    end
                end
            end
            epsilon = 1e-5
            fd = map(eachindex(u)) do j
                hi, lo = copy(u), copy(u)
                hi[j] += epsilon; lo[j] -= epsilon
                (contraction(hi) - contraction(lo)) / (2epsilon)
            end
            @test gradient ≈ fd rtol = 2e-8 atol = 2e-9
        end
    end
    gradient = fill(123.0,length(u))
    @test_throws ErrorException cell_pullback!(_ -> error("abort pass"), gradient, p, u)
    @test all(==(123.0), gradient)
    cell_pullback!(_ -> nothing, gradient, p, u)
    @test all(iszero, gradient)
    @test_throws ContractValidationError ad_gradient_hvp_and_third(q,
        ([0.6],), p, u[[2]], [2], 1)
    other = prepare_compact_cell_ad(prepare_ad_third, built, bound, :y, u, [2],
        _CS_THIRD, ([0.6],); partitioned = :z, locality = :error)
    cell_pullback!(gradient, p, u) do batch
        @test_throws ContractValidationError ad_gradient_hvp_and_third(other,
            ([0.6],), batch, u[[2]], [2], 1)
    end
    @test_throws ContractValidationError cell_pullback!(gradient, p, u) do batch
        changed = copy(u); changed[1] += 0.1
        ad_gradient_hvp_and_third(q, ([0.6],), changed, changed[[2]], [2], 1)
        @test_throws ContractValidationError ad_gradient_hvp_and_third(q,
            ([0.6],), batch, u[[2]], [2], 1)
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
    localterm = partition_footprint_kernel(term; shared=false)
    localouter = partition_footprint_kernel(outer; shared=false)
    gradient = zeros(length(u))
    cell_pullback!(gradient,p,u) do batch
        @test outer(batch,[sum(rate[data.rows[1]])],1) ≈ outer(u,[sum(rate[data.rows[1]])],1)
        @test isempty(localouter(batch,[sum(rate[data.rows[1]])],1).unconstrained)
        for (j,r) in enumerate(data.rows[1])
            @test term(batch,1,j) ≈ rate[r]
            @test localterm(batch,1,j).unconstrained == [r+1]
            g, (hv,), (tv,) = ad_gradient_hvp_and_third(q, ([0.6],),
                batch, u[[r+1]], [r+1], 1, j)
            @test g ≈ [rate[r]]
            @test hv ≈ [rate[r]*0.6]
            @test tv ≈ [rate[r]*0.6^2]
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
    reference = ad_gradient_hvp_and_third(qouter, (([0.6],Float64[]),),
        u, [s], Float64[], Int[], 1)
    cell_pullback!(gradient,p,u) do batch
        @test ad_gradient_hvp_and_third(qouter, (([0.6],Float64[]),),
            batch, [s], Float64[], Int[], 1) == reference
        value, local_gradient = accumulate_cell!(batch,aouter,([0.6],Float64[]),1.0,
            [s],Float64[],Int[],1)
        @test length(local_gradient[1]) == 1
        @test isempty(local_gradient[2])
    end
    @test gradient[1] ≈ 0.6^2/(1+s)^2
    @test all(iszero,gradient[2:end])
end

@testset "Fixed-point local derivatives with many shared coordinates" begin
    measurements = NamedTuple[]
    for n in (8, 1024)
        data = (; id=collect(1:n), grid=collect(range(-0.7,0.9; length=n)), y=mod.(1:n,4))
        bound = _CS_WIDE(; data.id, data.grid) | (; data.y)
        built = build_kernel(bound)
        u = collect(range(-0.2,0.3; length=2n))
        active, indices, direction = u[[n+1]], [n+1], [0.6]
        q = prepare_compact_cell_ad(prepare_ad_third, built, bound, :y, u, indices,
            _CS_THIRD, (direction,); partitioned=:z, locality=:error)
        @test length(q.shared.shared_positions) == n
        p = prepare_cell_pullback(q, _CS_REVERSE, u)
        gradient = similar(u)
        reference = ad_gradient_hvp_and_third(q, (direction,), u, active, indices, 1)
        original = u[1]
        cell_pullback!(gradient, p, u) do batch
            # The callback owns a copy; changing the caller's vector cannot
            # silently refresh the fixed point used by local derivatives.
            u[1] += 0.5
            @test ad_gradient_hvp_and_third(q, (direction,), batch, active, indices, 1) == reference
            bytes = minimum(@allocated(ad_gradient_hvp_and_third(q,
                (direction,), batch, active, indices, 1)) for _ in 1:3)
            push!(measurements, (; n, bytes))
        end
        @test all(iszero, gradient) # ordinary verbs do not accumulate
        g, (hv,), (tv,) = ad_gradient_hvp_and_third(q, (direction,), u, active, indices, 1)
        rate = exp(active[1] + u[1]*data.grid[1])
        @test g ≈ [data.y[1]-rate]
        @test hv ≈ [-rate*0.6]
        @test tv ≈ [-rate*0.6^2]
        u[1] = original
        a = prepare_cell_accumulator(q, _CS_THIRD, u; order=2)
        cell_pullback!(gradient, p, u) do batch
            for i in 1:n
                indices[1] = n+i; active[1] = u[n+i]
                accumulate_cell!(batch, a, direction, 1.0, active, indices, i)
            end
        end
        expected = -exp.(u[n+1:end] .+ u[1:n].*data.grid) .* 0.6^2
        @test gradient[n+1:end] ≈ expected
        @test gradient[1:n] ≈ expected.*data.grid
    end
    @info "fixed-point local derivative allocations" measurements
    @test last(measurements).bytes <= first(measurements).bytes + 2048
end

function _cs_sweep!(gradient, p, a, u, direction, active, indices)
    cell_pullback!(gradient, p, u) do batch
        for i in 1:(length(u)-1)
            indices[1] = i+1
            active[1] = u[i+1]
            accumulate_cell!(batch, a, direction, 1.0, active, indices, i)
        end
    end
end

@testset "Shared derivative storage scales with the point" begin
    measurements = NamedTuple[]
    for n in (8, 128, 1024)
        data = (; id = collect(1:n), grid = collect(range(-0.7,0.9; length=n)),
            y = mod.(1:n,4))
        bound = _CS_LOCAL(; data.id, data.grid) | (; data.y)
        built = build_kernel(bound)
        u = collect(range(-0.2,0.3; length=n+1))
        active, indices, direction = u[[2]], [2], [0.6]
        q = prepare_compact_cell_ad(prepare_ad_third, built, bound, :y, u, indices,
            _CS_THIRD, (direction,); partitioned=:z, locality=:error)
        @test q.shared.cache !== nothing
        @test any(v -> v isa AbstractArray && length(v) == n, q.shared.cache.values)
        a = prepare_cell_accumulator(q, _CS_THIRD, u; order=2)
        p = prepare_cell_pullback(q, _CS_REVERSE, u)
        gradient = similar(u)
        _cs_sweep!(gradient, p, a, u, direction, active, indices)
        expected = -exp.(u[2:end] .+ u[1].*data.grid) .* 0.6^2
        @test gradient[2:end] ≈ expected
        @test gradient[1] ≈ sum(expected.*data.grid)
        bytes = minimum(@allocated(_cs_sweep!(gradient, p, a, u, direction,
                                             active, indices)) for _ in 1:3)
        active[1] = u[2]; indices[1] = 2
        ad_gradient_hvp_and_third(q, (direction,), u, active, indices, 1)
        local_bytes = minimum(@allocated(ad_gradient_hvp_and_third(q,
            (direction,), u, active, indices, 1)) for _ in 1:3)
        push!(measurements, (; n, bytes, local_bytes,
            storage = Base.summarysize((p, a))))
    end
    @info "shared derivative allocation measurements" measurements
    small, middle, large = measurements
    @test large.local_bytes <= small.local_bytes + 2048
    # Bound the observed slope across two intervals, allowing fixed overhead.
    @test large.bytes <= 10middle.bytes + 65536
    @test large.storage <= 10middle.storage + 65536
end

@testset "Conditional cells without a prefix and unsafe shared cuts" begin
    data = (; id=[1,2,3], grid=[-0.3,0.2,0.7], y=[1,2,3])
    bound = _CS_NO_PREFIX(; data.id) | (; data.y)
    built = build_kernel(bound)
    u = [0.1,-0.2,0.3]
    q = prepare_compact_cell_ad(prepare_ad_third, built, bound, :y, u, [1],
        _CS_THIRD, ([0.6],); partitioned=:z, locality=:error)
    @test q.shared.cache === nothing
    p = prepare_cell_pullback(q, _CS_REVERSE, u)
    a = prepare_cell_accumulator(q, _CS_THIRD, u; order=2)
    gradient = similar(u)
    cell_pullback!(gradient,p,u) do batch
        for i in eachindex(u)
            accumulate_cell!(batch,a,[0.6],1.0,u[[i]],[i],i)
        end
    end
    @test gradient ≈ -exp.(u).*0.6^2
    varying = _CS_VARYING(; data.id, data.grid) | (; data.y)
    varying_built = build_kernel(varying)
    @test_throws ReactiveKernels.PartitionLocalityError prepare_compact_cell_ad(
        prepare_ad_third,varying_built,varying,:y,zeros(4),[2],_CS_THIRD,([0.6],);
        partitioned=:z,locality=:error)
end
