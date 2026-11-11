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
# The prefix caches sigma's transform; `a` is shared too, but only the residual
# reads it, at rows. In both orders the prefix reads fewer coordinates than are
# shared, and which one it reads comes first in the second model.
const _CS_READS_FIRST = @rkppl begin
    sigma ~ Exponential(1)
    a ~ Normal(0, 1)
    z[levels(id)] .~ Normal.(0, 1)
    mu = a .+ z[id] .* grid
    y .~ Normal.(mu, sigma)
end
const _CS_READS_SECOND = @rkppl begin
    a ~ Normal(0, 1)
    sigma ~ Exponential(1)
    z[levels(id)] .~ Normal.(0, 1)
    mu = a .+ z[id] .* grid
    y .~ Normal.(mu, sigma)
end
const _CS_READS_BLOCK = @rkppl begin
    sigma ~ Exponential(1)
    w[1:2] .~ Normal.(0, 1)
    z[1:4] .~ Normal.(0, 1)
    @plate for k in eachindex(y)
        y[k] ~ Normal(w[grp[k]] + sum(z[rows[k]]), sigma)
    end
end
const _CS_READS_SUM = @rkppl begin
    sigma ~ Exponential(1)
    w[1:2] .~ Normal.(0, 1)
    z[levels(id)] .~ Normal.(0, 1)
    rate = exp.(z[id])
    @plate for k in eachindex(y)
        y[k] ~ Normal(log1p(sum(rate[rows[k]])) + w[grp[k]], sigma)
    end
end
# A shared value that is a NamedTuple of scalars beside a matrix, as a BRM
# PK source record is (snag rkppl-zero-cell-a7c54028).
_cs_source(grid, beta) = (; scale = exp(beta), weights = reshape(grid .* beta, :, 1),
    offset = beta^2 / 2)
_cs_source_rate(source, k) = source.scale * exp(source.weights[k, 1]) + source.offset
const _CS_NAMED = @rkppl begin
    beta ~ Normal(0, 1)
    z[levels(id)] .~ Normal.(0, 1)
    source = _cs_source(grid, beta)
    @plate for k in eachindex(y)
        y[k] ~ Poisson(exp(z[id[k]]) * _cs_source_rate(source, k))
    end
end
# A cell that reads more shared values than DifferentiationInterface and
# Enzyme splat statically: each crosses the cell's derivative as an inactive
# HAVE port (snag varyingsource3-b-82246291). Cell k's mean is
# (z[id[k]] + Σ_j grid[k] (beta + j)) / K.
const _CS_MANY_K = 30
_cs_shift(grid, beta, j) = grid .* (beta + j)
const _CS_MANY = let body = Expr(:block, :(beta ~ Normal(0, 1)),
                                 :(z[levels(id)] .~ Normal.(0, 1)))
    for j in 1:_CS_MANY_K
        push!(body.args, :($(Symbol(:w, j)) = _cs_shift(grid, beta, $j)))
    end
    mean = Expr(:call, :+, :(z[id[k]]), (:($(Symbol(:w, j))[k]) for j in 1:_CS_MANY_K)...)
    loop = quote
        @plate for k in eachindex(y)
            y[k] ~ Normal($mean / $_CS_MANY_K, 1.0)
        end
    end
    append!(body.args, loop.args)
    @eval @rkppl $body
end
@isdefined(_static_enzyme) || include("static_enzyme.jl")

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

@testset "A prefix reads only some shared coordinates: read at $read, not at $unread" for
        (model, read, unread) in ((_CS_READS_FIRST, 1, 2), (_CS_READS_SECOND, 2, 1))
    data = (; id = repeat(1:4, inner = 2), grid = sin.(1:8), y = cos.(1:8))
    bound = model(; data.id, data.grid) | (; data.y)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.4; length = built.layout.total))
    local_position(i) = 2 + data.id[i]
    residual(point, i) = data.y[i] - point[unread] - point[local_position(i)] * data.grid[i]
    density(point, i) = logpdf(Normal(0, exp(point[read])), residual(point, i))
    moved(position) = (point = copy(u); point[position] += 0.25; point)

    q = prepare_cell_query(built, bound, :y; partitioned = :z, locality = :ignore)
    # The shape under test: without it, nothing below exercises a partial read.
    @test q.shared.cache.positions == [read]
    @test q(u, 1) ≈ density(u, 1)
    before, values = q.shared.generation, q.shared.cache.values
    for i in (1, 1, 2, 8)
        @test q(u, i) ≈ density(u, i)
    end
    @test q.shared.generation == before
    # A local or residual-only shared coordinate changes the value, not the cache.
    for position in (3, unread)
        @test q(moved(position), 1) ≈ density(moved(position), 1)
        @test q.shared.generation == before
        @test q.shared.cache.values === values
    end
    @test q(moved(read), 1) ≈ density(moved(read), 1)
    @test q.shared.generation == before + 1
    @test q.shared.cache.values !== values
    @test q(u, 1) ≈ density(u, 1)

    fp = partition_footprint_kernel(q; shared = false)
    for _ in 1:2
        @test fp(u, 1).unconstrained == [3]
    end
    # A live pass reads its own copied point and the cache. The residual-only
    # coordinate reaches neither, so changing it elsewhere leaves the pass exact.
    result = with_cell_batch(q, u) do batch
        @test q(batch, 1) ≈ density(u, 1)
        @test q(u, 2) ≈ density(u, 2)
        @test q(moved(unread), 1) ≈ density(moved(unread), 1)
        @test q(batch, 1) ≈ density(u, 1)
        @test fp(batch, 1).unconstrained == [3]
        :result
    end
    @test result === :result
    @test_throws ContractValidationError with_cell_batch(q, u) do batch
        q(moved(read), 1)
        q(batch, 1)
    end

    c = prepare_compact_cell_ad(prepare_ad_third, built, bound, :y, u, [3],
        _CS_THIRD, ([0.6],); partitioned = :z, locality = :ignore)
    @test c.shared.cache.positions == [read]
    @test_throws ContractValidationError prepare_compact_cell_ad(prepare_ad_third,
        built, bound, :y, u, [unread], _CS_THIRD, ([0.6],);
        partitioned = :z, locality = :ignore)
    # Analytic local derivatives of each cell's density in its own z.
    local_gradient(point, i) = residual(point, i) * data.grid[i] / exp(2point[read])
    local_curvature(i, point) = -data.grid[i]^2 / exp(2point[read])
    for i in (1, 1, 2)
        g, (hv,), (tv,) = ad_gradient_hvp_and_third(c, ([0.6],),
            u, u[[local_position(i)]], [local_position(i)], i)
        @test only(g) ≈ local_gradient(u, i)
        @test only(hv) ≈ 0.6 * local_curvature(i, u)
        @test only(tv) ≈ 0.0 atol = 1e-12
    end
    p = prepare_cell_pullback(c, _CS_REVERSE, u)
    weights = [0.3, -0.4, 1.2, 0.8, -0.5, 0.7, 0.2, -1.1]
    contraction(point, order) = sum(eachindex(data.y)) do i
        weights[i] * (order == 0 ? density(point, i) :
            order == 1 ? 0.6 * local_gradient(point, i) : 0.6^2 * local_curvature(i, point))
    end
    for (order, backend) in ((0, _CS_REVERSE), (1, _CS_SECOND), (2, _CS_THIRD))
        a = prepare_cell_accumulator(c, backend, u; order)
        gradient = zeros(length(u))
        for point in (u, moved(unread), moved(read))
            cell_pullback!(gradient, p, point) do batch
                for i in eachindex(data.y)
                    accumulate_cell!(batch, a, order == 0 ? nothing : [0.6], weights[i],
                        point[[local_position(i)]], [local_position(i)], i)
                end
            end
            epsilon = 1e-5
            fd = map(eachindex(point)) do j
                hi, lo = copy(point), copy(point)
                hi[j] += epsilon; lo[j] -= epsilon
                (contraction(hi, order) - contraction(lo, order)) / (2epsilon)
            end
            @test gradient ≈ fd rtol = 2e-8 atol = 2e-8
        end
    end
end

@testset "A prefix reads only some shared coordinates beside a local block read at rows" begin
    data = (; grp = [1, 2, 1], rows = [[1, 2], [3, 4], [2, 3]], y = [0.2, 0.4, 0.0])
    bound = _CS_READS_BLOCK(; data.grp, data.rows) | (; data.y)
    built = build_kernel(bound)
    u = [0.2, -0.1, 0.3, 0.4, 0.1, -0.2, 0.25]
    c = prepare_compact_cell_ad(prepare_ad, built, bound, :y, u, [4, 5], _CS_REVERSE;
        partitioned = :z => collect(1:4), locality = :ignore)
    @test c.shared.cache.positions == [1]
    # `_local_cell_rows` picks each row from the gathered base or the active
    # vector (static_enzyme.jl). Dogfood validation requires the gradient.
    _static_enzyme() do
        for i in (1, 1, 2, 3)
            selection = 3 .+ data.rows[i]
            mu = u[1 + data.grp[i]] + sum(u[selection])
            value, gradient = ad_value_and_gradient(c, u, u[selection], selection, i)
            @test value ≈ logpdf(Normal(mu, exp(u[1])), data.y[i])
            @test gradient ≈ fill((data.y[i] - mu) / exp(2u[1]), 2)
        end
    end
end

@testset "Factors whose prefix reads only some shared coordinates" begin
    data = (; id = collect(1:4), grp = [1, 2, 1], rows = [[1, 2, 2], [3, 4], Int[]],
        y = [0.2, 0.4, 0.0])
    bound = _CS_READS_SUM(; data.id, data.grp, data.rows) | (; data.y)
    built = build_kernel(bound)
    u = [0.2, -0.1, 0.3, 0.4, 0.1, -0.2, 0.25]
    f = prepare_cell_factors(built, bound, :y; partitioned = :z, u0 = u, locality = :ignore)
    # The shape under test: the outer reads `w` at rows, the prefix only sigma.
    @test f.shared.cache.positions == [1]
    term = prepare_cell_factor_query(f, 1)
    outer = prepare_cell_factor_query(f, :outer)
    rate(point) = exp.(point[4:7])
    total(point, i) = sum(rate(point)[data.rows[i]]; init = 0.0)
    location(point, s, i) = log1p(s) + point[1 + data.grp[i]]
    outer_density(point, s, i) = logpdf(Normal(location(point, s, i), exp(point[1])), data.y[i])
    function check(point)
        for i in 1:3
            s = total(point, i)
            @test outer(point, [s], i) ≈ outer_density(point, s, i)
            for (t, row) in enumerate(data.rows[i])
                @test term(point, i, t) ≈ rate(point)[row]
            end
        end
    end
    check(u); check(u)
    before = f.shared.generation
    residual_only = copy(u); residual_only[2] += 0.25
    check(residual_only)
    @test f.shared.generation == before
    read = copy(u); read[1] += 0.25
    check(read)
    @test f.shared.generation == before + 1
    s = total(u, 1)
    with_cell_batch(f, u) do batch
        @test outer(batch, [s], 1) ≈ outer_density(u, s, 1)
        @test outer(residual_only, [s], 1) ≈ outer_density(residual_only, s, 1)
        @test outer(batch, [s], 1) ≈ outer_density(u, s, 1)
    end

    q = prepare_cell_factor_ad(prepare_ad_third, f, 1, u, [4], _CS_THIRD, ([0.6],))
    for (i, t, row) in ((1, 1, 1), (1, 1, 1), (2, 1, 3))
        g, (hv,), (tv,) = ad_gradient_hvp_and_third(q, ([0.6],), u, u[[row + 3]], [row + 3], i, t)
        @test only(g) ≈ rate(u)[row]
        @test only(hv) ≈ rate(u)[row] * 0.6
        @test only(tv) ≈ rate(u)[row] * 0.6^2
    end
    a = prepare_cell_accumulator(q, _CS_THIRD, u; order = 2)
    p = prepare_cell_pullback(f, _CS_REVERSE, u)
    gradient = zeros(length(u))
    cell_pullback!(gradient, p, u) do batch
        for (t, row) in enumerate(data.rows[1])
            accumulate_cell!(batch, a, [0.6], 1.0, u[[row + 3]], [row + 3], 1, t)
        end
    end
    @test gradient[4:7] ≈ [1, 2, 0, 0] .* rate(u) .* 0.6^2
    @test all(iszero, gradient[1:3])

    # Sum values are independent: this is the mixed derivative at a fixed `s`, so
    # sigma comes through the prefix's pullback and `w` from the packed base.
    directions = (([0.6], Float64[]),)
    qouter = prepare_cell_factor_ad(prepare_ad_third, f, :outer, u, Int[], _CS_THIRD, directions)
    gouter, _, _ = ad_gradient_hvp_and_third(qouter, directions, u, [s], Float64[], Int[], 1)
    @test only(first(gouter)) ≈ (data.y[1] - location(u, s, 1)) / exp(2u[1]) / (1 + s)
    curvature(point) = 0.6^2 * -(1 + data.y[1] - location(point, s, 1)) / (exp(2point[1]) * (1 + s)^2)
    aouter = prepare_cell_accumulator(qouter, _CS_THIRD, u; order = 2)
    cell_pullback!(gradient, p, u) do batch
        accumulate_cell!(batch, aouter, only(directions), 1.0, [s], Float64[], Int[], 1)
    end
    epsilon = 1e-5
    fd = map(eachindex(u)) do j
        hi, lo = copy(u), copy(u)
        hi[j] += epsilon; lo[j] -= epsilon
        (curvature(hi) - curvature(lo)) / (2epsilon)
    end
    @test gradient ≈ fd rtol = 2e-8 atol = 2e-8
end

@testset "A shared NamedTuple value with scalar fields" begin
    n = 4
    data = (; id = collect(1:n), grid = collect(range(-0.7, 0.9; length = n)),
        y = mod.(1:n, 4))
    bound = _CS_NAMED(; data.id, data.grid) | (; data.y)
    built = build_kernel(bound)
    u = collect(range(-0.2, 0.3; length = built.layout.total))
    fd(f, point) = map(eachindex(point)) do j
        hi, lo = copy(point), copy(point)
        hi[j] += 1e-5; lo[j] -= 1e-5
        (f(hi) - f(lo)) / 2e-5
    end
    q = prepare_compact_cell_ad(prepare_ad_third, built, bound, :y, u, [2],
        _CS_THIRD, ([0.6],); partitioned = :z, locality = :error)
    @test only(q.shared.cache.values) isa NamedTuple{(:scale, :weights, :offset)}
    p = prepare_cell_pullback(q, _CS_REVERSE, u)
    reference = prepare_cell_query(built, bound, :y; locality = :ignore)
    weights = [0.3, -0.4, 1.2, 0.8]
    for (order, backend) in ((0, _CS_REVERSE), (1, _CS_SECOND), (2, _CS_THIRD))
        a = prepare_cell_accumulator(q, backend, u; order)
        gradient = zeros(length(u))
        cell_pullback!(gradient, p, u) do batch
            for i in 1:n
                accumulate_cell!(batch, a, order == 0 ? nothing : [0.6],
                    weights[i], u[[i+1]], [i+1], i)
            end
        end
        contraction(point) = sum(1:n) do i
            order == 0 && return weights[i] * reference(point, i)
            g, (hv,), _ = ad_gradient_hvp_and_third(q, ([0.6],),
                point, point[[i+1]], [i+1], i)
            weights[i] * 0.6 * only(order == 1 ? g : hv)
        end
        @test gradient ≈ fd(contraction, u) rtol = 1e-7 atol = 1e-8
        @test gradient[1] != 0
    end

    partition = prepare_partition(built, bound, :y; by = :id, locality = :ignore)
    for compact in (false, true)
        sampler = prepare_partition_sampler(built, bound, :y, u; by = :id,
            backend = _CS_REVERSE, compact, locality = :error, partitioned = :z)
        @test sampler.prefix !== nothing
        for label in 1:n
            footprint = partition_footprint(sampler, label)
            g = zeros(compact ? length(footprint) : length(u))
            partition_value_and_gradient!(sampler, g, u, label)
            expected = fd(point -> partition(point, label), u)
            @test g ≈ (compact ? expected[footprint] : expected) rtol = 1e-7 atol = 1e-8
        end
        g = zeros(length(u))
        partition_batch_value_and_gradient!(sampler, g, u, collect(1:n))
        @test g ≈ fd(point -> sum(label -> partition(point, label), 1:n), u) rtol = 1e-7 atol = 1e-8
    end
end

@testset "A cell that reads more shared values than the splat limit" begin
    # Its compact Hessian-vector products raised `EnzymeRuntimeActivityError`
    # under both compositions, as BRM's sparse Laplace runs them for NLME
    # `check_protocol` (snag varyingsource3-b-82246291).
    n, K = 4, _CS_MANY_K
    data = (; id = collect(1:n), grid = collect(range(-0.7, 0.9; length = n)),
        y = [0.1, -0.3, 0.5, 0.2])
    bound = _CS_MANY(; data.id, data.grid) | (; data.y)
    built = build_kernel(bound)
    u = collect(range(-0.2, 0.3; length = built.layout.total))
    cell_mean(point, i) = (point[i+1] + sum(point[1] + j for j in 1:K) * data.grid[i]) / K
    fd(f, point) = map(eachindex(point)) do j
        hi, lo = copy(point), copy(point)
        hi[j] += 1e-5; lo[j] -= 1e-5
        (f(hi) - f(lo)) / 2e-5
    end
    for second in (_CS_SECOND, SecondOrder(AutoEnzyme(; mode = Enzyme.Forward), _CS_REVERSE))
        q = prepare_compact_cell_ad(prepare_ad_hvp, built, bound, :y, u, [2],
            second, ([0.6],); partitioned = :z, locality = :error)
        @test length(q.shared.cache.values) == K
        for i in 1:n
            g, (hv,) = ad_gradient_and_hvp(q, ([0.6],), u, u[[i+1]], [i+1], i)
            @test only(g) ≈ (data.y[i] - cell_mean(u, i)) / K
            @test only(hv) ≈ -0.6 / K^2
        end
    end

    q = prepare_compact_cell_ad(prepare_ad_third, built, bound, :y, u, [2],
        _CS_THIRD, ([0.6],); partitioned = :z, locality = :error)
    p = prepare_cell_pullback(q, _CS_REVERSE, u)
    weights = [0.3, -0.4, 1.2, 0.8]
    for (order, backend) in ((0, _CS_REVERSE), (1, _CS_SECOND), (2, _CS_THIRD))
        a = prepare_cell_accumulator(q, backend, u; order)
        gradient = zeros(length(u))
        cell_pullback!(gradient, p, u) do batch
            for i in 1:n
                accumulate_cell!(batch, a, order == 0 ? nothing : [0.6],
                    weights[i], u[[i+1]], [i+1], i)
            end
        end
        # The weighted cell value, first or second directional derivative
        # along z_i with 0.6.
        contraction(point) = sum(1:n) do i
            order == 0 && return weights[i] * logpdf(Normal(cell_mean(point, i), 1.0), data.y[i])
            weights[i] * (order == 1 ? 0.6 * (data.y[i] - cell_mean(point, i)) / K : -0.36 / K^2)
        end
        @test gradient ≈ fd(contraction, u) rtol = 1e-7 atol = 1e-8
    end
end
