using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme, SecondOrder

module _CFADomains
using ReactiveKernels
@kernel live(x::AbstractVector{Float64}, y::AbstractVector{Float64}) = begin
    cells = plate(x, y) do xi, yi
        -(xi - yi)^2 / 2
    end
    return cells
end
@kernel scalarcheck(a::Float64, b::Float64, y::AbstractVector{Float64}) = begin
    intermediate = plate(a, b) do ai, bi
        ai + bi
    end
    cells = plate(intermediate, y) do m, yi
        m - yi
    end
    return cells
end
end

@testset "factor domains without bound operands" begin
    spec = plate_cell(_CFADomains.live, :cells; index=:i, locality=:error)
    f = ReactiveKernels._cell_factor_graph(spec, :cells_cell; have=(:x,:y,:i), locality=:error)
    q = prepare(f.spec; have=(:x,:y,:i), want=:cells_cell)
    fp = partition_footprint_kernel(q)
    for n in (1, 4, 32)
        x, y = sin.(1:n), cos.(1:n)
        for i in (1, n)
            @test q(x,y,i) ≈ -(x[i]-y[i])^2/2
            @test q(x,[0.4],i) ≈ -(x[i]-0.4)^2/2
            @test Base.invokelatest(fp,x,y,i).x == [i]
        end
        @test_throws BoundsError q(x,y,n+1)
    end
    @test_throws DimensionMismatch q(ones(2),ones(3),1)
    @test_throws BoundsError q(Float64[],Float64[],1)

    # A later array domain must not make an authored scalar-only plate valid.
    invalid = plate_cell(_CFADomains.scalarcheck, :cells; index=:i)
    original = prepare(invalid; have=(:a,:b,:y,:i), want=:cells_cell, bound=(;y=ones(3)))
    fbad = ReactiveKernels._cell_factor_graph(invalid, :cells_cell;
        have=(:a,:b,:y,:i), bound=(;y=ones(3)))
    factored = prepare(fbad.spec; have=(:a,:b,:y,:i), want=:cells_cell, bound=(;y=ones(3)))
    for query in (original, factored)
        err = try
            query(0.2,0.3,1)
        catch e
            e
        end
        @test err isa MethodError && err.f === Base.Broadcast.combine_axes && isempty(err.args)
    end
end

const _CFA_GAUSSIAN = @rkppl begin
    offset ~ Normal(0, 1)
    field[1:n] .~ Normal.(prior_mean, prior_scale)
    prediction = offset .+ field[idx]
    y .~ Normal.(prediction, 0.6)
end

# An array-valued scale and a two-axis random-effect block retain the shape
# used by generated random-intercept predictors. The shared helper stays
# parameter-dependent, so mixed derivatives must traverse the prefix.
_cfa_field(grid, beta) = map(t -> exp(beta * t), grid)
const _CFA_INTERCEPT = @rkppl begin
    offset ~ Normal(0, 1)
    beta ~ Normal(0, 1)
    scale[1:1] .~ Exponential.(1)
    z[1:n, 1:1] .~ Normal.(0, 1)
    effects = z .* scale[1]
    eta = offset .+ effects[idx, 1]
    field = _cfa_field(grid, beta)
    y .~ Poisson.(exp.(eta) .* field)
end

const _CFA_REVERSE = AutoEnzyme(; mode = Enzyme.Reverse)
const _CFA_SECOND = SecondOrder(_CFA_REVERSE, AutoEnzyme(; mode = Enzyme.Forward))
const _CFA_THIRD = SecondOrder(_CFA_SECOND, AutoEnzyme(; mode = Enzyme.Forward))

@testset "cached factor projections retain broadcast domains" begin
    for n in (1, 4, 32)
        idx = [1, n, 1]
        y = [0.2, -0.4, 0.8]
        bound = _CFA_GAUSSIAN(; n, idx, prior_mean=zeros(n), prior_scale=ones(n)) | (; y)
        built = build_kernel(bound)
        u = collect(range(-0.2, 0.3; length=built.layout.total))
        names = coordinate_names(built.layout)
        offset = findfirst(==(:offset), names)
        positions = [findfirst(==(Symbol("field.$j")), names) for j in 1:n]
        f = prepare_cell_factors(built, bound, :y;
            partitioned=(:field => collect(1:n),), u0=u, locality=:error)
        @test isempty(cell_reductions(f))
        q = prepare_cell_factor_query(f, :outer)
        fp = partition_footprint_kernel(q.kernel)
        op = prepare_cell_factor_ad(prepare_ad_third, f, :outer, u, [positions[1]],
            _CFA_THIRD, ((Float64[], [0.7]),))
        pullback = prepare_cell_pullback(f, _CFA_REVERSE, u)
        accumulator = prepare_cell_accumulator(op, _CFA_SECOND, u; order=1)
        for shift in (-0.2, 0.4)
            u[offset] = shift
            for i in eachindex(y)
                j = positions[idx[i]]
                mu = shift + u[j]
                @test q(u, Float64[], i) ≈ logpdf(Normal(mu, 0.6), y[i])
                @test Base.invokelatest(fp, u, Float64[], i).unconstrained == sort([offset, j])
                g, (hv,), (tv,) = ad_gradient_hvp_and_third(op,
                    ((Float64[], [0.7]),), u, Float64[], u[[j]], [j], i)
                @test isempty(g[1])
                @test g[2] ≈ [(y[i] - mu) / 0.6^2]
                @test hv[2] ≈ [-0.7 / 0.6^2]
                @test tv[2] ≈ [0.0] atol=1e-12
            end
            gradient = similar(u)
            cell_pullback!(gradient, pullback, u) do batch
                for i in eachindex(y)
                    j = positions[idx[i]]
                    accumulate_cell!(batch, accumulator, (Float64[], [0.7]), 1.0,
                        Float64[], u[[j]], [j], i)
                end
            end
            expected = zeros(length(u))
            expected[offset] = -length(y) * 0.7 / 0.6^2
            for j in idx
                expected[positions[j]] -= 0.7 / 0.6^2
            end
            @test gradient ≈ expected
        end
    end
end

@testset "cached generated-shape random intercept factors" begin
    n = 4
    idx = [1, 3, 1, 4]
    grid = [-0.4, 0.2, 0.6, 0.9]
    y = [1, 0, 2, 3]
    bound = _CFA_INTERCEPT(; n, idx, grid) | (; y)
    built = build_kernel(bound)
    names = coordinate_names(built.layout)
    u = collect(range(-0.2, 0.3; length=built.layout.total))
    offset = findfirst(==(:offset), names)
    beta = findfirst(==(:beta), names)
    scale = findfirst(==(Symbol("scale.1")), names)
    positions = [findfirst(==(Symbol("z.$j.1")), names) for j in 1:n]
    f = prepare_cell_factors(built, bound, :y;
        partitioned=(:z => collect(1:n),), u0=u, locality=:error)
    q = prepare_cell_factor_query(f, :outer)
    op = prepare_cell_factor_ad(prepare_ad_third, f, :outer, u, [positions[1]],
        _CFA_THIRD, ((Float64[], [0.7]),))
    pullback = prepare_cell_pullback(f, _CFA_REVERSE, u)
    accumulator = prepare_cell_accumulator(op, _CFA_REVERSE, u; order=0)
    for shift in (-0.2, 0.4)
        u[beta] = shift
        s = exp(u[scale])
        expected = zeros(length(u))
        for i in eachindex(y)
            j = positions[idx[i]]
            rate = exp(u[offset] + s*u[j] + shift*grid[i])
            @test q(u, Float64[], i) ≈ logpdf(Poisson(rate), y[i])
            g, (hv,), (tv,) = ad_gradient_hvp_and_third(op,
                ((Float64[], [0.7]),), u, Float64[], u[[j]], [j], i)
            @test g[2] ≈ [s*(y[i]-rate)]
            @test hv[2] ≈ [-s^2*rate*0.7]
            @test tv[2] ≈ [-s^3*rate*0.7^2]
            expected[offset] += y[i]-rate
            expected[beta] += grid[i]*(y[i]-rate)
            expected[scale] += s*u[j]*(y[i]-rate)
            expected[j] += s*(y[i]-rate)
        end
        gradient = similar(u)
        cell_pullback!(gradient, pullback, u) do batch
            for i in eachindex(y)
                j = positions[idx[i]]
                accumulate_cell!(batch, accumulator, nothing, 1.0,
                    Float64[], u[[j]], [j], i)
            end
        end
        @test gradient ≈ expected
    end
end
