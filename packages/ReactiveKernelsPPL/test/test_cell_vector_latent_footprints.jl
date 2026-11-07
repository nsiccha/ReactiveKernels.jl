module CellVectorLatentFootprintTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions
using DifferentiationInterface: AutoEnzyme
import Enzyme

# A dotted cell latent `x[i] .~ D.(args…)` packs one block of coordinates per
# index. The cell reads its own block through a view of the packed vector,
# indexed by a bound table of unit ranges, so a strict cell query's footprint,
# a partition's footprint and the cell's gradient read that block only, not
# every index's coordinates (snag `rkppl-dotted-pla-1629a747`).

function central(f, v)
    map(eachindex(v)) do j
        plus, minus = copy(v), copy(v)
        plus[j] += 1e-6
        minus[j] -= 1e-6
        (f(plus) - f(minus)) / 2e-6
    end
end

const RESPONSE = quote
    sigma ~ Exponential(1)
    s ~ Exponential(1)
    @plate for i in eachindex(y)
        x[i] .~ Normal.(loc[i], s)
        x_cell = x[i]
        y[i] .~ Normal.(mu[i] .+ x_cell, sigma)
    end
end
const POSITIVE = quote
    sigma ~ Exponential(1)
    s ~ Exponential(1)
    @plate for i in eachindex(y)
        x[i] .~ LogNormal.(loc[i], s)
        y[i] .~ Normal.(mu[i] .+ x[i], sigma)
    end
end

function fixture(ast, loc; transform = identity)
    n = length(loc)
    data = (; loc, mu = [0.1i for i in 1:n], g = collect(1:n),
        y = [0.3 .* sin.(1:length(l)) .+ 0.1i for (i, l) in enumerate(loc)])
    bound = bind_data(lower_rkppl(ast, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(bound)
    u = [0.05 + 0.07k for k in 1:built.layout.total]
    names = coordinate_names(built.layout)
    offsets = cumsum([2; length.(loc)])
    # The cell's observation: sigma (coordinate 1) and its own block.
    observation(v, i) = (block = transform.(v[offsets[i] + 1:offsets[i + 1]]);
        sum(logpdf.(Normal.(data.mu[i] .+ block, exp(v[1])), data.y[i]); init = 0.0))
    (; data, bound, built, u, names, offsets, observation)
end

block(fx, i) = [1; collect(fx.offsets[i] + 1:fx.offsets[i + 1])]

@testset "a dotted cell latent's cell reads only its own block" begin
    for (ast, transform) in ((RESPONSE, identity), (POSITIVE, exp)),
            loc in ([[0.1, -0.2], [0.3], [0.0, 0.4, -0.1], [0.2, 0.2]],
                    [[0.1, -0.2], Float64[], [0.3]])
        fx = fixture(ast, loc; transform)
        saved = deepcopy((fx.data, fx.u))
        cell = prepare_cell_query(fx.built, fx.bound, :y; locality = :error)
        footprint = partition_footprint_kernel(cell.kernel)
        partition = prepare_partition(fx.built, fx.bound, :y; by = :y => :g,
            locality = :error)
        ad = prepare_ad(cell.kernel, AutoEnzyme(; mode = Enzyme.Reverse), fx.u, 1;
            active = :unconstrained)
        for i in eachindex(loc)
            read = Base.invokelatest(footprint, fx.u, i).unconstrained
            parted = partition_footprint(partition.kernel, fx.u, i, nothing).unconstrained
            if ast === RESPONSE
                @test read == block(fx, i)
                @test parted == block(fx, i)
            else
                # A transformed latent's cell reads a view of the constrained
                # vector, which the cell-window pushdown does not compute at
                # the view's rows yet: the footprint is the whole latent, as
                # for a top-level `view(x, r[i])` of a LogNormal vector.
                @test issubset(block(fx, i), read)
                @test_broken read == block(fx, i)
                @test parted == read
            end
            expected = fx.observation(fx.u, i)
            @test Base.invokelatest(cell, fx.u, i) ≈ expected
            @test partition(fx.u, i) ≈ expected
            value, gradient = ad_value_and_gradient(ad, fx.u, i)
            @test value ≈ expected
            @test gradient ≈ central(v -> fx.observation(v, i), fx.u) atol = 1e-8 rtol = 1e-6
            @test issubset(findall(!iszero, gradient), read)
        end
        @test (fx.data, fx.u) == saved
    end
end

@testset "matrix and number blocks read their own coordinates" begin
    # A matrix per index keeps its dimensions; its cell still reads its block.
    A = [[1.0 2.0; 3.0 4.0], [5.0 6.0 7.0]]
    data = (; A, y = [1.0, 2.0])
    bound = bind_data(lower_rkppl(quote
            @plate for i in eachindex(A)
                M[i] .~ Normal.(A[i], 1)
                y[i] ~ Normal(sum(M[i]), 1)
            end
        end, data; conditioned = (:y,)), data)
    built = build_kernel(bound)
    u = collect(0.1:0.1:0.7)
    cell = prepare_cell_query(built, bound, :y; locality = :error)
    footprint = partition_footprint_kernel(cell.kernel)
    @test Base.invokelatest(footprint, u, 1).unconstrained == 1:4
    @test Base.invokelatest(footprint, u, 2).unconstrained == 5:7
    @test Base.invokelatest(cell, u, 1) ≈ logpdf(Normal(sum(u[1:4]), 1), 1.0)
    @test Base.invokelatest(cell, u, 2) ≈ logpdf(Normal(sum(u[5:7]), 1), 2.0)
    # Lowered from data names alone, numbers at every index are one
    # coordinate per index, read as a number.
    m, ys = [0.5, -1.0, 2.0], [0.2, 0.1, -0.3]
    bound = bind_data(lower_rkppl(quote
            @plate for i in eachindex(m)
                th[i] .~ Normal.(m[i], 1)
                ys[i] ~ Normal(th[i], 1)
            end
        end, (:m, :ys); conditioned = (:ys,)), (; m, ys))
    built = build_kernel(bound)
    u = [0.1, 0.2, 0.3]
    cell = prepare_cell_query(built, bound, :ys; locality = :error)
    footprint = partition_footprint_kernel(cell.kernel)
    for i in 1:3
        @test Base.invokelatest(footprint, u, i).unconstrained == [i]
        @test Base.invokelatest(cell, u, i) ≈ logpdf(Normal(u[i], 1), ys[i])
    end
end

astsize(x::Expr) = 1 + sum(astsize, x.args; init = 0)
astsize(x) = 0

@testset "a dotted cell latent's footprint program does not grow with the data" begin
    sizes = Int[]
    for (groups, len) in ((3, 2), (300, 2), (3, 400))
        fx = fixture(RESPONSE, [fill(0.1, len) for _ in 1:groups])
        cell = prepare_cell_query(fx.built, fx.bound, :y; locality = :error)
        footprint = partition_footprint_kernel(cell.kernel)
        @test Base.invokelatest(footprint, fx.u, groups).unconstrained == block(fx, groups)
        push!(sizes, astsize(code_expr(cell.kernel)) + astsize(code_expr(footprint)))
    end
    @test allequal(sizes)
end

end
