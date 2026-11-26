using Test, Random, Distributions, ReactiveKernels, ReactiveKernelsPPL
using LinearAlgebra: istril
import Enzyme, DifferentiationInterface as DI

module DrawFixtures
using ReactiveKernelsPPL
using Random
struct Shift{T}
    lower::T
end
ReactiveKernelsPPL.sampling_draw(rng, d::Shift) = d.lower + randexp(rng)
ReactiveKernelsPPL.sampling_logdensity(d::Shift, x) = x > d.lower ? d.lower - x : -Inf
ReactiveKernelsPPL.sampling_geometry(::Type{Shift}, args, shape) =
    ParameterGeometry(shape, 1; support=:lower_bounded,
        constrain=(u, shape, lo) -> lo + exp(only(u)),
        unconstrain=(v, shape, lo) -> [log(v-lo)],
        logjac=(u, shape, lo) -> only(u))

struct PairEvent{T}
    center::T
end
function ReactiveKernelsPPL.sampling_draw(rng, d::PairEvent)
    x = d.center + randn(rng)
    return [x, -x]
end

foreign(mu) = error("fragment must expand structurally")
const fragment = RKPPLSubmodel(:foreign, [:mu], quote
    b ~ Shift(mu)
    return b
end, @__MODULE__)
ReactiveKernelsPPL.sampling_fragment(::typeof(foreign)) = fragment
@rkppl wrapped(mu) = begin
    inner ~ foreign(mu)
    return inner
end

center(x) = 2x
positions(xs) = eachindex(xs)
@rkppl readings(mu, xs) = begin
    delta ~ Shift(offset)
    offset ~ Normal(center(mu), 1)
    @plate for j in positions(xs)
        reading[j] = xs[j] * delta + offset
    end
    return reading
end
end

@testset "joint prior uses a single realization of each ancestral site" begin
    # The dependent site is deliberately declared first.
    m = @rkppl begin
        b ~ DrawFixtures.Shift(a)
        a ~ Normal(mu, 1)
        v[1:k] .~ Normal.(b, 0.5)
        s ~ Dirichlet(alpha)
        L ~ LKJCholesky(3, 1.5)
        readout = design .* b
    end
    fixed = (; mu=0.4, k=3, alpha=[1.0,2.0,3.0])
    target(x) = bind_data(m(; fixed..., design=x))
    plan = target([1.,2.])
    q = prepare_prior(plan)
    rng = Xoshiro(29)
    draws = q(rng, 3)
    @test draws == q(rng, 3)
    @test draws[1] == q(rng)
    @test draws != q(Xoshiro(30), 3)
    @test isempty(q(rng, 0))
    @test rand(copy(rng)) == rand(Xoshiro(29))
    r = Xoshiro(29)
    a = rand(r, Normal(fixed.mu,1))
    @test draws[1].a == a
    @test draws[1].b == a + randexp(r)
    @test draws[1].v == [rand(r, Normal(draws[1].b,0.5)) for _ in 1:3]
    @test all(d -> sum(d.s) ≈ 1 && all(>(0),d.s) && istril(d.L), draws)
    function replay(d, x)
        p = target(x)
        built = build_kernel(p)
        u = unconstrain(built.layout, d)
        @test constrain(built.layout, u).b ≈ d.b
        return Base.invokelatest(prepare_query(built, p, (:readout,)), u)
    end
    d = first(draws)
    designs = [[-1.,0.5], [4.,3.,2.]]
    forward = [replay(d,x) for x in designs]
    @test forward == reverse([replay(d,x) for x in reverse(designs)])
    @test all(forward .≈ [x .* d.b for x in designs])
    @test fixed.alpha == [1.,2.,3.]

    nested = @rkppl begin
        branch ~ DrawFixtures.wrapped(location)
    end
    scoped = prepare_prior(nested(; location=2.0))(rng)
    @test scoped.branch.inner.b == 2.0 + randexp(copy(rng))
end

@testset "generated scopes and their random inputs are outside the sampler" begin
    m = @rkppl begin
        a ~ Normal(0,1)
        @generate begin
            b ~ DrawFixtures.wrapped(a)
            local_b = b.inner.b
            z[1:n] .~ DrawFixtures.Shift.(b .+ offset)
            pair[1:2] ~ DrawFixtures.PairEvent(a)
        end
    end
    u = [0.45]
    rng = Xoshiro(7)
    baseline = bind_data((@rkppl begin a ~ Normal(0,1) end)())
    names = coordinate_names(assign_layout(baseline))
    sources = Expr[]
    for n in (0, 3, 11)
        offset = collect(1.:n)
        p = bind_data(m(; n, offset))
        @test assign_layout(p).total == 1
        @test coordinate_names(assign_layout(p)) == names
        q = prepare_generated(p)
        draw = q(u, rng)
        @test draw == q((; a=only(u)), Xoshiro(7))
        @test draw != q(u, Xoshiro(8))
        @test draw.b == draw.local_b > only(u)
        @test size(draw.z) == (n,)
        @test all(draw.z .> draw.b .+ offset)
        @test draw.pair[1] == -draw.pair[2]
        @test offset == collect(1.:n) && u == [0.45] && rand(copy(rng)) == rand(Xoshiro(7))
        ex = Base.remove_linenums!(deepcopy(code_expr(q)))
        push!(sources, ex)
        replay = Core.eval(@__MODULE__, Meta.parse(string(ex)))
        @test Base.invokelatest(replay, copy(rng), q.inputs, (; a=only(u))) == draw
    end
    @test all(==(first(sources)), sources)
    only_gq = @rkppl begin
        @generate x ~ DrawFixtures.PairEvent(1.0)
    end
    @test assign_layout(bind_data(only_gq())).total == 0
    @test length(prepare_generated(only_gq())(Float64[], rng).x) == 2
end

@testset "generated plate submodels return values and share their local draws" begin
    model = @rkppl begin
        theta ~ Normal(0, 1)
        y ~ Normal(theta, 1)
        @generate begin
            @plate for i in eachindex(groups)
                value[i] ~ DrawFixtures.readings(theta + groups[i], xs[i])
                same[i] = value[i].offset
            end
            offsets = value.offset
            deltas = value.delta
        end
    end
    baseline = (@rkppl begin
        theta ~ Normal(0, 1)
        y ~ Normal(theta, 1)
    end)() | (; y=0.7)
    sources = Tuple[]
    for xs in (Vector{Float64}[], [[1., 2.], Float64[]], [[0.], [-1., 4., 2.], [8.]])
        groups = collect(1.:length(xs))
        p = model(; groups, xs) | (; y=0.7)
        @test coordinate_names(assign_layout(p)) == coordinate_names(assign_layout(baseline))
        q = prepare_generated(p)
        cell_sources = map(cell -> ReactiveKernels._program_expression(cell.native), q.inputs.cells)
        source = map(ex -> Base.remove_linenums!(deepcopy(ex)), (code_expr(q), cell_sources...))
        push!(sources, source)
        @test length(cell_sources) == 2
        replay, cells... = map(ex -> Core.eval(@__MODULE__, Meta.parse(string(ex))), source)
        bindings = (; data=q.inputs.data, cells=Tuple(cells))
        @test isconcretetype(only(Base.return_types(q.program,
            Tuple{Xoshiro,typeof(q.inputs),@NamedTuple{theta::Float64}})))
        for u in ([1.25], [-0.5]), seed in (17, 18)
            rng = Xoshiro(seed)
            result = q(u, rng)
            @test keys(result) == (:value, :same, :offsets, :deltas)
            @test result == q((; theta=only(u)), rng)
            @test result == Base.invokelatest(replay, copy(rng), bindings, (; theta=only(u)))
            oracle = copy(rng)
            for i in eachindex(groups)
                a = rand(oracle, Normal(2 * (only(u) + groups[i]), 1))
                b = a + randexp(oracle)
                @test result.offsets[i] == result.same[i] == a
                @test result.deltas[i] == b
                @test result.value[i] == xs[i] .* b .+ a
            end
            @test length(result.value) == length(result.offsets) == length(groups)
            @test eltype(result.offsets) === Float64
            @test eltype(result.value) === Vector{Float64}
            @test rand(rng) == rand(Xoshiro(seed))
        end
    end
    @test all(==(first(sources)), sources)

    # Retain both physical draws from an empty reading schedule and use
    # them on either design without revisiting the stochastic query.
    groups = [1., 3.]
    generate(xs) = prepare_generated(model(; groups, xs) | (; y=0.7))([1.25], Xoshiro(17))
    retained = generate([Float64[], Float64[]])
    readout(xs) = [row .* b .+ a for (row, a, b) in zip(xs, retained.offsets, retained.deltas)]
    designs = ([[1., 3.], Float64[]], [Float64[], [-1., 4., 0.]])
    for design in reverse(designs)
        d = generate(design)
        @test (d.offsets, d.deltas) == (retained.offsets, retained.deltas)
        @test d.value == readout(design)
    end

    # A direct return can share its identifier with the local that supplied it.
    direct = @rkppl begin
        theta ~ Normal(0, 1)
        @generate begin
            @plate for i in eachindex(groups)
                result[i] ~ DrawFixtures.foreign(theta + groups[i])
            end
            local_b = result.b
        end
    end
    groups = [1., 3.]
    q = prepare_generated(direct(; groups))
    oracle = Xoshiro(17)
    expected = [1.25 + g + randexp(oracle) for g in groups]
    @test q([1.25], Xoshiro(17)) == (; result=expected, local_b=expected)

    lexical = @rkppl begin
        theta ~ Normal(0, 1)
        @plate for i in eachindex(y)
            y[i] ~ Normal(theta, 1)
        end
        @generate begin
            shift = theta + 1
            @plate for i in eachindex(groups)
                shift = groups[i] + theta
                adjusted[i] = 2 * shift
            end
            after = shift
        end
    end
    q = prepare_generated(lexical(; groups) | (; y=[0.7]))
    @test q([1.25], Xoshiro(17)) ==
        (; shift=2.25, adjusted=2 .* (groups .+ 1.25), after=2.25)
end

@testset "structured and conditioned draws" begin
    constrained = @rkppl begin
        cuts ~ Ordered(Normal(0, 1), 3)
        bounded ~ truncated(Normal(0, 1), -1, 2)
    end
    draw = prepare_prior(constrained())(Xoshiro(8))
    @test length(draw.cuts) == 3 && issorted(draw.cuts)
    @test -1 <= draw.bounded <= 2
    m = @rkppl begin
        L ~ LKJCholesky(2, 1.5)
        scales[1:2] .~ HalfNormal.(scale)
        F = scales .* L
        eachrow(B[levels(group), 1:2]) .~ MvNormalCholesky(zeros(2), F)
    end
    for groups in ([2,1,1,3], [4,4])
        fit = m(; group=groups, scale=0.7)
        q = prepare_prior(fit)
        d = q(Xoshiro(9))
        @test size(d.B) == (length(unique(groups)),2)
        @test all(>(0), d.scales)
        p = bind_data(fit)
        @test constrain(assign_layout(p), unconstrain(assign_layout(p), d)).B ≈ d.B
        source = Meta.parse(string(code_expr(q)))
        replay = Core.eval(@__MODULE__, source)
        @test Base.invokelatest(replay, Xoshiro(9), q.inputs, (;)) == d
    end
    empty_model = @rkppl begin
        x[1:n] .~ Normal.(0, 1)
        @generate eachrow(B[1:n,1:2]) .~ DrawFixtures.PairEvent(0.)
    end
    empty_fit = empty_model(; n=0)
    empty_draw = prepare_prior(empty_fit)(Xoshiro(9))
    @test size(empty_draw.x) == (0,)
    @test size(prepare_generated(empty_fit)(empty_draw, Xoshiro(9)).B) == (0,2)
    m = @rkppl begin
        fixed ~ Normal(0,1)
        child ~ DrawFixtures.Shift(fixed)
        @generate replica ~ DrawFixtures.Shift(child)
    end
    fit = m() | (; fixed=1.8)
    seed = Xoshiro(2)
    d = prepare_prior(fit)(seed)
    @test keys(d) == (:child,)
    @test d.child == 1.8 + randexp(copy(seed))
    @test prepare_generated(fit)(d,seed).replica == d.child + randexp(copy(seed))
end

@testset "generation eligibility" begin
    density_only = @rkppl begin
        @generate y ~ LogDensity(abs)
    end
    @test_throws "sampling_draw" prepare_generated(density_only())(Float64[], Xoshiro(2))
    flat = @rkppl begin x ~ Flat() end
    @test_throws "normalized conditional" prepare_prior(flat())
    wrong_shape = @rkppl begin
        @generate x[1:3] ~ DrawFixtures.PairEvent(1.)
    end
    @test_throws DimensionMismatch prepare_generated(wrong_shape())(Float64[], Xoshiro(2))
    duplicate = @rkppl begin
        x ~ Normal()
        @generate x ~ Normal()
    end
    @test_throws "redefines" bind_data(duplicate())
    leak = @rkppl begin
        @generate x ~ Normal()
        y ~ Normal(x,1)
    end
    @test_throws "reads generated" bind_data(leak())
end

@testset "ordinary native sampler gradient excludes generated sites" begin
    u = [0.45]
    baseline = bind_data((@rkppl begin a ~ Normal(0,1) end)())
    function vg(plan)
        sampler = prepare_sampler(build_kernel(plan), plan, u;
            backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
        return sampler_value_and_gradient!(sampler, zeros(length(u)), u)
    end
    density = vg(baseline)
    @test density[2] ≈ -u
    model = @rkppl begin
        a ~ Normal(0,1)
        @generate replica[1:n] .~ Normal.(a,1)
    end
    for n in (0, 3, 11)
        p = bind_data(model(; n))
        g = prepare_generated(p)
        g(u, Xoshiro(1))
        g(u, Xoshiro(2))
        @test vg(p) == density
    end

    repeated = @rkppl begin
        a ~ Normal(0,1)
        y ~ Normal(a,1)
        @generate begin
            @plate for i in eachindex(xs)
                value[i] ~ DrawFixtures.readings(a, xs[i])
            end
            delta = value.delta
        end
    end
    observed = @rkppl begin
        a ~ Normal(0,1)
        y ~ Normal(a,1)
    end
    reference = vg(observed() | (; y=0.7))
    @test reference[1] ≈ logpdf(Normal(), only(u)) + logpdf(Normal(only(u), 1), 0.7)
    @test reference[2] ≈ [0.7 - 2 * only(u)]
    for xs in (Vector{Float64}[], [[2.0], Float64[]])
        p = repeated(; xs) | (; y=0.7)
        prepare_generated(p)(u, Xoshiro(17))
        @test vg(p) == reference
    end
end
