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
end
