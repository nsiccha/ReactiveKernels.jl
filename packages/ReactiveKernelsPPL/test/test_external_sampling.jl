using Test, ReactiveKernels, ReactiveKernelsPPL
using ReactiveKernelsDistributionKernels.DistributionKernelSources
import DifferentiationInterface as DI
import Enzyme

module ExternalSamplingFixtures
using ReactiveKernelsPPL
import Distributions
struct OrdinaryDistribution{T} <: Distributions.ContinuousUnivariateDistribution
    mu::T
end
Distributions.logpdf(d::OrdinaryDistribution, x::Real) = normal_lpdf(x, d.mu)
struct NormalDensity{T}
    mu::T
end
const standard_normal = NormalDensity(0.0)
normal_lpdf(x, mu) = -log(2pi)/2 - (x-mu)^2/2
keyword_lpdf(x, lower; rate=1.0) = x > lower ? log(rate)-rate*(x-lower) : -Inf
ReactiveKernelsPPL.sampling_logdensity(d::NormalDensity, x) = normal_lpdf(x, d.mu)

struct ShiftExp{T}
    lower::T
end
ReactiveKernelsPPL.sampling_logdensity(d::ShiftExp, x) = x > d.lower ? -(x-d.lower) : -Inf
shift_constrain(u, shape, lower) = lower + exp(u[1])
shift_unconstrain(x, shape, lower) = [log(x-lower)]
shift_logjac(u, shape, lower) = u[1]
ReactiveKernelsPPL.sampling_geometry(::Type{ShiftExp}, args, shape) =
    ParameterGeometry(shape, 1; support=(:lower, args[1]),
        constrain=shift_constrain, unconstrain=shift_unconstrain, logjac=shift_logjac)

struct PairSimplex end
ReactiveKernelsPPL.sampling_logdensity(::PairSimplex, x) =
    all(x .> 0) && sum(x) ≈ 1 ? log(6.0) + log(x[1]) + log(x[2]) : -Inf
function pair_constrain(u, shape)
    s = 1/(1+exp(-u[1]))
    return [s, 1-s]
end
pair_unconstrain(x, shape) = [log(x[1])-log(x[2])]
function pair_logjac(u, shape)
    s = 1/(1+exp(-u[1]))
    return log(s)+log1p(-s)
end
ReactiveKernelsPPL.sampling_geometry(::Type{PairSimplex}, args, shape) =
    ParameterGeometry(shape, 1; support=:simplex,
        constrain=pair_constrain, unconstrain=pair_unconstrain, logjac=pair_logjac,
        coordinate_names=(:balance,))

identity_constrain(u, shape, mu) = reshape(copy(u), shape)
identity_unconstrain(x, shape, mu) = vec(copy(x))
identity_logjac(u, shape, mu) = 0.0
ReactiveKernelsPPL.sampling_geometry(::Type{NormalDensity}, args, shape) =
    ParameterGeometry(shape, only(shape); support=:real,
        constrain=identity_constrain, unconstrain=identity_unconstrain, logjac=identity_logjac)

struct ClosureExp end
ReactiveKernelsPPL.sampling_logdensity(::ClosureExp, x) = x > 0 ? -x : -Inf
ReactiveKernelsPPL.sampling_geometry(::Type{ClosureExp}, args, shape) =
    ParameterGeometry(shape, 1; support=:positive,
        constrain=(u, shape)->exp(u[1]),
        unconstrain=(x, shape)->[log(x)], logjac=(u, shape)->u[1])

struct KeywordExp{T,R}
    lower::T
    rate::R
end
KeywordExp(lower; rate=1.0) = KeywordExp(lower, rate)
ReactiveKernelsPPL.sampling_logdensity(d::KeywordExp, x) =
    x > d.lower ? log(d.rate)-d.rate*(x-d.lower) : -Inf
function ReactiveKernelsPPL.sampling_geometry(::Type{KeywordExp}, args, shape)
    ParameterGeometry(shape, 1; support=:lower_bounded,
        constrain=(u, shape, lower; rate=1.0)->lower+exp(u[1])/rate,
        unconstrain=(x, shape, lower; rate=1.0)->[log(rate*(x-lower))],
        logjac=(u, shape, lower; rate=1.0)->u[1]-log(rate))
end

# A foreign frontend emits ordinary AST through the public fragment API.
# Neither function is called as an opaque statistical model at runtime.
foreign_latent(mu) = error("the frontend must expand this RHS")
const LATENT = RKPPLSubmodel(:foreign_latent, [:mu], quote
    b ~ Normal(mu, 1)
    return b
end, @__MODULE__)
ReactiveKernelsPPL.sampling_fragment(::typeof(foreign_latent)) = LATENT
foreign_stream(mu) = error("the frontend must expand this RHS")
const STREAM = RKPPLSubmodel(:foreign_stream, [:mu], quote
    slot .~ LogDensity.(normal_lpdf, mu)
    return slot
end, @__MODULE__)
ReactiveKernelsPPL.sampling_fragment(::typeof(foreign_stream)) = STREAM
foreign_constant() = error("the frontend must expand this RHS")
const CONSTANT = RKPPLSubmodel(:foreign_constant, Symbol[], quote
    slot .~ standard_normal
    return slot
end, @__MODULE__)
ReactiveKernelsPPL.sampling_fragment(::typeof(foreign_constant)) = CONSTANT
@rkppl nested(mu) = begin
    inner ~ foreign_latent(mu)
    return inner
end


end

function _external_built(ast, data=Dict{Symbol,Any}())
    before = deepcopy(data)
    plan = bind_data(lower_rkppl(ast, data; mod=@__MODULE__, conditioned=keys(data)), data)
    built = build_kernel(plan)
    @test data == before
    return plan, built
end
function _external_value(built, plan, u, query=:sampler)
    q = prepare_query(built, plan, query)
    return Base.invokelatest(q, u)
end
function _external_gradient(built, plan, u, reference)
    before = copy(u)
    sampler = prepare_sampler(built, plan, u; backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = sampler_value_and_gradient!(sampler, zeros(length(u)), u)
    @test value ≈ reference(u) atol=1e-11
    # Independent oracle, not differences of the generated implementation.
    h = 1e-5
    oracle = map(eachindex(u)) do i
        plus, minus = copy(u), copy(u)
        plus[i] += h; minus[i] -= h
        (reference(plus)-reference(minus))/(2h)
    end
    @test gradient ≈ oracle atol=1e-8 rtol=1e-7
    @test u == before
end

@testset "open observation RHS, broadcasting and named density cuts" begin
    for n in (0, 3, 11), helper in (false, true)
        y = collect(range(-0.2, 0.5; length=n))
        rhs = helper ? :(LogDensity.(ExternalSamplingFixtures.normal_lpdf, a)) :
            :(ExternalSamplingFixtures.NormalDensity.(a))
        ast = quote a ~ Normal(0, 1); y .~ $rhs end
        plan, built = _external_built(ast, Dict(:y=>y))
        u = [0.3]
        prior = w -> -log(2pi)/2-w[1]^2/2
        likelihood = w -> sum(-log(2pi)/2-(x-w[1])^2/2 for x in y; init=0.0)
        @test built.layout.total == 1
        @test _external_value(built, plan, u) ≈ prior(u)+likelihood(u)
        @test _external_value(built, plan, u, :prior) ≈ prior(u)
        @test _external_value(built, plan, u, :likelihood) ≈ likelihood(u)
        @test _external_value(built, plan, u, :pointwise).y ≈
            [-log(2pi)/2-(x-u[1])^2/2 for x in y]
        _external_gradient(built, plan, u, w->prior(w)+likelihood(w))
    end
end

@testset "custom parameter geometry: live parent and distinct dimensions" begin
    ast = quote
        a ~ Normal(0, 1)
        lower = a + 0.7
        b ~ ExternalSamplingFixtures.ShiftExp(lower)
        y .~ ExternalSamplingFixtures.NormalDensity.(b)
    end
    y = [0.2, 1.5, -0.1]
    plan, built = _external_built(ast, Dict(:y=>y))
    u = [0.15, -0.4]
    nt = constrain(built.layout, u)
    @test nt.b ≈ nt.a+0.7+exp(u[2])
    @test unconstrain(built.layout, nt) ≈ u
    @test logjac(built.layout, u) ≈ u[2]
    oracle = w -> begin
        b = w[1]+0.7+exp(w[2])
        -log(2pi)/2-w[1]^2/2-exp(w[2])+w[2] +
            sum(-log(2pi)/2-(x-b)^2/2 for x in y)
    end
    _external_gradient(built, plan, u, oracle)

    pair_ast = quote p[1:2] ~ ExternalSamplingFixtures.PairSimplex() end
    pair_plan, pair_built = _external_built(pair_ast)
    v = [0.25]
    @test pair_built.layout.total == 1
    @test coordinate_names(pair_built.layout) == [Symbol("p.balance")]
    pair = constrain(pair_built.layout, v).p
    @test length(pair) == 2
    @test sum(pair) ≈ 1
    @test unconstrain(pair_built.layout, (p=pair,)) ≈ v
    reference = w -> begin
        s = 1/(1+exp(-w[1]))
        log(6.0)+2log(s)+2log1p(-s)
    end
    @test logjac(pair_built.layout, v) ≈ log(pair[1])+log(pair[2])
    _external_gradient(pair_built, pair_plan, v, reference)
end

@testset "external fragments share RKPPL scope and nesting" begin
    ast = quote
        a ~ ExternalSamplingFixtures.nested(0.2)
        y ~ ExternalSamplingFixtures.foreign_stream(a)
    end
    y = [0.1, -0.3, 0.4]
    plan, built = _external_built(ast, Dict(:y=>y))
    @test coordinate_names(built.layout) == [Symbol("a.inner.b")]
    u = [0.15]
    @test constrain(built.layout, u).a.inner.b ≈ u[1]
    oracle = w -> -log(2pi)/2-(w[1]-0.2)^2/2 +
        sum(-log(2pi)/2-(x-w[1])^2/2 for x in y)
    _external_gradient(built, plan, u, oracle)
    constant_plan, constant_built = _external_built(quote
        a ~ Normal(0, 1)
        y ~ ExternalSamplingFixtures.foreign_constant()
    end, Dict(:y=>y))
    _external_gradient(constant_built, constant_plan, u,
        w -> -log(2pi)/2-w[1]^2/2 +
            sum(ExternalSamplingFixtures.normal_lpdf.(y, 0.0)))
end

@testset "data-sized custom parameters and density-only RHS values" begin
    for n in (0, 3, 11)
        ast = quote
            a ~ Normal(0, 1)
            v[1:N] .~ ExternalSamplingFixtures.NormalDensity.(a)
        end
        plan, built = _external_built(ast, Dict(:N=>n))
        u = vcat(0.2, collect(range(-0.1, 0.4; length=n)))
        @test built.layout.total == n+1
        nt = constrain(built.layout, u)
        @test nt.v == u[2:end]
        @test unconstrain(built.layout, nt) ≈ u
        @test logjac(built.layout, u) == 0
        oracle = w -> -log(2pi)/2-w[1]^2/2 +
            sum(-log(2pi)/2-(x-w[1])^2/2 for x in w[2:end]; init=0.0)
        _external_gradient(built, plan, u, oracle)
    end
    for broadcast in (false, true)
        statement = broadcast ? :(y .~ ExternalSamplingFixtures.standard_normal) :
            :(y ~ ExternalSamplingFixtures.standard_normal)
        y = broadcast ? [0.1, -0.2] : 0.1
        plan, built = _external_built(Expr(:block, :(a ~ Normal(0, 1)), statement), Dict(:y=>y))
        @test built.layout.total == 1
        reference = w -> -log(2pi)/2-w[1]^2/2 +
            (broadcast ? sum(ExternalSamplingFixtures.normal_lpdf.(y, 0.0)) :
                ExternalSamplingFixtures.normal_lpdf(y, 0.0))
        _external_gradient(built, plan, [0.2], reference)
    end
    # Whole-event observations do not acquire a parameter geometry or layout.
    plan, built = _external_built(quote
        a ~ Normal(0, 1)
        y ~ ExternalSamplingFixtures.PairSimplex()
    end, Dict(:y=>[0.3, 0.7]))
    @test built.layout.total == 1
    @test _external_value(built, plan, [0.2], :likelihood) ≈ log(6*0.3*0.7)
end

@testset "custom RHS source replay and observation expressions" begin
    plan, built = _external_built(quote
        b ~ ExternalSamplingFixtures.ClosureExp()
        residual = y .- b
        residual .~ ExternalSamplingFixtures.standard_normal
    end, Dict(:y=>[0.2, 0.4]))
    u = [-0.3]
    reference = w -> -exp(w[1])+w[1] +
        sum(ExternalSamplingFixtures.normal_lpdf.( [0.2, 0.4] .- exp(w[1]), 0.0))
    _external_gradient(built, plan, u, reference)
    source = string(kernel_expr(plan, built.layout))
    replay = Core.eval(@__MODULE__, :(@kernel $(Meta.parse(source))))
    replay_query = prepare_query((;spec=replay, layout=built.layout), plan, :sampler)
    @test Base.invokelatest(replay_query, u) ≈ reference(u)
end

@testset "ordinary distributions and independent observation axes" begin
    for n in (0, 3, 11)
        y = collect(range(-0.2, 0.3; length=n))
        z = reshape([0.1, 0.4, -0.2, 0.0], 2, 2)
        x = reshape([0.2, -0.1], 2, 1)
        plan, built = _external_built(quote
            a ~ Normal(0, 1)
            y .~ Normal.(a, 1)
            mu = a .+ x
            z .~ ExternalSamplingFixtures.OrdinaryDistribution.(mu)
        end, Dict(:y=>y, :z=>z, :x=>x))
        oracle = w -> -log(2pi)/2-w[1]^2/2 +
            sum(ExternalSamplingFixtures.normal_lpdf.(y, w[1])) +
            sum(ExternalSamplingFixtures.normal_lpdf.(z, w[1] .+ x))
        _external_gradient(built, plan, [0.15], oracle)
        @test size(_external_value(built, plan, [0.15], :pointwise).z) == size(z)
    end
end

@testset "open RHS in retained scalar plate cells" begin
    for n in (0, 3, 7)
        y = collect(range(0.3, 1.1; length=n))
        ast = quote
            a ~ Normal(0, 1)
            @plate for i in eachindex(y)
                b[i] ~ ExternalSamplingFixtures.ShiftExp(a)
                y[i] ~ ExternalSamplingFixtures.NormalDensity(b[i])
            end
        end
        plan, built = _external_built(ast, Dict(:y=>y))
        u = vcat(0.2, fill(-0.4, n))
        @test built.layout.total == n+1
        nt = constrain(built.layout, u)
        @test nt.b ≈ u[1] .+ exp.(u[2:end])
        @test unconstrain(built.layout, nt) ≈ u
        @test logjac(built.layout, u) ≈ sum(u[2:end])
        oracle = w -> -log(2pi)/2-w[1]^2/2 +
            sum(-exp(w[i+1])+w[i+1] +
                ExternalSamplingFixtures.normal_lpdf(y[i], w[1]+exp(w[i+1]))
                for i in 1:n; init=0.0)
        _external_gradient(built, plan, u, oracle)
    end
end

@testset "RHS values, rebinding and missing capabilities" begin
    ast = quote
        a ~ Normal(0, 1)
        rhs = ExternalSamplingFixtures.NormalDensity(a)
        y .~ rhs
    end
    y = [0.1, 0.3]
    plan, built = _external_built(ast, Dict(:y=>y))
    _external_gradient(built, plan, [0.2], w -> -log(2pi)/2-w[1]^2/2 +
        sum(ExternalSamplingFixtures.normal_lpdf.(y, w[1])))
    sized = quote v[1:N] .~ ExternalSamplingFixtures.NormalDensity.(0.0) end
    original, small = _external_built(sized, Dict(:N=>2))
    rebound = bind_data(original, Dict(:N=>5))
    @test build_kernel(rebound).layout.total == 5
    @test small.layout.total == 2
    @test original.columns[:N] == 2
    # Missing geometry is diagnosed before any statistical constructor runs.
    @test_throws ArgumentError lower_rkppl(quote x ~ LogDensity(identity) end, ())
    @test_throws ArgumentError sampling_logdensity(1, 0.0)
    @test_throws ContractValidationError _external_built(sized, Dict(:N=>-1))
    @test_throws ContractValidationError _external_built(sized, Dict(:N=>2.5))
    active = quote
        N ~ Normal(0, 1)
        v[1:N] .~ ExternalSamplingFixtures.NormalDensity.(0.0)
    end
    @test_throws ContractValidationError _external_built(active)
end

@testset "ordinary RHS keyword constructors and geometry" begin
    y = [0.4, 0.8]
    plan, built = _external_built(quote
        a ~ Normal(0, 1)
        b ~ ExternalSamplingFixtures.KeywordExp(a; rate=1+exp(a))
        y .~ ExternalSamplingFixtures.KeywordExp.(a; rate=1+exp(a))
        z .~ LogDensity.(ExternalSamplingFixtures.keyword_lpdf, a; rate=1+exp(a))
    end, Dict(:y=>y, :z=>copy(y)))
    u = [0.2, -0.3]
    nt = constrain(built.layout, u)
    @test nt.b ≈ u[1]+exp(u[2])/(1+exp(u[1]))
    @test unconstrain(built.layout, nt) ≈ u
    @test logjac(built.layout, u) ≈ u[2]-log(1+exp(u[1]))
    reference = w -> begin
        rate = 1+exp(w[1])
        -log(2pi)/2-w[1]^2/2-exp(w[2])+w[2] +
            2sum(log(rate)-rate*(x-w[1]) for x in y)
    end
    _external_gradient(built, plan, u, reference)
    @test _external_value(built, plan, u, :pointwise).y ==
        _external_value(built, plan, u, :pointwise).z
    source = string(kernel_expr(plan, built.layout))
    replay = Core.eval(@__MODULE__, :(@kernel $(Meta.parse(source))))
    query = prepare_query((;spec=replay, layout=built.layout), plan, :sampler)
    @test Base.invokelatest(query, u) ≈ reference(u)
end
