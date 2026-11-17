module GibbsTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme
using WarmupHMC, LogDensityProblems, Random, LinearAlgebra
using DifferentiationInterface: AutoEnzyme
const PPL = ReactiveKernelsPPL
const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

@rkppl gibbs_child(lo, hi) = begin
    k ~ DiscreteUniform(lo, hi)
    return k
end

const RAG = @rkppl begin
    p ~ Beta(a, b)
    c ~ Binomial(n, p)
    r = h - c
    r ~ Binomial(n - c, 0.5)
end

@testset "RAG conditionals and constant density terms" begin
    plan = RAG(; n=12, h=8, a=2.0, b=3.0) | (;)
    built = build_kernel(plan)
    u, discrete = [0.2], (; c=4)
    problem = prepare_gibbs(built, plan, u; discrete, backend=BACKEND)
    for x in (-1.0, 0.0, 1.2)
        p = 1 / (1 + exp(-x))
        conditional = only(problem.integers).kernel([x], discrete)
        @test vec(first.(conditional.states)) == 0:12
        @test PPL._gibbs_weights(conditional.logweights) ≈
            [pdf(Binomial(8, 2p / (1+p)), c) for c in 0:12]
        # The discrete block omits p's prior and its coordinate Jacobian.
        for c in 0:8
            @test conditional.logweights[c+1] ≈ logpdf(Binomial(12, p), c) +
                logpdf(Binomial(12-c, 0.5), 8-c)
        end
        for c in (0, 4, 8)
            problem.continuous.discrete = (; c)
            expected = logpdf(Beta(2, 3), p) + logpdf(Binomial(12, p), c) +
                log(p) + log1p(-p)
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem.continuous, [x])
            @test value ≈ expected
            @test only(gradient) ≈ 2 + c - 17p
            # The honest-claims factor is absent from the continuous target.
            @test problem.joint([x], (; c)) - value ≈ logpdf(Binomial(12-c, 0.5), 8-c)
        end
    end

    state = initialize_gibbs(Xoshiro(42), problem, u; discrete, stepsize=0.3)
    copy_u, copy_d = copy(u), deepcopy(discrete)
    rng = copy(state.rng)
    conditional = only(problem.integers).kernel(u, discrete)
    index = PPL._integer_draw_index(rng, PPL._gibbs_weights(conditional.logweights))
    c = only(conditional.states[index])
    problem.continuous.discrete = (; c)
    reference = fixed_kernel_nuts_step(state.nuts, problem.continuous; blas_threads=1)
    transition = gibbs_step!(state)
    @test state.discrete == (; c)
    @test state.position == reference.state.position
    @test transition.divergent == reference.divergent
    @test u == copy_u && discrete == copy_d
    draws = gibbs_draws(state, 32; discard=8)
    saved = deepcopy(draws)
    gibbs_step!(state)
    @test draws == saved
    @test all(c -> 0 <= c <= 8, draws.parameters.c)
    @test all(p -> 0 < p < 1, draws.parameters.p)
    @test size(draws.positions) == (1, 32)
    @test eltype(draws.parameters.c) === Int
    @test size(gibbs_draws(state, 0).positions) == (1, 0)
    a = initialize_gibbs(Xoshiro(72), problem, u; discrete, stepsize=0.3)
    b = initialize_gibbs(Xoshiro(72), problem, u; discrete, stepsize=0.3)
    @test gibbs_draws(a, 12) == gibbs_draws(b, 12)
    @test_throws ArgumentError initialize_gibbs(Xoshiro(1), problem, u; discrete=(; c=9), stepsize=0.3)
    @test_throws ArgumentError initialize_gibbs(Xoshiro(1), problem, u; discrete,
        metric=Diagonal([1.0]))
    @test_throws ArgumentError initialize_gibbs(Xoshiro(1), problem, u; discrete, warmup=-1)
    @test state.warmup.sweeps == 0      # a supplied kernel runs no warm-up
    @test_throws ArgumentError gibbs_draws(state, -1)
    @test_throws ArgumentError prepare_gibbs(built, plan, u; discrete, backend=BACKEND, blocks=((:c,:c),))
end

# Monte Carlo standard error of a chain average by batch means.
function batch_mcse(x; batches=40)
    m = length(x) ÷ batches
    means = [sum(view(x, (b - 1) * m + 1:b * m)) / m for b in 1:batches]
    return sqrt(sum(abs2, means .- sum(means) / batches) / (batches - 1)) / sqrt(batches)
end

@testset "adaptive warm-up tunes the continuous kernel" begin
    n, h, a, b = 12, 8, 2.0, 3.0
    plan = RAG(; n, h, a, b) | (;)
    built = build_kernel(plan)
    u, discrete = [0.2], (; c=4)
    problem = prepare_gibbs(built, plan, u; discrete, backend=BACKEND)
    # The exact posterior with c summed out: h | p ~ Binomial(n, (1 + p) / 2),
    # and c | p, h ~ Binomial(h, 2p / (1 + p)).
    grid = range(1e-7, 1 - 1e-7; length=20001)
    w = [pdf(Beta(a, b), p) * pdf(Binomial(n, (1 + p) / 2), h) for p in grid]
    w ./= sum(w)
    mean_p = sum(w .* grid)
    mean_c = sum(w .* (h .* 2 .* grid ./ (1 .+ grid)))
    ps, cs, divergent = Float64[], Float64[], 0
    for seed in 1:4
        state = initialize_gibbs(Xoshiro(seed), problem, u; discrete, warmup=300)
        @test state.warmup.sweeps == 300
        @test state.warmup.evaluations > 300
        draws = gibbs_draws(state, 2000)
        append!(ps, draws.parameters.p)
        append!(cs, draws.parameters.c)
        divergent += count(draws.divergent)
    end
    @test abs(sum(ps) / length(ps) - mean_p) < 5 * batch_mcse(ps)
    @test abs(sum(cs) / length(cs) - mean_c) < 5 * batch_mcse(cs)
    @test divergent <= 8
    # The same seed reproduces warm-up and draws.
    a1 = initialize_gibbs(Xoshiro(9), problem, u; discrete, warmup=50)
    a2 = initialize_gibbs(Xoshiro(9), problem, u; discrete, warmup=50)
    @test gibbs_draws(a1, 10) == gibbs_draws(a2, 10)
    @test a1.warmup == a2.warmup
end

@testset "exact finite Gibbs transition preserves the joint law" begin
    model = @rkppl begin
        k ~ Bernoulli(0.3)
        z ~ Bernoulli(0.15 + 0.7k)
        y ~ Normal(k + z, 1)
    end
    plan = model() | (; y=0.4)
    built = build_kernel(plan)
    problem = prepare_gibbs(built, plan, Float64[]; discrete=(; k=0, z=0))
    states = [(; k, z) for k in 0:1 for z in 0:1]
    joint = PPL._gibbs_weights([problem.joint(Float64[], d) for d in states])
    transition = zeros(4, 4)
    for (i, old) in enumerate(states), (j, new) in enumerate(states)
        first_block = problem.integers[1].kernel(Float64[], old)
        second_block = problem.integers[2].kernel(Float64[], (; k=new.k, z=old.z))
        transition[i,j] = PPL._gibbs_weights(first_block.logweights)[new.k+1] *
            PPL._gibbs_weights(second_block.logweights)[new.z+1]
    end
    @test vec(sum(transition; dims=2)) ≈ ones(4)
    @test transpose(transition) * joint ≈ joint
    chain = initialize_gibbs(Xoshiro(5), problem, Float64[]; discrete=(; k=0,z=0))
    @test chain.warmup.sweeps == 0      # nothing to adapt without real coordinates
    draws = gibbs_draws(chain, 100)
    @test size(draws.positions) == (0,100)
    @test all(d -> d in states, draws.discrete)
    @test !any(draws.divergent)
end

@testset "joint blocks and malformed conditionals" begin
    model = @rkppl begin
        k ~ Bernoulli(0.3)
        z ~ Bernoulli(k)
    end
    plan = model() | (;)
    built = build_kernel(plan)
    problem = prepare_gibbs(built, plan, Float64[];
        discrete=(; k=0,z=0), blocks=((:k,:z),))
    conditional = only(problem.integers).kernel(Float64[], (; k=0,z=0))
    @test PPL._gibbs_weights(conditional.logweights) ≈ [0.7,0.0,0.0,0.3]
    chain = initialize_gibbs(Xoshiro(8), problem, Float64[]; discrete=(; k=0,z=0))
    draws = gibbs_draws(chain, 100)
    @test all(d -> d.k == d.z, draws.discrete)
    @test Set(draws.parameters.k) == Set((0,1))
    @test_throws ArgumentError PPL._gibbs_weights([-Inf,-Inf])
    @test_throws ArgumentError PPL._gibbs_weights([0.0,NaN])
    @test_throws ArgumentError PPL._gibbs_weights([0.0,Inf])
    @test_throws ArgumentError PPL._gibbs_weights(Float64[])
end

@testset "nested categorical and integer-range blocks" begin
    model = @rkppl begin
        t ~ gibbs_child(lo, hi)
        z ~ Categorical(probabilities)
        y ~ Normal(t + z, 1)
    end
    plan = model(; lo=-1, hi=2, probabilities=[0.2,0.3,0.5]) | (; y=1.2)
    built = build_kernel(plan)
    discrete = (; t=(; k=0), z=1)
    problem = prepare_gibbs(built, plan, Float64[]; discrete,
        blocks=((Symbol("t.k"),:z),))
    conditional = only(problem.integers).kernel(Float64[], discrete)
    reference = [logpdf(DiscreteUniform(-1,2), k) + logpdf(Categorical([0.2,0.3,0.5]),z) +
        logpdf(Normal(k+z,1),1.2) for (k,z) in conditional.states]
    @test conditional.logweights ≈ reference
    state = initialize_gibbs(Xoshiro(9), problem, Float64[]; discrete)
    draws = gibbs_draws(state, 30)
    @test all(d -> -1 <= d.t.k <= 2 && 1 <= d.z <= 3, draws.discrete)
    @test draws.parameters.t.k == [d.t.k for d in draws.discrete]

    uniform = @rkppl begin
        k ~ DiscreteUniform(lo,hi)
    end
    sizes = Int[]
    nodes(x) = x isa Expr ? 1 + sum(nodes, x.args; init=0) : 1
    for hi in (0,2,300)
        bound = uniform(; lo=0,hi) | (;)
        prepared = prepare_gibbs(build_kernel(bound), bound, Float64[]; discrete=(; k=UInt8(0)))
        block = only(prepared.integers)
        value = block.kernel(Float64[], (; k=0))
        @test length(value.states) == hi+1
        @test PPL._gibbs_weights(value.logweights) ≈ fill(1/(hi+1),hi+1)
        push!(sizes, nodes(code_expr(block.kernel)))
        chain = initialize_gibbs(Xoshiro(20), prepared, Float64[]; discrete=(; k=UInt8(0)))
        output = gibbs_draws(chain, 50)
        @test eltype(output.parameters.k) === Int
        hi == 300 && @test maximum(output.parameters.k) > typemax(UInt8)
    end
    @test maximum(sizes) < 2minimum(sizes)
    println("GIBBS_SUPPORT_SOURCE_NODES ", sizes)
end

@testset "integer-dependent transforms retain their Jacobian" begin
    model = @rkppl begin
        k ~ Bernoulli(0.3)
        x ~ Uniform(0, k+1)
    end
    plan = model() | (;)
    built = build_kernel(plan)
    problem = prepare_gibbs(built, plan, [0.2]; discrete=(; k=0), backend=BACKEND)
    for x in (-0.7, 0.2, 1.4)
        conditional = only(problem.integers).kernel([x], (; k=0))
        @test PPL._gibbs_weights(conditional.logweights) ≈ [0.7,0.3]
        for k in 0:1
            # Holding packed coordinates fixed changes the physical x with k.
            # The joint density in those coordinates includes that Jacobian.
            @test conditional.logweights[k+1] ≈ problem.joint([x], (; k))
        end
    end
end

function inventory(ex)
    ex isa Expr || return (nodes=1, loops=0, functions=0)
    children = map(inventory, ex.args)
    return (nodes=1+sum(c->c.nodes,children;init=0),
        loops=Int(ex.head in (:for,:while))+sum(c->c.loops,children;init=0),
        functions=Int(ex.head in (:function,:(->)))+sum(c->c.functions,children;init=0))
end
inventory(program::ReactiveKernels._PlateProgram) = inventory(program.ast)

@testset "Gibbs enumeration retains runtime bodies" begin
    model = @rkppl begin
        p ~ Beta(2,3)
        k ~ Binomial(n,p)
        y .~ Normal.(k*p,1)
    end
    u = [0.3]
    p = 1/(1+exp(-only(u)))
    inventories = NamedTuple[]
    for (n,ny) in ((0,0),(2,2),(20,7),(50,40))
        y = collect(range(0.2,1.7;length=ny))
        plan = model(;n) | (;y)
        problem = prepare_gibbs(build_kernel(plan),plan,u;discrete=(;k=0),backend=BACKEND)
        kernel = only(problem.integers).kernel
        scores = [logpdf(Binomial(n,p),k)+sum(logpdf.(Normal(k*p,1),y)) for k in 0:n]
        value = kernel(u,(;k=0))
        @test vec(value.logweights) ≈ scores
        @test PPL._gibbs_weights(value.logweights) ≈ PPL._gibbs_weights(scores)
        metrics = inventory(code_expr(kernel))
        push!(inventories,(;n,ny,metrics...))
    end
    @test all(i->i.loops>0,inventories[2:end])
    @test inventories[end].nodes < 2inventories[2].nodes
    @test inventories[end].functions <= inventories[2].functions+2
    println("GIBBS NATIVE BODY INVENTORY ",inventories)
end
@testset "integer arrays: elementwise blocks draw every element exactly" begin
    y = [-2.1, -1.9, 2.0, 2.2, 1.8]
    model = @rkppl begin
        mu[1:3] .~ Normal.(0, 5)
        z[axes(y, 1)] .~ Categorical(p)
        y .~ Normal.(mu[z], 1.0)
    end
    p, mu = [0.2, 0.3, 0.5], [-2.0, 0.5, 2.0]
    plan = model(; p) | (; y)
    built = build_kernel(plan)
    d = (; z = [1, 1, 1, 1, 1])
    u = unconstrain(built.layout, (; mu, z = d.z))
    problem = prepare_gibbs(built, plan, u; discrete=d, backend=BACKEND)
    block = only(problem.integers)
    @test block isa PPL.GibbsArrayBlock && block.elementwise
    support = collect(block.support(u, d))
    @test support == 1:3
    # Every element's exact conditional, from one evaluation per support value.
    L = reduce(hcat, [block.cells(u, d, s) for s in support])
    for i in eachindex(y)
        exact = [p[k] * pdf(Normal(mu[k], 1.0), y[i]) for k in 1:3]
        @test PPL._gibbs_weights(L[i, :]) ≈ exact ./ sum(exact) rtol=1e-12
    end
    # A one-coordinate model whose posterior of `m` is known by quadrature.
    model2 = @rkppl begin
        m ~ Normal(0, 3)
        z[axes(y, 1)] .~ Bernoulli(0.3)
        y .~ Normal.(m .+ delta .* z, 1.0)
    end
    delta = 2.5
    plan2 = model2(; delta) | (; y)
    built2 = build_kernel(plan2)
    d2 = (; z = zeros(Int, 5))
    problem2 = prepare_gibbs(built2, plan2, [0.0]; discrete=d2, backend=BACKEND)
    @test only(problem2.integers).elementwise
    grid = range(-8, 8; length=8001)
    post = [pdf(Normal(0, 3), m) * prod(0.7 * pdf(Normal(m, 1), yi) +
        0.3 * pdf(Normal(m + delta, 1), yi) for yi in y) for m in grid]
    post ./= sum(post)
    ms = Float64[]
    for seed in 1:4
        state = initialize_gibbs(Xoshiro(seed), problem2, [0.0]; discrete=d2, warmup=300)
        draws = gibbs_draws(state, 3000)
        @test all(z -> all(in(0:1), z.z), draws.discrete)
        append!(ms, draws.parameters.m)
    end
    @test abs(sum(ms) / length(ms) - sum(post .* grid)) < 5 * batch_mcse(ms)
    @test_throws ArgumentError prepare_gibbs(built2, plan2, [0.0]; discrete=d2,
        backend=BACKEND, blocks=((:z,),))
end

@testset "integer arrays: coupled elements use exact single-site updates" begin
    model = @rkppl begin
        z[axes(y, 1)] .~ Bernoulli(0.4)
        y .~ Normal.(2.0 .* z, 1.0)
        s = sum(z)
        t ~ Normal(s, 0.7)
    end
    y, t = [0.3, 1.9, 1.2, -0.4], 2.6
    plan = model() | (; y, t)
    built = build_kernel(plan)
    d = (; z = [0, 0, 0, 0])
    problem = prepare_gibbs(built, plan, Float64[]; discrete=d)
    @test !only(problem.integers).elementwise   # `sum(z)` reads every element
    states = vec([collect(c) for c in Iterators.product(0:1, 0:1, 0:1, 0:1)])
    lw = [sum(log(zi == 1 ? 0.4 : 0.6) for zi in s) + sum(logpdf.(Normal.(2.0 .* s, 1.0), y)) +
        logpdf(Normal(sum(s), 0.7), t) for s in states]
    exact = exp.(lw .- maximum(lw))
    exact ./= sum(exact)
    draws = gibbs_draws(initialize_gibbs(Xoshiro(3), problem, Float64[]; discrete=d), 40000)
    frequency = [count(x -> x.z == s, draws.discrete) / 40000 for s in states]
    @test maximum(abs.(frequency .- exact)) < 0.02
end

end
