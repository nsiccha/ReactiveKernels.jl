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
    L = reduce(hcat, [block.cells(u, (; z = fill(s, length(y)))) for s in support])
    for i in eachindex(y)
        exact = [p[k] * pdf(Normal(mu[k], 1.0), y[i]) for k in 1:3]
        @test PPL._gibbs_weights(L[i, :]) ≈ exact ./ sum(exact) rtol=1e-12
    end
    # All elements are drawn in one pass: each row's probabilities equal
    # `_gibbs_weights`, and the draw takes one uniform per element in order,
    # as drawing each element on its own does.
    P = PPL._gibbs_probabilities!(copy(L))
    @test all(i -> P[i, :] ≈ PPL._gibbs_weights(L[i, :]), eachindex(y))
    batch = PPL._draw_array_block(block, Xoshiro(11), u, d).z
    reference = let rng = Xoshiro(11)
        [support[PPL._integer_draw_index(rng, PPL._gibbs_weights(L[i, :]))] for i in eachindex(y)]
    end
    @test batch == reference
    @test_throws ArgumentError PPL._gibbs_probabilities!([0.0 NaN])
    @test_throws ArgumentError PPL._gibbs_probabilities!([-Inf -Inf])
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

# An integer array sized by an observation (`z[axes(y, 1)]`) has an extent the
# program may not read, since a design binding may lack `y`. Its programs keep
# no binding size: one type serves every number of observations, so a new N
# compiles nothing (the Gen-home-turf mixture recompiled about 12 s per N).
# The host still checks the size wherever an integer state enters.
@testset "integer arrays sized by an observation: one program for every size" begin
    model = @rkppl begin
        mu[1:3] .~ Normal.(0, 5)
        z[axes(y, 1)] .~ Categorical(p)
        y .~ Normal.(mu[z], 1.0)
    end
    p, mu = [0.2, 0.3, 0.5], [-2.0, 0.5, 2.0]
    function prepared(N)
        y = collect(range(-2.5, 2.5; length = N))
        plan = model(; p) | (; y)
        built = build_kernel(plan)
        d = (; z = [mod1(i, 3) for i in 1:N])
        u = unconstrain(built.layout, (; mu, z = d.z))
        problem = prepare_gibbs(built, plan, u; discrete = d, backend = BACKEND)
        (; problem, built, plan, u, d, y)
    end
    a, b = prepared(5), prepared(9)
    @test typeof(a.problem.continuous) === typeof(b.problem.continuous)
    @test typeof(a.problem.integers) === typeof(b.problem.integers)
    @test typeof(a.problem.joint) === typeof(b.problem.joint)
    # The same program evaluates each size's own conditional.
    for (s, N) in ((a, 5), (b, 9))
        block = only(s.problem.integers)
        L = reduce(hcat, [block.cells(s.u, (; z = fill(k, N))) for k in 1:3])
        @test size(L) == (N, 3)
        for i in 1:N
            exact = [p[k] * pdf(Normal(mu[k], 1.0), s.y[i]) for k in 1:3]
            @test PPL._gibbs_weights(L[i, :]) ≈ exact ./ sum(exact) rtol = 1e-12
        end
    end
    # A state of another size is refused by the host check.
    @test_throws ArgumentError prepare_gibbs(a.built, a.plan, a.u;
        discrete = (; z = ones(Int, 4)), backend = BACKEND)
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

# The exact sweep of single-site updates over every joint state: element `i`
# moves by its conditional, the others stay.
function _gibbs_sweep_matrix(block, states; name = :k)
    state(x) = NamedTuple{(name,)}((x,))
    T = Matrix{Float64}(I, length(states), length(states))
    support = PPL._gibbs_block_supports(block, Float64[], state(states[1]))
    for i in eachindex(states[1])
        own = PPL._gibbs_element_support(support, i)
        K = zeros(length(states), length(states))
        for (a, old) in enumerate(states)
            w = PPL._gibbs_weights(PPL._gibbs_site_logweights!(zeros(length(own)),
                block, Float64[], state(old), copy(old), i, own))
            for (j, s) in enumerate(own)
                new = copy(old); new[i] = s
                K[a, findfirst(==(new), states)] += w[j]
            end
        end
        T = T * K
    end
    return T
end

@testset "integer arrays: each element may have its own support" begin
    # Elementwise: Binomial trials and DiscreteUniform bounds read per element.
    n, p, y = [2, 5, 1], 0.35, [1.2, 3.9, 0.1]
    model = @rkppl begin
        k[axes(n, 1)] .~ Binomial.(n, p)
        y .~ Normal.(k, 0.7)
    end
    plan = model(; n, p) | (; y)
    built = build_kernel(plan)
    d = (; k = [0, 0, 0])
    problem = prepare_gibbs(built, plan, Float64[]; discrete=d)
    block = only(problem.integers)
    @test block.elementwise
    @test block.support(Float64[], d) == [0:2, 0:5, 0:1]
    candidates, L = PPL._gibbs_elementwise_logweights(block, Float64[], d)
    @test candidates == 0:5               # each value evaluated once, for every element
    for i in eachindex(n)
        exact = [s <= n[i] ? pdf(Binomial(n[i], p), s) * pdf(Normal(s, 0.7), y[i]) : 0.0
            for s in candidates]
        @test PPL._gibbs_weights(L[i, :]) ≈ exact ./ sum(exact) rtol=1e-12
    end
    chain = initialize_gibbs(Xoshiro(2), problem, Float64[]; discrete=d)
    draws = gibbs_draws(chain, 2000)
    @test all(x -> all(x.k .<= n) && all(>=(0), x.k), draws.discrete)
    @test any(x -> x.k[2] >= 3, draws.discrete)  # beyond the other elements' supports

    # Disjoint ranges: only values some element can take are evaluated.
    lo, hi, y2 = [0, 5, 2], [1, 6, 3], [0.8, 5.6, 2.2]
    model2 = @rkppl begin
        k[axes(lo, 1)] .~ DiscreteUniform.(lo, hi)
        y .~ Normal.(k, 0.5)
    end
    plan2 = model2(; lo, hi) | (; y = y2)
    built2 = build_kernel(plan2)
    d2 = (; k = copy(lo))
    block2 = only(prepare_gibbs(built2, plan2, Float64[]; discrete=d2).integers)
    candidates2, L2 = PPL._gibbs_elementwise_logweights(block2, Float64[], d2)
    @test candidates2 == [0, 1, 2, 3, 5, 6]
    for i in eachindex(lo)
        exact = [lo[i] <= s <= hi[i] ? pdf(Normal(s, 0.5), y2[i]) : 0.0 for s in candidates2]
        @test PPL._gibbs_weights(L2[i, :]) ≈ exact ./ sum(exact) rtol=1e-12
    end

    # A restriction read per element intersects each element's own support.
    nr, lor, hir, yr = [4, 5, 3], [1, 0, 2], [3, 2, 3], [2.2, 0.4, 2.9]
    modelr = @rkppl begin
        k[axes(n, 1)] .~ restricted.(Binomial.(n, 0.4), lo, hi)
        y .~ Normal.(k, 0.6)
    end
    planr = modelr(; n = nr, lo = lor, hi = hir) | (; y = yr)
    builtr = build_kernel(planr)
    dr = (; k = [2, 1, 2])
    blockr = only(prepare_gibbs(builtr, planr, Float64[]; discrete=dr).integers)
    @test blockr.support(Float64[], dr) == [1:3, 0:2, 2:3]
    candidatesr, Lr = PPL._gibbs_elementwise_logweights(blockr, Float64[], dr)
    for i in eachindex(nr)
        exact = [lor[i] <= s <= hir[i] ? pdf(Binomial(nr[i], 0.4), s) *
            pdf(Normal(s, 0.6), yr[i]) : 0.0 for s in candidatesr]
        @test PPL._gibbs_weights(Lr[i, :]) ≈ exact ./ sum(exact) rtol=1e-12
    end

    # The loop spelling reads its trials per cell, as the array does.
    model3 = @rkppl begin
        @plate for i in eachindex(y)
            k[i] ~ Binomial(n[i], p)
            y[i] ~ Normal(k[i], 0.7)
        end
    end
    plan3 = model3(; n, p) | (; y)
    built3 = build_kernel(plan3)
    problem3 = prepare_gibbs(built3, plan3, Float64[]; discrete=d)
    @test only(problem3.integers).elementwise
    a = gibbs_draws(initialize_gibbs(Xoshiro(7), problem, Float64[]; discrete=d), 200)
    b = gibbs_draws(initialize_gibbs(Xoshiro(7), problem3, Float64[]; discrete=d), 200)
    @test [x.k for x in a.discrete] == [x.k for x in b.discrete]

    # Coupled elements: single-site updates over each element's own support
    # leave the exact joint law invariant.
    model4 = @rkppl begin
        k[axes(n, 1)] .~ Binomial.(n, p)
        s = sum(k)
        t ~ Normal(s, 0.8)
    end
    n4, t = [1, 2, 1], 2.3
    plan4 = model4(; n = n4, p) | (; t)
    built4 = build_kernel(plan4)
    d4 = (; k = [0, 0, 0])
    block4 = only(prepare_gibbs(built4, plan4, Float64[]; discrete=d4).integers)
    @test !block4.elementwise
    states = vec([collect(c) for c in Iterators.product(0:1, 0:2, 0:1)])
    joint = [prod(pdf.(Binomial.(n4, p), s)) * pdf(Normal(sum(s), 0.8), t) for s in states]
    joint ./= sum(joint)
    T = _gibbs_sweep_matrix(block4, states)
    @test vec(sum(T; dims=2)) ≈ ones(length(states))
    @test transpose(T) * joint ≈ joint rtol=1e-12

    # `Categorical.(ws)` over data holding one probability vector per element
    # reads element `i`'s law from `ws[i]`, as in Julia.
    ws, y5, mu = [[0.5, 0.5], [0.2, 0.3, 0.5], [1.0]], [0.1, 2.0, -1.0], [-1.0, 0.3, 1.2]
    model5 = @rkppl begin
        mu[1:3] .~ Normal.(0, 5)
        z[axes(y, 1)] .~ Categorical.(ws)
        y .~ Normal.(mu[z], 1.0)
    end
    plan5 = model5(; ws) | (; y = y5)
    built5 = build_kernel(plan5)
    q5 = prepare_query(built5, plan5, :sampler)
    for z in ([1, 3, 1], [2, 2, 1])
        u5 = unconstrain(built5.layout, (; mu, z))
        @test q5(u5, (; z)) ≈ sum(logpdf.(Normal(0, 5), mu)) +
            sum(log(ws[i][z[i]]) for i in 1:3) + sum(logpdf.(Normal.(mu[z], 1.0), y5))
    end
    d5 = (; z = [1, 1, 1])
    u5 = unconstrain(built5.layout, (; mu, z = d5.z))
    sampler5 = prepare_sampler(built5, plan5, u5; backend=BACKEND, discrete=d5)
    g5 = zeros(3)
    sampler_value_and_gradient!(sampler5, g5, u5)
    @test g5 ≈ [(q5(u5 .+ 1e-6 .* (1:3 .== j), d5) - q5(u5 .- 1e-6 .* (1:3 .== j), d5)) / 2e-6
        for j in 1:3] rtol=1e-6
    block5 = only(prepare_gibbs(built5, plan5, u5; discrete=d5, backend=BACKEND).integers)
    @test block5.elementwise
    @test block5.support(u5, d5) == [1:2, 1:3, 1:1]
    candidates5, L5 = PPL._gibbs_elementwise_logweights(block5, u5, d5)
    for i in 1:3
        exact = [k <= length(ws[i]) ? ws[i][k] * pdf(Normal(mu[k], 1.0), y5[i]) : 0.0
            for k in candidates5]
        @test PPL._gibbs_weights(L5[i, :]) ≈ exact ./ sum(exact) rtol=1e-12
    end
    err = try build_kernel(model5(; ws = ws[1:2]) | (; y = y5)); nothing catch e; e end
    @test err isa PPL.ContractValidationError
    @test occursin("3 elements but its prior arg1 has 2", sprint(showerror, err))

    # A two-axis array keeps each element's own support in its shape.
    N = [1 3; 2 0]
    model6 = @rkppl begin
        k[1:2, 1:2] .~ Binomial.(N, 0.5)
        y .~ Normal.(vec(k), 1.0)
    end
    plan6 = model6(; N) | (; y = [0.4, 1.5, 2.2, 0.0])
    built6 = build_kernel(plan6)
    d6 = (; k = zeros(Int, 2, 2))
    problem6 = prepare_gibbs(built6, plan6, Float64[]; discrete=d6)
    block6 = only(problem6.integers)
    @test block6.support(Float64[], d6) == reshape([0:1, 0:2, 0:3, 0:0], 2, 2)
    draws6 = gibbs_draws(initialize_gibbs(Xoshiro(4), problem6, Float64[]; discrete=d6), 500)
    @test all(x -> size(x.k) == (2, 2) && all(0 .<= x.k .<= N), draws6.discrete)
    @test any(x -> x.k[1, 2] == 3, draws6.discrete)
end


@testset "collapsed integers: Gibbs beside marginalized integers" begin
    # A cell body naming a NamedTuple field after an enclosing value still
    # reads that value: marginalized programs build `pointwise = (y = …,)`.
    reads = PPL._statement_value_reads!(Set{Symbol}(),
        :(plate(s) do i; pointwise = (y = y[i] + w,); pointwise end))
    @test :y in reads && :w in reads && :pointwise ∉ reads
    # `a` is summed out; `k`'s conditional is p(k | p, y) ∝ Σₐ p(k, a, y | p).
    n, y = 4, 2.6
    model = @rkppl begin
        p ~ Beta(2, 2)
        a ~ Bernoulli(0.3)
        k ~ Binomial(n, p)
        y ~ Normal(k + 2a, 1)
    end
    plan = marginalize(model(; n), :a) | (; y)
    built = build_kernel(plan)
    d = (; k = 1)
    u = unconstrain(built.layout, (; p = 0.45, k = 1))
    problem = prepare_gibbs(built, plan, u; discrete=d, backend=BACKEND)
    @test [b.names for b in problem.integers] == [(:k,)]
    conditional = only(problem.integers).kernel(u, d)
    exact = [pdf(Binomial(n, 0.45), k) * (0.7 * pdf(Normal(k, 1), y) +
        0.3 * pdf(Normal(k + 2, 1), y)) for k in 0:n]
    @test PPL._gibbs_weights(conditional.logweights) ≈ exact ./ sum(exact) rtol=1e-12
    @test problem.joint(u, d) ≈ logpdf(Beta(2, 2), 0.45) + logpdf(Binomial(n, 0.45), 1) +
        log(0.7 * pdf(Normal(1, 1), y) + 0.3 * pdf(Normal(3, 1), y)) + log(0.45 * 0.55)
    draws = gibbs_draws(initialize_gibbs(Xoshiro(1), problem, u; discrete=d, warmup=200), 300)
    @test all(x -> x.k in 0:n, draws.discrete)

    # Two explicit scalars beside a marginalized one: the sweep leaves the
    # collapsed joint law invariant.
    model2 = @rkppl begin
        a ~ Bernoulli(0.4)
        k ~ Bernoulli(0.3 + 0.4a)
        z ~ Bernoulli(0.2 + 0.5k)
        y ~ Normal(k + z + a, 1)
    end
    plan2 = marginalize(model2(), :a) | (; y = 1.3)
    built2 = build_kernel(plan2)
    problem2 = prepare_gibbs(built2, plan2, Float64[]; discrete=(; k=0, z=0))
    states = [(; k, z) for k in 0:1 for z in 0:1]
    joint = [sum((a == 1 ? 0.4 : 0.6) * pdf(Bernoulli(0.3 + 0.4a), s.k) *
        pdf(Bernoulli(0.2 + 0.5s.k), s.z) * pdf(Normal(s.k + s.z + a, 1), 1.3) for a in 0:1)
        for s in states]
    joint ./= sum(joint)
    @test PPL._gibbs_weights([problem2.joint(Float64[], s) for s in states]) ≈ joint rtol=1e-12
    T = zeros(4, 4)
    for (i, old) in enumerate(states), (j, new) in enumerate(states)
        first_block = problem2.integers[1].kernel(Float64[], old)
        second_block = problem2.integers[2].kernel(Float64[], (; k=new.k, z=old.z))
        T[i, j] = PPL._gibbs_weights(first_block.logweights)[new.k+1] *
            PPL._gibbs_weights(second_block.logweights)[new.z+1]
    end
    @test transpose(T) * joint ≈ joint rtol=1e-12

    # An explicit integer array beside a marginalized scalar: its terms are
    # sums over the marginalized states, so elements update one by one.
    y3 = [0.2, 2.9, 1.1]
    model3 = @rkppl begin
        a ~ Bernoulli(0.5)
        z[axes(y, 1)] .~ Bernoulli(0.4)
        y .~ Normal.(z .+ 2a, 1)
    end
    plan3 = marginalize(model3(), :a) | (; y = y3)
    built3 = build_kernel(plan3)
    block3 = only(prepare_gibbs(built3, plan3, Float64[]; discrete=(; z=[0, 0, 0])).integers)
    @test !block3.elementwise
    states3 = vec([collect(c) for c in Iterators.product(0:1, 0:1, 0:1)])
    joint3 = [sum(0.5 * prod(pdf.(Bernoulli(0.4), s)) * prod(pdf.(Normal.(s .+ 2a, 1), y3))
        for a in 0:1) for s in states3]
    joint3 ./= sum(joint3)
    T3 = _gibbs_sweep_matrix(block3, states3; name = :z)
    @test transpose(T3) * joint3 ≈ joint3 rtol=1e-12

    # Nothing left to update: the remedy is NUTS on the real coordinates.
    plan4 = marginalize(@rkppl(begin
        a ~ Bernoulli(0.3)
        y ~ Normal(2a, 1)
    end)(), :a) | (; y = 0.5)
    err = try prepare_gibbs(build_kernel(plan4), plan4, Float64[]; discrete=nothing); nothing catch e; e end
    @test err isa ArgumentError && occursin("NUTS", sprint(showerror, err))
end

end
