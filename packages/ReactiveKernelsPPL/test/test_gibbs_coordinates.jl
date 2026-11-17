module GibbsCoordinateTests
using Test, ReactiveKernels, ReactiveKernelsPPL, Random, LinearAlgebra
const PPL = ReactiveKernelsPPL

const LOGISTIC = @rkppl begin
    beta[axes(X, 2)] .~ Normal.(0, sigma)
    y .~ Bernoulli.(logistic.(X * beta))
end
const SCALED = @rkppl begin
    beta[axes(X, 2)] .~ Normal.(0, 2.0)
    sigma ~ Exponential(1.0)
    y .~ Normal.(X * beta, sigma)
end
const TWO = @rkppl begin
    a[axes(X, 2)] .~ Normal.(0, 1.0)
    b[axes(Z, 2)] .~ Normal.(0, 3.0)
    y .~ Bernoulli.(logistic.(X * a .+ Z * b))
end
const HIERARCHICAL = @rkppl begin
    beta[axes(X, 2)] .~ Normal.(0, tau)
    tau ~ Exponential(1.0)
    y .~ Bernoulli.(logistic.(X * beta))
end
const POSITIVE = @rkppl begin
    beta[axes(X, 2)] .~ Exponential.(tau)
    tau ~ Exponential(1.0)
    y .~ Bernoulli.(logistic.(X * beta))
end
const PLAIN = @rkppl begin
    m ~ Normal(0, 3)
    s ~ Exponential(1.0)
    y .~ Normal.(m, s)
end

at(u, k, z) = (v = copy(u); v[k] = z; v)

# Trials at random values with a coin deciding each move. A conditional may
# omit terms that are constant in the coordinates, so compare it with the
# joint density through differences from the committed point.
function exercise!(rng, state; sweeps, trials, scale=0.5)
    joint, u = state.problem.joint, state.position
    worst, moves = 0.0, 0
    for _ in 1:sweeps, k in eachindex(u)
        here = PPL.coordinate_logdensity(state, k, u[k])
        base = joint(u)
        z = u[k]
        for _ in 1:trials
            z = u[k] + scale * randn(rng)
            before = copy(u)
            cached = PPL.coordinate_logdensity(state, k, z)
            @test u == before
            worst = max(worst, abs((cached - here) - (joint(at(u, k, z)) - base)))
        end
        if rand(rng) < 0.5
            PPL.set_coordinate!(state, k, z)
            @test u[k] == z
            moves += 1
        end
    end
    return (; worst, moves)
end

@testset "cached logistic conditional equals the joint density" begin
    rng = Xoshiro(20261117)
    X = randn(rng, 80, 6)
    y = Int.(rand(rng, 80) .< 0.4)
    plan = LOGISTIC(; X, sigma=10.0) | (; y)
    problem = PPL.prepare_coordinate_conditional(build_kernel(plan), plan, zeros(6))
    @test length(problem.products) == 1
    # One block for the coefficient vector; its prior is read one cell at a time.
    @test [(block.range, length(block.cells)) for block in problem.blocks] == [(1:6, 1)]
    u0 = 0.3 .* randn(rng, 6)
    start = copy(u0)
    state = PPL.coordinate_state(problem, u0)
    # Closed forms at the start. A coefficient's conditional keeps the
    # likelihood and its own prior cell; the other cells are constant in it.
    eta = X * u0
    cells = @. -0.5 * (u0 / 10.0)^2 - log(10.0) - 0.5 * log(2pi)
    loglikelihood = sum(@. y * eta - log1p(exp(eta)))
    @test problem.joint(u0) ≈ loglikelihood + sum(cells)
    for k in (1, 3, 6)
        @test PPL.coordinate_logdensity(state, k, u0[k]) ≈ loglikelihood + cells[k]
    end
    result = exercise!(rng, state; sweeps=20, trials=3, scale=2.0)
    @test result.moves > 20
    @test result.worst < 1e-9
    @test u0 == start                       # the caller's start is not written
    @test only(state.products) ≈ X * state.position
    @test PPL.refresh_products!(state) < 1e-10
    @test only(state.products) == X * state.position

    # Independent states share the prepared problem and nothing else.
    other = PPL.coordinate_state(problem, -u0)
    kept = copy(only(other.products))
    exercise!(rng, state; sweeps=2, trials=2)
    @test only(other.products) == kept
    @test only(other.products) !== only(state.products)
    @test exercise!(rng, other; sweeps=5, trials=2).worst < 1e-9

    # Another binding of the same model, with other sizes.
    X2 = randn(rng, 17, 3)
    y2 = Int.(rand(rng, 17) .< 0.5)
    plan2 = LOGISTIC(; X=X2, sigma=0.7) | (; y=y2)
    problem2 = PPL.prepare_coordinate_conditional(build_kernel(plan2), plan2, zeros(3))
    state2 = PPL.coordinate_state(problem2, randn(rng, 3))
    @test exercise!(rng, state2; sweeps=10, trials=3).worst < 1e-9
    @test exercise!(rng, state; sweeps=2, trials=2).worst < 1e-9

    @test_throws DimensionMismatch PPL.coordinate_state(problem, zeros(5))
    @test_throws BoundsError PPL.coordinate_logdensity(state, 7, 0.0)
    @test_throws BoundsError PPL.set_coordinate!(state, 0, 0.0)
end

@testset "coordinates outside a product and two products" begin
    rng = Xoshiro(5)
    X, Z = randn(rng, 40, 4), randn(rng, 40, 3)
    yn = X * [0.5, -1.0, 0.2, 0.0] .+ 0.7 .* randn(rng, 40)
    plan = SCALED(; X) | (; y=yn)
    problem = PPL.prepare_coordinate_conditional(build_kernel(plan), plan, zeros(5))
    @test length(problem.products) == 1
    @test [(block.range, length(block.cells)) for block in problem.blocks] == [(2:5, 1)]
    state = PPL.coordinate_state(problem, [0.1, -0.2, 0.3, 0.0, 0.2])
    result = exercise!(rng, state; sweeps=15, trials=3)
    @test result.moves > 10 && result.worst < 1e-9
    @test PPL.refresh_products!(state) < 1e-10

    yb = Int.(rand(rng, 40) .< 0.5)
    plan2 = TWO(; X, Z) | (; y=yb)
    problem2 = PPL.prepare_coordinate_conditional(build_kernel(plan2), plan2, zeros(7))
    @test length(problem2.products) == 2
    @test [(block.range, length(block.cells)) for block in problem2.blocks] == [(1:4, 1), (5:7, 1)]
    state2 = PPL.coordinate_state(problem2, 0.2 .* randn(rng, 7))
    result2 = exercise!(rng, state2; sweeps=15, trials=3)
    @test result2.moves > 10 && result2.worst < 1e-9
    @test PPL.refresh_products!(state2) < 1e-10
end

@testset "a moved coefficient reads its own prior and Jacobian cells" begin
    rng = Xoshiro(11)
    X = randn(rng, 30, 5)
    y = Int.(rand(rng, 30) .< 0.5)
    # A scale shared by the vector's prior: the scale's own coordinate reads
    # every term, a coefficient reads its likelihood and one prior cell.
    plan = HIERARCHICAL(; X) | (; y)
    problem = PPL.prepare_coordinate_conditional(build_kernel(plan), plan, zeros(6))
    @test length(problem.products) == 1
    @test [(block.range, length(block.cells)) for block in problem.blocks] == [(2:6, 1)]
    state = PPL.coordinate_state(problem, 0.3 .* randn(rng, 6))
    result = exercise!(rng, state; sweeps=15, trials=3)
    @test result.moves > 10 && result.worst < 1e-9
    @test PPL.refresh_products!(state) < 1e-10

    # A constrained vector: its product is recomputed, and both its prior and
    # its Jacobian are read at the moved cell.
    plan2 = POSITIVE(; X) | (; y)
    problem2 = PPL.prepare_coordinate_conditional(build_kernel(plan2), plan2, zeros(6))
    @test isempty(problem2.products)
    @test [(block.range, length(block.cells)) for block in problem2.blocks] == [(2:6, 2)]
    state2 = PPL.coordinate_state(problem2, 0.3 .* randn(rng, 6))
    result2 = exercise!(rng, state2; sweeps=15, trials=3)
    @test result2.moves > 10 && result2.worst < 1e-9
end

@testset "a coefficient trial does not materialize the product" begin
    rng = Xoshiro(3)
    n = 4000
    X = randn(rng, n, 3)
    plan = LOGISTIC(; X, sigma=2.0) | (; y=Int.(rand(rng, n) .< 0.5))
    problem = PPL.prepare_coordinate_conditional(build_kernel(plan), plan, zeros(3))
    state = PPL.coordinate_state(problem, [0.1, -0.2, 0.05])
    trial(state) = PPL.coordinate_logdensity(state, 2, 0.3)
    trial(state)
    # The shifted product feeds the likelihood without a vector of its own.
    @test (@allocated trial(state)) < 8n
    @test exercise!(rng, state; sweeps=2, trials=2).worst < 1e-8
end

@testset "a program without a product needs no cache" begin
    rng = Xoshiro(9)
    plan = PLAIN(;) | (; y=[0.3, -0.1, 0.8, 0.4])
    problem = PPL.prepare_coordinate_conditional(build_kernel(plan), plan, zeros(2))
    @test isempty(problem.products)
    @test problem.blocks === ()
    state = PPL.coordinate_state(problem, [0.2, -0.3])
    @test state.products === ()
    result = exercise!(rng, state; sweeps=10, trials=3)
    @test result.moves > 3 && result.worst < 1e-9
    @test PPL.refresh_products!(state) == 0.0
end

@testset "the conditional program does not grow with the data" begin
    inventory(ex) = ex isa Expr ? 1 + sum(inventory, ex.args; init=0) : 1
    inventory(program::ReactiveKernels._PlateProgram) = inventory(program.ast)
    sizes = Int[]
    for (n, d) in ((5, 2), (60, 9), (300, 40))
        rng = Xoshiro(n)
        X = randn(rng, n, d)
        plan = LOGISTIC(; X, sigma=1.5) | (; y=Int.(rand(rng, n) .< 0.5))
        problem = PPL.prepare_coordinate_conditional(build_kernel(plan), plan, zeros(d))
        push!(sizes, inventory(code_expr(problem.conditional)) +
            inventory(code_expr(only(problem.blocks).kernel)))
        state = PPL.coordinate_state(problem, 0.1 .* randn(rng, d))
        @test exercise!(rng, state; sweeps=1, trials=2).worst < 1e-9
    end
    println("COORDINATE CONDITIONAL NODE INVENTORY ", sizes)
    # Inventories may differ by folded constants, never by the data's extent.
    @test maximum(sizes) < 2minimum(sizes)
    @test sizes[3] <= sizes[2] + 8
end

@testset "free integer parameters are not combined yet" begin
    model = @rkppl begin
        p ~ Beta(2, 3)
        c ~ Binomial(n, p)
    end
    plan = model(; n=4) | (;)
    @test_throws ArgumentError PPL.prepare_coordinate_conditional(
        build_kernel(plan), plan, zeros(1))
end
end
