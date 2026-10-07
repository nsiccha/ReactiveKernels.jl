using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

# A dotted `@plate` cell broadcasts over its own iteration's values, as the
# Julia loop it states does. A response holding one array per index (ragged,
# possibly empty) is observed entry by entry inside a retained group plate.

ReactiveKernels.@kernel _cba_reader(t, dose, log_k) = begin
    cells = ReactiveKernels.plate(t, dose, Ref(log_k)) do ti, di, lk
        di .* exp.(-exp(lk) .* ti)
    end
    return cells
end

function _cba_data(sizes)
    n = length(sizes)
    x = [Float64[0.1 * i + 0.3 * j for j in 1:k] for (i, k) in enumerate(sizes)]
    y = [0.3 .+ 0.5 .* sin.(3 .* xi .+ i) for (i, xi) in enumerate(x)]
    counts = [[mod(3i + j, 4) for j in 1:k] for (i, k) in enumerate(sizes)]
    w = [Float64[0.5 + 0.25 * j for j in 1:k] for k in sizes]
    dose = collect(range(1.0, 2.0; length = n))
    mu = collect(range(-0.3, 0.4; length = n))
    return (; x, y, counts, w, dose, mu)
end

# The observation cell and its per-entry oracle for each kind; parameters
# `a`, `b`, `sigma` are declared in every program.
function _cba_case(kind)
    kind === :reader && return (
        :(loc = _cba_reader(x, dose, a)),
        :(y[i] .~ Normal.(loc[i], sigma)),
        (d, q, i) -> logpdf.(Normal.(d.dose[i] .* exp.(-exp(q.a) .* d.x[i]), q.sigma), d.y[i]))
    kind === :data && return (nothing,
        :(y[i] .~ Normal.(x[i], sigma)),
        (d, q, i) -> logpdf.(Normal.(d.x[i], q.sigma), d.y[i]))
    kind === :computed && return (nothing,
        :(y[i] .~ Normal.(a .+ b .* x[i], sigma)),
        (d, q, i) -> logpdf.(Normal.(q.a .+ q.b .* d.x[i], q.sigma), d.y[i]))
    kind === :local && return (nothing,
        Expr(:block, :(m = a .+ b .* x[i]), :(y[i] .~ Normal.(m, sigma))),
        (d, q, i) -> logpdf.(Normal.(q.a .+ q.b .* d.x[i], q.sigma), d.y[i]))
    kind === :group_scalar && return (nothing,
        :(y[i] .~ Normal.(mu[i] + a, sigma)),
        (d, q, i) -> logpdf.(Normal(d.mu[i] + q.a, q.sigma), d.y[i]))
    kind === :shared && return (nothing,
        :(y[i] .~ Normal.(a, sigma)),
        (d, q, i) -> logpdf.(Normal(q.a, q.sigma), d.y[i]))
    kind === :poisson && return (nothing,
        :(counts[i] .~ Poisson.(exp.(a .+ b .* x[i]))),
        (d, q, i) -> logpdf.(Poisson.(exp.(q.a .+ q.b .* d.x[i])), d.counts[i]))
    kind === :weighted && return (nothing,
        :(y[i] .~ weighted.(Normal.(a .+ x[i], sigma), w[i])),
        (d, q, i) -> d.w[i] .* logpdf.(Normal.(q.a .+ d.x[i], q.sigma), d.y[i]))
    kind === :censored && return (nothing,
        :(y[i] .~ censored.(Normal.(a .+ b .* x[i], sigma), -0.1, 0.7)),
        (d, q, i) -> logpdf.(censored.(Normal.(q.a .+ q.b .* d.x[i], q.sigma), -0.1, 0.7),
            clamp.(d.y[i], -0.1, 0.7)))
    error("unknown kind $kind")
end

function _cba_fixture(kind, sizes; unbound = nothing)
    d = _cba_data(sizes)
    kind === :censored && (d = merge(d, (; y = [clamp.(v, -0.1, 0.7) for v in d.y])))
    definition, cell, entry = _cba_case(kind)
    response = kind === :poisson ? :counts : :y
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
    end
    definition === nothing || push!(ast.args, definition)
    push!(ast.args, Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
        Expr(:for, :(i = eachindex($response)), cell isa Expr && cell.head === :block ?
            cell : Expr(:block, cell))))
    data = Dict(pairs(d))
    delete!(data, response === :y ? :counts : :y)
    unbound === nothing &&
        (unbound = lower_rkppl(ast, data; conditioned = (response,), mod = @__MODULE__))
    bound = bind_data(unbound, data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; a = 0.25, b = -0.3, sigma = 0.8))
    pointwise(v) = [entry(d, constrain(built.layout, v), i) for i in eachindex(sizes)]
    function oracle(v)
        q = constrain(built.layout, v)
        logpdf(Normal(), q.a) + logpdf(Normal(), q.b) + logpdf(Exponential(1.0), q.sigma) +
            logjac(built.layout, v) + sum(sum(pw; init = 0.0) for pw in pointwise(v); init = 0.0)
    end
    return (; kind, response, data, unbound, bound, built, u, oracle, pointwise)
end

function _cba_fd(f, u)
    h = cbrt(eps(Float64))
    [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end for i in eachindex(u)]
end

_cba_structure(program) = [(e.kind, e.depth) for e in recipe_inventory(program)
    if e.kind !== :ordinary]

@testset "dotted plate cells over one array per index: native values and gradients" begin
    for kind in (:reader, :data, :computed, :local, :group_scalar, :shared,
            :poisson, :weighted, :censored)
        for sizes in ((3, 2, 0, 1), (0, 4, 1, 5, 2, 3, 0, 2))
            fx = _cba_fixture(kind, sizes)
            saved = deepcopy(fx.data)
            sampler = prepare_sampler(fx.built, fx.bound, fx.u;
                backend = AutoEnzyme(; mode = Enzyme.Reverse))
            for shift in (0.0, 0.07)
                u = fx.u .+ shift
                value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
                @test value ≈ fx.oracle(u) rtol = 1e-10
                @test gradient ≈ _cba_fd(fx.oracle, u) rtol = 1e-5 atol = 1e-7
                pw = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), u)
                expected = fx.pointwise(u)
                # The pointwise result keeps the response's shape: one array of
                # observation densities per index.
                @test length(pw[fx.response]) == length(sizes)
                @test all(length.(pw[fx.response]) .== collect(sizes))
                @test all(pw[fx.response] .≈ expected)
                lik = Base.invokelatest(prepare_query(fx.built, fx.bound, :likelihood), u)
                @test lik ≈ sum(sum(p; init = 0.0) for p in expected) rtol = 1e-10
            end
            @test isequal(fx.data, saved)
        end
    end
end

@testset "dotted plate cells over arrays retain a nested group plate" begin
    small = _cba_fixture(:computed, (3, 2, 0, 1))
    large = _cba_fixture(:computed, (0, 4, 1, 5, 2, 3, 0, 2, 6, 1, 1, 3))
    # One observation plate per index, nested in the group plate (RK native
    # nested plates); the computed location is its own retained cell.
    @test (:plate, 1) in _cba_structure(small.built.spec)
    @test _cba_structure(small.built.spec) == _cba_structure(large.built.spec)
    query(fx) = prepare_query(fx.built, fx.bound, :likelihood)
    @test _cba_structure(query(small)) == _cba_structure(query(large))
    @test count(==((:plate, 1)), _cba_structure(query(small))) == 1
end

@testset "dotted plate cells over arrays: empty domains and rebinding" begin
    m = @rkppl begin
        a ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(y)
            y[i] .~ Normal.(a .+ x[i], sigma)
        end
    end
    u = [0.2, log(0.8)]
    for (x, y) in (([Float64[], Float64[]], [Float64[], Float64[]]),
            (Vector{Float64}[], Vector{Float64}[]))
        plan = m(; x) | (; y)
        built = build_kernel(plan)
        @test Base.invokelatest(prepare_query(built, plan, :likelihood), u) == 0.0
    end
    # A literal loop over every index observes each index's whole array.
    literal = @rkppl begin
        a ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
        @plate for i in 1:4
            y[i] .~ Normal.(a .+ x[i], sigma)
        end
    end
    d = _cba_data((3, 2, 0, 1))
    plan = literal(; x = d.x) | (; y = d.y)
    built = build_kernel(plan)
    @test Base.invokelatest(prepare_query(built, plan, :likelihood), u) ≈
        sum(sum(logpdf.(Normal.(0.2 .+ d.x[i], 0.8), d.y[i]); init = 0.0) for i in 1:4)
    # A names-only plan binds arrays per index of any lengths.
    fx = _cba_fixture(:computed, (3, 2, 0, 1))
    other = _cba_fixture(:computed, (1, 0, 4); unbound = fx.unbound)
    lik = Base.invokelatest(prepare_query(other.built, other.bound, :likelihood), other.u)
    @test lik ≈ sum(sum(p; init = 0.0) for p in other.pointwise(other.u)) rtol = 1e-10
end

@testset "observing arrays per index outside a dotted cell follows Julia" begin
    y = [[0.7, 0.3, 0.1], [1.4, 0.6], Float64[]]
    whole = @rkppl begin
        a ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
        y .~ Normal.(a, sigma)
    end
    # refused: `y .~ D.(…)` broadcasts a univariate distribution over the
    # arrays of `y`, which takes no array argument in Julia (standard-Julia
    # semantics, standing @rkppl principle 3)
    @test_throws ContractValidationError whole() | (; y)
    @test_throws "dotted `@plate` cell" whole() | (; y)
    undotted = @rkppl begin
        a ~ Normal(0, 1)
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(y)
            y[i] ~ Normal(a, sigma)
        end
    end
    # refused: an undotted cell observes `y[i]` itself, an array, with a
    # univariate distribution (standard-Julia semantics, principle 3)
    @test_throws "`y[i] .~ D.(…)`" undotted() | (; y)
    data_location = @rkppl begin
        sigma ~ Exponential(1.0)
        y .~ Normal.(x, sigma)
    end
    # refused: the same Julia broadcast over a data operand holding arrays
    @test_throws "holds one array per entry" data_location(; x = y) | (; y)
    mismatched = @rkppl begin
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(y)
            y[i] .~ Normal.(x[i], sigma)
        end
    end
    # refused: `x[i]` and `y[i]` must broadcast together, as Julia requires
    # (DimensionMismatch)
    @test_throws "does not broadcast" mismatched(; x = [[0.1, 0.2], [0.3, 0.4], Float64[]]) | (; y)
end
