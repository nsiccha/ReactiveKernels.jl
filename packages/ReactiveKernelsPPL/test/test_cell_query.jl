using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

# One cell of an observation plate: one iteration of an `@plate for` loop
# (one group's observations), or one entry of an elementwise observation
# (snag `rkppl-per-subjec-a3d6e6d8`, user decision `06z2prg`, option 1a).

# Grouped exponential-decay curves: per-group predictors with a correlated
# non-centred group block, one loop iteration per group observing that group's
# series (ragged, one group empty).
const _CQ_GROUPED = quote
    sigma ~ Exponential(1)
    a ~ Normal(0, 1)
    bw ~ Normal(0, 1)
    v0 ~ Normal(0, 1)
    sd[1:2] .~ Exponential.(1)
    L ~ LKJCholesky(2, 2.0)
    Z[levels(group), 1:2] .~ Normal.(0, 1)
    F = sd .* L
    R = Z * F'
    log_k = a .+ bw .* w .+ R[group, 1]
    log_v = v0 .+ R[group, 2]
    @plate for i in eachindex(y)
        k = exp(log_k[i])
        v = exp(log_v[i])
        y[i] .~ Normal.(amount[i] / v .* exp.(-k .* t[i]), sigma)
    end
end

function _cq_grouped_data()
    t = [[0.5, 1.0, 2.0], [0.5, 3.0], Float64[], [1.0, 2.0, 4.0, 8.0], [0.25]]
    y = [[1.8, 1.5, 1.1], [2.0, 0.8], Float64[], [1.6, 1.2, 0.7, 0.3], [2.4]]
    (; group = ["g1", "g2", "g3", "g4", "g5"], w = [0.1, -0.2, 0.3, 0.0, 0.5],
       amount = [10.0, 8.0, 12.0, 9.0, 11.0], t, y)
end

function _cq_bound(ast, data, conditioned)
    plan = lower_rkppl(ast, data; conditioned, mod = @__MODULE__)
    bound = bind_data(plan, data)
    bound, build_kernel(bound)
end

_cq_central(f, u, j; h = 1e-6) =
    (f(setindex!(copy(u), u[j] + h, j)) - f(setindex!(copy(u), u[j] - h, j))) / (2h)

@testset "cell query: one group of an @plate for loop" begin
    data = _cq_grouped_data()
    bound, built = _cq_bound(_CQ_GROUPED, data, (:y,))
    u = collect(range(-0.4, 0.5; length = built.layout.total))
    pointwise = Base.invokelatest(prepare_query(built, bound, :pointwise), u)
    likelihood = Base.invokelatest(prepare_query(built, bound, :likelihood), u)
    q = prepare_cell_query(built, bound, :y)
    @test q isa CellQuery
    cells = [q(u, i) for i in eachindex(data.y)]
    @test cells ≈ [sum(x; init = 0.0) for x in pointwise.y]
    @test q(u, 3) == 0.0
    @test sum(cells) ≈ likelihood
    @test_throws BoundsError q(u, length(data.y) + 1)

    # The gradient of one group's density is the full likelihood gradient on
    # that group's block, and zero on every other group's block.
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    s = prepare_cell_sampler(built, bound, :y, u; backend)
    full = prepare_ad(prepare_query(built, bound, :likelihood), backend, u;
        active = :unconstrained)
    _, gfull = Base.invokelatest(ad_value_and_gradient, full, u)
    names = coordinate_names(built.layout)
    block(i) = findall(n -> startswith(String(n), "Z.$i."), names)
    for i in (1, 4)
        g = similar(u)
        value, _ = cell_value_and_gradient!(s, g, u, i)
        @test value ≈ cells[i]
        @test g[block(i)] ≈ gfull[block(i)]
        @test all(iszero, g[block(j)] for j in eachindex(data.y) if j != i)
        for j in (2, first(block(i)))
            @test g[j] ≈ _cq_central(x -> q(x, i), u, j) rtol = 1e-5
        end
    end
    # Gradients leave the bound per-group arrays and later values unchanged.
    @test data == _cq_grouped_data()
    @test [q(u, i) for i in eachindex(data.y)] ≈ cells
end

@testset "cell query: a group's cost does not grow with the number of groups" begin
    # The predictors `log_k`, `log_v` and the group block `R` are computed at the
    # queried group only (increment 1b), so a cell allocates the same at 10
    # groups as at 2,000.
    function data(n)
        (; group = ["g$j" for j in 1:n], w = collect(range(-0.3, 0.3; length = n)),
           amount = fill(10.0, n), t = [[0.5, 1.0, 2.0] for _ in 1:n],
           y = [[1.8, 1.5, 1.1] for _ in 1:n])
    end
    measured = map((10, 2_000)) do n
        bound, built = _cq_bound(_CQ_GROUPED, data(n), (:y,))
        q = prepare_cell_query(built, bound, :y)
        u = collect(range(-0.4, 0.5; length = built.layout.total))
        pointwise = Base.invokelatest(prepare_query(built, bound, :pointwise), u)
        q(u, 2)
        (; value = q(u, 2), reference = sum(pointwise.y[2]), bytes = @allocated q(u, 2))
    end
    @test all(m -> m.value ≈ m.reference, measured)
    @test measured[1].bytes == measured[2].bytes
end

@testset "cell query: elementwise and scalar-cell observations" begin
    x = [0.2, -0.4, 1.1, 0.7]
    y = [0.5, -0.1, 1.8, 1.0]
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        s ~ Exponential(1)
        y .~ Normal.(a .+ b .* x, s)
    end
    bound, built = _cq_bound(ast, (; x, y), (:y,))
    u = [0.3, -0.2, 0.1]
    pointwise = Base.invokelatest(prepare_query(built, bound, :pointwise), u)
    q = prepare_cell_query(built, bound, (:y,))
    @test [q(u, i) for i in eachindex(y)] ≈ pointwise.y

    looped = quote
        a ~ Normal(0, 1)
        s ~ Exponential(1)
        @plate for i in eachindex(y)
            y[i] ~ Normal(a * x[i], s)
            z[i] ~ Normal(a + x[i], 2s)
        end
    end
    z = [0.1, 0.4, -0.6, 0.9]
    bound, built = _cq_bound(looped, (; x, y, z), (:y, :z))
    u = [0.4, -0.3]
    pointwise = Base.invokelatest(prepare_query(built, bound, :pointwise), u)
    both = prepare_cell_query(built, bound, (:y, :z))
    @test [both(u, i) for i in eachindex(y)] ≈ pointwise.y .+ pointwise.z
    only_z = prepare_cell_query(built, bound, :z)
    @test [only_z(u, i) for i in eachindex(z)] ≈ pointwise.z
end

# An observation in an `@plate for` cell whose argument the cell computes
# (`m = f(w[i], la[i])`) reads that argument's plate at the cell's own index,
# so a cell query composes the plate and calls `f` once per observation (snag
# `prepare-cell-que-30f4f1bf`: the plate iterated the resolved
# `Base.eachindex(y1)`, the response range kept `eachindex(y1)`, and a whole
# gather between them ran the plate at every index).
const _CQ_CALLS = Ref(0)
_cq_scaled(w, rate) = (_CQ_CALLS[] += 1; w * exp(rate))

@testset "cell query: an in-cell observation's computed argument runs at that cell" begin
    n = 9
    w = collect(range(0.3, 1.1; length = n))
    x = collect(range(-0.4, 0.6; length = n))
    data = (; w, x, y1 = w .+ 0.1, y2 = 2 .* w)
    u = [0.2, 0.1]
    m = w .* exp.(u[1] .+ x)
    s = exp(u[2])
    one_observation(iterator) = quote
        a ~ Normal(0, 1)
        s ~ Exponential(1.0)
        la = a .+ x
        @plate for i in $iterator
            m = _cq_scaled(w[i], la[i])
            y1[i] ~ Normal(m, s)
        end
        y2 .~ Normal.(2 .* w, s)
    end
    two_observations = quote
        a ~ Normal(0, 1)
        s ~ Exponential(1.0)
        la = a .+ x
        @plate for i in eachindex(y1)
            m = _cq_scaled(w[i], la[i])
            y1[i] ~ Normal(m, s)
            y2[i] ~ Normal(2 * m, s)
        end
    end
    y1_density(i) = logpdf(Normal(m[i], s), data.y1[i])
    y2_density(i) = logpdf(Normal(2 * m[i], s), data.y2[i])
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    for (ast, observations, calls, density) in (
            (one_observation(:(eachindex(y1))), (:y1,), 1, y1_density),
            (one_observation(:(axes(y1, 1))), (:y1,), 1, y1_density),
            (two_observations, (:y1,), 1, y1_density),
            # Each observation's argument has its own plate: one call each.
            (two_observations, (:y1, :y2), 2, i -> y1_density(i) + y2_density(i)))
        bound, built = _cq_bound(ast, data, (:y1, :y2))
        pointwise = Base.invokelatest(prepare_query(built, bound, :pointwise), u)
        q = prepare_cell_query(built, bound, observations)
        cells = [q(u, i) for i in 1:n]
        @test cells ≈ [density(i) for i in 1:n] rtol = 1e-12
        @test cells ≈ sum(vec(getfield(pointwise, o)) for o in observations) rtol = 1e-12
        for i in (1, 5, 9)
            _CQ_CALLS[] = 0
            q(u, i)
            @test _CQ_CALLS[] == calls
        end
        sampler = prepare_cell_sampler(built, bound, observations, u; backend)
        for i in (2, 7)
            g = similar(u)
            value, _ = cell_value_and_gradient!(sampler, g, u, i)
            @test value ≈ cells[i] rtol = 1e-12
            @test g ≈ [_cq_central(v -> q(v, i), u, j) for j in eachindex(u)] rtol = 1e-6
        end
    end
    @test data.w == collect(range(0.3, 1.1; length = n))
    @test data.y1 == data.w .+ 0.1
end

@testset "cell query: refusals" begin
    data = _cq_grouped_data()
    bound, built = _cq_bound(_CQ_GROUPED, data, (:y,))
    @test_throws ContractValidationError prepare_cell_query(built, bound, :amount)
    @test_throws ContractValidationError prepare_cell_query(built, bound, ())
    scalar = quote
        theta ~ Beta(1, 1)
        k ~ Binomial(n, theta)
    end
    plan = lower_rkppl(scalar, (; n = 5, k = 2); conditioned = (:k,), mod = @__MODULE__)
    sbound = bind_data(plan, (; n = 5, k = 2))
    @test_throws ContractValidationError prepare_cell_query(build_kernel(sbound), sbound, :k)
end
