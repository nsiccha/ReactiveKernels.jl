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

_cq_direct_compact_value(q, base, active, indices, cell) =
    q(ReactiveKernelsPPL._CompactCellPoint(base, active,
        ReactiveKernelsPPL._compact_cell_slots(length(base), indices)), cell)

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
    compact = prepare_compact_cell_sampler(built, bound, :y, u,
        sort!([2; block(1)]);
        backend = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse)))
    for i in (1, 4)
        g = similar(u)
        value, _ = cell_value_and_gradient!(s, g, u, i)
        @test value ≈ cells[i]
        @test g[block(i)] ≈ gfull[block(i)]
        @test all(iszero, g[block(j)] for j in eachindex(data.y) if j != i)
        for j in (2, first(block(i)))
            @test g[j] ≈ _cq_central(x -> q(x, i), u, j) rtol = 1e-5
        end
        indices = sort!([2; block(i)])
        base = copy(u)
        active = u[indices]
        active[1] += 0.1
        overlaid = copy(u)
        overlaid[indices] = active
        compact_gradient = similar(active)
        compact_value, returned = cell_value_and_gradient!(
            compact, compact_gradient, base, active, indices, i)
        full_gradient = similar(u)
        full_value, _ = cell_value_and_gradient!(s, full_gradient, overlaid, i)
        @test compact_value ≈ full_value ≈ q(overlaid, i)
        @test returned === compact_gradient
        @test compact_gradient ≈ full_gradient[indices]
        @test base == u
        @test active == overlaid[indices]
        if i == 1
            # BRM's direct-Enzyme workaround differentiates the compact η
            # vector and then evaluates the cell again for its value.
            direct_gradient = zeros(length(active))
            Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse),
                _cq_direct_compact_value, Enzyme.Active, Enzyme.Const(q),
                Enzyme.Const(base), Enzyme.Duplicated(active, direct_gradient),
                Enzyme.Const(indices), Enzyme.Const(i))
            direct_value = _cq_direct_compact_value(q, base, active, indices, i)
            @test compact_value ≈ direct_value
            @test compact_gradient ≈ direct_gradient
        end
    end
    @test_throws ReactiveKernelsPPL.ContractValidationError cell_value_and_gradient!(
        compact, zeros(2), u, zeros(2), [2, 2], 1)
    # The sampler maps the previous call's selection; a caller that changes its
    # `indices` in place between calls gets the new selection, and a refused
    # selection leaves the previous one in force.
    let indices = sort!([2; block(1)]), base = copy(u)
        function expected(indices, cell = 1)
            overlaid = copy(base)
            overlaid[indices] .+= 0.1
            g = similar(u)
            value, _ = cell_value_and_gradient!(s, g, overlaid, cell)
            value, g[indices]
        end
        compact_gradient = zeros(length(indices))
        # The loop above last selected group 4's block; the refusal kept it.
        previous = sort!([2; block(4)])
        value, _ = cell_value_and_gradient!(compact, compact_gradient, base,
            base[previous] .+ 0.1, previous, 4)
        @test value ≈ expected(previous, 4)[1]
        @test compact_gradient ≈ expected(previous, 4)[2]
        value, _ = cell_value_and_gradient!(compact, compact_gradient, base,
            base[indices] .+ 0.1, indices, 1)
        @test value ≈ expected(indices)[1]
        @test compact_gradient ≈ expected(indices)[2]
        indices[1] = 1
        value, _ = cell_value_and_gradient!(compact, compact_gradient, base,
            base[indices] .+ 0.1, indices, 1)
        @test value ≈ expected(indices)[1]
        @test compact_gradient ≈ expected(indices)[2]
        indices[1] = 2
        value, _ = cell_value_and_gradient!(compact, compact_gradient, base,
            base[indices] .+ 0.1, indices, 1)
        @test value ≈ expected(indices)[1]
        @test compact_gradient ≈ expected(indices)[2]
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
        uview = view(u, :)
        pointwise = Base.invokelatest(prepare_query(built, bound, :pointwise), u)
        q(u, 2); q(uview, 2)
        names = coordinate_names(built.layout)
        indices = sort!([2; findall(name -> startswith(String(name), "Z.2."), names)])
        active = u[indices]
        compact = prepare_compact_cell_sampler(built, bound, :y, u, indices;
            backend = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse)))
        compact_gradient = similar(active)
        compact_value, _ = cell_value_and_gradient!(
            compact, compact_gradient, u, active, indices, 2)
        (; value = q(u, 2), viewed = q(uview, 2),
           reference = sum(pointwise.y[2]), compact_value,
           bytes = (@allocated q(u, 2)), view_bytes = (@allocated q(uview, 2)),
           compact_bytes = (@allocated cell_value_and_gradient!(
               compact, compact_gradient, u, active, indices, 2)))
    end
    @test all(m -> m.value ≈ m.reference, measured)
    @test all(m -> m.viewed ≈ m.reference, measured)
    @test measured[1].bytes == measured[2].bytes
    @test measured[1].view_bytes == measured[2].view_bytes
    @test all(m -> m.compact_value ≈ m.reference, measured)
    @test measured[2].compact_bytes <= measured[1].compact_bytes + 1024
    # The group row of `R = Z * F'` reads `Z` through the compact overlay, which
    # is not strided; RK's native product computes it with its own loop, where
    # LinearAlgebra's generic `mul!` allocated 88 KB per gradient on
    # Julia 1.10 (snag `compact-cell-gra-ebf0a114`).
    @test all(m -> m.compact_bytes < 8_192, measured)
end

# BRM's NLME emission for a caller-provided scalar law: a kernel plate writes
# each subject's prediction, and the observation plate reads it per subject
# through `LogDensity` (snag `brm-s-nlme-subje-f6f0efd3`). A subject's cell
# runs the kernel plate, and the predictors it reads, at that subject only, as
# the same program observed by `Normal` does.
ReactiveKernels.@kernel _cq_scalar_law(value, location, scale) = begin
    residual = (value - location) / scale
    density = -0.5 * log(2 * pi) - log(scale) - 0.5 * residual * residual
    return density
end
_cq_flatten_cells(cells) = reduce(vcat, cells; init = Float64[])

_cq_kernel_plate(observation) = quote
    sigma ~ Exponential(1)
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    s1 ~ Exponential(1)
    s2 ~ Exponential(1)
    z1[levels(subject)] .~ Normal.(0, 1)
    z2[levels(subject)] .~ Normal.(0, 1)
    log_CL = a .+ s1 .* z1[subject]
    log_V = b .+ s2 .* z2[subject]
    @plate for i = eachindex(log_CL)
        CL = exp(log_CL[i])
        Vc = exp(log_V[i])
        pred_cell = (dose[i] / Vc) .* exp.(-(CL / Vc) .* t[i])
        pred_cells[i] = pred_cell
        pred_argument_dv_1[i] = pred_cell
    end
    pred = _cq_flatten_cells(pred_cells)
    @plate for i = eachindex(dv)
        $observation
    end
end

function _cq_pk_data(n)
    t = [[0.5, 1.0, 2.0, 4.0][1:mod1(j, 4)] for j in 1:n]
    (; subject = ["s$j" for j in 1:n], dose = fill(100.0, n), t,
       dv = [5 .* exp.(-0.15 .* tj) .* (1 .+ 0.05 .* sin.(j .+ tj))
             for (j, tj) in enumerate(t)])
end

@testset "cell query: a caller-owned law reads a kernel plate's row" begin
    law = :(dv[i] .~ LogDensity.(_cq_scalar_law, pred_argument_dv_1[i], sigma))
    normal = :(dv[i] .~ Normal.(pred_argument_dv_1[i], sigma))
    data = _cq_pk_data(5)
    bound, built = _cq_bound(_cq_kernel_plate(law), data, (:dv,))
    nbound, nbuilt = _cq_bound(_cq_kernel_plate(normal), data, (:dv,))
    @test coordinate_names(built.layout) == coordinate_names(nbuilt.layout)
    u = collect(range(-0.4, 0.5; length = built.layout.total))
    pointwise = Base.invokelatest(prepare_query(built, bound, :pointwise), u)
    q = prepare_cell_query(built, bound, :dv)
    nq = prepare_cell_query(nbuilt, nbound, :dv)
    cells = [q(u, i) for i in eachindex(data.dv)]
    @test cells ≈ [sum(x; init = 0.0) for x in pointwise.dv]
    @test cells ≈ [nq(u, i) for i in eachindex(data.dv)]
    @test_throws BoundsError q(u, length(data.dv) + 1)

    names = coordinate_names(built.layout)
    block(i) = findall(n -> n in (Symbol("z1.$i"), Symbol("z2.$i")), names)
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    s = prepare_cell_sampler(built, bound, :dv, u; backend)
    full = prepare_ad(prepare_query(built, bound, :likelihood), backend, u;
        active = :unconstrained)
    _, gfull = Base.invokelatest(ad_value_and_gradient, full, u)
    compact = prepare_compact_cell_sampler(built, bound, :dv, u,
        sort!([1:5; block(1)]);
        backend = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse)))
    for i in (1, 4)
        g = similar(u)
        value, _ = cell_value_and_gradient!(s, g, u, i)
        @test value ≈ cells[i]
        @test g[block(i)] ≈ gfull[block(i)]
        @test all(iszero, g[block(j)] for j in eachindex(data.dv) if j != i)
        for j in (1, 2, first(block(i)))
            @test g[j] ≈ _cq_central(x -> q(x, i), u, j) rtol = 1e-5
        end
        indices = sort!([1:5; block(i)])
        active = u[indices]
        compact_gradient = similar(active)
        compact_value, _ = cell_value_and_gradient!(
            compact, compact_gradient, u, active, indices, i)
        @test compact_value ≈ value
        @test compact_gradient ≈ g[indices]
    end
    @test data == _cq_pk_data(5)

    # One subject's cell allocates the same at 10 subjects as at 2,000.
    measured = map((10, 2_000)) do n
        data = _cq_pk_data(n)
        bound, built = _cq_bound(_cq_kernel_plate(law), data, (:dv,))
        u = collect(range(-0.4, 0.5; length = built.layout.total))
        q = prepare_cell_query(built, bound, :dv)
        names = coordinate_names(built.layout)
        indices = sort!([1:5; findall(n -> n in (Symbol("z1.2"), Symbol("z2.2")), names)])
        active = u[indices]
        compact = prepare_compact_cell_sampler(built, bound, :dv, u, indices;
            backend = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse)))
        compact_gradient = similar(active)
        cell_value_and_gradient!(compact, compact_gradient, u, active, indices, 2)
        q(u, 2)
        (; bytes = (@allocated q(u, 2)),
           compact_bytes = (@allocated cell_value_and_gradient!(
               compact, compact_gradient, u, active, indices, 2)))
    end
    @test measured[1].bytes == measured[2].bytes
    @test measured[1].compact_bytes == measured[2].compact_bytes
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

@testset "cell query: computed scale and missing observation cells" begin
    n = 9
    w = collect(range(0.3, 1.1; length = n))
    x = collect(range(-0.4, 0.6; length = n))
    u = [0.2]
    scale_model = quote
        a ~ Normal(0, 1)
        la = a .+ x
        @plate for i in eachindex(y)
            s = _cq_scaled(w[i], la[i])
            y[i] ~ Normal(x[i], s)
        end
    end
    complete = (; w, x, y = w .+ 0.1)
    bound, built = _cq_bound(scale_model, complete, (:y,))
    q = prepare_cell_query(built, bound, :y)
    for i in (1, 5, 9)
        _CQ_CALLS[] = 0
        @test q(u, i) ≈ logpdf(Normal(x[i], w[i] * exp(u[1] + x[i])), complete.y[i])
        @test _CQ_CALLS[] == 1
    end
    sampler = prepare_cell_sampler(built, bound, :y, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    g = similar(u)
    value, _ = cell_value_and_gradient!(sampler, g, u, 5)
    @test value ≈ q(u, 5)
    @test g[1] ≈ _cq_central(v -> q(v, 5), u, 1) rtol = 1e-6

    missing_y = Union{Missing,Float64}[complete.y...]
    missing_y[3] = missing
    incomplete = (; w, x, y = missing_y)
    location_model = quote
        a ~ Normal(0, 1)
        la = a .+ x
        @plate for i in eachindex(y)
            m = _cq_scaled(w[i], la[i])
            y[i] ~ Normal(m, 1.3)
        end
    end
    bound, built = _cq_bound(location_model, incomplete, (:y,))
    q = prepare_cell_query(built, bound, :y)
    for i in (1, 3, 5, 9)
        _CQ_CALLS[] = 0
        expected = ismissing(incomplete.y[i]) ? 0.0 :
            logpdf(Normal(w[i] * exp(u[1] + x[i]), 1.3), incomplete.y[i])
        @test q(u, i) ≈ expected
        @test _CQ_CALLS[] == (ismissing(incomplete.y[i]) ? 0 : 1)
    end
    sampler = prepare_cell_sampler(built, bound, :y, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    for i in (3, 5)
        g = similar(u)
        value, _ = cell_value_and_gradient!(sampler, g, u, i)
        @test value ≈ q(u, i)
        @test g[1] ≈ _cq_central(v -> q(v, i), u, 1) atol = 1e-10 rtol = 1e-6
    end
    @test ismissing(incomplete.y[3])
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

# The cell groups of a regional panel (snag `compact-cell-sam-d85d96ae`): a
# total reads its rows of a predictor whose design has a computed column, and
# an observation reads a mean with a per-row stratified effect. Each cell costs
# its own rows, however many rows the model has.
const _CQ_TOTALS = quote
    sigma_t ~ Exponential(1)
    beta[1:3] .~ Normal.(0, 1)
    tau ~ Exponential(1)
    u[levels(g)] .~ Normal.(0, 1)
    c0 ~ Normal(4, 1)
    c = lx .- c0
    X = hcat(one, c, year)
    log_w = X * beta .+ tau .* u[g]
    @plate for k in eachindex(total)
        total[k] ~ Normal(sum(exp.(log_w[rows[k]])), sigma_t * total[k])
    end
end

const _CQ_STRATA = quote
    sigma ~ Exponential(1)
    a ~ Normal(0, 1)
    ts[levels(s), 1:1] .~ Exponential.(1)
    zg[levels(g), 1:1] .~ Normal.(0, 1)
    @plate for i in eachindex(g)
        b[i, 1:1] = ts[s[i], :] .* zg[g[i], :]
    end
    mu = a .+ b[:, 1]
    xo .~ Normal.(mu[obs], sigma)
end

function _cq_panel_data(n; size = 4)
    rows = [collect(j:min(j + size - 1, n)) for j in 1:size:n]
    (; lx = 4.0 .+ 0.1 .* sin.(1:n), one = ones(n),
       year = collect(range(-1.0, 1.0; length = n)),
       g = [string("g", 1 + (j - 1) ÷ 2) for j in 1:n], rows,
       total = [10.0 + k for k in eachindex(rows)],
       s = [isodd(j) ? "a" : "b" for j in 1:n], obs = collect(1:2:n),
       xo = [0.1 * j for j in 1:2:n])
end

# The totals' design as a data matrix: the cell gradient, compact and dense.
const _CQ_TOTALS_DATA = quote
    sigma_t ~ Exponential(1)
    beta[1:3] .~ Normal.(0, 1)
    tau ~ Exponential(1)
    u[levels(g)] .~ Normal.(0, 1)
    log_w = X * beta .+ tau .* u[g]
    @plate for k in eachindex(total)
        total[k] ~ Normal(sum(exp.(log_w[rows[k]])), sigma_t * total[k])
    end
end

_cq_panel_data_matrix(n) = let d = _cq_panel_data(n)
    (; X = hcat(d.one, d.lx, d.year), d.g, d.rows, d.total)
end

@testset "cell query: totals over rows and stratified effects cost their own rows" begin
    backend = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
    function measure(ast, data, obs; gradients)
        map((40, 4_000)) do n
            bound, built = _cq_bound(ast, data(n), (obs,))
            q = prepare_cell_query(built, bound, obs)
            u = collect(range(-0.4, 0.5; length = built.layout.total))
            pointwise = Base.invokelatest(prepare_query(built, bound, :pointwise), u)
            q(u, 2)
            measured = (; value = q(u, 2), reference = getproperty(pointwise, obs)[2],
                bytes = (@allocated q(u, 2)))
            gradients || return measured
            dense = prepare_cell_sampler(built, bound, obs, u; backend)
            dense_value, dense_gradient = cell_value_and_gradient!(dense, similar(u), u, 2)
            indices = collect(1:6)
            active = u[indices]
            compact = prepare_compact_cell_sampler(built, bound, obs, u, indices; backend)
            compact_gradient = similar(active)
            compact_value, _ = cell_value_and_gradient!(
                compact, compact_gradient, u, active, indices, 2)
            (; measured..., dense_value, compact_value,
               parity = compact_gradient ≈ dense_gradient[indices],
               compact_bytes = (@allocated cell_value_and_gradient!(
                   compact, compact_gradient, u, active, indices, 2)))
        end
    end
    for (ast, data, obs, gradients) in (
            (_CQ_TOTALS, _cq_panel_data, :total, false),
            (_CQ_TOTALS_DATA, _cq_panel_data_matrix, :total, true),
            (_CQ_STRATA, _cq_panel_data, :xo, true))
        measured = measure(ast, data, obs; gradients)
        @test all(m -> m.value ≈ m.reference, measured)
        @test measured[1].bytes == measured[2].bytes
        gradients || continue
        @test all(m -> m.value ≈ m.dense_value ≈ m.compact_value, measured)
        @test all(m -> m.parity, measured)
        @test measured[1].compact_bytes == measured[2].compact_bytes
    end
end

# Each preparation used to generate the whole program twice: once to check that
# the binding generates the program the graph was built from, once to find the
# observation's plate. On a 224k-coordinate BRM program that was about 45 s per
# preparation of any one cell (snag `brm-s-nlme-cell-d70abd20`). A binding is
# now checked against a build once, and the plates are read from the program
# the build recorded.
@testset "cell query: another preparation does not generate the program again" begin
    data = _cq_grouped_data()
    plan = lower_rkppl(_CQ_GROUPED, data; conditioned = (:y,), mod = @__MODULE__)
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    u = collect(range(-0.4, 0.5; length = built.layout.total))
    pointwise = Base.invokelatest(prepare_query(built, bound, :pointwise), u)
    check() = @allocated ReactiveKernelsPPL._check_built_program(built, bound)
    # `prepare_query` above checked this binding: no check generates it again.
    q = prepare_cell_query(built, bound, :y)
    hit, again = check(), check()
    @test hit < 10_000
    @test again < 10_000
    @test [q(u, i) for i in eachindex(data.y)] ≈ [sum(x; init = 0.0) for x in pointwise.y]

    # Another binding of the same program (its own column table) is checked
    # once, on its first preparation.
    rebound = bind_data(plan, data)
    generate = @allocated ReactiveKernelsPPL._check_built_program(built, rebound)
    @test generate > 100 * hit
    @test (@allocated ReactiveKernelsPPL._check_built_program(built, rebound)) < 10_000

    # A binding of another program (one group fewer, so `Z` has other sizes) is
    # refused on every preparation, before and after the first one.
    fewer = map(x -> x isa AbstractVector ? x[1:4] : x, data)
    other = bind_data(plan, fewer)
    for _ in 1:2
        @test_throws ContractValidationError prepare_cell_query(built, other, :y)
        @test_throws ContractValidationError prepare_query(built, other, :pointwise)
    end
    # A binding checked against one build is checked against another.
    own = build_kernel(other)
    @test prepare_cell_query(own, other, :y) isa CellQuery
    @test_throws ContractValidationError prepare_cell_query(own, bound, :y)

    # A graph made outside `build_kernel` records no program: the plates are
    # found in the program the binding generates, as before.
    unrecorded = prepare_cell_query((; built.spec, built.layout), bound, :y)
    @test [unrecorded(u, i) for i in eachindex(data.y)] == [q(u, i) for i in eachindex(data.y)]

    # The record holds neither the binding's column table nor the program.
    record = ReactiveKernelsPPL._structural_cache(bound).entries[
        ReactiveKernelsPPL._checked_program_key(bound)]
    @test record.columns isa WeakRef && record.columns.value === bound.columns
    @test all(ref -> ref isa WeakRef, record.programs)
    @test any(ref -> ref.value === built.program, record.programs)
end

# Values a cell reads through gathers of model-level vectors (snag
# `on-arbeitsplatza-6065a1ef`): a group-effect predictor `a .+ b .* x .+ z[g]`
# (its coefficients as a typed coordinate view), a covariate completed from
# drawn entries `vcat(free, grouped)` and gathered onto the rows that read it,
# and a NegativeBinomial2 location gathered from a sum of that predictor at
# two row indices (one gathered value, two index vectors; RKPPL reads the
# location through a typed alias), as BRM emits mixture intensities. Each cell
# computes them at its own rows.
const _CQ_GATHERED = quote
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    s ~ Exponential(1)
    phi ~ Exponential(1)
    z[levels(g)] .~ Normal.(0, 1)
    zf[axes(free_rows, 1)] .~ Normal.(0, 1)
    zs[axes(share_rows, 1)] .~ Normal.(0, 1)
    yf = exp.(mf .+ s .* zf)
    ys = exp.(0.5 .* zs)
    completed = observed .+ mask .* vcat(yf, ys)[lookup]
    c = log.(completed) .- 0.5
    eta = a .+ b .* x .+ z[g] .+ c[pos]
    y .~ Poisson.(exp.(eta))
    lam = exp.(eta[r1]) .+ exp.(eta[r2])
    log_m = log.(lam) .+ a
    m2 = exp.(log_m[r3])
    w .~ NegativeBinomial2.(m2, phi)
end

function _cq_gathered_data(n)
    # A covariate of `rows` entries, some missing: the first `nfree` missing
    # entries are free draws, the others grouped ones. An observed entry looks
    # up drawn entry 1 and masks it out, as BRM's completions do.
    rows = max(4, n ÷ 2)
    nfree = nshare = max(1, rows ÷ 8)
    missing_rows = collect(range(1; step = 3, length = nfree + nshare))
    lookup, mask = ones(Int, rows), zeros(rows)
    for (k, row) in enumerate(missing_rows)
        lookup[row], mask[row] = k, 1.0
    end
    observed = (1 .- mask) .* collect(range(0.5, 3.0; length = rows))
    (; x = collect(range(-1, 1; length = n)), g = ["g$(mod1(7j, max(2, n ÷ 10)))" for j in 1:n],
       free_rows = collect(1:nfree), share_rows = collect(1:nshare),
       mf = collect(range(-0.2, 0.2; length = nfree)), observed, mask, lookup,
       pos = [mod1(5j, rows) for j in 1:n], y = [j % 4 for j in 1:n],
       r1 = [mod1(3j, n) for j in 1:n], r2 = [mod1(11j + 1, n) for j in 1:n],
       r3 = [mod1(j + 2, n) for j in 1:n], w = [j % 3 for j in 1:n])
end

@testset "cell query: gathered model-level values and their completions cost their own rows" begin
    backend = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
    for obs in (:y, :w)
        measured = map((40, 4_000)) do n
            data = _cq_gathered_data(n)
            bound, built = _cq_bound(_CQ_GATHERED, data, (:y, :w))
            q = prepare_cell_query(built, bound, obs)
            u = collect(range(-0.4, 0.5; length = built.layout.total))
            pointwise = getproperty(
                Base.invokelatest(prepare_query(built, bound, :pointwise), u), obs)
            values = [q(u, i) for i in (1, 2, n)]
            q(u, 2)
            dense = prepare_cell_sampler(built, bound, obs, u; backend)
            dense_value, dense_gradient = cell_value_and_gradient!(dense, similar(u), u, 2)
            support = findall(!iszero, dense_gradient)
            indices = sort!(union(support, 1:4))
            compact = prepare_compact_cell_sampler(built, bound, obs, u, indices; backend)
            compact_gradient = similar(u[indices])
            compact_value, _ = cell_value_and_gradient!(
                compact, compact_gradient, u, u[indices], indices, 2)
            j = last(support)
            (; values, reference = pointwise[[1, 2, n]], bytes = (@allocated q(u, 2)),
               dense_value, compact_value, support = length(support),
               parity = compact_gradient ≈ dense_gradient[indices],
               central = isapprox(dense_gradient[j], _cq_central(v -> q(v, 2), u, j);
                   rtol = 1e-5))
        end
        @test all(m -> m.values ≈ m.reference, measured)
        @test measured[1].bytes == measured[2].bytes
        @test all(m -> m.values[2] ≈ m.dense_value ≈ m.compact_value, measured)
        @test all(m -> m.parity && m.central, measured)
        # The gradient's support is the cell's own rows: the coefficients,
        # one group effect and one drawn entry per row read (and its scale).
        @test all(m -> m.support <= 11, measured)
    end
end

const _CQ_NUMBER_READERS = quote
    sigma ~ Exponential(1)
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    z[levels(g)] .~ Normal.(0, 1)
    # A number the cell computes, read by two predictors the cell computes at
    # its rows (snag `cell-take-rows-s-e48675d0`).
    s = exp.(b)
    l1 = s .* x
    l2 = s .+ w
    mu = a .+ l1 .+ l2 .+ z[g]
    y .~ Normal.(mu, sigma)
end

@testset "cell query: a number read by two predictors of a cell" begin
    backend = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
    measured = map((30, 3_000)) do n
        data = (; y = cos.(1:n), x = sin.(1:n), w = collect(range(0.1, 1.0; length = n)),
            g = [mod1(j, 5) for j in 1:n])
        bound, built = _cq_bound(_CQ_NUMBER_READERS, data, (:y,))
        q = prepare_cell_query(built, bound, :y)
        u = collect(range(-0.4, 0.5; length = built.layout.total))
        pointwise = Base.invokelatest(prepare_query(built, bound, :pointwise), u).y
        values = [q(u, i) for i in (1, 2, n)]
        q(u, 2)
        dense = prepare_cell_sampler(built, bound, :y, u; backend)
        value, gradient = cell_value_and_gradient!(dense, similar(u), u, 2)
        support = findall(!iszero, gradient)
        compact = prepare_compact_cell_sampler(built, bound, :y, u, support; backend)
        compact_gradient = similar(u[support])
        compact_value, _ = cell_value_and_gradient!(
            compact, compact_gradient, u, u[support], support, 2)
        (; values, reference = pointwise[[1, 2, n]], bytes = (@allocated q(u, 2)), value,
           compact_value, parity = compact_gradient ≈ gradient[support],
           central = all(j -> isapprox(gradient[j], _cq_central(v -> q(v, 2), u, j);
               rtol = 1e-5), support))
    end
    @test all(m -> m.values ≈ m.reference, measured)
    @test measured[1].bytes == measured[2].bytes
    @test all(m -> m.values[2] ≈ m.value ≈ m.compact_value, measured)
    @test all(m -> m.parity && m.central, measured)
end

# A covariate completed by a kernel that gathers straight from an inline
# `vcat` of its observed and drawn entries, as BRM emits it, and the same
# completion as a definition, which RKPPL reads through a read-only gather
# (snag `inline-vcat-gath-bb73ae09`): each subject's cell computes the
# completion at its own row, so an observed subject reads no drawn entry and a
# missing one reads only its own, as with `stacked = vcat(...)` and
# `stacked[order]`.
ReactiveKernels.@kernel _cq_completed_covariate(observed, drawn, order) = begin
    completed = (vcat(observed, drawn))[order]
    return completed
end

const _CQ_INLINE_COMPLETION = quote
    a ~ Normal(0, 1)
    sigma ~ Exponential(1)
    z_mis[axes(missing_rows, 1)] .~ Normal.(0, 1)
    y_mis = 0 .+ 1 .* z_mis
    w = _cq_completed_covariate(w_observed, y_mis, w_order)
    y .~ Normal.(a .+ w, sigma)
end

const _CQ_INLINE_DEFINITION = quote
    a ~ Normal(0, 1)
    sigma ~ Exponential(1)
    z_mis[axes(missing_rows, 1)] .~ Normal.(0, 1)
    y_mis = 0 .+ 1 .* z_mis
    w = vcat(w_observed, y_mis)[w_order]
    y .~ Normal.(a .+ w, sigma)
end

# Even subjects observe the covariate; `w_order` is each subject's position in
# `vcat(w_observed, y_mis)`.
function _cq_inline_completion_data(n)
    seen = [iseven(j) for j in 1:n]
    nobs = count(seen)
    order, o, m = zeros(Int, n), 0, 0
    for j in 1:n
        order[j] = seen[j] ? (o += 1) : nobs + (m += 1)
    end
    (; w_observed = collect(range(0.2, 1.4; length = nobs)),
       missing_rows = collect(1:(n - nobs)), w_order = order,
       y = [0.3 * sin(j) for j in 1:n])
end

@testset "cell query: a gather from an inline vcat costs the cell's own rows" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    @testset "$label" for (label, ast) in (("kernel", _CQ_INLINE_COMPLETION),
                                          ("definition", _CQ_INLINE_DEFINITION))
        measured = map((20, 2_000)) do n
            data = _cq_inline_completion_data(n)
            bound, built = _cq_bound(ast, data, (:y,))
            q = prepare_cell_query(built, bound, :y)
            u = collect(range(-0.4, 0.5; length = built.layout.total))
            pointwise = Base.invokelatest(prepare_query(built, bound, :pointwise), u).y
            values = [q(u, i) for i in (1, 2, n)]
            q(u, 3)
            s = prepare_cell_sampler(built, bound, :y, u; backend)
            names = coordinate_names(built.layout)
            cells = map((2, 3)) do i
                g = similar(u)
                value, _ = cell_value_and_gradient!(s, g, u, i)
                j = last(findall(!iszero, g))
                (; value, drawn = filter(x -> startswith(String(x), "z_mis"), names[findall(!iszero, g)]),
                   central = isapprox(g[j], _cq_central(v -> q(v, i), u, j); rtol = 1e-5))
            end
            (; values, reference = pointwise[[1, 2, n]], bytes = (@allocated q(u, 3)), cells,
               cell_values = [q(u, i) for i in (2, 3)])
        end
        @test all(m -> m.values ≈ m.reference, measured)
        @test measured[1].bytes == measured[2].bytes
        for m in measured
            @test [c.value for c in m.cells] ≈ m.cell_values
            @test all(c -> c.central, m.cells)
            # Subject 2 observes its covariate; subject 3 is the second missing one.
            @test isempty(m.cells[1].drawn)
            @test m.cells[2].drawn == [Symbol("z_mis.2")]
        end
    end
end

# A joint population effect for several responses, as BRM emits it (snag
# `column-read-lowe-24508269`): a design with a gathered computed column, a
# submodel returning `X * beta` and one definition per response column. RKPPL
# reads each column `P[:, j]` of the per-observation product in place, through
# `view(P, :, j)`; a cell of one response computes the product, the design and
# the gathered column at its own row, so its cost does not grow with the
# observations.
@rkppl _cq_joint(X, K, T) = begin
    beta[1:K, 1:T] .~ Normal.(0.0, 1.0)
    return X * beta
end

const _CQ_JOINT_COLUMNS = quote
    sigma ~ Exponential(1)
    a ~ Normal(0, 1)
    z[levels(g)] .~ Normal.(0, 1)
    c = z[g]
    X = hcat(ones(length(x)), x, c)
    P ~ _cq_joint(X, 3, 3)
    m1 = P[:, 1]
    eta1 = m1 .+ a
    y1 .~ Normal.(eta1, sigma)
    y2 .~ Normal.(P[:, 2] .+ P[:, 3], sigma)
end

_cq_joint_columns_data(n) = (; x = collect(range(0.5, 1.5; length = n)),
    g = [mod1(3j, 5) for j in 1:n],
    y1 = [0.3 * sin(j) for j in 1:n], y2 = [0.2 * cos(j) for j in 1:n])

@testset "cell query: column views of a joint product cost the cell's own row" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    measured = map((20, 2_000)) do n
        data = _cq_joint_columns_data(n)
        bound, built = _cq_bound(_CQ_JOINT_COLUMNS, data, (:y1, :y2))
        @test occursin("Base.view(P", string(kernel_expr(bound, built.layout)))
        q = prepare_cell_query(built, bound, :y1)
        u = collect(range(-0.4, 0.5; length = built.layout.total))
        pointwise = Base.invokelatest(prepare_query(built, bound, :pointwise), u).y1
        values = [q(u, i) for i in (1, 2, n)]
        q(u, 3)
        s = prepare_cell_sampler(built, bound, :y1, u; backend)
        cells = map((2, 3)) do i
            g = similar(u)
            value, _ = cell_value_and_gradient!(s, g, u, i)
            (; value, central = all(j -> isapprox(g[j], _cq_central(v -> q(v, i), u, j);
                rtol = 1e-5, atol = 1e-8), eachindex(u)))
        end
        (; values, reference = pointwise[[1, 2, n]], bytes = (@allocated q(u, 3)), cells,
           cell_values = [q(u, i) for i in (2, 3)])
    end
    @test all(m -> m.values ≈ m.reference, measured)
    @test measured[1].bytes == measured[2].bytes
    for m in measured
        @test [c.value for c in m.cells] ≈ m.cell_values
        @test all(c -> c.central, m.cells)
    end
end
