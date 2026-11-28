module DottedCellGatherFootprintTests
using ReactiveKernels, ReactiveKernelsPPL, SparseArrays, Test
using DifferentiationInterface: AutoEnzyme
import Enzyme

# A dotted `@plate` observation whose argument gathers a model value at the
# cell's bound rows, `y[g] .~ Normal.(mu[rows[g]], sigma)`: the form a
# hand-written per-series likelihood takes. RKPPL reads such a gather in place
# (`_ppl_cell_gather`), in a plate of its own. RK's analysis of how a plate
# body reads a shared argument took that call for a read of the whole value,
# so a cell query computed `mu` whole in every cell, each cell's footprint held
# every series' states, and `locality = :error` raised nothing: `mu` was
# reported as shared by all cells (snag `dotted-plate-obs-4a6835eb`). The call
# is the read `mu[rows[g]]`, so the cell computes `mu` at its own rows.

# The gather written in the observation, and in a cell local.
const _DCG_INLINE = @rkppl begin
    sigma ~ Exponential(1)
    w[1:nw] .~ Normal.(0, 1)
    mu = w[pos]
    @plate for g in eachindex(y)
        y[g] .~ Normal.(mu[rows[g]], sigma)
    end
end

const _DCG_LOCAL = @rkppl begin
    sigma ~ Exponential(1)
    w[1:nw] .~ Normal.(0, 1)
    mu = w[pos]
    @plate for g in eachindex(y)
        m = mu[rows[g]]
        y[g] .~ Normal.(m, sigma)
    end
end

# The gathered value computed through a matrix product (named, inline, or
# returned by a submodel, the shape of a population effect), a `vcat`, and a
# product over a bound sparse factor.
@rkppl _dcg_population(X, ncoef, loc, scale) = begin
    beta[1:ncoef] .~ Normal.(loc, scale)
    return X * beta
end

const _DCG_PRODUCT = @rkppl begin
    z[1:ng] .~ Normal.(0, 1)
    beta[1:1] .~ Normal.(0, 1)
    pop = X * beta
    mu = z[group] .+ pop
    @plate for g in eachindex(y)
        y[g] .~ Normal.(mu[rows[g]], 1)
    end
end

const _DCG_INLINE_PRODUCT = @rkppl begin
    z[1:ng] .~ Normal.(0, 1)
    beta[1:1] .~ Normal.(0, 1)
    mu = z[group] .+ X * beta
    @plate for g in eachindex(y)
        y[g] .~ Normal.(mu[rows[g]], 1)
    end
end

const _DCG_SUBMODEL_PRODUCT = @rkppl begin
    z[1:ng] .~ Normal.(0, 1)
    pop ~ _dcg_population(X, 1, 0.0, 1.0)
    mu = z[group] .+ pop
    @plate for g in eachindex(y)
        y[g] .~ Normal.(mu[rows[g]], 1)
    end
end

const _DCG_VCAT = @rkppl begin
    z[1:ng] .~ Normal.(0, 1)
    left = z[head]
    right = z[tail]
    mu = vcat(left, right)
    @plate for g in eachindex(y)
        y[g] .~ Normal.(mu[rows[g]], 1)
    end
end

const _DCG_SPARSE_PRODUCT = @rkppl begin
    z[1:ng] .~ Normal.(0, 1)
    beta[1:2] .~ Normal.(0, 1)
    pop = S * beta
    mu = z[group] .+ pop
    @plate for g in eachindex(y)
        y[g] .~ Normal.(mu[rows[g]], 1)
    end
end

# Two observations of one loop gather the same value by different rows.
const _DCG_TWO = @rkppl begin
    sigma ~ Exponential(1)
    w[1:nw] .~ Normal.(0, 1)
    mu = w[pos]
    @plate for g in eachindex(y)
        y[g] .~ Normal.(mu[rows[g]], sigma)
        x[g] .~ Normal.(mu[lead[g]], 1)
    end
end

# The index is one Boolean mask per cell: it selects entries by value, so the
# read is not a read at rows.
const _DCG_MASK = @rkppl begin
    sigma ~ Exponential(1)
    w[1:nw] .~ Normal.(0, 1)
    mu = w[pos]
    @plate for g in eachindex(y)
        y[g] .~ Normal.(mu[mask[g]], sigma)
    end
end

# The same mask read by plain indexing in a cell local.
const _DCG_MASK_LOCAL = @rkppl begin
    sigma ~ Exponential(1)
    w[1:nw] .~ Normal.(0, 1)
    mu = w[pos]
    @plate for g in eachindex(t)
        m = mu[mask[g]]
        t[g] ~ Normal(sum(m), sigma)
    end
end

# The gathered value is a declared parameter: nothing computes it.
const _DCG_PARAMETER = @rkppl begin
    theta[1:nrater] .~ Normal.(0, 2)
    sigma ~ Exponential(1)
    @plate for j in eachindex(y)
        y[j] .~ Normal.(theta[rater[j]], sigma)
    end
end

# The gathered value is computed by an ordinary function, which has no row
# form.
const _DCG_OPAQUE = @rkppl begin
    sigma ~ Exponential(1)
    w[1:nw] .~ Normal.(0, 1)
    mu = cumsum(w[pos])
    @plate for g in eachindex(y)
        y[g] .~ Normal.(mu[rows[g]], sigma)
    end
end

normal_logpdf(y, mu, sigma) = -log(2pi) / 2 - log(sigma) - (y - mu)^2 / (2sigma^2)

# Each cell's packed-coordinate footprint by name, in the two forms a consumer
# asks for: the whole kernel, and the entries a partition owns (`partitioned`,
# `shared = false`, inside a cell batch), the form BRM's sparse Laplace uses.
function footprints(built, plan, u, cells; locality, partitioned)
    names = coordinate_names(built.layout)
    query = prepare_cell_query(built, plan, :y; locality)
    footprint = partition_footprint_kernel(query.kernel)
    whole = [names[Base.invokelatest(footprint, u, g).unconstrained] for g in cells]
    owning = prepare_cell_query(built, plan, :y; locality, partitioned, u0 = u)
    owned_footprint = partition_footprint_kernel(owning; shared = false)
    owned = with_cell_batch(owning, u) do batch
        [names[Base.invokelatest(owned_footprint, batch, g).unconstrained] for g in cells]
    end
    (; query, owning, whole, owned)
end

status(query, value) = only(e.status for e in ReactiveKernels.partition_locality(query)
                            if e.value === value)

# One cell's gradient by ordinary Enzyme Reverse against central differences
# of the cell query: it differentiates the read of the cell's window.
function central_differences(f, u; h = 1e-6)
    map(eachindex(u)) do k
        up, down = copy(u), copy(u)
        up[k] += h
        down[k] -= h
        (f(up) - f(down)) / 2h
    end
end
function check_cell_gradients(built, plan, query, u, cells; observations = :y)
    sampler = prepare_cell_sampler(built, plan, observations, u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    for g in cells
        value, gradient = cell_value_and_gradient!(sampler, similar(u), u, g)
        @test value ≈ Base.invokelatest(query, u, g)
        @test gradient ≈ central_differences(v -> Base.invokelatest(query, v, g), u) atol = 1e-6
    end
end
whole_values(query) = [e.value for e in ReactiveKernels.partition_locality(query)
                       if e.status === :whole]

# Series interleaved over the hours, as long-format data are: hour `r` belongs
# to series `mod1(r, S)` and reads that series' state of its week.
function series_data(S, W, H)
    hours = S * W * H
    series = [mod1(r, S) for r in 1:hours]
    week = [(cld(r, S) - 1) ÷ H + 1 for r in 1:hours]
    pos = [(series[r] - 1) * W + week[r] for r in 1:hours]
    rows = [findall(==(g), series) for g in 1:S]
    yrow = [0.3 * sin(1.7r) + 0.1 * series[r] for r in 1:hours]
    (; nw = S * W, pos, rows, y = [yrow[r] for r in rows])
end

@testset "a dotted cell's gather of a computed value reads its own rows" begin
    S, W, H = 3, 2, 3
    d = series_data(S, W, H)
    saved = deepcopy(d)
    states(g) = [Symbol("w.$k") for k in ((g - 1) * W + 1):(g * W)]
    for model in (_DCG_INLINE, _DCG_LOCAL)
        plan = model(; d.nw, d.pos, d.rows) | (; d.y)
        built = build_kernel(plan)
        u = collect(range(-0.4, 0.5; length = built.layout.total))
        names = coordinate_names(built.layout)
        (; sigma, w) = constrain(built.layout, u)
        mu = w[d.pos]
        reference(g) = sum(normal_logpdf.(d.y[g], mu[d.rows[g]], sigma))
        for locality in (:error, :ignore)
            fx = footprints(built, plan, u, 1:S; locality,
                partitioned = (:w => collect(1:d.nw),))
            for g in 1:S
                @test fx.whole[g] == [:sigma; states(g)]
                @test fx.owned[g] == states(g)
                @test Base.invokelatest(fx.query, u, g) ≈ reference(g)
            end
            @test isempty(whole_values(fx.query))
            @test isempty(whole_values(fx.owning))
            @test status(fx.query, :mu) === :rows
            @test status(fx.query, :w) === :rows
        end
        # One cell's gradient is nonzero only at the coordinates it reads.
        sampler = prepare_cell_sampler(built, plan, :y, u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        for g in 1:S
            value, gradient = cell_value_and_gradient!(sampler, similar(u), u, g)
            expected = zeros(length(u))
            residual = d.y[g] .- mu[d.rows[g]]
            # `sigma = exp(u[sigma])`: d/dlog(sigma) of the cell's density.
            expected[findfirst(==(:sigma), names)] =
                sum(residual .^ 2) / sigma^2 - length(residual)
            for (r, e) in zip(d.rows[g], residual)
                expected[findfirst(==(Symbol("w.$(d.pos[r])")), names)] += e / sigma^2
            end
            @test value ≈ reference(g)
            @test gradient ≈ expected rtol = 1e-10
        end
        # The whole density still reads the gather in place.
        @test occursin("_ppl_cell_gather(mu", string(kernel_expr(plan, built.layout)))
    end
    @test isequal(d, saved)
end

# Three groups of four rows; `S` gives the first group its own coefficient.
function group_data()
    G, n = 3, 4
    group = repeat(1:G; inner = n)
    rows = [collect(((g - 1) * n + 1):(g * n)) for g in 1:G]
    x = [cos(0.7r) for r in 1:(G * n)]
    yrow = [0.3 + 0.5 * x[r] + 0.2 * sin(3r) for r in 1:(G * n)]
    (; G, group, rows, x, X = hcat(x), y = [yrow[r] for r in rows],
       S = sparse(1:(G * n), [g == 1 ? 1 : 2 for g in group], x, G * n, 2),
       head = group[1:6], tail = group[7:12])
end

@testset "the gathered value through products, vcat and a sparse product" begin
    d = group_data()
    saved = deepcopy(d)
    cases = (
        (_DCG_PRODUCT(; ng = d.G, d.group, d.rows, d.X), g -> [Symbol("beta.1")]),
        (_DCG_INLINE_PRODUCT(; ng = d.G, d.group, d.rows, d.X), g -> [Symbol("beta.1")]),
        (_DCG_SUBMODEL_PRODUCT(; ng = d.G, d.group, d.rows, d.X), g -> [Symbol("pop.beta.1")]),
        (_DCG_VCAT(; ng = d.G, d.head, d.tail, d.rows), g -> Symbol[]),
        (_DCG_SPARSE_PRODUCT(; ng = d.G, d.group, d.rows, d.S),
            g -> [Symbol(g == 1 ? "beta.1" : "beta.2")]),
    )
    for (k, (model, coefficients)) in enumerate(cases)
        plan = model | (; d.y)
        built = build_kernel(plan)
        u = collect(range(-0.3, 0.6; length = built.layout.total))
        full = prepare_query(built, plan, :pointwise)
        pointwise = Base.invokelatest(full, u).y
        for locality in (:error, :ignore)
            fx = footprints(built, plan, u, 1:d.G; locality,
                partitioned = (:z => collect(1:d.G),))
            for g in 1:d.G
                @test Set(fx.whole[g]) == Set([Symbol("z.$g"); coefficients(g)])
                @test fx.owned[g] == [Symbol("z.$g")]
                @test Base.invokelatest(fx.query, u, g) ≈ sum(pointwise[g])
            end
            @test isempty(whole_values(fx.query))
            @test status(fx.query, :mu) === :rows
            # The dense and the sparse product hold their rows differently.
            locality === :error && k in (1, 5) &&
                check_cell_gradients(built, plan, fx.query, u, 1:d.G)
        end
    end
    @test isequal(d, saved)
end

@testset "a dotted cell's gather of a parameter reads the gathered entries" begin
    rater = [[1, 3], Int[], [2, 3, 4], [4]]
    y = [[0.1, -0.3], Float64[], [0.4, 0.2, -0.1], [0.7]]
    plan = _DCG_PARAMETER(; nrater = 4, rater) | (; y)
    built = build_kernel(plan)
    u = collect(range(-0.4, 0.5; length = built.layout.total))
    (; theta, sigma) = constrain(built.layout, u)
    for locality in (:error, :ignore)
        fx = footprints(built, plan, u, 1:4; locality,
            partitioned = (:theta => collect(1:4),))
        for j in 1:4
            entries = [Symbol("theta.$r") for r in rater[j]]
            @test Set(fx.whole[j]) == Set([entries; :sigma])
            @test fx.owned[j] == entries
            @test Base.invokelatest(fx.query, u, j) ≈
                sum(normal_logpdf.(y[j], theta[rater[j]], sigma); init = 0.0)
        end
        @test isempty(whole_values(fx.query))
        locality === :error && check_cell_gradients(built, plan, fx.query, u, 1:4)
    end
end

@testset "two observations gathering one value by different rows" begin
    # The cell's window of `mu` then holds the rows of both gathers, so each
    # read finds its rows among them.
    d = series_data(3, 2, 3)
    lead = [circshift(r, 1) for r in d.rows]
    x = [0.5 .* v for v in d.y]
    plan = _DCG_TWO(; d.nw, d.pos, d.rows, lead) | (; d.y, x)
    built = build_kernel(plan)
    u = collect(range(-0.4, 0.5; length = built.layout.total))
    (; sigma, w) = constrain(built.layout, u)
    mu = w[d.pos]
    query = prepare_cell_query(built, plan, (:y, :x); locality = :error)
    footprint = partition_footprint_kernel(query.kernel)
    names = coordinate_names(built.layout)
    for g in 1:3
        @test names[Base.invokelatest(footprint, u, g).unconstrained] ==
            [:sigma; [Symbol("w.$k") for k in (2g - 1):(2g)]]
        @test Base.invokelatest(query, u, g) ≈
            sum(normal_logpdf.(d.y[g], mu[d.rows[g]], sigma)) +
            sum(normal_logpdf.(x[g], mu[lead[g]], 1.0))
    end
    @test isempty(whole_values(query))
    check_cell_gradients(built, plan, query, u, 1:3; observations = (:y, :x))
end

@testset "a partition of several cells reads its cells' rows" begin
    # Cells 1 and 3 form one unit, cells 2 and 4 the other: a unit's window of
    # `mu` holds both cells' rows, sorted, and each cell finds its own.
    S, W = 4, 2
    d = series_data(S, W, 3)
    data = (; d.nw, d.pos, d.rows, d.y, unit = [1, 2, 1, 2])
    plan = bind_data(lower_rkppl(quote
        sigma ~ Exponential(1)
        w[1:nw] .~ Normal.(0, 1)
        mu = w[pos]
        @plate for g in eachindex(y)
            y[g] .~ Normal.(mu[rows[g]], sigma)
        end
    end, data; conditioned = (:y,), mod = @__MODULE__), data)
    built = build_kernel(plan)
    u = collect(range(-0.4, 0.5; length = built.layout.total))
    names = coordinate_names(built.layout)
    (; sigma, w) = constrain(built.layout, u)
    mu = w[d.pos]
    reference(g) = sum(normal_logpdf.(d.y[g], mu[d.rows[g]], sigma))
    for locality in (:error, :ignore)
        partition = prepare_partition(built, plan, :y; by = :unit, locality)
        @test collect(partition_labels(partition)) == [1, 2]
        for label in 1:2
            cells = findall(==(label), data.unit)
            @test Base.invokelatest(partition, u, label) ≈ sum(reference, cells)
            @test names[partition_footprint(partition.kernel, u, label, nothing).unconstrained] ==
                [:sigma; [Symbol("w.$k") for g in cells for k in ((g - 1) * W + 1):(g * W)]]
        end
        @test isempty(whole_values(partition.kernel))
    end
end

@testset "a Boolean mask index is read whole, and named" begin
    d = series_data(3, 2, 3)
    series = [mod1(r, 3) for r in eachindex(d.pos)]
    for mask in ([series .== g for g in 1:3], [collect(series .== g) for g in 1:3])
        plan = _DCG_MASK(; d.nw, d.pos, mask) | (; d.y)
        built = build_kernel(plan)
        u = collect(range(-0.4, 0.5; length = built.layout.total))
        (; sigma, w) = constrain(built.layout, u)
        mu = w[d.pos]
        err = try
            prepare_cell_query(built, plan, :y; locality = :error)
            nothing
        catch e
            e
        end
        @test err isa ReactiveKernels.PartitionLocalityError
        @test [v.value for v in err.violations] == [:mu]
        @test occursin("Boolean mask", sprint(showerror, err))
        query = prepare_cell_query(built, plan, :y; locality = :ignore)
        @test whole_values(query) == [:mu]
        for g in 1:3
            @test Base.invokelatest(query, u, g) ≈
                sum(normal_logpdf.(d.y[g], mu[mask[g]], sigma))
        end
        # Plain indexing by the mask took it for row positions and threw
        # `BoundsError` from the cell query; it is read whole as well.
        t = [0.2, -0.1, 0.4]
        plan = _DCG_MASK_LOCAL(; d.nw, d.pos, mask) | (; t)
        built = build_kernel(plan)
        @test_throws ReactiveKernels.PartitionLocalityError prepare_cell_query(
            built, plan, :t; locality = :error)
        query = prepare_cell_query(built, plan, :t; locality = :ignore)
        for g in 1:3
            @test Base.invokelatest(query, u, g) ≈
                normal_logpdf(t[g], sum(mu[mask[g]]), sigma)
        end
    end
end

@testset "a gathered value with no row form is named, not silently shared" begin
    d = series_data(3, 2, 3)
    plan = _DCG_OPAQUE(; d.nw, d.pos, d.rows) | (; d.y)
    built = build_kernel(plan)
    u = collect(range(-0.4, 0.5; length = built.layout.total))
    (; sigma, w) = constrain(built.layout, u)
    mu = cumsum(w[d.pos])
    err = try
        prepare_cell_query(built, plan, :y; locality = :error)
        nothing
    catch e
        e
    end
    @test err isa ReactiveKernels.PartitionLocalityError
    @test [v.value for v in err.violations] == [:mu]
    query = prepare_cell_query(built, plan, :y; locality = :ignore)
    @test whole_values(query) == [:mu]
    for g in 1:3
        @test Base.invokelatest(query, u, g) ≈
            sum(normal_logpdf.(d.y[g], mu[d.rows[g]], sigma))
    end
end

end
