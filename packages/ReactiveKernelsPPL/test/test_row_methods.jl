module RowMethodTests
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, Test
using DifferentiationInterface: AutoEnzyme

const PPL = ReactiveKernelsPPL
const BACKEND = AutoEnzyme(; mode=Enzyme.Reverse)

# A summand whose axes the lowering cannot see: a submodel's value that a
# kernel computes, as BRM emits a grouped HSGP. The location then gets its rows
# at run time (`_ppl_rows`).
@kernel field_graph(t, sigma, z, g) = begin
    rows = z[g, :]
    value = vec(sum(t .* rows; dims = 2)) .* sigma
    return value
end
@rkppl field(t, g) = begin
    sigma ~ LogNormal(0.0, 1.0)
    z[1:3, 1:2] .~ Normal.(0, 1)
    value = field_graph(t, sigma, z, g)
    return value
end
@rkppl group_effects(g) = begin
    tau[1:1] .~ LogNormal.(0, 1)
    z[1:length(levels(g)), 1:1] .~ Normal.(0, 1)
    return z .* tau[1]
end
@rkppl population(X, k) = begin
    beta[1:k] .~ Normal.(0, 1)
    return X * beta
end
@kernel leading_graph(w, n) = begin
    v = w[1:n]
    return v
end
@rkppl leading(n) = begin
    w[1:2] .~ Normal.(0, 1)
    value = leading_graph(w, n)
    return value
end
@rkppl trailing(n) = begin
    q[1:2] .~ Normal.(0, 1)
    value = leading_graph(q, n)
    return value
end

normal(y, m, s) = -0.5log(2pi) - log(s) - 0.5((y - m) / s)^2

function build(model, data)
    plan = bind_data(lower_rkppl(model, data; conditioned=(:y,), mod=@__MODULE__), data)
    plan, build_kernel(plan)
end

const DATA = (; subject=repeat(1:3, inner=2), t=collect(range(-1.0, 1.0; length=6)),
    y=collect(range(-0.5, 0.5; length=6)), X=hcat(ones(6), collect(range(-1.0, 1.0; length=6))))

# Whether the program gives a location its rows at run time, so a test over it
# exercises `_ppl_rows`'s row method.
reaches_rows(built) = occursin("_ppl_rows", sprint(show, built.program))

position(names, name) = findfirst(==(name), names)

function field_value(names, u, i)
    level = DATA.subject[i]
    DATA.t[i] * (u[position(names, Symbol("f.z.$level.1"))] +
        u[position(names, Symbol("f.z.$level.2"))]) * exp(u[position(names, Symbol("f.sigma"))])
end
ranef(names, u, i) = u[position(names, Symbol("b.z.$(DATA.subject[i]).1"))] *
    exp(u[position(names, Symbol("b.tau.1"))])

# The gradient of the strict compact partition against differences of the cell
# values it sums.
function check_gradient(built, plan, u, cell)
    sampler = prepare_partition_sampler(built, plan, :y, u; by=:subject, backend=BACKEND,
        compact=true, locality=:error)
    for label in unique(DATA.subject)
        cells = findall(==(label), DATA.subject)
        total(point) = sum(cell(point, i) for i in cells)
        fp = partition_footprint(sampler, label)
        gradient = zeros(length(fp))
        value, _ = partition_value_and_gradient!(sampler, gradient, u, label)
        @test value ≈ total(u)
        differences = map(eachindex(u)) do j
            step = zeros(length(u))
            step[j] = 1e-6
            (total(u .+ step) - total(u .- step)) / 2e-6
        end
        @test gradient ≈ differences[fp] atol = 1e-5
        @test all(iszero, differences[setdiff(eachindex(u), fp)])
    end
end

@testset "a location takes its rows at a cell's rows" begin
    # The spellings BRM's programs reach: effects and a field, and a population
    # product beside a field. A scalar beside them lowers through its affine
    # coefficient instead and stays local, so it needs no rows at run time.
    models = (
        (quote
            sigma ~ Exponential(1.0)
            b ~ group_effects(subject)
            r = b[subject, 1]
            f ~ field(t, subject)
            mu = .+(r, f)
            y .~ Normal.(mu, sigma)
        end, (names, u, i) -> ranef(names, u, i) + field_value(names, u, i), true),
        (quote
            sigma ~ Exponential(1.0)
            a ~ Normal(0, 1)
            b ~ group_effects(subject)
            r = b[subject, 1]
            f ~ field(t, subject)
            mu = .+(a, r, f)
            y .~ Normal.(mu, sigma)
        end, (names, u, i) -> u[position(names, :a)] + ranef(names, u, i) + field_value(names, u, i), false),
        (quote
            sigma ~ Exponential(1.0)
            pop ~ population(X, 2)
            f ~ field(t, subject)
            mu = .+(pop, f)
            y .~ Normal.(mu, sigma)
        end, (names, u, i) -> u[position(names, Symbol("pop.beta.1"))] +
            DATA.X[i, 2] * u[position(names, Symbol("pop.beta.2"))] + field_value(names, u, i), true),
    )
    for (model, location, runtime_rows) in models
        plan, built = build(model, DATA)
        @test reaches_rows(built) == runtime_rows
        names = coordinate_names(built.layout)
        u = collect(range(-0.4, 0.5; length=built.layout.total))
        cell = prepare_cell_query(built, plan, :y; locality=:error)
        footprint = partition_footprint_kernel(cell.kernel)
        s = exp(u[position(names, :sigma)])
        for i in eachindex(DATA.y)
            @test cell(u, i) ≈ normal(DATA.y[i], location(names, u, i), s)
            reads = names[Base.invokelatest(footprint, u, i).unconstrained]
            # This cell's own level, and no other subject's.
            level = DATA.subject[i]
            @test Symbol("f.z.$level.1") in reads
            @test !any(other -> Symbol("f.z.$other.1") in reads, setdiff(1:3, level))
        end
        @test !any(e -> e.status === :whole, partition_locality(cell.kernel))
        check_gradient(built, plan, u, cell)
    end
end

@testset "a location's rows keep the whole value's broadcast" begin
    # One entry repeats along the rows, in the cell as in the whole program.
    plan, built = build(quote
        f ~ leading(1)
        h ~ trailing(1)
        mu = .+(f, h)
        y .~ Normal.(mu, 1.0)
    end, DATA)
    @test reaches_rows(built)
    names = coordinate_names(built.layout)
    u = collect(range(-0.4, 0.5; length=built.layout.total))
    cell = prepare_cell_query(built, plan, :y; locality=:error)
    m = u[position(names, Symbol("f.w.1"))] + u[position(names, Symbol("h.q.1"))]
    for i in eachindex(DATA.y)
        @test cell(u, i) ≈ normal(DATA.y[i], m, 1.0)
    end

    # A value of neither one entry nor one per row keeps its dimension error.
    plan, built = build(quote
        f ~ leading(2)
        h ~ trailing(2)
        mu = .+(f, h)
        y .~ Normal.(mu, 1.0)
    end, DATA)
    @test reaches_rows(built)
    u = collect(range(-0.4, 0.5; length=built.layout.total))
    cell = prepare_cell_query(built, plan, :y; locality=:error)
    @test_throws DimensionMismatch cell(u, 1)
    whole = prepare_query(built, plan, :likelihood)
    @test_throws DimensionMismatch Base.invokelatest(whole, u)

    # What each call returns: a number fills the rows, a row-aligned window is
    # itself, and any other element type converts as the whole broadcast does.
    window = [0.5, 1.5]
    @test PPL._ppl_rows_at([3, 5], 2.0, 6) == [2.0, 2.0]
    @test PPL._ppl_rows_at(3, 2.0, 6) == [2.0]
    @test PPL._ppl_rows_at([3, 5], window, 6) === window
    @test PPL._ppl_rows_at([3, 5], [1, 2], 6) == [1.0, 2.0]
    @test PPL._ppl_rows_at([3, 5], [1, 2], 6) isa Vector{Float64}
end

@testset "a dotted expression's constant operands read no row" begin
    plan, built = build(quote
        sigma ~ Exponential(1.0)
        rho ~ Exponential(1.0)
        s[levels(subject)] .~ Exponential.(1.0)
        scale = s .* (rho .* sqrt(2pi)) .^ (1 / 2)
        mu = scale[subject]
        y .~ Normal.(mu, sigma)
    end, DATA)
    names = coordinate_names(built.layout)
    u = collect(range(-0.4, 0.5; length=built.layout.total))
    cell = prepare_cell_query(built, plan, :y; locality=:error)
    footprint = partition_footprint_kernel(cell.kernel)
    for i in eachindex(DATA.y)
        level = Symbol("s.$(DATA.subject[i])")
        m = exp(u[position(names, level)]) * (exp(u[position(names, :rho)]) * sqrt(2pi))^(1 / 2)
        @test cell(u, i) ≈ normal(DATA.y[i], m, exp(u[position(names, :sigma)]))
        reads = names[Base.invokelatest(footprint, u, i).unconstrained]
        @test level in reads
        @test count(n -> startswith(string(n), "s."), reads) == 1
    end
    check_gradient(built, plan, u, cell)

    # Only a proven number extrudes: `ones(3)` is an array a window could not
    # broadcast against, and a call of no argument may depend on state.
    constant(ex, params = Symbol[]) = ReactiveKernels._cell_constant_number(ex, params, @__MODULE__)
    @test constant(:(sqrt(2pi))) === Float64
    @test constant(:(1 / 2)) === Float64
    @test constant(:(2 + 3)) === Int
    @test constant(:pi) === Irrational{:π}
    @test constant(:(ones(3))) === nothing
    @test constant(:(rand())) === nothing
    @test constant(:(sqrt(x)), [:x]) === nothing
    @test constant(:x, [:x]) === nothing
    @test constant(:DATA) === nothing

    # The value such an operand gives stays Julia's, by whichever route it takes.
    plan, built = build(quote
        s[1:6] .~ Normal.(0, 1)
        mu = s .* ones(6)
        y .~ Normal.(mu, 1.0)
    end, DATA)
    u = collect(range(-0.4, 0.5; length=6))
    cell = prepare_cell_query(built, plan, :y; locality=:ignore)
    @test [cell(u, i) for i in 1:6] ≈ normal.(DATA.y, u, 1.0)
end

@testset "an LKJ factor's rows are the whole factor's, bit for bit" begin
    for K in 1:6
        P = K * (K - 1) ÷ 2
        u = [-1.2 + 0.35k for k in 1:P]
        z = tanh.(u)
        L = PPL.lkj_chol_constrain(u, K)
        diagonal = [L[j, j] for j in 1:K]
        @test PPL._ppl_lkj_lower(z, diagonal, K, P) == L
        @test PPL._ppl_lkj_upper(z, diagonal, K, P) == permutedims(L)
        for (rows_of, whole) in ((PPL._ppl_lkj_lower_rows, L),
                                 (PPL._ppl_lkj_upper_rows, permutedims(L)))
            for r in 1:K
                @test rows_of(r, z, diagonal, K, P) == whole[r:r, :]
            end
            rows = [K, 1, K, max(1, K - 1)]
            @test rows_of(rows, z, diagonal, K, P) == whole[rows, :]
            @test rows_of(Int[], z, diagonal, K, P) == whole[Int[], :]
        end
    end

    for uplo in ('L', 'U')
        plan, built = build(quote
            sigma ~ Exponential(1.0)
            L ~ LKJCholesky(3, 2.0, $uplo)
            z[1:3] .~ Normal.(0, 1)
            b = L * z
            mu = b[subject]
            y .~ Normal.(mu, sigma)
        end, DATA)
        names = coordinate_names(built.layout)
        u = collect(range(-0.4, 0.5; length=built.layout.total))
        cell = prepare_cell_query(built, plan, :y; locality=:error)
        locality = partition_locality(cell.kernel)
        @test only(e for e in locality if e.value === :L).status === :rows
        partials = [u[position(names, Symbol("L.$k"))] for k in 1:3]
        factor = PPL.lkj_chol_constrain(partials, 3)
        uplo === 'U' && (factor = permutedims(factor))
        draws = [u[position(names, Symbol("z.$k"))] for k in 1:3]
        s = exp(u[position(names, :sigma)])
        for i in eachindex(DATA.y)
            @test cell(u, i) ≈ normal(DATA.y[i], (factor * draws)[DATA.subject[i]], s)
        end
        check_gradient(built, plan, u, cell)
    end

    # A reader of the whole factor keeps it whole: its rows would not serve it.
    plan, built = build(quote
        L ~ LKJCholesky(3, 2.0)
        z[1:3] .~ Normal.(0, 1)
        total = sum(L)
        mu = (L * z)[subject] .+ total
        y .~ Normal.(mu, 1.0)
    end, DATA)
    cell = prepare_cell_query(built, plan, :y; locality=:warn)
    @test only(e for e in partition_locality(cell.kernel) if e.value === :L).status !== :rows
end
end
