module CellFactorNestedTests
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme, SecondOrder
using ReactiveKernels: partition_locality

# A sum read by the summands of another sum: a dose total scaling a density's
# row-wise location. The density total stays in the outer factor, reading the
# dose as one number, so neither factor reads the other's coordinates.
const NAMED = @rkppl begin
    sigma ~ Exponential(1)
    e[1:E] .~ Normal.(0, 1)
    w[1:E] .~ Normal.(0, 1)
    eta = 2 .* w
    @plate for i in eachindex(y)
        dose = sum(amount[events[i]] .* exp.(e[events[i]]))
        loc = 0.01 * dose .* exp.(eta[obs[i]])
        y[i] .~ Normal.(loc, sigma)
    end
end
const INLINE = @rkppl begin
    sigma ~ Exponential(1)
    e[1:E] .~ Normal.(0, 1)
    w[1:E] .~ Normal.(0, 1)
    eta = 2 .* w
    @plate for i in eachindex(y)
        loc = 0.01 .* exp.(eta[obs[i]]) .* sum(amount[events[i]] .* exp.(e[events[i]]))
        y[i] .~ Normal.(loc, sigma)
    end
end
# The dose in an earlier plate, as BRM emits a location its observation reads.
const EARLIER = @rkppl begin
    sigma ~ Exponential(1)
    e[1:E] .~ Normal.(0, 1)
    w[1:E] .~ Normal.(0, 1)
    eta = 2 .* w
    @plate for i in eachindex(events)
        dose = sum(amount[events[i]] .* exp.(e[events[i]]))
        loc_cell = (0.01dose) .* exp.(eta[obs[i]])
        loc[i] = loc_cell
    end
    @plate for i in eachindex(y)
        y[i] .~ Normal.(loc[i], sigma)
    end
end

const REVERSE = AutoEnzyme(; mode = Enzyme.Reverse)
const SECOND = SecondOrder(REVERSE, AutoEnzyme(; mode = Enzyme.Forward))

# Registered Enzyme's gap, as in test_cell_factors.jl. Dogfood validation sets
# RK_TEST_REQUIRE_STATIC_ENZYME=1, which makes it fatal.
function static(f)
    try
        f()
    catch err
        err isa Enzyme.Compiler.EnzymeRuntimeActivityError || rethrow()
        get(ENV, "RK_TEST_REQUIRE_STATIC_ENZYME", "0") == "1" && rethrow()
        @test occursin("Mismatched activity", sprint(showerror, err))
        @test_broken false
        nothing
    end
end

# Cell 2 repeats an event, cell 3 has no events, cell 4 no observations.
function data(n)
    E = 2n
    events = [[2c - 1, 2c] for c in 1:n]
    obs = [[2c, 2c - 1] for c in 1:n]
    events[2] = [3, 3, 4]
    events[3] = Int[]
    obs[4] = Int[]
    amount = 1.0 .+ collect(1:E) ./ E
    y = [0.1 .* c .+ 0.05 .* collect(eachindex(obs[c])) for c in 1:n]
    (; E, events, obs, amount, y)
end

function positions(built)
    names = string.(coordinate_names(built.layout))
    at(name) = findfirst(==(name), names)
    E = count(startswith("e."), names)
    (; sigma = at("sigma"), e = [at("e.$k") for k in 1:E], w = [at("w.$k") for k in 1:E])
end

# The density, with no RK code.
function oracle(u, d, at, c)
    sigma = exp(u[at.sigma])
    dose = sum((d.amount[r] * exp(u[at.e[r]]) for r in d.events[c]); init = 0.0)
    sum((logpdf(Normal(0.01 * dose * exp(2 * u[at.w[r]]), sigma), d.y[c][t])
         for (t, r) in enumerate(d.obs[c])); init = 0.0)
end

function central(f, u, k; h = 1e-5)
    a, b = copy(u), copy(u)
    a[k] += h
    b[k] -= h
    (f(a) - f(b)) / 2h
end

function prepared(model, n; kw...)
    d = data(n)
    bound = model(; d.E, d.events, d.obs, d.amount) | (; d.y)
    built = build_kernel(bound)
    u = collect(range(-0.2, 0.3; length = built.layout.total))
    f = prepare_cell_factors(built, bound, :y; locality = :error, kw...)
    (; d, bound, built, u, f, at = positions(built))
end

struct ReadPoint <: AbstractVector{Float64}
    data::Vector{Float64}
    reads::Vector{Int}
end
Base.size(u::ReadPoint) = size(u.data)
Base.IndexStyle(::Type{ReadPoint}) = IndexLinear()
Base.getindex(u::ReadPoint, i::Int) = (push!(u.reads, i); u.data[i])

support(fp, u, args...) = sort(Base.invokelatest(fp, u, args...).unconstrained)
nodes(x) = x isa Expr ? 1 + sum(nodes, x.args; init = 0) : 1

options(n, u) = (("uncached", (;)),
                 ("partitioned", (; partitioned = (:w => collect(1:2n),), u0 = copy(u))))

@testset "a sum its neighbour's summands read is the cell's factor ($name)" for (name, model) in
        (("named", NAMED), ("inline", INLINE), ("earlier plate", EARLIER))
    n = 6
    base = prepared(model, n)
    for (label, kw) in options(n, base.u)
        (; d, u, f, at) = prepared(model, n; kw...)
        # The density total reads the dose, so only the dose is a reduction.
        @test length(cell_reductions(f)) == 1
        term = prepare_cell_factor_query(f, 1)
        outer = prepare_cell_factor_query(f, :outer)
        for c in 1:n
            @test reduction_rows(f, 1, c) == d.events[c]
            terms = [term(u, c, t) for t in eachindex(d.events[c])]
            # The dose's terms carry no density and no 0.01.
            @test terms ≈ [d.amount[r] * exp(u[at.e[r]]) for r in d.events[c]]
            @test outer(u, [sum(terms; init = 0.0)], c) ≈ oracle(u, d, at, c)
        end
    end
end

@testset "neither factor computes the other's values, at any size ($name)" for (name, model) in
        (("named", NAMED), ("inline", INLINE), ("earlier plate", EARLIER))
    sizes = map((6, 48)) do n
        (; d, u, f, at) = prepared(model, n)
        term = prepare_cell_factor_query(f, 1)
        outer = prepare_cell_factor_query(f, :outer)
        @test !any(e -> e.status === :whole, partition_locality(outer.kernel))
        @test !any(e -> e.status === :whole, partition_locality(term.kernel))
        term_fp = partition_footprint_kernel(term.kernel)
        outer_fp = partition_footprint_kernel(outer.kernel)
        for c in (1, 2, 5)
            direct = sort([at.sigma; at.w[d.obs[c]]])
            @test support(outer_fp, u, [0.7], c) == direct
            point = ReadPoint(u, Int[])
            outer(point, [0.7], c)
            @test sort(unique(point.reads)) == direct
            for (t, r) in enumerate(d.events[c])
                @test support(term_fp, u, c, t) == [at.e[r]]
                point = ReadPoint(u, Int[])
                term(point, c, t)
                @test unique(point.reads) == [at.e[r]]
            end
        end
        (nodes(code_expr(outer.kernel)), nodes(code_expr(term.kernel)))
    end
    @test sizes[1] == sizes[2]
end

@testset "empty sums and rows, a repeated row" begin
    n = 6
    (; d, u, f, at) = prepared(NAMED, n)
    term = prepare_cell_factor_query(f, 1)
    outer = prepare_cell_factor_query(f, :outer)
    @test isempty(reduction_rows(f, 1, 3))
    @test outer(u, [0.0], 3) ≈ oracle(u, d, at, 3)
    # No observations: an empty density total, however large the dose.
    @test outer(u, [5.0], 4) == 0.0
    @test reduction_rows(f, 1, 2) == [3, 3, 4]
    @test term(u, 2, 1) == term(u, 2, 2)
end

@testset "chain rule through the outer factor" begin static() do
    n = 6
    (; d, u, f, at) = prepared(NAMED, n)
    N = length(u)
    term_indices(c, t) = [at.e[d.events[c][t]]]
    outer_indices(c) = sort([at.sigma; at.w[d.obs[c]]])
    indices = outer_indices(1)
    width = length(indices)
    term_g = prepare_cell_factor_ad(prepare_ad, f, 1, u, term_indices(1, 1), REVERSE)
    term_h = prepare_cell_factor_ad(prepare_ad_hvp, f, 1, u, term_indices(1, 1), SECOND, ([1.0],))
    outer_g = prepare_cell_factor_ad(prepare_ad, f, :outer, u, indices, REVERSE)
    units = ntuple(width + 1) do i
        unit = Float64[j == i for j in 1:width+1]
        (unit[1:1], unit[2:end])
    end
    outer_h = prepare_cell_factor_ad(prepare_ad_hvp, f, :outer, u, indices, SECOND, units)
    gradient_at(x, c) = [central(y -> oracle(y, d, at, c), x, k) for k in 1:N]
    for c in (1, 2, 5)
        J, Hs, s = zeros(N), zeros(N, N), 0.0
        for t in eachindex(d.events[c])
            k = term_indices(c, t)
            value, gradient = ad_value_and_gradient(term_g, u, u[k], k, c, t)
            _, (column,) = ad_gradient_and_hvp(term_h, ([1.0],), u, u[k], k, c, t)
            s += value
            J[k] .+= gradient
            Hs[k, k] .+= column
        end
        direct = outer_indices(c)
        point = (u, [s], u[direct], direct, c)
        value, (F_s, F_u) = ad_value_and_gradient(outer_g, point...)
        _, columns = ad_gradient_and_hvp(outer_h, units, point...)
        @test value ≈ oracle(u, d, at, c)
        gradient = J .* only(F_s)
        gradient[direct] .+= F_u
        @test gradient ≈ gradient_at(u, c) rtol = 1e-7 atol = 1e-7
        # H = Jᵀ F_ss J + F_s ∇²s, with the mixed sum and direct blocks.
        F_ss = only(columns[1][1])
        F_su = columns[1][2]
        F_uu = reduce(hcat, (columns[i + 1][2] for i in 1:width))
        H = F_ss .* (J * J') .+ only(F_s) .* Hs
        H[direct, :] .+= F_su * J'
        H[:, direct] .+= J * F_su'
        H[direct, direct] .+= F_uu
        step(k) = 1e-4 .* (1:N .== k)
        numeric = reduce(hcat, ((gradient_at(u .+ step(k), c) .- gradient_at(u .- step(k), c)) ./ 2e-4
                                for k in 1:N))
        @test H ≈ numeric rtol = 1e-4 atol = 1e-4
    end
end end

# Under a shared prefix a factor differentiates its partitioned coordinates only.
@testset "derivatives under a shared prefix" begin static() do
    n = 6
    base = prepared(NAMED, n)
    @testset "the outer factor over the observations' coordinates" begin
        (; d, u, f, at) = prepared(NAMED, n; partitioned = (:w => collect(1:2n),), u0 = copy(base.u))
        outer = prepare_cell_factor_query(f, :outer)
        term = prepare_cell_factor_query(f, 1)
        g = prepare_cell_factor_ad(prepare_ad, f, :outer, u, sort(at.w[d.obs[1]]), REVERSE)
        for c in (1, 2, 5)
            s = sum((term(u, c, t) for t in eachindex(d.events[c])); init = 0.0)
            direct = sort(at.w[d.obs[c]])
            value, (F_s, F_u) = ad_value_and_gradient(g, u, [s], u[direct], direct, c)
            @test value ≈ oracle(u, d, at, c)
            @test F_u ≈ [central(x -> oracle(x, d, at, c), u, k) for k in direct] rtol = 1e-7
            @test only(F_s) ≈ (outer(u, [s + 1e-5], c) - outer(u, [s - 1e-5], c)) / 2e-5 rtol = 1e-7
        end
    end
    @testset "a summand over the sum's coordinates" begin
        (; d, u, f, at) = prepared(NAMED, n; partitioned = (:e => collect(1:2n),), u0 = copy(base.u))
        g = prepare_cell_factor_ad(prepare_ad, f, 1, u, [at.e[1]], REVERSE)
        for c in (1, 2, 5), (t, r) in enumerate(d.events[c])
            k = [at.e[r]]
            value, gradient = ad_value_and_gradient(g, u, u[k], k, c, t)
            @test value ≈ d.amount[r] * exp(u[at.e[r]])
            @test gradient ≈ [value]
        end
    end
end end

# The locality rule still covers what stays in the outer factor. Compiler guide,
# "Locality": a value every reader needs at some rows is computed at those rows or
# named. Sparse Laplace relies on it: a silent whole read couples every cell.
opaque_row(x) = reverse(x)
const OPAQUE = @rkppl begin
    sigma ~ Exponential(1)
    e[1:E] .~ Normal.(0, 1)
    w[1:E] .~ Normal.(0, 1)
    eta = 2 .* w
    @plate for i in eachindex(y)
        dose = sum(amount[events[i]] .* exp.(e[events[i]]))
        loc = 0.01 * dose .* opaque_row(eta) .* amount[obs[i]]
        y[i] .~ Normal.(loc, sigma)
    end
end
@testset "an operand with no row form stays refused in the outer factor" begin
    d = data(6)
    bound = OPAQUE(; d.E, d.events, d.obs, d.amount) | (; d.y)
    built = build_kernel(bound)
    f = prepare_cell_factors(built, bound, :y; locality = :error)
    @test length(cell_reductions(f)) == 1
    err = try
        prepare_cell_factor_query(f, :outer)
        nothing
    catch caught
        caught
    end
    @test err isa PartitionLocalityError
    @test occursin("opaque_row", sprint(showerror, err))
end

# A sum that reads a sum another sum reads has no known type where the total asks
# for it at rows, so the outer factor is refused, naming it. The values it would
# give are not in question; this records the limit.
const TWO_LEVELS = @rkppl begin
    sigma ~ Exponential(1)
    e[1:E] .~ Normal.(0, 1)
    w[1:E] .~ Normal.(0, 1)
    eta = 2 .* w
    @plate for i in eachindex(y)
        inner = sum(exp.(e[events[i]]))
        middle = sum(inner .* eta[events[i]])
        y[i] .~ Normal.(middle .* amount[obs[i]], sigma)
    end
end
@testset "two levels of nesting" begin
    d = data(6)
    bound = TWO_LEVELS(; d.E, d.events, d.obs, d.amount) | (; d.y)
    built = build_kernel(bound)
    f = prepare_cell_factors(built, bound, :y; locality = :error)
    @test length(cell_reductions(f)) == 1
    err = try
        prepare_cell_factor_query(f, :outer)
        nothing
    catch caught
        caught
    end
    @test_broken err === nothing
    err === nothing || @test occursin("middle", sprint(showerror, err)) &&
                             occursin("sum(", sprint(showerror, err))
end

# Sums at one level stay separate factors: nothing here reads another sum.
const SIBLINGS = @rkppl begin
    sigma ~ Exponential(1)
    e[1:E] .~ Normal.(0, 1)
    @plate for k in eachindex(y)
        y[k] ~ Normal(sum(exp.(e[r1[k]])) + sum(e[r2[k]] .^ 2), sigma)
    end
end
@testset "sums that read no other sum stay reductions" begin
    bound = SIBLINGS(; E = 4, r1 = [[1, 2], [3]], r2 = [[4], [1, 2]]) | (; y = [0.1, 0.2])
    built = build_kernel(bound)
    f = prepare_cell_factors(built, bound, :y; locality = :error)
    @test length(cell_reductions(f)) == 2
end

end
