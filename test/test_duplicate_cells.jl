# A summed plate whose bound rows repeat runs once per distinct row, weighted
# by its row count (`src/duplicate_cells.jl`, snag `summed-plates-ov-67f319ae`).
module DuplicateCellAcceptance
using ReactiveKernels, Enzyme, DifferentiationInterface, Test
using LogExpFunctions: log1pexp, logistic, logsumexp, logaddexp
include("type_stability_audit.jl")
using .TypeStabilityAudit: findings, ad_call_findings
const RK = ReactiveKernels

# A Bernoulli-logit regression on a bound covariate and response with shared
# coefficients.
@kernel logit_rows(coef::Vector{Float64}, x::Vector{Float64}, y::Vector{Bool}) = begin
    alpha::Float64 = coef[1]
    beta::Float64 = coef[2]
    terms = plate(alpha, beta, x, y) do a, b, z, o
        eta = a + b * z
        -log1pexp(ifelse(o, -eta, eta))
    end
    total::Float64 = sum(terms)
    return total
end
# The same, with coefficients whose types only the exemplar proves.
@kernel logit_untyped(coef::Vector{Float64}, x::Vector{Float64}, y::Vector{Bool}) = begin
    alpha = coef[1]
    beta = coef[2]
    terms = plate(alpha, beta, x, y) do a, b, z, o
        eta = a + b * z
        -log1pexp(ifelse(o, -eta, eta))
    end
    total::Float64 = sum(terms)
    return total
end
# Intercepts gathered through a bound group index.
@kernel logit_groups(q::Vector{Float64}, idx::Vector{Int}, x::Vector{Float64},
                     y::Vector{Bool}) = begin
    intercept = ReactiveKernels._readonly_gather(q, idx)
    slope::Float64 = q[1]
    terms = plate(intercept, slope, x, y) do a, b, z, o
        eta = a + b * z
        -log1pexp(ifelse(o, -eta, eta))
    end
    total::Float64 = sum(terms)
    return total
end
# Per-cell parameters: the cells differ even where the data repeat.
@kernel logit_latent(z::Vector{Float64}, x::Vector{Float64}, y::Vector{Bool}) = begin
    terms = plate(z, x, y) do a, w, o
        eta = a + w
        -log1pexp(ifelse(o, -eta, eta))
    end
    total::Float64 = sum(terms)
    return total
end

# A whole-event mixture: one component governs a whole (ragged) history, and
# each position's lazy branch reads the history. The capture-recapture shape:
# the varying operand is one bound `Vector{Bool}` per cell.
@kernel event_law(observed::Vector{Bool}, P::Matrix{Float64}, w::Vector{Float64}) = begin
    components = plate(1:length(w)) do k
        cells = plate(1:length(observed)) do i
            cell::Float64 = observed[i] ? log(P[i, k]) : log1p(-P[i, k])
            cell
        end
        sum(cells)
    end
    log_weights = log.(w)
    weighted = plate(components, log_weights) do s, lw
        s + lw
    end
    return logsumexp(weighted)
end
@kernel histories_model(p::Vector{Float64}, histories::Vector{Vector{Bool}},
                        absent::Vector{Float64}, omega::Float64) = begin
    terms = plate(histories) do h
        P = hcat(p, absent)
        w = [omega, 1 - omega]
        event_law(h, P, w)
    end
    total::Float64 = sum(terms)
    return total
end
# A lane index whose bound gather preparation caches: the cell no longer
# reads the index, only the gathered value.
@kernel indexed_logs(s::Vector{Float64}, values::Vector{Float64},
                     idx::Vector{Int}) = begin
    terms = plate(idx) do i
        cell::Float64 = log(values[i] * s[1])
        cell
    end
    total::Float64 = sum(terms)
    return total
end

# A gathered operand the cell ignores: the gather still reads every row.
@kernel unread_gather(q::Vector{Float64}, idx::Vector{Int}, x::Vector{Float64}) = begin
    gathered = ReactiveKernels._readonly_gather(q, idx)
    terms = plate(gathered, x) do a, z
        cell::Float64 = log(z * q[1])
        cell
    end
    total::Float64 = sum(terms)
    return total
end

function histories_oracle(p, histories, absent, omega)
    sum(histories; init = 0.0) do h
        a = sum(i -> h[i] ? log(p[i]) : log1p(-p[i]), eachindex(h); init = 0.0)
        b = sum(i -> h[i] ? log(absent[i]) : log1p(-absent[i]), eachindex(h); init = 0.0)
        logaddexp(log(omega) + a, log(1 - omega) + b)
    end
end

function rows_oracle(coef, x, y)
    value = 0.0; g = zero(coef)
    for i in eachindex(x)
        s = (y[i] ? -1 : 1) * (coef[1] + coef[2] * x[i])
        value -= log1pexp(s)
        d = -logistic(s) * (y[i] ? -1 : 1)
        g[1] += d; g[2] += d * x[i]
    end
    value, g
end
function groups_oracle(q, idx, x, y)
    value = 0.0; g = zero(q)
    for i in eachindex(x)
        s = (y[i] ? -1 : 1) * (q[idx[i]] + q[1] * x[i])
        value -= log1pexp(s)
        d = -logistic(s) * (y[i] ? -1 : 1)
        g[idx[i]] += d; g[1] += d * x[i]
    end
    value, g
end
grouped(k) = any(op -> op === RK._cell_multiplicity_weighted, RK._kernel_ops(k))
# The row counts of the grouped plates: the constants named `<cells>_multiplicity`.
multiplicities(k) = [r.op.value for r in getfield(k, :lowered_recipes)
                     if r.op isa RK._BoundConstant &&
                        endswith(string(only(r.outputs).name), "_multiplicity")]
function verify(k, u, ref; active = :coef)
    pristine = copy(u)
    @test k(u) ≈ ref[1] rtol = 1e-12
    ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), u; active,
                    native_forward_batch = false)
    g = zero(u); v, _ = ad_value_and_gradient!(ad, g, u)
    @test v ≈ ref[1] rtol = 1e-12
    @test g ≈ ref[2] rtol = 1e-11 atol = 1e-11
    @test u == pristine
end

# 120 rows, 12 distinct (x, y) pairs.
const X = repeat([-1.0, 0.0, 0.5, 2.0], 30)
const Y = repeat([true, false, false], 40)

@testset "duplicate cells: one weighted cell per distinct row" begin
    coef = [0.3, -0.7]
    x, y = copy(X), copy(Y)
    k = prepare(logit_rows; bound = (; x, y))
    @test grouped(k)
    counts = only(multiplicities(k))
    @test sum(counts) == length(x)
    @test length(counts) == length(unique(zip(x, y)))
    verify(k, coef, rows_oracle(coef, x, y))
    verify(k, [-1.2, 0.4], rows_oracle([-1.2, 0.4], x, y))
    @test x == X && y == Y
    @test isempty(findings(k, (Vector{Float64},)))
    ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), coef; active = :coef,
                    native_forward_batch = false)
    @test isempty(ad_call_findings(ad, coef))
end

@testset "duplicate cells: rebinding shares the grouped template" begin
    cache = PreparationCache(); coef = [0.3, -0.7]
    k1 = prepare!(cache, logit_rows; bound = (x = copy(X), y = copy(Y)))
    x2 = repeat([1.5, -0.5], 50); y2 = repeat([true, false, true, true, false], 20)
    k2 = prepare!(cache, logit_rows; bound = (x = x2, y = y2))
    @test grouped(k1) && grouped(k2)
    @test k1.f === k2.f
    verify(k2, coef, rows_oracle(coef, x2, y2))
    # Earlier kernels keep their own groups after a rebinding.
    verify(k1, coef, rows_oracle(coef, X, Y))
    # Rows that barely repeat keep the plate's own loop: another program.
    x3 = [sin(i / 3) for i in 1:60]; y3 = isodd.(1:60)
    k3 = prepare!(cache, logit_rows; bound = (x = x3, y = y3))
    @test !grouped(k3)
    @test k3.f !== k1.f
    verify(k3, coef, rows_oracle(coef, x3, y3))
    # No rows: the plate's own loop, which adds nothing.
    k0 = prepare!(cache, logit_rows; bound = (x = Float64[], y = Bool[]))
    @test !grouped(k0)
    @test k0(coef) == 0.0
end

@testset "duplicate cells: gathers through a bound index" begin
    q = [0.2, -0.4, 0.9, 0.1]
    idx = repeat([1, 2, 3, 4, 2], 24); x = repeat([0.5, -1.0, 0.5], 40)
    y = repeat([true, false], 60)
    k = prepare(logit_groups; bound = (; idx, x, y))
    @test grouped(k)
    @test length(only(multiplicities(k))) == length(unique(zip(idx, x, y)))
    # The gather's index runs no longer apply.
    @test !any(op -> op === RK._readonly_gather_runs, RK._kernel_ops(k))
    verify(k, q, groups_oracle(q, idx, x, y); active = :q)
    verify(k, -q, groups_oracle(-q, idx, x, y); active = :q)
    @test isempty(findings(k, (Vector{Float64},)))
    # An out-of-range index is still read.
    invalid = prepare(logit_groups; bound = (idx = fill(5, 32), x = fill(0.5, 32),
                                             y = fill(true, 32)))
    @test grouped(invalid)
    @test_throws BoundsError invalid(q)
end

@testset "duplicate cells: rows of per-cell arrays and unread lane indices" begin
    # 40 ragged histories, 4 distinct.
    base = [Bool[0, 0, 0, 0], Bool[0, 1, 1, 0], Bool[1, 0], Bool[0, 0, 1]]
    histories = base[repeat([1, 2, 1, 3, 1, 4, 2, 1, 3, 1], 4)]
    absent = zeros(4); omega = 0.7
    k = prepare(histories_model; bound = (; histories, absent, omega))
    @test grouped(k)
    @test sort(only(multiplicities(k))) ==
          sort([count(==(h), histories) for h in unique(histories)])
    p = [0.3, 0.45, 0.6, 0.2]
    value = histories_oracle(p, histories, absent, omega)
    @test k(p) ≈ value rtol = 1e-12
    ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), p; active = :p,
                    native_forward_batch = false)
    g = zero(p); v, _ = ad_value_and_gradient!(ad, g, p)
    @test v ≈ value rtol = 1e-12
    step = 1e-6
    fd = map(eachindex(p)) do i
        e = zeros(length(p)); e[i] = step
        (histories_oracle(p .+ e, histories, absent, omega) -
         histories_oracle(p .- e, histories, absent, omega)) / (2step)
    end
    @test g ≈ fd rtol = 1e-6
    @test isempty(findings(k, (Vector{Float64},)))
    # A position no history takes keeps its arm untaken: at p[1] = 1 every
    # history must capture at position 1, and the gradient stays finite.
    captured = [Bool[1, 0], Bool[1, 1], Bool[1, 0], Bool[1, 0]]
    k = prepare(histories_model; bound = (histories = captured, absent = [0.5, 0.5],
                                          omega = 0.6))
    @test grouped(k)
    ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), [1.0, 0.4]; active = :p,
                    native_forward_batch = false)
    g = zeros(2); ad_value_and_gradient!(ad, g, [1.0, 0.4])
    @test all(isfinite, g)

    # The cell reads `values[i]`, which preparation caches per row, and not
    # `i`: rows group by the cached value.
    values = [2.0, 3.0, 2.0]
    idx = repeat([1, 2, 3, 1], 6)
    k = prepare(indexed_logs; bound = (; values, idx))
    @test grouped(k)
    @test sort(only(multiplicities(k))) == [6, 18]
    @test k([0.5]) ≈ sum(log(values[i] * 0.5) for i in idx) rtol = 1e-13
    ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), [0.5]; active = :s,
                    native_forward_batch = false)
    g = zeros(1); ad_value_and_gradient!(ad, g, [0.5])
    @test only(g) ≈ length(idx) / 0.5 rtol = 1e-12
    # Its first throwing row still throws.
    bad = prepare(indexed_logs; bound = (values = [2.0, -1.0], idx = repeat([1, 2], 8)))
    @test grouped(bad)
    @test_throws DomainError bad([0.5])

    # A gather the cell ignores still tells rows apart: every index is read,
    # and an out-of-range one still throws.
    x = repeat([2.0, 3.0], 12)
    k = prepare(unread_gather; bound = (idx = repeat([1, 2, 3], 8), x))
    @test length(only(multiplicities(k))) == length(unique(zip(repeat([1, 2, 3], 8), x)))
    @test k([0.5, 1.0, 2.0]) ≈ sum(log.(x .* 0.5)) rtol = 1e-13
    invalid = prepare(unread_gather; bound = (idx = vcat(fill(1, 23), 9), x))
    @test_throws BoundsError invalid([0.5, 1.0, 2.0])
end

@testset "duplicate cells: which plates are grouped" begin
    coef = [0.3, -0.7]
    x, y = copy(X), copy(Y)
    # Untyped shared operands: the exemplar proves them numbers.
    k = prepare(logit_untyped; bound = (; x, y), exemplars = (coef,))
    @test grouped(k)
    verify(k, coef, rows_oracle(coef, x, y))
    # Without it they are not proven shared, and the plate keeps its loop.
    k = prepare(logit_untyped; bound = (; x, y))
    @test !grouped(k)
    verify(k, coef, rows_oracle(coef, x, y))
    # `simplify = false` keeps the plate's own loop.
    k = prepare(logit_rows; bound = (; x, y), simplify = false)
    @test !grouped(k)
    verify(k, coef, rows_oracle(coef, x, y))
    # A plate read cell by cell keeps its loop.
    query = prepare(logit_rows; bound = (; x, y), want = (:terms, :total))
    @test !grouped(query)
    terms, total = query(coef)
    @test length(terms) == length(x)
    @test total ≈ rows_oracle(coef, x, y)[1] rtol = 1e-12
    # A per-cell parameter makes every cell its own.
    z = collect(range(-1, 1; length = length(x)))
    k = prepare(logit_latent; bound = (; x, y))
    @test !grouped(k)
    @test k(z) ≈ sum(-log1pexp((y[i] ? -1 : 1) * (z[i] + x[i])) for i in eachindex(x))
    # Signed zeros stay apart; NaNs compare equal.
    @test RK._cell_group_lanes(([0.0, -0.0, 0.0, -0.0], trues(4))) == ([1, 2], [2, 2])
    @test RK._cell_group_lanes(([NaN, 1.0, NaN],)) == ([1, 2], [2, 1])
end

@testset "duplicate cells: retained loops across data sizes" begin
    inventories = Dict{Symbol,Int}[]
    for (distinct, copies) in ((4, 8), (40, 8), (400, 16))
        x = repeat(collect(range(-2, 2; length = distinct)), copies)
        y = repeat(isodd.(1:distinct), copies)
        k = prepare(logit_rows; bound = (; x, y))
        @test grouped(k)
        @test length(only(multiplicities(k))) == distinct
        counts = Dict{Symbol,Int}()
        RK._index_run_visit(k.ast) do ex
            counts[ex.head] = get(counts, ex.head, 0) + 1
        end
        @test get(counts, :while, 0) > 0
        push!(inventories, counts)
        verify(k, [0.3, -0.7], rows_oracle([0.3, -0.7], x, y))
    end
    @test all(==(first(inventories)), inventories)
end
end
