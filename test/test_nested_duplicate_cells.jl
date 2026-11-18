# A summed plate nested in another plate's cell runs once per distinct row of
# its bind-time caches, per lane of the enclosing plates, weighted by its row
# count; a size query of a live array whose shape is bound is a bind-time value
# (`src/nested_duplicate_cells.jl`, `src/bound_shapes.jl`, snag
# `butterfly-positi-b8e56042`).
module NestedDuplicateCellAcceptance
using ReactiveKernels, Enzyme, DifferentiationInterface, Test
using LogExpFunctions: log1pexp, logsumexp, logaddexp
const RK = ReactiveKernels

# A whole-event two-component law, written as the product-mixture laws write
# theirs: each component's log-odds form one column of a matrix with one row,
# or one row per position. A position with a positive count is a success.
@kernel event_law(observed::Vector{Int}, logits::Matrix{Float64},
                  weights::Vector{Float64}) = begin
    checked = (size(logits, 1) == 1 || size(logits, 1) == length(observed)) ?
        logits : throw(DimensionMismatch("logits need one row or one per position"))
    nrows = size(checked, 1)
    components = plate(1:size(checked, 2)) do k
        cells = plate(1:length(observed)) do i
            row = ifelse(nrows == 1, 1, i)
            l = checked[row, k]
            y = observed[i]
            cell::Float64 = y > 0 ? -log1pexp(-l) : -log1pexp(l)
            cell
        end
        sum(cells)
    end
    log_weights = log.(weights)
    weighted = plate(components, log_weights) do s, lw
        s + lw
    end
    return logsumexp(weighted)
end

# Each event's first component has the same log-odds at every position; the
# second never succeeds (log-odds `-Inf`, bound as data).
@kernel filled_model(a::Vector{Float64}, events::Vector{Vector{Int}},
                     never::Vector{Vector{Float64}}, omega::Float64) = begin
    terms = plate(a, events, never) do ae, h, nv
        P = hcat(fill(ae, length(h)), nv)
        event_law(h, P, [omega, 1 - omega])
    end
    total::Float64 = sum(terms)
    return total
end
# The same with one row per event.
@kernel row_model(a::Vector{Float64}, events::Vector{Vector{Int}}, omega::Float64) = begin
    terms = plate(a, events) do ae, h
        P = [ae -Inf]
        event_law(h, P, [omega, 1 - omega])
    end
    total::Float64 = sum(terms)
    return total
end
# Positions with their own log-odds: no two positions are alike.
@kernel position_model(a::Vector{Float64}, events::Vector{Vector{Int}},
                       offsets::Vector{Vector{Float64}}, never::Vector{Vector{Float64}},
                       omega::Float64) = begin
    terms = plate(a, events, offsets, never) do ae, h, o, nv
        P = hcat(ae .+ o, nv)
        event_law(h, P, [omega, 1 - omega])
    end
    total::Float64 = sum(terms)
    return total
end

function oracle(a, events, omega; offsets = nothing)
    sum(eachindex(events); init = 0.0) do e
        h = events[e]
        member = sum(eachindex(h); init = 0.0) do i
            l = a[e] + (offsets === nothing ? 0.0 : offsets[e][i])
            h[i] > 0 ? -log1pexp(-l) : -log1pexp(l)
        end
        outsider = any(>(0), h) ? -Inf : 0.0
        logaddexp(log(omega) + member, log1p(-omega) + outsider)
    end
end
function oracle_gradient(a, events, omega; offsets = nothing)
    step = 1e-6
    map(eachindex(a)) do i
        e = zeros(length(a)); e[i] = step
        (oracle(a .+ e, events, omega; offsets) - oracle(a .- e, events, omega; offsets)) /
            (2step)
    end
end

grouped(k) = any(op -> op === RK._cell_multiplicity_weighted, RK._kernel_ops(k))
# The row counts of each event's grouped cells, nested as the plates are.
multiplicities(k) = [r.op.value for r in getfield(k, :lowered_recipes)
                     if r.op isa RK._BoundConstant &&
                        endswith(string(only(r.outputs).name), "_multiplicity")]

function verify(k, a, events, omega; offsets = nothing)
    pristine = copy(a)
    value = oracle(a, events, omega; offsets)
    @test k(a) ≈ value rtol = 1e-12
    ad = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), a; active = :a,
                    native_forward_batch = false)
    g = zero(a); v, _ = ad_value_and_gradient!(ad, g, a)
    @test v ≈ value rtol = 1e-12
    @test g ≈ oracle_gradient(a, events, omega; offsets) rtol = 1e-6 atol = 1e-8
    @test a == pristine
    g
end

# 12 events of 6 positions: 5 never detected, the others with few distinct
# counts.
const EVENTS = [[0, 2, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0], [3, 3, 0, 0, 0, 3],
                [0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0],
                [0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0], [0, 1, 0, 2, 0, 0], [5, 5, 5, 5, 5, 5]]
const A = [0.3 * sin(e) - 0.2 for e in eachindex(EVENTS)]

@testset "nested duplicate cells: a filled column and a bound one" begin
    events = deepcopy(EVENTS)
    never = [fill(-Inf, length(h)) for h in events]
    omega = 0.6
    k = prepare(filled_model; bound = (; events, never, omega))
    @test grouped(k)
    counts = only(multiplicities(k))
    # No cache varies by component, so both components of an event share one
    # count vector: one count per distinct count of the event, summing to its
    # positions.
    @test length(counts) == length(events)
    for (e, h) in enumerate(events)
        @test sum(counts[e]) == length(h)
        @test length(counts[e]) == length(unique(h))
    end
    g = verify(k, A, events, omega)
    verify(k, -2 .* A, events, omega)
    # Saturated log-odds keep their tails.
    verify(k, fill(40.0, length(A)), events, omega)
    @test events == EVENTS
    # `simplify = false` keeps every position, with the same value and gradient.
    plain = prepare(filled_model; bound = (; events, never, omega), simplify = false)
    @test !grouped(plain)
    @test verify(plain, A, events, omega) ≈ g rtol = 1e-12
end

@testset "nested duplicate cells: one row per event" begin
    events = deepcopy(EVENTS)
    omega = 0.35
    k = prepare(row_model; bound = (; events, omega))
    @test grouped(k)
    verify(k, A, events, omega)
end

@testset "nested duplicate cells: what keeps every position" begin
    events = deepcopy(EVENTS)
    never = [fill(-Inf, length(h)) for h in events]
    # Per-position log-odds tell positions apart.
    offsets = [0.1 .* collect(1:length(h)) for h in events]
    k = prepare(position_model; bound = (; events, offsets, never, omega = 0.6))
    @test !grouped(k)
    verify(k, A, events, 0.6; offsets)
    # A bound column whose rows differ is read at its own rows.
    varied = [[-Inf, -Inf, 0.0, -Inf, -Inf, -Inf] for _ in events]
    k = prepare(filled_model; bound = (; events, never = varied, omega = 0.6))
    @test k(A) ≈ sum(eachindex(events)) do e
        h = events[e]
        member = sum(i -> h[i] > 0 ? -log1pexp(-A[e]) : -log1pexp(A[e]), eachindex(h))
        outsider = sum(i -> h[i] > 0 ? -log1pexp(-varied[e][i]) : -log1pexp(varied[e][i]),
                       eachindex(h))
        logaddexp(log(0.6) + member, log(0.4) + outsider)
    end rtol = 1e-12
    # Events whose positions barely repeat keep their loops.
    distinct = [collect(0:5) for _ in 1:4]
    k = prepare(filled_model; bound = (events = distinct,
        never = [fill(-Inf, 6) for _ in 1:4], omega = 0.6))
    @test !grouped(k)
    verify(k, A[1:4], distinct, 0.6)
    # Operands whose rows differ still throw in every evaluation.
    short = [h[1:5] for h in events]
    k = prepare(filled_model; bound = (events = short, never, omega = 0.6))
    @test_throws DimensionMismatch k(A)
end

@testset "nested duplicate cells: empty events and rebinding" begin
    cache = PreparationCache()
    events = deepcopy(EVENTS)
    never = [fill(-Inf, length(h)) for h in events]
    k1 = prepare!(cache, filled_model; bound = (; events, never, omega = 0.6))
    with_empty = [events[1:3]; [Int[]]; events[4:end]]
    k2 = prepare!(cache, filled_model;
        bound = (events = with_empty, never = [fill(-Inf, length(h)) for h in with_empty],
                 omega = 0.6))
    @test grouped(k1) && grouped(k2)
    @test k1.f === k2.f
    a2 = [A[1:3]; 0.7; A[4:end]]
    verify(k2, a2, with_empty, 0.6)
    verify(k1, A, events, 0.6)
end
end
