module PlateConsumerNativeADTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

pack_lanes(lanes) = vec(stack(lanes))
@kernel weighted_columns(q, X, weights) = begin
    lanes = plate(eachcol(X), Ref(q)) do xs, p
        p[1] .* xs
    end
    packed = pack_lanes(lanes)
    total = sum(packed .* weights)
    return total
end

@kernel weighted_broadcast(q, x, weights) = begin
    total = sum((q[1] .* x) .* weights)
    return total
end

@kernel split_arm_scale(y::Vector{Float64}, scale::Float64) = begin
    pointwise = plate(y, scale) do yi, si
        cell::Float64 = yi <= 0 ? (si > 0 ? log(si) : -Inf) : si * yi
        cell
    end
    total::Float64 = sum(pointwise)
end

@testset "bound split-arm plates under unannotated native Reverse" begin
    datasets = (Float64[], [0.0, 1.0, 2.0], repeat([0.0, 1.0, 2.0], 11),
                [-2.0, 0.0, -1.0], [1.0, 2.0, 3.0])
    for y in datasets
        saved_y = copy(y)
        k = prepare(split_arm_scale; have=(:y, :scale), want=:total, bound=(; y))
        for scale in (1.3, -0.4, 0.0, 3.1)
            expected, derivative = 0.0, 0.0
            for yi in y
                if yi <= 0
                    expected += scale > 0 ? log(scale) : -Inf
                    derivative += scale > 0 ? inv(scale) : 0.0
                else
                    expected += scale * yi
                    derivative += yi
                end
            end
            @test k(scale) ≈ expected
            @test y == saved_y
            @test only(Enzyme.gradient(Enzyme.Reverse, k, scale)) ≈ derivative
            @test y == saved_y
        end
    end
end

# Plain Julia control with fresh output and a runtime loop, independent of RK.
function weighted_loop(q, X, weights)
    packed = similar(X, eltype(q), length(X))
    for i in eachindex(X)
        packed[i] = q[1] * X[i]
    end
    sum(packed .* weights)
end

@testset "array-valued plate consumers under ordinary native Reverse" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    for T in (Float32, Float64), count in (1, 3, 9, 33)
        X = reshape(T.(cos.(1:4count)), 4, count)
        weights = T.(sin.(1:4count))
        saved_X, saved_weights = copy(X), copy(weights)
        q = T[0.7]
        unbound = prepare(weighted_columns)
        bound = prepare(weighted_columns; bound=(; X, weights))
        unbound_ad = prepare_ad(unbound, backend, q, X, weights; active=:q)
        bound_ad = prepare_ad(bound, backend, q; active=:q)
        all_ad = prepare_ad(unbound, backend, q, X, weights;
                            active=(:q, :X, :weights))
        for scale in T.((0.7, -0.2))
            q = [scale]
            expected_gradient = [sum(vec(X) .* weights)]
            expected_value = scale * only(expected_gradient)
            @test weighted_loop(q, X, weights) ≈ expected_value
            @test first(Enzyme.gradient(Enzyme.Reverse, weighted_loop, q,
                                      Enzyme.Const(X), Enzyme.Const(weights))) ≈ expected_gradient
            @test unbound(q, X, weights) ≈ expected_value
            @test bound(q) ≈ expected_value
            @test ad_gradient(unbound_ad, q, X, weights) ≈ expected_gradient
            @test ad_gradient(bound_ad, q) ≈ expected_gradient
            value, gradients = ad_value_and_gradient(all_ad, q, X, weights)
            @test value ≈ expected_value
            @test gradients[1] ≈ expected_gradient
            @test gradients[2] ≈ scale .* reshape(weights, size(X))
            @test gradients[3] ≈ scale .* vec(X)
            replacement_X = X .+ T(0.25)
            replacement_weights = weights .* T(-0.5)
            @test ad_gradient(unbound_ad, q, replacement_X, replacement_weights) ≈
                [sum(vec(replacement_X) .* replacement_weights)]
            @test replacement_X == X .+ T(0.25)
            @test replacement_weights == weights .* T(-0.5)
            @test q == [scale]
            @test X == saved_X
            @test weights == saved_weights
        end
    end
    for T in (Float32, Float64)
        q, x, weights = T[0.7], T[], T[]
        k = prepare(weighted_broadcast; bound=(; x, weights))
        ad = prepare_ad(k, backend, q; active=:q)
        @test k(q) == zero(T)
        @test ad_gradient(ad, q) == T[0]
        @test q == T[0.7]
        @test isempty(x) && isempty(weights)
    end
end

# Base's scalar-array `*` and unary `-` broadcast into a fresh array of their
# operand's length. Over an empty gather, after a branch on that length (here
# the dimension check of the dotted result's materialization), native Enzyme
# on Julia 1.10 splits that allocation into a never-written empty array and
# raises EnzymeRuntimeActivityError
# (benchmark/repro_enzyme_length_branch_allocation_split.jl, Enzyme only).
# A capability gap of the backend: the pins fail on an Enzyme that legalizes
# the empty allocation, and then go.
@kernel gathered_location(base, slope, rate, times, idx) = begin
    level = exp.(-rate .* times)
    scaled = slope * (level[idx] ./ 2.0)
    negated = -(level[idx] ./ 4.0)
    chained = 0.5 * slope * level[idx]
    location = base .+ scaled .+ negated .+ chained
    return location
end
@kernel stepped_location(base, slope, rate, times, idx) = begin
    level = exp.(-rate .* times)
    halved = level[idx] ./ 2.0
    scaled = slope * halved
    location = base .+ scaled
    return location
end
for (name, location) in ((:gathered_groups, :gathered_location),
                         (:stepped_groups, :stepped_location))
    @eval @kernel $name(q::Vector{Float64}, groups::Int, times, idxs, ys) = begin
        cells = plate(1:groups, Ref(q), Ref(times), Ref(idxs), Ref(ys)) do g, q, times, idxs, ys
            location = $location(q[g], q[groups + g], exp(q[2groups + g]),
                                 times[g], idxs[g])
            sum(-0.5 .* (ys[g] .- location) .^ 2)
        end
        total::Float64 = sum(cells)
        return total
    end
end

function located_groups_loop(location, q, times, idxs, ys)
    groups, total = length(idxs), 0.0
    for g in 1:groups
        base, slope, rate = q[g], q[groups + g], exp(q[2groups + g])
        for (k, i) in enumerate(idxs[g])
            level = exp(-rate * times[g][i])
            total += -0.5 * (ys[g][k] - location(base, slope, level))^2
        end
    end
    total
end
gathered_loop(base, slope, level) =
    base + slope * (level / 2.0) - level / 4.0 + 0.5 * slope * level
stepped_loop(base, slope, level) = base + slope * (level / 2.0)

@testset "Base scalar-array operations over empty gathers under native Reverse" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    for (spec, location) in ((gathered_groups, gathered_loop), (stepped_groups, stepped_loop)),
        idxs in ([[1, 2], [3], [2, 4]], [[1, 2], Int[], Int[]],
                 [Int[], [1, 3], Int[]], [Int[], Int[]])
        groups = length(idxs)
        times = [collect(0.5:0.5:2.0) for _ in 1:groups]
        ys = [0.1 .* idx .+ 1.0 for idx in idxs]
        saved = deepcopy((times, idxs, ys))
        q = collect(range(0.1, 0.9; length=3groups))
        k = prepare(spec; bound=(; groups, times, idxs, ys))
        @test k(q) ≈ located_groups_loop(location, q, times, idxs, ys)
        ad = prepare_ad(k, backend, q; active=:q)
        if VERSION < v"1.11-" && any(isempty, idxs)
            @test_broken (ad_value_and_gradient(ad, q); true)
        else
            value, gradient = ad_value_and_gradient(ad, q)
            @test value ≈ located_groups_loop(location, q, times, idxs, ys)
            step = 1e-6
            central = map(eachindex(q)) do j
                e = zeros(length(q)); e[j] = step
                (located_groups_loop(location, q .+ e, times, idxs, ys) -
                 located_groups_loop(location, q .- e, times, idxs, ys)) / 2step
            end
            @test gradient ≈ central atol=1e-7
        end
        @test q == collect(range(0.1, 0.9; length=3groups))
        @test (times, idxs, ys) == saved
    end
end
end
