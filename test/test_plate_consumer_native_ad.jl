module PlateConsumerNativeADTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test
using ReactiveKernelsDistributionKernels.DistributionKernelSources: gamma

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

@kernel guarded_gamma_plates(y1::Vector{Float64}, y2::Vector{Float64},
                             scale::Float64) = begin
    p1 = plate(y1, scale) do y, s
        density::Float64 = s > 0 ? gamma(2.0, 1.0 / s).logpdf(y) : -Inf
        density
    end
    p2 = plate(y2, scale) do y, s
        density::Float64 = s > 0 ? gamma(2.0, 1.0 / s).logpdf(y) : -Inf
        density
    end
    total::Float64 = sum(p1) + sum(p2)
    return total
end

@testset "native Reverse keeps nested endpoint guards lazy in scalar plates" begin
    y1 = [0.55, 0.6, 0.65, 0.7]
    y2 = [0.5, 0.75, 0.9]
    for bound in (false, true)
        kernel = bound ? prepare(guarded_gamma_plates; bound=(; y1, y2)) :
            prepare(guarded_gamma_plates)
        arguments(s) = bound ? (s,) : (y1, y2, s)
        for scale in (0.4, -0.5)
            args = arguments(scale)
            ad = prepare_ad(kernel, AutoEnzyme(; mode=Enzyme.Reverse), args...;
                            active=:scale)
            value, gradient = ad_value_and_gradient(ad, args...)
            if scale > 0
                expected = sum(2log(inv(scale)) + log(y) - y / scale
                               for y in (y1..., y2...))
                derivative = sum(-2 / scale + y / scale^2
                                 for y in (y1..., y2...))
                @test value ≈ expected
                @test gradient ≈ derivative
            else
                @test value == -Inf
                @test gradient == 0.0
            end
        end
        @test kernel(arguments(0.0)...) == -Inf
    end
    # Both response lengths may change without duplicating either loop body.
    small = string(readable_code(prepare(guarded_gamma_plates; bound=(; y1, y2))))
    large = string(readable_code(prepare(guarded_gamma_plates;
        bound=(; y1=repeat(y1, 7), y2=repeat(y2, 5)))))
    loops(code) = length(collect(eachmatch(r"(?m)^\s*for ", code)))
    @test loops(small) == loops(large) == 2
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

# Untyped plate arguments emit each cell recipe once, inside the loop (snag
# prepared-kernel-ae180d1a): a cell value cached across coordinates, and a
# scalar invariant computed at the first coordinate, under native Reverse.
@kernel untyped_domain_cells(xs, s) = begin
    pointwise = plate(xs, s) do x, si
        ls = log(si)
        shifted = x * si + ls
        shifted * shifted
    end
    total = sum(pointwise)
    return total
end
untyped_domain_loop(xs, s) = sum(x -> (x * s + log(s))^2, xs; init = 0.0)

@testset "untyped plate arguments under native Reverse" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    k = prepare(untyped_domain_cells)
    for xs in ([0.25, -1.0, 2.0], 1:4, (0.5, 1.5)), s in (0.7, 1.9)
        saved = copy(collect(xs))
        @test k(xs, s) ≈ untyped_domain_loop(xs, s)
        value, gradient = ad_value_and_gradient(
            prepare_ad(k, backend, xs, s; active=:s), xs, s)
        @test value ≈ untyped_domain_loop(xs, s)
        @test gradient ≈ sum(2 * (x * s + log(s)) * (x + inv(s)) for x in xs)
        @test collect(xs) == saved
    end
    # A matrix domain is not one-dimensional: its cells run through the
    # coordinate-change test instead.
    M = [0.25 -1.0; 2.0 0.5; 1.5 3.0]
    value, gradient = ad_value_and_gradient(prepare_ad(k, backend, M, 1.3; active=:s), M, 1.3)
    @test value ≈ untyped_domain_loop(vec(M), 1.3)
    @test gradient ≈ sum(2 * (x * 1.3 + log(1.3)) * (x + inv(1.3)) for x in M)

    # Bound arguments lower as their class declares (todo 0hc187j): a bound
    # domain is the static axis, with no runtime axis test of its own (the
    # live untyped `s` keeps its guard), and a bound scalar operand is
    # computed above the loop.
    for xs in ([0.25, -1.0, 2.0], 1:4, (0.5, 1.5), M)
        bound = prepare(untyped_domain_cells; bound = (; xs))
        code = string(readable_code(bound))
        @test !occursin("_authored_plate_is_axis(xs)", code)
        @test !occursin("_authored_plate_marker", code)
        value, gradient = ad_value_and_gradient(
            prepare_ad(bound, backend, 0.7; active=:s), 0.7)
        @test value ≈ untyped_domain_loop(vec(collect(xs)), 0.7)
        @test gradient ≈ sum(2 * (x * 0.7 + log(0.7)) * (x + inv(0.7)) for x in xs)
    end
    xs = [0.25, -1.0, 2.0]
    bound = prepare(untyped_domain_cells; bound = (; s = 0.7))
    value, gradient = ad_value_and_gradient(prepare_ad(bound, backend, xs; active=:xs), xs)
    @test value ≈ untyped_domain_loop(xs, 0.7)
    @test gradient ≈ [2 * (x * 0.7 + log(0.7)) * 0.7 for x in xs]
    @test xs == [0.25, -1.0, 2.0]
end

# A cell whose result is an inactive argument's element, or a field of one,
# stores pointers into constant memory in the fresh pointwise buffer, which
# then feeds an active result. Enzyme's static activity analysis rejects that
# buffer exactly as it rejects plain `map(identity, groups)`
# (`benchmark/repro_enzyme_constant_element_container.jl`, docs/src/constraints.md).
struct AliasedSubject
    xs::Vector{Float64}
end
@kernel identity_cells(groups, rates) = begin
    per = plate(groups, Ref(rates)) do g, rates
        identity(g)
    end
    flat = convert(Vector{Float64}, reduce(vcat, per; init = Float64[]))
    total = sum(flat .* rates)
end
@kernel bare_cells(groups, rates) = begin
    per = plate(groups) do g
        g
    end
    total = sum(reduce(vcat, per) .* rates)
end
@kernel field_cells(subjects, rates) = begin
    per = plate(subjects) do s
        s.xs
    end
    total = sum(reduce(vcat, per) .* rates)
end
@kernel copied_cells(groups, rates) = begin
    per = plate(groups) do g
        copy(g)
    end
    total = sum(reduce(vcat, per) .* rates)
end

@testset "plate cells returning inactive argument storage under native Reverse" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    groups = [[0.5, 1.0, 1.0], [0.5], [1.0, 1.0]]
    subjects = AliasedSubject.(groups)
    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    flat = reduce(vcat, groups)
    expected = sum(flat .* rates)
    saved = deepcopy(groups)
    gradient_of(spec, data; bound = (;)) = begin
        k = prepare(spec; bound)
        args = isempty(bound) ? (data, rates) : (rates,)
        @test k(args...) ≈ expected
        ad = prepare_ad(k, backend, args...; active=:rates)
        ad_value_and_gradient(ad, args...)
    end
    # Fresh cell results differentiate; so does a data-only plate under
    # `bound=`, which preparation hoists out of the gradient.
    for (spec, data, bound) in ((copied_cells, groups, (;)),
                                (identity_cells, groups, (; groups)),
                                (bare_cells, groups, (; groups)),
                                (field_cells, subjects, (; subjects)))
        value, gradient = gradient_of(spec, data; bound)
        @test value ≈ expected
        @test gradient ≈ flat
    end
    # The same cells over an unbound argument: the backend boundary.
    for (spec, data) in ((identity_cells, groups), (bare_cells, groups),
                         (field_cells, subjects))
        @test_broken (gradient_of(spec, data)[2] ≈ flat)
    end
    @test groups == saved
    @test [s.xs for s in subjects] == saved
end

# A plate with more batched operands than Julia's 32-element splat limit. Base's
# `combine_axes(A, B...)` then recurses through `Core._apply_iterate` over one
# tuple of the remaining operands, and native Reverse rejected the constant
# ragged arrays stored there beside the live vector. Under `bound=` each
# per-cell array the cell reads from bound data becomes a cached plate operand
# (the declared live rank makes the cell eligible), which is how wide consumer
# reader plates crossed the limit (snag `rk-cached-bound-aa544610`).
const WIDE_PORTS = 33
let ports = [Symbol(:xs, k) for k in 1:WIDE_PORTS],
    cells = [Symbol(:c, k) for k in 1:WIDE_PORTS],
    shifted = [Symbol(:w, k) for k in 1:WIDE_PORTS]
    reads = (:($w = $c .+ 1.0) for (w, c) in zip(shifted, cells))
    terms = (:(sum(l .* $w; init = 0.0)) for w in shifted)
    @eval @kernel wide_plate(live::Vector{Float64}, $(ports...)) = begin
        out = plate(live, $(ports...)) do l, $(cells...)
            $(reads...)
            $(Expr(:call, :+, :(l * l), terms...))
        end
        total = sum(out)
    end
end

@testset "a plate wider than Julia's splat limit under native Reverse" begin
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    live = [0.1, 0.2, 0.3]
    data = Tuple([[Float64(k + i + j) for j in 1:mod(i, 3)] for i in 1:3]
                 for k in 1:WIDE_PORTS)
    names = Tuple(Symbol(:xs, k) for k in 1:WIDE_PORTS)
    saved = deepcopy(data)
    shifted(i) = sum(sum(d[i] .+ 1.0; init = 0.0) for d in data)
    expected = sum(l^2 + l * shifted(i) for (i, l) in enumerate(live))
    gradient = [2l + shifted(i) for (i, l) in enumerate(live)]
    @test prepare(wide_plate)(live, data...) ≈ expected
    plain = prepare_ad(wide_plate, backend, live, data...;
                       active=:live, want=:total)
    value, g = ad_value_and_gradient(plain, live, data...)
    @test value ≈ expected
    @test g ≈ gradient
    bound = prepare_ad(wide_plate, backend, live; active=:live, want=:total,
                       bound=NamedTuple{names}(data))
    cached = filter(r -> r.op isa ReactiveKernels._BoundConstant &&
        startswith(String(only(r.outputs).name), "bound_plate_"),
        bound.kernel.plan.recipes)
    @test length(cached) == WIDE_PORTS
    @test all(r -> r.op.value isa Vector{Vector{Float64}}, cached)
    value, g = ad_value_and_gradient(bound, live)
    @test value ≈ expected
    @test g ≈ gradient
    @test_throws DimensionMismatch prepare(wide_plate)(
        live, data[1:end-1]..., [[1.0], [2.0]])
    @test data == saved
    @test live == [0.1, 0.2, 0.3]
end
end
