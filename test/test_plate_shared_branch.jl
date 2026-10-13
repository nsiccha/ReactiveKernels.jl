module PlateSharedBranchTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# A plate cell whose lazy branch reads only shared values takes the same arm
# at every cell. The native lowering decides it once per call and runs the
# taken arm's cells alone, so the arm's own shared work is a cell invariant
# (snag `rkppl-normal-lik-31ec97aa`). The counter only shows where a value is
# computed; nothing else in these cells has an effect.
const _ARM_CALLS = Ref(0)
_counted_log(s) = (_ARM_CALLS[] += 1; log(s))

# The support guard of a Normal observation with a shared scale.
@kernel _guarded_normal(y::Vector{Float64}, mu::Vector{Float64}, sigma::Float64) = begin
    pointwise = plate(y, mu, sigma) do y, mu, sigma
        if sigma > 0
            (normal(mu, sigma)).logpdf(y)
        else
            -Inf
        end
    end
    total::Float64 = sum(pointwise)
    return total
end
@kernel _guarded_pointwise(y::Vector{Float64}, mu::Vector{Float64}, sigma::Float64) = begin
    pointwise = plate(y, mu, sigma) do y, mu, sigma
        sigma > 0 ? (normal(mu, sigma)).logpdf(y) : -Inf
    end
    return pointwise
end
# A captured scale is shared; the arm's inline call is an invariant of that arm.
@kernel _guarded_captured(x, s) = begin
    cells = plate(x) do xi
        s > 0 ? xi * _counted_log(s) : -Inf
    end
    total = sum(cells)
    return total
end
@kernel _guarded_elseif(x, s) = begin
    cells = plate(x) do xi
        if s > 1
            xi * _counted_log(s)
        elseif s > 0
            xi + _counted_log(s)
        else
            -Inf
        end
    end
    total = sum(cells)
    return total
end
# A condition that throws on an invalid shared value is not evaluated over an
# empty domain, exactly as the per-cell branch.
@kernel _guarded_throwing(x, s) = begin
    cells = plate(x) do xi
        sqrt(s) > 1 ? xi * _counted_log(s) : 0.0
    end
    total = sum(cells)
    return total
end
# A condition that reads the cell keeps its branch in every cell; the arm's
# shared work runs at the first cell that takes the arm.
@kernel _lane_guard(x, s) = begin
    cells = plate(x) do xi
        xi > 0 ? xi * _counted_log(s) : -xi
    end
    return cells
end
@kernel _lane_guard_total(x, s) = begin
    cells = plate(x) do xi
        xi > 0 ? xi * _counted_log(s) : -xi
    end
    total = sum(cells)
    return total
end
# The throwing shared value runs only once a cell takes its arm.
@kernel _lane_throwing(x, s) = begin
    cells = plate(x) do xi
        xi > 0 ? xi * sqrt(s) : -xi
    end
    total = sum(cells)
    return total
end

# The branch belongs to a constructed endpoint rather than the plate cell's
# own source. Its positive arm contains a shared subexpression inside a mixed
# lane/shared recipe; the first positive lane computes it for all later lanes.
@kernel _endpoint_law(scale::Float64) = begin
    logpdf(x::Float64)::Float64 =
        x > 0 ? (twice = 2x; 0.5twice * _counted_log(scale)) : -x
end
@kernel _endpoint_law_plate(x::Vector{Float64}, scale::Float64) = begin
    cells = plate(x) do xi
        _endpoint_law(scale).logpdf(xi)
    end
    total::Float64 = sum(cells)
end
@kernel _endpoint_law_inferred(x::Vector{Float64}, offset::Float64) = begin
    scale = 1 - offset
    cells = plate(x, scale) do xi, si
        _endpoint_law(si).logpdf(xi)
    end
    total::Float64 = sum(cells)
end
@kernel _endpoint_law_math(scale::Float64) = begin
    logpdf(x::Float64)::Float64 = x > 0 ? x * log(scale) : -x
end
@kernel _endpoint_law_math_plate(x::Vector{Float64}, scale::Float64) = begin
    cells = plate(x) do xi
        _endpoint_law_math(scale).logpdf(xi)
    end
    total::Float64 = sum(cells)
end

_reference_normal(y, mu, sigma) =
    sigma > 0 ? sum(@. -0.5 * log(2π) - 0.5 * ((y - mu) / sigma)^2 - log(sigma)) : -Inf

function _program(k)
    names = Dict{String,String}()
    replace(string(ReactiveKernels._readable_expr(code_expr(k), k)),
            r"var\"[^\"]*\"" => m -> get!(names, m, "v$(length(names) + 1)"))
end

@testset "a shared plate branch is decided once per call" begin
    y = [0.5, -1.0, 2.0, 0.25, 1.5]
    mu = [0.1, 0.2, -0.3, 0.4, 0.0]
    k = prepare(_guarded_normal)
    for sigma in (1.3, 0.7, 0.0, -1.0, NaN, Inf, -Inf)
        @test k(y, mu, sigma) ≈ _reference_normal(y, mu, sigma) nans = true
    end
    @test k(y, mu, 1.3) isa Float64
    @test k(Float64[], Float64[], -1.0) == 0.0
    @test k(Float64[], Float64[], 1.0) == 0.0
    pointwise = prepare(_guarded_pointwise)
    @test pointwise(y, mu, 1.3) ≈ @. -0.5 * log(2π) - 0.5 * ((y - mu) / 1.3)^2 - log(1.3)
    @test pointwise(y, mu, -1.0) == fill(-Inf, length(y))
    @test pointwise(Float64[], Float64[], -1.0) == Float64[]
    @test pointwise(y, mu, 1.3) isa Vector{Float64}

    # The taken arm's scale logarithm runs once, above the cell loop.
    program = _program(k)
    @test occursin("log(sigma)", program)
    @test findfirst("log(sigma)", program).start < findfirst("for ", program).start
    @test count("condition", program) == 1
    # The graph keeps the authored lazy branch; only the native loop changes.
    @test [(e.kind, e.depth) for e in recipe_inventory(k)
           if e.kind !== :ordinary] == [(:plate, 0)]

    x = [1.0, 2.0, 3.0, 4.0]
    for (s, calls, value) in ((2.0, 1, sum(x .* log(2.0))), (-1.0, 0, -Inf),
                              (0.0, 0, -Inf))
        _ARM_CALLS[] = 0
        @test prepare(_guarded_captured)(x, s) ≈ value
        @test _ARM_CALLS[] == calls
    end
    _ARM_CALLS[] = 0
    @test prepare(_guarded_captured)(Float64[], 2.0) == 0.0
    @test _ARM_CALLS[] == 0

    elseif_kernel = prepare(_guarded_elseif)
    for (s, value) in ((3.0, sum(x) * log(3.0)), (0.5, sum(x .+ log(0.5))),
                       (-2.0, -Inf))
        _ARM_CALLS[] = 0
        @test elseif_kernel(x, s) ≈ value
        @test _ARM_CALLS[] == (s > 0 ? 1 : 0)
    end

    throwing = prepare(_guarded_throwing)
    @test throwing(Float64[], -1.0) == 0.0
    @test_throws DomainError throwing(x, -1.0)
    _ARM_CALLS[] = 0
    @test throwing(x, 4.0) ≈ sum(x) * log(4.0)
    @test _ARM_CALLS[] == 1

    lane = prepare(_lane_guard)
    _ARM_CALLS[] = 0
    @test lane([1.0, -2.0, 3.0], 0.5) ≈ [log(0.5), 2.0, 3.0 * log(0.5)]
    @test _ARM_CALLS[] == 1
    _ARM_CALLS[] = 0
    @test lane([-1.0, -2.0], 0.5) == [1.0, 2.0]
    @test _ARM_CALLS[] == 0
    lane_throwing = prepare(_lane_throwing)
    @test lane_throwing([-1.0, -2.0], -1.0) == 3.0
    @test_throws DomainError lane_throwing([-1.0, 2.0], -1.0)
    @test lane_throwing([-1.0, 4.0, 9.0], 4.0) == 1.0 + 26.0
end

@testset "a lane branch runs its arm's shared work at the first taking cell" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    x = [1.0, -2.0, 3.0, 0.5]
    k = prepare(_lane_guard_total)
    for s in (0.5, 2.0)
        prepared = prepare_ad(k, backend, x, s; active = :s)
        value, gradient = ad_value_and_gradient(prepared, x, s)
        positive = filter(>(0), x)
        @test value ≈ sum(positive) * log(s) + 2.0
        @test gradient ≈ sum(positive) / s
        prepared = prepare_ad(k, backend, x, s; active = :x)
        value, gradient = ad_value_and_gradient(prepared, x, s)
        @test gradient ≈ [xi > 0 ? log(s) : -1.0 for xi in x]
    end
end

@testset "constructed endpoint arm shares work across taking lanes" begin
    x = [-2.0, 1.0, -3.0, 0.5, 4.0]
    k = prepare(_endpoint_law_plate)
    _ARM_CALLS[] = 0
    @test k(x, 2.0) ≈ 5.0 + 5.5log(2.0)
    @test _ARM_CALLS[] == 1
    _ARM_CALLS[] = 0
    @test k([-2.0, -3.0], -1.0) == 5.0
    @test _ARM_CALLS[] == 0
    _ARM_CALLS[] = 0
    @test k(Float64[], -1.0) == 0.0
    @test _ARM_CALLS[] == 0
    @test_throws DomainError k(x, -1.0)

    # The generated scalar assignment has no declared type, as in RKPPL's
    # synthesized distribution argument. Its inferred Number type still makes
    # the endpoint's scale shared.
    inferred = prepare(_endpoint_law_inferred)
    _ARM_CALLS[] = 0
    @test inferred(x, -1.0) ≈ 5.0 + 5.5log(2.0)
    @test _ARM_CALLS[] == 1

    math = prepare(_endpoint_law_math_plate)
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    pb = prepare_ad(math, backend, x, 2.0; active = :scale)
    value, gradient = ad_value_and_gradient(pb, x, 2.0)
    @test value ≈ 5.0 + 5.5log(2.0)
    @test gradient ≈ 5.5 / 2.0
end

@testset "bound data and native Reverse through a shared plate branch" begin
    y = [0.5, -1.0, 2.0, 0.25, 1.5]
    mu = [0.1, 0.2, -0.3, 0.4, 0.0]
    bound = prepare(_guarded_normal; have = (:y, :mu, :sigma), want = :total,
                    bound = (; y))
    @test bound(mu, 1.3) ≈ _reference_normal(y, mu, 1.3)
    @test bound(mu, -1.0) == -Inf

    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    k = prepare(_guarded_normal)
    for sigma in (1.3, 0.4)
        z = (y .- mu) ./ sigma
        prepared = prepare_ad(k, backend, y, mu, sigma; active = :sigma)
        value, gradient = ad_value_and_gradient(prepared, y, mu, sigma)
        @test value ≈ _reference_normal(y, mu, sigma)
        @test gradient ≈ sum(z .^ 2 ./ sigma .- 1 / sigma)
        prepared = prepare_ad(k, backend, y, mu, sigma; active = :mu)
        value, gradient = ad_value_and_gradient(prepared, y, mu, sigma)
        @test gradient ≈ z ./ sigma
    end
    prepared = prepare_ad(k, backend, y, mu, -1.0; active = :sigma)
    value, gradient = ad_value_and_gradient(prepared, y, mu, -1.0)
    @test value == -Inf
    @test gradient == 0.0
end

end
