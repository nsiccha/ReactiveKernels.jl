module LazyArmChildADTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# A function-shaped child called where the splicer cannot reach it: a lazy
# branch arm (top-level recipe, plate cell, scan step) or a deferred closure.
# Its planned recipes are inlined at the call, so no runtime `KernelSpec`
# application plans the child inside the differentiated kernel.
@kernel stable_softplus(z) = begin
    value = if z > 0.0
        z + log1p(exp(-z))
    else
        log1p(exp(z))
    end
    return value
end

@kernel first_plus(x, values) = begin
    result = x + values[1]
    return result
end

@kernel cumulative(xs, gain) = begin
    updates = scan(xs, Ref(gain); init = 0.0) do carry, x, g
        next = carry + x * g
        (next, next)
    end
    total = sum(updates)
    return total
end

@kernel top_arm(s) = begin
    v = if s > 0.0
        stable_softplus(s)
    else
        0.0
    end
    return v
end

@kernel plate_arms(xs, s) = begin
    values = plate(xs, Ref(s)) do x, ss
        v = if x > 0.0
            stable_softplus(x * ss)
        elseif x < 0.0
            -LazyArmChildADTests.stable_softplus(-x * ss)
        else
            0.0
        end
        v
    end
    total = sum(values)
    return total
end

@kernel scan_arm(xs, s) = begin
    trajectory = scan(xs, Ref(s); init = 0.0) do c, x, ss
        n = x > 0.0 ? c + stable_softplus(x * ss) : c
        (n, n)
    end
    total = sum(trajectory)
    return total
end

@kernel deferred_child(xs, s) = begin
    total = sum(x -> stable_softplus(x * s), xs)
    return total
end

@kernel guarded_cells(xs, values) = begin
    cells = plate(xs, Ref(values)) do x, v
        x > 0.0 ? first_plus(x, v) : 0.0
    end
    return cells
end

@kernel scaled_sum(xs, gain) = begin
    cells = plate(xs, Ref(gain)) do x, g
        x * g + log1p(exp(x * g))
    end
    total = sum(cells)
    return total
end

@kernel first_partial(xs) = begin
    partial = scan(xs; init = 0.0) do carry, x
        next = carry + x
        (next, next)
    end
    head = partial[1]
    return head
end

# Children holding a plate or scan are prepared once when the caller is
# defined and applied in the arm.
@kernel scan_child_arm(xs, gain) = begin
    v = gain > 0.0 ? cumulative(xs, gain) : 0.0
    return v
end

@kernel plate_child_arm(xs, gain) = begin
    v = gain > 0.0 ? scaled_sum(xs, gain) : 0.0
    return v
end

@kernel cell_scan_child(groups, gain) = begin
    cells = plate(groups, Ref(gain)) do xs, g
        cell::Float64 = g > 0.0 ? cumulative(xs, g) : 0.0
        cell
    end
    total = sum(cells)
    return total
end

@kernel guarded_region_child(xs, flag) = begin
    v = flag > 0 ? first_partial(xs) : 0.0
    return v
end

softplus(z) = z > 0 ? z + log1p(exp(-z)) : log1p(exp(z))
logistic(z) = inv(1 + exp(-z))

function plate_reference(xs, s)
    total, gradient = 0.0, 0.0
    for x in xs
        if x > 0
            total += softplus(x * s); gradient += x * logistic(x * s)
        elseif x < 0
            total -= softplus(-x * s); gradient += x * logistic(-x * s)
        end
    end
    total, gradient
end

function scan_reference(xs, s)
    carry, dcarry, total, gradient = 0.0, 0.0, 0.0, 0.0
    for x in xs
        if x > 0
            carry += softplus(x * s); dcarry += x * logistic(x * s)
        end
        total += carry; gradient += dcarry
    end
    total, gradient
end

inlined(kernel) = !occursin("stable_softplus(", sprint(show, readable_code(kernel)))

@testset "function-shaped children under lazy arms differentiate natively" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    xs = [-1.0, 0.0, 0.5, 2.0]
    saved = copy(xs)

    top = prepare(top_arm)
    @test inlined(top)
    for s in (0.7, -0.4)
        ad = prepare_ad(top, backend, s; active = :s)
        value, gradient = ad_value_and_gradient(ad, s)
        @test value ≈ (s > 0 ? softplus(s) : 0.0)
        @test gradient ≈ (s > 0 ? logistic(s) : 0.0)
    end

    for (spec, reference) in ((plate_arms, plate_reference),
                              (scan_arm, scan_reference),
                              (deferred_child, (xs, s) -> (
                                  sum(softplus.(xs .* s)),
                                  sum(xs .* logistic.(xs .* s)))))
        kernel = prepare(spec)
        @test inlined(kernel)
        ad = prepare_ad(kernel, backend, xs, 0.7; active = :s)
        for s in (0.7, 1.3, -0.2)
            expected, dexpected = reference(xs, s)
            @test kernel(xs, s) ≈ expected
            value, gradient = ad_value_and_gradient(ad, xs, s)
            @test value ≈ expected
            @test gradient ≈ dexpected
            @test xs == saved
        end
    end
end

@testset "inlined lazy-arm children keep their arm's laziness" begin
    guarded = prepare(guarded_cells)
    @test guarded([-1.0, 0.0], Float64[]) == [0.0, 0.0]
    @test guarded([-1.0, 2.0], [3.0]) == [0.0, 5.0]
end

cumulative_reference(xs, g) = (g * sum(cumsum(xs)), sum(cumsum(xs)))
scaled_reference(xs, g) = (sum(x * g + softplus(x * g) for x in xs),
                           sum(x * (1 + logistic(x * g)) for x in xs))

@testset "children holding a plate or scan in a lazy arm are prepared once" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    xs = [1.0, 2.0, 0.5]
    groups = [xs, [2.0], Float64[]]
    saved, saved_groups = copy(xs), deepcopy(groups)
    cases = ((scan_child_arm, xs, cumulative_reference),
             (plate_child_arm, xs, scaled_reference),
             (cell_scan_child, groups, (gs, g) -> (
                 sum(first(cumulative_reference(x, g)) for x in gs),
                 sum(last(cumulative_reference(x, g)) for x in gs))))
    for (spec, data, reference) in cases
        kernel = prepare(spec)
        # The arm names the authored child; it is applied, not re-prepared.
        @test occursin(r"cumulative\(|scaled_sum\(", sprint(show, readable_code(kernel)))
        kernel(data, 0.5)
        @test (@allocated kernel(data, 0.5)) < 16_384
        ad = prepare_ad(kernel, backend, data, 0.5; active = :gain)
        for g in (0.5, 1.3)
            expected, dexpected = reference(data, g)
            @test kernel(data, g) ≈ expected
            value, gradient = ad_value_and_gradient(ad, data, g)
            @test value ≈ expected
            @test gradient ≈ dexpected
        end
        value, gradient = ad_value_and_gradient(ad, data, -0.5)
        @test value == 0.0
        @test gradient == 0.0
    end
    @test xs == saved
    @test groups == saved_groups
    # Preparing the child at definition evaluates nothing: an untaken arm
    # never reads `partial[1]` of an empty scan.
    guarded = prepare(guarded_region_child)
    @test guarded(Float64[], 0) == 0.0
    @test guarded([2.0, 3.0], 1) == 2.0
    @test_throws BoundsError guarded(Float64[], 1)
end
end
