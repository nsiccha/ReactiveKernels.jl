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

@kernel scan_child_arm(xs, gain) = begin
    v = gain > 0.0 ? cumulative(xs, gain) : 0.0
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

@testset "children with a scan in a lazy arm keep their runtime application" begin
    kernel = prepare(scan_child_arm)
    @test occursin("cumulative(", sprint(show, readable_code(kernel)))
    @test kernel([1.0, 2.0], 0.5) ≈ 0.5 + 1.5
    @test kernel([1.0, 2.0], -0.5) == 0.0
end
end
