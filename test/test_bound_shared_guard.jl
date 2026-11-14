module BoundSharedGuardTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test
using Distributions: Normal, cdf

const calls = Ref(0)
counted_exp(x) = (calls[] += 1; exp(x))

@kernel guarded_score(y::Float64, a::Float64, upper::Float64) = begin
    z = isinf(upper) ? 1.0 : exp(a + upper)
    result = y * z
    return result
end
@kernel direct_score(y::Float64, a::Float64, upper::Float64) = begin
    z = exp(a + upper)
    result = y * z
    return result
end
@kernel guarded_total(y, upper, a::Float64) = begin
    scores = plate(y, upper) do yi, ui
        guarded_score(yi, a, ui)
    end
    total = sum(scores)
end
@kernel direct_total(y, upper, a::Float64) = begin
    scores = plate(y, upper) do yi, ui
        direct_score(yi, a, ui)
    end
    total = sum(scores)
end
@kernel whole_bound_arm(y, upper, a::Float64) = begin
    scores = plate(y, upper) do yi, ui
        isinf(ui) ? yi : yi * exp(a + ui)
    end
    total = sum(scores)
end
struct ReadCountVector{T} <: AbstractVector{T}
    values::Vector{T}
    reads::Base.RefValue{Int}
end
Base.size(x::ReadCountVector) = size(x.values)
Base.IndexStyle(::Type{<:ReadCountVector}) = IndexLinear()
Base.getindex(x::ReadCountVector, i::Int) = (x.reads[] += 1; x.values[i])
@kernel counted(y, upper, a::Float64) = begin
    scores = plate(y, upper) do yi, ui
        z = isinf(ui) ? 1.0 : counted_exp(a + ui)
        yi * z
    end
    total = sum(scores)
end
@kernel shared_result(y, upper, a::Float64) = begin
    scores = plate(y, upper) do yi, ui
        isinf(ui) ? 1.0 : counted_exp(a + ui)
    end
    total = sum(scores)
end
@kernel nested(y, upper, a::Float64) = begin
    scores = plate(y, upper) do yi, ui
        z = isinf(ui) ? 1.0 : (ui > 0 ? counted_exp(a + ui) : 0.0)
        yi * z
    end
    total = sum(scores)
end
@kernel safe_log(y, upper, a::Float64) = begin
    scores = plate(y, upper) do yi, ui
        z = isinf(ui) ? 1.0 : log(a + ui)
        yi * z
    end
    total = sum(scores)
end

# A lognormal delay convolved with a uniform window: the finite upper-bound
# normalizer contains nested data guards and composed kernel calls, while its
# moment is a separate shared recipe. This is the reported normalizer shape.
@kernel integrated_cdf(q::Float64, location::Float64, scale::Float64,
                       moment::Float64) = begin
    result = if q <= 0
        zero(q * location * scale)
    else
        z = (log(q) - location) / scale
        q * cdf(Normal(), z) - moment * cdf(Normal(), z - scale)
    end
    return result
end
@kernel window_cdf(q::Float64, width::Float64, location::Float64,
                  scale::Float64, moment::Float64) = begin
    result = if q <= 0
        zero(q * width * location * scale)
    elseif isinf(q)
        one(q * width * location * scale)
    else
        (integrated_cdf(q, location, scale, moment) -
         integrated_cdf(q - width, location, scale, moment)) / width
    end
    return result
end
@kernel normalized_score(y::Float64, width::Float64, upper::Float64,
                         location::Float64, scale::Float64) = begin
    moment = exp(location + scale^2 / 2)
    normalizer = isinf(upper) ? 1.0 :
        window_cdf(upper, width, location, scale, moment)
    return y / normalizer
end
@kernel normalized_total(y, width, upper, location::Float64, scale::Float64) = begin
    scores = plate(y, width, upper) do yi, wi, ui
        normalized_score(yi, wi, ui, location, scale)
    end
    total = sum(scores)
end

const backend = AutoEnzyme(; mode=Enzyme.Reverse)
nodes(x) = x isa Expr ? 1 + sum(nodes, x.args; init=0) : 1

@testset "a wholly bound arm retains its prepared reduction" begin
    for n in (0, 1, 7, 257), branch_arms in (:taken, :all)
        values = collect(range(0.25; step=0.125, length=n))
        reads = Ref(0)
        y = ReadCountVector(values, reads)
        k = prepare(whole_bound_arm; want=:total, bound=(; y, upper=Inf), branch_arms)
        reads[] = 0
        @test k(0.2) ≈ sum(values)
        @test k(-0.7) ≈ sum(values)
        @test reads[] == 0
        ad = prepare_ad(k, backend, 0.2; active=:a)
        @test ad_value_and_gradient(ad, 0.2) == (sum(values), 0.0)
        @test reads[] == 0
    end
end

@testset "a shared numerical guard stays runtime across bindings" begin
    for branch_arms in (:taken, :all)
        for n in (0, 1, 7, 257), repeated in (false, true)
            y = repeated ? fill(0.5, n) : collect(range(0.25; step=0.125, length=n))
            kernels = [prepare(guarded_total; want=:total,
                bound=(; y, upper), branch_arms) for upper in (2.0, Inf)]
            # Changing this shared guard must not change the program. Row
            # grouping is independent: repeated and distinct data can have
            # different programs, both with bounded loop-body growth.
            same_type = typeof(kernels[1]) == typeof(kernels[2])
            same_code = code_expr(kernels[1]) == code_expr(kernels[2])
            @test same_type
            @test same_code
            for (k, upper) in zip(kernels, (2.0, Inf)), a in (-0.25, 0.5)
                expected = sum(y) * (isinf(upper) ? 1.0 : exp(a + upper))
                @test k(a) ≈ expected
            end
        end
        for repeated in (false, true)
            sizes = map((7, 257)) do n
                y = repeated ? fill(0.5, n) : collect(range(0.25; step=0.125, length=n))
                k = prepare(guarded_total; want=:total,
                    bound=(; y, upper=2.0), branch_arms)
                nodes(code_expr(k))
            end
            @test sizes[1] == sizes[2]
        end
    end
end

@testset "a bound shared guard keeps its arm shared" begin
    # User principle 1om0yie: plate-invariant work is hoisted automatically.
    # Count executions as well as checking values: LLVM may erase an unused
    # argument, but that does not repair the dependencies in the prepared graph.
    for branch_arms in (:taken, :all), n in (0, 1, 7, 257), upper in (2.0, Inf)
        y = collect(range(0.25; step=0.125, length=n))
        saved = copy(y)
        bound = (; y, upper)
        for spec in (counted, nested, shared_result)
            k = prepare(spec; have=(:y, :upper, :a), want=(:scores, :total),
                        bound, branch_arms)
            for a in (0.5, -0.25)
                calls[] = 0
                scores, total = k(a)
                z = isinf(upper) ? 1.0 : exp(a + upper)
                expected = spec === shared_result ? fill(z, n) : y .* z
                @test scores ≈ expected
                @test total ≈ sum(expected)
                @test calls[] == (n > 0 && isfinite(upper) ? 1 : 0)
                @test y == saved
            end
        end

        k = prepare(guarded_total; have=(:y, :upper, :a), want=:total,
                    bound, branch_arms)
        ad = prepare_ad(k, backend, 0.5; active=:a)
        value, gradient = ad_value_and_gradient(ad, 0.5)
        expected = sum(y) * (isinf(upper) ? 1.0 : exp(2.5))
        @test value ≈ expected
        @test gradient ≈ (isinf(upper) ? 0.0 : expected)
        if isfinite(upper)
            direct = prepare(direct_total; have=(:y, :upper, :a), want=:total,
                             bound, branch_arms)
            @test k(0.5) ≈ direct(0.5)
        end
    end
    # Emission keeps data-driven iteration; a larger binding does not duplicate
    # arm bodies (CONTRIBUTING.md, "Keep sizes and numeric values out
    # of generated programs").
    for spec in (guarded_total, normalized_total)
        sizes = map((7, 257)) do n
            bound = spec === guarded_total ? (y=ones(n), upper=2.0) :
                (y=ones(n), width=1.0, upper=2.0)
            k = prepare(spec; bound, want=:total)
            nodes(code_expr(k))
        end
        @test sizes[1] == sizes[2]
    end
end

@testset "shared arm laziness and lane-dependent controls" begin
    for branch_arms in (:taken, :all)
        for upper in (2.0, Inf)
            k = prepare(safe_log; have=(:y, :upper, :a), want=:total,
                        bound=(y=Float64[], upper), branch_arms)
            @test k(-3.0) == 0.0
            ad = prepare_ad(k, backend, -3.0; active=:a)
            @test ad_value_and_gradient(ad, -3.0) == (0.0, 0.0)
        end
        k = prepare(safe_log; have=(:y, :upper, :a), want=:total,
                    bound=(y=[1.0, 2.0], upper=Inf), branch_arms)
        @test k(-3.0) == 3.0
        bad = prepare(safe_log; have=(:y, :upper, :a), want=:total,
                      bound=(y=[1.0, 2.0], upper=2.0), branch_arms)
        @test_throws DomainError bad(-3.0)

        y = [1.0, 2.0, 3.0, 4.0]
        upper = [2.0, Inf, 3.0, Inf]
        k = prepare(counted; have=(:y, :upper, :a), want=:total,
                    bound=(; y, upper), branch_arms)
        calls[] = 0
        @test k(0.5) ≈ exp(2.5) + 2 + 3exp(3.5) + 4
        @test calls[] == 2 # distinct per-lane inputs remain per-lane work
    end
end

@testset "composed finite-window normalizer" begin
    integral(q, a, b) = q <= 0 ? 0.0 :
        q * cdf(Normal(), (log(q) - a) / b) -
        exp(a + b^2 / 2) * cdf(Normal(), (log(q) - a) / b - b)
    normalizer(u, w, a, b) = isinf(u) ? 1.0 :
        (integral(u, a, b) - integral(u - w, a, b)) / w
    for n in (0, 1, 7, 257), upper in (0.75, 2.0, Inf)
        y = collect(range(0.25; step=0.125, length=n))
        width, a, b = 1.0, 0.2, 0.8
        k = prepare(normalized_total; have=(:y, :width, :upper, :location, :scale),
                    want=:total, bound=(; y, width, upper))
        ref(a, b) = sum(y) / normalizer(upper, width, a, b)
        @test k(a, b) ≈ ref(a, b)
        ad = prepare_ad(k, backend, a, b; active=:location)
        value, gradient = ad_value_and_gradient(ad, a, b)
        h = 1.0e-5
        @test value ≈ ref(a, b)
        @test gradient ≈ (ref(a + h, b) - ref(a - h, b)) / (2h) atol=1.0e-7 rtol=1.0e-6
    end
end
end # module
