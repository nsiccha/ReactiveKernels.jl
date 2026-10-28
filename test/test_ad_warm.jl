module ADWarmTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# `prepare_ad(...; warm = true)` compiles the prepared gradient during
# preparation by evaluating it once at a copy of the exemplar (snag
# nested-enzyme-co-3b60d53e). Later evaluations return exactly what an
# unwarmed preparation returns, and the exemplar stays untouched.

const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
const EVALUATIONS = Ref(0)

counted_half_norm(q) = (EVALUATIONS[] += 1; sum(abs2, q) / 2)

@kernel warm_objective(q::Vector{Float64}, scale::Float64) = begin
    total = scale * counted_half_norm(q) + scale^2
end

function evaluations(f)
    before = EVALUATIONS[]
    result = f()
    result, EVALUATIONS[] - before
end

@testset "prepare_ad warm compiles during preparation" begin
    q, scale = [0.3, -1.2, 2.0], 1.7
    exemplar = copy(q)
    cold, cold_runs = evaluations(() ->
        prepare_ad(warm_objective, BACKEND, q, scale; want = :total, active = :q))
    warm, warm_runs = evaluations(() -> prepare_ad(warm_objective, BACKEND,
        q, scale; want = :total, active = :q, warm = true))
    @test warm_runs > cold_runs
    @test q == exemplar
    for point in (q, [1.0, 0.5, -0.25])
        expected = ad_value_and_gradient!(cold, zeros(3), point, scale)
        @test ad_value_and_gradient!(warm, zeros(3), point, scale) == expected
        @test expected[2] ≈ scale .* point
    end

    # A tuple selector warms the out-of-place call.
    selector = (:q, :scale)
    cold = prepare_ad(warm_objective, BACKEND, q, scale; want = :total,
        active = selector)
    warm, warm_runs = evaluations(() -> prepare_ad(warm_objective, BACKEND,
        q, scale; want = :total, active = selector, warm = true))
    @test warm_runs > 0
    @test q == exemplar
    @test ad_value_and_gradient(warm, q, scale) == ad_value_and_gradient(cold, q, scale)

    # The low-level positional boundary accepts the same option.
    kernel = prepare(warm_objective; want = :total)
    warm, warm_runs = evaluations(() ->
        prepare_ad(kernel, BACKEND, q, scale; active = :q, warm = true))
    @test warm_runs > 0
    @test ad_value_and_gradient!(warm, zeros(3), q, scale)[2] ≈ scale .* q
end

end
