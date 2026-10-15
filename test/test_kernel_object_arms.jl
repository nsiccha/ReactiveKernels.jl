module KernelObjectArmTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test

# An endpoint call (a sibling, a bound child or an `inv`) in a lazy branch arm
# of a kernel-object method is evaluated only when the arm is taken
# (`docs/src/constraints.md`, "Preserve lazy branches"). It used to be hoisted
# into its own recipe above the branch, so `f(-1.0)` ran `log(-1.0)` and threw
# a `DomainError` (todo `ReactiveKernels:kernel-let/2026-10-13T10-32-37-142-0dsoym5`).
@kernel guarded(scale::Float64) = begin
    sib(x::Float64)::Float64 = log(x) * scale
    f(x::Float64)::Float64 = x > 0 ? sib(x) : 0.0
    g(x::Float64)::Float64 = x > 0 && sib(x) > 0 ? 1.0 : 0.0
    # A straight call still splices as graph recipes.
    straight(x::Float64)::Float64 = sib(x) + 1.0
end

@kernel half_log() = begin
    logpdf(z::Float64)::Float64 = log(z)
end

@kernel guarded_scale(standard, location::Float64, scale::Float64) = begin
    standardized(x::Float64)::Float64 = (x - location) / scale
    inv(standardized, z::Float64)::Float64 = location + scale * sqrt(z)
    logpdf(x::Float64)::Float64 =
        x > location ? standard.logpdf(standardized(x)) - log(scale) : -Inf
    back(z::Float64)::Float64 = z > 0 ? inv(standardized, z) : location
end

@kernel guarded_half = guarded_scale(half_log)

@kernel arm_caller(scale::Float64, x::Float64) = begin
    y::Float64 = guarded(; scale = scale).f(x)
end

# A caller arm renders `f`'s recipes in place, carrying its prepared child.
@kernel nested_arm_caller(scale::Float64, x::Float64, c::Bool) = begin
    y::Float64 = c ? guarded(; scale = scale).f(x) : 1.0
end

@testset "endpoint calls in lazy arms of kernel-object methods" begin
    f = prepare(guarded.f)
    @test f(1.5, 2.0) == log(2.0) * 1.5
    @test f(1.5, -1.0) == 0.0
    g = prepare(guarded.g)
    @test (g(1.5, 2.0), g(1.5, 0.5), g(1.5, -1.0)) == (1.0, 0.0, 0.0)
    @test prepare(guarded.straight)(1.5, 2.0) == log(2.0) * 1.5 + 1.0
    @test prepare(arm_caller)(1.5, -1.0) == 0.0
    nested = prepare(nested_arm_caller)
    @test (nested(1.5, 2.0, true), nested(1.5, -1.0, true), nested(1.5, -1.0, false)) ==
          (log(2.0) * 1.5, 0.0, 1.0)
    # Prepared code calls each body as one operation (`code_expr`); its recipe
    # source shows where the sibling is called.
    top_sources(k) = [e.recipe.source for e in recipe_inventory(k) if e.depth == 0]
    outputs(k) = [v.name for e in recipe_inventory(k) for v in e.recipe.outputs]
    # The caller's arm calls `f`'s operation, which carries its prepared child.
    nested_source = string(only(top_sources(nested)))
    @test occursin(r"^if c\b", nested_source)
    @test occursin("_KernelPreparedChild", nested_source)
    @test !(:sib in outputs(nested))
    # No sibling value is computed ahead of the branch.
    @test !occursin("sib = ", string(code_expr(f)))
    @test !(:sib in outputs(f))
    @test occursin(r"if x > 0\s+var\"prepared sib\"\(scale, x\)",
                   string(only(top_sources(f))))

    logpdf = prepare(guarded_half.logpdf)
    @test logpdf(1.0, 2.0, 5.0) == log((5.0 - 1.0) / 2.0) - log(2.0)
    @test logpdf(1.0, 2.0, 0.5) == -Inf
    back = prepare(guarded_half.back)
    @test back(1.0, 2.0, 4.0) == 5.0
    @test back(1.0, 2.0, -4.0) == 1.0

    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    ad = prepare_ad(f, backend, 1.5, 2.0; active = :x)
    @test ad_value_and_gradient(ad, 1.5, 2.0) == (log(2.0) * 1.5, 1.5 / 2.0)
    @test ad_value_and_gradient(ad, 1.5, -1.0) == (0.0, 0.0)
    ad = prepare_ad(logpdf, backend, 1.0, 2.0, 5.0; active = :x)
    value, gradient = ad_value_and_gradient(ad, 1.0, 2.0, 5.0)
    @test value == log(2.0) - log(2.0)
    @test gradient ≈ 1 / (5.0 - 1.0)
    @test ad_value_and_gradient(ad, 1.0, 2.0, 0.5) == (-Inf, 0.0)
end

end
