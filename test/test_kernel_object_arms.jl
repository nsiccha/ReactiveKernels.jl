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

@kernel quadratic_law() = begin
    logpdf(x::Float64)::Float64 = -x^2
end
@kernel scaled_law(base, scale::Float64) = begin
    logpdf(x::Float64)::Float64 = begin
        z::Float64 = x / scale
        scale == 0 ? -Inf : base.logpdf(z) - log(scale)
    end
end
@kernel scaled_quadratic = scaled_law(quadratic_law)
@kernel scaled_log = scaled_law(half_log)

@kernel quadratic_scores(data::Vector{Float64}, scale::Float64) = begin
    values = plate(data) do x
        scaled_quadratic(scale).logpdf(x)
    end
    return sum(values)
end
@kernel log_scores(data::Vector{Float64}, scale::Float64) = begin
    values = plate(data) do x
        scaled_log(scale).logpdf(x)
    end
    return sum(values)
end
@kernel guarded_scores(data::Vector{Float64}, location::Float64, scale::Float64) = begin
    values = plate(data) do x
        guarded_half(location, scale).logpdf(x)
    end
    return sum(values)
end

# Ordinary lexical values and bare callees have the same reconstruction
# contract as a prepared bound endpoint. A same-named global is a decoy.
const captured_transform = x -> -1000.0
const lexical_law = let captured_transform = x -> 2x, offset = [3.0]
    @kernel lexical(scale::Float64, x::Float64) = begin
        result::Float64 = scale > 0 ? captured_transform(x) : offset[1] + x
    end
end
@kernel lexical_scores(data::Vector{Float64}, scale::Float64) = begin
    values = plate(data) do x
        lexical_law(scale, x)
    end
    return sum(values)
end

const boxed = let offset = 1.0
    law = @kernel boxed_spec(scale::Float64, x::Float64) = begin
        result::Float64 = scale > 0 ? x + offset : -x
    end
    (law, value -> (offset = value))
end
const boxed_law = first(boxed)
@kernel boxed_scores(data::Vector{Float64}, scale::Float64) = begin
    values = plate(data) do x
        boxed_law(scale, x)
    end
    return sum(values)
end

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

@testset "recovered plate arms preserve bound endpoint captures" begin
    backend = AutoEnzyme(; mode = Enzyme.Reverse)
    function regions(ex, counts = Dict(:for => 0, :while => 0, :if => 0))
        ex isa Expr || return counts
        haskey(counts, ex.head) && (counts[ex.head] += 1)
        foreach(arg -> regions(arg, counts), ex.args)
        counts
    end
    structures = Dict{Any,Any}()
    # The shared scale guard makes lowering recover the arm's graph from its
    # source. Its prepared base endpoint must remain a lexical capture in
    # every split part, including on subsequent bindings and other aliases.
    for n in (0, 2, 19)
        data = [isodd(i) ? 0.2 : 0.7 for i in 1:n]
        saved = copy(data)
        unbound = prepare(quadratic_scores; type_stability = :error)
        for policy in (nothing, :ignore)
            bound = prepare(quadratic_scores; bound = (; data), on_error = policy,
                            type_stability = :error)
            for scale in (1.0, 2.0)
                expected = -sum(abs2, data) / scale^2 - n * log(scale)
                @test unbound(data, scale) ≈ expected
                @test bound(scale) ≈ expected
                ad = prepare_ad(bound, backend, scale; active = :scale)
                value, gradient = ad_value_and_gradient(ad, scale)
                @test value ≈ expected
                @test gradient ≈ 2sum(abs2, data) / scale^3 - n / scale
            end
            @test bound(0.0) == (n == 0 ? 0.0 : -Inf)
            # Growing bound data retains the same emitted loop/branch bodies.
            # An empty domain may simplify them away (constraints.md).
            if n > 0
                shape = regions(code_expr(bound))
                @test shape == get!(structures, policy, shape)
                @test shape[:for] + shape[:while] > 0
            end
        end
        @test data == saved
    end
    @test prepare(quadratic_scores; bound = (data = [0.2, 0.7],))(1.0) ≈ -0.53

    logarithmic = prepare(log_scores; bound = (data = [0.2, 0.7],))
    @test logarithmic(2.0) ≈ sum(log, [0.2, 0.7]) - 4log(2.0)
    @test ad_value_and_gradient(prepare_ad(logarithmic, backend, 2.0;
        active = :scale), 2.0)[2] ≈ -2.0
    inactive = prepare(log_scores; bound = (data = [-1.0],))
    @test inactive(0.0) == -Inf # log(-Inf) in the untaken endpoint must not run.
    @test ad_value_and_gradient(prepare_ad(inactive, backend, 0.0;
        active = :scale), 0.0) == (-Inf, 0.0)
    @test_throws DomainError inactive(1.0)

    # A lane-dependent condition uses the other arm-recovery entry point.
    lane = prepare(guarded_scores; bound = (data = [0.2, 0.7],))
    @test lane(0.0, 2.0) ≈ sum(log, [0.2, 0.7]) - 4log(2.0)
    ad = prepare_ad(lane, backend, 0.0, 2.0; active = :scale)
    @test ad_value_and_gradient(ad, 0.0, 2.0)[2] ≈ -2.0
    @test ad_value_and_gradient(ad, 1.0, 2.0) == (-Inf, 0.0)

    lexical = prepare(lexical_scores; bound = (data = [1.0, 2.0],))
    @test lexical(1.0) == 6.0
    @test lexical(-1.0) == 9.0
    shared_cell = prepare(boxed_scores; bound = (data = [1.0, 2.0],))
    @test shared_cell(1.0) == 5.0
    last(boxed)(4.0)
    @test shared_cell(1.0) == 11.0
    @test shared_cell(-1.0) == -3.0
end

end
