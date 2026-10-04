# Temporary manual IR compatibility for an existing consumer.
# Statistical surface syntax and production model bodies are retired.
using ReactiveKernelsPPL, Test

function _tv_draws(;
        group::Symbol = :g,
        kind::Symbol = :correlated,
        margins::Vector{VaryingMargin} = VaryingMargin[
            VaryingMargin(:Intercept, VaryingZRecipe(:ones, :none, nothing)),
            VaryingMargin(:x, VaryingZRecipe(:column, :x, nothing))],
        lkj_eta::Float64 = 1.0,
        label::Symbol = :draws_g,
        suffix::String = "g")
    return VaryingDraws(group, kind, margins, lkj_eta, label, suffix)
end

function _tv_effect_term(group::Symbol = :g)
    label = Symbol("r_mu_" * string(group))
    return TermSpec(VaryingEffectTerm, [group],
        (draws = Symbol("draws_" * string(group)),), label, label)
end

# Base GLM plan to inject draws into for contract-validator tests.
function _tv_base(; with_effect::Bool = false)
    plan = lower_rkppl(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 2)
            sigma ~ Exponential(1)
            mu = a .+ b .* x
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :g); conditioned = (:y, :x, :g))
    if with_effect
        pred = plan.predictors[1]
        terms = vcat(pred.terms, _tv_effect_term(:g))
        preds = PredictorSpec[pred for pred in plan.predictors]
        preds[1] = PredictorSpec(pred.name, pred.link, terms, pred.label)
        return StructuralPlan(plan.responses, preds, plan.population_priors,
            plan.parameters, plan.assignments, plan.columns, plan.n_obs;
            roles = plan.roles, derived = plan.derived,
            levelmaps = plan.levelmaps,
            plate_parameters = plan.plate_parameters, scans = plan.scans,
            varying_draws = VaryingDraws[_tv_draws()],
            varying_slices = VaryingSlice[VaryingSlice(:draws_g, 1:2, :mu)])
    end
    return plan
end

@testset "varying contract validation" begin
    base = _tv_base()
    # Happy path validates.
    @test validate_structure(_tv_base(; with_effect = true)) === nothing
    # Duplicate labels / bad kind / non-partition slices / margin mismatch.
    dup = _tv_base(; with_effect = true)
    push!(dup.varying_draws, _tv_draws())
    # refused: duplicate draws labels (IR contract)
    @test_throws ContractValidationError validate_structure(dup)
    badkind = _tv_base(; with_effect = true)
    badkind.varying_draws[1] = VaryingDraws(:g, :slope1,
        _tv_draws().margins, NaN, :draws_g, "g")
    # refused: unknown draws kind (IR contract)
    @test_throws ContractValidationError validate_structure(badkind)
    # The retired K=1 kinds are refused on hand-built plans too.
    k1m = [VaryingMargin(:Intercept, VaryingZRecipe(:ones, :none, nothing))]
    retired = _tv_base(; with_effect = true)
    retired.varying_draws[1] = VaryingDraws(:g, :intercept1, k1m, NaN,
        :draws_g, "g")
    # refused: retired K=1 kind `:intercept1` (IR contract)
    @test_throws ContractValidationError validate_structure(retired)
    # K=1 draws carry the canonical eta 1.0.
    k1eta = _tv_base(; with_effect = true)
    k1eta.varying_draws[1] = VaryingDraws(:g, :correlated, k1m, 2.0,
        :draws_g, "g")
    k1eta.varying_slices[1] = VaryingSlice(:draws_g, 1:1, :mu)
    # refused: K=1 draws carry canonical eta 1.0 (IR contract)
    @test_throws ContractValidationError validate_structure(k1eta)
    k1ok = _tv_base(; with_effect = true)
    k1ok.varying_draws[1] = VaryingDraws(:g, :correlated, k1m, 1.0,
        :draws_g, "g")
    k1ok.varying_slices[1] = VaryingSlice(:draws_g, 1:1, :mu)
    @test validate_structure(k1ok) === nothing
    badslice = _tv_base(; with_effect = true)
    badslice.varying_slices[1] = VaryingSlice(:draws_g, 1:1, :mu)
    # refused: slices must partition 1:K exactly once (IR contract)
    @test_throws ContractValidationError validate_structure(badslice)
    # Effect term over unknown draws / dangling slice.
    noeffect = _tv_base()
    pred = noeffect.predictors[1]
    preds = PredictorSpec[pred for pred in noeffect.predictors]
    badterm = TermSpec(VaryingEffectTerm, [:g], (draws = :draws_zz,),
        :r_mu_g, :r_mu_g)
    preds[1] = PredictorSpec(pred.name, pred.link,
        vcat(pred.terms, badterm), pred.label)
    orphan = StructuralPlan(noeffect.responses, preds,
        noeffect.population_priors, noeffect.parameters, noeffect.assignments,
        noeffect.columns, noeffect.n_obs; roles = noeffect.roles,
        derived = noeffect.derived, levelmaps = noeffect.levelmaps,
        plate_parameters = noeffect.plate_parameters, scans = noeffect.scans,
        varying_draws = VaryingDraws[_tv_draws()],
        varying_slices = VaryingSlice[VaryingSlice(:draws_g, 1:2, :mu)])
    # refused: effect term references unknown draws (IR contract)
    @test_throws ContractValidationError validate_structure(orphan)
    dangling = _tv_base()
    dangling2 = StructuralPlan(dangling.responses, dangling.predictors,
        dangling.population_priors, dangling.parameters, dangling.assignments,
        dangling.columns, dangling.n_obs; roles = dangling.roles,
        derived = dangling.derived, levelmaps = dangling.levelmaps,
        plate_parameters = dangling.plate_parameters, scans = dangling.scans,
        varying_draws = VaryingDraws[_tv_draws()],
        varying_slices = VaryingSlice[VaryingSlice(:draws_g, 1:2, :mu)])
    # refused: dangling slice with no effect term (IR contract)
    @test_throws ContractValidationError validate_structure(dangling2)
end

@testset "manual varying IR binds and builds for the existing consumer" begin
    cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5],
        :x => [0.5, -1.0, 1.5, 0.0], :g => [1, 2, 1, 2])
    plan = _tv_base(; with_effect = true)
    bound = bind_data(plan, cols)
    @test isbound(bound)
    @test bound.roles[:g] === :group
    @test_throws ContractValidationError bind_data(plan,
        Dict{Symbol,AbstractVector}(:y => cols[:y], :x => cols[:x]))
    built = build_kernel(bound)
    @test built.layout.total == 10
    query = prepare_query(built, bound, :sampler)
    @test isfinite(Base.invokelatest(query, fill(0.1, built.layout.total)))
end
