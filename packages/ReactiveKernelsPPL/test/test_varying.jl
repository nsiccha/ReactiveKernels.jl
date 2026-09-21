# Varying stages A–C: draws/slice IR + surface lowering + validation
# (Stage A), K=1 codegen (Stage B), LKJ-correlated codegen (Stage C). All
# three geometries build end to end (tested below), in both the fused
# (`r ~ varying_effect(g, [margins...])`) and split (`d ~
# varying_draws(g, [margins...])` + `r ~ varying_slice(d, cols)`)
# spellings.

using SpecialFunctions: loggamma

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
        end, (:y, :x, :g))
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

@testset "varying effect lowering fused" begin
    got = lower_rkppl(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 2)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [1, x]; eta = 1.0)
            mu = a .+ b .* x .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :g))
    @test _draws_equal(got.varying_draws, VaryingDraws[_tv_draws()])
    @test only(got.varying_draws).suffix == "g"
    @test got.varying_slices == [VaryingSlice(:draws_g, 1:2, :mu)]
    t = last(got.predictors[1].terms)
    @test t.kind === VaryingEffectTerm
    @test t.columns == [:g]
    @test t.options == (draws = :draws_g,)
    @test t.addressee === t.label === :r_mu_g
    # Multi-target slices: the split form with explicit column ranges.
    multi = lower_rkppl(quote
            d ~ varying_draws(g, [1, x, 1])
            r1 ~ varying_slice(d, 1:2)
            r2 ~ varying_slice(d, 3)
            mu = a .+ r1
            sg = c .+ r2
            y .~ Normal.(mu, 1.5)
            y2 .~ Normal.(sg, 1.5)
        end, (:y, :y2, :g, :x))
    b = only(multi.varying_draws)
    @test multi.varying_slices ==
        [VaryingSlice(:draws_g, 1:2, :mu), VaryingSlice(:draws_g, 3:3, :sg)]
    @test [m.coefficient for m in b.margins] == [:Intercept, :x, :Intercept]
    @test b.lkj_eta == 1.0
end

@testset "varying lowering K=1 kinds" begin
    one = lower_rkppl(quote
            r ~ varying_effect(g, [1])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :g))
    b = only(one.varying_draws)
    @test b.kind === :intercept1
    @test isnan(b.lkj_eta)
    @test b.label === :draws_g
    @test b.suffix == "g"
    slope = lower_rkppl(quote
            r ~ varying_effect(g, [x])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    bs = only(slope.varying_draws)
    @test bs.kind === :slope1
    @test isnan(bs.lkj_eta)
    # K=1 WITH eta takes the vacuous-1x1-LKJ correlated route.
    kid = lower_rkppl(quote
            r ~ varying_effect(g, [1]; eta = 2.0)
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :g))
    bk = only(kid.varying_draws)
    @test bk.kind === :correlated
    @test bk.lkj_eta == 2.0
end

@testset "varying dummy margins" begin
    got = lower_rkppl(quote
            r ~ varying_effect(g, [1, dummy(c, 2), dummy(s, "a")])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :c, :s, :g))
    b = only(got.varying_draws)
    @test b.kind === :correlated
    @test _vmargins_equal([b.margins[2]],
        [VaryingMargin(:c_dummy_2, VaryingZRecipe(:dummy, :c, 2))])
    @test b.margins[3].coefficient === :s_dummy_a
    @test b.margins[3].z.level == "a"
end

@testset "varying surface failures" begin
    D = (:y, :x, :g)
    # Quoted id position is gone (blocks on one grouping disambiguate by
    # binding name, not labels).
    @test_throws SurfaceLoweringError lower_rkppl(quote
            r ~ varying_effect(:ID, g, [1, x])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, D)
    # Non-data group.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            r ~ varying_effect(h, [1])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, D)
    # Non-positive eta.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            r ~ varying_effect(g, [1, x]; eta = 0.0)
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, D)
    # Duplicate binding (single assignment).
    @test_throws SurfaceLoweringError lower_rkppl(quote
            r ~ varying_effect(g, [1])
            r ~ varying_effect(g, [1])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, D)
    # One slice per (draws, target): two slices of one draws into one
    # predictor.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            d ~ varying_draws(g, [1, x])
            r1 ~ varying_slice(d, 1)
            r2 ~ varying_slice(d, 2)
            mu = a .+ r1 .+ r2
            y .~ Normal.(mu, 1.5)
        end, D)
    # Bad margin integer / unknown margin / tuple-not-vect / bare margin.
    for bad in (:([2]), :([zz]), :((1, x)), 1)
        @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
                Expr(:call, :~, :r,
                    Expr(:call, :varying_effect, :g, bad)),
                :(mu = a .+ r), :(y .~ Normal.(mu, 1.5))), D)
    end
    # Unknown draws / unconsumed margins.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            r2 ~ varying_slice(zz, 1)
            mu = a .+ r2
            y .~ Normal.(mu, 1.5)
        end, D)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            d ~ varying_draws(g, [1, x])
            r1 ~ varying_slice(d, 1)
            mu = a .+ r1
            y .~ Normal.(mu, 1.5)
        end, D)
    # Negated / nested contributions.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            r ~ varying_effect(g, [1])
            mu = a .- r
            y .~ Normal.(mu, 1.5)
        end, D)
    @test_throws SurfaceLoweringError lower_rkppl(quote
            r ~ varying_effect(g, [1])
            mu = a .+ r .* x
            y .~ Normal.(mu, 1.5)
        end, D)
    # Contribution inside a non-predictor definition.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            r ~ varying_effect(g, [1])
            w = r .+ 1.0
            mu = a .+ b .* x .+ r
            y .~ Normal.(mu, sigma)
            sigma ~ Exponential(1)
        end, (:y, :x, :g))
    # Contribution inside a STRUCTURAL definition (coefficient reference inlines
    # even vector defs — the statement-level screen must catch it before
    # absorption silently turns the alias into a direct summand).
    @test_throws SurfaceLoweringError lower_rkppl(quote
            r ~ varying_effect(g, [1])
            w = r .+ b .* x
            mu = a .+ w
            y .~ Normal.(mu, sigma)
            sigma ~ Exponential(1)
        end, (:y, :x, :g))
    # Reserved names.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            dummy ~ Normal(0, 1)
            y .~ Normal.(mu, 1.5)
        end, (:y,))
end

@testset "varying contract validation" begin
    base = _tv_base()
    # Happy path validates.
    @test validate_structure(_tv_base(; with_effect = true)) === nothing
    # Duplicate labels / bad kind / non-partition slices / margin mismatch.
    dup = _tv_base(; with_effect = true)
    push!(dup.varying_draws, _tv_draws())
    @test_throws ContractValidationError validate_structure(dup)
    badkind = _tv_base(; with_effect = true)
    badkind.varying_draws[1] = VaryingDraws(:g, :slope1,
        _tv_draws().margins, NaN, :draws_g, "g")
    @test_throws ContractValidationError validate_structure(badkind)
    badslice = _tv_base(; with_effect = true)
    badslice.varying_slices[1] = VaryingSlice(:draws_g, 1:1, :mu)
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
    @test_throws ContractValidationError validate_structure(orphan)
    dangling = _tv_base()
    dangling2 = StructuralPlan(dangling.responses, dangling.predictors,
        dangling.population_priors, dangling.parameters, dangling.assignments,
        dangling.columns, dangling.n_obs; roles = dangling.roles,
        derived = dangling.derived, levelmaps = dangling.levelmaps,
        plate_parameters = dangling.plate_parameters, scans = dangling.scans,
        varying_draws = VaryingDraws[_tv_draws()],
        varying_slices = VaryingSlice[VaryingSlice(:draws_g, 1:2, :mu)])
    @test_throws ContractValidationError validate_structure(dangling2)
end

@testset "varying bind validation" begin
    cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5],
        :x => [0.5, -1.0, 1.5, 0.0], :g => [1, 2, 1, 2])
    plan = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [1])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    bound = bind_data(plan, cols)
    @test bound.roles[:g] === :group
    # Group column missing.
    @test_throws ContractValidationError bind_data(plan,
        Dict{Symbol,AbstractVector}(:y => cols[:y], :x => cols[:x]))
    # Non-numeric continuous Z.
    strz = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [s])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :s, :g))
    @test_throws ContractValidationError bind_data(strz,
        Dict{Symbol,AbstractVector}(:y => cols[:y],
            :s => ["a", "b", "a", "b"], :g => cols[:g]))
    # Dummy membership ok + fail.
    dym = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [dummy(s, "a")])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :s, :g))
    bound2 = bind_data(dym, Dict{Symbol,AbstractVector}(:y => cols[:y],
        :s => ["a", "b", "a", "b"], :g => cols[:g]))
    @test isbound(bound2)
    dymbad = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [dummy(s, "z")])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :s, :g))
    @test_throws ContractValidationError bind_data(dymbad,
        Dict{Symbol,AbstractVector}(:y => cols[:y],
            :s => ["a", "b", "a", "b"], :g => cols[:g]))
end

@testset "varying Stage-C builds" begin
    cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5],
        :x => [0.5, -1.0, 1.5, 0.0], :g => [1, 2, 1, 2])
    # K=1 draws build (intercept + slope); LKJ-correlated builds too.
    iplan = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [1])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    ibuilt = build_kernel(bind_data(iplan, cols))
    # a + log_scale_g + 2 xi cells.
    @test ibuilt.layout.total == 4
    splan = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [x])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    sbuilt = build_kernel(bind_data(splan, cols))
    @test sbuilt.layout.total == 4
    cplan = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [1, x])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    cbuilt = build_kernel(bind_data(cplan, cols))
    # a + 1 theta + 2 tau + 2*2 z cells.
    @test cbuilt.layout.total == 8
    k1cplan = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [1]; eta = 1.0)
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    # K=1 with eta is :correlated (L packs zero coords).
    k1cbuilt = build_kernel(bind_data(k1cplan, cols))
    # a + 0 thetas + 1 tau + 2 z cells.
    @test k1cbuilt.layout.total == 4
end

@testset "varying K=1 names" begin
    iplan = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [1])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    ib = only(iplan.varying_draws)
    @test ib.kind === :intercept1
    @test ReactiveKernelsPPL._varying_k1_names(ib) ===
        (:log_scale_g, :xi_g)
    splan = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [x])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    sb = only(splan.varying_draws)
    @test sb.kind === :slope1
    @test ReactiveKernelsPPL._varying_k1_names(sb) === (:tau_g, :xi_g)
    cplan = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [1, x])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    @test_throws ContractValidationError ReactiveKernelsPPL._varying_k1_names(
        only(cplan.varying_draws))
    # Claims: user definitions cannot collide with K=1 names.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 1)
            tau_g = 1.0
            r ~ varying_effect(g, [x])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [1])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
            xi_g ~ Normal(0, 1)
        end, (:y, :x, :g))
    # Name tables: a hand-built parameter under a K=1 name fails.
    clash = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [1])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    push!(clash.parameters, SampledParameter(:log_scale_g, :normal,
        (arg1 = 0, arg2 = 1), nothing, :log_scale_g))
    @test_throws ContractValidationError validate_structure(clash)
end

function _tv_k1_cols()
    g = ["b", "a", "c", "a", "b", "c", "a", "b"]
    y = [0.5, -0.2, 0.8, 0.1, -0.5, 0.3, 0.0, 0.2]
    return Dict{Symbol,AbstractVector}(:g => g, :y => y), g, y
end

@testset "varying K=1 layout" begin
    cols, _, _ = _tv_k1_cols()
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [1])
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :g))
    bound = bind_data(plan, cols)
    layout = assign_layout(bound)
    @test [e.kind for e in layout.entries] ==
        [:coefficient, :sampled, :sampled, :varying]
    @test [e.size for e in layout.entries] == [1, 1, 1, 3]
    @test [e.transform for e in layout.entries] ==
        [:identity, :exp, :identity, :identity]
    @test layout.total == 6
    @test coordinate_names(layout)[2:3] ==
        [:sigma, :log_scale_g]
    u = [0.2, 0.1, -0.3, 0.4, 0.0, -0.1]
    nt = constrain(layout, u)
    @test nt.log_scale_g == -0.3 && nt.xi_g == [0.4, 0.0, -0.1]
    @test unconstrain(layout, nt) ≈ u
    @test logjac(layout, u) ≈ u[2]
    # Slope draws: tau rides :exp (Stan lower-bound Jacobian, no renorm).
    splan = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [x])
            mu = a .+ r
            y .~ Normal.(mu, 1.0)
        end, (:y, :x, :g))
    scols = Dict{Symbol,AbstractVector}(:g => cols[:g], :y => cols[:y],
        :x => collect(1.0:8.0))
    slayout = assign_layout(bind_data(splan, scols))
    @test [e.kind for e in slayout.entries] ==
        [:coefficient, :sampled, :varying]
    @test [e.transform for e in slayout.entries] ==
        [:identity, :exp, :identity]
    @test coordinate_names(slayout)[2] === :tau_g
    us = [0.5, 0.3, 0.1, 0.2, 0.0]
    @test logjac(slayout, us) ≈ us[2]
end

# Independent intercept reference: SB `exp(log_scale) * xi[idx]` shape
# with explicit names/order (no contract helpers — this pins them).
function _tv_ref_intercept(bound, nt)
    idx = [findfirst(==(v), ["a", "b", "c"]) for v in bound.columns[:g]]
    r = exp(nt.log_scale_g) .* nt.xi_g[idx]
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), bound.columns[:y]))
    pr = logpdf(Normal(0, 5), nt.mu[1]) + logpdf(Exponential(1), nt.sigma) +
        logpdf(Normal(0, 1), nt.log_scale_g) +
        sum(logpdf.(Normal(0, 1), nt.xi_g))
    return (; ll, pr)
end

@testset "varying intercept e2e values and gradient" begin
    cols, _, _ = _tv_k1_cols()
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [1])
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :g))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = [0.2, 0.1, -0.3, 0.4, 0.0, -0.1]
    nt = constrain(built.layout, u)
    ref = _tv_ref_intercept(bound, nt)
    @test _query(built.spec, bound, :likelihood, u) ≈ ref.ll
    @test _query(built.spec, bound, :prior, u) ≈ ref.pr
    @test _query(built.spec, bound, :posterior, u) ≈ ref.ll + ref.pr + u[2]
    _check_gradient(built.spec, bound, u)
end

# Independent slope reference: SB `tau * (xi[idx] .* Z)` association.
function _tv_ref_slope(bound, nt, Z)
    idx = [findfirst(==(v), [1, 2, 3]) for v in bound.columns[:g]]
    r = nt.tau_g .* (nt.xi_g[idx] .* Z)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), bound.columns[:y]))
    pr = logpdf(Normal(0, 5), nt.mu[1]) + logpdf(Exponential(1), nt.sigma) +
        logpdf(Normal(0, 1), nt.tau_g) +
        sum(logpdf.(Normal(0, 1), nt.xi_g))
    return (; ll, pr)
end

@testset "varying slope e2e values and gradient" begin
    _, _, y = _tv_k1_cols()
    g2 = [2, 1, 3, 1, 2, 3, 1, 2]
    x2 = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0, 2.0, -1.5]
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [x])
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :g))
    bound = bind_data(plan,
        Dict{Symbol,AbstractVector}(:g => g2, :y => y, :x => x2))
    built = build_kernel(bound)
    u = [0.2, 0.1, -0.2, 0.3, 0.0, -0.1]
    nt = constrain(built.layout, u)
    ref = _tv_ref_slope(bound, nt, x2)
    @test _query(built.spec, bound, :likelihood, u) ≈ ref.ll
    # No +log(2): SB Stan-convention tau (see the generator comment).
    @test _query(built.spec, bound, :prior, u) ≈ ref.pr
    @test _query(built.spec, bound, :posterior, u) ≈
        ref.ll + ref.pr + u[2] + u[3]
    _check_gradient(built.spec, bound, u)
    # Dummy-Z slope: same shape, indicator Z.
    c3 = [1, 2, 2, 1, 2, 1, 2, 1]
    dplan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [dummy(c, 2)])
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :c, :g))
    dbound = bind_data(dplan,
        Dict{Symbol,AbstractVector}(:g => g2, :y => y, :c => c3))
    dbuilt = build_kernel(dbound)
    dnt = constrain(dbuilt.layout, u)
    dref = _tv_ref_slope(dbound, dnt, Float64.([v == 2 for v in c3]))
    @test _query(dbuilt.spec, dbound, :likelihood, u) ≈ dref.ll
    @test _query(dbuilt.spec, dbound, :prior, u) ≈ dref.pr
    @test _query(dbuilt.spec, dbound, :posterior, u) ≈
        dref.ll + dref.pr + u[2] + u[3]
end

@testset "varying correlated names" begin
    cplan = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [1, x])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    cb = only(cplan.varying_draws)
    @test cb.kind === :correlated
    @test ReactiveKernelsPPL._varying_corr_names(cb) ===
        (:L_g, :tau_g, :z_flat_g)
    # Same-group second draws take the `group_binding` suffix.
    two = lower_rkppl(quote
            a1 ~ Normal(0, 1)
            a2 ~ Normal(0, 1)
            a3 ~ Normal(0, 1)
            d ~ varying_draws(g, [1, x]; eta = 1.0)
            r1 ~ varying_slice(d, 1)
            r2 ~ varying_slice(d, 2)
            e ~ varying_draws(g, [1]; eta = 1.0)
            r3 ~ varying_slice(e, 1)
            mu1 = a1 .+ r1
            mu2 = a2 .+ r2
            mu3 = a3 .+ r3
            y1 .~ Normal.(mu1, 1.5)
            y2 .~ Normal.(mu2, 1.5)
            y3 .~ Normal.(mu3, 1.5)
        end, (:y1, :y2, :y3, :x, :g))
    second = only(d for d in two.varying_draws if d.label === :draws_g_e)
    @test second.suffix == "g_e"
    @test ReactiveKernelsPPL._varying_corr_names(second) ===
        (:L_g_e, :tau_g_e, :z_flat_g_e)
    # K=1 kinds own no correlated names.
    k1plan = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [1])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    @test_throws ContractValidationError ReactiveKernelsPPL._varying_corr_names(
        only(k1plan.varying_draws))
    # Claims: user definitions cannot collide with correlated names.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 1)
            L_g = 1.0
            r ~ varying_effect(g, [1, x])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [1, x])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
            z_flat_g ~ Normal(0, 1)
        end, (:y, :x, :g))
    # The derived draws are claimed too (constrain-output key).
    @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 1)
            b_g = 1.0
            r ~ varying_effect(g, [1, x])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    # Name tables: a hand-built parameter under a correlated name fails.
    clash = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [1, x])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    push!(clash.parameters, SampledParameter(:tau_g, :normal,
        (arg1 = 0, arg2 = 1), nothing, :tau_g))
    @test_throws ContractValidationError validate_structure(clash)
    # Empty slice ranges fail closed (a vacuous slice).
    empty = _tv_base(; with_effect = true)
    empty.varying_slices[1] = VaryingSlice(:draws_g, 2:1, :mu)
    @test_throws ContractValidationError validate_structure(empty)
end

# NOTE (F1, 2026-09-19): this testset used to pin `lkj_chol_logjac`
# against a central-difference 1/2 logdet(J'J) of u -> vec(L) — the
# hyperspherical sphere-volume Gram factor. That expectation WAS the
# bug: the Stan-verbatim `lkj_corr_cholesky_logpdf` it pairs with is
# a density w.r.t. the correlation-matrix (Omega-pullback) volume
# element, whose Gram exponent is (i-j), one log-sin per angle more
# (posterior parity vs SBBRMI failed on L.2.1/L.2.2 until the fix).
# The Jacobian oracle now lives in test_lkj_jacobian.jl (Omega-volume
# identity + K=2 closed form); this testset keeps roundtrip/shape/
# fail-closed coverage only.

@testset "varying LKJ transform" begin
    # Packed-dim inversion + fail-closed on non-triangular lengths.
    @test ReactiveKernelsPPL._lkj_dim(0) == 1
    @test ReactiveKernelsPPL._lkj_dim(1) == 2
    @test ReactiveKernelsPPL._lkj_dim(3) == 3
    @test ReactiveKernelsPPL._lkj_dim(6) == 4
    @test_throws ContractValidationError ReactiveKernelsPPL._lkj_dim(2)
    @test_throws ContractValidationError ReactiveKernelsPPL._lkj_dim(5)
    # Roundtrip + Cholesky validity, K = 1..4.
    for (K, u) in ((1, Float64[]), (2, [0.3]), (3, [0.3, -0.5, 0.7]),
            (4, [0.3, -0.5, 0.7, 0.1, -0.2, 0.4]))
        L = lkj_chol_constrain(u, K)
        @test size(L) == (K, K)
        @test L[1, 1] == 1.0
        for i in 1:K, j in i+1:K
            @test L[i, j] == 0.0
        end
        for i in 1:K
            @test sum(L[i, 1:i] .^ 2) ≈ 1.0
            @test L[i, i] > 0.0
        end
        @test lkj_chol_unconstrain(L, K) ≈ u
    end
    # Log-Jacobian oracle: test_lkj_jacobian.jl (see NOTE above).
    @test lkj_chol_logjac(Float64[], 1) == 0.0
    # Fail-closed: length mismatch, non-square, off-hemisphere.
    @test_throws ContractValidationError lkj_chol_constrain([0.1], 3)
    @test_throws ContractValidationError lkj_chol_unconstrain([1.0 0.0], 2)
    @test_throws ContractValidationError lkj_chol_unconstrain(
        [1.0 0.0; 0.5 -0.5], 2)
end

@testset "varying LKJ constant" begin
    # K=1 is ±0.0 (Stan's K=1 term: no diagonal, zero constant).
    @test lkj_logconst(1, 1.0) == 0.0
    @test lkj_logconst(1, 2.5) == 0.0
    # K=2 closed form from the Beta integral ∫(1-r²)^{η-1}dr
    # (independent of the LKJ09-theorem-5 port): hand-evaluated at
    # half-integer/integer etas where the gammas telescope.
    @test lkj_logconst(2, 0.5) ≈ -log(pi)
    @test lkj_logconst(2, 1.0) ≈ -log(2.0)
    @test lkj_logconst(2, 2.0) ≈ log(0.75)
    @test lkj_logconst(2, 3.0) ≈ log(0.9375)
    # K=3/K=4 eta==1.0 branches, hand-evaluated from Stan's formula.
    @test lkj_logconst(3, 1.0) ≈ log(2.0) - 2 * log(pi)
    @test lkj_logconst(4, 1.0) ≈ 3 * log(6.0) - 2 * log(pi) - 8 * log(2.0)
    # lpdf: K=1 exactly zero; K=2 eta==1.0 is the bare constant.
    @test lkj_corr_cholesky_logpdf(reshape([1.0], 1, 1), 2.0) == 0.0
    L = lkj_chol_constrain([0.3], 2)
    @test lkj_corr_cholesky_logpdf(L, 1.0) ≈ -log(2.0)
    # K=2 general branch: const + (2η-2) log L22.
    @test lkj_corr_cholesky_logpdf(L, 2.0) ≈ log(0.75) + 2 * log(L[2, 2])
    @test_throws ContractValidationError lkj_corr_cholesky_logpdf(
        [1.0 0.0], 1.0)
end

@testset "varying correlated layout" begin
    cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
        :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
        :g => [1, 2, 1, 3, 2, 3])
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [1, x])
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :g))
    bound = bind_data(plan, cols)
    layout = assign_layout(bound)
    # `a` rides the intercept coefficient; sampled = [sigma, draws triple].
    @test [e.kind for e in layout.entries] ==
        [:coefficient, :sampled, :varying_corr, :varying, :varying]
    @test [e.name for e in layout.entries] ==
        [:mu_coef, :sigma, :L_g, :tau_g, :z_flat_g]
    @test [e.size for e in layout.entries] == [1, 1, 1, 2, 6]
    @test [e.transform for e in layout.entries] ==
        [:identity, :exp, :lkj, :exp, :identity]
    @test layout.total == 11
    @test coordinate_names(layout)[2] === :sigma
    @test coordinate_names(layout)[3] == Symbol("L_g.1")
    @test coordinate_names(layout)[4:5] ==
        [Symbol("tau_g.1"), Symbol("tau_g.2")]
    @test coordinate_names(layout)[6] == Symbol("z_flat_g.1")
    u = collect(range(-0.5, 0.5; length = layout.total))
    nt = constrain(layout, u)
    @test size(nt.L_g) == (2, 2)
    @test nt.tau_g ≈ exp.(u[4:5])
    @test nt.z_flat_g == u[6:11]
    # Derived draws: shape + the trivially hand-checkable b[1,1].
    @test size(nt.b_g) == (3, 2)
    @test nt.b_g[1, 1] ≈ nt.tau_g[1] * nt.z_flat_g[1]
    # Roundtrip ignores the derived b (constrain output feeds unconstrain).
    @test unconstrain(layout, nt) ≈ u
    # Jacobian: sigma + tau exps + the K=2 theta term, hand-summed
    # (the (i-j) = 1 exponent keeps one log-sin term — F1 fix; the old
    # pure-logistic pin asserted the sphere-volume bug).
    s = 1 / (1 + exp(-u[3]))
    lkj = log(sin(pi * s)) + log(pi) + log(s) + log1p(-s)
    @test logjac(layout, u) ≈ u[2] + u[4] + u[5] + lkj
end

# Independent correlated-contribution reference: SB `(diag(tau)*L*z)'`
# shape with explicit per-margin/per-group loops (never the fused forms),
# over the global margin subset `js` with Z columns `Zs` (Zs[j] is the
# j-th GLOBAL margin's column; `:ones` margins pass `ones(n)`).
# `levels` overrides the numbering order (declared-order tests); default
# is sort order (the bind fill for plain vectors).
function _tv_ref_corr_r(bound, groupcol, L, tau, zflat, Zs, js; levels = nothing)
    K = length(Zs)
    lv = levels === nothing ? sort!(unique(bound.columns[groupcol])) : levels
    idx = [findfirst(==(v), lv) for v in bound.columns[groupcol]]
    r = zeros(Float64, length(idx))
    for m in eachindex(idx)
        g = idx[m]
        for j in js
            acc = 0.0
            for s in 1:j
                acc += tau[j] * L[j, s] * zflat[s + (g - 1) * K]
            end
            r[m] += Zs[j][m] * acc
        end
    end
    return r
end

# K=2 LKJ from the Beta-integral closed form (independent of the
# LKJ09-theorem-5 `lkj_logconst` port the emitter inlines).
function _tv_ref_lkj_k2(L, eta)
    c = loggamma(eta + 0.5) - loggamma(eta) - 0.5 * log(pi)
    return c + (2 * eta - 2) * log(L[2, 2])
end

@testset "varying correlated e2e values and gradient" begin
    gv = [1, 2, 1, 3, 2, 3]
    xv = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    yv = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    cols = Dict{Symbol,AbstractVector}(:g => gv, :x => xv, :y => yv)
    # Both LKJ emission branches: eta == 1.0 fast path + general.
    for eta in (1.0, 2.0)
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                sigma ~ Exponential(1)
                r ~ varying_effect(g, [1, x]; eta = $eta)
                mu = a .+ r
                y .~ Normal.(mu, sigma)
            end, (:y, :x, :g))
        bound = bind_data(plan, cols)
        built = build_kernel(bound)
        u = collect(range(-0.4, 0.4; length = built.layout.total))
        nt = constrain(built.layout, u)
        r = _tv_ref_corr_r(bound, :g, nt.L_g, nt.tau_g, nt.z_flat_g,
            [ones(6), xv], 1:2)
        ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), yv))
        pr = logpdf(Normal(0, 5), nt.mu[1]) +
            logpdf(Exponential(1), nt.sigma) +
            _tv_ref_lkj_k2(nt.L_g, eta) +
            sum(logpdf.(Normal(0, 1), nt.tau_g)) +
            sum(logpdf.(Normal(0, 1), nt.z_flat_g))
        @test _query(built.spec, bound, :likelihood, u) ≈ ll
        # No +log(2): SB Stan-convention tau (see the generator comment).
        @test _query(built.spec, bound, :prior, u) ≈ pr
        s = 1 / (1 + exp(-u[3]))
        jac = u[2] + u[4] + u[5] + log(sin(pi * s)) + log(pi) + log(s) +
            log1p(-s)
        @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
        _check_gradient(built.spec, bound, u)
    end
end

@testset "varying K=3 e2e values and gradient" begin
    gv = [1, 2, 1, 3, 2, 3]
    x1v = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    x2v = [1.0, 0.5, -0.5, 2.0, -1.5, 0.0]
    yv = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    cols = Dict{Symbol,AbstractVector}(:g => gv, :x1 => x1v, :x2 => x2v,
        :y => yv)
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [1, x1, x2])
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :x1, :x2, :g))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    # coef + sigma + 3 thetas + 3 tau + 3*3 z cells.
    @test built.layout.total == 17
    u = collect(range(-0.4, 0.4; length = built.layout.total))
    nt = constrain(built.layout, u)
    r = _tv_ref_corr_r(bound, :g, nt.L_g, nt.tau_g, nt.z_flat_g,
        [ones(6), x1v, x2v], 1:3)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), yv))
    # K=3 eta==1.0 LKJ, hand-derived from Stan's do_lkj_constant
    # (const = log2 - 2logpi; diag (3-2)logL22 + 0*logL33).
    lkj = log(2.0) - 2 * log(pi) + log(nt.L_g[2, 2])
    pr = logpdf(Normal(0, 5), nt.mu[1]) +
        logpdf(Exponential(1), nt.sigma) + lkj +
        sum(logpdf.(Normal(0, 1), nt.tau_g)) +
        sum(logpdf.(Normal(0, 1), nt.z_flat_g))
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    # Jacobian: sigma + tau exps + row-2/row-3 theta terms (the
    # (i-j) Gram exponents: 1, 2, 1 — F1 fix).
    s3 = 1 / (1 + exp(-u[3]))
    s4 = 1 / (1 + exp(-u[4]))
    s5 = 1 / (1 + exp(-u[5]))
    lj = log(sin(pi * s3)) + log(pi) + log(s3) + log1p(-s3) +
        2 * log(sin(pi * s4)) + log(pi) + log(s4) + log1p(-s4) +
        log(sin(pi * s5)) + log(pi) + log(s5) + log1p(-s5)
    jac = u[2] + u[6] + u[7] + u[8] + lj
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
    _check_gradient(built.spec, bound, u)
end

@testset "varying K=1 correlated e2e values and gradient" begin
    gv = [1, 2, 1, 2]
    xv = [0.5, -1.0, 1.5, 0.0]
    yv = [1.0, 2.0, 1.5, 2.5]
    cols = Dict{Symbol,AbstractVector}(:g => gv, :x => xv, :y => yv)
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [x]; eta = 1.0)
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :g))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    # coef + sigma + 0 thetas + 1 tau + 2 z cells.
    @test built.layout.total == 5
    u = [0.2, 0.1, -0.3, 0.4, -0.1]
    nt = constrain(built.layout, u)
    @test nt.L_g == [1.0;;]
    idx = [findfirst(==(v), [1, 2]) for v in gv]
    r = nt.tau_g[1] .* (nt.z_flat_g[idx] .* xv)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), yv))
    # No LKJ term: Stan's K=1 LKJ contributes exactly 0.0.
    pr = logpdf(Normal(0, 5), nt.mu[1]) +
        logpdf(Exponential(1), nt.sigma) +
        logpdf(Normal(0, 1), nt.tau_g[1]) +
        sum(logpdf.(Normal(0, 1), nt.z_flat_g))
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + u[2] + u[3]
    _check_gradient(built.spec, bound, u)
end

@testset "varying multislice e2e values and gradient" begin
    gv = [1, 2, 1, 3, 2, 3]
    xv = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    y1v = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    y2v = [0.5, 1.5, 1.0, 2.0, 2.5, 1.5]
    cols = Dict{Symbol,AbstractVector}(:g => gv, :x => xv, :y1 => y1v,
        :y2 => y2v)
    plan = lower_rkppl(quote
            a1 ~ Normal(0, 5)
            a2 ~ Normal(0, 5)
            s ~ Exponential(1)
            d ~ varying_draws(g, [1, x])
            r1 ~ varying_slice(d, 1)
            r2 ~ varying_slice(d, 2)
            mu1 = a1 .+ r1
            mu2 = a2 .+ r2
            y1 .~ Normal.(mu1, s)
            y2 .~ Normal.(mu2, s)
        end, (:y1, :y2, :x, :g))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    @test [e.kind for e in built.layout.entries] ==
        [:coefficient, :coefficient, :sampled, :varying_corr, :varying, :varying]
    # 2 coefs + s + 1 theta + 2 tau + 2*3 z cells.
    @test built.layout.total == 12
    u = collect(range(-0.4, 0.4; length = built.layout.total))
    nt = constrain(built.layout, u)
    Zs = [ones(6), xv]
    r1 = _tv_ref_corr_r(bound, :g, nt.L_g, nt.tau_g, nt.z_flat_g,
        Zs, 1:1)
    r2 = _tv_ref_corr_r(bound, :g, nt.L_g, nt.tau_g, nt.z_flat_g,
        Zs, 2:2)
    ll = sum(logpdf.(Normal.(nt.mu1[1] .+ r1, nt.s), y1v)) +
        sum(logpdf.(Normal.(nt.mu2[1] .+ r2, nt.s), y2v))
    pr = logpdf(Normal(0, 5), nt.mu1[1]) + logpdf(Normal(0, 5), nt.mu2[1]) +
        logpdf(Exponential(1), nt.s) + _tv_ref_lkj_k2(nt.L_g, 1.0) +
        sum(logpdf.(Normal(0, 1), nt.tau_g)) +
        sum(logpdf.(Normal(0, 1), nt.z_flat_g))
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    s4 = 1 / (1 + exp(-u[4]))
    jac = u[3] + u[5] + u[6] + log(sin(pi * s4)) + log(pi) + log(s4) +
        log1p(-s4)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
    _check_gradient(built.spec, bound, u)
end

function _tv_derived_cols()
    xv = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    zv = [1.0, 0.5, -0.5, 2.0, -1.5, 0.25]
    yv = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    gv = [1, 2, 1, 3, 2, 3]
    return xv, zv, yv, gv
end

@testset "varying derived margin K=1 slope vs baked twin" begin
    xv, zv, yv, gv = _tv_derived_cols()
    wv = xv .* zv
    dplan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            w = x .* z
            r ~ varying_effect(g, [w])
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :z, :g))
    db = only(dplan.varying_draws)
    @test db.kind === :slope1
    @test only(db.margins).z == VaryingZRecipe(:column, :w, nothing)
    @test any(d -> d.name === :w, dplan.derived)
    # Forward reference: draws before the definition lowers identically
    # (the partition-time gate admits defined names; shape proves later).
    fwd = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [w])
            mu = a .+ r
            y .~ Normal.(mu, sigma)
            w = x .* z
        end, (:y, :x, :z, :g))
    @test only(only(fwd.varying_draws).margins).z ==
        VaryingZRecipe(:column, :w, nothing)
    # Emitter-baked twin: `w` arrives as a raw column (option-A shape).
    tplan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [w])
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :w, :g))
    dbound = bind_data(dplan, Dict{Symbol,AbstractVector}(:y => yv, :x => xv,
        :z => zv, :g => gv))
    tbound = bind_data(tplan, Dict{Symbol,AbstractVector}(:y => yv, :w => wv,
        :g => gv))
    dbuilt = build_kernel(dbound)
    tbuilt = build_kernel(tbound)
    @test dbuilt.layout.total == tbuilt.layout.total
    @test coordinate_names(dbuilt.layout) == coordinate_names(tbuilt.layout)
    u = [0.2, 0.1, -0.2, 0.3, 0.0, -0.1]
    for q in (:likelihood, :prior, :posterior)
        @test _query(dbuilt.spec, dbound, q, u) ≈
            _query(tbuilt.spec, tbound, q, u)
    end
    _check_gradient(dbuilt.spec, dbound, u)
end

@testset "varying derived margin correlated slice vs baked twin" begin
    xv, zv, yv, gv = _tv_derived_cols()
    wv = xv .* zv
    dplan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            w = x .* z
            r ~ varying_effect(g, [1, w])
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :z, :g))
    db = only(dplan.varying_draws)
    @test db.kind === :correlated
    @test [m.z.kind for m in db.margins] == [:ones, :column]
    @test db.margins[2].z.column === :w
    tplan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [1, w])
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :w, :g))
    dbound = bind_data(dplan, Dict{Symbol,AbstractVector}(:y => yv, :x => xv,
        :z => zv, :g => gv))
    tbound = bind_data(tplan, Dict{Symbol,AbstractVector}(:y => yv, :w => wv,
        :g => gv))
    dbuilt = build_kernel(dbound)
    tbuilt = build_kernel(tbound)
    @test dbuilt.layout.total == 11
    @test dbuilt.layout.total == tbuilt.layout.total
    @test coordinate_names(dbuilt.layout) == coordinate_names(tbuilt.layout)
    u = collect(range(-0.4, 0.4; length = dbuilt.layout.total))
    for q in (:likelihood, :prior, :posterior)
        @test _query(dbuilt.spec, dbound, q, u) ≈
            _query(tbuilt.spec, tbound, q, u)
    end
    _check_gradient(dbuilt.spec, dbound, u)
end

@testset "varying derived margin failures" begin
    # Scalar derived local stays rejected (vector-shaped only).
    @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 5)
            m = mean(x)
            r ~ varying_effect(g, [m])
            mu = a .+ r
            y .~ Normal.(mu, m)
        end, (:y, :x, :g))
    # Inline expressions bind via an assignment first.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            r ~ varying_effect(g, [x .* z])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :z, :g))
    # A predictor location inlines and emits no Z column.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            r ~ varying_effect(g, [mu])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    # Sampled parameters are not Z columns.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 5)
            r ~ varying_effect(g, [a])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    # Predictor structure absorbed into the LP emits no column either.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 5)
            b ~ Normal(0, 2)
            sigma ~ Exponential(1)
            w = b .* x
            r ~ varying_effect(g, [w])
            mu = a .+ w .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :g))
    # `dummy` needs a raw column (level membership needs bound values).
    @test_throws SurfaceLoweringError lower_rkppl(quote
            w = x .* z
            r ~ varying_effect(g, [dummy(w, 1)])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :z, :g))
    # Grouping columns stay raw.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            w = x .* z
            r ~ varying_effect(w, [1])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :z, :g))
end

@testset "varying correlated restore_draws" begin
    cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5],
        :x => [0.5, -1.0, 1.5, 0.0], :g => [1, 2, 1, 2])
    plan = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [1, x])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :x, :g))
    layout = assign_layout(bind_data(plan, cols))
    U = hcat(collect(range(-0.4, 0.4; length = layout.total)),
        collect(range(0.4, -0.4; length = layout.total)))
    draws = restore_draws(layout, U)
    # LKJ factors + derived draws restore as vectors of matrices.
    @test draws.L_g isa Vector{Matrix{Float64}}
    @test draws.b_g isa Vector{Matrix{Float64}}
    @test size(draws.L_g[1]) == (2, 2)
    @test size(draws.b_g[2]) == (2, 2)
    @test draws.L_g[1] ≈ constrain(layout, U[:, 1]).L_g
    @test draws.b_g[2] ≈ constrain(layout, U[:, 2]).b_g
    # Empty draws keep the keys, with empty matrix vectors.
    none = restore_draws(layout, Matrix{Float64}(undef, layout.total, 0))
    @test Tuple(keys(none)) == Tuple(keys(draws))
    @test isempty(none.L_g) && isempty(none.b_g)
end

# Swap declared levels onto one draws block of a lowered (unbound) plan.
function _tv_with_levels(plan::StructuralPlan, levels::Vector, which::Int = 1)
    d = plan.varying_draws[which]
    plan.varying_draws[which] = VaryingDraws(d.group, d.kind,
        d.margins, d.lkj_eta, d.label, d.suffix, levels)
    return plan
end

@testset "declared codes helper" begin
    dc = ReactiveKernelsPPL._declared_codes
    # Declared order, never sorted: levels [3, 1, 2].
    @test dc([1, 3, 2, 1], [3, 1, 2]) == [2, 1, 3, 2]
    # Strings in declared order.
    @test dc(["b", "a", "c", "b"], ["c", "b", "a"]) == [2, 3, 1, 2]
    # Single level; repeated values.
    @test dc([7, 7], [7]) == [1, 1]
    # Unobserved declared levels keep their positions.
    @test dc(["a", "c"], ["c", "b", "a"]) == [3, 1]
    # Symbols.
    @test dc([:x, :y], [:y, :x]) == [2, 1]
end

@testset "varying declared levels structure validation" begin
    b = _tv_draws()
    mklevels(lv) = VaryingDraws(b.group, b.kind, b.margins,
        b.lkj_eta, b.label, b.suffix, lv)
    # Empty levels fail.
    bad = _tv_base(; with_effect = true)
    bad.varying_draws[1] = mklevels([])
    @test_throws ContractValidationError validate_structure(bad)
    # Duplicate levels fail.
    dup = _tv_base(; with_effect = true)
    dup.varying_draws[1] = mklevels([1, 2, 1])
    @test_throws ContractValidationError validate_structure(dup)
    # Non-literal-embeddable levels fail.
    nonemb = _tv_base(; with_effect = true)
    nonemb.varying_draws[1] = mklevels([1, missing])
    @test_throws ContractValidationError validate_structure(nonemb)
    # Valid declared levels (order ≠ sorted) pass.
    ok = _tv_base(; with_effect = true)
    ok.varying_draws[1] = mklevels(["b", "a"])
    @test validate_structure(ok) === nothing
end

@testset "varying declared levels bind" begin
    cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5],
        :g => [2, 1, 3, 1])
    mkplan() = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [1])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :g))
    # `nothing` fills sort-ordered observed levels (SB numbering for
    # plain vectors).
    bound = bind_data(mkplan(), cols)
    @test only(bound.varying_draws).levels == [1, 2, 3]
    # Provided levels pass through; G counts unobserved declared levels.
    bound2 = bind_data(_tv_with_levels(mkplan(), [3, 1, 2, 4]), cols)
    @test only(bound2.varying_draws).levels == [3, 1, 2, 4]
    @test assign_layout(bound2).total == 6 # a + log_scale + 4 xi cells
    # Observed-but-undeclared values fail closed (they would encode 0).
    bad = _tv_with_levels(mkplan(), [1, 2])
    @test_throws ContractValidationError bind_data(bad, cols)
    # Hand-built bound plans with `levels === nothing` fail loud.
    hand = bind_data(mkplan(), cols)
    hb = only(hand.varying_draws)
    hand.varying_draws[1] = VaryingDraws(hb.group, hb.kind, hb.margins,
        hb.lkj_eta, hb.label, hb.suffix, nothing)
    @test_throws ContractValidationError validate_data(hand)
    # Same-group draws must agree on levels (order included).
    two = lower_rkppl(quote
            a ~ Normal(0, 1)
            dA ~ varying_draws(g, [1])
            rA ~ varying_slice(dA, 1)
            dB ~ varying_draws(g, [1])
            rB ~ varying_slice(dB, 1)
            mu = a .+ rA .+ rB
            y .~ Normal.(mu, 1.5)
        end, (:y, :g))
    agree = bind_data(_tv_with_levels(_tv_with_levels(two, [2, 1, 3], 1),
        [2, 1, 3], 2), cols)
    @test agree.varying_draws[1].levels == agree.varying_draws[2].levels
    disagree = lower_rkppl(quote
            a ~ Normal(0, 1)
            dA ~ varying_draws(g, [1])
            rA ~ varying_slice(dA, 1)
            dB ~ varying_draws(g, [1])
            rB ~ varying_slice(dB, 1)
            mu = a .+ rA .+ rB
            y .~ Normal.(mu, 1.5)
        end, (:y, :g))
    _tv_with_levels(_tv_with_levels(disagree, [2, 1, 3], 1), [1, 2, 3], 2)
    @test_throws ContractValidationError bind_data(disagree, cols)
end

# Surface-level draws with a spliced `levels` value (`QuoteNode` mirrors
# how `[:c]` parses; bare Symbols stay bare names).
_tv_levels_lit(v::Symbol) = QuoteNode(v)
_tv_levels_lit(v) = v
function _tv_levels_plan(lv)
    call = Expr(:call, :varying_effect,
        Expr(:parameters, Expr(:kw, :levels, lv)), :g, Expr(:vect, 1))
    return lower_rkppl(Expr(:block,
            Expr(:call, :~, :r, call),
            :(mu = a .+ r), :(y .~ Normal.(mu, 1.5))),
        (:y, :g))
end
_tv_levels_vals(vals::Vector) =
    _tv_levels_plan(Expr(:vect, (_tv_levels_lit(v) for v in vals)...))

@testset "varying levels surface spelling" begin
    # Declared order lands verbatim (never sorted); every
    # literal-embeddable element shape rides.
    for vals in (["c", "a", "b", "d"], [3, 1, 2, 4], [:c, :a],
            ['c', 'a'], [true, false], [1.5, 2.5], Any[1, "a"])
        @test only(_tv_levels_vals(vals).varying_draws).levels == vals
    end
    # Eta combo; omitted levels stay `nothing` (bind derives).
    combo = lower_rkppl(quote
            r ~ varying_effect(g, [1]; eta = 2.0, levels = [:c, :a])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :g))
    b = only(combo.varying_draws)
    @test b.levels == [:c, :a] && b.lkj_eta == 2.0
    bare = lower_rkppl(quote
            r ~ varying_effect(g, [1])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :g))
    @test only(bare.varying_draws).levels === nothing
end

@testset "varying levels surface failures" begin
    # Not a literal vector: string / bare column / range.
    for lv in ("ab", :g, Expr(:call, :(:), 1, 3))
        @test_throws SurfaceLoweringError _tv_levels_plan(lv)
    end
    # Empty / duplicates.
    @test_throws SurfaceLoweringError _tv_levels_plan(Expr(:vect))
    @test_throws SurfaceLoweringError _tv_levels_plan(Expr(:vect, "a", "a"))
    # Bare names are not level values (one or all).
    @test_throws SurfaceLoweringError _tv_levels_plan(Expr(:vect, :a, :b))
    @test_throws SurfaceLoweringError _tv_levels_plan(Expr(:vect, 1, "a", :b))
    # Non-literal elements: nested vector / call / non-embeddable value.
    @test_throws SurfaceLoweringError _tv_levels_plan(Expr(:vect, 1, Expr(:vect, 2)))
    @test_throws SurfaceLoweringError _tv_levels_plan(Expr(:vect, 1, Expr(:call, :f, 2)))
    @test_throws SurfaceLoweringError _tv_levels_plan(Expr(:vect, 1, missing))
    # Unknown keyword (the reworded two-keyword gate).
    @test_throws SurfaceLoweringError lower_rkppl(quote
            r ~ varying_effect(g, [1]; foo = 1)
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :g))
    # The two guidance messages are pinned: literal-vector requirement and
    # the bare-name quote-it fix.
    err = try
        _tv_levels_plan("ab")
        nothing
    catch e
        e
    end
    @test err isa SurfaceLoweringError &&
        occursin("takes a literal level vector", sprint(showerror, err))
    err = try
        _tv_levels_plan(Expr(:vect, :a))
        nothing
    catch e
        e
    end
    @test err isa SurfaceLoweringError &&
        occursin("is a bare name", sprint(showerror, err)) &&
        occursin("`:a`", sprint(showerror, err))
end

# Declared-order K=1 intercept reference: SB `exp(log_scale) * xi[idx]`
# with `idx` in DECLARED position order (never sorted) and `xi` sized
# by the declared count (unobserved levels are prior-only).
function _tv_ref_declared_intercept(bound, nt, levels)
    idx = [findfirst(==(v), levels) for v in bound.columns[:g]]
    r = exp(nt.log_scale_g) .* nt.xi_g[idx]
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), bound.columns[:y]))
    pr = logpdf(Normal(0, 5), nt.mu[1]) + logpdf(Exponential(1), nt.sigma) +
        logpdf(Normal(0, 1), nt.log_scale_g) +
        sum(logpdf.(Normal(0, 1), nt.xi_g))
    return (; ll, pr)
end

@testset "varying declared-order intercept e2e values and gradient" begin
    _, _, y = _tv_k1_cols()
    g = ["a", "c", "b", "a", "c", "b", "a", "c"]
    levels = ["c", "a", "b", "d"] # declared ≠ sorted; "d" unobserved
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [1])
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :g))
    bound = bind_data(_tv_with_levels(plan, levels),
        Dict{Symbol,AbstractVector}(:g => g, :y => y))
    built = build_kernel(bound)
    # a + sigma + log_scale + 4 xi cells (d is prior-only).
    @test built.layout.total == 7
    u = [0.2, 0.1, -0.3, 0.4, 0.0, -0.1, 0.25]
    nt = constrain(built.layout, u)
    ref = _tv_ref_declared_intercept(bound, nt, levels)
    @test _query(built.spec, bound, :likelihood, u) ≈ ref.ll
    @test _query(built.spec, bound, :prior, u) ≈ ref.pr
    @test _query(built.spec, bound, :posterior, u) ≈ ref.ll + ref.pr + u[2]
    _check_gradient(built.spec, bound, u)
end

@testset "varying declared-order correlated e2e values and gradient" begin
    gv = ["a", "c", "b", "c", "a", "b"]
    xv = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    yv = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    levels = ["c", "b", "a", "d"] # declared ≠ sorted; "d" unobserved
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [1, x])
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :g))
    bound = bind_data(_tv_with_levels(plan, levels),
        Dict{Symbol,AbstractVector}(:g => gv, :x => xv, :y => yv))
    built = build_kernel(bound)
    # coef + sigma + 1 theta + 2 tau + 2*4 z cells.
    @test built.layout.total == 13
    u = collect(range(-0.4, 0.4; length = built.layout.total))
    nt = constrain(built.layout, u)
    r = _tv_ref_corr_r(bound, :g, nt.L_g, nt.tau_g, nt.z_flat_g,
        [ones(6), xv], 1:2; levels)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), yv))
    pr = logpdf(Normal(0, 5), nt.mu[1]) +
        logpdf(Exponential(1), nt.sigma) +
        _tv_ref_lkj_k2(nt.L_g, 1.0) +
        sum(logpdf.(Normal(0, 1), nt.tau_g)) +
        sum(logpdf.(Normal(0, 1), nt.z_flat_g))
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    s = 1 / (1 + exp(-u[3]))
    jac = u[2] + u[4] + u[5] + log(sin(pi * s)) + log(pi) + log(s) +
        log1p(-s)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
    _check_gradient(built.spec, bound, u)
end

@testset "varying surface levels intercept e2e values and gradient" begin
    _, _, y = _tv_k1_cols()
    g = ["a", "c", "b", "a", "c", "b", "a", "c"]
    levels = ["c", "a", "b", "d"] # declared ≠ sorted; "d" unobserved
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [1]; levels = ["c", "a", "b", "d"])
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :g))
    # Surface text carries declared levels into the draws (the P2 channel).
    @test only(plan.varying_draws).levels == levels
    bound = bind_data(plan, Dict{Symbol,AbstractVector}(:g => g, :y => y))
    @test only(bound.varying_draws).levels == levels
    built = build_kernel(bound)
    # a + sigma + log_scale + 4 xi cells (d is prior-only).
    @test built.layout.total == 7
    u = [0.2, 0.1, -0.3, 0.4, 0.0, -0.1, 0.25]
    nt = constrain(built.layout, u)
    ref = _tv_ref_declared_intercept(bound, nt, levels)
    @test _query(built.spec, bound, :likelihood, u) ≈ ref.ll
    @test _query(built.spec, bound, :prior, u) ≈ ref.pr
    @test _query(built.spec, bound, :posterior, u) ≈ ref.ll + ref.pr + u[2]
    _check_gradient(built.spec, bound, u)
end

@testset "varying surface levels correlated bind" begin
    gv = ["a", "c", "b", "c", "a", "b"]
    xv = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    yv = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    levels = ["c", "b", "a", "d"] # declared ≠ sorted; "d" unobserved
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [1, x]; levels = ["c", "b", "a", "d"])
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :g))
    @test only(plan.varying_draws).levels == levels
    bound = bind_data(plan,
        Dict{Symbol,AbstractVector}(:g => gv, :x => xv, :y => yv))
    # Surface levels pass bind through; G counts the unobserved level.
    @test only(bound.varying_draws).levels == levels
    @test assign_layout(bound).total == 13 # coef + sigma + theta + 2 tau + 2×4 z
end

# Swap per-margin sd priors onto one draws block of a lowered (unbound) plan.
function _tv_with_sd(plan::StructuralPlan, sd::Vector{VaryingSdPrior},
        which::Int = 1)
    d = plan.varying_draws[which]
    plan.varying_draws[which] = VaryingDraws(d.group, d.kind,
        d.margins, d.lkj_eta, d.label, d.suffix, d.levels, sd)
    return plan
end

@testset "varying sd priors structure validation" begin
    # Default is empty (all-`:std_normal`); pre-sd-prior arities keep it.
    @test _tv_draws().sd_priors == VaryingSdPrior[]
    b = _tv_draws()
    @test VaryingDraws(b.group, b.kind, b.margins, b.lkj_eta, b.label,
        b.suffix).sd_priors == VaryingSdPrior[]
    @test VaryingDraws(b.group, b.kind, b.margins, b.lkj_eta, b.label,
        b.suffix, [1, 2]).sd_priors == VaryingSdPrior[]
    mksd(sd) = _tv_with_sd(_tv_base(; with_effect = true), sd)
    # Wrong length fails (K = 2 here).
    @test_throws ContractValidationError validate_structure(
        mksd([VaryingSdPrior(:exponential, 1 / 3)]))
    # Unknown family fails.
    @test_throws ContractValidationError validate_structure(mksd(
        [VaryingSdPrior(:gamma, 2.0), VaryingSdPrior(:std_normal, 1.0)]))
    # Non-finite params fail, even on `:std_normal`.
    @test_throws ContractValidationError validate_structure(mksd(
        [VaryingSdPrior(:std_normal, NaN), VaryingSdPrior(:std_normal, 1.0)]))
    # Non-positive scale/sd fail.
    @test_throws ContractValidationError validate_structure(mksd(
        [VaryingSdPrior(:exponential, 0.0), VaryingSdPrior(:std_normal, 1.0)]))
    @test_throws ContractValidationError validate_structure(mksd(
        [VaryingSdPrior(:normal, -1.0), VaryingSdPrior(:std_normal, 1.0)]))
    # A finite `:std_normal` param is ignored, not rejected.
    @test validate_structure(mksd([VaryingSdPrior(:std_normal, 2.0),
        VaryingSdPrior(:std_normal, 1.0)])) === nothing
    # Mixed per-margin priors pass on `:correlated`.
    @test validate_structure(mksd([VaryingSdPrior(:exponential, 1 / 3),
        VaryingSdPrior(:normal, 2.0)])) === nothing
    # K=1 kinds fail closed on explicit sd priors (SB has no K=1
    # override path — the message names the vacuous-eta route).
    for (m, want) in ((1, :intercept1), (:x, :slope1))
        k1 = lower_rkppl(quote
                a ~ Normal(0, 1)
                r ~ varying_effect(g, [$m])
                mu = a .+ r
                y .~ Normal.(mu, 1.5)
            end, (:y, :x, :g))
        d = only(k1.varying_draws)
        @test d.kind === want
        k1.varying_draws[1] = VaryingDraws(d.group, d.kind,
            d.margins, d.lkj_eta, d.label, d.suffix, d.levels,
            [VaryingSdPrior(:exponential, 1.0)])
        err = try
            validate_structure(k1)
            nothing
        catch e
            e
        end
        @test err isa ContractValidationError
        @test occursin("vacuous-1x1-LKJ", sprint(showerror, err))
    end
end

@testset "varying sd priors emission shape" begin
    cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
        :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0], :g => [1, 2, 1, 3, 2, 3])
    # The `tau` PRIOR statements only (the effect summand reads
    # `tau_g[j]` scalar refs on every path — the LKJ-sandwich shape).
    function _tv_tau_src(sd)
        plan = lower_rkppl(quote
                a ~ Normal(0, 5)
                sigma ~ Exponential(1)
                r ~ varying_effect(g, [1, x]; eta = 2.0)
                mu = a .+ r
                y .~ Normal.(mu, sigma)
            end, (:y, :x, :g))
        bound = bind_data(_tv_with_sd(plan, sd), cols)
        def = kernel_expr(bound, assign_layout(bound))
        strs = [repr(st) for st in def.args[2].args]
        taus = filter(s -> occursin("tau_g", s) && startswith(s, ":(_ppl_p"),
            strs)
        return join(taus, "\n")
    end
    # Default stays the historical plate (no scalar `tau_g[k]` refs).
    default_src = _tv_tau_src(VaryingSdPrior[])
    @test occursin("_ppl_pw_prior_tau_g", default_src)
    @test occursin("plate(tau_g)", default_src)
    @test occursin("(normal(0.0, 1.0)).logpdf(_ppl_c1)", default_src)
    @test !occursin("tau_g[", default_src)
    # Uniform configured prior stays one plate with the mapped family.
    uni_src = _tv_tau_src([VaryingSdPrior(:exponential, 1 / 3),
        VaryingSdPrior(:exponential, 1 / 3)])
    @test occursin("_ppl_pw_prior_tau_g", uni_src)
    @test occursin("plate(tau_g)", uni_src)
    @test occursin("(exponential(0.3333333333333333)).logpdf(_ppl_c1)",
        uni_src)
    # Mixed margins unroll to one scalar density per margin.
    mix_src = _tv_tau_src([VaryingSdPrior(:exponential, 1 / 3),
        VaryingSdPrior(:normal, 2.0)])
    @test occursin("(exponential(0.3333333333333333)).logpdf(tau_g[1])",
        mix_src)
    @test occursin("(normal(0.0, 2.0)).logpdf(tau_g[2])", mix_src)
    @test occursin("_ppl_prior_tau_g", mix_src)
end

@testset "varying sd priors survive bind" begin
    cols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
        :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0], :g => [1, 2, 1, 3, 2, 3])
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [1, x]; eta = 2.0)
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :g))
    sd = [VaryingSdPrior(:exponential, 1 / 3),
        VaryingSdPrior(:normal, 2.0)]
    bound = bind_data(_tv_with_sd(plan, sd), cols)
    # The levels-fill rebuild threads the config through.
    @test only(bound.varying_draws).levels == [1, 2, 3]
    got = only(bound.varying_draws).sd_priors
    @test [(p.family, p.param) for p in got] ==
        [(:exponential, 1 / 3), (:normal, 2.0)]
end

@testset "varying uniform exponential sd e2e values and gradient" begin
    gv = [1, 2, 1, 3, 2, 3]
    xv = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    yv = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    cols = Dict{Symbol,AbstractVector}(:g => gv, :x => xv, :y => yv)
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [1, x]; eta = 2.0)
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :g))
    # SB `sd ~ Exponential(0.3333)`: rate 3.0 inverts to scale θ = 1/3.
    sd = [VaryingSdPrior(:exponential, 1 / 3),
        VaryingSdPrior(:exponential, 1 / 3)]
    bound = bind_data(_tv_with_sd(plan, sd), cols)
    built = build_kernel(bound)
    # Layout unchanged by the prior: coef + sigma + theta + 2 tau + 6 z.
    @test built.layout.total == 11
    @test [e.transform for e in built.layout.entries] ==
        [:identity, :exp, :lkj, :exp, :identity]
    u = collect(range(-0.4, 0.4; length = built.layout.total))
    nt = constrain(built.layout, u)
    r = _tv_ref_corr_r(bound, :g, nt.L_g, nt.tau_g, nt.z_flat_g,
        [ones(6), xv], 1:2)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), yv))
    pr = logpdf(Normal(0, 5), nt.mu[1]) +
        logpdf(Exponential(1), nt.sigma) +
        _tv_ref_lkj_k2(nt.L_g, 2.0) +
        sum(logpdf.(Exponential(1 / 3), nt.tau_g)) +
        sum(logpdf.(Normal(0, 1), nt.z_flat_g))
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    # No truncation renormalizer: Stan lower-bound kernel semantics.
    @test _query(built.spec, bound, :prior, u) ≈ pr
    s = 1 / (1 + exp(-u[3]))
    jac = u[2] + u[4] + u[5] + log(sin(pi * s)) + log(pi) + log(s) +
        log1p(-s)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
    _check_gradient(built.spec, bound, u)
end

@testset "varying mixed sd priors e2e values and gradient" begin
    gv = [1, 2, 1, 3, 2, 3]
    xv = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
    yv = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
    cols = Dict{Symbol,AbstractVector}(:g => gv, :x => xv, :y => yv)
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [1, x]; eta = 2.0)
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :g))
    sd = [VaryingSdPrior(:exponential, 0.5),
        VaryingSdPrior(:normal, 2.0)]
    bound = bind_data(_tv_with_sd(plan, sd), cols)
    built = build_kernel(bound)
    @test built.layout.total == 11
    u = collect(range(-0.4, 0.4; length = built.layout.total))
    nt = constrain(built.layout, u)
    r = _tv_ref_corr_r(bound, :g, nt.L_g, nt.tau_g, nt.z_flat_g,
        [ones(6), xv], 1:2)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), yv))
    pr = logpdf(Normal(0, 5), nt.mu[1]) +
        logpdf(Exponential(1), nt.sigma) +
        _tv_ref_lkj_k2(nt.L_g, 2.0) +
        logpdf(Exponential(0.5), nt.tau_g[1]) +
        logpdf(Normal(0, 2.0), nt.tau_g[2]) +
        sum(logpdf.(Normal(0, 1), nt.z_flat_g))
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    s = 1 / (1 + exp(-u[3]))
    jac = u[2] + u[4] + u[5] + log(sin(pi * s)) + log(pi) + log(s) +
        log1p(-s)
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
    _check_gradient(built.spec, bound, u)
end

@testset "varying K=1 correlated sd prior e2e values and gradient" begin
    gv = [1, 2, 1, 2]
    xv = [0.5, -1.0, 1.5, 0.0]
    yv = [1.0, 2.0, 1.5, 2.5]
    cols = Dict{Symbol,AbstractVector}(:g => gv, :x => xv, :y => yv)
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            r ~ varying_effect(g, [x]; eta = 1.0)
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :g))
    # The vacuous-1x1-LKJ route takes sd priors like any `:correlated`.
    bound = bind_data(
        _tv_with_sd(plan, [VaryingSdPrior(:exponential, 1.0)]), cols)
    built = build_kernel(bound)
    @test built.layout.total == 5
    u = [0.2, 0.1, -0.3, 0.4, -0.1]
    nt = constrain(built.layout, u)
    idx = [findfirst(==(v), [1, 2]) for v in gv]
    r = nt.tau_g[1] .* (nt.z_flat_g[idx] .* xv)
    ll = sum(logpdf.(Normal.(nt.mu[1] .+ r, nt.sigma), yv))
    pr = logpdf(Normal(0, 5), nt.mu[1]) +
        logpdf(Exponential(1), nt.sigma) +
        logpdf(Exponential(1.0), nt.tau_g[1]) +
        sum(logpdf.(Normal(0, 1), nt.z_flat_g))
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + u[2] + u[3]
    _check_gradient(built.spec, bound, u)
end
