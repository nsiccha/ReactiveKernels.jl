using Distributions
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# SB-mirroring `me(x, sd)` measurement-error latent-predictor support:
# a length-n_obs plate vector with shared scalar Normal args, usable as
# (a) a ContinuousTerm design column with a free coefficient and (b) a
# likelihood mean (`x_obs ~ Normal(x_true, sd)`, scalar constant sd).
# Oracles are independent per-row Distributions.jl loops, never the
# emitted plate forms; gradients cross-check Enzyme against central
# differences (the test_generator.jl idiom).

function _me_columns()
    cols = Dict{Symbol,AbstractVector}(
        :y => [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
        :x_obs => [0.4, -0.9, 1.4, 0.1, -0.4, 1.1],
    )
    return cols, 6
end

# Canonical me surface program: `mu = a .+ b .* x_true` over the latent
# plus the latent-mean observation of `x_obs`.
function _me_ast(; loc = 0.5, scale = 1.5, sd = 0.5)
    return Expr(:block,
        :(a ~ Normal(0, 1)),
        :(b ~ Normal(0, 2)),
        :(sigma ~ Exponential(1)),
        :(mu = a .+ b .* x_true),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(5),
            Expr(:for, Expr(:(=), :i, :(eachindex(x_obs))),
                Expr(:block, :(x_true[i] ~ Normal($loc, $scale))))),
        :(y .~ Normal.(mu, sigma)),
        :(x_obs .~ Normal.(x_true, $sd)))
end

# Independent me oracle: main likelihood + observation likelihood +
# latent prior + coefficient priors + sigma prior (no Jacobian — the
# posterior test adds it, mirroring the gaussian value test).
function _me_oracle(cols, a, b, sigma, x_true; loc = 0.5, scale = 1.5,
        sd = 0.5)
    mu = a .+ b .* x_true
    ll_main = sum(logpdf.(Normal.(mu, sigma), cols[:y]))
    ll_obs = sum(logpdf.(Normal.(x_true, sd), cols[:x_obs]))
    pr_lat = sum(logpdf.(Normal(loc, scale), x_true))
    pr_coef = logpdf(Normal(0, 1), a) + logpdf(Normal(0, 2), b)
    pr_sig = logpdf(Exponential(1), sigma)
    return (; ll_main, ll_obs, pr_lat, pr_coef, pr_sig)
end

@testset "surface me lowering: LP use plus observation" begin
    plan = lower_rkppl(_me_ast(), (:y, :x_obs))
    @test [p.name for p in plan.predictors] == [:mu, :x_obs_loc]
    mu = only(p for p in plan.predictors if p.name === :mu)
    @test [t.kind for t in mu.terms] == [InterceptTerm, ContinuousTerm]
    @test mu.terms[2].columns == [:x_true]
    @test mu.terms[2].addressee === :x_true
    # The free coefficient keeps its stated prior, addressed at the latent.
    priors = Dict((p.predictor, p.addressee) => p for p in plan.population_priors)
    @test priors[(:mu, :Intercept)].location == 0.0
    @test priors[(:mu, :x_true)].location == 0.0
    @test priors[(:mu, :x_true)].scale == 2.0
    # loc/scale ride the plate's shared scalar args; the observation keeps
    # the established LatentTerm location path (identical math).
    pp = only(plan.plate_parameters)
    @test pp.name === :x_true && pp.family === :normal
    @test pp.args == (arg1 = 0.5, arg2 = 1.5)
    loc = only(p for p in plan.predictors if p.name === :x_obs_loc)
    @test only(loc.terms).kind === LatentTerm
    @test only(loc.terms).columns == [:x_true]
    @test [r.predictor for r in plan.responses] == [:mu, :x_obs_loc]
end

@testset "surface me lowering: default coefficient prior" begin
    ast = Expr(:block,
        :(mu = a .+ b .* x_true),
        :(a ~ Normal(0, 1)),
        :(sigma ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(4),
            Expr(:for, Expr(:(=), :i, :(eachindex(x_obs))),
                Expr(:block, :(x_true[i] ~ Normal(0, 1))))),
        :(y .~ Normal.(mu, sigma)),
        :(x_obs .~ Normal.(x_true, 0.5)))
    plan = lower_rkppl(ast, (:y, :x_obs))
    priors = Dict((p.predictor, p.addressee) => p for p in plan.population_priors)
    @test priors[(:mu, :x_true)] == PopulationPrior(:mu, :x_true, 0.0, 1.0)
end

@testset "surface me lowering: bare latent stays fail-closed" begin
    Dn = (:y, :x_obs)
    plate = Expr(:macrocall, Symbol("@plate"), LineNumberNode(1),
        Expr(:for, Expr(:(=), :i, :(eachindex(x_obs))),
            Expr(:block, :(x_true[i] ~ Normal(0, 1)),
                :(x_obs[i] ~ Normal.(x_true[i], 0.5)))))
    # Bare latent in an inline location (no named derived to transform).
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        :(a ~ Normal(0, 1)), :(sigma ~ Exponential(1)), plate,
        :(y .~ Normal.(a .+ x_true, sigma))), Dn)
    # Two coefficients on one latent column.
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        :(a ~ Normal(0, 1)), :(b ~ Normal(0, 1)), :(c ~ Normal(0, 1)),
        :(sigma ~ Exponential(1)),
        :(mu = a .+ b .* x_true .+ c .* x_true), plate,
        :(y .~ Normal.(mu, sigma))), Dn)
    # A latent is not a factor index: the surface screens the inline
    # spelling, and the contract screens indexing in a named derived.
    @test_throws SurfaceLoweringError lower_rkppl(Expr(:block,
        :(a ~ Normal(0, 1)), :(sigma ~ Exponential(1)), plate,
        :(y .~ Normal.(a .+ c[x_true], sigma))), Dn)
    @test_throws ContractValidationError lower_rkppl(Expr(:block,
        :(a ~ Normal(0, 1)), :(sigma ~ Exponential(1)),
        :(mu = a .+ c[x_true]), plate,
        :(y .~ Normal.(mu, sigma))), Dn)
end

@testset "surface me lowering: unscaled latent stays a transform" begin
    # A named definition that reads a latent WITHOUT coefficient scaling
    # keeps the LatentTerm location path (the pre-me behavior): `a` stays
    # a scalar parameter, never a coefficient.
    ast = Expr(:block,
        :(a ~ Normal(0, 1)),
        :(sigma ~ Exponential(1)),
        :(mu = a .+ x_true),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(4),
            Expr(:for, Expr(:(=), :i, :(eachindex(x_obs))),
                Expr(:block, :(x_true[i] ~ Normal(0, 1))))),
        :(y .~ Normal.(mu, sigma)),
        :(x_obs .~ Normal.(x_true, 0.5)))
    plan = lower_rkppl(ast, (:y, :x_obs))
    yloc = only(p for p in plan.predictors if p.name === :y_loc)
    @test only(yloc.terms).kind === LatentTerm
    @test :a in [p.name for p in plan.parameters]
    @test isempty(plan.population_priors)
    # Undotted scaling keeps Julia scalar×vector semantics through the
    # transform path too (the dotted spelling takes the design path).
    ast2 = Expr(:block,
        :(a ~ Normal(0, 1)),
        :(b ~ Normal(0, 2)),
        :(sigma ~ Exponential(1)),
        :(mu = a .+ b * x_true),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(5),
            Expr(:for, Expr(:(=), :i, :(eachindex(x_obs))),
                Expr(:block, :(x_true[i] ~ Normal(0, 1))))),
        :(y .~ Normal.(mu, sigma)),
        :(x_obs .~ Normal.(x_true, 0.5)))
    plan2 = lower_rkppl(ast2, (:y, :x_obs))
    yloc2 = only(p for p in plan2.predictors if p.name === :y_loc)
    @test only(yloc2.terms).kind === LatentTerm
    @test :b in [p.name for p in plan2.parameters]
end

@testset "surface me lowering: data-plus-latent stays a transform" begin
    # No coefficient structure: `x .+ x_true` keeps the LatentTerm
    # location path (the pre-me composition is preserved).
    ast = Expr(:block,
        :(sigma ~ Exponential(1)),
        :(eta = x .+ x_true),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(3),
            Expr(:for, Expr(:(=), :i, :(eachindex(x_obs))),
                Expr(:block, :(x_true[i] ~ Normal(0, 1))))),
        :(y .~ Normal.(eta, sigma)),
        :(x_obs .~ Normal.(x_true, 0.5)))
    plan = lower_rkppl(ast, (:y, :x, :x_obs))
    yloc = only(p for p in plan.predictors if p.name === :y_loc)
    @test only(yloc.terms).kind === LatentTerm
    @test :eta in [d.name for d in plan.derived]
end

@testset "surface me lowering: scalar-param read stays a transform" begin
    # A non-coefficient sampled read (Exponential `tau`) marks a latent
    # transform even with coefficient structure elsewhere.
    ast = Expr(:block,
        :(mu ~ Normal(0, 5)), :(tau ~ Exponential(1)),
        :(sigma ~ Exponential(1)),
        :(eta = mu .+ tau .* x_true),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(4),
            Expr(:for, Expr(:(=), :i, :(eachindex(x_obs))),
                Expr(:block, :(x_true[i] ~ Normal(0, 1))))),
        :(y .~ Normal.(eta, sigma)),
        :(x_obs .~ Normal.(x_true, 0.5)))
    plan = lower_rkppl(ast, (:y, :x_obs))
    yloc = only(p for p in plan.predictors if p.name === :y_loc)
    @test only(yloc.terms).kind === LatentTerm
end

function _me_bound_plan()
    cols, n = _me_columns()
    return bind_data(lower_rkppl(_me_ast(), (:y, :x_obs)), cols)
end

@testset "contract me: ContinuousTerm over a plate" begin
    plan = _me_bound_plan()
    @test validate_plan(plan) === nothing
    shape = design_shape(plan.predictors[1], plan.columns;
        levelmaps = plan.levelmaps)
    @test shape.width == 2
    @test shape.blocks[2].kind === ContinuousTerm
    @test shape.blocks[2].column === :x_true
    # The design recipe materializes the latent view so the hcat stays
    # homogeneous (Enzyme), while data columns stay bare.
    recs = preprocessing_recipes(plan)
    drec = only(r for r in recs if r.args[1] === design_name(:mu))
    @test drec == :(_ppl_design_mu =
        Float64.(hcat(ones(6), Float64.(x_true))))
end

@testset "contract me: non-continuous terms over a plate fail" begin
    cols, n = _me_columns()
    mkplan(terms) = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu,
            :sigma, nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink, terms, :mu)],
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :x_true, 0.0, 2.0)],
        SampledParameter[SampledParameter(:sigma, :exponential,
            (arg1 = 1.0,), nothing, :sigma)],
        AssignmentSpec[], cols, n;
        plate_parameters = PlateParameter[PlateParameter(:x_true, :normal,
            (arg1 = 0.0, arg2 = 1.0), nothing)])
    good = mkplan(TermSpec[
        TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
            :intercept),
        TermSpec(ContinuousTerm, [:x_true], NamedTuple(), :x_true,
            :x_true_term)])
    @test validate_plan(good) === nothing
    # Offset over a latent: the latent takes a free coefficient, never a
    # bare add.
    off = mkplan(TermSpec[
        TermSpec(InterceptTerm, ColumnRef[], NamedTuple(), :Intercept,
            :intercept),
        TermSpec(OffsetTerm, [:x_true], NamedTuple(), :x_true, :x_true_off)])
    @test_throws ContractValidationError validate_plan(off)
    # Factor over a latent: latents carry no levels. (The LevelMap gets
    # the plan past structure validation so the term check fires.)
    fac = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu,
            :sigma, nothing, _none_evidence(), :y_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(FactorTerm, [:x_true], NamedTuple(), :x_true,
                :x_true_fac)], :mu)],
        PopulationPrior[PopulationPrior(:mu, :x_true, 0.0, 2.0)],
        SampledParameter[SampledParameter(:sigma, :exponential,
            (arg1 = 1.0,), nothing, :sigma)],
        AssignmentSpec[], cols, n;
        levelmaps = LevelMap[LevelMap(:mu, :x_true, [1, 2], :levels, :)],
        plate_parameters = PlateParameter[PlateParameter(:x_true, :normal,
            (arg1 = 0.0, arg2 = 1.0), nothing)])
    @test_throws ContractValidationError validate_plan(fac)
    # The free coefficient needs its population prior.
    noprior = StructuralPlan(good.responses, good.predictors,
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0)],
        good.parameters, good.assignments, cols, n;
        roles = good.roles, levelmaps = good.levelmaps,
        plate_parameters = good.plate_parameters)
    @test_throws ContractValidationError validate_plan(noprior)
end

# Hand-built direct plate-mean plan: the observation names the plate in
# `predictor` (the scan-state precedent), no LatentTerm wrapper.
function _me_direct_plan(cols, n; sd = 0.5)
    return StructuralPlan(
        LikelihoodSpec[
            LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, :sigma,
                nothing, _none_evidence(), :y_resp),
            LikelihoodSpec(GaussianFam, IdentityLink, :x_obs, :x_true, sd,
                nothing, _none_evidence(), :x_obs_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                    :Intercept, :intercept),
                TermSpec(ContinuousTerm, [:x_true], NamedTuple(), :x_true,
                    :x_true_term)], :mu)],
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu, :x_true, 0.0, 2.0)],
        SampledParameter[SampledParameter(:sigma, :exponential,
            (arg1 = 1.0,), nothing, :sigma)],
        AssignmentSpec[], cols, n;
        plate_parameters = PlateParameter[PlateParameter(:x_true, :normal,
            (arg1 = 0.5, arg2 = 1.5), nothing)])
end

@testset "contract me: direct plate-mean observation" begin
    cols, n = _me_columns()
    @test validate_plan(_me_direct_plan(cols, n)) === nothing
    # A non-literal sd fails closed (SB: sd is a positive constant).
    @test_throws ContractValidationError validate_plan(
        _me_direct_plan(cols, n; sd = :sd_p))
    # Non-Gaussian plate-mean locations fail closed.
    badfam = StructuralPlan(
        LikelihoodSpec[LikelihoodSpec(BernoulliLogitFam, LogitLink, :yb,
            :x_true, nothing, nothing, _none_evidence(), :yb_resp)],
        PredictorSpec[PredictorSpec(:mu, IdentityLink,
            TermSpec[TermSpec(InterceptTerm, ColumnRef[], NamedTuple(),
                :Intercept, :intercept)], :mu)],
        PopulationPrior[PopulationPrior(:mu, :Intercept, 0.0, 1.0)],
        SampledParameter[], AssignmentSpec[],
        Dict{Symbol,AbstractVector}(:yb => [true, false, true, false,
            true, false]),
        n; plate_parameters = PlateParameter[PlateParameter(:x_true,
            :normal, (arg1 = 0.0, arg2 = 1.0), nothing)])
    @test_throws ContractValidationError validate_structure(badfam)
    # Weights, evidence, and ranges fail closed on a plate-mean response.
    wplan = _me_direct_plan(cols, n)
    wresp = wplan.responses[2]
    wcols = Dict{Symbol,AbstractVector}(:y => cols[:y],
        :x_obs => cols[:x_obs], :w => ones(6))
    wbad = StructuralPlan(
        LikelihoodSpec[wplan.responses[1], LikelihoodSpec(wresp.family,
            wresp.link, wresp.response, wresp.predictor, wresp.scale, :w,
            wresp.evidence, wresp.label, wresp.trials, wresp.range)],
        wplan.predictors, wplan.population_priors, wplan.parameters,
        wplan.assignments, wcols, n;
        plate_parameters = wplan.plate_parameters)
    @test_throws ContractValidationError validate_plan(wbad)
    ev = ResponseEvidence(:truncated, 0.0, nothing)
    evbad = StructuralPlan(
        LikelihoodSpec[wplan.responses[1], LikelihoodSpec(wresp.family,
            wresp.link, wresp.response, wresp.predictor, wresp.scale,
            wresp.weights, ev, wresp.label, wresp.trials, wresp.range)],
        wplan.predictors, wplan.population_priors, wplan.parameters,
        wplan.assignments, cols, n;
        plate_parameters = wplan.plate_parameters)
    @test_throws ContractValidationError validate_plan(evbad)
    rbad = StructuralPlan(
        LikelihoodSpec[wplan.responses[1], LikelihoodSpec(wresp.family,
            wresp.link, wresp.response, wresp.predictor, wresp.scale,
            wresp.weights, wresp.evidence, wresp.label, wresp.trials,
            1:n)],
        wplan.predictors, wplan.population_priors, wplan.parameters,
        wplan.assignments, cols, n;
        plate_parameters = wplan.plate_parameters)
    @test_throws ContractValidationError validate_plan(rbad)
end

@testset "generator me: surface-model values and gradient" begin
    plan = _me_bound_plan()
    built = build_kernel(plan)
    @test built.layout.total == 9
    u = [0.1, -0.2, 0.3, 0.5, -0.4, 0.2, -0.1, 0.0, 0.15]
    nt = constrain(built.layout, u)
    a, b = nt.mu[1], nt.mu[2]
    ref = _me_oracle(plan.columns, a, b, nt.sigma, Vector(nt.x_true))
    @test abs(_query(built.spec, plan, :likelihood, u) -
        (ref.ll_main + ref.ll_obs)) < 1e-12
    @test abs(_query(built.spec, plan, :prior, u) -
        (ref.pr_lat + ref.pr_coef + ref.pr_sig)) < 1e-12
    # Posterior adds the sigma exp-Jacobian (u[3] is log-sigma).
    @test abs(_query(built.spec, plan, :posterior, u) -
        (ref.ll_main + ref.ll_obs + ref.pr_lat + ref.pr_coef + ref.pr_sig +
            u[3])) < 1e-12
    _check_gradient(built.spec, plan, u)
end

@testset "generator me: direct plate-mean matches the surface path" begin
    cols, n = _me_columns()
    surf = _me_bound_plan()
    direct = _me_direct_plan(cols, n)
    @test validate_plan(direct) === nothing
    bs, bd = build_kernel(surf), build_kernel(direct)
    # Same packed length (the surface path's LatentTerm predictor owns a
    # width-0 coefficient block): identical packing, identical posterior.
    @test bs.layout.total == bd.layout.total == 9
    u = [0.1, -0.2, 0.3, 0.5, -0.4, 0.2, -0.1, 0.0, 0.15]
    @test abs(_query(bs.spec, surf, :posterior, u) -
        _query(bd.spec, direct, :posterior, u)) < 1e-12
    @test abs(_query(bd.spec, direct, :likelihood, u) -
        _query(bs.spec, surf, :likelihood, u)) < 1e-12
    @test abs(_query(bd.spec, direct, :prior, u) -
        _query(bs.spec, surf, :prior, u)) < 1e-12
    _check_gradient(bd.spec, direct, u)
end

@testset "generator me: hierarchical shared-scalar args" begin
    # loc/scale override rides scalar parameter refs (the SB
    # `latent(...) ~ Normal(...)` vehicle, resolved emitter-side).
    ast = Expr(:block,
        :(a ~ Normal(0, 1)),
        :(b ~ Normal(0, 2)),
        :(sigma ~ Exponential(1)),
        :(xloc ~ Normal(0, 1)),
        :(xsca ~ Exponential(1)),
        :(mu = a .+ b .* x_true),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(7),
            Expr(:for, Expr(:(=), :i, :(eachindex(x_obs))),
                Expr(:block, :(x_true[i] ~ Normal(xloc, xsca))))),
        :(y .~ Normal.(mu, sigma)),
        :(x_obs .~ Normal.(x_true, 0.5)))
    cols, n = _me_columns()
    plan = bind_data(lower_rkppl(ast, (:y, :x_obs)), cols)
    @test only(plan.plate_parameters).args == (arg1 = :xloc, arg2 = :xsca)
    built = build_kernel(plan)
    u = [0.1, -0.2, 0.3, -0.15, 0.25, 0.5, -0.4, 0.2, -0.1, 0.0, 0.15]
    @test length(u) == built.layout.total
    nt = constrain(built.layout, u)
    ref = _me_oracle(plan.columns, nt.mu[1], nt.mu[2], nt.sigma,
        Vector(nt.x_true); loc = nt.xloc, scale = nt.xsca)
    pr_extra = logpdf(Normal(0, 1), nt.xloc) + logpdf(Exponential(1), nt.xsca)
    @test _query(built.spec, plan, :likelihood, u) ≈
        ref.ll_main + ref.ll_obs
    @test _query(built.spec, plan, :prior, u) ≈
        ref.pr_lat + ref.pr_coef + ref.pr_sig + pr_extra
    _check_gradient(built.spec, plan, u)
end
