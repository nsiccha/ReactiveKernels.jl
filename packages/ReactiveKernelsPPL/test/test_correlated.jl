# Correlated outcomes: joint MvNormalCholesky responses over an LKJ
# covariance factor (`L ~ LKJCovarianceFactor`, `[y1..yK] ~ MvNormalCholesky`).
# Native axes only (the exact-GP precedent): Reactant primal of this emission
# shape is verified in scratch, and the Reactant-compiled gradient axis is
# upstream-blocked (no `stablehlo.triangular_solve` adjoint — recorded in
# the generator, not pinned here).

using LinearAlgebra: Diagonal
using Distributions: MvNormal, Normal, Exponential, logpdf

_corr_data2() = Dict{Symbol,AbstractVector}(
    :y1 => [0.5, -0.2, 0.8, 0.1],
    :y2 => [0.1, 0.4, -0.3, 0.2],
    :x => [0.5, -1.0, 1.5, 0.0])

function _corr_plan2()
    return lower_rkppl(quote
            a1 ~ Normal(0, 1)
            b1 ~ Normal(0, 1)
            a2 ~ Normal(0, 1)
            b2 ~ Normal(0, 1)
            mu1 = a1 .+ b1 .* x
            mu2 = a2 .+ b2 .* x
            L ~ LKJCovarianceFactor(2, Exponential(1.0), 2.0)
            [y1, y2] ~ MvNormalCholesky([mu1, mu2], L)
        end, (:y1, :y2, :x))
end

function _corr_oracle2(nt, cols)
    s = Vector(nt.L_scales)
    Lc = Matrix(nt.L_L_corr)
    L = Diagonal(s) * Lc
    c1, c2 = Vector(nt.mu1), Vector(nt.mu2)
    m1 = c1[1] .+ c1[2] .* cols[:x]
    m2 = c2[1] .+ c2[2] .* cols[:x]
    Σ = L * L'
    ll = sum(logpdf(MvNormal([m1[i], m2[i]], Σ), [cols[:y1][i], cols[:y2][i]])
        for i in 1:length(cols[:x]))
    pr = sum(logpdf(Normal(0, 1), c) for c in [c1; c2]) +
        sum(logpdf(Exponential(1.0), si) for si in s) +
        lkj_corr_cholesky_logpdf(Lc, 2.0)
    return ll, pr
end

@testset "correlated factor lowering" begin
    plan = _corr_plan2()
    @test length(plan.responses) == 1
    r = only(plan.responses)
    @test r.family === MvNormalCholeskyFam
    @test r.link === IdentityLink
    @test r.response === :y1
    @test r.extra_responses == [:y2]
    @test r.predictor === :mu1
    @test r.extra_predictors == [:mu2]
    @test r.factor_scales === :L_scales
    @test r.factor_corr === :L_L_corr
    @test r.label === :y1_y2_resp
    @test length(plan.vector_parameters) == 2
    sc, cr = plan.vector_parameters
    @test (sc.name, sc.family, sc.args, sc.size) ==
        (:L_scales, :positive_exponential, (arg1 = 1.0,), 2)
    @test (cr.name, cr.family, cr.args, cr.size) ==
        (:L_L_corr, :cholesky_corr_lkj, (arg1 = 2.0,), 2)
    # A sampled scale hyperparameter rides through as a symbol.
    hier = lower_rkppl(quote
            a1 ~ Normal(0, 1)
            b1 ~ Normal(0, 1)
            a2 ~ Normal(0, 1)
            b2 ~ Normal(0, 1)
            tau ~ Exponential(0.5)
            mu1 = a1 .+ b1 .* x
            mu2 = a2 .+ b2 .* x
            L ~ LKJCovarianceFactor(2, Exponential(tau), 2.0)
            [y1, y2] ~ MvNormalCholesky([mu1, mu2], L)
        end, (:y1, :y2, :x))
    @test hier.vector_parameters[1].args == (arg1 = :tau,)
    # An assignment-valued scale survives absorption (param-arg edge).
    det = lower_rkppl(quote
            a1 ~ Normal(0, 1)
            b1 ~ Normal(0, 1)
            a2 ~ Normal(0, 1)
            b2 ~ Normal(0, 1)
            t0 ~ Normal(0, 1)
            tau = exp(t0)
            mu1 = a1 .+ b1 .* x
            mu2 = a2 .+ b2 .* x
            L ~ LKJCovarianceFactor(2, Exponential(tau), 2.0)
            [y1, y2] ~ MvNormalCholesky([mu1, mu2], L)
        end, (:y1, :y2, :x))
    @test any(a -> a.name === :tau, det.assignments)
    # K=1 lowers uniformly (single outcome, no tail).
    one = lower_rkppl(quote
            a1 ~ Normal(0, 1)
            b1 ~ Normal(0, 1)
            mu1 = a1 .+ b1 .* x
            L ~ LKJCovarianceFactor(1, Exponential(1.0), 2.0)
            [y1] ~ MvNormalCholesky([mu1], L)
        end, (:y1, :x))
    r1 = only(one.responses)
    @test r1.extra_responses == Symbol[] && r1.extra_predictors == Symbol[]
    @test [p.size for p in one.vector_parameters] == [1, 1]
end

@testset "correlated layout edges" begin
    bound = bind_data(_corr_plan2(), _corr_data2())
    layout = assign_layout(bound)
    @test layout.total == 7 # 2+2 coefs + 2 scales + 1 theta
    kinds = [(e.kind, e.name, e.size, e.transform) for e in layout.entries]
    @test kinds[3] == (:vector, :L_scales, 2, :exp)
    @test kinds[4] == (:cholesky_corr, :L_L_corr, 1, :lkj)
    u = [0.5, -0.25, 0.1, 0.2, 0.3, -0.1, 0.7]
    nt = constrain(layout, u)
    @test Vector(nt.L_scales) ≈ exp.([0.3, -0.1])
    @test Matrix(nt.L_L_corr) ≈ lkj_chol_constrain([0.7], 2)
    @test !haskey(nt, :b_corr) # no ranef derived draws leak
    @test unconstrain(layout, nt) ≈ u
    @test logjac(layout, u) ≈ (0.3 + -0.1) + lkj_chol_logjac([0.7], 2)
    @test length(coordinate_names(layout)) == 7
end

@testset "correlated build value and gradient" begin
    bound = bind_data(_corr_plan2(), _corr_data2())
    built = build_kernel(bound)
    u = [0.5, -0.25, 0.1, 0.2, 0.3, -0.1, 0.7]
    nt = constrain(built.layout, u)
    ll, pr = _corr_oracle2(nt, bound.columns)
    jac = logjac(built.layout, u)
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    @test _query(built.spec, bound, :log_jacobian, u) ≈ jac
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
    _check_gradient(built.spec, bound, u)
end

@testset "correlated K=1 degenerates to Normal" begin
    plan = lower_rkppl(quote
            a1 ~ Normal(0, 1)
            b1 ~ Normal(0, 1)
            mu1 = a1 .+ b1 .* x
            L ~ LKJCovarianceFactor(1, Exponential(1.0), 2.0)
            [y1] ~ MvNormalCholesky([mu1], L)
        end, (:y1, :x))
    cols = Dict{Symbol,AbstractVector}(
        :y1 => [0.5, -0.2, 0.8, 0.1], :x => [0.5, -1.0, 1.5, 0.0])
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    @test built.layout.total == 3 # 2 coefs + 1 scale + 0 thetas
    u = [0.5, -0.25, 0.3]
    nt = constrain(built.layout, u)
    s1 = only(Vector(nt.L_scales))
    c1 = Vector(nt.mu1)
    m1 = c1[1] .+ c1[2] .* cols[:x]
    ll = sum(logpdf(Normal(m1[i], s1), cols[:y1][i]) for i in 1:4)
    pr = sum(logpdf(Normal(0, 1), c) for c in c1) +
        logpdf(Exponential(1.0), s1) # LKJ K=1 term is 0.0
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + u[3]
    _check_gradient(built.spec, bound, u)
end

@testset "correlated K=3 eta=1 sampled scale" begin
    plan = lower_rkppl(quote
            a1 ~ Normal(0, 1)
            b1 ~ Normal(0, 1)
            a2 ~ Normal(0, 1)
            b2 ~ Normal(0, 1)
            a3 ~ Normal(0, 1)
            b3 ~ Normal(0, 1)
            tau ~ Exponential(0.5)
            mu1 = a1 .+ b1 .* x
            mu2 = a2 .+ b2 .* x
            mu3 = a3 .+ b3 .* x
            L ~ LKJCovarianceFactor(3, Exponential(tau), 1.0)
            [y1, y2, y3] ~ MvNormalCholesky([mu1, mu2, mu3], L)
        end, (:y1, :y2, :y3, :x))
    cols = Dict{Symbol,AbstractVector}(
        :y1 => [0.5, -0.2, 0.8, 0.1],
        :y2 => [0.1, 0.4, -0.3, 0.2],
        :y3 => [-0.4, 0.6, 0.0, -0.1],
        :x => [0.5, -1.0, 1.5, 0.0])
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    @test built.layout.total == 13 # 6 coefs + tau + 3 scales + 3 thetas
    u = [0.5, -0.25, 0.1, 0.2, -0.15, 0.05, 0.0, 0.3, -0.1, 0.2,
        0.7, -0.4, 0.1]
    nt = constrain(built.layout, u)
    tau = nt.tau
    s = Vector(nt.L_scales)
    Lc = Matrix(nt.L_L_corr)
    L = Diagonal(s) * Lc
    cs = [Vector(nt.mu1), Vector(nt.mu2), Vector(nt.mu3)]
    ms = [c[1] .+ c[2] .* cols[:x] for c in cs]
    ys = [cols[:y1], cols[:y2], cols[:y3]]
    Σ = L * L'
    ll = sum(logpdf(MvNormal([ms[1][i], ms[2][i], ms[3][i]], Σ),
            [ys[1][i], ys[2][i], ys[3][i]]) for i in 1:4)
    pr = sum(logpdf(Normal(0, 1), c) for c in vcat(cs...)) +
        logpdf(Exponential(0.5), tau) +
        sum(logpdf(Exponential(tau), si) for si in s) +
        lkj_corr_cholesky_logpdf(Lc, 1.0)
    jac = logjac(built.layout, u)
    @test _query(built.spec, bound, :likelihood, u) ≈ ll
    @test _query(built.spec, bound, :prior, u) ≈ pr
    @test _query(built.spec, bound, :posterior, u) ≈ ll + pr + jac
    _check_gradient(built.spec, bound, u)
end

# Plan-level admission: mutate the K=2 plan and pin the failure.
function _corr_mutate(; resp = nothing, vps = nothing)
    plan = _corr_plan2()
    rs = resp === nothing ? plan.responses : resp
    vs = vps === nothing ? plan.vector_parameters : vps
    return StructuralPlan(rs, plan.predictors, plan.population_priors,
        plan.parameters, plan.assignments, plan.columns, plan.n_obs;
        roles = plan.roles, vector_parameters = vs)
end
_corr_r() = only(_corr_plan2().responses)

@testset "correlated contract admission" begin
    base = _corr_r()
    # Link / predictor shape.
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        resp = [LikelihoodSpec(base.family, LogitLink, base.response,
            base.predictor, base.scale, base.weights, base.evidence,
            base.label, base.trials, base.range;
            extra_responses = base.extra_responses,
            extra_predictors = base.extra_predictors,
            factor_scales = base.factor_scales, factor_corr = base.factor_corr)]))
    no_tail = LikelihoodSpec(base.family, base.link, base.response,
        base.predictor, base.scale, base.weights, base.evidence, base.label,
        base.trials, base.range; extra_responses = base.extra_responses,
        extra_predictors = Symbol[], factor_scales = base.factor_scales,
        factor_corr = base.factor_corr)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        resp = [no_tail]))
    dup_out = LikelihoodSpec(base.family, base.link, base.response,
        base.predictor, base.scale, base.weights, base.evidence, base.label,
        base.trials, base.range; extra_responses = [:y1],
        extra_predictors = base.extra_predictors,
        factor_scales = base.factor_scales, factor_corr = base.factor_corr)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        resp = [dup_out]))
    dup_pred = LikelihoodSpec(base.family, base.link, base.response,
        base.predictor, base.scale, base.weights, base.evidence, base.label,
        base.trials, base.range; extra_responses = base.extra_responses,
        extra_predictors = [:mu1], factor_scales = base.factor_scales,
        factor_corr = base.factor_corr)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        resp = [dup_pred]))
    unknown_pred = LikelihoodSpec(base.family, base.link, base.response,
        base.predictor, base.scale, base.weights, base.evidence, base.label,
        base.trials, base.range; extra_responses = base.extra_responses,
        extra_predictors = [:nope], factor_scales = base.factor_scales,
        factor_corr = base.factor_corr)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        resp = [unknown_pred]))
    # A non-identity mean predictor fails (same rebuild, predictor swapped).
    plan = _corr_plan2()
    preds = PredictorSpec[p for p in plan.predictors]
    preds[2] = PredictorSpec(:mu2, LogitLink, preds[2].terms, preds[2].label)
    @test_throws ContractValidationError validate_structure(StructuralPlan(
        plan.responses, preds, plan.population_priors, plan.parameters,
        plan.assignments, plan.columns, plan.n_obs; roles = plan.roles,
        vector_parameters = plan.vector_parameters))
    # Factor linkage.
    for (sc, cr) in ((nothing, base.factor_corr),
            (base.factor_scales, nothing),
            (:nope, base.factor_corr),
            (base.factor_scales, :nope),
            (base.factor_corr, base.factor_scales))
        bad = LikelihoodSpec(base.family, base.link, base.response,
            base.predictor, base.scale, base.weights, base.evidence,
            base.label, base.trials, base.range;
            extra_responses = base.extra_responses,
            extra_predictors = base.extra_predictors, factor_scales = sc,
            factor_corr = cr)
        @test_throws ContractValidationError validate_structure(
            _corr_mutate(; resp = [bad]))
    end
    # Factor sizes disagree with the joint width.
    vs = VectorParameter[p for p in plan.vector_parameters]
    vs[1] = VectorParameter(vs[1].name, vs[1].family, vs[1].args, 3,
        vs[1].label)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        vps = vs))
    # An unlinked factor piece fails (linkage is exactly-once).
    vs2 = vcat(plan.vector_parameters,
        [VectorParameter(:stray, :positive_exponential, (arg1 = 1.0,), 2,
            :stray)])
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        vps = vs2))
    # A factor piece shared by two joint responses fails.
    r2 = LikelihoodSpec(base.family, base.link, :y3, :mu3, base.scale,
        base.weights, base.evidence, :y3_y4_resp, base.trials, base.range;
        extra_responses = [:y4], extra_predictors = [:mu4],
        factor_scales = base.factor_scales, factor_corr = base.factor_corr)
    preds34 = vcat(plan.predictors,
        [PredictorSpec(:mu3, IdentityLink, plan.predictors[1].terms, :mu3),
            PredictorSpec(:mu4, IdentityLink, plan.predictors[1].terms, :mu4)])
    priors34 = vcat(plan.population_priors,
        [PopulationPrior(:mu3, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu3, :x, 0.0, 1.0),
            PopulationPrior(:mu4, :Intercept, 0.0, 1.0),
            PopulationPrior(:mu4, :x, 0.0, 1.0)])
    @test_throws ContractValidationError validate_structure(StructuralPlan(
        [base, r2], preds34, priors34, plan.parameters, plan.assignments,
        Dict{Symbol,AbstractVector}(), 0;
        vector_parameters = plan.vector_parameters))
    # Joint-only fields on a Gaussian response fail.
    g = LikelihoodSpec(GaussianFam, IdentityLink, :y1, :mu1, :sigma, nothing,
        ResponseEvidence(:none, nothing, nothing), :y_resp, nothing, nothing;
        extra_responses = [:y2])
    gplan = StructuralPlan([g], [plan.predictors[1]],
        plan.population_priors[1:2],
        [SampledParameter(:sigma, :exponential, (arg1 = 1.0,), nothing,
            :sigma)],
        AssignmentSpec[], Dict{Symbol,AbstractVector}(), 0)
    @test_throws ContractValidationError validate_structure(gplan)
    # Non-joint extras on a joint response fail.
    for kw in (Dict(:n_levels => 2), Dict(:thresholds => :t),
            Dict(:count_columns => [:c2]))
        bad = LikelihoodSpec(base.family, base.link, base.response,
            base.predictor, base.scale, base.weights, base.evidence,
            base.label, base.trials, base.range;
            extra_responses = base.extra_responses,
            extra_predictors = base.extra_predictors,
            factor_scales = base.factor_scales, factor_corr = base.factor_corr,
            kw...)
        @test_throws ContractValidationError validate_structure(
            _corr_mutate(; resp = [bad]))
    end
    weights_bad = LikelihoodSpec(base.family, base.link, base.response,
        base.predictor, base.scale, :w, base.evidence, base.label,
        base.trials, base.range; extra_responses = base.extra_responses,
        extra_predictors = base.extra_predictors,
        factor_scales = base.factor_scales, factor_corr = base.factor_corr)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        resp = [weights_bad]))
    trials_bad = LikelihoodSpec(base.family, base.link, base.response,
        base.predictor, base.scale, base.weights, base.evidence, base.label,
        :n, base.range; extra_responses = base.extra_responses,
        extra_predictors = base.extra_predictors,
        factor_scales = base.factor_scales, factor_corr = base.factor_corr)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        resp = [trials_bad]))
    scale_bad = LikelihoodSpec(base.family, base.link, base.response,
        base.predictor, :sigma, base.weights, base.evidence, base.label,
        base.trials, base.range; extra_responses = base.extra_responses,
        extra_predictors = base.extra_predictors,
        factor_scales = base.factor_scales, factor_corr = base.factor_corr)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        resp = [scale_bad]))
    ev = LikelihoodSpec(base.family, base.link, base.response, base.predictor,
        base.scale, base.weights,
        ResponseEvidence(:censored, :lo, :hi), base.label, base.trials,
        base.range; extra_responses = base.extra_responses,
        extra_predictors = base.extra_predictors,
        factor_scales = base.factor_scales, factor_corr = base.factor_corr)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        resp = [ev]))
    # Factor args: bad scale, unknown scale name, bad shape, missing size.
    for bad_vp in (
            VectorParameter(:L_scales, :positive_exponential, (arg1 = -1.0,),
                2, :L),
            VectorParameter(:L_scales, :positive_exponential,
                (arg1 = :nope,), 2, :L),
            VectorParameter(:L_L_corr, :cholesky_corr_lkj, (arg1 = 0.0,), 2,
                :L),
            VectorParameter(:L_L_corr, :cholesky_corr_lkj, (arg1 = :eta,), 2,
                :L),
            VectorParameter(:L_scales, :positive_exponential, (arg1 = 1.0,),
                nothing, :L))
        vsbad = bad_vp.name === :L_scales ?
            VectorParameter[bad_vp, plan.vector_parameters[2]] :
            VectorParameter[plan.vector_parameters[1], bad_vp]
        @test_throws ContractValidationError validate_structure(_corr_mutate(;
            vps = vsbad))
    end
end

@testset "correlated data admission" begin
    plan = _corr_plan2()
    good = _corr_data2()
    # A missing tail outcome fails at bind.
    missing_tail = copy(good)
    delete!(missing_tail, :y2)
    @test_throws ContractValidationError bind_data(plan, missing_tail)
    # A non-numeric tail outcome fails.
    bad_tail = copy(good)
    bad_tail[:y2] = ["a", "b", "c", "d"]
    @test_throws ContractValidationError bind_data(plan, bad_tail)
    # A ragged tail outcome fails the uniform-n_obs rule.
    ragged = copy(good)
    ragged[:y2] = [0.1, 0.4]
    @test_throws ContractValidationError bind_data(plan, ragged)
end

# Shared two-mean block, spliced (flat) in front of each admission case.
_corr_mu() = quote
    a1 ~ Normal(0, 1)
    b1 ~ Normal(0, 1)
    a2 ~ Normal(0, 1)
    b2 ~ Normal(0, 1)
    mu1 = a1 .+ b1 .* x
    mu2 = a2 .+ b2 .* x
end
_corr_surface(extra::Expr...) =
    Expr(:block, _corr_mu().args..., extra...)

@testset "correlated surface admission" begin
    F2 = :(L ~ LKJCovarianceFactor(2, Exponential(1.0), 2.0))
    J2 = :([y1, y2] ~ MvNormalCholesky([mu1, mu2], L))
    # Joint form errors.
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(F2,
            :([y1, y2] .~ MvNormalCholesky([mu1, mu2], L))), (:y1, :y2, :x))
    @test_throws SurfaceLoweringError lower_rkppl(_corr_surface(J2),
        (:y1, :y2, :x))
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(F2, :([y1, y2] ~ MvNormalCholesky([mu1], L))),
        (:y1, :y2, :x))
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(F2, :([y1, y3] ~ MvNormalCholesky([mu1, mu2], L))),
        (:y1, :y2, :x))
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(F2, :([y1, y1] ~ MvNormalCholesky([mu1, mu2], L))),
        (:y1, :y2, :x))
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(F2, :([y1, y2] ~ Normal([mu1, mu2], L))), (:y1, :y2, :x))
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(F2, :(y1 .~ MvNormalCholesky.([mu1, mu2], L))),
        (:y1, :y2, :x))
    # Factor-statement errors.
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(:(L ~ LKJCovarianceFactor(0, Exponential(1.0), 2.0)),
            J2), (:y1, :y2, :x))
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(:(L ~ LKJCovarianceFactor(2, Gamma(2.0, 1.0), 2.0)),
            J2), (:y1, :y2, :x))
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(:(L ~ LKJCovarianceFactor(2, Exponential(1.0), 0.0)),
            J2), (:y1, :y2, :x))
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(:(eta ~ Exponential(1.0)),
            :(L ~ LKJCovarianceFactor(2, Exponential(1.0), eta)), J2),
        (:y1, :y2, :x))
    # A factor K that disagrees with the joint width fails contract
    # validation inside lowering.
    @test_throws ContractValidationError lower_rkppl(
        _corr_surface(:(L ~ LKJCovarianceFactor(3, Exponential(1.0), 2.0)),
            J2), (:y1, :y2, :x))
    # The factor stem is not a coefficient.
    @test_throws SurfaceLoweringError lower_rkppl(quote
            mu1 = L .+ b1 .* x
            mu2 = a2 .+ b2 .* x
            a2 ~ Normal(0, 1)
            b1 ~ Normal(0, 1)
            b2 ~ Normal(0, 1)
            L ~ LKJCovarianceFactor(2, Exponential(1.0), 2.0)
            [y1, y2] ~ MvNormalCholesky([mu1, mu2], L)
        end, (:y1, :y2, :x))
    # A user definition colliding with a derived factor piece fails.
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(:(L_scales ~ Normal(0, 1)), F2, J2), (:y1, :y2, :x))
end

@testset "correlated emission shape" begin
    bound = bind_data(_corr_plan2(), _corr_data2())
    built = build_kernel(bound)
    src = string(kernel_expr(bound, built.layout))
    @test occursin("_ppl_mvn_Le_", src) # row-scaled L entries
    @test occursin("_ppl_prior_L_L_corr", src) # LKJ prior node
    @test occursin("plate", src) # the row-grouped plate
end
