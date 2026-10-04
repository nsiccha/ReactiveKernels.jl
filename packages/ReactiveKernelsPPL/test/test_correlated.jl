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
        end, (:y1, :y2, :x); conditioned = (:y1, :y2, :x))
end

function _corr_oracle2(nt, cols)
    s = Vector(nt.L_scales)
    Lc = Matrix(nt.L_L_corr)
    L = Diagonal(s) * Lc
    c1, c2 = [nt.a1, nt.b1], [nt.a2, nt.b2]
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
        end, (:y1, :y2, :x); conditioned = (:y1, :y2, :x))
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
        end, (:y1, :y2, :x); conditioned = (:y1, :y2, :x))
    @test any(a -> a.name === :tau, det.assignments)
    # K=1 lowers uniformly (single outcome, no tail).
    one = lower_rkppl(quote
            a1 ~ Normal(0, 1)
            b1 ~ Normal(0, 1)
            mu1 = a1 .+ b1 .* x
            L ~ LKJCovarianceFactor(1, Exponential(1.0), 2.0)
            [y1] ~ MvNormalCholesky([mu1], L)
        end, (:y1, :x); conditioned = (:y1, :x))
    r1 = only(one.responses)
    @test r1.extra_responses == Symbol[] && r1.extra_predictors == Symbol[]
    @test [p.size for p in one.vector_parameters] == [1, 1]
end

@testset "correlated layout edges" begin
    bound = bind_data(_corr_plan2(), _corr_data2())
    layout = assign_layout(bound)
    @test layout.total == 7 # 2+2 coefs + 2 scales + 1 theta
    kinds = [(e.kind, e.name, e.size, e.transform) for e in layout.entries]
    @test only(k for k in kinds if k[2] === :L_scales) ==
        (:vector, :L_scales, 2, :exp)
    @test only(k for k in kinds if k[2] === :L_L_corr) ==
        (:cholesky_corr, :L_L_corr, 1, :lkj)
    u = [0.5, -0.25, 0.1, 0.2, 0.3, -0.1, 0.7]
    nt = constrain(layout, u)
    @test Vector(nt.L_scales) ≈ exp.([0.3, -0.1])
    @test Matrix(nt.L_L_corr) ≈ lkj_chol_constrain([0.7], 2)
    @test !haskey(nt, :b_corr) # no varying derived draws leak
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
        end, (:y1, :x); conditioned = (:y1, :x))
    cols = Dict{Symbol,AbstractVector}(
        :y1 => [0.5, -0.2, 0.8, 0.1], :x => [0.5, -1.0, 1.5, 0.0])
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    @test built.layout.total == 3 # 2 coefs + 1 scale + 0 thetas
    u = [0.5, -0.25, 0.3]
    nt = constrain(built.layout, u)
    s1 = only(Vector(nt.L_scales))
    c1 = [nt.a1, nt.b1]
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
        end, (:y1, :y2, :y3, :x); conditioned = (:y1, :y2, :y3, :x))
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
    cs = [[nt.a1, nt.b1], [nt.a2, nt.b2], [nt.a3, nt.b3]]
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
    # refused: joint MvNormalCholesky requires identity link (IR contract)
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
    # refused: extra predictors must match extra responses (IR contract)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        resp = [no_tail]))
    dup_out = LikelihoodSpec(base.family, base.link, base.response,
        base.predictor, base.scale, base.weights, base.evidence, base.label,
        base.trials, base.range; extra_responses = [:y1],
        extra_predictors = base.extra_predictors,
        factor_scales = base.factor_scales, factor_corr = base.factor_corr)
    # refused: duplicate outcome in joint response (IR contract)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        resp = [dup_out]))
    dup_pred = LikelihoodSpec(base.family, base.link, base.response,
        base.predictor, base.scale, base.weights, base.evidence, base.label,
        base.trials, base.range; extra_responses = base.extra_responses,
        extra_predictors = [:mu1], factor_scales = base.factor_scales,
        factor_corr = base.factor_corr)
    # refused: duplicate predictor in joint response (IR contract)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        resp = [dup_pred]))
    unknown_pred = LikelihoodSpec(base.family, base.link, base.response,
        base.predictor, base.scale, base.weights, base.evidence, base.label,
        base.trials, base.range; extra_responses = base.extra_responses,
        extra_predictors = [:nope], factor_scales = base.factor_scales,
        factor_corr = base.factor_corr)
    # refused: unknown extra predictor (IR contract)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        resp = [unknown_pred]))
    # Each mean predictor keeps its written link; the joint response itself
    # consumes those values on the identity scale (rkppl-use, value locations).
    plan = _corr_plan2()
    preds = PredictorSpec[p for p in plan.predictors]
    preds[2] = PredictorSpec(:mu2, LogitLink, preds[2].terms, preds[2].label)
    @test validate_structure(StructuralPlan(
        plan.responses, preds, plan.population_priors, plan.parameters,
        plan.assignments, plan.columns, plan.n_obs; roles = plan.roles,
        vector_parameters = plan.vector_parameters)) === nothing
    # Factor linkage.
    # refused: missing factor scales (IR contract)
    for (sc, cr) in ((nothing, base.factor_corr),
            # refused: missing factor corr (IR contract)
            (base.factor_scales, nothing),
            # refused: unknown factor scales (IR contract)
            (:nope, base.factor_corr),
            # refused: unknown factor corr (IR contract)
            (base.factor_scales, :nope),
            # refused: swapped factor pieces (IR contract)
            (base.factor_corr, base.factor_scales))
        bad = LikelihoodSpec(base.family, base.link, base.response,
            base.predictor, base.scale, base.weights, base.evidence,
            base.label, base.trials, base.range;
            extra_responses = base.extra_responses,
            extra_predictors = base.extra_predictors, factor_scales = sc,
            factor_corr = cr)
        # refused: factor linkage exactly-once (IR contract); see per-entry lines
        @test_throws ContractValidationError validate_structure(
            _corr_mutate(; resp = [bad]))
    end
    # Factor sizes disagree with the joint width.
    vs = VectorParameter[p for p in plan.vector_parameters]
    vs[1] = VectorParameter(vs[1].name, vs[1].family, vs[1].args, 3,
        vs[1].label)
    # refused: factor size disagrees with joint width (IR contract)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        vps = vs))
    # A declared parameter may contribute its prior without a likelihood use.
    vs2 = vcat(plan.vector_parameters,
        [VectorParameter(:stray, :positive_exponential, (arg1 = 1.0,), 2,
            :stray)])
    @test validate_structure(_corr_mutate(; vps = vs2)) === nothing
    # Two joint responses may consume the same covariance factor.
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
    @test validate_structure(StructuralPlan(
        [base, r2], preds34, priors34, plan.parameters, plan.assignments,
        Dict{Symbol,AbstractVector}(), 0;
        vector_parameters = plan.vector_parameters)) === nothing
    # Joint-only fields on a Gaussian response fail.
    g = LikelihoodSpec(GaussianFam, IdentityLink, :y1, :mu1, :sigma, nothing,
        ResponseEvidence(:none, nothing, nothing), :y_resp, nothing, nothing;
        extra_responses = [:y2])
    gplan = StructuralPlan([g], [plan.predictors[1]],
        PopulationPrior[],
        [filter(p -> p.name in (:a1, :b1), plan.parameters)...,
            SampledParameter(:sigma, :exponential, (arg1 = 1.0,), nothing,
                :sigma)],
        AssignmentSpec[], Dict{Symbol,AbstractVector}(), 0)
    # refused: joint-only fields on a Gaussian response (IR contract)
    @test_throws ContractValidationError validate_structure(gplan)
    # Non-joint extras on a joint response fail.
    # refused: n_levels on joint response (IR contract)
    for kw in (Dict(:n_levels => 2), Dict(:thresholds => :t),
            # refused: count_columns on joint response (IR contract)
            Dict(:count_columns => [:c2]))
        bad = LikelihoodSpec(base.family, base.link, base.response,
            base.predictor, base.scale, base.weights, base.evidence,
            base.label, base.trials, base.range;
            extra_responses = base.extra_responses,
            extra_predictors = base.extra_predictors,
            factor_scales = base.factor_scales, factor_corr = base.factor_corr,
            kw...)
        # refused: non-joint extras on a joint response (IR contract); see per-entry lines
        @test_throws ContractValidationError validate_structure(
            _corr_mutate(; resp = [bad]))
    end
    weights_bad = LikelihoodSpec(base.family, base.link, base.response,
        base.predictor, base.scale, :w, base.evidence, base.label,
        base.trials, base.range; extra_responses = base.extra_responses,
        extra_predictors = base.extra_predictors,
        factor_scales = base.factor_scales, factor_corr = base.factor_corr)
    # refused: weights on joint response (IR contract)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        resp = [weights_bad]))
    trials_bad = LikelihoodSpec(base.family, base.link, base.response,
        base.predictor, base.scale, base.weights, base.evidence, base.label,
        :n, base.range; extra_responses = base.extra_responses,
        extra_predictors = base.extra_predictors,
        factor_scales = base.factor_scales, factor_corr = base.factor_corr)
    # refused: trials on joint response (IR contract)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        resp = [trials_bad]))
    scale_bad = LikelihoodSpec(base.family, base.link, base.response,
        base.predictor, :sigma, base.weights, base.evidence, base.label,
        base.trials, base.range; extra_responses = base.extra_responses,
        extra_predictors = base.extra_predictors,
        factor_scales = base.factor_scales, factor_corr = base.factor_corr)
    # refused: scalar scale on joint response (scale lives in the factor; IR contract)
    @test_throws ContractValidationError validate_structure(_corr_mutate(;
        resp = [scale_bad]))
    # Factor args: bad scale, unknown scale name, bad shape, missing size.
    for bad_vp in (
            # refused: negative exponential scale (IR contract)
            VectorParameter(:L_scales, :positive_exponential, (arg1 = -1.0,),
                2, :L),
            # refused: unknown scale name (IR contract)
            VectorParameter(:L_scales, :positive_exponential,
                (arg1 = :nope,), 2, :L),
            # refused: LKJ eta 0.0 (IR contract)
            VectorParameter(:L_L_corr, :cholesky_corr_lkj, (arg1 = 0.0,), 2,
                :L),
            # refused: each factor entry has valid dimensions, distribution arguments and declared names (IR contract; P6, 05oe96l)
            VectorParameter(:L_L_corr, :cholesky_corr_lkj, (arg1 = :eta,), 2,
                :L),
            # refused: missing factor size (IR contract)
            VectorParameter(:L_scales, :positive_exponential, (arg1 = 1.0,),
                nothing, :L))
        vsbad = bad_vp.name === :L_scales ?
            VectorParameter[bad_vp, plan.vector_parameters[2]] :
            VectorParameter[plan.vector_parameters[1], bad_vp]
        # refused: factor arg validation (IR contract); see per-entry lines
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
    # refused: missing data name (tail outcome `y2`)
    @test_throws ContractValidationError bind_data(plan, missing_tail)
    # A non-numeric tail outcome fails.
    bad_tail = copy(good)
    bad_tail[:y2] = ["a", "b", "c", "d"]
    # refused: wrong eltype (non-numeric tail outcome)
    @test_throws ContractValidationError bind_data(plan, bad_tail)
    # A ragged tail outcome fails the uniform-n_obs rule.
    ragged = copy(good)
    ragged[:y2] = [0.1, 0.4]
    # refused: length mismatch for observation-aligned outcome
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
    # refused: `.~` broadcasts; a joint multivariate observation is `~` (P3)
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(F2,
            :([y1, y2] .~ MvNormalCholesky([mu1, mu2], L))), (:y1, :y2, :x); conditioned = (:y1, :y2, :x))
    # refused: undeclared factor `L` (P6, 05oe96l)
    @test_throws SurfaceLoweringError lower_rkppl(_corr_surface(J2),
        (:y1, :y2, :x); conditioned = (:y1, :y2, :x))
    # refused: mean width != outcome width (malformed distribution)
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(F2, :([y1, y2] ~ MvNormalCholesky([mu1], L))),
        (:y1, :y2, :x); conditioned = (:y1, :y2, :x))
    # refused: `y3` is not data (missing data name / undeclared, P6)
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(F2, :([y1, y3] ~ MvNormalCholesky([mu1, mu2], L))),
        (:y1, :y2, :x); conditioned = (:y1, :y2, :x))
    # refused: outcome listed twice (single assignment)
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(F2, :([y1, y1] ~ MvNormalCholesky([mu1, mu2], L))),
        (:y1, :y2, :x); conditioned = (:y1, :y2, :x))
    # refused: `Normal` over a vector mean and matrix factor is a Julia MethodError (P3; malformed distribution)
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(F2, :([y1, y2] ~ Normal([mu1, mu2], L))), (:y1, :y2, :x); conditioned = (:y1, :y2, :x))
    # refused: broadcasting a multivariate over means for one outcome is malformed (P3)
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(F2, :(y1 .~ MvNormalCholesky.([mu1, mu2], L))),
        (:y1, :y2, :x); conditioned = (:y1, :y2, :x))
    # Factor-statement errors.
    # refused: LKJ dimension K = 0 (mathematically invalid)
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(:(L ~ LKJCovarianceFactor(0, Exponential(1.0), 2.0)),
            J2), (:y1, :y2, :x); conditioned = (:y1, :y2, :x))
    # The covariance factor is an explicit value with ordinary scale priors.
    @test validate_structure(lower_rkppl(_corr_surface(
        :(sd[1:2] .~ Gamma.(2.0, 1.0)), :(C ~ LKJCholesky(2, 2.0)),
        :(L = sd .* C), J2), (:y1, :y2, :x);
        conditioned = (:y1, :y2, :x))) === nothing
    # refused: LKJ eta must be > 0 (mathematically invalid input)
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(:(L ~ LKJCovarianceFactor(2, Exponential(1.0), 0.0)),
            J2), (:y1, :y2, :x); conditioned = (:y1, :y2, :x))
    @test validate_structure(lower_rkppl(_corr_surface(
        :(eta ~ Exponential(1.0)), :(sd[1:2] .~ Exponential.(1.0)),
        :(C ~ LKJCholesky(2, eta)), :(L = sd .* C), J2),
        (:y1, :y2, :x); conditioned = (:y1, :y2, :x))) === nothing
    # A factor K that disagrees with the joint width fails contract
    # validation inside lowering.
    # refused: factor K != joint width (dimension mismatch)
    @test_throws ContractValidationError lower_rkppl(
        _corr_surface(:(L ~ LKJCovarianceFactor(3, Exponential(1.0), 2.0)),
            J2), (:y1, :y2, :x); conditioned = (:y1, :y2, :x))
    # The legacy factor stem is not an authored value. Value expressions
    # lower generically; binding rejects its undeclared assignment input.
    stemmean = lower_rkppl(quote
            mu1 = L .+ b1 .* x
            mu2 = a2 .+ b2 .* x
            a2 ~ Normal(0, 1)
            b1 ~ Normal(0, 1)
            b2 ~ Normal(0, 1)
            L ~ LKJCovarianceFactor(2, Exponential(1.0), 2.0)
            [y1, y2] ~ MvNormalCholesky([mu1, mu2], L)
        end, (:y1, :y2, :x); conditioned = (:y1, :y2, :x))
    @test_throws "assignment references unknown name L" bind_data(
        stemmean, _corr_data2())
    # A user definition colliding with a derived factor piece fails.
    # refused: name collides with construct-minted factor piece `L_scales` (reserved)
    @test_throws SurfaceLoweringError lower_rkppl(
        _corr_surface(:(L_scales ~ Normal(0, 1)), F2, J2), (:y1, :y2, :x); conditioned = (:y1, :y2, :x))
end

@testset "correlated emission shape" begin
    bound = bind_data(_corr_plan2(), _corr_data2())
    built = build_kernel(bound)
    src = string(kernel_expr(bound, built.layout))
    @test occursin("_ppl_mvn_Le_", src) # row-scaled L entries
    @test occursin("_ppl_prior_L_L_corr", src) # LKJ prior node
    @test occursin("plate", src) # the row-grouped plate
end
