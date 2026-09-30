# IRT family (matrix-b): composed predictors v2 — varying-effect
# sub-predictors (`th = r_t`, brms `theta ~ 0 + (1 | person)`), dotted
# `exp.` maps (`a = exp(log_a)`), and names bound to compositions
# (`d = exp.(la) .* th; eta1 = d .- s1`) — lowering, SB parity against
# the pair partner's BridgeStan numbers, independent Distributions
# oracles at non-uniform probes, Enzyme-vs-findiff, and Reactant/XLA.
#
# SB numbers: briefs 2026-09-27T22-26-42-812-1jgk1py (numbers) and
# 2026-09-27T22-26-43-751-1l3mnz4 (verdict) on
# BayesianRegressionModels:rk:kernel:matrix-b (BRM 88c5621, StanBlocks
# 342436de, BridgeStan 2.9.0; full posterior, propto=false, Jacobian
# included). Partner formulas + data recovered from their run scripts
# (matrix-b-spikes3/4.jl, -numbers5.jl, -hier2pl-numbers.jl,
# -oracles6.jl).
#
# Parity design. S-LSAT/S-LSAT2, I-2pl and I-hier2pl share SB's
# parameterization (per-level coefficients, non-centered tau/z_flat
# draws, tanh-CPC 2x2 LKJ factor), so they compare DIRECTLY at the same
# u (the banked probes are uniform, so coordinate ORDER is covered by
# the non-uniform oracle legs instead). I-latreg / I-gpcm / I-grsm are
# emitted by SB in a COLLAPSED-TOTALS parameterization: the per-person
# totals t_j = Intercept + r_j are the sampled coordinates and the
# Intercept is integrated out analytically (t ~ N(loc, J/prec + tau^2 I));
# gpcm/grsm additionally represent the Student-t(3) Intercept prior as a
# Gamma(nu/2, rate nu/2) scale mixture with its own sampled coordinate
# `mix` (Intercept | mix ~ N(0, 1/mix)). RK samples the Intercept and the
# non-centered z_j directly, so these legs integrate the RK kernel over
# the Intercept: with z_j = (t_j - t0)/tau and
#   h(t0) = RK(u(t0)) - n*log(tau)
# (the non-centered -> centered Jacobian), h is exactly quadratic in t0
# (Gaussian prior x Gaussian person terms; the likelihood depends on the
# totals only), so the Laplace form h(m) + log(2pi/P)/2 is the exact
# integral. The legs assert quadraticity (h at m +/- delta) and the
# integral == SB banked; gpcm/grsm run the conditional-on-mix variant
# (RK Intercept prior N(0, 1/sqrt(mix))) and add SB's mixture terms
# log Gamma(mix) + u_mix. The actual Student-t programs are covered by
# the Distributions oracle legs.
# (`_check_gradient` / `_GEN_BACKEND` come from test_generator.jl.)
using Distributions: Normal, Cauchy, Exponential, Gamma, LKJ, TDist,
    LocationScale, logpdf
using LinearAlgebra: Symmetric
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

function _irt_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

_irt_cols(nt) = Dict{Symbol,AbstractVector}(k => collect(v)
    for (k, v) in pairs(nt))
_irt_val(kern, u) = Base.invokelatest(kern, u)
_irt_logit_lpmf(y, e) = y * e - log1p(exp(e))
# Reference-coded 3-category logit: class 1 is the zero reference.
_irt_cat_lpmf(y, e1, e2) = (0.0, e1, e2)[y] -
    log(1.0 + exp(e1) + exp(e2))
_irt_t3(x) = logpdf(LocationScale(0.0, 1.0, TDist(3)), x)
# Non-uniform probe (distinct coordinates catch mapping bugs the
# uniform banked probes cannot).
_irt_probe(n) = [0.3 * sin(1.7i) + 0.05i for i in 1:n]

const _IRT_PD = (y = [1, 0, 1, 0, 1, 1], person = [1, 1, 2, 2, 3, 3],
    item = [1, 2, 1, 2, 1, 2],
    w1 = [0.5, 0.5, -1.0, -1.0, 1.5, 1.5],
    w2 = [1.0, 1.0, 0.5, 0.5, -0.5, -0.5])
const _IRT_CD = merge(_IRT_PD, (y = [1, 3, 2, 1, 3, 2],))
const _IRT_HD = (y = [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1],
    person = repeat(1:4, inner = 3), item = repeat(1:3, 4))
const _IRT_SD = (y = [1, 0, 1, 0, 1, 1], student = [1, 1, 2, 2, 3, 3],
    question = [1, 2, 1, 2, 1, 2])

const _IRT_LSAT = quote
    c_th[levels(student)] .~ Normal.(0, 1)
    c_al[levels(question)] .~ Normal.(0, 100)
    th = c_th[student]
    al = c_al[question]
    be ~ Normal(0, 100)
    eta = be .* th .- al
    y .~ Bernoulli.(logistic.(eta))
end

const _IRT_2PL = quote
    r_t ~ varying_effect(person, [1]; eta = 1.0, sd = Cauchy(0, 2))
    r_a ~ varying_effect(item, [1]; eta = 1.0, sd = Cauchy(0, 2))
    r_b ~ varying_effect(item, [1]; eta = 1.0, sd = Cauchy(0, 2))
    b0 ~ Normal(0, 5)
    th = r_t
    la = r_a
    b = b0 .+ r_b
    eta = exp.(la) .* (th .- b)
    y .~ Bernoulli.(logistic.(eta))
end

# Intercept prior spliced in: the item program uses `Normal(0, 1)`; the
# collapsed-totals parity leg reuses the body verbatim.
_irt_latreg(t0prior) = quote
    t0 ~ $t0prior
    bw1 ~ StudentT(3, 0, 1)
    bw2 ~ StudentT(3, 0, 1)
    r_t ~ varying_effect(person, [1]; eta = 1.0, sd = Exponential(1))
    c_la[levels(item)] .~ Normal.(1, 1)
    c_b[levels(item)] .~ Normal.(0, 3)
    th = t0 .+ bw1 .* w1 .+ bw2 .* w2 .+ r_t
    la = c_la[item]
    b = c_b[item]
    eta = exp.(la) .* (th .- b)
    y .~ Bernoulli.(logistic.(eta))
end

_irt_gpcm(t0prior) = quote
    t0 ~ $t0prior
    bw1 ~ StudentT(3, 0, 1)
    bw2 ~ StudentT(3, 0, 1)
    r_t ~ varying_effect(person, [1]; eta = 1.0, sd = Exponential(1))
    c_la[levels(item)] .~ Normal.(1, 1)
    c_s1[levels(item)] .~ Normal.(0, 3)
    c_s2[levels(item)] .~ Normal.(0, 3)
    th = t0 .+ bw1 .* w1 .+ bw2 .* w2 .+ r_t
    la = c_la[item]
    s1 = c_s1[item]
    s2 = c_s2[item]
    d = exp.(la) .* th
    eta1 = d .- s1
    eta2 = d .+ d .- s1 .- s2
    y .~ CategoricalLogit.(eta1, eta2)
end

_irt_grsm(t0prior) = quote
    t0 ~ $t0prior
    bw1 ~ StudentT(3, 0, 1)
    bw2 ~ StudentT(3, 0, 1)
    r_t ~ varying_effect(person, [1]; eta = 1.0, sd = Exponential(1))
    c_la[levels(item)] .~ Normal.(1, 1)
    c_b[levels(item)] .~ Normal.(0, 3)
    k1 ~ Normal(0, 3)
    k2 ~ Normal(0, 3)
    th = t0 .+ bw1 .* w1 .+ bw2 .* w2 .+ r_t
    la = c_la[item]
    b = c_b[item]
    d = exp.(la) .* th
    eta1 = d .- b .- k1
    eta2 = d .+ d .- b .- b .- k1 .- k2
    y .~ CategoricalLogit.(eta1, eta2)
end

const _IRT_HIER2PL = quote
    c_th[levels(person)] .~ Normal.(0, 1)
    x1 ~ Normal(0, 1)
    x2 ~ Normal(0, 5)
    dx ~ varying_draws(item, [1, 1]; eta = 4.0, sd = Exponential(10))
    r1 ~ varying_slice(dx, 1)
    r2 ~ varying_slice(dx, 2)
    th = c_th[person]
    xi1 = x1 .+ r1
    xi2 = x2 .+ r2
    eta = exp.(xi1) .* (th .- xi2)
    y .~ Bernoulli.(logistic.(eta))
end

@testset "composed v2 lowering" begin
    # Varying-effect sub-predictors: effect-only and intercept+effect.
    plan = lower_rkppl(_IRT_2PL, (:y, :person, :item))
    @test [(p.name, [t.kind for t in p.terms]) for p in plan.predictors] ==
        [(:la, [VaryingEffectTerm]), (:th, [VaryingEffectTerm]),
         (:b, [InterceptTerm, VaryingEffectTerm]), (:eta, [ComposedTerm])]
    t = only(plan.predictors[4].terms)
    @test t.options.tree == :(exp.(la) .* (th .- b))
    @test t.options.subs == [:la, :th, :b]
    # A name bound to a composition inlines (and never emits).
    gp = lower_rkppl(_irt_gpcm(:(Normal(0, 1))),
        (:y, :person, :item, :w1, :w2))
    names = [p.name for p in gp.predictors]
    @test :d ∉ names
    @test only(gp.predictors[findfirst(==(:eta1), names)].terms).options.tree ==
        :(exp.(la) .* th .- s1)
    @test isempty(gp.derived)
    # A product over a factor sub in an additive sum composes (LSAT).
    ls = lower_rkppl(_IRT_LSAT, (:y, :student, :question))
    @test only(ls.predictors[3].terms).options.tree == :(be .* th .- al)
    # Fail-closed: unadmitted elementwise maps name the admitted set.
    bad_map = quote
        c[levels(g)] .~ Normal.(0, 1)
        th = c[g]
        be ~ Normal(0, 1)
        eta = log.(th) .* be
        y .~ Bernoulli.(logistic.(eta))
    end
    err = try lower_rkppl(bad_map, (:y, :g)); nothing catch e; e end
    @test err isa SurfaceLoweringError
    @test occursin("exp.", sprint(showerror, err))
    # A bare varying contribution (not a sub alias) stays out of trees.
    bare = quote
        r ~ varying_effect(g, [1]; eta = 1.0, sd = Cauchy(0, 2))
        be ~ Normal(0, 1)
        eta = be .* r
        y .~ Bernoulli.(logistic.(eta))
    end
    @test_throws SurfaceLoweringError lower_rkppl(bare, (:y, :g))
end

@testset "IRT SB parity: direct (LSAT, 2PL, hier2pl)" begin
    # S-LSAT (zeros(6)) / S-LSAT2 (0.3^6): per-level coefficients +
    # scalar-by-factor product; names cat_theta_student_beta.1-3,
    # cat_alpha_question_beta.1-2, be == RK order.
    _, built, kern, lay = _irt_query(_IRT_LSAT, _irt_cols(_IRT_SD))
    @test lay.total == 6
    @test _irt_val(kern, zeros(6)) ≈ -23.488024840551986 atol = 1e-12
    @test _irt_val(kern, fill(0.3, 6)) ≈ -23.86605274332301 atol = 1e-12
    # I-2pl (0.3^11): non-centered person/item draws, Cauchy(0,2) sds,
    # b0 ~ Normal(0,5).
    _, _, kern, lay = _irt_query(_IRT_2PL,
        _irt_cols((; y = _IRT_PD.y, person = _IRT_PD.person,
            item = _IRT_PD.item)))
    @test lay.total == 11
    @test _irt_val(kern, fill(0.3, 11)) ≈ -19.77660090228615 atol = 1e-12
    # I-hier2pl (0.3^15): correlated item draws across two predictors
    # (LKJ(4) 2x2, Exponential(10) sds = Stan exponential(0.1)).
    _, _, kern, lay = _irt_query(_IRT_HIER2PL, _irt_cols(_IRT_HD))
    @test lay.total == 15
    @test _irt_val(kern, fill(0.3, 15)) ≈ -28.5192727576817 atol = 1e-12
end

# Exact intercept integral of the RK kernel (see header): returns
# (log-integral, quadraticity residual).
function _irt_intercept_integral(kern, lay, base::Dict{Symbol,Float64},
        t0sym::Symbol, zsyms, tausym::Symbol, totals, s0)
    tau = exp(base[tausym])
    n = length(totals)
    names = coordinate_names(lay)
    function h(t0)
        d = copy(base)
        d[t0sym] = t0
        for (zs, t) in zip(zsyms, totals)
            d[zs] = (t - t0) / tau
        end
        return _irt_val(kern, [d[nm] for nm in names]) - n * log(tau)
    end
    P = 1 / s0^2 + n / tau^2
    m = (sum(totals) / tau^2) / P
    delta = 0.7
    resid = max(abs(h(m + delta) - (h(m) - P * delta^2 / 2)),
        abs(h(m - delta) - (h(m) - P * delta^2 / 2)))
    return h(m) + 0.5 * log(2pi / P), resid
end

@testset "IRT SB parity: collapsed totals (latreg, gpcm, grsm)" begin
    zs = [Symbol("z_flat_person.$j") for j in 1:3]
    # I-latreg (0.3^10): SB totals t_j = 0.3; Intercept ~ N(0, 1).
    _, _, kern, lay = _irt_query(_irt_latreg(:(Normal(0, 1))),
        _irt_cols(_IRT_PD))
    @test lay.total == 11
    base = Dict{Symbol,Float64}(nm => 0.3 for nm in coordinate_names(lay))
    val, resid = _irt_intercept_integral(kern, lay, base,
        Symbol("th.Intercept"), zs, Symbol("tau_person.1"), fill(0.3, 3),
        1.0)
    @test resid < 1e-10
    @test val ≈ -17.61000720604677 atol = 1e-10
    # I-gpcm / I-grsm (0.3^13): SB mixture coordinate u_mix = 0.3,
    # mix = e^0.3 ~ Gamma(shape 1.5, rate 1.5) (Distributions: scale
    # 1/1.5), plus its log-transform Jacobian u_mix.
    mix = exp(0.3)
    sbmix = logpdf(Gamma(1.5, 1 / 1.5), mix) + 0.3
    s0 = 1 / sqrt(mix)
    for (label, mk, banked) in (("I-gpcm", _irt_gpcm, -25.007930680103343),
                                 ("I-grsm", _irt_grsm, -24.751951505370183))
        _, _, kern, lay = _irt_query(mk(:(Normal(0, $s0))),
            _irt_cols(_IRT_CD))
        @test lay.total == 13
        base = Dict{Symbol,Float64}(nm => 0.3 for nm in coordinate_names(lay))
        val, resid = _irt_intercept_integral(kern, lay, base,
            Symbol("th.Intercept"), zs, Symbol("tau_person.1"),
            fill(0.3, 3), s0)
        @test resid < 1e-10
        @test val + sbmix ≈ banked atol = 1e-10
    end
end

@testset "IRT oracles (non-uniform probes)" begin
    pd = _IRT_PD
    # LSAT.
    _, _, kern, lay = _irt_query(_IRT_LSAT, _irt_cols(_IRT_SD))
    u = _irt_probe(lay.total)
    c = Dict(zip(coordinate_names(lay), u))
    th = [c[Symbol("th.student_$j")] for j in 1:3]
    al = [c[Symbol("al.question_$k")] for k in 1:2]
    want = sum(logpdf.(Normal(0, 1), th)) + sum(logpdf.(Normal(0, 100), al)) +
        logpdf(Normal(0, 100), c[:be]) +
        sum(_irt_logit_lpmf(_IRT_SD.y[i],
            c[:be] * th[_IRT_SD.student[i]] - al[_IRT_SD.question[i]])
            for i in 1:6)
    @test _irt_val(kern, u) ≈ want atol = 1e-12

    # 2PL.
    _, _, kern, lay = _irt_query(_IRT_2PL,
        _irt_cols((; y = pd.y, person = pd.person, item = pd.item)))
    u = _irt_probe(lay.total)
    c = Dict(zip(coordinate_names(lay), u))
    lt, la_, lb = c[Symbol("tau_person.1")], c[Symbol("tau_item.1")],
        c[Symbol("tau_item_r_b.1")]
    zt = [c[Symbol("z_flat_person.$j")] for j in 1:3]
    za = [c[Symbol("z_flat_item.$k")] for k in 1:2]
    zb = [c[Symbol("z_flat_item_r_b.$k")] for k in 1:2]
    th = exp(lt) .* zt
    la = exp(la_) .* za
    b = c[Symbol("b.Intercept")] .+ exp(lb) .* zb
    want = logpdf(Normal(0, 5), c[Symbol("b.Intercept")]) +
        sum(logpdf(Cauchy(0, 2), exp(l)) + l for l in (lt, la_, lb)) +
        sum(logpdf.(Normal(0, 1), vcat(zt, za, zb))) +
        sum(_irt_logit_lpmf(pd.y[i], exp(la[pd.item[i]]) *
            (th[pd.person[i]] - b[pd.item[i]])) for i in 1:6)
    @test _irt_val(kern, u) ≈ want atol = 1e-12

    # latreg / gpcm / grsm (the actual Student-t / Normal programs).
    function theta_parts(c)
        tau = exp(c[Symbol("tau_person.1")])
        z = [c[Symbol("z_flat_person.$j")] for j in 1:3]
        wp = [(0.5, 1.0), (-1.0, 0.5), (1.5, -0.5)]
        th = [c[Symbol("th.Intercept")] + c[Symbol("th.w1")] * w[1] +
              c[Symbol("th.w2")] * w[2] + tau * z[j]
              for (j, w) in enumerate(wp)]
        pr = logpdf(Exponential(1), tau) + c[Symbol("tau_person.1")] +
            sum(logpdf.(Normal(0, 1), z)) +
            _irt_t3(c[Symbol("th.w1")]) + _irt_t3(c[Symbol("th.w2")])
        la = [c[Symbol("la.item_$k")] for k in 1:2]
        pr += sum(logpdf.(Normal(1, 1), la))
        return th, exp.(la), pr
    end
    _, _, kern, lay = _irt_query(_irt_latreg(:(Normal(0, 1))),
        _irt_cols(pd))
    u = _irt_probe(lay.total)
    c = Dict(zip(coordinate_names(lay), u))
    th, a, pr = theta_parts(c)
    b = [c[Symbol("b.item_$k")] for k in 1:2]
    want = pr + logpdf(Normal(0, 1), c[Symbol("th.Intercept")]) +
        sum(logpdf.(Normal(0, 3), b)) +
        sum(_irt_logit_lpmf(pd.y[i],
            a[pd.item[i]] * (th[pd.person[i]] - b[pd.item[i]])) for i in 1:6)
    @test _irt_val(kern, u) ≈ want atol = 1e-12

    cd = _IRT_CD
    _, _, kern, lay = _irt_query(_irt_gpcm(:(StudentT(3, 0, 1))),
        _irt_cols(cd))
    u = _irt_probe(lay.total)
    c = Dict(zip(coordinate_names(lay), u))
    th, a, pr = theta_parts(c)
    s1 = [c[Symbol("s1.item_$k")] for k in 1:2]
    s2 = [c[Symbol("s2.item_$k")] for k in 1:2]
    want = pr + _irt_t3(c[Symbol("th.Intercept")]) +
        sum(logpdf.(Normal(0, 3), vcat(s1, s2))) +
        sum(1:6) do i
            dd = a[cd.item[i]] * th[cd.person[i]]
            k = cd.item[i]
            _irt_cat_lpmf(cd.y[i], dd - s1[k], 2dd - s1[k] - s2[k])
        end
    @test _irt_val(kern, u) ≈ want atol = 1e-12

    _, _, kern, lay = _irt_query(_irt_grsm(:(StudentT(3, 0, 1))),
        _irt_cols(cd))
    u = _irt_probe(lay.total)
    c = Dict(zip(coordinate_names(lay), u))
    th, a, pr = theta_parts(c)
    b = [c[Symbol("b.item_$k")] for k in 1:2]
    k1, k2 = c[:k1], c[:k2]
    want = pr + _irt_t3(c[Symbol("th.Intercept")]) +
        sum(logpdf.(Normal(0, 3), vcat(b, [k1, k2]))) +
        sum(1:6) do i
            dd = a[cd.item[i]] * th[cd.person[i]]
            k = cd.item[i]
            _irt_cat_lpmf(cd.y[i], dd - b[k] - k1,
                2dd - 2b[k] - k1 - k2)
        end
    @test _irt_val(kern, u) ≈ want atol = 1e-12

    # hier2pl (the partner's inline oracle, RK coordinate names).
    hd = _IRT_HD
    _, _, kern, lay = _irt_query(_IRT_HIER2PL, _irt_cols(hd))
    u = _irt_probe(lay.total)
    c = Dict(zip(coordinate_names(lay), u))
    L21 = tanh(c[Symbol("L_item.1")])
    L22 = sqrt(1 - L21^2)
    t1, t2 = exp(c[Symbol("tau_item.1")]), exp(c[Symbol("tau_item.2")])
    zf = [c[Symbol("z_flat_item.$i")] for i in 1:6]
    B = ([t1 0.0; 0.0 t2] * [1.0 0.0; L21 L22] * reshape(zf, 2, 3))'
    thv = [c[Symbol("th.person_$j")] for j in 1:4]
    xi1 = c[Symbol("xi1.Intercept")] .+ B[:, 1]
    xi2 = c[Symbol("xi2.Intercept")] .+ B[:, 2]
    want = logpdf(LKJ(2, 4.0), Symmetric([1.0 L21; L21 1.0])) +
        log(1 - L21^2) +
        logpdf(Exponential(10), t1) + logpdf(Exponential(10), t2) +
        c[Symbol("tau_item.1")] + c[Symbol("tau_item.2")] +
        sum(logpdf.(Normal(0, 1), zf)) + sum(logpdf.(Normal(0, 1), thv)) +
        logpdf(Normal(0, 1), c[Symbol("xi1.Intercept")]) +
        logpdf(Normal(0, 5), c[Symbol("xi2.Intercept")]) +
        sum(_irt_logit_lpmf(hd.y[n], exp(xi1[hd.item[n]]) *
            (thv[hd.person[n]] - xi2[hd.item[n]])) for n in 1:12)
    @test _irt_val(kern, u) ≈ want atol = 1e-12
end

const _IRT_ITEMS = (
    ("lsat", _IRT_LSAT, _IRT_SD),
    ("2pl", _IRT_2PL, (; y = _IRT_PD.y, person = _IRT_PD.person,
        item = _IRT_PD.item)),
    ("latreg", _irt_latreg(:(Normal(0, 1))), _IRT_PD),
    ("gpcm", _irt_gpcm(:(StudentT(3, 0, 1))), _IRT_CD),
    ("grsm", _irt_grsm(:(StudentT(3, 0, 1))), _IRT_CD),
    ("hier2pl", _IRT_HIER2PL, _IRT_HD),
)

@testset "IRT Enzyme-vs-findiff ($label)" for (label, prog, data) in _IRT_ITEMS
    bound, built, _, lay = _irt_query(prog, _irt_cols(data))
    _check_gradient(built.spec, bound, _irt_probe(lay.total))
end

function _irt_reactant(prog::Expr, cols)
    bound, built, post_q, lay = _irt_query(prog, cols)
    u = _irt_probe(lay.total)
    return Base.invokelatest(_irt_reactant_measure, built, bound, post_q, u)
end

function _irt_reactant_measure(built, bound, post_q, u)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), native, primal, val, g,
        rval = Float64(rval), rgrad = Array(rgrad))
end

@testset "IRT under Reactant ($label)" for (label, prog, data) in _IRT_ITEMS
    fx = _irt_reactant(prog, _irt_cols(data))
    @test fx.primal ≈ fx.native rtol = 1e-9
    @test fx.rval ≈ fx.val rtol = 1e-9
    @test fx.rgrad ≈ fx.g rtol = 1e-7 atol = 1e-9
end

@testset "IRT XLA data-length invariance (2pl)" begin
    small = _irt_reactant(_IRT_2PL, _irt_cols((; y = [1, 0, 1, 0],
        person = [1, 1, 2, 2], item = [1, 2, 1, 2])))
    large = _irt_reactant(_IRT_2PL, _irt_cols((;
        y = [1, 0, 1, 0, 1, 1, 0, 1],
        person = [1, 1, 2, 2, 1, 1, 2, 2], item = [1, 2, 1, 2, 1, 2, 1, 2])))
    @test small.lines == large.lines
end
