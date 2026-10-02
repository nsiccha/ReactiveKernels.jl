# Computed coefficients, expression-valued coefficient priors and
# hierarchical identifiability (decision `1cmodra` prong `fallback`;
# hunt-priors `coef_grammar`; hunt-emitter `coef-priors`): a predictor
# summand that is not an affine term lowers as an in-graph derived column
# (an inline `z[g]` reads the declared array `z` as a value; its named
# alias `zg = z[g]` is a factor sub-predictor — one density, both legal);
# coefficient priors take names and
# expressions; a coefficient another statement reads is an ordinary
# parameter; intercept + full-cover factor is identified by a hierarchical
# scale. Every density is checked against a Distributions.jl oracle.
# (`_findiff_grad` / `_GEN_BACKEND` come from test_generator.jl, included
# first.)
using Distributions: Normal, Exponential, Cauchy, Uniform, Beta, Dirichlet,
    truncated, logpdf
using ReactiveKernels
using ReactiveKernelsPPL
using Test

const _FB_COLS = Dict{Symbol,AbstractVector}(
    :y => [0.3, -1.2, 0.8, 1.9, -0.4, 0.6, 1.1, -0.7, 0.2],
    :x => [0.5, -1.0, 1.5, 0.0, -0.5, 1.0, 0.25, -0.75, 2.0],
    :x1 => [1.0, 0.2, -0.3, 0.7, -1.1, 0.4, 0.9, -0.2, -0.6],
    :x2 => [-0.4, 0.8, 0.1, -1.3, 0.6, 0.0, 1.2, -0.9, 0.3],
    :g => [1, 2, 3, 1, 2, 3, 1, 2, 3])

_fb_halfnormal(s) = truncated(Normal(0, s), 0, Inf)
_fb_halfcauchy(s) = truncated(Cauchy(0, s), 0, Inf)
_fb_col(k) = Vector{Float64}(_FB_COLS[k])

function _fb_cols(names)
    return Dict{Symbol,AbstractVector}(k => _FB_COLS[k] for k in names)
end

# Lower + bind + build; returns (plan, bound, built, sampler kernel).
function _fb_build(prog::Expr, names)
    cols = _fb_cols(names)
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return plan, bound, built, kern
end

# The packed vector at a constrained probe; entries the probe leaves out
# (the empty coefficient block of a coefficient-free composed predictor)
# come from the layout itself.
_fb_unconstrain(lay, q::NamedTuple) =
    unconstrain(lay, merge(constrain(lay, zeros(lay.total)), q))

# Posterior at a constrained probe vs `want(q)` (likelihood + priors at
# the constrained values) plus the layout's log-Jacobian.
function _fb_check(prog::Expr, names, q::NamedTuple, want)
    plan, bound, built, kern = _fb_build(prog, names)
    lay = built.layout
    u = _fb_unconstrain(lay, q)
    got = Base.invokelatest(kern, u)
    @test got ≈ want(q) + logjac(lay, u) rtol = 1e-12
    return plan
end

_fb_normal_ll(mu, sigma) = sum(logpdf.(Normal.(mu, sigma), _fb_col(:y)))

# The intercept and slope a probe reads: a predictor's coefficient block
# (`mu`, in Intercept/column order) when the affine path owns them, else
# the ordinary scalars (a composition keeps `a`, `b` as names).
_fb_a(q) = haskey(q, :a) ? q.a : q.mu[1]
_fb_b(q) = haskey(q, :b) ? q.b : q.mu[2]

const _FB_NC_ALIAS = quote
    a ~ Normal(0, 5); sg ~ HalfNormal(1)
    z[levels(g)] .~ Normal.(0, 1); sigma ~ Exponential(1)
    zg = z[g]
    mu = a .+ sg .* zg
    y .~ Normal.(mu, sigma)
end
const _FB_NC_INLINE = quote
    a ~ Normal(0, 5); sg ~ HalfNormal(1)
    z[levels(g)] .~ Normal.(0, 1); sigma ~ Exponential(1)
    mu = a .+ sg .* z[g]
    y .~ Normal.(mu, sigma)
end
const _FB_SLOPE_ALIAS = quote
    a ~ Normal(0, 5); b ~ Normal(0, 1); sg ~ HalfNormal(1)
    c[levels(g)] .~ Normal.(0, sg); sigma ~ Exponential(1)
    cg = c[g]
    mu = a .+ b .* x .+ cg .* x
    y .~ Normal.(mu, sigma)
end
const _FB_SLOPE_INLINE = quote
    a ~ Normal(0, 5); b ~ Normal(0, 1); sg ~ HalfNormal(1)
    c[levels(g)] .~ Normal.(0, sg); sigma ~ Exponential(1)
    mu = a .+ b .* x .+ c[g] .* x
    y .~ Normal.(mu, sigma)
end
const _FB_COMPUTED = quote
    a ~ Normal(0, 5); lam ~ HalfCauchy(1); tau ~ HalfCauchy(1)
    z ~ Normal(0, 1)
    b = z * lam * tau
    sigma ~ Exponential(1)
    mu = a .+ b .* x
    y .~ Normal.(mu, sigma)
end
const _FB_EXPR_SCALE = quote
    a ~ Normal(0, 5); b ~ Normal(0, 1 / sqrt(var(x))); sigma ~ Exponential(1)
    mu = a .+ b .* x
    y .~ Normal.(mu, sigma)
end

@testset "fallback: inline and named spellings both lower" begin
    # Naming a subexpression never changes legality. The inline gather
    # keys the declared array under the author's name (a derived column);
    # the alias is a factor sub-predictor. The oracles below check that
    # both give one density.
    for (alias, inline, arr) in ((_FB_NC_ALIAS, _FB_NC_INLINE, :z),
            (_FB_SLOPE_ALIAS, _FB_SLOPE_INLINE, :c))
        pa = lower_rkppl(alias, (:y, :x, :g))
        pinl = lower_rkppl(inline, (:y, :x, :g))
        @test arr in Set(p.name for p in pinl.array_parameters)
        @test any(t -> t.kind === ComposedTerm, last(pa.predictors).terms)
    end
    plan = lower_rkppl(_FB_NC_INLINE, (:y, :g))
    @test [t.kind for t in only(plan.predictors).terms] ==
        [InterceptTerm, OffsetTerm]
    @test only(plan.derived).expr == :(sg .* z[g])
end

@testset "fallback: computed coefficients lower as derived columns" begin
    # P2/P5: a computed scalar times a column is an in-graph derived
    # column (offset term); its parameters are ordinary parameters.
    plan = lower_rkppl(_FB_COMPUTED, (:y, :x))
    mu = only(plan.predictors)
    @test [t.kind for t in mu.terms] == [InterceptTerm, OffsetTerm]
    @test only(plan.derived).expr == :((z * lam * tau) .* x)
    @test Set(p.name for p in plan.parameters) == Set([:lam, :tau, :z, :sigma])
    # The `./ 1.0` spelling that used to be the only door is the same plan
    # up to the extracted expression.
    divided = lower_rkppl(quote
        a ~ Normal(0, 5); lam ~ HalfCauchy(1); tau ~ HalfCauchy(1)
        z ~ Normal(0, 1)
        b = z * lam * tau
        sigma ~ Exponential(1)
        mu = a .+ b .* x ./ 1.0
        y .~ Normal.(mu, sigma)
    end, (:y, :x))
    @test [t.kind for t in only(divided.predictors).terms] ==
        [InterceptTerm, OffsetTerm]
    # A scalar summand has no column to scale.
    @test_throws SurfaceLoweringError lower_rkppl(quote
        a ~ Normal(0, 5); s ~ HalfNormal(1); sigma ~ Exponential(1)
        mu = a .+ s .+ 0.0 .* x
        y .~ Normal.(mu, sigma)
    end, (:y, :x))
end

@testset "fallback: Distributions oracles" begin
    x, x1, x2, g = _fb_col(:x), _fb_col(:x1), _fb_col(:x2), _FB_COLS[:g]
    @testset "P1 sampled coefficient scale" begin
        plan = _fb_check(quote
            s ~ HalfNormal(1); a ~ Normal(0, 5); b ~ Normal(0, s)
            sigma ~ Exponential(1)
            mu = a .+ b .* x
            y .~ Normal.(mu, sigma)
        end, (:y, :x), (mu = [0.4, -0.7], s = 0.8, sigma = 1.3), q -> begin
            a, b = q.mu
            _fb_normal_ll(a .+ b .* x, q.sigma) +
                logpdf(_fb_halfnormal(1), q.s) + logpdf(Normal(0, 5), a) +
                logpdf(Normal(0, q.s), b) + logpdf(Exponential(1), q.sigma)
        end)
        @test plan.population_priors[2].scale === :s
    end
    @testset "P2 computed coefficient through an assignment" begin
        _fb_check(quote
            s ~ HalfNormal(1); a ~ Normal(0, 5); z ~ Normal(0, 1)
            b = s * z; sigma ~ Exponential(1)
            mu = a .+ b .* x
            y .~ Normal.(mu, sigma)
        end, (:y, :x), (mu = [0.4], s = 0.8, z = -0.6, sigma = 1.3), q -> begin
            _fb_normal_ll(q.mu[1] .+ q.s * q.z .* x, q.sigma) +
                logpdf(_fb_halfnormal(1), q.s) +
                logpdf(Normal(0, 5), q.mu[1]) + logpdf(Normal(0, 1), q.z) +
                logpdf(Exponential(1), q.sigma)
        end)
    end
    nc_want(zv) = q -> _fb_normal_ll(_fb_a(q) .+ q.sg .* q[zv][g], q.sigma) +
        logpdf(Normal(0, 5), _fb_a(q)) + logpdf(_fb_halfnormal(1), q.sg) +
        sum(logpdf.(Normal(0, 1), q[zv])) + logpdf(Exponential(1), q.sigma)
    @testset "P3 / A2 non-centered intercept, inline and alias" begin
        _fb_check(_FB_NC_INLINE, (:y, :g), (z = [0.3, -0.9, 1.4],
            mu = [0.2], sg = 0.7, sigma = 1.1), nc_want(:z))
        _fb_check(_FB_NC_ALIAS, (:y, :g), (zg = [0.3, -0.9, 1.4],
            a = 0.2, sg = 0.7, sigma = 1.1), nc_want(:zg))
    end
    @testset "Q5 reordered `z[g] .* sg`" begin
        _fb_check(quote
            a ~ Normal(0, 5); sg ~ HalfNormal(1)
            z[levels(g)] .~ Normal.(0, 1); sigma ~ Exponential(1)
            mu = a .+ z[g] .* sg
            y .~ Normal.(mu, sigma)
        end, (:y, :g), (z = [0.3, -0.9, 1.4], mu = [0.2], sg = 0.7,
            sigma = 1.1), nc_want(:z))
    end
    @testset "P3c intercept + hierarchical full-cover factor" begin
        plan = _fb_check(quote
            a ~ Normal(0, 5); sg ~ HalfNormal(1)
            c[levels(g)] .~ Normal.(0, sg); sigma ~ Exponential(1)
            mu = a .+ c[g]
            y .~ Normal.(mu, sigma)
        end, (:y, :g), (mu = [0.2, 0.3, -0.9, 1.4], sg = 0.7, sigma = 1.1),
        q -> begin
            a, c = q.mu[1], q.mu[2:4]
            _fb_normal_ll(a .+ c[g], q.sigma) + logpdf(Normal(0, 5), a) +
                logpdf(_fb_halfnormal(1), q.sg) +
                sum(logpdf.(Normal(0, q.sg), c)) +
                logpdf(Exponential(1), q.sigma)
        end)
        @test [t.kind for t in only(plan.predictors).terms] ==
            [InterceptTerm, FactorTerm]
    end
    @testset "S6 centered intercept beside a population intercept" begin
        # The plain spelling (aliased factor index under `.+`) is the
        # affine Intercept + FactorTerm block; the hierarchical scale
        # identifies it.
        _fb_check(quote
            a ~ Normal(0, 5)
            r_sg ~ HalfNormal(1)
            r_c[levels(g)] .~ Normal.(0, r_sg)
            r = r_c[g]
            sigma ~ Exponential(1)
            mu = a .+ r
            y .~ Normal.(mu, sigma)
        end, (:y, :g), (mu = [0.2, 0.3, -0.9, 1.4], r_sg = 0.7,
            sigma = 1.1), q -> begin
            a, c = q.mu[1], q.mu[2:4]
            _fb_normal_ll(a .+ c[g], q.sigma) + logpdf(Normal(0, 5), a) +
                logpdf(_fb_halfnormal(1), q.r_sg) +
                sum(logpdf.(Normal(0, q.r_sg), c)) +
                logpdf(Exponential(1), q.sigma)
        end)
    end
    @testset "P4 varying slope, inline and alias" begin
        slope_want(cv) = q -> begin
            c = q[cv]
            a, b = _fb_a(q), _fb_b(q)
            _fb_normal_ll(a .+ b .* x .+ c[g] .* x, q.sigma) +
                logpdf(Normal(0, 5), a) + logpdf(Normal(0, 1), b) +
                logpdf(_fb_halfnormal(1), q.sg) +
                sum(logpdf.(Normal(0, q.sg), c)) +
                logpdf(Exponential(1), q.sigma)
        end
        _fb_check(_FB_SLOPE_INLINE, (:y, :x, :g), (c = [0.3, -0.9, 1.4],
            mu = [0.2, -0.4], sg = 0.7, sigma = 1.1), slope_want(:c))
        _fb_check(_FB_SLOPE_ALIAS, (:y, :x, :g), (cg = [0.3, -0.9, 1.4],
            a = 0.2, b = -0.4, sg = 0.7, sigma = 1.1), slope_want(:cg))
    end
    hs_want(q) = _fb_normal_ll(q.mu[1] .+ (q.z * q.lam * q.tau) .* x,
        q.sigma) + logpdf(Normal(0, 5), q.mu[1]) +
        logpdf(_fb_halfcauchy(1), q.lam) + logpdf(_fb_halfcauchy(1), q.tau) +
        logpdf(Normal(0, 1), q.z) + logpdf(Exponential(1), q.sigma)
    @testset "P5 horseshoe spelled out" begin
        _fb_check(_FB_COMPUTED, (:y, :x), (mu = [0.4], lam = 0.9, tau = 0.3,
            z = -1.2, sigma = 1.3), hs_want)
    end
    @testset "Q1/Q4 reordered `x .* (z * lam * tau)`" begin
        _fb_check(quote
            a ~ Normal(0, 5); lam ~ HalfCauchy(1); tau ~ HalfCauchy(1)
            z ~ Normal(0, 1); sigma ~ Exponential(1)
            mu = a .+ x .* (z * lam * tau)
            y .~ Normal.(mu, sigma)
        end, (:y, :x), (mu = [0.4], lam = 0.9, tau = 0.3, z = -1.2,
            sigma = 1.3), hs_want)
    end
    @testset "Q3 computed coefficient in a named column" begin
        # The named column is a coefficient-free predictor over the
        # in-graph derived column; `mu` composes it with the scalar `a`,
        # which stays an ordinary parameter (no intercept coefficient).
        plan = _fb_check(quote
            a ~ Normal(0, 5); lam ~ HalfCauchy(1); tau ~ HalfCauchy(1)
            z ~ Normal(0, 1); sigma ~ Exponential(1)
            eff = (z * lam * tau) .* x
            mu = a .+ eff
            y .~ Normal.(mu, sigma)
        end, (:y, :x), (a = 0.4, lam = 0.9, tau = 0.3, z = -1.2,
            sigma = 1.3), q -> hs_want(merge(q, (mu = [q.a],))))
        @test only(plan.derived).expr == :((z * lam * tau) .* x)
        @test only(only(plan.predictors[1].terms).columns) ===
              only(plan.derived).name
    end
    @testset "Q7 parameter-scaled column beside an affine term" begin
        _fb_check(quote
            a ~ Normal(0, 5); s ~ HalfNormal(1); b ~ Normal(0, 1)
            sigma ~ Exponential(1)
            mu = a .+ b .* x .+ s .* x1
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :x1), (mu = [0.4, -0.3], s = 0.8, sigma = 1.3),
        q -> _fb_normal_ll(q.mu[1] .+ q.mu[2] .* x .+ q.s .* x1, q.sigma) +
            logpdf(Normal(0, 5), q.mu[1]) + logpdf(Normal(0, 1), q.mu[2]) +
            logpdf(_fb_halfnormal(1), q.s) + logpdf(Exponential(1), q.sigma))
    end
    @testset "expression coefficient scale (assignment)" begin
        plan = _fb_check(_FB_EXPR_SCALE, (:y, :x), (mu = [0.4, -0.3],
            sigma = 1.3), q -> _fb_normal_ll(q.mu[1] .+ q.mu[2] .* x,
            q.sigma) + logpdf(Normal(0, 5), q.mu[1]) +
            logpdf(Normal(0, 1 / sqrt(sum(abs2, x .- sum(x) / length(x)) /
                (length(x) - 1))), q.mu[2]) + logpdf(Exponential(1), q.sigma))
        @test plan.population_priors[2].scale === :_rkppl_b_arg2
        @test only(plan.assignments).expr == :(1 / sqrt(var(x)))
        # Literal arithmetic folds to a literal.
        lit = lower_rkppl(quote
            a ~ Normal(0, 1 / 2); sigma ~ Exponential(1)
            mu = a .+ 0.0 .* x
            y .~ Normal.(mu, sigma)
        end, (:y, :x))
        @test lit.population_priors[1].scale == 0.5
    end
    @testset "positive-support coefficient is an ordinary parameter" begin
        _fb_check(quote
            a ~ Normal(0, 5); b ~ HalfNormal(2); sigma ~ Exponential(1)
            mu = a .+ b .* x
            y .~ Normal.(mu, sigma)
        end, (:y, :x), (mu = [0.4], b = 0.6, sigma = 1.3),
        q -> _fb_normal_ll(q.mu[1] .+ q.b .* x, q.sigma) +
            logpdf(Normal(0, 5), q.mu[1]) + logpdf(_fb_halfnormal(2), q.b) +
            logpdf(Exponential(1), q.sigma))
    end
    @testset "a coefficient another prior reads is an ordinary parameter" begin
        plan = _fb_check(quote
            a ~ Normal(0, 5); b ~ Normal(0, 1); c ~ Normal(b, 1)
            sigma ~ Exponential(1)
            mu = a .+ b .* x .+ c .* x1
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :x1), (mu = [0.4, 0.9], b = -0.5, sigma = 1.3),
        q -> _fb_normal_ll(q.mu[1] .+ q.b .* x .+ q.mu[2] .* x1, q.sigma) +
            logpdf(Normal(0, 5), q.mu[1]) + logpdf(Normal(0, 1), q.b) +
            logpdf(Normal(q.b, 1), q.mu[2]) + logpdf(Exponential(1), q.sigma))
        @test plan.population_priors[2].location === :b
        # One coefficient used on two columns: one parameter, two columns.
        _fb_check(quote
            a ~ Normal(0, 5); b ~ Normal(0, 1); sigma ~ Exponential(1)
            mu = a .+ b .* x .+ b .* x1
            y .~ Normal.(mu, sigma)
        end, (:y, :x, :x1), (mu = [0.4], b = -0.5, sigma = 1.3),
        q -> _fb_normal_ll(q.mu[1] .+ q.b .* (x .+ x1), q.sigma) +
            logpdf(Normal(0, 5), q.mu[1]) + logpdf(Normal(0, 1), q.b) +
            logpdf(Exponential(1), q.sigma))
    end
    @testset "signed asymmetric and hierarchical priors" begin
        # `Uniform(0, 2)` under `.-` keeps b in (0, 2): the stored
        # coordinate is -b in (-2, 0).
        _fb_check(quote
            a ~ Normal(0, 5); b ~ Uniform(0, 2); sigma ~ Exponential(1)
            mu = a .- b .* x
            y .~ Normal.(mu, sigma)
        end, (:y, :x), (mu = [0.4, -1.5], sigma = 1.3),
        q -> _fb_normal_ll(q.mu[1] .+ q.mu[2] .* x, q.sigma) +
            logpdf(Normal(0, 5), q.mu[1]) +
            logpdf(Uniform(0, 2), -q.mu[2]) + logpdf(Exponential(1), q.sigma))
        # A sign-flipped hierarchical location negates through an
        # assignment instead of refusing.
        _fb_check(quote
            m ~ Normal(0, 1); s ~ HalfNormal(1)
            c[levels(g)] .~ Normal.(m, s); sigma ~ Exponential(1)
            mu = .-(c[g])
            y .~ Normal.(mu, sigma)
        end, (:y, :g), (mu = [-0.3, 0.9, -1.4], m = 0.5, s = 0.7,
            sigma = 1.1), q -> _fb_normal_ll(q.mu[g], q.sigma) +
            logpdf(Normal(0, 1), q.m) + logpdf(_fb_halfnormal(1), q.s) +
            sum(logpdf.(Normal(q.m, q.s), .-q.mu)) +
            logpdf(Exponential(1), q.sigma))
    end
    # Element reads of a simplex (P6, P6b, Q6): `phi[1]` is a scalar in a
    # coefficient prior argument (hoisted), in a per-element vector prior
    # argument and inside a computed coefficient.
    @testset "simplex element reads" begin
        x1, x2 = _fb_col(:x1), _fb_col(:x2)
        v(c) = sum(abs2, c .- sum(c) / length(c)) / (length(c) - 1)
        r2d2_want(q) = _fb_normal_ll(q.mu[1] .+ q.mu[2] .* x1 .+
                q.mu[3] .* x2, q.sigma) + logpdf(Normal(0, 5), q.mu[1]) +
            logpdf(Beta(1, 1), q.R2) + logpdf(Dirichlet([1.0, 1.0]), q.phi) +
            logpdf(_fb_halfnormal(1), q.tau) +
            logpdf(Normal(0, sqrt(q.phi[1] * q.R2 * q.tau^2 / v(x1))),
                q.mu[2]) +
            logpdf(Normal(0, sqrt(q.phi[2] * q.R2 * q.tau^2 / v(x2))),
                q.mu[3]) + logpdf(Exponential(1), q.sigma)
        q = (mu = [0.4, 0.7, -0.5], R2 = 0.3, tau = 0.8, sigma = 1.3,
            phi = [0.35, 0.65])
        # P6: the R2D2 prior spelled out per coefficient.
        _fb_check(quote
            a ~ Normal(0, 5); R2 ~ Beta(1, 1); phi ~ Dirichlet([1.0, 1.0])
            tau ~ HalfNormal(1); sigma ~ Exponential(1)
            b1 ~ Normal(0, sqrt(phi[1] * R2 * tau^2 / var(x1)))
            b2 ~ Normal(0, sqrt(phi[2] * R2 * tau^2 / var(x2)))
            mu = a .+ b1 .* x1 .+ b2 .* x2
            y .~ Normal.(mu, sigma)
        end, (:y, :x1, :x2), q, r2d2_want)
        # P6b: the same prior on a design-matrix coefficient vector.
        _fb_check(quote
            a ~ Normal(0, 5); R2 ~ Beta(1, 1); phi ~ Dirichlet([1.0, 1.0])
            tau ~ HalfNormal(1); sigma ~ Exponential(1)
            X = hcat(x1, x2)
            b[axes(X, 2)] .~ Normal.(0, [sqrt(phi[1] * R2 * tau^2 / var(x1)),
                sqrt(phi[2] * R2 * tau^2 / var(x2))])
            mu = a .+ X * b
            y .~ Normal.(mu, sigma)
        end, (:y, :x1, :x2), q, r2d2_want)
        # Q6: simplex elements as computed coefficients.
        _fb_check(quote
            a ~ Normal(0, 5); phi ~ Dirichlet([1.0, 1.0])
            sigma ~ Exponential(1)
            mu = a .+ x1 .* phi[1] .+ x2 .* phi[2]
            y .~ Normal.(mu, sigma)
        end, (:y, :x1, :x2), (mu = [0.4], sigma = 1.3, phi = [0.35, 0.65]),
        q -> _fb_normal_ll(q.mu[1] .+ x1 .* q.phi[1] .+ x2 .* q.phi[2],
                q.sigma) + logpdf(Normal(0, 5), q.mu[1]) +
            logpdf(Dirichlet([1.0, 1.0]), q.phi) +
            logpdf(Exponential(1), q.sigma))
        # The whole simplex read as a scalar stays refused.
        @test_throws SurfaceLoweringError lower_rkppl(quote
            a ~ Normal(0, 5); phi ~ Dirichlet([1.0, 1.0])
            sigma ~ Exponential(1)
            mu = a .+ phi .* x1
            y .~ Normal.(mu, sigma)
        end, (:y, :x1))
    end
    # One predictor mixing an inline array gather, an n-ary computed
    # coefficient and a simplex element; the emitted kernel is the same
    # for every observation count (no data-length unrolling).
    @testset "computed coefficients beside an inline array gather" begin
        x, x1, g = _fb_col(:x), _fb_col(:x1), _FB_COLS[:g]
        prog = quote
            a ~ Normal(0, 5); lam ~ HalfCauchy(1); tau ~ HalfCauchy(1)
            z ~ Normal(0, 1); sg ~ HalfNormal(1)
            zz[levels(g)] .~ Normal.(0, 1); phi ~ Dirichlet([1.0, 1.0])
            sigma ~ Exponential(1)
            mu = a .+ x .* (z * lam * tau) .+ sg .* zz[g] .+ x1 .* phi[1]
            y .~ Normal.(mu, sigma)
        end
        q = (mu = [0.4], lam = 0.9, tau = 0.3, z = -1.2, sg = 0.7,
            zz = [0.2, -0.5, 0.9], phi = [0.35, 0.65], sigma = 1.3)
        _fb_check(prog, (:y, :x, :x1, :g), q, q -> _fb_normal_ll(_fb_a(q) .+
                x .* (q.z * q.lam * q.tau) .+ q.sg .* q.zz[g] .+
                x1 .* q.phi[1], q.sigma) + logpdf(Normal(0, 5), _fb_a(q)) +
            logpdf(_fb_halfcauchy(1), q.lam) +
            logpdf(_fb_halfcauchy(1), q.tau) + logpdf(Normal(0, 1), q.z) +
            logpdf(_fb_halfnormal(1), q.sg) +
            sum(logpdf.(Normal(0, 1), q.zz)) +
            logpdf(Dirichlet([1.0, 1.0]), q.phi) +
            logpdf(Exponential(1), q.sigma))
        # The alias spelling routes the same summands through the
        # composition path (n-ary `*` folds like Julia's).
        alias = Expr(:block, filter(a -> !(a isa Expr && a.head === :(=) &&
            a.args[1] === :mu), prog.args)...)
        insert!(alias.args, length(alias.args), :(zg = zz[g]))
        insert!(alias.args, length(alias.args), :(mu = a .+ x .* (z * lam *
            tau) .+ sg .* zg .+ x1 .* phi[1]))
        qa = merge(Base.structdiff(q, NamedTuple{(:zz, :mu)}),
            (zg = q.zz, a = q.mu[1]))
        _fb_check(alias, (:y, :x, :x1, :g), qa, q -> _fb_normal_ll(q.a .+
                x .* (q.z * q.lam * q.tau) .+ q.sg .* q.zg[g] .+
                x1 .* q.phi[1], q.sigma) + logpdf(Normal(0, 5), q.a) +
            logpdf(_fb_halfcauchy(1), q.lam) +
            logpdf(_fb_halfcauchy(1), q.tau) + logpdf(Normal(0, 1), q.z) +
            logpdf(_fb_halfnormal(1), q.sg) +
            sum(logpdf.(Normal(0, 1), q.zg)) +
            logpdf(Dirichlet([1.0, 1.0]), q.phi) +
            logpdf(Exponential(1), q.sigma))
        nodes(e) = e isa Expr ? 1 + sum(nodes, e.args; init = 0) : 1
        sizes = map((9, 90)) do n
            cols = Dict{Symbol,AbstractVector}(:y => randn(n), :x => randn(n),
                :x1 => randn(n), :g => repeat([1, 2, 3], n ÷ 3))
            bound = bind_data(lower_rkppl(prog, keys(cols)), cols)
            nodes(ReactiveKernelsPPL.kernel_expr(bound,
                ReactiveKernelsPPL.assign_layout(bound)))
        end
        @test sizes[1] == sizes[2]
    end
    # A dar parameter scaling a column: the posterior equals the dar-only
    # model on `y - beta .* x` at the same unconstrained point.
    @testset "parameter-scaled column beside a dar summand" begin
        x, y = _fb_col(:x), _fb_col(:y)
        dar_prog(loc) = quote
            a ~ Normal(0, 1); b ~ Normal(0, 1)
            beta ~ truncated(Normal(0.5, 0.2), 0, 1)
            sigmad ~ HalfNormal(0.2); sigma ~ Exponential(1)
            $loc
            y .~ Normal.(mu, sigma)
        end
        function post(prog, ycol)
            cols = Dict{Symbol,AbstractVector}(:x => x, :y => ycol)
            bound = bind_data(lower_rkppl(prog, keys(cols)), cols)
            built = build_kernel(bound)
            return built.layout, prepare_query(built, bound, :sampler)
        end
        layA, kA = post(dar_prog(:(mu = a .+ beta .* x .+ dar(beta, sigmad))), y)
        u = [0.05 * k * (-1)^k for k in 1:layA.total]
        beta = constrain(layA, u).beta
        layC, kC = post(dar_prog(:(mu = a .+ dar(beta, sigmad))), y .- beta .* x)
        @test coordinate_names(layC) == coordinate_names(layA)
        @test Base.invokelatest(kA, u) ≈ Base.invokelatest(kC, u) rtol = 1e-12
    end
end

@testset "fallback: identifiability counts hierarchical priors" begin
    hier(prior) = quote
        a ~ Normal(0, 5); s ~ HalfNormal(1); m ~ Normal(0, 1)
        c[levels(g)] .~ $prior; sigma ~ Exponential(1)
        mu = a .+ c[g]
        y .~ Normal.(mu, sigma)
    end
    @test lower_rkppl(hier(:(Normal.(0, s))), (:y, :g)) isa StructuralPlan
    @test lower_rkppl(hier(:(Normal.(0, 2 * s))), (:y, :g)) isa StructuralPlan
    for (prior, msg) in ((:(Normal.(0, 1)), "fixed prior"),
            (:(Normal.(m, s)), "trade off"))
        err = try
            lower_rkppl(hier(prior), (:y, :g))
            nothing
        catch e
            e
        end
        @test err isa SurfaceLoweringError
        @test occursin(msg, sprint(showerror, err))
    end
end

@testset "fallback: Enzyme gradients" begin
    for (prog, names, q) in (
            (_FB_NC_INLINE, (:y, :g), (z = [0.3, -0.9, 1.4],
                mu = [0.2], sg = 0.7, sigma = 1.1)),
            (_FB_COMPUTED, (:y, :x), (mu = [0.4], lam = 0.9, tau = 0.3,
                z = -1.2, sigma = 1.3)),
            (_FB_EXPR_SCALE, (:y, :x), (mu = [0.4, -0.3], sigma = 1.3)))
        _, bound, built, kern = _fb_build(prog, names)
        u = _fb_unconstrain(built.layout, q)
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        grad = similar(u)
        sampler_value_and_gradient!(prep, grad, u)
        @test all(isfinite, grad)
        @test isapprox(grad, _findiff_grad(w -> Base.invokelatest(kern, w), u);
            rtol = 1e-5, atol = 1e-7)
    end
end
