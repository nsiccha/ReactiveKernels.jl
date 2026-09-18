using ReactiveKernelsPPLExamples.Irt2plExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, cauchy, lognormal, bernoulli
using LogExpFunctions: logistic, log1pexp
using DifferentiationInterface
import Enzyme

# Scalar posterior log-density (prior + log Jacobian + Bernoulli_logit
# likelihood), recomputed independently and differentiably from the Stan
# unconstrained layout — the reference for the native plain-Enzyme gradient axis.
function _irt_2pl_reference_density(q, y)
    I = size(y, 1)
    J = size(y, 2)
    u_sigma_theta = q[1]
    sigma_theta = exp(u_sigma_theta)
    sigma_a = exp(q[J + 2])
    sigma_b = exp(q[J + 4 + I])
    mu_b = q[J + 3 + I]

    nld(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    cld(x, loc, sc) = -log(π) - log(sc) - log1p(((x - loc) / sc)^2)

    lp = cld(sigma_theta, 0.0, 2.0) + cld(sigma_a, 0.0, 2.0) + cld(sigma_b, 0.0, 2.0) +
         nld(mu_b, 0.0, 5.0)
    jac = u_sigma_theta + q[J + 2] + q[J + 4 + I]
    for j in 1:J
        lp += nld(q[1 + j], 0.0, sigma_theta)          # theta[j]
    end
    for i in 1:I
        u_ai = q[J + 2 + i]                             # log a[i]
        ai = exp(u_ai)
        # lognormal(0, sigma_a) lpdf at a[i], plus its exp-transform Jacobian u_ai
        lp += -log(ai) - log(sigma_a) - 0.5 * log(2π) - 0.5 * ((log(ai)) / sigma_a)^2
        jac += u_ai
        bi = q[J + 4 + I + i]                           # b[i]
        lp += nld(bi, mu_b, sigma_b)
    end
    like = 0.0
    for j in 1:J
        theta_j = q[1 + j]
        for i in 1:I
            ai = exp(q[J + 2 + i])
            bi = q[J + 4 + I + i]
            eta = ai * (theta_j - bi)
            like += y[i, j] ? -log1pexp(-eta) : -log1pexp(eta)
        end
    end
    lp + jac + like
end

# Graph-independent reference oracle for posteriordb irt_2pl: a 2PL IRT model with
# Cauchy(0,2) scale priors, Normal ability/difficulty priors, LogNormal
# discriminations, and a Bernoulli_logit response matrix. Recomputes the value
# decomposition (parameters, log_jacobian, prior, likelihood, posterior)
# independently of the RK graph, from the exact Stan unconstrained layout.
function _irt_2pl_reference(q, y)
    I = size(y, 1)
    J = size(y, 2)
    u_sigma_theta = q[1]
    theta = q[2:(J + 1)]
    u_sigma_a = q[J + 2]
    u_a = q[(J + 3):(J + 2 + I)]
    mu_b = q[J + 3 + I]
    u_sigma_b = q[J + 4 + I]
    b = q[(J + 5 + I):(J + 4 + 2I)]

    sigma_theta = exp(u_sigma_theta)
    sigma_a = exp(u_sigma_a)
    a = exp.(u_a)
    sigma_b = exp(u_sigma_b)
    log_jacobian = u_sigma_theta + u_sigma_a + sum(u_a) + u_sigma_b

    nld(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    cld(x, loc, sc) = -log(π) - log(sc) - log1p(((x - loc) / sc)^2)
    lnld(x, loc, sc) = -log(x) - log(sc) - 0.5 * log(2π) - 0.5 * ((log(x) - loc) / sc)^2

    prior = cld(sigma_theta, 0.0, 2.0)
    prior += sum(nld(t, 0.0, sigma_theta) for t in theta)
    prior += cld(sigma_a, 0.0, 2.0)
    prior += sum(lnld(ai, 0.0, sigma_a) for ai in a)
    prior += nld(mu_b, 0.0, 5.0)
    prior += cld(sigma_b, 0.0, 2.0)
    prior += sum(nld(bi, mu_b, sigma_b) for bi in b)

    likelihood = 0.0
    for i in 1:I, j in 1:J
        eta = a[i] * (theta[j] - b[i])
        likelihood += y[i, j] ? -log1pexp(-eta) : -log1pexp(eta)
    end

    (; parameters = (; sigma_theta, theta, sigma_a, a, mu_b, sigma_b, b),
       log_jacobian, prior, likelihood, posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — irt_2pl (posteriordb)" begin
    artifact = evaluate_irt_2pl_source()
    @test artifact.source == strip(IRT_2PL_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = artifact.inputs.q
    reference = _irt_2pl_reference(q, IRT_2PL_Y)

    @testset "authored on the current baseline surface" begin
        @test occursin("bernoulli(; logit = e)", IRT_2PL_SOURCE)
        @test occursin("cauchy(0.0, 2.0).logpdf", IRT_2PL_SOURCE)
        @test occursin("lognormal(0.0, sa).logpdf", IRT_2PL_SOURCE)
        @test occursin("eta::Matrix{Float64} = a .* (transpose(theta) .- b)", IRT_2PL_SOURCE)
        @test !occursin("struct ", IRT_2PL_SOURCE)
        # direct logit HAVE, never a bernoulli(logistic(eta)) round trip
        @test !occursin("bernoulli(logistic", IRT_2PL_SOURCE)
        @test artifact.bernoulli_object === bernoulli
        @test artifact.cauchy_object === cauchy
        @test artifact.lognormal_object === lognormal

        raw_generated = code_expr(artifact.kernel)
        readable = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false,
        )
        @test !occursin(r"__ops__\[\d+\]", readable)
    end

    @testset "constrain-only prunes the likelihood work" begin
        # The unconstrained→constrained layout is data-shape-dependent
        # (dim = 2I + J + 4), so `parameters` needs y for its shape; the
        # likelihood work is still pruned when only `parameters` is wanted.
        p = plan(model.graph; have = (model.unconstrained, model.y),
                 want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        parameters = prepare(p)(q, IRT_2PL_Y)
        @test parameters.sigma_theta ≈ reference.parameters.sigma_theta
        @test parameters.a ≈ reference.parameters.a
        @test parameters.mu_b ≈ reference.parameters.mu_b
        @test collect(parameters.b) ≈ reference.parameters.b
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.y),
                 want = (model.prior, model.pointwise, model.likelihood,
                         model.log_jacobian, model.posterior))
        prior, pointwise, likelihood, log_jacobian, posterior =
            prepare(p)(q, IRT_2PL_Y)
        @test all(isfinite, pointwise)
        @test length(pointwise) == length(IRT_2PL_Y)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test posterior ≈ reference.posterior
    end

    @testset "generated quantity p = inv_logit(eta) prunes the density" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.y), want = (model.p,))
        produced = Set(canon_id(model.graph, o.id)
                       for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        probs = prepare(p)(q, IRT_2PL_Y)
        I = size(IRT_2PL_Y, 1)
        J = size(IRT_2PL_Y, 2)
        a = reference.parameters.a
        theta = reference.parameters.theta
        b = reference.parameters.b
        expected = vec([logistic(a[i] * (theta[j] - b[i])) for i in 1:I, j in 1:J])
        @test probs ≈ expected
    end

    @testset "one authored plate preserves the pointwise-to-total reduction" begin
        pointwise_kernel = prepare(model; have = (:unconstrained, :y), want = :pointwise)
        likelihood_kernel = prepare(model; have = (:unconstrained, :y), want = :likelihood)
        pw = pointwise_kernel(q, IRT_2PL_Y)
        @test likelihood_kernel(q, IRT_2PL_Y) ≈ sum(pw)
        # These generated-source checks distinguish the selected pointwise
        # buffer from the reduced total; they are not runtime allocation claims.
        @test occursin("similar", string(code_expr(pointwise_kernel)))
        @test !occursin("similar", string(code_expr(likelihood_kernel)))
    end

    @testset "native plain-Enzyme reverse gradient vs reference oracle" begin
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        kernel = prepare(model; have = (:unconstrained, :y), want = :posterior,
                         bound = (; y = IRT_2PL_Y))
        @test kernel(q) ≈ _irt_2pl_reference_density(q, IRT_2PL_Y)
        prep = prepare_ad(kernel, backend, q; active = :unconstrained)
        g_rk = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
        g_ref = DifferentiationInterface.gradient(
            qq -> _irt_2pl_reference_density(qq, IRT_2PL_Y), backend, q)
        @test all(isfinite, g_rk)
        @test length(g_rk) == length(q)
        @test all(isapprox.(g_rk, g_ref; rtol = 1e-6))
    end

    @testset "data-generic: the same graph handles an alternate small shape" begin
        # The dims are read from the bound response matrix, never hardcoded, so
        # the SAME built template accepts a different (I, J). 3 items × 4 persons.
        ys = Bool[(i + j) % 2 == 0 for i in 1:3, j in 1:4]
        g = build_irt_2pl_graph()
        kb = prepare(g; have = (:unconstrained, :y), want = :posterior, bound = (; y = ys))
        dim = 2 * 3 + 4 + 4                 # 2I + J + 4 = 14
        qs = 0.1 .* collect(1:dim)
        vs = kb(qs)
        @test isfinite(vs)
        @test vs ≈ _irt_2pl_reference_density(qs, ys)
        prep = prepare_ad(kb, AutoEnzyme(; mode = Enzyme.Reverse), qs; active = :unconstrained)
        gs = ReactiveKernels.ad_value_and_gradient!(prep, similar(qs), qs)[2]
        @test all(isfinite, gs)
        @test length(gs) == dim
    end
end
