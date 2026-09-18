using ReactiveKernelsPPLExamples.Hier2plExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, exponential, bernoulli
using LogExpFunctions: log1pexp
using DifferentiationInterface
import Enzyme

# Independent reference for posteriordb hier_2pl: exp/tanh support transforms,
# Normal/Exponential hyperpriors, the analytic K=2 LKJ(4) Cholesky density, a
# bivariate multi_normal_cholesky item prior, and a Bernoulli_logit likelihood.
function _hier_2pl_density(q, ii, jj, y, I, J)
    theta = q[1:J]
    xi1 = q[(J + 1):(J + I)]
    xi2 = q[(J + I + 1):(J + 2I)]
    mu1 = q[J + 2I + 1]; mu2 = q[J + 2I + 2]
    u_tau1 = q[J + 2I + 3]; u_tau2 = q[J + 2I + 4]
    w = q[J + 2I + 5]
    tau1 = exp(u_tau1); tau2 = exp(u_tau2)
    L21 = tanh(w); L22 = sqrt(1 - L21 * L21)
    log_jac = u_tau1 + u_tau2 + log(1 - L21 * L21)
    nld(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    expld(x, sc) = -log(sc) - x / sc
    lp = sum(nld(t, 0.0, 1.0) for t in theta) + nld(mu1, 0.0, 1.0) + nld(mu2, 0.0, 5.0) +
         expld(tau1, 10.0) + expld(tau2, 10.0) + (6.0 * log(L22) - log(6.0 / 6.5625))
    sL21 = tau2 * L21; sL22 = tau2 * L22
    mnc_const = -log(2π) - log(tau1) - log(tau2) - log(L22)
    for i in 1:I
        w1 = (xi1[i] - mu1) / tau1
        w2 = ((xi2[i] - mu2) - sL21 * w1) / sL22
        lp += mnc_const - 0.5 * (w1 * w1 + w2 * w2)
    end
    like = zero(eltype(q))
    for n in eachindex(y)
        eta = exp(xi1[ii[n]]) * (theta[jj[n]] - xi2[ii[n]])
        like += y[n] ? -log1pexp(-eta) : -log1pexp(eta)
    end
    lp + like + log_jac
end

function _hier_2pl_reference(q, ii, jj, y, I, J)
    theta = q[1:J]
    xi1 = q[(J + 1):(J + I)]; xi2 = q[(J + I + 1):(J + 2I)]
    mu1 = q[J + 2I + 1]; mu2 = q[J + 2I + 2]
    tau1 = exp(q[J + 2I + 3]); tau2 = exp(q[J + 2I + 4])
    w = q[J + 2I + 5]; L21 = tanh(w); L22 = sqrt(1 - L21 * L21)
    log_jac = q[J + 2I + 3] + q[J + 2I + 4] + log(1 - L21 * L21)
    alpha = exp.(xi1); beta = xi2
    nld(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    expld(x, sc) = -log(sc) - x / sc
    lp = sum(nld(t, 0.0, 1.0) for t in theta) + nld(mu1, 0.0, 1.0) + nld(mu2, 0.0, 5.0) +
         expld(tau1, 10.0) + expld(tau2, 10.0) + (6.0 * log(L22) - log(6.0 / 6.5625))
    sL21 = tau2 * L21; sL22 = tau2 * L22
    mnc_const = -log(2π) - log(tau1) - log(tau2) - log(L22)
    for i in 1:I
        w1 = (xi1[i] - mu1) / tau1
        w2 = ((xi2[i] - mu2) - sL21 * w1) / sL22
        lp += mnc_const - 0.5 * (w1 * w1 + w2 * w2)
    end
    like = 0.0
    for n in eachindex(y)
        eta = alpha[ii[n]] * (theta[jj[n]] - beta[ii[n]])
        like += y[n] ? -log1pexp(-eta) : -log1pexp(eta)
    end
    (; parameters = (; theta, alpha, beta, mu1, mu2, tau1, tau2, L21, L22),
       log_jacobian = log_jac, prior = lp, likelihood = like, posterior = lp + like + log_jac)
end

@testset "PPL graph — hier_2pl (posteriordb)" begin
    artifact = evaluate_hier_2pl_source()
    @test artifact.source == strip(HIER_2PL_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = artifact.inputs.q
    reference = _hier_2pl_reference(q, HIER_2PL_II, HIER_2PL_JJ, HIER_2PL_Y, HIER_2PL_I, HIER_2PL_J)

    @testset "authored on the current baseline surface" begin
        @test occursin("bernoulli(; logit = e)", HIER_2PL_SOURCE)
        @test occursin("exponential(10.0).logpdf", HIER_2PL_SOURCE)
        @test occursin("L21::Float64 = tanh(w)", HIER_2PL_SOURCE)
        @test occursin("log(6.0 / 6.5625)", HIER_2PL_SOURCE)
        @test !occursin("bernoulli(logistic", HIER_2PL_SOURCE)
        @test !occursin("struct ", HIER_2PL_SOURCE)
        @test artifact.exponential_object === exponential

        raw_generated = code_expr(artifact.kernel)
        readable = sprint(Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false)
        @test !occursin(r"__ops__\[\d+\]", readable)
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.ii, model.jj, model.y, model.I, model.J),
                 want = (model.prior, model.pointwise, model.likelihood,
                         model.log_jacobian, model.posterior))
        prior, pointwise, likelihood, log_jacobian, posterior =
            prepare(p)(q, HIER_2PL_II, HIER_2PL_JJ, HIER_2PL_Y, HIER_2PL_I, HIER_2PL_J)
        @test all(isfinite, pointwise)
        @test length(pointwise) == length(HIER_2PL_Y)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test posterior ≈ reference.posterior
    end

    @testset "constrain-only prunes the likelihood work" begin
        p = plan(model.graph; have = (model.unconstrained, model.I, model.J),
                 want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id) for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        parameters = prepare(p)(q, HIER_2PL_I, HIER_2PL_J)
        @test parameters.alpha ≈ reference.parameters.alpha
        @test parameters.tau1 ≈ reference.parameters.tau1
        @test parameters.L22 ≈ reference.parameters.L22
    end

    @testset "native plain-Enzyme reverse gradient vs reference oracle" begin
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        kernel = prepare(model; have = (:unconstrained, :ii, :jj, :y, :I, :J),
                         want = :posterior,
                         bound = (; ii = HIER_2PL_II, jj = HIER_2PL_JJ, y = HIER_2PL_Y,
                                    I = HIER_2PL_I, J = HIER_2PL_J))
        @test kernel(q) ≈ _hier_2pl_density(q, HIER_2PL_II, HIER_2PL_JJ, HIER_2PL_Y, HIER_2PL_I, HIER_2PL_J)
        prep = prepare_ad(kernel, backend, q; active = :unconstrained)
        g_rk = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
        g_ref = DifferentiationInterface.gradient(
            qq -> _hier_2pl_density(qq, HIER_2PL_II, HIER_2PL_JJ, HIER_2PL_Y, HIER_2PL_I, HIER_2PL_J),
            backend, q)
        @test all(isfinite, g_rk)
        @test length(g_rk) == length(q)
        @test all(isapprox.(g_rk, g_ref; rtol = 1e-5))
    end

    @testset "data-generic: the same graph handles an alternate small shape" begin
        # I=2 items, J=3 persons, N=6 dense.
        Is = 2; Js = 3
        iis = Int[i for j in 1:Js for i in 1:Is]
        jjs = Int[j for j in 1:Js for i in 1:Is]
        ys = Bool[(i + j) % 2 == 0 for j in 1:Js for i in 1:Is]
        g = build_hier_2pl_graph()
        kb = prepare(g; have = (:unconstrained, :ii, :jj, :y, :I, :J), want = :posterior,
                     bound = (; ii = iis, jj = jjs, y = ys, I = Is, J = Js))
        dim = Js + 2Is + 5                          # = 12
        qs = 0.1 .* collect(1:dim)
        vs = kb(qs)
        @test isfinite(vs)
        @test vs ≈ _hier_2pl_density(qs, iis, jjs, ys, Is, Js)
        prep = prepare_ad(kb, AutoEnzyme(; mode = Enzyme.Reverse), qs; active = :unconstrained)
        gs = ReactiveKernels.ad_value_and_gradient!(prep, similar(qs), qs)[2]
        @test all(isfinite, gs)
        @test length(gs) == dim
    end
end
