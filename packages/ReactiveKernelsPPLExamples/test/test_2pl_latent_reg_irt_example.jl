using ReactiveKernelsPPLExamples.TwoplLatentRegIrtExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, lognormal, student_t, bernoulli
using LogExpFunctions: log1pexp
using SpecialFunctions: loggamma
using Statistics: std
using DifferentiationInterface
import Enzyme

# Independent Stan-faithful covariate standardization (obtain_adjustments +
# centering/scaling), reproducing the upstream operator-precedence quirk so it
# matches BridgeStan: column 1 → (0, 1); k≥2 → centered by mean, scaled by 2·sd.
function _2pl_W_adj(W)
    J, K = size(W)
    W_adj = similar(W)
    for k in 1:K
        col = @view W[:, k]
        if k == 1
            a1 = 0.0; a2 = 1.0
        else
            mn = minimum(col); mx = maximum(col); a1 = sum(col) / J
            mc = 0
            for j in 1:J
                mc = (((mc + col[j]) == mn) || (col[j] == mx)) ? 1 : 0
            end
            a2 = mc == J ? (mx - mn) : 2 * std(col)
        end
        @views @. W_adj[:, k] = (col - a1) / a2
    end
    W_adj
end

# Full value decomposition (parameters, log_jacobian, prior, likelihood).
function _2pl_reference(q, ii, jj, y, W, I)
    J, K = size(W); N = length(y)
    u_alpha = q[1:I]; alpha = exp.(u_alpha)
    beta_free = q[(I + 1):(2I - 1)]
    theta = q[(2I):(2I - 1 + J)]
    lambda = q[(2I + J):(2I + J - 1 + K)]
    beta = vcat(beta_free, -sum(beta_free))
    log_jacobian = sum(u_alpha)
    mu = _2pl_W_adj(W) * lambda
    nld(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    lnld(x, l, s) = -log(x) - log(s) - 0.5 * log(2π) - 0.5 * ((log(x) - l) / s)^2
    stt(x, nu, l, s) = loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * log(nu * π) -
                       log(s) - ((nu + 1) / 2) * log1p(((x - l) / s)^2 / nu)
    prior = sum(lnld(a, 1.0, 1.0) for a in alpha) + sum(nld(b, 0.0, 3.0) for b in beta) +
            sum(stt(l, 3.0, 0.0, 1.0) for l in lambda) +
            sum(nld(theta[j], mu[j], 1.0) for j in 1:J)
    like = 0.0
    for n in 1:N
        eta = alpha[ii[n]] * theta[jj[n]] - beta[ii[n]]
        like += y[n] ? -log1pexp(-eta) : -log1pexp(eta)
    end
    (; parameters = (; alpha, beta, theta, lambda_adj = lambda),
       log_jacobian, prior, likelihood = like, posterior = prior + like + log_jacobian)
end

# Scalar differentiable posterior (W_adj passed in precomputed, so the
# differentiated body is free of the standardization control flow).
function _2pl_density(q, ii, jj, y, W_adj, I)
    J = size(W_adj, 1); K = size(W_adj, 2); N = length(y)
    u_alpha = q[1:I]; alpha = exp.(u_alpha)
    beta_free = q[(I + 1):(2I - 1)]
    theta = q[(2I):(2I - 1 + J)]
    lambda = q[(2I + J):(2I + J - 1 + K)]
    beta = vcat(beta_free, -sum(beta_free))
    nld(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
    lnld(x, l, s) = -log(x) - log(s) - 0.5 * log(2π) - 0.5 * ((log(x) - l) / s)^2
    stt(x, nu, l, s) = loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * log(nu * π) -
                       log(s) - ((nu + 1) / 2) * log1p(((x - l) / s)^2 / nu)
    # Explicit (non-BLAS) mu = W_adj·lambda so Enzyme reverse can differentiate
    # the reference (BLAS gemv on a const matrix is rejected as not-readonly).
    theta_lp = zero(eltype(q))
    for j in 1:J
        muj = zero(eltype(q))
        for k in 1:K
            muj += W_adj[j, k] * lambda[k]
        end
        theta_lp += nld(theta[j], muj, 1.0)
    end
    lp = sum(lnld(a, 1.0, 1.0) for a in alpha) + sum(nld(b, 0.0, 3.0) for b in beta) +
         sum(stt(l, 3.0, 0.0, 1.0) for l in lambda) + theta_lp + sum(u_alpha)
    like = zero(eltype(q))
    for n in 1:N
        eta = alpha[ii[n]] * theta[jj[n]] - beta[ii[n]]
        like += y[n] ? -log1pexp(-eta) : -log1pexp(eta)
    end
    lp + like
end

@testset "PPL graph — 2pl_latent_reg_irt (posteriordb)" begin
    artifact = evaluate_2pl_latent_reg_irt_source()
    @test artifact.source == strip(TWOPL_LR_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = artifact.inputs.q
    reference = _2pl_reference(q, TWOPL_LR_II, TWOPL_LR_JJ, TWOPL_LR_Y, TWOPL_LR_W, TWOPL_LR_I)

    @testset "authored on the current baseline surface" begin
        @test occursin("bernoulli(; logit = e)", TWOPL_LR_SOURCE)
        @test occursin("student_t(3.0, 0.0, 1.0).logpdf", TWOPL_LR_SOURCE)
        @test occursin("lognormal(1.0, 1.0).logpdf", TWOPL_LR_SOURCE)
        @test occursin("W_adj::Matrix{Float64} = _obtain_W_adj(W)", TWOPL_LR_SOURCE)
        @test occursin("beta::Vector{Float64} = S * beta_free", TWOPL_LR_SOURCE)
        @test !occursin("bernoulli(logistic", TWOPL_LR_SOURCE)
        @test !occursin("struct ", TWOPL_LR_SOURCE)
        @test artifact.bernoulli_object === bernoulli
        @test artifact.student_t_object === student_t

        raw_generated = code_expr(artifact.kernel)
        readable = sprint(Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false)
        @test !occursin(r"__ops__\[\d+\]", readable)
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.ii, model.jj, model.y, model.W, model.I),
                 want = (model.prior, model.pointwise, model.likelihood,
                         model.log_jacobian, model.posterior))
        prior, pointwise, likelihood, log_jacobian, posterior =
            prepare(p)(q, TWOPL_LR_II, TWOPL_LR_JJ, TWOPL_LR_Y, TWOPL_LR_W, TWOPL_LR_I)
        @test all(isfinite, pointwise)
        @test length(pointwise) == length(TWOPL_LR_Y)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test posterior ≈ reference.posterior
    end

    @testset "constrain-only prunes the likelihood work" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.ii, model.jj, model.y, model.W, model.I),
                 want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id) for r in p.recipes for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        parameters = prepare(p)(q, TWOPL_LR_II, TWOPL_LR_JJ, TWOPL_LR_Y, TWOPL_LR_W, TWOPL_LR_I)
        @test parameters.alpha ≈ reference.parameters.alpha
        @test collect(parameters.beta) ≈ reference.parameters.beta
        @test parameters.lambda_adj ≈ reference.parameters.lambda_adj
    end

    @testset "native plain-Enzyme reverse gradient vs reference oracle" begin
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        W_adj = _2pl_W_adj(TWOPL_LR_W)
        kernel = prepare(model; have = (:unconstrained, :ii, :jj, :y, :W, :I),
                         want = :posterior,
                         bound = (; ii = TWOPL_LR_II, jj = TWOPL_LR_JJ, y = TWOPL_LR_Y,
                                    W = TWOPL_LR_W, I = TWOPL_LR_I))
        @test kernel(q) ≈ _2pl_density(q, TWOPL_LR_II, TWOPL_LR_JJ, TWOPL_LR_Y, W_adj, TWOPL_LR_I)
        prep = prepare_ad(kernel, backend, q; active = :unconstrained)
        g_rk = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
        g_ref = DifferentiationInterface.gradient(
            qq -> _2pl_density(qq, TWOPL_LR_II, TWOPL_LR_JJ, TWOPL_LR_Y, W_adj, TWOPL_LR_I),
            backend, q)
        @test all(isfinite, g_rk)
        @test length(g_rk) == length(q)
        @test all(isapprox.(g_rk, g_ref; rtol = 1e-5))
    end

    @testset "data-generic: the same graph handles an alternate small shape" begin
        # I=3 items, J=4 persons, K=2 covariates (col1 intercept), N=12 dense.
        Is = 3; Js = 4; Ks = 2
        iis = Int[i for j in 1:Js for i in 1:Is]
        jjs = Int[j for j in 1:Js for i in 1:Is]
        ys = Bool[(i + j) % 2 == 0 for j in 1:Js for i in 1:Is]
        Ws = Float64[k == 1 ? 1.0 : Float64((j + k) % 2) for j in 1:Js, k in 1:Ks]
        g = build_2pl_latent_reg_irt_graph()
        kb = prepare(g; have = (:unconstrained, :ii, :jj, :y, :W, :I), want = :posterior,
                     bound = (; ii = iis, jj = jjs, y = ys, W = Ws, I = Is))
        dim = 2 * Is + Js + Ks - 1                 # = 11
        qs = 0.1 .* collect(1:dim)
        vs = kb(qs)
        @test isfinite(vs)
        @test vs ≈ _2pl_density(qs, iis, jjs, ys, _2pl_W_adj(Ws), Is)
        prep = prepare_ad(kb, AutoEnzyme(; mode = Enzyme.Reverse), qs; active = :unconstrained)
        gs = ReactiveKernels.ad_value_and_gradient!(prep, similar(qs), qs)[2]
        @test all(isfinite, gs)
        @test length(gs) == dim
    end
end
