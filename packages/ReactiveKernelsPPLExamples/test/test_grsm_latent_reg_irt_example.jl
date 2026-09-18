using ReactiveKernelsPPLExamples.GrsmLatentRegIrtExample
using ReactiveKernels
import ReactiveKernelsPPLExamples
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, lognormal, student_t
using LogExpFunctions: logsumexp
using SpecialFunctions: loggamma
using Statistics: std
using DifferentiationInterface
import Enzyme

const _GRSM_AE = AutoEnzyme(mode = Enzyme.Reverse)

# Stan-faithful covariate design (obtain_adjustments precedence quirk: 2·sd for k≥2).
function _grsm_W_adj(W)
    J, K = size(W); W_adj = similar(W)
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

_grsm_dim(I, J, K, m) = 2 * I + m + J + K - 2
_grsm_parts(q, I, J, K, m) =
    (q[1:I], q[(I + 1):(2 * I - 1)], q[(2 * I):(2 * I + m - 2)],
     q[(2 * I + m - 1):(2 * I + m - 2 + J)],
     q[(2 * I + m - 1 + J):(2 * I + m - 2 + J + K)])

# Differentiable scalar posterior (explicit non-BLAS operations), transcribed
# from the reference `.stan` (rsm + priors + inverse-ILR-free identity
# transforms; only alpha is log-transformed, Jacobian sum(u_alpha)).
function _grsm_density(q, ii, jj, y, W_adj, I)
    J, K = size(W_adj, 1), size(W_adj, 2)
    m = maximum(y)
    u_alpha, beta_free, kappa_free, theta, lambda = _grsm_parts(q, I, J, K, m)
    alpha = exp.(u_alpha)
    beta = vcat(beta_free, -sum(beta_free))
    kappa = vcat(kappa_free, -sum(kappa_free))
    nld(x, mm, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - mm) / s)^2
    lnld(x, l, s) = -log(x) - log(s) - 0.5 * log(2π) - 0.5 * ((log(x) - l) / s)^2
    stt(x, nu, l, s) = loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * log(nu * π) -
                       log(s) - ((nu + 1) / 2) * log1p(((x - l) / s)^2 / nu)
    lp = sum(lnld(a, 1.0, 1.0) for a in alpha) + sum(nld(b, 0.0, 3.0) for b in beta) +
         sum(nld(k, 0.0, 3.0) for k in kappa) + sum(stt(l, 3.0, 0.0, 1.0) for l in lambda) +
         sum(u_alpha)
    for j in 1:J
        muj = zero(eltype(q))
        for k in 1:K; muj += W_adj[j, k] * lambda[k]; end
        lp += nld(theta[j], muj, 1.0)
    end
    like = zero(eltype(q))
    for n in eachindex(y)
        theta_s = theta[jj[n]] * alpha[ii[n]]
        b = beta[ii[n]]
        Lv = [(v * (theta_s - b) - (v == 0 ? zero(eltype(q)) : sum(kappa[1:v]))) for v in 0:m]
        like += Lv[y[n] + 1] - (maximum(Lv) + log(sum(exp.(Lv .- maximum(Lv)))))
    end
    lp + like
end

function _grsm_reference(q, ii, jj, y, W, I)
    J, K = size(W, 1), size(W, 2)
    m = maximum(y)
    u_alpha, beta_free, kappa_free, theta, lambda = _grsm_parts(q, I, J, K, m)
    alpha = exp.(u_alpha)
    beta = vcat(beta_free, -sum(beta_free))
    kappa = vcat(kappa_free, -sum(kappa_free))
    log_jacobian = sum(u_alpha)
    mu = _grsm_W_adj(W) * lambda
    nld(x, mm, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - mm) / s)^2
    lnld(x, l, s) = -log(x) - log(s) - 0.5 * log(2π) - 0.5 * ((log(x) - l) / s)^2
    stt(x, nu, l, s) = loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * log(nu * π) -
                       log(s) - ((nu + 1) / 2) * log1p(((x - l) / s)^2 / nu)
    prior = sum(lnld(a, 1.0, 1.0) for a in alpha) + sum(nld(b, 0.0, 3.0) for b in beta) +
            sum(nld(k, 0.0, 3.0) for k in kappa) + sum(stt(l, 3.0, 0.0, 1.0) for l in lambda) +
            sum(nld(theta[j], mu[j], 1.0) for j in 1:J)
    like = 0.0
    for n in eachindex(y)
        theta_s = theta[jj[n]] * alpha[ii[n]]
        b = beta[ii[n]]
        Lv = [v * (theta_s - b) - sum(kappa[1:v]; init = 0.0) for v in 0:m]
        like += Lv[y[n] + 1] - logsumexp(Lv)
    end
    (; parameters = (; alpha, beta, kappa, theta, lambda_adj = lambda),
       log_jacobian, prior, likelihood = like,
       posterior = prior + like + log_jacobian, m)
end

const _GRSM_SENTINEL_BEFORE = ReactiveKernelsPPLExamples._DEMO_TAIL_EXECUTIONS[]
const _GRSM_HAVE = (:unconstrained, :ii, :jj, :y, :W, :I)
const _GRSM_BOUND = (; ii = GRSM_LR_II, jj = GRSM_LR_JJ, y = GRSM_LR_Y,
                      W = GRSM_LR_W, I = GRSM_LR_I)

@testset "eager template contract and first use" begin
    q0 = fill(0.1, _grsm_dim(GRSM_LR_I, size(GRSM_LR_W, 1), size(GRSM_LR_W, 2),
                              maximum(GRSM_LR_Y)))
    function _first_use_values()
        g1 = build_grsm_latent_reg_irt_graph()
        g2 = build_grsm_latent_reg_irt_graph()
        (prepare(g1; have = _GRSM_HAVE, want = :posterior, bound = _GRSM_BOUND)(q0),
         prepare(g2; have = _GRSM_HAVE, want = :posterior, bound = _GRSM_BOUND)(q0))
    end
    v1, v2 = _first_use_values()
    @test isfinite(v1)
    @test v1 == v2
    @test ReactiveKernelsPPLExamples._DEMO_TAIL_EXECUTIONS[] == _GRSM_SENTINEL_BEFORE
end

@testset "PPL graph — grsm_latent_reg_irt (posteriordb)" begin
    artifact = evaluate_grsm_latent_reg_irt_source()
    @test ReactiveKernelsPPLExamples._DEMO_TAIL_EXECUTIONS[] ==
          _GRSM_SENTINEL_BEFORE + 1
    @test artifact.source == strip(GRSM_LR_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = artifact.inputs.q
    reference = _grsm_reference(q, GRSM_LR_II, GRSM_LR_JJ, GRSM_LR_Y, GRSM_LR_W,
                                GRSM_LR_I)

    @testset "authored on the current baseline surface" begin
        @test occursin("logsumexp", GRSM_LR_SOURCE)
        @test occursin("student_t(3.0, 0.0, 1.0).logpdf", GRSM_LR_SOURCE)
        @test occursin("lognormal(1.0, 1.0).logpdf", GRSM_LR_SOURCE)
        @test occursin("m::Int = maximum(y)", GRSM_LR_SOURCE)
        @test occursin("beta::Vector{Float64} = SB * beta_free", GRSM_LR_SOURCE)
        @test occursin("kappa::Vector{Float64} = SK * kappa_free", GRSM_LR_SOURCE)
        @test occursin("vcat(0.0, cumsum(kappa))", GRSM_LR_SOURCE)
        @test occursin("_obtain_W_adj(W)", GRSM_LR_SOURCE)
        @test !occursin("struct ", GRSM_LR_SOURCE)
        @test artifact.student_t_object === student_t

        raw_generated = code_expr(artifact.kernel)
        readable = sprint(Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false)
        @test !occursin(r"__ops__\[\d+\]", readable)
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.ii, model.jj, model.y,
                         model.W, model.I),
                 want = (model.prior, model.pointwise, model.likelihood,
                         model.log_jacobian, model.posterior))
        prior, pointwise, likelihood, log_jacobian, posterior =
            prepare(p)(q, GRSM_LR_II, GRSM_LR_JJ, GRSM_LR_Y, GRSM_LR_W, GRSM_LR_I)
        @test all(isfinite, pointwise)
        @test length(pointwise) == length(GRSM_LR_Y)
        @test prior ≈ reference.prior
        @test likelihood ≈ reference.likelihood
        @test likelihood ≈ sum(pointwise)
        @test log_jacobian ≈ reference.log_jacobian
        @test posterior ≈ reference.posterior
    end

    @testset "constrain-only prunes the likelihood work" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.ii, model.jj, model.y,
                         model.W, model.I),
                 want = (model.parameters,))
        produced = Set(canon_id(model.graph, o.id) for r in p.recipes
                       for o in r.outputs)
        @test !(canon_id(model.graph, model.likelihood.id) in produced)
        parameters = prepare(p)(q, GRSM_LR_II, GRSM_LR_JJ, GRSM_LR_Y, GRSM_LR_W,
                                GRSM_LR_I)
        @test parameters.alpha ≈ reference.parameters.alpha
        @test collect(parameters.beta) ≈ reference.parameters.beta
        @test collect(parameters.kappa) ≈ reference.parameters.kappa
    end

    @testset "short-N control (responses are independent)" begin
        ns = 40
        ii_s, jj_s, y_s = GRSM_LR_II[1:ns], GRSM_LR_JJ[1:ns], GRSM_LR_Y[1:ns]
        kb = prepare(model; have = _GRSM_HAVE, want = :posterior,
                     bound = (; ii = ii_s, jj = jj_s, y = y_s,
                                W = GRSM_LR_W, I = GRSM_LR_I))
        @test kb(q) ≈ _grsm_reference(q, ii_s, jj_s, y_s, GRSM_LR_W,
                                      GRSM_LR_I).posterior
    end

    @testset "native plain-Enzyme reverse gradient vs reference oracle" begin
        backend = _GRSM_AE
        W_adj = _grsm_W_adj(GRSM_LR_W)
        kernel = prepare(model; have = _GRSM_HAVE, want = :posterior,
                         bound = _GRSM_BOUND)
        @test kernel(q) ≈ _grsm_density(q, GRSM_LR_II, GRSM_LR_JJ, GRSM_LR_Y,
                                        W_adj, GRSM_LR_I)
        prep = prepare_ad(kernel, backend, q; active = :unconstrained)
        g_rk = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
        g_ref = DifferentiationInterface.gradient(
            qq -> _grsm_density(qq, GRSM_LR_II, GRSM_LR_JJ, GRSM_LR_Y, W_adj,
                                GRSM_LR_I),
            backend, q)
        @test all(isfinite, g_rk)
        @test length(g_rk) == length(q)
        @test all(isapprox.(g_rk, g_ref; rtol = 1e-5))
    end

    @testset "data-generic: the same graph handles an alternate shape" begin
        # I=3 items, J=4 persons, K=2 covariates, m = max(y) = 2 (3 categories).
        Is = 3; Js = 4; Ks = 2
        iis = Int[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        jjs = Int[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        ys = Int[0, 2, 1, 1, 0, 0, 0, 1, 1, 1, 2, 0]
        Ws = Float64[k == 1 ? 1.0 : Float64((j + k) % 2) for j in 1:Js, k in 1:Ks]
        @test maximum(ys) == 2
        g = build_grsm_latent_reg_irt_graph()
        kb = prepare(g; have = _GRSM_HAVE, want = :posterior,
                     bound = (; ii = iis, jj = jjs, y = ys, W = Ws, I = Is))
        dim = _grsm_dim(Is, Js, Ks, maximum(ys))    # = 6 + 2 + 4 + 2 − 2 = 12
        qs = 0.1 .* collect(1:dim)
        vs = kb(qs)
        @test isfinite(vs)
        @test vs ≈ _grsm_density(qs, iis, jjs, ys, _grsm_W_adj(Ws), Is)
        prep = prepare_ad(kb, _GRSM_AE, qs; active = :unconstrained)
        gs = ReactiveKernels.ad_value_and_gradient!(prep, similar(qs), qs)[2]
        @test all(isfinite, gs)
        @test length(gs) == dim
    end
end
