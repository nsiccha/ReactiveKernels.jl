using ReactiveKernelsPPLExamples.GpcmLatentRegIrtExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, lognormal, student_t
using LogExpFunctions: logsumexp
using SpecialFunctions: loggamma
using Statistics: std
using DifferentiationInterface
import Enzyme

# Stan-faithful covariate design (obtain_adjustments precedence quirk: 2·sd for k≥2).
function _gpcm_W_adj(W)
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
function _gpcm_m(y, ii, I)
    m = zeros(Int, I)
    for n in eachindex(y); y[n] > m[ii[n]] && (m[ii[n]] = y[n]); end
    m
end
function _gpcm_pos(m)
    I = length(m); pos = ones(Int, I)
    for i in 2:I; pos[i] = m[i - 1] + pos[i - 1]; end
    pos
end

# Differentiable scalar posterior (explicit non-BLAS mu, online logsumexp).
function _gpcm_density(q, ii, jj, y, W_adj, I, m, pos, sum_m)
    J = size(W_adj, 1); K = size(W_adj, 2)
    u_alpha = q[1:I]; alpha = exp.(u_alpha)
    beta_free = q[(I + 1):(I + sum_m - 1)]
    theta = q[(I + sum_m):(I + sum_m - 1 + J)]
    lambda = q[(I + sum_m + J):(I + sum_m + J - 1 + K)]
    beta = vcat(beta_free, -sum(beta_free))
    nld(x, mm, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - mm) / s)^2
    lnld(x, l, s) = -log(x) - log(s) - 0.5 * log(2π) - 0.5 * ((log(x) - l) / s)^2
    stt(x, nu, l, s) = loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * log(nu * π) -
                       log(s) - ((nu + 1) / 2) * log1p(((x - l) / s)^2 / nu)
    lp = sum(lnld(a, 1.0, 1.0) for a in alpha) + sum(nld(b, 0.0, 3.0) for b in beta) +
         sum(stt(l, 3.0, 0.0, 1.0) for l in lambda) + sum(u_alpha)
    for j in 1:J
        muj = zero(eltype(q))
        for k in 1:K; muj += W_adj[j, k] * lambda[k]; end
        lp += nld(theta[j], muj, 1.0)
    end
    like = zero(eltype(q))
    for n in eachindex(y)
        i = ii[n]; mi = m[i]
        theta_s = theta[jj[n]] * alpha[i]
        cb = zero(eltype(q)); run_max = -Inf; run_sum = zero(eltype(q)); sel = zero(eltype(q))
        for v in 0:mi
            Lvv = v * theta_s - cb
            v == y[n] && (sel = Lvv)
            if Lvv > run_max
                run_sum = run_sum * exp(run_max - Lvv) + one(eltype(q)); run_max = Lvv
            else
                run_sum += exp(Lvv - run_max)
            end
            v < mi && (cb += beta[pos[i] + v])
        end
        like += sel - (run_max + log(run_sum))
    end
    lp + like
end

function _gpcm_reference(q, ii, jj, y, W, I)
    J = size(W, 1); K = size(W, 2)
    m = _gpcm_m(y, ii, I); pos = _gpcm_pos(m); sum_m = sum(m)
    u_alpha = q[1:I]; alpha = exp.(u_alpha)
    beta_free = q[(I + 1):(I + sum_m - 1)]
    theta = q[(I + sum_m):(I + sum_m - 1 + J)]
    lambda = q[(I + sum_m + J):(I + sum_m + J - 1 + K)]
    beta = vcat(beta_free, -sum(beta_free))
    log_jacobian = sum(u_alpha)
    mu = _gpcm_W_adj(W) * lambda
    nld(x, mm, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - mm) / s)^2
    lnld(x, l, s) = -log(x) - log(s) - 0.5 * log(2π) - 0.5 * ((log(x) - l) / s)^2
    stt(x, nu, l, s) = loggamma((nu + 1) / 2) - loggamma(nu / 2) - 0.5 * log(nu * π) -
                       log(s) - ((nu + 1) / 2) * log1p(((x - l) / s)^2 / nu)
    prior = sum(lnld(a, 1.0, 1.0) for a in alpha) + sum(nld(b, 0.0, 3.0) for b in beta) +
            sum(stt(l, 3.0, 0.0, 1.0) for l in lambda) +
            sum(nld(theta[j], mu[j], 1.0) for j in 1:J)
    like = 0.0
    for n in eachindex(y)
        i = ii[n]; mi = m[i]; theta_s = theta[jj[n]] * alpha[i]
        Lv = [v * theta_s - sum(beta[pos[i]:(pos[i] + v - 1)]; init = 0.0) for v in 0:mi]
        like += Lv[y[n] + 1] - logsumexp(Lv)
    end
    (; parameters = (; alpha, beta, theta, lambda_adj = lambda),
       log_jacobian, prior, likelihood = like, posterior = prior + like + log_jacobian,
       m, pos, sum_m)
end

@testset "PPL graph — gpcm_latent_reg_irt (posteriordb)" begin
    artifact = evaluate_gpcm_latent_reg_irt_source()
    @test artifact.source == strip(GPCM_LR_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    q = artifact.inputs.q
    reference = _gpcm_reference(q, GPCM_LR_II, GPCM_LR_JJ, GPCM_LR_Y, GPCM_LR_W, GPCM_LR_I)

    @testset "authored on the current baseline surface" begin
        @test occursin("logsumexp", GPCM_LR_SOURCE)
        @test occursin("student_t(3.0, 0.0, 1.0).logpdf", GPCM_LR_SOURCE)
        @test occursin("lognormal(1.0, 1.0).logpdf", GPCM_LR_SOURCE)
        @test occursin("_gpcm_m(y, ii, I)", GPCM_LR_SOURCE)
        @test occursin("beta::Vector{Float64} = S * beta_free", GPCM_LR_SOURCE)
        @test occursin("BSEG::Matrix{Float64} = beta[POSIDX]", GPCM_LR_SOURCE)
        @test !occursin("struct ", GPCM_LR_SOURCE)
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
            prepare(p)(q, GPCM_LR_II, GPCM_LR_JJ, GPCM_LR_Y, GPCM_LR_W, GPCM_LR_I)
        @test all(isfinite, pointwise)
        @test length(pointwise) == length(GPCM_LR_Y)
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
        parameters = prepare(p)(q, GPCM_LR_II, GPCM_LR_JJ, GPCM_LR_Y, GPCM_LR_W, GPCM_LR_I)
        @test parameters.alpha ≈ reference.parameters.alpha
        @test collect(parameters.beta) ≈ reference.parameters.beta
    end

    @testset "native plain-Enzyme reverse gradient vs reference oracle" begin
        backend = AutoEnzyme(; mode = Enzyme.Reverse)
        W_adj = _gpcm_W_adj(GPCM_LR_W)
        m = reference.m; pos = reference.pos; sum_m = reference.sum_m
        kernel = prepare(model; have = (:unconstrained, :ii, :jj, :y, :W, :I),
                         want = :posterior,
                         bound = (; ii = GPCM_LR_II, jj = GPCM_LR_JJ, y = GPCM_LR_Y,
                                    W = GPCM_LR_W, I = GPCM_LR_I))
        @test kernel(q) ≈ _gpcm_density(q, GPCM_LR_II, GPCM_LR_JJ, GPCM_LR_Y, W_adj, GPCM_LR_I, m, pos, sum_m)
        prep = prepare_ad(kernel, backend, q; active = :unconstrained)
        g_rk = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
        g_ref = DifferentiationInterface.gradient(
            qq -> _gpcm_density(qq, GPCM_LR_II, GPCM_LR_JJ, GPCM_LR_Y, W_adj, GPCM_LR_I, m, pos, sum_m),
            backend, q)
        @test all(isfinite, g_rk)
        @test length(g_rk) == length(q)
        @test all(isapprox.(g_rk, g_ref; rtol = 1e-5))
    end

    @testset "data-generic: the same graph handles an alternate ragged shape" begin
        # I=3 items with m=(1,2,1) → sum_m=4, K=2 covariates, J=4 persons.
        Is = 3; Js = 4; Ks = 2
        iis = Int[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        jjs = Int[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        ys = Int[0, 2, 1, 1, 0, 0, 0, 1, 1, 1, 2, 0]   # item2 reaches category 2 → m2=2
        Ws = Float64[k == 1 ? 1.0 : Float64((j + k) % 2) for j in 1:Js, k in 1:Ks]
        m = _gpcm_m(ys, iis, Is); pos = _gpcm_pos(m); sum_m = sum(m)
        @test m == [1, 2, 1] && sum_m == 4
        g = build_gpcm_latent_reg_irt_graph()
        kb = prepare(g; have = (:unconstrained, :ii, :jj, :y, :W, :I), want = :posterior,
                     bound = (; ii = iis, jj = jjs, y = ys, W = Ws, I = Is))
        dim = Is + sum_m + Js + Ks - 1                 # = 3+4+4+2-1 = 12
        qs = 0.1 .* collect(1:dim)
        vs = kb(qs)
        @test isfinite(vs)
        @test vs ≈ _gpcm_density(qs, iis, jjs, ys, _gpcm_W_adj(Ws), Is, m, pos, sum_m)
        prep = prepare_ad(kb, AutoEnzyme(; mode = Enzyme.Reverse), qs; active = :unconstrained)
        gs = ReactiveKernels.ad_value_and_gradient!(prep, similar(qs), qs)[2]
        @test all(isfinite, gs)
        @test length(gs) == dim
    end
end
