using ReactiveKernelsPPLExamples.HmmDrive1Example
using ReactiveKernels
import ReactiveKernelsPPLExamples
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, dirichlet
import Enzyme
using DifferentiationInterface
using LogExpFunctions: logsumexp
using SpecialFunctions: loggamma

const _HMM1_AE = AutoEnzyme(mode = Enzyme.Reverse)

# Graph-independent reference oracle for posteriordb `hmm_drive_1` (posterior
# `bball_drive_event_1-hmm_drive_1`): the Stan 2.39 inverse-ILR simplex[2] and
# ordered[2] transforms with their exact Jacobians, the Dirichlet transit and
# Normal emission-mean priors, and the plain forward algorithm over the two
# observation streams, written exactly as the reference `.stan` loops write it.
_normal_ld(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
_dirichlet_ld(theta, alpha) =
    loggamma(sum(alpha)) - sum(loggamma(a) for a in alpha) +
    sum((a - 1) * log(t) for (a, t) in zip(alpha, theta))

function _hmm_drive_1_reference(q, u, v, alpha, tau, rho)
    K = 2
    N = length(u)
    s1 = [q[1] / sqrt(2), -q[1] / sqrt(2)]
    lse1 = logsumexp(s1)
    logrow1 = s1 .- lse1
    theta1 = exp.(logrow1)
    s2 = [q[2] / sqrt(2), -q[2] / sqrt(2)]
    lse2 = logsumexp(s2)
    logrow2 = s2 .- lse2
    theta2 = exp.(logrow2)
    phi = [q[3], q[3] + exp(q[4])]
    lambda = [q[5], q[5] + exp(q[6])]
    logtheta = permutedims(hcat(logrow1, logrow2))
    log_jacobian = (-2 * lse1 + 0.5 * log(2)) + (-2 * lse2 + 0.5 * log(2)) +
                   q[4] + q[6]
    prior = _dirichlet_ld(theta1, vec(alpha[1, :])) +
            _dirichlet_ld(theta2, vec(alpha[2, :])) +
            _normal_ld(phi[1], 0.0, 1.0) + _normal_ld(phi[2], 3.0, 1.0) +
            _normal_ld(lambda[1], 0.0, 1.0) + _normal_ld(lambda[2], 3.0, 1.0)
    gamma = Matrix{Float64}(undef, N, K)
    for k in 1:K
        gamma[1, k] = _normal_ld(u[1], phi[k], tau) + _normal_ld(v[1], lambda[k], rho)
    end
    for t in 2:N, k in 1:K
        acc = [gamma[t - 1, j] + logtheta[j, k] +
               _normal_ld(u[t], phi[k], tau) + _normal_ld(v[t], lambda[k], rho)
               for j in 1:K]
        gamma[t, k] = logsumexp(acc)
    end
    likelihood = logsumexp(gamma[N, :])
    (; theta1, theta2, phi, lambda, prior, log_jacobian, likelihood,
       posterior = prior + log_jacobian + likelihood)
end

const _HMM1_SENTINEL_BEFORE = ReactiveKernelsPPLExamples._DEMO_TAIL_EXECUTIONS[]
const _HMM1_HAVE = (:unconstrained, :u, :v, :alpha, :tau, :rho)
const _HMM1_BOUND = (; u = HMM_DRIVE_1_U, v = HMM_DRIVE_1_V, alpha = HMM_DRIVE_1_ALPHA,
                tau = HMM_DRIVE_1_TAU, rho = HMM_DRIVE_1_RHO)
_HMM1_BOUND_N(n) = (; u = HMM_DRIVE_1_U[1:n], v = HMM_DRIVE_1_V[1:n],
                      alpha = HMM_DRIVE_1_ALPHA, tau = HMM_DRIVE_1_TAU,
                      rho = HMM_DRIVE_1_RHO)

@testset "eager template contract and first use" begin
    # Sentinel captured at include time, BEFORE any full-source evaluation in
    # this file: template builds below must leave it unchanged (model-only), no
    # matter how many earlier test files ran full evaluations in this process.
    # First use and a repeated fresh build happen inside this ordinary
    # function, not at top level; the two fresh compositions must agree.
    q0 = [0.3, -0.2, 0.0, 0.0, 0.0, 0.0]
    function _first_use_values()
        g1 = build_hmm_drive_1_graph()
        g2 = build_hmm_drive_1_graph()
        (prepare(g1; have = _HMM1_HAVE, want = :posterior, bound = _HMM1_BOUND)(q0),
         prepare(g2; have = _HMM1_HAVE, want = :posterior, bound = _HMM1_BOUND)(q0))
    end
    v1, v2 = _first_use_values()
    @test isfinite(v1)
    @test v1 == v2
    @test ReactiveKernelsPPLExamples._DEMO_TAIL_EXECUTIONS[] == _HMM1_SENTINEL_BEFORE
end

@testset "PPL graph — hmm_drive_1 (posteriordb basketball-drive HMM)" begin
    artifact = evaluate_hmm_drive_1_source()
    @test ReactiveKernelsPPLExamples._DEMO_TAIL_EXECUTIONS[] ==
          _HMM1_SENTINEL_BEFORE + 1
    @test artifact.source == strip(HMM_DRIVE_1_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model

    @testset "authored on the current baseline surface" begin
        @test occursin("scan(eachrow(scan_rows)", HMM_DRIVE_1_SOURCE)
        @test occursin("mapslices(logsumexp, transitioned; dims = 1)", HMM_DRIVE_1_SOURCE)
        @test occursin("dirichlet(alpha1).logpdf(theta1)", HMM_DRIVE_1_SOURCE)
        @test occursin("normal(3.0, 1.0).logpdf(phi2)", HMM_DRIVE_1_SOURCE)
        @test occursin("logaddexp(s1a, s1b)", HMM_DRIVE_1_SOURCE)
        @test !occursin("struct ", HMM_DRIVE_1_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.dirichlet_object === dirichlet
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        kb = prepare(model;
            have = (:unconstrained, :u, :v, :alpha, :tau, :rho),
            want = (:parameters, :prior, :log_jacobian, :likelihood, :posterior),
            bound = (; u = HMM_DRIVE_1_U, v = HMM_DRIVE_1_V, alpha = HMM_DRIVE_1_ALPHA,
                     tau = HMM_DRIVE_1_TAU, rho = HMM_DRIVE_1_RHO))
        for q in ([0.3, -0.2, 0.0, 0.0, 0.0, 0.0],
                  [0.4, 0.25, -0.1, 0.2, 0.05, -0.15],
                  [-0.5, -0.1, 0.2, -0.3, 0.1, 0.4])
            parameters, prior, log_jacobian, likelihood, posterior = kb(q)
            reference = _hmm_drive_1_reference(q, HMM_DRIVE_1_U, HMM_DRIVE_1_V,
                HMM_DRIVE_1_ALPHA, HMM_DRIVE_1_TAU, HMM_DRIVE_1_RHO)
            @test prior ≈ reference.prior
            @test log_jacobian ≈ reference.log_jacobian
            @test likelihood ≈ reference.likelihood
            @test posterior ≈ reference.posterior
            @test isfinite(posterior)
            @test parameters.theta1 ≈ reference.theta1
            @test parameters.theta2 ≈ reference.theta2
            @test parameters.phi ≈ reference.phi
            @test parameters.lambda ≈ reference.lambda
            @test isapprox(sum(parameters.theta1), 1.0; atol = 1e-12)
            @test isapprox(sum(parameters.theta2), 1.0; atol = 1e-12)
            @test issorted(parameters.phi)
            @test issorted(parameters.lambda)
        end
    end

    @testset "bound raw-data query equals the raw-data query" begin
        plain = prepare(model;
            have = (:unconstrained, :u, :v, :alpha, :tau, :rho), want = :posterior)
        bound = prepare(model;
            have = (:unconstrained, :u, :v, :alpha, :tau, :rho), want = :posterior,
            bound = (; u = HMM_DRIVE_1_U, v = HMM_DRIVE_1_V, alpha = HMM_DRIVE_1_ALPHA,
                     tau = HMM_DRIVE_1_TAU, rho = HMM_DRIVE_1_RHO))
        q = [0.3, -0.2, 0.0, 0.0, 0.0, 0.0]
        @test plain(q, HMM_DRIVE_1_U, HMM_DRIVE_1_V, HMM_DRIVE_1_ALPHA,
                    HMM_DRIVE_1_TAU, HMM_DRIVE_1_RHO) == bound(q)
    end

    @testset "N=1 and short-N controls (Stan data contract N ≥ 1)" begin
        model = artifact.model
        for n in (1, 3)
            kb = prepare(model; have = _HMM1_HAVE, want = :posterior, bound = _HMM1_BOUND_N(n))
            reference = _hmm_drive_1_reference([0.3, -0.2, 0.0, 0.0, 0.0, 0.0],
                HMM_DRIVE_1_U[1:n], HMM_DRIVE_1_V[1:n], HMM_DRIVE_1_ALPHA,
                HMM_DRIVE_1_TAU, HMM_DRIVE_1_RHO)
            @test kb([0.3, -0.2, 0.0, 0.0, 0.0, 0.0]) ≈ reference.posterior
        end
    end

    @testset "plain-Enzyme reverse gradient — documented static-activity gap" begin
        # Natural source, known RK/Enzyme limitation: with the authored scan in
        # the combined :posterior graph, static-activity plain Enzyme.Reverse
        # fails EnzymeRuntimeActivityError (snag scan-prior-enzym-d67d4ac1;
        # <=3 endpoint terms around the same scan pass, and runtime-activity
        # mode is finite and correct). Assert the documented failure so the gap
        # is pinned, not hidden — if RK fixes the snag, replace this block with
        # full gradient assertions against the reference density.
        kb = prepare(model;
            have = (:unconstrained, :u, :v, :alpha, :tau, :rho), want = :posterior,
            bound = (; u = HMM_DRIVE_1_U, v = HMM_DRIVE_1_V, alpha = HMM_DRIVE_1_ALPHA,
                     tau = HMM_DRIVE_1_TAU, rho = HMM_DRIVE_1_RHO))
        q = [0.3, -0.2, 0.0, 0.0, 0.0, 0.0]
        prep = prepare_ad(kb, _HMM1_AE, q; active = :unconstrained)
        err = try
            ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)
            nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing
        @test occursin("EnzymeRuntimeActivityError", err)
    end

end
