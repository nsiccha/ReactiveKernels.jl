using ReactiveKernels
using ReactiveKernelsPPLExamples.GARCH11Example
using DifferentiationInterface
import Enzyme

const _GARCH_AE = AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)

# Graph-independent reference oracle: recomputes the GARCH(1,1) density from
# first principles (support transforms + Jacobian, sequential sd recursion,
# Normal likelihood), matching Stan's propto=false, jacobian=true convention.
_garch_normal(x, m, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - m) / s)^2
_garch_log1pexp(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
_garch_logistic(x) = 1 / (1 + exp(-x))

function _garch11_reference(q, y, sigma1)
    mu, ua0, ua1, ub1 = q[1], q[2], q[3], q[4]
    alpha0 = exp(ua0)
    alpha1 = _garch_logistic(ua1)
    beta1 = (1 - alpha1) * _garch_logistic(ub1)
    jac = ua0 +
          (-_garch_log1pexp(-ua1) - _garch_log1pexp(ua1)) +
          (log(1 - alpha1) - _garch_log1pexp(-ub1) - _garch_log1pexp(ub1))
    T = length(y)
    sigma = Vector{Float64}(undef, T)
    sigma[1] = sigma1
    for t in 2:T
        sigma[t] = sqrt(alpha0 + alpha1 * (y[t - 1] - mu)^2 + beta1 * sigma[t - 1]^2)
    end
    lik = sum(_garch_normal(y[t], mu, sigma[t]) for t in 1:T)
    (; sigma, log_jacobian = jac, likelihood = lik, posterior = lik + jac)
end

function _garch_central_fd(f, q; h = 1e-6)
    g = similar(q)
    for i in eachindex(q)
        qp = copy(q); qp[i] += h
        qm = copy(q); qm[i] -= h
        g[i] = (f(qp) - f(qm)) / (2h)
    end
    g
end

@testset "PPL graph — garch11" begin
    artifact = evaluate_garch11_source()
    @test artifact.source == strip(GARCH11_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    y = GARCH11_Y
    sigma1 = GARCH11_SIGMA1
    q = [0.05, log(0.1), 0.3, -0.2]

    @testset "authored on the current baseline surface" begin
        @test occursin("normal(m, st).logpdf", GARCH11_SOURCE)
        @test occursin("scan(y_lag", GARCH11_SOURCE)
        @test occursin("pointwise = plate(", GARCH11_SOURCE)
        @test !occursin("struct ", GARCH11_SOURCE)
    end

    @testset "density decomposition vs the independent reference oracle" begin
        p = prepare(model; have = (:unconstrained, :y, :sigma1),
            want = (:sigma, :log_jacobian, :pointwise, :likelihood, :posterior),
            bound = (; y, sigma1))
        sigma, log_jacobian, pointwise, likelihood, posterior = p(q)
        ref = _garch11_reference(q, y, sigma1)
        @test length(pointwise) == length(y)
        @test all(isfinite, pointwise)
        @test sigma ≈ ref.sigma
        @test sigma[1] == sigma1
        @test all(>(0.0), sigma)
        @test likelihood ≈ ref.likelihood
        @test likelihood ≈ sum(pointwise)
        @test log_jacobian ≈ ref.log_jacobian
        @test posterior ≈ ref.posterior
    end

    @testset "native primal + plain-Enzyme gradient vs finite differences" begin
        pk = prepare(model; have = (:unconstrained, :y, :sigma1), want = :posterior,
            bound = (; y, sigma1))
        @test pk(q) ≈ _garch11_reference(q, y, sigma1).posterior
        prep = prepare_ad(pk, _GARCH_AE, q; active = :unconstrained)
        g = ReactiveKernels.ad_value_and_gradient!(prep, similar(q), q)[2]
        @test all(isfinite, g)
        gfd = _garch_central_fd(qq -> _garch11_reference(qq, y, sigma1).posterior, q)
        @test g ≈ gfd rtol = 1e-5
        # repeated use returns the same value
        @test pk(q) == pk(q)
    end

    @testset "the summed likelihood equals the pointwise total" begin
        lk = prepare(model; have = (:unconstrained, :y, :sigma1), want = :likelihood,
            bound = (; y, sigma1))
        pw = prepare(model; have = (:unconstrained, :y, :sigma1), want = :pointwise,
            bound = (; y, sigma1))
        @test lk(q) ≈ sum(pw(q))
        @test length(pw(q)) == length(y)
    end

    @testset "one-step-ahead volatility forecast from a constrained boundary" begin
        # `parameters` is an authoritative HAVE boundary (inverse edges); the
        # forecast reruns the recursion off the constrained parameters.
        ref = _garch11_reference(q, y, sigma1)
        params = (; mu = q[1], alpha0 = exp(q[2]),
                  alpha1 = _garch_logistic(q[3]),
                  beta1 = (1 - _garch_logistic(q[3])) * _garch_logistic(q[4]))
        fk = prepare(model; have = (:parameters, :y, :sigma1), want = :forecast_sigma,
            bound = (; y, sigma1))
        expected = sqrt(params.alpha0 + params.alpha1 * (y[end] - params.mu)^2 +
                        params.beta1 * ref.sigma[end]^2)
        @test fk(params) ≈ expected
    end

    @testset "data-generic: an alternate short series lowers and matches the oracle" begin
        y_short = y[1:24]
        s1 = 0.7
        q2 = [-0.1, log(0.2), 0.1, 0.4]
        pk = prepare(model; have = (:unconstrained, :y, :sigma1), want = :posterior)
        @test pk(q2, y_short, s1) ≈ _garch11_reference(q2, y_short, s1).posterior
    end
end
