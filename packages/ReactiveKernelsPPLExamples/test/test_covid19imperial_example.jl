using ReactiveKernels
using ReactiveKernelsPPLExamples
using ReactiveKernelsPPLExamples.Covid19ImperialExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, gamma, exponential
using DifferentiationInterface: AutoEnzyme
using Enzyme
using SpecialFunctions: loggamma
using LogExpFunctions: xlogy
using Random

# Graph-independent oracle for the Imperial covid19 renewal model, written
# directly from the shipped `.stan` semantics and parameterized by the RAW
# posteriordb arrays (the same arrays the graph binds as ports). Includes
# Stan's day-1 special case E_deaths[1,m] = 1e-15 * prediction[1,m] (NOT
# multiplied by ifr_noise) and supports observation ranges that start inside
# the imputation window 1:N0.
function _covid_reference(q, X, EpidemicStart, N, deaths, SI, fmat, pop,
                          M, P, N0, N2)
    normlp(x, mu, s) = -0.5 * log(2π) - log(s) - 0.5 * ((x - mu) / s)^2
    gammalp(x, a, b) = a * log(b) - loggamma(a) + (a - 1) * log(x) - b * x
    nb2lp(y, mu, phi) = loggamma(y + phi) - loggamma(phi) - loggamma(y + 1) +
                        phi * log(phi) - phi * log(phi + mu) + xlogy(y, mu) -
                        y * log(phi + mu)
    mu = exp.(q[1:M]); alpha_hier = exp.(q[M+1:M+P]); kappa = exp(q[M+P+1])
    y = exp.(q[M+P+2:M+P+1+M]); phi = exp(q[2M+P+2]); tau = exp(q[2M+P+3])
    ifr = exp.(q[2M+P+4:2M+P+3+M]); logjac = sum(q)
    alpha = alpha_hier .- log(1.05) / 6
    prior = log(0.03) - 0.03 * tau
    prior += sum(-log(tau) .- y ./ tau)
    prior += normlp(phi, 0.0, 5.0) + normlp(kappa, 0.0, 0.5)
    prior += sum(normlp(mu[m], 3.28, kappa) for m in 1:M)
    prior += sum(gammalp(alpha_hier[i], 0.1667, 1.0) for i in 1:P)
    prior += sum(normlp(ifr[m], 1.0, 0.1) for m in 1:M)
    loglik = 0.0
    for m in 1:M
        pred = zeros(N2); pred[1:N0] .= y[m]; cumm = zeros(N2)
        for i in 2:N0; cumm[i] = cumm[i-1] + pred[i]; end
        Rt = mu[m] .* exp.(-(X[m, :, :] * alpha))
        for i in N0+1:N2
            conv = 0.0; for j in 1:i-1; conv += pred[j] * SI[i-j]; end
            cumm[i] = cumm[i-1] + pred[i-1]
            pred[i] = ((pop[m] - cumm[i]) / pop[m]) * Rt[i] * conv
        end
        for i in EpidemicStart[m]:N[m]
            if i == 1
                ed = 1e-15 * pred[1]           # Stan's day-1 special case
            else
                ed = 0.0; for j in 1:i-1; ed += pred[j] * fmat[i-j, m]; end
                ed *= ifr[m]
            end
            loglik += nb2lp(deaths[i, m], ed, phi)
        end
    end
    (; prior, logjac, loglik, posterior = prior + logjac + loglik)
end

_covid_have() = (:unconstrained, :X, :EpidemicStart, :N, :deaths, :SI, :fmat,
                 :pop, :M, :P, :N0, :N2)
_covid_bind() = (; X = COVID19IMPERIAL_X,
                 EpidemicStart = COVID19IMPERIAL_EPIDEMICSTART,
                 N = COVID19IMPERIAL_N, deaths = COVID19IMPERIAL_DEATHS,
                 SI = COVID19IMPERIAL_SI, fmat = COVID19IMPERIAL_F,
                 pop = COVID19IMPERIAL_POP, M = COVID19IMPERIAL_M,
                 P = COVID19IMPERIAL_P, N0 = COVID19IMPERIAL_N0,
                 N2 = COVID19IMPERIAL_N2)

# Ordinary reverse-mode AD (no function_annotation substitution), per the
# standing acceptance rule for anything consumed as ReactiveKernels.
const _COVID_AE_ORDINARY = AutoEnzyme(mode = Enzyme.Reverse)

@testset "PPL graph — covid19imperial (posteriordb)" begin
    # The demo-tail sentinel is process-global. Snapshot it so this test is
    # valid standalone and after earlier hosted example-package tests; only
    # this test's full source evaluation should advance the delta by one.
    demo_tail_before = ReactiveKernelsPPLExamples._DEMO_TAIL_EXECUTIONS[]
    template_kernel = prepare(build_covid19imperial_graph();
        have = _covid_have(), want = (:prior, :log_jacobian, :loglik, :posterior),
        bound = _covid_bind())
    dim = 3 * COVID19IMPERIAL_M + COVID19IMPERIAL_P + 3
    @test dim == 51
    q0 = zeros(dim)
    cold = template_kernel(q0)
    @test isfinite(cold[4])

    artifact = evaluate_covid19imperial_source()
    @test ReactiveKernelsPPLExamples._DEMO_TAIL_EXECUTIONS[] == demo_tail_before + 1
    @test artifact.source == strip(COVID19IMPERIAL_SOURCE, '\n')
    @test artifact.output == Base.invokelatest(artifact.kernel, artifact.inputs.q)
    model = artifact.model

    @testset "authored on the natural source surface" begin
        # The renewal recurrence is one batched scan over i = 2:N2; day 1 has
        # its own node; all raw-data preprocessing is in-graph; the
        # exponential scale convention is explicit.
        @test occursin("scan(eachrow(xs)", COVID19IMPERIAL_SOURCE) # xs is the pure param-derived renewal block
        @test occursin("1e-15 .* y", COVID19IMPERIAL_SOURCE)
        @test occursin("observed_mask = (icol .>= ESrow) .& (icol .<= Nrow)",
                       COVID19IMPERIAL_SOURCE)
        @test occursin("XX = reshape(permutedims(X, (2, 1, 3))", COVID19IMPERIAL_SOURCE)
        @test occursin("exponential(1 / 0.03).logpdf(tau)", COVID19IMPERIAL_SOURCE)
        @test occursin("edmat = vcat(permutedims(1e-15 .* y), ed_early,", COVID19IMPERIAL_SOURCE)
        @test !occursin("struct ", COVID19IMPERIAL_SOURCE)
        @test artifact.exponential_object === exponential
        @test artifact.gamma_object === gamma
        @test artifact.normal_object === normal
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        for q in (q0, 0.1 .* [(-1)^i for i in 1:dim])
            reference = _covid_reference(q, COVID19IMPERIAL_X,
                COVID19IMPERIAL_EPIDEMICSTART, COVID19IMPERIAL_N,
                COVID19IMPERIAL_DEATHS, COVID19IMPERIAL_SI, COVID19IMPERIAL_F,
                COVID19IMPERIAL_POP, COVID19IMPERIAL_M, COVID19IMPERIAL_P,
                COVID19IMPERIAL_N0, COVID19IMPERIAL_N2)
            prior, log_jacobian, loglik, posterior = template_kernel(q)
            @test isfinite(posterior)
            @test prior ≈ reference.prior
            @test log_jacobian ≈ reference.logjac
            @test loglik ≈ reference.loglik
            @test posterior ≈ reference.posterior
        end
    end

    @testset "model_only template ≡ full source graph (native equality)" begin
        gfu = compose(evaluate_covid19imperial_source().model)
        k(g) = prepare(g; have = _covid_have(), want = :posterior,
                       bound = _covid_bind())
        @test isfinite(k(gfu)(q0))
        @test template_kernel(q0)[4] == k(gfu)(q0)
    end

    # Small-shape controls: the SAME raw ports bind completely different
    # (M, P, N0, N2) shapes, including observation windows that start inside
    # the imputation days and a dataset with no renewal tail at all.
    function small_case(; M, P, N0, N2, ES, Nlast, frac = false, q = nothing)
        X = frac ? reshape(collect(0.5:(0.5 + M * N2 * P - 1)) .* 0.25, M, N2, P) :
                   reshape(collect(Float64, 1:(M * N2 * P)), M, N2, P)
        pop = frac ? [1000.5, 2000.25][1:M] : Float64[1000.0, 2000.0][1:M]
        deaths = fill(2, N2, M)
        SI = collect(1.0:N2) ./ sum(1.0:N2)
        f = reshape(collect(1.0:(N2 * M)) ./ (N2 * M), N2, M)
        d = 3M + P + 3
        q === nothing && (q = 0.25 .* [(-1)^i for i in 1:d])
        bound = (; X = X, EpidemicStart = fill(ES, M), N = fill(Nlast, M),
                 deaths = deaths, SI = SI, fmat = f, pop = pop,
                 M = M, P = P, N0 = N0, N2 = N2)
        ref = _covid_reference(q, X, fill(ES, M), fill(Nlast, M), deaths, SI,
                               f, pop, M, P, N0, N2)
        kernel = prepare(build_covid19imperial_graph();
            have = _covid_have(),
            want = (:prior, :log_jacobian, :loglik, :posterior), bound = bound)
        (; kernel, q, ref, bound, d)
    end

    @testset "early observations include the imputation days (M=2, P=1)" begin
        c = small_case(; M = 2, P = 1, N0 = 3, N2 = 8, ES = 2, Nlast = 8)
        prior, log_jacobian, loglik, posterior = c.kernel(c.q)
        @test isfinite(posterior)
        @test prior ≈ c.ref.prior
        @test log_jacobian ≈ c.ref.logjac
        @test loglik ≈ c.ref.loglik
        @test posterior ≈ c.ref.posterior
    end

    @testset "day-1 observation uses Stan's 1e-15 special case (M=1)" begin
        c = small_case(; M = 1, P = 1, N0 = 3, N2 = 6, ES = 1, Nlast = 6)
        prior, log_jacobian, loglik, posterior = c.kernel(c.q)
        @test isfinite(posterior)
        @test loglik ≈ c.ref.loglik
        @test posterior ≈ c.ref.posterior
    end

    @testset "empty renewal tail N2 = N0 (M=1)" begin
        c = small_case(; M = 1, P = 1, N0 = 3, N2 = 3, ES = 1, Nlast = 3)
        prior, log_jacobian, loglik, posterior = c.kernel(c.q)
        @test isfinite(posterior)
        @test prior ≈ c.ref.prior
        @test log_jacobian ≈ c.ref.logjac
        @test loglik ≈ c.ref.loglik
        @test posterior ≈ c.ref.posterior
    end

    @testset "fractional raw X and pop bind through the REAL ports" begin
        c = small_case(; M = 2, P = 1, N0 = 3, N2 = 8, ES = 4, Nlast = 8,
                       frac = true)
        prior, log_jacobian, loglik, posterior = c.kernel(c.q)
        @test isfinite(posterior)
        @test prior ≈ c.ref.prior
        @test loglik ≈ c.ref.loglik
        @test posterior ≈ c.ref.posterior
    end

    @testset "nonpositive forecast E_deaths stays masked and finite" begin
        # Susceptible depletion drives a late, UNOBSERVED step's prediction
        # (hence E_deaths) negative; the oracle — like Stan, which never
        # evaluates unobserved cells — stays finite, and the masked whole-grid
        # graph must match it.
        q = [log(100.0), 0.0, 0.0, log(5.0), 0.0, 0.0, 0.0]
        c = small_case(; M = 1, P = 1, N0 = 3, N2 = 8, ES = 4, Nlast = 4, q = q)
        @test isfinite(c.ref.posterior)
        prior, log_jacobian, loglik, posterior = c.kernel(q)
        @test isfinite(posterior)
        @test loglik ≈ c.ref.loglik
        @test posterior ≈ c.ref.posterior
    end

    @testset "ordinary reverse-mode gradient (no annotation substitution)" begin
        # Bundled data and one boundary shape: the ordinary AutoEnzyme Reverse
        # gradient is finite and agrees with manual central differences.
        full_values = prepare(build_covid19imperial_graph();
            have = _covid_have(),
            want = (:prior, :log_jacobian, :loglik, :posterior),
            bound = _covid_bind())
        # AD preparation needs a single selected WANT port.
        full_scalar = prepare(build_covid19imperial_graph();
            have = _covid_have(), want = :posterior, bound = _covid_bind())
        small = small_case(; M = 1, P = 1, N0 = 3, N2 = 6, ES = 1, Nlast = 6)
        small_scalar = prepare(build_covid19imperial_graph();
            have = _covid_have(), want = :posterior, bound = small.bound)
        for (ad_kernel, value_kernel, q) in (
                (full_scalar, full_values, 0.05 .* [(-1)^i for i in 1:dim]),
                (small_scalar, small.kernel, 0.2 .* [(-1)^i for i in 1:7]))
            prep = prepare_ad(ad_kernel, _COVID_AE_ORDINARY, q; active = :unconstrained)
            v, g = ReactiveKernels.ad_value_and_gradient(prep, q)
            @test isfinite(v)
            @test all(isfinite, g)
            h = 1e-6
            for i in (1, 2, 4)
                qp = copy(q); qp[i] += h
                qm = copy(q); qm[i] -= h
                fd = (value_kernel(qp)[4] - value_kernel(qm)[4]) / (2h)
                @test isapprox(g[i], fd; rtol = 1e-4, atol = 1e-6)
            end
        end
    end
end
