using ReactiveKernelsPPLExamples.ProphetExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal, laplace

# Graph-independent reference oracle: recomputes the prophet (LINEAR trend)
# density from first principles, matching the Stan model block (propto = false,
# jacobian = true).
_pr_normal(x, mu, sigma) = -0.5 * log(2π) - log(sigma) - 0.5 * ((x - mu) / sigma)^2
_pr_laplace(x, mu, b) = -log(2 * b) - abs(x - mu) / b

function _prophet_reference(q, t, t_change, X, sigmas, tau, s_a, s_m, y)
    S = length(t_change); K = size(X, 2)
    k = q[1]; m = q[2]
    delta = q[3:(2 + S)]
    log_sigma_obs = q[3 + S]
    beta = q[(4 + S):(3 + S + K)]
    sigma_obs = exp(log_sigma_obs)
    log_jacobian = log_sigma_obs
    A = Float64.(t .>= t_change')                     # T×S changepoint incidence
    trend = (k .+ A * delta) .* t .+ (m .- A * (t_change .* delta))
    mean_response = trend .* (1 .+ X * (beta .* s_m)) .+ X * (beta .* s_a)
    prior = _pr_normal(k, 0, 5) + _pr_normal(m, 0, 5) +
            sum(_pr_laplace(d, 0, tau) for d in delta) +
            _pr_normal(sigma_obs, 0, 0.5) +
            sum(_pr_normal(beta[j], 0, sigmas[j]) for j in 1:K)
    likelihood = sum(_pr_normal(y[i], mean_response[i], sigma_obs) for i in 1:length(y))
    (; prior, likelihood, log_jacobian, posterior = prior + likelihood + log_jacobian)
end

@testset "PPL graph — prophet (linear trend)" begin
    artifact = evaluate_prophet_source()
    @test artifact.source == strip(PROPHET_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    have = (:unconstrained, :t, :t_change, :X, :sigmas, :tau, :s_a, :s_m, :y)
    data = (PROPHET_T, PROPHET_T_CHANGE, PROPHET_X, PROPHET_SIGMAS,
            PROPHET_TAU, PROPHET_S_A, PROPHET_S_M, PROPHET_Y)
    S = length(PROPHET_T_CHANGE); K = size(PROPHET_X, 2)
    dim = 3 + S + K

    @testset "authored on the reusable distribution-object surface" begin
        @test occursin("laplace(0.0, b).logpdf", PROPHET_SOURCE)
        @test occursin("(t .>= t_change') .* 1.0", PROPHET_SOURCE)
        @test occursin("plate(y, mean_response, sigma_obs)", PROPHET_SOURCE)
        @test !occursin("struct ", PROPHET_SOURCE)
        # The dishonest -Inf sentinel gate must NOT be present.
        @test !occursin("-Inf", PROPHET_SOURCE)
        @test !occursin("trend_indicator", PROPHET_SOURCE)
        # The natural logistic recurrence is preserved as a reference source.
        @test occursin("logistic_gamma", PROPHET_LOGISTIC_TREND_REFERENCE_SOURCE)
        @test occursin("k_s[i] / k_s[i+1]", PROPHET_LOGISTIC_TREND_REFERENCE_SOURCE)
        @test artifact.normal_object === normal
        @test artifact.laplace_object === laplace
        raw_generated = code_expr(artifact.kernel)
        readable = sprint(
            Base.show_unquoted,
            ReactiveKernels._readable_expr(raw_generated, artifact.kernel);
            context = :limit => false,
        )
        @test !occursin(r"__ops__\[\d+\]", readable)
        @test !occursin(r"\boperation\(", readable)
    end

    @testset "density decomposition vs the independent reference oracle" begin
        kernel = prepare(model; have = have,
            want = (:prior, :likelihood, :log_jacobian, :posterior))
        for q in ([zeros(dim)],
                  [0.02 .* collect(1:dim) .- 0.05],
                  [0.1 .* cos.(0.4 .* (1:dim))])
            qi = q[1]
            prior, likelihood, log_jacobian, posterior = kernel(qi, data...)
            ref = _prophet_reference(qi, data...)
            @test isfinite(posterior)
            @test prior ≈ ref.prior
            @test likelihood ≈ ref.likelihood
            @test log_jacobian ≈ ref.log_jacobian
            @test posterior ≈ ref.posterior
        end
    end

    @testset "linear-only capability: the trend flag is validated, not silently ignored" begin
        # The authoritative data is linear; build validates the supported mode.
        @test build_prophet_graph() isa ReactiveKernels.KernelSpec
        @test build_prophet_graph(; trend_indicator = 0) isa ReactiveKernels.KernelSpec
        # The logistic trend is a real (finite-density) Stan mode that is NOT
        # implemented — it must throw an explicit unsupported-mode error, never a
        # wrong or sentinel density.
        @test_throws ArgumentError build_prophet_graph(; trend_indicator = 1)
        @test_throws ArgumentError build_prophet_graph(; trend_indicator = 2)
    end

    @testset "wrong behavior is detectable — perturbing q moves the density" begin
        kernel = prepare(model; have = have, want = :posterior)
        q = zeros(dim)
        q2 = copy(q); q2[1] = 0.8   # bump k (base growth rate)
        @test kernel(q, data...) != kernel(q2, data...)
    end

    @testset "data-generic: alternate horizon / changepoints / regressors" begin
        # Tiny synthetic linear-trend problem (T = 5, S = 2, K = 2).
        t = [0.0, 0.25, 0.5, 0.75, 1.0]
        t_change = [0.3, 0.7]
        X = [0.1 0.4; -0.2 0.3; 0.5 -0.1; 0.0 0.2; -0.3 0.1]
        sigmas = [1.0, 0.5]
        tau = 0.05
        s_a = [1.0, 0.0]
        s_m = [0.0, 1.0]
        y = [0.2, -0.1, 0.4, 0.05, -0.3]
        small_dim = 3 + length(t_change) + size(X, 2)
        q = collect(range(-0.2, 0.2; length = small_dim))
        small = (t, t_change, X, sigmas, tau, s_a, s_m, y)
        post = prepare(model; have = have, want = :posterior)(q, small...)
        @test post ≈ _prophet_reference(q, small...).posterior
    end
end
