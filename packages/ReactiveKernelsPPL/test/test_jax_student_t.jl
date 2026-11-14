using ReactiveKernels, ReactiveKernelsPPL, Random, Test
using DifferentiationInterface: AutoEnzyme, gradient
using ReactiveKernelsDistributionKernels.DistributionKernelSources: _student_log_tail
using SpecialFunctions: logbeta
import Enzyme
isdefined(@__MODULE__, :rk_jax_evaluate) ||
    include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

const _JAX_STUDENT_TOY = @rkppl begin
    mu ~ Normal(0, prior_scale)
    sigma ~ Exponential(sigma_prior)
    nu ~ Gamma(nu_shape, nu_scale)
    y_obs .~ StudentT.(nu, mu, sigma .* ones_obs)
    y_cens .~ censored.(StudentT.(nu, mu, sigma .* ones_cens), limit, Inf)
end

@kernel jax_student_logtails(u::Vector{Float64}, z::Vector{Float64}) = begin
    nu = u[1]
    log_beta = logbeta(nu / 2, 0.5)
    values = plate(z) do zi
        q = (zi - u[2]) / exp(u[3])
        _student_log_tail(nu, q, log_beta, false) +
            0.3 * _student_log_tail(nu, q, log_beta, true)
    end
    total = sum(values)
    return total
end

@testset "Student-t JAX log tails retain centre and extreme-tail derivatives" begin
    source = jax_source(jax_student_logtails; have=(:u,:z), want=:total)
    native = prepare(jax_student_logtails; have=(:u,:z), want=:total)
    z = [-1e30, -37.0, -4.0, -1e-7, 0.0, 1e-7, 0.7, 41.0, 1e30]
    # One loop: the fraction's value and partials share `_beta_cf_math`.
    @test count("jax.lax.fori_loop(", string(source)) == 1
    if rk_jax_available()
        backend = AutoEnzyme(; mode=Enzyme.Reverse)
        for nu in (0.5, 1.0, 4.0, 100.0)
            u = [nu, 0.0, 0.0]
            query = v -> native(v, z)
            got = rk_jax_evaluate(source, "kernel", (u, z); grad=true)
            @test got.value ≈ query(u) rtol=1e-11
            @test all(isfinite, got.grad)
            @test got.grad ≈ gradient(query, backend, u) rtol=2e-9 atol=1e-10
            @test got.grad ≈ rk_fd_gradient(query, u) rtol=2e-5 atol=2e-7
        end
    else
        @test_skip false
    end
end

function jax_student_case(n)
    limit = -0.5
    y = 0.3 .+ 0.8 .* randn(Xoshiro(613), n)
    y_obs = y[y .> limit]
    y_cens = fill(limit, count(<=(limit), y))
    data = (; ones_obs=ones(length(y_obs)), ones_cens=ones(length(y_cens)),
             limit, prior_scale=2.0, sigma_prior=1.0, nu_shape=2.0, nu_scale=5.0)
    plan = _JAX_STUDENT_TOY(; data...) | (; y_obs, y_cens)
    return plan, merge(data, (; y_obs, y_cens))
end

@testset "censored Student-t JAX module shares its fraction across data sizes" begin
    cases = jax_student_case.((0, 1, 17, 100))
    built = [build_kernel(plan) for (plan, _) in cases]
    sources = [jax_source(b, plan) for (b, (plan, _)) in zip(built, cases)]
    @test all(==(string(first(sources))), string.(sources))
    # One loop, shared by the fraction's value and partials, independent of
    # observation count and the lower/upper tail call sites in a censored law.
    @test count("jax.lax.fori_loop(", string(first(sources))) == 1
    if rk_jax_available()
        source = first(sources)
        inputs = first.(only(f for f in source.functions if f.name == "prepare_data").inputs)
        backend = AutoEnzyme(; mode=Enzyme.Reverse)
        for (b, (plan, data)) in zip(built, cases)
            query = prepare_query(b, plan, :sampler)
            u = unconstrain(b.layout, (; mu=0.1, sigma=0.9, nu=5.0))
            raw = Tuple(getproperty(data, Symbol(name)) for name in inputs)
            got = rk_jax_evaluate(source, "log_density", (u, raw...);
                                  grad=true, prepare=:arguments)
            @test got.value ≈ query(u) rtol=1e-11
            @test got.grad ≈ gradient(query, backend, u) rtol=2e-9 atol=1e-10
            @test got.grad ≈ rk_fd_gradient(query, u) rtol=2e-5 atol=2e-7
        end
    else
        @test_skip false
    end
end

const _JAX_STUDENT_THRESHOLD = @rkppl begin
    location ~ Normal(location_prior, location_scale)
    scale ~ Exponential(scale_prior)
    nu ~ Gamma(df_shape, df_scale)
    y .~ censored.(StudentT.(nu, location, scale), threshold, Inf)
end

@testset "censored Student-t with a per-observation threshold emits JAX" begin
    # The lower tail's beta fraction is a derivative rule whose module-level
    # functions run under the censoring branch; from RK 0.1.233 until those
    # functions were translated without their caller's condition, this
    # raised "a function's active condition is outside its caller's scope".
    priors = (; location_prior=0.0, location_scale=2.0, scale_prior=1.0,
              df_shape=2.0, df_scale=2.0)
    # Both observations at their thresholds; then one above its threshold
    # between two at theirs, so both arms of the censoring branch run.
    for (threshold, y) in (([-0.5, 0.25], [-0.5, 0.25]),
                           ([-0.5, 0.25, 0.0], [-0.5, 1.3, 0.0]))
        plan = _JAX_STUDENT_THRESHOLD(; priors..., threshold) | (; y)
        built = build_kernel(plan)
        source = jax_source(built, plan)
        @test occursin("UNTRANSLATED = {}", string(source))
        rk_jax_available() || (@test_skip false; continue)
        data = merge(priors, (; threshold, y))
        inputs = first.(only(f for f in source.functions if f.name == "prepare_data").inputs)
        raw = Tuple(getproperty(data, Symbol(name)) for name in inputs)
        query = prepare_query(built, plan, :sampler)
        for point in ((; location=0.1, scale=0.9, nu=5.0),
                      (; location=-0.3, scale=1.7, nu=0.8),
                      (; location=0.0, scale=0.4, nu=60.0))
            u = unconstrain(built.layout, point)
            got = rk_jax_evaluate(source, "log_density", (u, raw...);
                                  grad=true, prepare=:arguments)
            @test got.value ≈ query(u) rtol=1e-11
            @test got.grad ≈ rk_fd_gradient(query, u) rtol=2e-5 atol=2e-7
        end
    end
end
