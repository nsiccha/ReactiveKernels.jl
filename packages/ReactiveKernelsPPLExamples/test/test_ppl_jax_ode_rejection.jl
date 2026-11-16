# The JAX module of an `@rkppl` model with an ODE (snag
# rk-ode-solve-fai-3c664466): where the solve fails, the native sampler target
# answers -Inf, and so do the module's density functions, which a NumPyro or
# BlackJAX sampler then rejects. Functions that do not solve the ODE are
# unaffected. Needs the ODE solvers, so it runs with the examples.
using ReactiveKernels, ReactiveKernelsPPL, Test, Enzyme
using DifferentiationInterface: AutoEnzyme
isdefined(@__MODULE__, :rk_jax_evaluate) ||
    include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

# dy/dt = k y^2 from y0 blows up at t = 1 / (k y0).
_ppl_ode_blowup(t, y, k) = k .* y .^ 2
_ppl_ode_last(k, y0, ts) =
    rk_ode_rk45_tol(_ppl_ode_blowup, [y0], 0.0, ts, 1e-8, 1e-8, 10^4, k)[end, 1]

const PPL_ODE_REJECTION = @rkppl begin
    logk ~ Normal(-2, 1)
    sigma ~ Exponential(1)
    @plate for i in eachindex(y)
        y[i] ~ Normal(_ppl_ode_last(exp(logk), y0[i], ts), sigma)
    end
end

@testset "RKPPL's JAX density functions answer -inf where its native target does" begin
    raw = (; y0 = [0.2, 0.3, 0.25], ts = [0.5, 1.0, 2.0], y = [0.21, 0.32, 0.27])
    plan = PPL_ODE_REJECTION(; Base.structdiff(raw, (; y = raw.y))...) | (; raw.y)
    built = build_kernel(plan)
    names = [e.name for e in built.layout.entries]
    ik = built.layout.entries[findfirst(==(:logk), names)].offset
    # At k = exp(1.5) the solve from 0.2 blows up at t = 1.12, before t = 2.
    good = zeros(built.layout.total); good[ik] = -2.0
    bad = copy(good); bad[ik] = 1.5
    q = prepare_sampler(built, plan, good; backend = AutoEnzyme(; mode = Enzyme.Reverse))
    g = zeros(built.layout.total)
    value, _ = sampler_value_and_gradient!(q, g, good)
    @test isfinite(value) && all(isfinite, g)
    @test q(bad) == -Inf
    src = jax_source(built, plan)
    @test occursin("checkify.check(", string(src))
    if rk_jax_available() && success(`$RK_JAX_PYTHON -c "import diffrax"`)
        inputs = first.(only(f for f in src.functions if f.name == "prepare_data").inputs)
        args = Tuple(getproperty(raw, Symbol(name)) for name in inputs)
        got = rk_jax_evaluate(src, :log_density, (good, args...); grad = true, prepare = :arguments)
        @test got.value ≈ value rtol = 1e-6
        @test got.grad ≈ g rtol = 1e-4
        @test rk_jax_evaluate(src, :log_density, (bad, args...); prepare = :arguments).value == -Inf
        @test rk_jax_evaluate(src, :log_likelihood, (bad, args...); prepare = :arguments).value == -Inf
        # The prior and the constrained values do not solve the ODE.
        prior = prepare_query(built, plan, :prior)
        @test rk_jax_evaluate(src, :log_prior, (bad, args...); prepare = :arguments).value ≈
              prior(bad) rtol = 1e-10
        @test rk_jax_evaluate(src, :constrain, (bad, args...); prepare = :arguments).value["logk"] ≈ 1.5
    else
        @test_skip false
    end
end
