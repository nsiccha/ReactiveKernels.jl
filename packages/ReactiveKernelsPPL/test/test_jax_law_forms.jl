using ReactiveKernels, ReactiveKernelsPPL, Test
include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

# A distribution law spliced into a model reads as authored code does (snag
# jax-source-on-rk-d32d60bd): a `censored(Normal)` law's arms that are one
# expression are `lambda:`, its single-use values stay expressions under the
# law's own names (`logcdf_z = jax.lax.cond(...)`), and nothing binds a
# `value_N` or a `let` result. A literal prior location is read as the
# literal (`amp != 0.0`), not bound as `location_binding = 0.0`.

const jax_law_censored_model = @rkppl begin
    "Mean."
    mu ~ Normal(mu_loc, mu_scale)
    "Noise SD."
    noise ~ Exponential(noise_scale)
    "Observations, left-censored at `lower`."
    y .~ censored.(Normal.(mu, noise), lower, Inf)
end

const jax_law_literal_model = @rkppl begin
    "Mean."
    mu ~ Normal(mu_loc, mu_scale)
    "Amplitude."
    amp ~ Normal(0, amp_scale)
    "Noise SD."
    noise ~ Exponential(noise_scale)
    m = mu .+ amp .* x
    y .~ Normal.(m, noise)
end

function jax_law_check(src, built, plan, raw)
    names = first.(only(f for f in src.functions if f.name == "prepare_data").inputs)
    args = Tuple(getproperty(raw, Symbol(name)) for name in names)
    u = collect(range(-0.2, 0.3; length = built.layout.total))
    native = prepare_query(built, plan, :sampler)
    got = rk_jax_evaluate(src, :log_density, (u, args...); grad = true, prepare = :arguments)
    @test got.value ≈ native(u) atol = 1e-11
    finite_difference = [(native(u + 1e-5 .* (1:length(u) .== i)) -
                          native(u - 1e-5 .* (1:length(u) .== i))) / 2e-5
                         for i in eachindex(u)]
    @test got.grad ≈ finite_difference rtol = 2e-6 atol = 1e-7
end

@testset "a spliced censored law keeps the forms of authored code" begin
    texts = String[]
    for n in (0, 3, 19)
        y = collect(range(-1.0, 1.0; length = n))
        # Uncensored, censored at the observation, and below it.
        lower = [k % 3 == 0 ? -Inf : k % 3 == 1 ? y[k] : y[k] - 0.5 for k in 1:n]
        raw = (; mu_loc = 0.0, mu_scale = 1.0, noise_scale = 1.0, lower, y)
        plan = jax_law_censored_model(; Base.structdiff(raw, (; y = raw.y))...) | (; raw.y)
        built = build_kernel(plan)
        src = jax_source(built, plan)
        text = src.text
        @test !occursin(r"(?m)^\s*(value|let_value)(_\d+)? = ", text)
        @test occursin(r"logcdf_z\w* = jax\.lax\.cond\([^\n]*, lambda: jnp\.ones_like\(", text)
        @test occursin("lambda: jax.lax.cond(lower", text)
        n > 0 && push!(texts, text)
        rk_jax_available() ? jax_law_check(src, built, plan, raw) : @test_skip false
    end
    # One module serves every data length: nothing is replicated per entry.
    @test allequal(texts)
end

@testset "a literal prior location is read as the literal" begin
    texts = String[]
    for n in (0, 3, 19)
        x = collect(range(0.0, 2.0; length = n))
        raw = (; mu_loc = 0.0, mu_scale = 1.0, amp_scale = 1.5, noise_scale = 1.0, x,
               y = 0.3 .* x .- 0.1)
        plan = jax_law_literal_model(; Base.structdiff(raw, (; y = raw.y))...) | (; raw.y)
        built = build_kernel(plan)
        src = jax_source(built, plan)
        text = src.text
        @test !occursin("location_binding", text)
        @test occursin("amp != 0.0", text)
        n > 0 && push!(texts, text)
        rk_jax_available() ? jax_law_check(src, built, plan, raw) : @test_skip false
    end
    @test allequal(texts)
end
