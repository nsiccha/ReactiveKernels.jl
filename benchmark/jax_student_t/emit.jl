using ReactiveKernels, ReactiveKernelsPPL, Random
length(ARGS) == 1 || error("usage: emit.jl OUTPUT_DIRECTORY")
const OUTPUT = abspath(only(ARGS))
mkpath(OUTPUT)
isdefined(@__MODULE__, :rk_jax_evaluate) || include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))
const RK = ReactiveKernels
const BASELINE = Ref(false)
const SOURCES = Dict{String,String}()
# Diagnostic baseline: ordinary AD of the exact same value-only recurrence.
# This dispatch exists only in this benchmark process.
function RK._jax_call_positional(ctx::RK._JaxContext,
        rule::typeof(RK._rk_beta_inc_cf), args::Vector{RK._PyVal})
    BASELINE[] ? RK._jax_traceable_call(ctx, RK._beta_cf_value, args) :
                 RK._jax_scalar_rule(ctx, rule, args)
end
const TOY = @rkppl begin
    mu ~ Normal(0, prior_scale)
    sigma ~ Exponential(sigma_prior)
    nu ~ Gamma(nu_shape, nu_scale)
    y_obs .~ StudentT.(nu, mu, sigma .* ones_obs)
    y_cens .~ censored.(StudentT.(nu, mu, sigma .* ones_cens), limit, Inf)
end
for n in (100,1000)
    limit = -0.5
    # The reporter supplied these seeds and the exact emitted input bundle.
    y = 0.3 .+ 0.8 .* randn(Xoshiro(n), n)
    observed = y[y .> limit]
    censored_values = fill(limit, count(<=(limit),y))
    data = (;y_obs=observed, y_cens=censored_values, limit,
        ones_obs=ones(length(observed)), ones_cens=ones(length(censored_values)),
        prior_scale=2.0,sigma_prior=1.0,nu_shape=2.0,nu_scale=5.0)
    model_data = Base.structdiff(data, (; y_obs=observed, y_cens=censored_values))
    plan = TOY(;model_data...) | (;y_obs=observed,y_cens=censored_values)
    built = build_kernel(plan)
    query = prepare_query(built, plan, :sampler)
    u = 0.1 .* randn(Xoshiro(1), built.layout.total)
    println("N=$n censored=$(length(censored_values)) native=$(query(u))")
    for baseline in (true,false)
        BASELINE[] = baseline
        source = jax_source(built,plan)
        tag = baseline ? "baseline" : "candidate"
        path = joinpath(OUTPUT, "$(tag)_$(n).py")
        write(path, source)
        preparation = only(f for f in source.functions if f.name == "prepare_data")
        raw = [getproperty(data, Symbol(name)) for (name, _) in preparation.inputs]
        write(joinpath(OUTPUT, "data_$(n).py"),
            "NAMES = [" * join((repr(name) for (name, _) in preparation.inputs), ", ") * "]\n" *
            "RAW = [" * join(map(_rk_py_input, raw), ", ") * "]\n" *
            "POINT = np.asarray(" * _rk_py(u) * ")\n" *
            "NATIVE = " * repr(query(u)) * "\n")
        text = string(source)
        if haskey(SOURCES, tag)
            @assert text == SOURCES[tag] "emitted source changed with observation count"
        else
            SOURCES[tag] = text
        end
        lines = count('\n', text)
        loops = count("jax.lax.fori_loop(",text)
        conditions = count("jax.lax.cond(",text)
        println("$tag lines=$lines loops=$loops cond=$conditions")
    end
end
