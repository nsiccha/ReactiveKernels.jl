using ReactiveKernels, ReactiveKernelsPPL, Random, Test

include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

# JAX modules of built `@rkppl` programs (todo 1l3nni7). `log_density` is the
# `:sampler` value over the same boundary as `prepare_query`, and `jax.grad`
# of it matches that query's finite differences.

@traceable function jax_ppl_recurrence(R::AbstractVector, rho::Real, kappa::Real)
    T = promote_type(eltype(R), typeof(rho), typeof(kappa))
    W = Vector{T}(undef, length(R))
    C = one(T)
    recover = 1 - exp(-rho)
    for t in eachindex(R)
        C = C + (1 - C) * recover
        dep = C * (1 - exp(-kappa * R[t]))
        C = C - dep
        W[t] = dep
    end
    return W
end

const _JAX_RNG = Xoshiro(20261024)
const _JAX_N = 30
const _JAX_X = randn(_JAX_RNG, _JAX_N)
const _JAX_XM = randn(_JAX_RNG, _JAX_N, 3)
const _JAX_G = rand(_JAX_RNG, 1:4, _JAX_N)
const _JAX_LABELS = ["west", "north", "east", "south"][_JAX_G]
const _JAX_Y = 1 .+ 2 .* _JAX_X .+ 0.5 .* randn(_JAX_RNG, _JAX_N)
const _JAX_LENGTHS = [9, 4, 13, 6]
const _JAX_R = [abs.(randn(_JAX_RNG, n)) for n in _JAX_LENGTHS]
const _JAX_YR = [0.1 .+ 0.05 .* jax_ppl_recurrence(R, 0.7, 0.3) .+ 0.01 .* randn(_JAX_RNG, length(R))
                 for R in _JAX_R]
const _JAX_TRIALS = rand(_JAX_RNG, 10:20, _JAX_N)
const _JAX_SUCCESSES = [rand(_JAX_RNG, 0:n) for n in _JAX_TRIALS]
const _JAX_HURDLE = [rand(_JAX_RNG) < 0.3 ? 0 : rand(_JAX_RNG, 1:5) for _ in 1:_JAX_N]

const _JAX_CASES = (
    gaussian = () -> (@rkppl(begin
        a ~ Normal(0, 5); b ~ Normal(0, 2); s ~ Exponential(1.0)
        mu = a .+ b .* x
        y .~ Normal.(mu, s)
    end))(; x = _JAX_X) | (; y = _JAX_Y),
    bernoulli_logit = () -> (@rkppl(begin
        a ~ Normal(0, 1); b ~ Normal(0, 2)
        y .~ Bernoulli.(logistic.(a .+ b .* x))
    end))(; x = _JAX_X) | (; y = Int.(_JAX_Y .> 1)),
    negative_binomial = () -> (@rkppl(begin
        a ~ Normal(0, 1); phi ~ Exponential(1.0)
        y .~ NegativeBinomial2.(exp.(a .+ 0.1 .* x), phi)
    end))(; x = _JAX_X) | (; y = rand(_JAX_RNG, 0:6, _JAX_N)),
    design_groups = () -> (@rkppl(begin
        beta[1:3] .~ Normal(0, 2); tau ~ Exponential(1.0); s ~ Exponential(1.0)
        alpha[1:4] .~ Normal(0, tau)
        y .~ Normal.(X * beta .+ alpha[g], s)
    end))(; X = _JAX_XM, g = _JAX_G) | (; y = _JAX_Y),
    # A `levels(g)` axis over string labels: `prepare_data` computes the
    # labels and the level codes (no label is a literal of the program).
    string_levels = () -> (@rkppl(begin
        a ~ Normal(0, 1); tau ~ Exponential(1.0); s ~ Exponential(1.0)
        c[levels(g)] .~ Normal.(0, tau)
        y .~ Normal.(a .+ c[g], s)
    end))(; g = _JAX_LABELS) | (; y = _JAX_Y),
    simplex = () -> (@rkppl(begin
        theta ~ Dirichlet(ones(3))
        y .~ Categorical(theta)
    end))(;) | (; y = rand(_JAX_RNG, 1:3, _JAX_N)),
    # A scan of two states, whose step outputs a named tuple.
    two_state_scan = () -> (@rkppl(begin
        beta ~ Uniform(0, 1); sigma ~ Exponential(1.0); s ~ Exponential(1.0)
        @scan begin
            level[1] = 0.0
            increment[1] = 0.0
            for t in 2:T
                z ~ Normal(0, 1)
                increment[t] = beta * increment[t - 1] + sigma * z
                level[t] = level[t - 1] + increment[t]
            end
        end
        y .~ Normal.(level, s)
    end))(; T = 12) | (; y = cumsum(0.3 .* _JAX_X[1:12])),
    # `logbeta` of arguments above 8 (`jax.scipy.special.betaln` is not
    # float64-accurate there).
    beta_binomial = () -> (@rkppl(begin
        a ~ Normal(0, 1); phi ~ Exponential(10.0)
        c .~ BetaBinomial2.(n, logistic.(a .+ 0.3 .* x), phi)
    end))(; x = _JAX_X, n = _JAX_TRIALS) | (; c = _JAX_SUCCESSES),
    # Nested lazy branches inside a plate cell.
    hurdle = () -> (@rkppl(begin
        a ~ Normal(0, 1); e ~ Normal(0, 1)
        y .~ HurdlePoisson.(exp.(a .+ 0.2 .* x), logistic.(e .+ 0.1 .* x))
    end))(; x = _JAX_X) | (; y = _JAX_HURDLE),
    # An LKJ Cholesky factor (a loop whose branch arms differ in value type),
    # and multivariate normal slices with their runtime factor checks.
    lkj = () -> (@rkppl(begin
        L ~ LKJCholesky(3, 2.0)
        sd[1:3] .~ Exponential.(1)
        F = sd .* L
        eachcol(B[1:3, 1:4]) .~ MvNormalCholesky(zeros(3), F)
        s ~ Exponential(1)
        y .~ Normal.(B[1, g] .+ B[2, g] .* x, s)
    end))(; x = _JAX_X, g = _JAX_G) | (; y = _JAX_Y),
    ragged_recurrence = () -> (@rkppl(begin
        rho ~ LogNormal(log(0.7), 0.3); kappa ~ LogNormal(log(0.3), 1.0)
        b0 ~ Normal(0.1, 0.1); tau ~ Exponential(0.05); s ~ Exponential(0.05)
        b[1:P] .~ Normal(b0, tau)
        @plate for i in eachindex(y)
            W = jax_ppl_recurrence(R[i], rho, kappa)
            mu = b[i] .+ 0.05 .* W
            y[i] .~ Normal.(mu, s)
        end
    end))(; P = 4, R = _JAX_R) | (; y = _JAX_YR),
)

@testset "jax_source(built, plan): module" begin
    plan = _JAX_CASES.gaussian()
    built = build_kernel(plan)
    src = jax_source(built, plan; values = (:mu,))
    text = string(src)
    @test [f.name for f in src.functions] ==
          ["prepare_data", "log_density", "log_prior", "log_likelihood", "log_jacobian", "constrain"]
    # `prepare_data` takes the raw data; the functions, the prepared inputs.
    @test first.(src.functions[1].inputs) == ["x", "y"]
    @test all(f -> first.(f.inputs) == ["unconstrained", "x", "y"], src.functions[2:end])
    @test occursin("DIMENSION = 3", text)
    @test occursin("PARAMETERS = {\"a\": {\"start\": 0, \"stop\": 1", text)
    @test occursin("\"s\": {\"start\": 2, \"stop\": 3, \"transform\": \"exp\"", text)
    @test occursin("return {\"a\": ", text)
    ragged = _JAX_CASES.ragged_recurrence()
    rsrc = jax_source(build_kernel(ragged), ragged)
    @test occursin("def _rk_pad_rows(rows):", string(rsrc))
    @test "R_lengths" in first.(rsrc.functions[2].inputs)
    @test occursin("\"ragged\": True", string(rsrc))
    @test occursin("jax.lax.fori_loop(", string(rsrc))
end

@testset "jax_source(built, plan): log density and gradient ($name)" for name in keys(_JAX_CASES)
    if !rk_jax_available()
        @info "RK_JAX_PYTHON is not set: JAX parity for $name skipped"
        @test_skip false
    else
        plan = _JAX_CASES[name]()
        built = build_kernel(plan)
        src = jax_source(built, plan)
        query = prepare_query(built, plan, :sampler)
        u = 0.2 .* randn(Xoshiro(7), built.layout.total)
        data = ReactiveKernelsPPL._query_bound(plan)
        r = rk_jax_evaluate(src, "log_density", (u, Tuple(data)...); grad = true, prepare = true)
        @test r.value ≈ query(u) rtol = 1e-10
        reference = rk_fd_gradient(query, u)
        @test maximum(abs.(r.grad .- reference) ./ max.(1, abs.(reference))) < 1e-5
    end
end

@testset "jax_source(built, plan): constrain" begin
    if !rk_jax_available()
        @test_skip false
    else
        plan = _JAX_CASES.gaussian()
        built = build_kernel(plan)
        src = jax_source(built, plan; values = (:mu,))
        u = [0.9, 1.8, log(0.4)]
        r = rk_jax_evaluate(src, "constrain", (u, _JAX_X, _JAX_Y); prepare = true)
        @test r.value["a"] ≈ 0.9
        @test r.value["s"] ≈ 0.4
        @test r.value["mu"] ≈ 0.9 .+ 1.8 .* _JAX_X
    end
end
