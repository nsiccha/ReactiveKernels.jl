using ReactiveKernels, ReactiveKernelsPPL, Random, Test, LinearAlgebra
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal_hmm_log_weights_logdensity,
    lower_truncated_lognormal, upper_truncated_lognormal

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

# The data the caller bound, in `prepare_data`'s order: every input is a
# name the caller wrote (a column, a scalar value), never a value only the
# Julia binder computes.
function jax_raw_data(plan, src)
    names = first.(only(f for f in src.functions if f.name == "prepare_data").inputs)
    # A value the program reads only through its bound copy, and the observed
    # value of a conditioned declaration, are passed by their authored names;
    # the plan holds them as `_rkppl_value_<name>` and
    # `_rkppl_conditioned_<name>`. A definition `prepare_data` takes as an
    # input (`UNTRANSLATED`) is the binding's own value of it (a submodel's
    # `r.g` is the input `r_g`).
    held = plan.data_values === nothing ? Symbol[] : collect(keys(plan.data_values.values))
    function value(n)
        for k in (Symbol(n), Symbol("_rkppl_value_", n), Symbol("_rkppl_conditioned_", n))
            haskey(plan.columns, k) && return plan.columns[k]
        end
        ReactiveKernelsPPL._bound_value(plan, plan.columns,
            only(h for h in held if replace(String(h), "." => "_") == n))
    end
    Tuple(value(n) for n in names)
end

# A data helper with a loop and no recorded source: no host translation.
function jax_ppl_opaque_cumsum(x::AbstractVector)
    out = similar(x, Float64)
    acc = 0.0
    for i in eachindex(x)
        acc += x[i]
        out[i] = acc
    end
    return out
end

# A group's sorted distinct times, built with `push!` and an `isequal` run
# test, and each time's position among them: data-only cell recipes that only
# `prepare_data` computes (snag jax-source-cell-a3ec8540).
@traceable function jax_ppl_levels(x)
    levels = Float64[]
    for j in eachindex(x)
        v = convert(Float64, x[j])
        if isempty(levels) || !isequal(v, last(levels))
            push!(levels, v)
        end
    end
    unique!(sort!(levels))
end
@traceable function jax_ppl_positions(levels, x)
    idx = zeros(Int, length(x))
    for j in eachindex(x)
        idx[j] = searchsortedfirst(levels, x[j])
    end
    idx
end

# Submodels whose names reach the layout and the data in their own scope
# (`b.z`, `v.mask`).
@rkppl jax_ppl_group_effects(K) = begin
    tau ~ LogNormal(0, 1)
    z[1:K] .~ Normal.(0, 1)
    return z .* tau
end
@rkppl jax_ppl_walk(T) = begin
    sigma ~ Exponential(1.0)
    @scan begin
        level[1] = 0.0
        for t in 2:T
            z ~ Normal(0, 1)
            level[t] = level[t - 1] + sigma * z
        end
    end
    return level
end
@rkppl jax_ppl_masked_effect(x) = begin
    mask = Float64.(x .> 0)
    a ~ Normal(0, 1)
    return a .* mask
end
# BRM's monotonic-effects block (`mo()`, its RK emission): one increment
# simplex per predictor as the columns of `simplex_incr`, and their cumulative
# table (snag `jax-source-simpl-249ccdc6`).
@rkppl jax_ppl_monotonic(alpha, nincr, ntargets) = begin
    eachcol(simplex_incr[1:nincr, 1:ntargets]) .~ Dirichlet(alpha)
    beta[1:ntargets] .~ Normal.(0, 1)
    return cumsum(vcat(zeros(1, ntargets), simplex_incr); dims = 1) .* beta'
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
# Per-row evidence bounds, infinite on the rows without one (rkppl-use §9).
const _JAX_LIMIT = [isodd(i) ? -Inf : 1.0 for i in 1:_JAX_N]
const _JAX_LOWER = [i % 3 == 0 ? -Inf : -6.0 for i in 1:_JAX_N]
const _JAX_UPPER = [i % 4 == 0 ? Inf : 9.0 for i in 1:_JAX_N]
const _JAX_INTERVAL = [isodd(i) ? Inf : y + 0.5 for (i, y) in enumerate(_JAX_Y)]

# Two monotonic predictors of a factor with `levels` levels; two levels give
# one increment, a one-entry simplex with no coordinates.
jax_monotonic_plan(levels) = (@rkppl(begin
    a ~ Normal(0, 2); s ~ Exponential(1.0)
    mo ~ jax_ppl_monotonic(alpha, nincr, 2)
    y .~ Normal.(a .+ mo[g, 1] .+ mo[g, 2] .* x, s)
end))(; g = mod1.(_JAX_G, levels), x = _JAX_X, alpha = ones(levels - 1),
       nincr = levels - 1) | (; y = _JAX_Y)

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
    # Evidence bounds read as data, `-Inf`/`Inf` on rows without a bound.
    infinite_limits = () -> (@rkppl(begin
        a ~ Normal(0, 5); b ~ Normal(0, 2); s ~ Exponential(1.0)
        mu = a .+ b .* x
        y .~ censored.(Normal.(mu, s), lo, Inf)
        t .~ truncated.(Normal.(mu, s), lower, upper)
        w .~ interval_censored.(Normal.(mu, s), hi)
    end))(; x = _JAX_X, lo = _JAX_LIMIT, lower = _JAX_LOWER, upper = _JAX_UPPER,
           hi = _JAX_INTERVAL) | (; y = max.(_JAX_Y, _JAX_LIMIT),
           t = clamp.(_JAX_Y, -5.9, 8.9), w = _JAX_Y),
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
    # Simplex slices (stick-breaking per column or row, a data concentration)
    # and ordered slices: RKPPL's slice transforms and their log-Jacobians.
    simplex_slices = () -> (@rkppl(begin
        eachcol(S[1:3, 1:4]) .~ Dirichlet(alpha)
        eachrow(P[1:4, 1:3]) .~ Dirichlet(ones(3))
        b ~ Normal(0, 1); s ~ Exponential(1.0)
        y .~ Normal.(b .* (S[1, g] .- P[g, 2]), s)
    end))(; g = _JAX_G, alpha = [1.0, 2.0, 1.5]) | (; y = _JAX_Y),
    ordered_slices = () -> (@rkppl(begin
        eachcol(C[1:3, 1:4]) .~ Ordered(Normal(0, 2), 3)
        eachrow(D[1:4, 1:2]) .~ Ordered(Normal(0, 2), 2)
        s ~ Exponential(1.0)
        y .~ Normal.(C[1, g] .+ C[3, g] .+ D[g, 2], s)
    end))(; g = _JAX_G) | (; y = _JAX_Y),
    monotonic = () -> jax_monotonic_plan(4),
    monotonic_two_levels = () -> jax_monotonic_plan(2),
    # Evidence wrappers read the family's CDFs, whose lazy arms call a
    # prepared endpoint (`standardized`): a detection limit, interval
    # readings, a truncated response and truncated priors.
    censored_normal = () -> (@rkppl(begin
        a ~ Normal(0, 2); b ~ Normal(0, 2); s ~ Exponential(1.0)
        y .~ censored.(Normal.(a .+ b .* x, s), lod, Inf)
    end))(; x = _JAX_X, lod = 0.0) | (; y = max.(_JAX_Y, 0.0)),
    interval_censored = () -> (@rkppl(begin
        a ~ Normal(0, 2); b ~ Normal(0, 2); s ~ Exponential(1.0)
        y .~ interval_censored.(Normal.(a .+ b .* x, s), hi)
    end))(; x = _JAX_X, hi = floor.(_JAX_Y) .+ 1) | (; y = floor.(_JAX_Y)),
    truncated = () -> (@rkppl(begin
        m ~ truncated(Normal(0, 1), 0, Inf); s ~ Exponential(1.0)
        p ~ truncated(Beta(2, 2), 0.1, 0.9)
        y .~ truncated.(LogNormal.(m .* p, s), 0.01, 100.0)
    end))(;) | (; y = clamp.(exp.(0.3 .* _JAX_X), 0.02, 50.0)),
    # A submodel's group effects: its block ends at a data-derived position
    # named in the submodel's scope.
    grouped_submodel = () -> (@rkppl(begin
        a ~ Normal(0, 2); s ~ Exponential(1.0)
        b ~ jax_ppl_group_effects(K)
        y .~ Normal.(a .+ b[g], s)
    end))(; K = 4, g = _JAX_G) | (; y = _JAX_Y),
    # A submodel's data-only definition (`v.mask`), computed by prepare_data.
    submodel_data = () -> (@rkppl(begin
        v ~ jax_ppl_masked_effect(x)
        y .~ Normal.(v, 1)
    end))(; x = _JAX_X) | (; y = _JAX_Y),
    # Chained data-only definitions, one named like a Base function, and
    # `sinpi`/`cospi` of a data period.
    chained_definitions = () -> (@rkppl(begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1.0)
        angle = 2 .* x ./ period
        y .~ Normal.(a .* sinpi.(angle) .+ b .* cospi.(angle) .+ cos.(angle), s)
    end))(; x = _JAX_X, period = 7.0) | (; y = _JAX_Y),
    # A ragged plate whose cell reads a per-group scalar.
    ragged_group_means = () -> (@rkppl(begin
        mu0 ~ Normal(0, 2); tau ~ Exponential(1.0); s ~ Exponential(1.0)
        m[1:G] .~ Normal(mu0, tau)
        @plate for i in eachindex(y)
            y[i] .~ Normal.(m[i], s)
        end
    end))(; G = 4) | (; y = _JAX_YR),
    # Per-group recurrences over series of different lengths: an observed
    # AR(1), a deterministic recurrence of a ragged input, and a random walk
    # with one innovation per step (a one-entry group included).
    group_ar1 = () -> (@rkppl(begin
        phi_raw ~ Normal(0, 1); sigma ~ Exponential(1.0)
        phi = tanh(phi_raw)
        @plate for g in eachindex(y)
            @scan begin
                y[g][1] ~ Normal(mu[g][1], sigma / sqrt(1 - phi^2))
                for t in 2:length(y[g])
                    y[g][t] ~ Normal(mu[g][t] + phi * (y[g][t - 1] - mu[g][t - 1]), sigma)
                end
            end
        end
    end))(; mu = [0.1 .* collect(1.0:length(r)) for r in _JAX_R]) | (; y = _JAX_YR),
    group_recurrence = () -> (@rkppl(begin
        a ~ Normal(0, 1)
        @plate for g in eachindex(y)
            @scan begin
                h[1] = x[g][1]
                for t in 2:length(x[g])
                    h[t] = a * h[t - 1] + x[g][t]
                end
            end
            y[g] .~ Normal.(h[idx[g]], 1)
        end
    end))(; x = _JAX_R, idx = [[1, length(r)] for r in _JAX_R]) | (; y = [[0.2, 0.1] for _ in _JAX_R]),
    group_random_walk = () -> (@rkppl(begin
        s ~ Exponential(1)
        @plate for g in eachindex(y)
            @scan begin
                h[1] = 0.0
                for t in 2:length(y[g])
                    e ~ Normal(0, 1)
                    h[t] = h[t - 1] + s * e
                end
            end
            y[g] .~ Normal.(h, 1)
        end
    end))() | (; y = [_JAX_YR[1], [0.3], _JAX_YR[3]]),
    # A ragged integer index column with an empty row (a group with no
    # readings): its padding keeps the column integer.
    empty_index_row = () -> (@rkppl(begin
        a ~ Normal(0, 1)
        @plate for g in eachindex(idx)
            y[g] .~ Normal.(a .* x[idx[g]], 1)
        end
    end))(; x = [0.1, 0.4, -0.2, 0.3], idx = [[1, 3], Int[], [2]]) | (; y = [[0.2, 0.1], Float64[], [0.9]]),
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
    # A hidden-Markov law: shape checks decided while
    # tracing, a history scan over the (step, state) cells, and folds over
    # the states as loops.
    # A multivariate normal per ragged cell (an empty row included): a
    # Cholesky factor of the cell's leading block, its padding an identity.
    mvnormal_cells = () -> (@rkppl(begin
        mu ~ Normal(0, 1); s ~ Exponential(1.0)
        @plate for g in eachindex(y)
            y[g] ~ MvNormal(fill(mu, length(y[g])), s^2 * I(length(y[g])))
        end
    end))() | (; y = [[0.1, 0.2, 0.0], Float64[], [0.0, 0.1]]),
    # A diagonal covariance per cell (`Diagonal.(rows)` of the cells' vectors).
    mvnormal_diagonal = () -> (@rkppl(begin
        mu ~ Normal(0, 1); s ~ Exponential(1.0)
        @plate for g in eachindex(y)
            y[g] ~ MvNormal(fill(mu, length(y[g])), Diagonal(fill(s^2, length(y[g]))))
        end
    end))() | (; y = [[0.1, 0.2, 0.0], [0.3], [0.0, 0.1]]),
    # A dense covariance literal per cell.
    mvnormal_dense = () -> (@rkppl(begin
        mu ~ Normal(0, 1); s ~ Exponential(1.0); r ~ Uniform(-0.5, 0.5)
        @plate for g in eachindex(y)
            y[g] ~ MvNormal(fill(mu, 2), [s^2 r*s^2; r*s^2 s^2])
        end
    end))() | (; y = [[0.1, 0.2], [0.3, -0.1], [0.0, 0.1]]),
    # Each group's distinct times and positions, computed per cell on the
    # host, then a decay per distinct time.
    cell_host_order = () -> (@rkppl(begin
        k ~ Exponential(1.0); s ~ Exponential(1.0)
        @plate for g in eachindex(y)
            levels = jax_ppl_levels(t[g])
            idx = jax_ppl_positions(levels, t[g])
            decay = exp.(-k .* levels)
            y[g] .~ Normal.(decay[idx], s)
        end
    end))(; t = [[0.5, 1.0, 0.5], [2.0, 1.0, 2.0, 3.0, 0.5], [1.5], [0.25, 0.25]]) |
        (; y = [[0.6, 0.5, 0.7], [0.3, 0.45, 0.2, 0.1, 0.6], [0.4], [0.8, 0.85]]),
    hmm_law = () -> (@rkppl(begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1.0)
        locations = [a, b]
        y ~ LogDensity(normal_hmm_log_weights_logdensity, locations, s, log_transition, log_initial)
    end))(; log_transition = log.([0.9 0.1; 0.2 0.8]), log_initial = log.([0.5, 0.5])) |
        (; y = sin.(1:20)),
    # Structural zeros (snag jax-grad-of-the-3dac142a): a left-to-right chain
    # (state 2 never returns to state 1) starting in state 1, where no move
    # reaches state 2 at step 2, so its incoming log-sum-exp folds two -Inf.
    # `jnp.logaddexp` made every gradient entry NaN there.
    # The LogNormal mean graphs guard their scale with a throw on a negative
    # scale, a runtime check in the module (snag jax-source-refus-05f10f2c):
    # the ordinary mean (a lower bound of 0) and one-sided truncated means of
    # a plate's cells.
    lognormal_mean = () -> (@rkppl(begin
        mu ~ Normal(0, 1); log_sigma ~ Normal(0, 0.5)
        m = lower_truncated_lognormal(mu, exp(log_sigma), 0.0).mean()
        y ~ Normal(m, 0.5)
    end))() | (; y = 1.3),
    truncated_lognormal_means = () -> (@rkppl(begin
        mu ~ Normal(0, 1); log_sigma ~ Normal(0, 1); log_bound ~ Normal(0, 1)
        @plate for i in eachindex(y)
            lower[i] = lower_truncated_lognormal(mu, exp(log_sigma), exp(log_bound)).mean()
            upper[i] = upper_truncated_lognormal(mu, exp(log_sigma), exp(log_bound)).mean()
            y[i] ~ Normal(lower[i] - upper[i], 0.5)
        end
    end))() | (; y = [1.4, 2.1, 0.6, 1.9]),
    hmm_unreached = () -> (@rkppl(begin
        a ~ Normal(0, 1); b ~ Normal(0, 1); s ~ Exponential(1.0)
        locations = [a, b]
        y ~ LogDensity(normal_hmm_log_weights_logdensity, locations, s, log_transition, log_initial)
    end))(; log_transition = cat([log(0.5) -Inf; -Inf -Inf], [log(0.5) log(0.5); -Inf 0.0],
                                 [log(0.5) log(0.5); -Inf 0.0]; dims = 3),
           log_initial = [0.0, -Inf]) | (; y = [0.3, 0.1, -0.4, 0.2]),
)
# ICAR priors (snag independent-icar-2e025fad): one field over a graph, and K
# independent fields over the same graph as the rows or the columns of one
# array, K and N data. Each translates its edge plate (and field plate).
const _JAX_ICAR_NODE1 = [1, 1, 2, 3, 4, 5, 2]
const _JAX_ICAR_NODE2 = [2, 3, 4, 4, 5, 6, 5]
const _JAX_ICAR_K = repeat(1:3, inner = 6)
const _JAX_ICAR_N = repeat(1:6, outer = 3)
const _JAX_ICAR_CASES = (
    icar_field = () -> (@rkppl(begin
        a ~ Normal(0, 1); s ~ Exponential(1.0)
        phi[1:N] ~ ICAR(node1, node2)
        phi_sum = sum(phi)
        phi_sum ~ Normal(0, 0.001 * N)
        y .~ Normal.(a .+ phi, s)
    end))(; N = 6, node1 = _JAX_ICAR_NODE1, node2 = _JAX_ICAR_NODE2) | (; y = sin.(1:6)),
    icar_rows = () -> (@rkppl(begin
        a ~ Normal(0, 1); s ~ Exponential(1.0)
        eachrow(phi[1:K, 1:N]) .~ ICAR(node1, node2)
        @plate for i in eachindex(k)
            mu[i] = a + phi[k[i], n[i]]
        end
        y .~ Normal.(mu, s)
    end))(; K = 3, N = 6, node1 = _JAX_ICAR_NODE1, node2 = _JAX_ICAR_NODE2,
           k = _JAX_ICAR_K, n = _JAX_ICAR_N) | (; y = sin.(1:18)),
    icar_cols = () -> (@rkppl(begin
        a ~ Normal(0, 1); s ~ Exponential(1.0)
        eachcol(phi[1:N, 1:K]) .~ ICAR(node1, node2)
        @plate for i in eachindex(k)
            mu[i] = a + phi[n[i], k[i]]
        end
        y .~ Normal.(mu, s)
    end))(; K = 3, N = 6, node1 = _JAX_ICAR_NODE1, node2 = _JAX_ICAR_NODE2,
           k = _JAX_ICAR_K, n = _JAX_ICAR_N) | (; y = sin.(1:18)),
)

# The same law with one location per state, so the state count is data.
function _jax_hmm_states(T, K)
    m = @rkppl begin
        s ~ Exponential(1.0)
        mu[axes(log_initial, 1)] .~ Normal.(0, 1)
        y ~ LogDensity(normal_hmm_log_weights_logdensity, mu, s, log_transition, log_initial)
    end
    stay = [k == j ? 0.8 : 0.2 / (K - 1) for k in 1:K, j in 1:K]
    m(; log_transition = log.(stay), log_initial = fill(log(1 / K), K)) | (; y = sin.(1:T))
end

@testset "jax_source(built, plan): module" begin
    plan = _JAX_CASES.gaussian()
    built = build_kernel(plan)
    src = jax_source(built, plan; values = (:mu,))
    text = string(src)
    @test [f.name for f in src.functions] ==
          ["prepare_data", "log_density", "log_prior", "log_likelihood", "log_jacobian", "constrain",
           "dimension", "parameters"]
    # `prepare_data` takes the raw data; the functions, the prepared inputs.
    @test first.(src.functions[1].inputs) == ["x", "y"]
    @test all(f -> first.(f.inputs) == ["unconstrained", "x", "y"], src.functions[2:6])
    @test all(f -> first.(f.inputs) == ["prepared"], src.functions[7:8])
    @test occursin("DIMENSION = 3", text)
    @test occursin("PARAMETERS = {\"a\": {\"start\": 0, \"stop\": 1", text)
    @test occursin("\"s\": {\"start\": 2, \"stop\": 3, \"transform\": \"exp\"", text)
    @test occursin("return {\"a\": ", text)
    ragged = _JAX_CASES.ragged_recurrence()
    rsrc = jax_source(build_kernel(ragged), ragged)
    @test occursin("def _rk_pad_rows(rows, dtype):", string(rsrc))
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
        # Standalone: `prepare_data` computes every data definition from the
        # raw data (a nonempty `UNTRANSLATED` is not standalone support).
        @test occursin("UNTRANSLATED = {}", string(src))
        r = rk_jax_evaluate(src, "log_density", (u, jax_raw_data(plan, src)...);
                            grad = true, prepare = true)
        @test r.value ≈ query(u) rtol = 1e-10
        # jit with the prepared data as arguments: one executable per shape;
        # an integer that sets a shape is static (`StaticInt`).
        @test rk_jax_evaluate(src, "log_density", (u, jax_raw_data(plan, src)...);
                              prepare = :arguments).value ≈ query(u) rtol = 1e-10
        reference = rk_fd_gradient(query, u)
        @test maximum(abs.(r.grad .- reference) ./ max.(1, abs.(reference))) < 1e-5
    end
end

@testset "jax_source(built, plan): ICAR priors ($name)" for name in keys(_JAX_ICAR_CASES)
    plan = _JAX_ICAR_CASES[name]()
    built = build_kernel(plan)
    src = jax_source(built, plan)
    @test occursin("UNTRANSLATED = {}", string(src))
    if !rk_jax_available()
        @info "RK_JAX_PYTHON is not set: JAX parity for $name skipped"
        @test_skip false
    else
        query = prepare_query(built, plan, :sampler)
        u = 0.2 .* randn(Xoshiro(7), built.layout.total)
        r = rk_jax_evaluate(src, "log_density", (u, jax_raw_data(plan, src)...);
                            grad = true, prepare = true)
        @test r.value ≈ query(u) rtol = 1e-10
        reference = rk_fd_gradient(query, u)
        @test maximum(abs.(r.grad .- reference) ./ max.(1, abs.(reference))) < 1e-5
    end
end

@testset "jax_source(built, plan): one hidden-Markov module across steps and states" begin
    plans = [_jax_hmm_states(T, K) for (T, K) in ((20, 2), (7, 2), (9, 3))]
    srcs = [jax_source(build_kernel(p), p) for p in plans]
    texts = string.(srcs)
    # Steps and states are data: one module text, the folds over the states
    # are loops and the forward pass is a scan.
    @test texts[1] == texts[2] == texts[3]
    @test occursin("jax.lax.fori_loop(", texts[1]) && occursin("jax.lax.scan(", texts[1])
    @test occursin("UNTRANSLATED = {}", texts[1])
    if !rk_jax_available()
        @test_skip false
    else
        plan, src = plans[3], srcs[3]
        built = build_kernel(plan)
        query = prepare_query(built, plan, :sampler)
        u = 0.2 .* randn(Xoshiro(11), built.layout.total)
        r = rk_jax_evaluate(src, "log_density", (u, jax_raw_data(plan, src)...);
                            grad = true, prepare = true)
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

# Slice-transformed matrices constrain as native `constrain` does: each
# column or row a simplex or an ordered vector, a one-entry simplex the
# constant 1 (snag `jax-source-simpl-249ccdc6`).
@testset "jax_source(built, plan): slice transforms constrain as native" begin
    if !rk_jax_available()
        @test_skip false
    else
        rows(v) = permutedims(reduce(hcat, v))     # the module's nested rows
        for (name, read) in ((:simplex_slices, x -> (x.S, x.P)), (:ordered_slices, x -> (x.C, x.D)),
                             (:monotonic, x -> (x.mo.simplex_incr,)),
                             (:monotonic_two_levels, x -> (x.mo.simplex_incr,)))
            plan = _JAX_CASES[name]()
            built = build_kernel(plan)
            src = jax_source(built, plan)
            u = 0.3 .* randn(Xoshiro(11), built.layout.total)
            r = rk_jax_evaluate(src, "constrain", (u, jax_raw_data(plan, src)...); prepare = true).value
            native = read(ReactiveKernelsPPL.constrain(built.layout, u))
            jax = name in (:monotonic, :monotonic_two_levels) ? (rows(r["mo"]["simplex_incr"]),) :
                Tuple(rows(r[k]) for k in (name === :simplex_slices ? ("S", "P") : ("C", "D")))
            @test all(map((a, b) -> size(a) == size(b) && a ≈ b, jax, native))
        end
    end
end

# `prepare_data` takes the data the caller binds, by the names the model
# reads them: scalar values and covariates, never the binder's internal
# copies or the data-only definitions it evaluates (`w`), which the module
# computes on the host. One module serves other sizes and values.
# A data scalar the program reads only through its bound copy
# (`_rkppl_value_r`): `prepare_data` takes it by its
# authored name and computes the copy.
@testset "jax_source(built, plan): authored names for bound copies" begin
    live = @rkppl begin
        a[1:2] .~ Normal.(0, 1)
        v = a .* r
        f = vec(sum(v .* permutedims(node); dims = 2))
        y .~ Normal.(f, 1)
    end
    shifted = @rkppl begin
        sd ~ Exponential(1.0); z[1:G] .~ Normal.(0, 1)
        m = m0 .+ s0 .* z
        @plate for j in eachindex(idx)
            y[j] .~ Normal.(m[j] .+ 0.0 .* idx[j], sd)
        end
    end
    filtered = @rkppl begin
        sd ~ Exponential(1.0)
        @plate for j in eachindex(idx)
            @scan begin
                a[1] = x[1, j]
                for t in 2:T
                    a[t] = 0.5 * a[t - 1] + x[t, j]
                end
            end
            y[j] .~ Normal.(a[idx[j]], sd)
        end
    end
    yr = [[0.1, 0.2], [0.3], [0.0, 0.1, 0.2]]
    cases = (
        (live(; node = [0.2, 0.5, 0.9], r = 2.0) | (; y = [0.1, 0.2]), ["node", "r", "y"]),
        (shifted(; G = 3, m0 = 0.5, s0 = 2.0, idx = [[1, 2], [3], [1, 1, 2]]) | (; y = yr),
         ["G", "idx", "m0", "s0", "y"]),
        (filtered(; T = 6, x = reshape(collect(1.0:18.0) ./ 10, 6, 3),
                  idx = [[1, 2], [3], [6, 4, 2]]) | (; y = yr), ["T", "idx", "x", "y"]))
    for (plan, expected) in cases
        built = build_kernel(plan)
        src = jax_source(built, plan)
        names = first.(only(f for f in src.functions if f.name == "prepare_data").inputs)
        @test sort(names) == expected
        @test occursin("UNTRANSLATED = {}", string(src))
        if rk_jax_available()
            query = prepare_query(built, plan, :sampler)
            u = 0.2 .* randn(Xoshiro(3), built.layout.total)
            r = rk_jax_evaluate(src, "log_density", (u, jax_raw_data(plan, src)...);
                                grad = true, prepare = true)
            @test r.value ≈ query(u) rtol = 1e-10
            reference = rk_fd_gradient(query, u)
            @test maximum(abs.(r.grad .- reference) ./ max.(1, abs.(reference))) < 1e-5
        end
    end
end

# A conditioned declaration's observed value (`y ~ LogDensity(f, …)` bound
# `| (; y)`, a parameter fixed `| (; a)`): the binding holds it as `_rkppl_conditioned_y`,
# and `prepare_data` takes it by its authored name. The module runs from the
# authored data alone.
@testset "jax_source(built, plan): authored names for conditioned declarations" begin
    hmm = @rkppl begin
        mu ~ Normal(0.0, 1.0)
        step ~ Normal(0.0, 1.0)
        sigma ~ Exponential(1.0)
        locations = reshape(permutedims(hcat(mu .+ step .+ z, mu .+ z)), 2, 1, N)
        scales = reshape(permutedims(hcat(sigma .+ z, sigma .+ z)), 2, 1, N)
        y ~ LogDensity(normal_hmm_log_weights_logdensity, locations, scales, log_transition, log_initial)
    end
    fixed = @rkppl begin
        a ~ Normal(0, 1); s ~ Exponential(1.0)
        y .~ Normal.(a .+ x, s)
    end
    N = 4
    T = repeat(log.([0.7 0.3; 0.4 0.6]), 1, 1, N - 1)
    cases = (
        ((; N, z = 0.1 .* (1:N), log_transition = T, log_initial = log.([0.6, 0.4])),
         (; y = [0.3, 0.1, -0.4, 0.2]), hmm, ["N", "log_initial", "log_transition", "y", "z"]),
        ((; x = [0.1, -0.3, 0.5]), (; a = 0.8, y = [0.2, -0.1, 0.9]), fixed, ["a", "x", "y"]))
    for (data, observed, model, expected) in cases
        plan = model(; data...) | observed
        @test any(startswith("_rkppl_conditioned_") ∘ String, keys(plan.columns))
        built = build_kernel(plan)
        src = jax_source(built, plan)
        names = first.(only(f for f in src.functions if f.name == "prepare_data").inputs)
        @test sort(names) == expected
        text = string(src)
        @test !occursin("def prepare_data(_rkppl_conditioned_", text)
        @test !occursin("\"_rkppl_conditioned_", text)
        @test occursin("UNTRANSLATED = {}", text)
        if rk_jax_available()
            authored = merge(data, observed)
            query = prepare_query(built, plan, :sampler)
            u = 0.2 .* randn(Xoshiro(5), built.layout.total)
            r = rk_jax_evaluate(src, "log_density", (u, (authored[Symbol(n)] for n in names)...);
                                grad = true, prepare = true)
            @test r.value ≈ query(u) rtol = 1e-10
            reference = rk_fd_gradient(query, u)
            @test maximum(abs.(r.grad .- reference) ./ max.(1, abs.(reference))) < 1e-5
        end
    end
end

@testset "jax_source(built, plan): prepare_data takes the bound data" begin
    model = @rkppl begin
        a ~ Normal(a_loc, a_scale)
        s ~ Exponential(1.0)
        w = cos.(x ./ period)
        y .~ Normal.(a .+ w, s)
    end
    bind(n, a_loc, a_scale, period) = let x = collect(range(0, 2; length = n))
        model(; x, a_loc, a_scale, period) | (; y = 1 .+ cos.(x ./ 3) .+ 0.1 .* sin.(1:n))
    end
    plan = bind(6, 0.0, 2.0, 3.0)
    built = build_kernel(plan)
    src = jax_source(built, plan)
    @test first.(src.functions[1].inputs) == ["a_loc", "a_scale", "period", "x", "y"]
    @test occursin("UNTRANSLATED = {}", string(src))
    @test occursin("np.cos(x / period)", string(src))
    # A data definition without a host translation is not computed: it is an
    # input, named in `UNTRANSLATED`, so the module is not standalone.
    opaque = (@rkppl(begin
        a ~ Normal(0, 1); s ~ Exponential(1.0)
        w = jax_ppl_opaque_cumsum(x)
        y .~ Normal.(a .+ w, s)
    end))(; x = [0.1, 0.2, 0.3]) | (; y = [0.5, 0.4, 0.8])
    osrc = jax_source(build_kernel(opaque), opaque)
    @test "w" in first.(osrc.functions[1].inputs)
    @test occursin(r"UNTRANSLATED = \{\"w\": \".+\"\}", string(osrc))
    other = bind(11, 0.5, 1.5, 4.0)
    @test string(jax_source(build_kernel(other), other)) == string(src)
    if !rk_jax_available()
        @test_skip false
    else
        for p in (plan, other)
            query = prepare_query(built, p, :sampler)
            u = [0.3, -0.2]
            r = rk_jax_evaluate(src, "log_density", (u, jax_raw_data(p, src)...);
                                grad = true, prepare = true)
            @test r.value ≈ query(u) rtol = 1e-12
            @test r.grad ≈ rk_fd_gradient(query, u) rtol = 1e-6
        end
    end
end

# A data-only value the program computes with a helper that has no
# translation (it fills a `Dict` in a loop) and reads by its fields, as BRM's
# `car_graph` bodies do: the value is an input of `prepare_data`, named in
# `UNTRANSLATED` and passed as a dict by field name, and `prepare_data`
# computes its field reads (a block's extent, a gather, a named definition)
# from it (snag jax-source-refus-8c6ed1ec).
function jax_ppl_levels_of(labels)
    d = Dict{String,Int}()
    for l in labels
        get!(d, l, length(d) + 1)
    end
    nodes = sort!(collect(keys(d)))
    (; n = length(d), pos = [d[l] for l in labels], nodes,
       scale = [1.0 + 0.5 * i for i in 1:length(d)])
end
@rkppl jax_ppl_level_effects(labels) = begin
    g = jax_ppl_levels_of(labels)
    z[1:g.n] .~ Normal.(0, 1)
    return z[g.pos]
end
@testset "jax_source(built, plan): a definition without a translation, read by its fields" begin
    models = (
        # The extent and the gather read `g` (the extent only through the layout).
        extent = labels -> (@rkppl(begin
            g = jax_ppl_levels_of(labels)
            z[1:g.n] .~ Normal.(0, 1)
            y .~ Normal.(z[g.pos], 1)
        end))(; labels),
        # A named field read: `n` was the input, with a wrong reason, and `g` was
        # listed though no input.
        named = labels -> (@rkppl(begin
            g = jax_ppl_levels_of(labels)
            n = g.n
            z[1:n] .~ Normal.(0, 1)
            y .~ Normal.(sum(z), 1)
        end))(; labels),
        # A float field scales the field, as BYM2's `inv_scale` does.
        scaled = labels -> (@rkppl(begin
            g = jax_ppl_levels_of(labels)
            s ~ Exponential(1.0)
            z[1:g.n] .~ Normal.(0, 1)
            y .~ Normal.((s .* sqrt.(g.scale) .* z)[g.pos], 1)
        end))(; labels),
        # In a submodel's scope, whose `r.g` is the input `r_g`.
        submodel = labels -> (@rkppl(begin
            s ~ Exponential(1.0)
            r ~ jax_ppl_level_effects(labels)
            y .~ Normal.(r, s)
        end))(; labels),
    )
    small = (labels = ["a", "b", "a", "c"], y = [0.1, -0.2, 0.3, 0.0])
    large = (labels = ["q", "r", "s", "q", "t", "r"], y = [0.5, 0.1, -0.3, 0.2, 0.0, 1.0])
    for (name, model) in pairs(models)
        plan, other = (model(d.labels) | (; d.y) for d in (small, large))
        built = build_kernel(plan)
        src = jax_source(built, plan)
        text = string(src)
        g = name === :submodel ? "r.g" : "g"
        inputs = first.(src.functions[1].inputs)
        @test replace(g, "." => "_") in inputs
        # The field reads are computed, not inputs.
        @test !("n" in inputs) && !any(startswith("_"), inputs)
        @test occursin("UNTRANSLATED = {\"" * g * "\": \"the function `jax_ppl_levels_of` " *
                       "has loops or branches", text)
        # One module serves every binding.
        @test string(jax_source(build_kernel(other), other)) == text
        rk_jax_available() || (@test_skip false; continue)
        for p in (plan, other)
            b = build_kernel(p)
            query = prepare_query(b, p, :sampler)
            u = collect(range(-0.4, 0.6; length = b.layout.total))
            r = rk_jax_evaluate(src, "log_density", (u, jax_raw_data(p, src)...);
                                grad = true, prepare = true)
            @test r.value ≈ query(u) rtol = 1e-12
            @test r.grad ≈ rk_fd_gradient(query, u) rtol = 1e-6
            @test rk_jax_layout(src, jax_raw_data(p, src)).dimension == b.layout.total
        end
    end
    # The value may also be given as its fields in order.
    if rk_jax_available()
        plan = models.extent(small.labels) | (; small.y)
        src = jax_source(build_kernel(plan), plan)
        query = prepare_query(build_kernel(plan), plan, :sampler)
        data = jax_raw_data(plan, src)
        u = [0.3, -0.2, 0.5]
        r = rk_jax_evaluate(src, "log_density",
                            (u, (d isa NamedTuple ? Tuple(d) : d for d in data)...); prepare = true)
        @test r.value ≈ query(u) rtol = 1e-12
    end
end

# A layout whose sizes follow the data (a level axis, a scan's length) is not
# stated in the module: `dimension(prepared)` and `parameters(prepared)` read
# it from the prepared data, so one module text serves every binding.
@testset "jax_source(built, plan): layout from the prepared data" begin
    mk(labels) = (@rkppl(begin
        a ~ Normal(0, 1); tau ~ Exponential(1.0); s ~ Exponential(1.0)
        c[levels(g)] .~ Normal.(0, tau)
        y .~ Normal.(a .+ c[g], s)
    end))(; g = labels) | (; y = collect(range(-1, 1; length = length(labels))))
    small, large = mk(["a", "b", "c", "a"]), mk(["a", "b", "c", "d", "e", "a", "e"])
    src = jax_source(build_kernel(small), small)
    @test string(jax_source(build_kernel(large), large)) == string(src)
    @test !occursin("DIMENSION", string(src))
    @test occursin("def dimension(prepared: dict) -> int:", string(src))
    if rk_jax_available()
        for (p, total, levels) in ((small, 6, 3), (large, 8, 5))
            r = rk_jax_layout(src, jax_raw_data(p, src))
            @test r.dimension == total == build_kernel(p).layout.total
            @test r.parameters["c"] == (start = 3, stop = total, dims = (levels,))
            @test r.parameters["s"] == (start = 2, stop = 3, dims = ())
        end
        # A point holds one binding's coordinates: the module refuses another
        # length, as native cuts do (snag `rk-prepared-cut-0e0b98ed`).
        u = 0.1 .* collect(1:6)
        query = prepare_query(build_kernel(small), small, :sampler)
        @test rk_jax_evaluate(src, "log_density", (u, jax_raw_data(small, src)...);
            prepare = true).value ≈ query(u) rtol = 1e-10
        for n in (5, 8)
            @test_throws "packed length" rk_jax_evaluate(src, "log_density",
                (0.1 .* collect(1:n), jax_raw_data(small, src)...); prepare = true)
        end
    end
end

# The module is a documented API: every public function states what it
# computes and annotates its parameters and result, and the model's own
# docstrings reach the module and `constrain` (downstream readability requirement).
@testset "jax_source(built, plan): documented functions" begin
    model = @rkppl begin
        "Mean of the observations (unit: metre)."
        mu ~ Normal(mu_loc, mu_scale)
        "Observation noise scale (unit: metre)."
        sigma ~ Exponential(sigma_prior)
        "Shifted mean, an authored definition."
        shifted = mu .+ offset
        y .~ Normal.(shifted, sigma)
    end
    plan = model(; mu_loc = 0.0, mu_scale = 1.0, sigma_prior = 1.0,
                 offset = [0.0, 0.5, 1.0]) | (; y = [0.1, -0.3, 0.5])
    text = string(jax_source(build_kernel(plan), plan; values = (:shifted,)))
    header = split(text, "\nimport")[1]
    @test occursin("- `mu`: Mean of the observations (unit: metre).", header)
    constrain_doc = split(split(text, "def constrain(")[2], "\"\"\"")[2]
    @test occursin("- `sigma`: Observation noise scale (unit: metre).", constrain_doc)
    @test occursin("- `shifted`: Shifted mean, an authored definition.", constrain_doc)
    for line in filter(l -> startswith(l, "def "), split(text, "\n"))
        @test occursin(") -> ", line)
        params = match(r"\((.*)\)", line)[1]
        @test all(p -> occursin(": ", p), split(params, ", "))
    end
    @test !occursin(r"^def rk_"m, text)
end

# `constrain` returns what native `constrain(layout, u)` does: a submodel's
# values nested under it, and a parameter with no coordinates in this binding
# as an empty array (one module for every binding). The functions take
# `(unconstrained, **prepared)` (user GO 08jy2dh).
@testset "jax_source(built, plan): constrain as native, keyword prepared data" begin
    walk = (@rkppl(begin
        lv ~ jax_ppl_walk(T)
        s ~ Exponential(1.0)
        y .~ Normal.(lv, s)
    end))(; T = 6) | (; y = sin.(1:6))
    mk(X) = (@rkppl(begin
        a ~ Normal(0, 1)
        b[axes(X, 2)] .~ Normal.(0, 1)
        y .~ Normal.(a .+ X * b, 1)
    end))(; X) | (; y = [0.1, -0.2, 0.3, 0.0, 0.5])
    empty, two = mk(zeros(5, 0)), mk(collect(reshape(0.1:0.1:1.0, 5, 2)))
    wsrc = jax_source(build_kernel(walk), walk)
    esrc = jax_source(build_kernel(empty), empty)
    @test string(esrc) == string(jax_source(build_kernel(two), two))
    @test occursin("def log_density(unconstrained: ArrayLike, **prepared: ArrayLike) -> jax.Array:",
                   string(wsrc))
    if rk_jax_available()
        built = build_kernel(walk)
        u = 0.1 .* collect(1:built.layout.total)
        native = ReactiveKernelsPPL.constrain(built.layout, u)
        r = rk_jax_evaluate(wsrc, "constrain", (u, jax_raw_data(walk, wsrc)...); prepare = true).value
        @test sort(collect(keys(r))) == ["lv", "s"]
        @test r["lv"]["sigma"] ≈ native.lv.sigma
        @test r["lv"]["_ppl_scan_z_level"] ≈ native.lv._ppl_scan_z_level
        r = rk_jax_evaluate(esrc, "constrain", ([0.2], jax_raw_data(empty, esrc)...); prepare = true).value
        @test r["a"] ≈ 0.2 && isempty(r["b"])
    end
end

# A per-index value of a ragged plate comes back from `constrain` padded,
# with its lengths beside it, so a caller can strip the padding.
@testset "jax_source(built, plan): ragged values in constrain" begin
    plan = (@rkppl(begin
        a ~ Normal(0, 1)
        @plate for g in eachindex(y)
            mu[g] = a .+ x[g]
            y[g] .~ Normal.(mu[g], 1)
        end
    end))(; x = [[0.1, 0.2, 0.3], [0.5], [0.0, 1.0]]) | (; y = [[0.2, 0.1, 0.0], [0.4], [0.3, 0.2]])
    src = jax_source(build_kernel(plan), plan; values = (:mu,))
    if rk_jax_available()
        r = rk_jax_evaluate(src, "constrain", ([0.3], jax_raw_data(plan, src)...); prepare = true).value
        lengths = Int.(r["mu_lengths"])
        @test lengths == [3, 1, 2]
        @test [r["mu"][g][1:n] for (g, n) in enumerate(lengths)] ≈ [[0.4, 0.5, 0.6], [0.8], [0.3, 1.3]]
    end
end

# The LogNormal variance graphs are still refused: a plate child spliced under
# a lazy arm (`_logmills_quadrature`) reaches the module as an execution copy
# without its plan, and the continued fraction's length is data (snag
# jax-source-refus-05f10f2c, follow-ups filed on ReactiveKernels' queue).
@testset "jax_source(built, plan): LogNormal variance graphs (not yet translated)" begin
    plan = (@rkppl(begin
        mu ~ Normal(0, 1); log_sigma ~ Normal(0, 0.5)
        v = lower_truncated_lognormal(mu, exp(log_sigma), 0.0).var()
        y ~ Normal(0.0, sqrt(v + 0.1))
    end))() | (; y = 1.3)
    built = build_kernel(plan)
    @test_broken (jax_source(built, plan); true)
end
