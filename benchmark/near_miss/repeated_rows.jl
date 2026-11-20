# Which arm a binding of a summed Normal plate runs, and what each arm costs.
#
# A Normal regression can run its row loop (over the distinct rows where rows
# repeat) or bound least-squares statistics. The binding chooses
# (`_ls_prefers_statistics` and `_ls_blocks_pay`,
# src/weighted_least_squares.jl). This probe times both arms on both sides of
# that choice:
#
# - rows repeated 60, 10 and 2 times and all distinct, for 1, 2 and 5
#   coefficients, for coefficients declared as numbers and read from a
#   declared vector, for a scale that is no parameter, and for a per-row scale;
# - one gathered coefficient vector with 2, 3, 4 and 8 rows per group (both
#   sides of its switch in `_ls_blocks_pay`), and two with 2, 4, 5, 20 and 200.
#
# usage: julia repeated_rows.jl [rows = 240] [rounds = 7]
#
# Columns: the rows the row program loops over (the distinct rows where the
# duplicate-row grouping applies); the arm the binding chose; density and
# value-and-gradient ns of that choice; of the row program
# (`with_bound_statistics(false)`); and of the statistics, forced, where a
# fit is usable. Every column is its own model object and preparation. Times
# are the minimum over the rounds, taken in alternation. Compare within a
# line, not across lines: a row loop's time depends on where its vectors were
# allocated.
#
# Five coefficients declared as numbers are lowered through a bound design
# matrix. That location and the gathered ones keep one program with a guarded
# region: their choice is the fit's bound `usable` flag, and "rows" then
# means the region's authored arm.
using ReactiveKernelsPPL, ReactiveKernels, Enzyme, ADTypes, Chairmarks, Random, LinearAlgebra, Printf
BLAS.set_num_threads(1)
const RK = ReactiveKernels
const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

intercept() = @rkppl begin
    m ~ Normal(0, 5); sigma ~ Exponential(1.0)
    y .~ Normal.(m, sigma)
end
# Coefficients read from a declared vector (the SesameOnePred spelling).
slope_vector() = @rkppl begin
    beta[1:2] .~ Flat()
    sigma ~ restricted(Flat(), 0, Inf)
    mu = beta[1] .+ beta[2] .* x1
    y .~ Normal.(mu, sigma)
end
slope() = @rkppl begin
    a ~ Normal(0, 5); b ~ Normal(0, 2); sigma ~ Exponential(1.0)
    y .~ Normal.(a .+ b .* x1, sigma)
end
five() = @rkppl begin
    a ~ Normal(0, 5); b1 ~ Normal(0, 2); b2 ~ Normal(0, 2); b3 ~ Normal(0, 2); b4 ~ Normal(0, 2)
    sigma ~ Exponential(1.0)
    y .~ Normal.(a .+ b1 .* x1 .+ b2 .* x2 .+ b3 .* x3 .+ b4 .* x4, sigma)
end
# A scale that is no parameter: its fit is the lazy location's.
fixed_scale() = @rkppl begin
    a ~ Normal(0, 5); b ~ Normal(0, 2)
    y .~ Normal.(a .+ b .* x1, 1.5)
end
row_scaled() = @rkppl begin
    a ~ Normal(0, 5); b ~ Normal(0, 2); sigma ~ Exponential(1.0)
    y .~ Normal.(a .+ b .* x1, sigma .* w)
end
# An intercept and a slope per group (the Rats shape).
per_group() = @rkppl begin
    mu_alpha ~ Normal(0, 100); mu_beta ~ Normal(0, 100)
    sigma ~ restricted(Flat(), 0, Inf)
    sigma_alpha ~ restricted(Flat(), 0, Inf); sigma_beta ~ restricted(Flat(), 0, Inf)
    alpha[1:N] .~ Normal.(mu_alpha, sigma_alpha)
    beta[1:N] .~ Normal.(mu_beta, sigma_beta)
    mu = alpha[g] .+ beta[g] .* (x1 .- xbar)
    y .~ Normal.(mu, sigma)
end
# An intercept per group alone (the RadonCounty shape).
group_intercept() = @rkppl begin
    mu_alpha ~ Normal(0, 100)
    sigma ~ restricted(Flat(), 0, Inf); sigma_alpha ~ restricted(Flat(), 0, Inf)
    alpha[1:N] .~ Normal.(mu_alpha, sigma_alpha)
    y .~ Normal.(alpha[g], sigma)
end
const MODELS = ("1 coefficient" => (intercept, ()), "2, from a vector" => (slope_vector, (:x1,)),
    "2 coefficients" => (slope, (:x1,)), "2, fixed scale" => (fixed_scale, (:x1,)),
    "5 coefficients" => (five, (:x1, :x2, :x3, :x4)),
    "2, row scale" => (row_scaled, (:x1, :w)))

# `distinct` distinct rows, each `rows ÷ distinct` times, in shuffled order.
function repeated(inputs, rows, distinct; seed = 7)
    rng = MersenneTwister(seed)
    columns = Dict(c => randn(rng, distinct) for c in (:x1, :x2, :x3, :x4))
    columns[:w] = 0.5 .+ rand(rng, distinct)
    y = 0.3 .+ 0.7 .* columns[:x1] .- 0.2 .* columns[:x2] .+ 0.5 .* randn(rng, distinct)
    at = repeat(1:distinct; inner = rows ÷ distinct)[randperm(rng, rows)]
    NamedTuple{inputs}(Tuple(columns[c][at] for c in inputs)), (; y = y[at])
end
# `groups` groups of `per` rows each, measured at shared ages.
function grouped(groups, per; seed = 7)
    rng = MersenneTwister(seed)
    g = repeat(1:groups; inner = per)
    x1 = repeat(Float64.(8 .+ 7 .* (0:per-1)), groups)
    y = 240 .+ 6 .* (x1 .- 22) .+ 10 .* randn(rng, groups)[g] .+ 5 .* randn(rng, groups * per)
    (; N = groups, g, x1, xbar = 22.0), (; y)
end

# 0: the binding's choice; 1: the rows; 2: the statistics whatever they cost.
# Forcing wraps the package's own rules, called in the world before the wrappers.
const FORCED = Ref(0)
const PACKAGE_WORLD = Base.get_world_counter()
@eval RK _ls_prefers_statistics(fit, rows::Int, grouped::Bool, factor::Int) =
    Main.FORCED[] == 2 ? fit.usable :
    Base.invoke_in_world($PACKAGE_WORLD, _ls_prefers_statistics, fit, rows, grouped, factor)::Bool
@eval RK _ls_blocks_pay(n::Int, k::Int, blocks::Int) = Main.FORCED[] == 2 ||
    Base.invoke_in_world($PACKAGE_WORLD, _ls_blocks_pay, n, k, blocks)::Bool

function prepared(model, inputs, observed, rows, arm)
    plan = model()(; inputs...) | observed
    built = build_kernel(plan)
    u = 0.1 .* randn(MersenneTwister(1), built.layout.total)
    FORCED[] = arm
    q = try
        RK.with_bound_statistics(arm != 1) do
            prepare_sampler(built, plan, u; backend = BACKEND)
        end
    finally
        FORCED[] = 0
    end
    g = similar(u)
    value = first(sampler_value_and_gradient!(q, g, u))
    fits = [RK._ls_fit(c) for c in q.kernel.constants if RK._ls_fit(c) !== nothing]
    # The grouping's row counts, one per distinct row, where it applied.
    counts = [length(c) for c in q.kernel.constants if c isa Vector{Int} && length(c) < rows]
    (; q, u, g, value, gradient = copy(g), fit = any(f -> f.usable, fits),
       looped = isempty(counts) ? rows : minimum(counts))
end
timed(c) = ((@b $(c.q)($(c.u)) seconds = 0.1).time * 1e9,
            (@b sampler_value_and_gradient!($(c.q), $(c.g), $(c.u)) seconds = 0.1).time * 1e9)

function line(label, model, inputs, observed, rows, shape, rounds)
    chosen = prepared(model, inputs, observed, rows, 0)
    plain = prepared(model, inputs, observed, rows, 1)
    forced = prepared(model, inputs, observed, rows, 2)
    @assert !plain.fit
    for c in (chosen, forced)
        @assert isapprox(c.value, plain.value; rtol = 1e-10)
        @assert isapprox(c.gradient, plain.gradient; rtol = 1e-8)
    end
    cases = forced.fit ? (chosen, plain, forced) : (chosen, plain)
    best = [(Inf, Inf) for _ in cases]
    for _ in 1:rounds, (i, c) in enumerate(cases)
        best[i] = min.(best[i], timed(c))
    end
    @printf("%-17s %5d %12s %6d | %-10s %7.1f %7.1f | %7.1f %7.1f |", label, rows, shape,
        plain.looped, chosen.fit ? "statistics" : "rows", best[1]..., best[2]...)
    forced.fit ? @printf(" %7.1f %7.1f\n", best[3]...) : println("  (no usable fit)")
    flush(stdout)
end

function main(rows, rounds)
    @printf("%-17s %5s %12s %6s | %-10s %7s %7s | %7s %7s | %7s %7s\n", "model", "rows", "shape",
        "looped", "chosen", "dens", "grad", "rows d", "rows g", "stats d", "stats g")
    for (label, (model, names)) in MODELS, repeats in (60, 10, 2, 1)
        distinct = rows ÷ repeats
        inputs, observed = repeated(names, rows, distinct)
        line(label, model, inputs, observed, rows, "$(distinct) distinct", rounds)
    end
    for per in (2, 3, 4, 8)
        inputs, observed = grouped(30, per)
        line("1 per group", group_intercept, (; N = inputs.N, g = inputs.g), observed, 30 * per,
            "$(per) per group", rounds)
    end
    for per in (2, 4, 5, 20, 200)
        inputs, observed = grouped(30, per)
        line("2 per group", per_group, inputs, observed, 30 * per, "$(per) per group", rounds)
    end
end
# Run as a script; `include` from a session and call `main` yourself.
abspath(PROGRAM_FILE) == @__FILE__() &&
    main(length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 240, length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 7)
