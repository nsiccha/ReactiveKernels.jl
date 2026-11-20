# Which arm a binding of a summed Normal plate runs, and what each arm costs.
#
# A Normal regression whose location is live numbers times bound columns can
# run its row loop (over the distinct rows where rows repeat) or bound
# least-squares statistics behind a scale guard. The binding chooses
# (`_ls_prefers_statistics`, src/weighted_least_squares.jl). This
# probe times both arms on both sides of that choice: rows repeated 60, 10 and
# 2 times and all distinct, for 1, 2 and 5 coefficients and a per-row scale.
#
# usage: julia repeated_rows.jl [rows = 240] [rounds = 7]
#
# Columns: the rows the row program loops over (the distinct rows where the
# duplicate-row grouping applies); the arm the binding chose; density and
# value-and-gradient ns of that choice; of the row program
# (`with_bound_statistics(false)`); and of the statistics, forced, where the
# fit is usable. Five coefficients are lowered through a bound design matrix:
# that location keeps the lazy statistics whatever the rows (not grouped, not
# chosen per binding). Every column is its own model
# object and preparation. Times are the minimum over the rounds, taken in
# alternation. Compare within a row, not across rows: the row loop's time
# depends on where its vectors were allocated.
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
row_scaled() = @rkppl begin
    a ~ Normal(0, 5); b ~ Normal(0, 2); sigma ~ Exponential(1.0)
    y .~ Normal.(a .+ b .* x1, sigma .* w)
end
const MODELS = ("1 coefficient" => (intercept, ()), "2, from a vector" => (slope_vector, (:x1,)),
    "2 coefficients" => (slope, (:x1,)), "5 coefficients" => (five, (:x1, :x2, :x3, :x4)),
    "2, row scale" => (row_scaled, (:x1, :w)))

# `distinct` distinct rows, each `rows ÷ distinct` times, in shuffled order.
function data(inputs, rows, distinct; seed = 7)
    rng = MersenneTwister(seed)
    columns = Dict(c => randn(rng, distinct) for c in (:x1, :x2, :x3, :x4))
    columns[:w] = 0.5 .+ rand(rng, distinct)
    y = 0.3 .+ 0.7 .* columns[:x1] .- 0.2 .* columns[:x2] .+ 0.5 .* randn(rng, distinct)
    at = repeat(1:distinct; inner = rows ÷ distinct)[randperm(rng, rows)]
    NamedTuple{inputs}(Tuple(columns[c][at] for c in inputs)), (; y = y[at])
end

# 0: the binding's choice; 1: the rows; 2: the statistics whatever they cost.
# Forcing wraps the package's own rule, called in the world before the wrapper.
const FORCED = Ref(0)
const PACKAGE_WORLD = Base.get_world_counter()
@eval RK _ls_prefers_statistics(fit, rows::Int, grouped::Bool, columns::Bool) =
    Main.FORCED[] == 2 ? fit.usable :
    Base.invoke_in_world($PACKAGE_WORLD, _ls_prefers_statistics, fit, rows, grouped, columns)::Bool

function prepared(model, inputs, rows, distinct, arm)
    ins, observed = data(inputs, rows, distinct)
    plan = model()(; ins...) | observed
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
    fit = any(c -> RK._ls_fit(c) !== nothing, q.kernel.constants)
    # The grouping's row counts, one per distinct row, where it applied.
    counts = [length(c) for c in q.kernel.constants if c isa Vector{Int}]
    looped = isempty(counts) ? rows : only(counts)
    (; q, u, g, value, gradient = copy(g), fit, looped)
end
timed(c) = ((@b $(c.q)($(c.u)) seconds = 0.1).time * 1e9,
            (@b sampler_value_and_gradient!($(c.q), $(c.g), $(c.u)) seconds = 0.1).time * 1e9)

function main(rows, rounds)
    @printf("%-17s %5s %8s %6s | %-10s %7s %7s | %7s %7s | %7s %7s\n", "model", "rows", "distinct",
        "looped", "chosen", "dens", "grad", "rows d", "rows g", "stats d", "stats g")
    for (label, (model, inputs)) in MODELS, repeats in (60, 10, 2, 1)
        distinct = rows ÷ repeats
        chosen = prepared(model, inputs, rows, distinct, 0)
        plain = prepared(model, inputs, rows, distinct, 1)
        forced = prepared(model, inputs, rows, distinct, 2)
        @assert !plain.fit
        # The authored arm beside the statistics visits the rows the plan
        # without them visits.
        @assert chosen.looped == plain.looped
        for c in (chosen, forced)
            @assert isapprox(c.value, plain.value; rtol = 1e-10)
            @assert isapprox(c.gradient, plain.gradient; rtol = 1e-8)
        end
        cases = forced.fit ? (chosen, plain, forced) : (chosen, plain)
        best = [(Inf, Inf) for _ in cases]
        for _ in 1:rounds, (i, c) in enumerate(cases)
            best[i] = min.(best[i], timed(c))
        end
        @printf("%-17s %5d %8d %6d | %-10s %7.1f %7.1f | %7.1f %7.1f |", label, rows, distinct,
            plain.looped, chosen.fit ? "statistics" : "rows", best[1]..., best[2]...)
        forced.fit ? @printf(" %7.1f %7.1f\n", best[3]...) : println("  (no usable fit)")
        flush(stdout)
    end
end
main(length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 240, length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 7)
