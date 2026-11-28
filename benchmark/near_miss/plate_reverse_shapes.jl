# Which summed plates the opt-in generated reverse (`with_plate_reverse`)
# takes, and what it costs or saves on each: ten model shapes at two row
# counts, the gradient with the generated reverse on against the gradient with
# it off, both prepared from one plan in one process and timed in alternating
# rounds. "Tape lengths" lists the one-pass tapes the preparation owns (none:
# every plate runs inline, as with the generated reverse off).
#   julia --project=<benchmark environment> benchmark/near_miss/plate_reverse_shapes.jl [<rows>[,<rows>...] [<model name pattern>]]
# A caller that has defined its own `@b` (one that reads CPU time on a loaded
# host, say) is used as it is; otherwise Chairmarks provides it.
using ReactiveKernelsPPL, ReactiveKernels, Enzyme, Printf
using DifferentiationInterface: AutoEnzyme
isdefined(@__MODULE__, Symbol("@b")) || @eval using Chairmarks
const RK = ReactiveKernels
const ROWS = parse.(Int, split(get(ARGS, 1, "60,2000"), ','))
const PATTERN = Regex(get(ARGS, 2, "."))

const MODELS = (
    grouped_poisson = @rkppl(begin
        a ~ Normal(0, 5)
        b ~ Normal(0, 1)
        sd ~ Exponential(1)
        z[1:J] .~ Normal.(0, 1)
        eta = a .+ b .* x .+ sd .* z[group]
        y .~ Poisson.(exp.(eta))
    end),
    poisson = @rkppl(begin
        a ~ Normal(0, 5)
        b ~ Normal(0, 1)
        y .~ Poisson.(exp.(a .+ b .* x))
    end),
    poisson_rows = @rkppl(begin
        a ~ Normal(0, 5)
        sd ~ Exponential(1)
        z[1:n] .~ Normal.(0, 1)
        y .~ Poisson.(exp.(a .+ sd .* z))
    end),
    bernoulli_logit = @rkppl(begin
        a ~ Normal(0, 5)
        b ~ Normal(0, 1)
        y .~ BernoulliLogit.(a .+ b .* x)
    end),
    bernoulli_logit_rows = @rkppl(begin
        a ~ Normal(0, 5)
        sd ~ Exponential(1)
        z[1:n] .~ Normal.(0, 1)
        y .~ BernoulliLogit.(a .+ sd .* z)
    end),
    grouped_bernoulli_logit = @rkppl(begin
        a ~ Normal(0, 5)
        b ~ Normal(0, 1)
        sd ~ Exponential(1)
        z[1:J] .~ Normal.(0, 1)
        y .~ BernoulliLogit.(a .+ b .* x .+ sd .* z[group])
    end),
    normal_exp_scale = @rkppl(begin
        a ~ Normal(0, 5)
        b ~ Normal(0, 1)
        ls ~ Normal(0, 1)
        y .~ Normal.(a .+ b .* x, exp(ls))
    end),
    grouped_normal_exp_scale = @rkppl(begin
        a ~ Normal(0, 5)
        sd ~ Exponential(1)
        ls ~ Normal(0, 1)
        z[1:J] .~ Normal.(0, 1)
        y .~ Normal.(a .+ sd .* z[group], exp(ls))
    end),
    grouped_negative_binomial = @rkppl(begin
        a ~ Normal(0, 5)
        sd ~ Exponential(1)
        phi ~ Exponential(1)
        z[1:J] .~ Normal.(0, 1)
        y .~ NegativeBinomial2.(exp.(a .+ sd .* z[group]), phi)
    end),
    grouped_normal = @rkppl(begin
        a ~ Normal(0, 5)
        sd ~ Exponential(1)
        sigma ~ Exponential(1)
        z[1:J] .~ Normal.(0, 1)
        y .~ Normal.(a .+ sd .* z[group], sigma)
    end),
)

# The data one model takes, by its name.
function plan_for(name::Symbol, n::Int)
    J = n <= 100 ? 10 : 20
    group = [mod1(7i, J) for i in 1:n]
    x = [0.3 * cos(i) for i in 1:n]
    counts = [i % 3 == 0 ? 0 : 1 + i % 4 for i in 1:n]
    flags = [i % 3 == 0 ? 0 : i % 2 for i in 1:n]
    reals = [0.5 * sin(i) + 0.1 * group[i] / J for i in 1:n]
    model = getfield(MODELS, name)
    name === :grouped_poisson && return model(; group, J, x) | (; y = counts)
    name === :poisson && return model(; x) | (; y = counts)
    name === :poisson_rows && return model(; n) | (; y = counts)
    name === :bernoulli_logit && return model(; x) | (; y = flags)
    name === :bernoulli_logit_rows && return model(; n) | (; y = flags)
    name === :grouped_bernoulli_logit && return model(; group, J, x) | (; y = flags)
    name === :normal_exp_scale && return model(; x) | (; y = reals)
    name === :grouped_normal_exp_scale && return model(; group, J) | (; y = reals)
    name === :grouped_negative_binomial && return model(; group, J) | (; y = counts)
    name === :grouped_normal && return model(; group, J) | (; y = reals)
    error("no data for $name")
end

println(join(("SHAPE", "model", "rows", "tape lengths", "declined", "density ns",
              "gradient ns, generated reverse off", "gradient ns, on", "on / off",
              "gradient bytes, on", "largest gradient difference"), '\t'))
const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
for name in keys(MODELS), n in ROWS
    occursin(PATTERN, String(name)) || continue
    try
        plan = plan_for(name, n)
        built = build_kernel(plan)
        u = [0.1 * sin(i) for i in 1:built.layout.total]
        off = RK.with_plate_reverse(false) do
            prepare_sampler(built, plan, u; backend = BACKEND, nonallocating = true)
        end
        on = RK.with_plate_reverse(true) do
            prepare_sampler(built, plan, u; backend = BACKEND, nonallocating = true)
        end
        g0, g1 = similar(u), similar(u)
        sampler_value_and_gradient!(off, g0, u)
        sampler_value_and_gradient!(on, g1, u)
        slots = [c for caches in RK._ad_owned_storage(on.ad) if caches isa Tuple for c in caches if c isa RK._PlateTapeSlot]
        density = (@b $on($u) seconds = 0.15).time * 1e9
        best = [Inf, Inf]; bytes = 0
        for round in 1:4, (k, q, g) in ((1, off, g0), (2, on, g1))
            trial = @b sampler_value_and_gradient!($q, $g, $u) seconds = 0.15
            best[k] = min(best[k], trial.time * 1e9)
            k == 2 && (bytes = trial.bytes)
        end
        @printf("SHAPE\t%s\t%d\t%s\t%s\t%.1f\t%.1f\t%.1f\t%.3f\t%d\t%.3g\n", name, n,
            join((length(s.tape) for s in slots), ","), join((s.declined for s in slots), ","),
            density, best[1], best[2], best[2] / best[1], bytes, maximum(abs, g1 .- g0))
    catch err
        @printf("SHAPE\t%s\t%d\terror: %s\n", name, n,
            first(replace(sprint(showerror, err), r"\s+" => " "), 300))
    end
    flush(stdout)
end
println("SHAPE-END")
