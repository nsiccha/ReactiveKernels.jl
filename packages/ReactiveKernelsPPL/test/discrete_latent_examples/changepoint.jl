using ReactiveKernelsPPL, Random, ReactiveKernels
using LogExpFunctions: logsumexp, softmax
using SpecialFunctions: loggamma

# A hundred yearly counts whose rate drops from 3 to 1 in year 41, the shape
# of the coal-mining disasters series.
rng = Xoshiro(1851)
rate = [t < 41 ? 3.0 : 1.0 for t in 1:100]
y = map(rate) do r                 # Poisson(r) draws by inversion
    k, p, u = 0, exp(-r), rand(rng)
    total = p
    while u > total
        k += 1
        p *= r / k
        total += p
    end
    k
end

# log p(y | the late rate starts in year s), for every s: the early rate
# before s, the late rate from s on (without the -log(y!) terms every s shares).
function switch_scores(y, early, late)
    before = cumsum(y .* log(early) .- early)
    after = cumsum(y .* log(late) .- late)
    vcat(0.0, before[1:end-1]) .+ (after[end] .- vcat(0.0, after[1:end-1]))
end

# The switch year is uniform over the years and summed out.
switch_logdensity(y, early, late) = logsumexp(switch_scores(y, early, late)) -
    log(length(y)) - sum(loggamma.(y .+ 1))
switch_probabilities(y, early, late) = softmax(switch_scores(y, early, late))

model = @rkppl begin
    early ~ Exponential(3.0)
    late ~ Exponential(3.0)
    y ~ LogDensity(switch_logdensity, early, late)
    switch = switch_probabilities(y, early, late)
end
plan = model() | (; y)
built = build_kernel(plan)

# The switch year's posterior is an authored value: query it like any other.
switch = ReactiveKernels.prepare(built.spec; have = built.spec.have_names,
    want = (:switch,), bound = (; y))
u = unconstrain(built.layout, (; early = 3.0, late = 1.0))
probabilities = Base.invokelatest(switch, u)
