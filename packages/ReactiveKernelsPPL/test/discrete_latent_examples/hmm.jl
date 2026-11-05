using ReactiveKernelsPPL, Random
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal_hmm_logdensity

# Sixty steps of a three-state chain; state k emits around k with noise 0.1,
# as in Turing's Bayesian HMM tutorial.
rng = Xoshiro(2019)
true_T = [0.90 0.05 0.05; 0.05 0.90 0.05; 0.05 0.05 0.90]
true_m = [1.0, 2.0, 3.0]
true_s = zeros(Int, 60)
true_s[1] = rand(rng, 1:3)
for t in 2:60
    true_s[t] = min(searchsortedfirst(cumsum(true_T[true_s[t - 1], :]), rand(rng)), 3)
end
y = true_m[true_s] .+ 0.1 .* randn(rng, 60)

# The state path is summed out by the forward algorithm: the parameters
# (transition rows T, state means m) are all the sampler sees.
model = @rkppl begin
    eachrow(T[1:K, 1:K]) .~ Dirichlet(ones(K) ./ K)
    m[1:K] .~ Normal.(1:K, 0.5)
    y ~ LogDensity(normal_hmm_logdensity, m, 0.1, T, ones(K) ./ K)
end
plan = model(; K = 3) | (; y)
built = build_kernel(plan)
latents = prepare_discrete_latents(built, plan)

# At one draw of the parameters (here the data-generating values): the
# posterior of every step's state, and one exact draw of the whole path.
u = unconstrain(built.layout, (; T = true_T, m = true_m))
posterior = latents(u).y          # probabilities, log_filtered, log_transition
path = discrete_latent_draws(latents, u, Xoshiro(1)).y
most_probable = [argmax(posterior.probabilities[t, :]) for t in 1:60]
