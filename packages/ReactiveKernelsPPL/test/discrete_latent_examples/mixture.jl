using ReactiveKernelsPPL, Random

# Eighty observations from two Gaussian components, 40 % around -1 and
# 60 % around 2, as in Turing's Gaussian mixture tutorial (one dimension).
rng = Xoshiro(7)
true_z = [rand(rng) < 0.4 ? 1 : 2 for _ in 1:80]
y = [-1.0, 2.0][true_z] .+ 0.6 .* randn(rng, 80)

# Each observation's component is summed out: the sampler sees the weights,
# the ordered means and the shared scale.
model = @rkppl begin
    w ~ Dirichlet(ones(2))
    mu ~ Ordered(Normal(0, 5), 2)
    sigma ~ Exponential(1.0)
    y .~ MixtureModel.(vcat.(Normal.(mu[1], sigma), Normal.(mu[2], sigma)), Ref(w))
end
plan = model() | (; y)
built = build_kernel(plan)
latents = prepare_discrete_latents(built, plan)

# At one draw of the parameters: each observation's component probabilities
# (80 × 2), and one exact draw of every component.
u = unconstrain(built.layout, (; w = [0.4, 0.6], mu = [-1.0, 2.0], sigma = 0.6))
memberships = latents(u).y.probabilities
components = discrete_latent_draws(latents, u, Xoshiro(1)).y
