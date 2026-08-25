using LinearAlgebra
using Random
using ReactiveKernels

# A complete graph-backed NUTS workflow. The Hamiltonian phase point owns an
# inspectable ReactiveProgram: integrator mutations of `pos` and `mom`
# invalidate the dependent fields, and generated getters recompute them lazily.
potential(position) = sum(abs2, position) / 2
potential_gradient(position) = (potential(position), copy(position))

rng = Xoshiro(20260825)
dimension = 4
point = euclidean_phasepoint(
    potential,
    potential_gradient,
    Diagonal(ones(dimension)),
    zeros(dimension),
    zeros(dimension),
)
sampler = nuts_state(
    point;
    rng,
    step_f = partial(leapfrog!; stepsize = 0.35),
    max_depth = 7,
)
chain = sample!(sampler, 1_000; discard_initial = 300)
means = vec(sum(chain.samples; dims = 2)) ./ size(chain.samples, 2)
variances = vec(sum(abs2, chain.samples .- means; dims = 2)) ./
    (size(chain.samples, 2) - 1)

println("selected plan:")
println(explain(plan(point)))
println("generated Hamiltonian getter:")
println(code_expr(point, :ham))
println("sample mean: ", means)
println("sample variance: ", variances)
println("divergences: ", count(stat -> stat.diverged, chain.diagnostics))
