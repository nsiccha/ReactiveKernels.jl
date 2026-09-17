# Declarative PPL kernel: multi_occupancy (marginalized occupancy)

This example ports the `multi_occupancy` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`butterfly-multi_occupancy`) into the declarative-`@kernel` style. It is the
Dorazio-Royle multi-species site-occupancy model with data augmentation (Stan
example-models): `n = 28` observed species over `J = 20` sites with `K = 18`
visits, augmented to a superpopulation of `S = 50`, authored on the FULL real
data loaded through PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/multi_occupancy.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/multi_occupancy.jl).

```math
\begin{aligned}
uv_i &\sim \operatorname{MVN}(0, \Sigma), \quad \Sigma = \begin{bmatrix}\sigma_1^2 & \rho\sigma_1\sigma_2\\ \rho\sigma_1\sigma_2 & \sigma_2^2\end{bmatrix}, \\
\text{logit\_psi}_i &= uv_{i,1} + \alpha, \qquad \text{logit\_theta}_i = uv_{i,2} + \beta, \\
\ell_0:\ & \operatorname{logaddexp}(\log\operatorname{inv\_logit}(\text{logit}\_\psi)+K\log\operatorname{inv\_logit}(-\text{logit}\_\theta),\ \log\operatorname{inv\_logit}(-\text{logit}\_\psi)), \\
\text{detected } (X>0):\ & \log\operatorname{inv\_logit}(\text{logit}\_\psi) + \operatorname{Binomial\_logit}(X \mid K, \text{logit}\_\theta), \\
\text{undetected}:\ & \ell_0, \\
\text{never detected}:\ & \operatorname{logaddexp}(\log(1-\Omega),\ \log(\Omega)+J\ell_0).
\end{aligned}
```

The latent occupancy/availability indicators are MARGINALIZED with
`log_sum_exp`. The graph binds ONLY the RAW `n × J` detection matrix `X` and the
dimensions `n`, `J`, `K`: the column-major flat counts `vec(X)` and the species
coordinate `repeat(1:n, J)` are built in-graph, the binomial detection normalizer
`log C(K, X)` is computed by the shared `binomial` object (the authored formula
is in-graph; no backend cache is claimed here), and the
detected/undetected split is the in-graph mask `X > 0`. `Omega ∈ [0,1]` (logit),
`rho_uv ∈ [-1,1]` (scaled logit), and `sigma_uv > 0` (log) carry their transform
Jacobians; the bivariate Normal is authored inline.

```text
unconstrained ──► alpha, beta, Omega, rho_uv, sigma_uv, uv1, uv2 ──► constrained parameters
  cauchy/beta priors + inline bivariate Normal over uv ──► log prior
X (raw) ──► vec(X), spec (in-graph) ──► gather uv by species ──► logit_psi/logit_theta
  ├─ observed sites: log_inv_logit(psi) + Binomial_logit(X|K,theta)   ┐
  ├─ undetected: log_sum_exp(...)                                     ├─► log likelihood
  └─ never-detected (augmented): log_sum_exp(...)                     ┘

log prior + log Jacobian + log likelihood ──► unconstrained log density
```

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :MultiOccupancyExample, :MULTI_OCC_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_multi_occupancy!,
)
```

## Reactant

The exact authored graph compiles and executes through the public Reactant
boundary with value and gradient parity — `benchmark/batch_latent_gate.jl`
`@compile`s both the primal and the gradient and asserts they match the native
evaluation and the reference `.stan` (via BridgeStan, `propto = false`,
`jacobian = true`). The in-graph `vec`/`repeat` recipes, the shared `binomial`
detection object, and the `log_sum_exp`
marginalization all lower cleanly.
Evidence is bounded to six reference-finite native points (Reactant uses their
first point) under BridgeStan 2.9 / Stan 2.39; finite-point parity is not an
all-input proof.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.MultiOccupancyExample.demo()'
```
