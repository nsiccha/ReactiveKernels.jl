# Declarative PPL kernel: irt_2pl (2-parameter logistic IRT)

This example ports the `irt_2pl` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`irt_2pl-irt_2pl`) into the declarative-`@kernel` style. It is a 2-parameter
logistic item-response model: each of `I` items has a discrimination
`a[i] > 0` and difficulty `b[i]`, each of `J` persons an ability `theta[j]`, and
the binary response matrix is `y[i,j] ~ Bernoulli_logit(a[i]·(theta[j] − b[i]))`.
Authored on the FULL real data (`I = 20` items, `J = 100` persons) loaded through
PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/irt_2pl.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/irt_2pl.jl).

```math
\begin{aligned}
y_{ij} &\sim \operatorname{Bernoulli\_logit}\!\big(a_i\,(\theta_j - b_i)\big), \\
\sigma_\theta,\ \sigma_a,\ \sigma_b &\sim \operatorname{Cauchy}(0, 2)\ \text{on } \sigma>0\ \text{(ordinary lpdf; no }{+}\log 2\text{ truncation term)}, \\
\theta_j &\sim \operatorname{Normal}(0, \sigma_\theta), \quad
a_i \sim \operatorname{LogNormal}(0, \sigma_a), \\
\mu_b &\sim \operatorname{Normal}(0, 5), \quad b_i \sim \operatorname{Normal}(\mu_b, \sigma_b).
\end{aligned}
```

Raw `y` (the `I×J` response matrix) is the **only** data HAVE. The per-cell
logit-scale linear predictor `eta[i,j] = a[i]·(theta[j] − b[i])` is a pure
in-graph broadcast `a .* (transpose(theta) .- b)` over the item (row) and person
(column) axes — the item/person structure is carried by the array axes
themselves, so no external per-cell index is materialized. The unconstrained
vector is `(u_sigma_theta, theta[1..J], u_sigma_a, u_a[1..I], mu_b, u_sigma_b,
b[1..I])`; `sigma_* = exp(u_sigma_*)`, `a = exp.(u_a)`, and each log coordinate
contributes its value to the log Jacobian. The likelihood uses the direct
logit-HAVE `Bernoulli` endpoint (no `logistic → logit` round trip). The
generated quantity `p = vec(inv_logit.(eta))` is the column-major flat
probability vector used by the scalar plate.

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel** derived from the executed kernel and selected plan,
and the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :Irt2plExample, :IRT_2PL_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_irt_2pl!,
)
```

## Reactant

On the focused gate's finite BridgeStan-valid probes (plus the explicit irt
probability-saturation stress probe), the exact authored graph compiles and
executes through the public Reactant boundary with value and gradient parity:
the primal and gradient both `@compile` and match native/the reference `.stan`
(via BridgeStan, `propto = false`, `jacobian = true`) to machine precision. The
`transpose`-broadcast linear predictor and the reused `Cauchy`/`Normal`/
`LogNormal`/`Bernoulli` endpoints all lower cleanly.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.Irt2plExample.demo()'
```
