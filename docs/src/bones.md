# Declarative PPL kernel: bones (graded-response latent trait)

This example ports the `bones_model` from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`bones_data-bones_model`) into the declarative-`@kernel` style. It is the BUGS
"bones" grade-of-ossification latent-trait model (WinBUGS vol. 1): `nChild = 13`
children with a latent skeletal maturity, and `nInd = 34` radiographic indicators
with `ncat ∈ {2,3,4,5}` ordered grades, authored on the FULL real data loaded
through PosteriorDB.jl.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/bones_model.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/bones_model.jl).

```math
\begin{aligned}
\theta_i &\sim \operatorname{Normal}(0, 36), \\
Q_{ijk} &= \operatorname{inv\_logit}\!\big(\delta_j (\theta_i - \gamma_{jk})\big), \\
p(\text{grade} = g) &= Q_{i,j,g-1} - Q_{i,j,g}, \qquad Q_{i,j,0} \equiv 1,\ Q_{i,j,\text{ncat}_j} \equiv 0,
\end{aligned}
```

so `p(1) = 1 - Q_1` and `p(\text{ncat}) = Q_{\text{ncat}-1}` fall out of the one
cumulative-gap formula; the log-likelihood sums `log p(grade_{ij})` over the
observed cells (missing grade is coded `-1` and contributes nothing).

The graph binds ONLY the RAW `grade`/`gamma`/`delta`/`ncat` block. Everything else
is derived in-graph over the full grid: the grid coordinate matrices are built
with `repeat` over the raw dimensions; the two bracketing cut indices are computed
from `grade` by arithmetic, safe-clamped, and used to GATHER the raw `gamma`;
`delta`/`theta` are gathered by the coordinates; the boundary and missing masks
are bound comparisons. `theta` is unconstrained (no Jacobian).

```text
unconstrained ──► theta ──► Normal(0,36) prior
grade,gamma,delta,ncat (raw) ─► in-graph coords, cut indices (clamp), gathers, masks
  └─► Q_lo, Q_hi ──► log(Q_lo - Q_hi) over observed cells ──► log likelihood

log prior + log likelihood ──► unconstrained log density
```

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :BonesModelExample, :BONES_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_bones!,
)
```

## Reactant

The exact authored graph compiles and executes through the public Reactant
boundary with value and gradient parity — `benchmark/batch_latent_gate.jl`
`@compile`s both the primal and the gradient and asserts they match the native
evaluation and the reference `.stan` (via BridgeStan, `propto = false`,
`jacobian = true`). The in-graph `repeat` coordinates, the `grade`-derived cut
indices, the `gamma`/`delta`/`theta` gathers and the comparison masks all lower
cleanly.
Evidence is bounded to six reference-finite native points (Reactant uses their
first point) under BridgeStan 2.9 / Stan 2.39; finite-point parity is not an
all-input proof.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.BonesModelExample.demo()'
```
