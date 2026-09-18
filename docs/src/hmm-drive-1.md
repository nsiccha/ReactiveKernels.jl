# Declarative PPL kernel: basketball-drive HMM (`hmm_drive_1`)

This example ports the `hmm_drive_1` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior
`bball_drive_event_1-hmm_drive_1`, the basketball-drive model from the
Pathfinder paper): a two-state hidden Markov model with **Normal emissions on
two observation streams** — `u` (one over speed) and `v` (hoop distance) —
fixed emission scales `tau`, `rho`, simplex transition rows `theta1`, `theta2`,
and `ordered` emission means `phi`, `lambda`. The likelihood is the
**forward-algorithm marginal** over the latent state path, an irreducibly
sequential K-vector recursion, authored inline with the
[`scan` primitive](scan.md): the carry is the per-state log-belief vector, the
iterated sequence is the bound two-stream observation matrix widened with a
0/1 first-step mask column (`scan(eachrow(scan_rows), …)`), and the step updates
`gamma[k] = logsumexp_j(gamma[j] + mask_t·logtheta[j,k]) + emit_k(u_t, v_t)`
(the compiled module's `stablehlo.while` count is measured per query shape —
see the Reactant section — not asserted from other models).
The masked first step is the identity in log space (the carry is seeded with
the uniform initial log mass, so `logsumexp(seed) = 0`), which reproduces
Stan's bare first-emission vector exactly and keeps the graph valid for every
`N ≥ 1` in the model's data contract — `N = 1` included — without any special
case. The
transition rows use the Stan 2.39 inverse-ILR simplex transform and the
emission means the `ordered` transform, both with their exact
change-of-variables Jacobians; the transit prior reuses the shared Dirichlet
endpoint and the emission-mean priors the shared Normal endpoints. The Stan
`generated quantities` block is a Viterbi decode that does not enter `target`,
so it is outside the density this graph reproduces.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/hmm_drive_1.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/hmm_drive_1.jl).

```math
\begin{aligned}
\gamma_{1k} &= \log N(u_1 \mid \phi_k, \tau) + \log N(v_1 \mid \lambda_k, \rho),\\
\gamma_{tk} &= \log\sum_j \exp\!\big(\gamma_{t-1,j} + \log\theta_{jk}\big)
              + \log N(u_t \mid \phi_k, \tau) + \log N(v_t \mid \lambda_k, \rho),\\
\ell &= \log\sum_k \exp(\gamma_{Nk}), \qquad
\theta_k \sim \operatorname{Dirichlet}(\alpha_k),\quad
\phi_1, \lambda_1 \sim N(0,1),\quad \phi_2, \lambda_2 \sim N(3,1).
\end{aligned}
```

```text
unconstrained ──► simplex rows (inverse-ILR) ──► theta1, theta2 ──► logtheta
             ├─► ordered phi, lambda ──────────► emission means
u, v ──► scan_rows = [u v mask] ──► scan(carry = K-vector log-belief)
                                          │  (compiled HLO shape measured per
                                          │   query; valid for every N ≥ 1)
log prior + log Jacobian + forward log-likelihood ──► unconstrained log density
```

The panel below shows the raw authored source, the readable generated kernel,
and the compute DAG of the selected density plan.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :HmmDrive1Example, :HMM_DRIVE_1_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_hmm_drive_1!,
)
```

## Reactant

The natural forward recursion **lowers through Reactant**, and the two query
boundaries are measured separately (`benchmark/structured_gate.jl`, axes 3a/3b):
the public **all-bound** query (raw `u`/`v`/`alpha`/`tau`/`rho` bound) compiles
and matches the native density exactly, with the compiler specializing the
recurrence for the query shape (its emitted HLO contains no `stablehlo.while`
region; the specialization is recorded with a byte-size control, and no
carry-loop claim is made for it). The **traced-stream** query (the same graph
with the data ports free and traced) also matches native exactly and, on the
current pin carrying ReactiveKernels `90acd41c`, lowers as a **single
`stablehlo.while` carry loop** (~99 KB of HLO, asserted by the gate — the
earlier pin unrolled it into a ~9.6 MB module). The counts are read from
`repr(Reactant.@code_hlo ...)` — the same surface the repository's
authored-scan tests assert on — because `module_string` is empty on the pinned
Reactant and a count against it would be vacuous.

The **native plain-Enzyme reverse gradient** is a documented UNSUPPORTED axis
(snag `scan-prior-enzym-d67d4ac1`): it fails Enzyme's *static* activity
analysis (`EnzymeRuntimeActivityError`) under both the ordinary and
Const-annotated configurations, while the same graph's components, the same
priors without the scan, and a three-prior scan control all pass, and
`Enzyme.set_runtime_activity` matches BridgeStan's gradient. The
Reactant-compiled gradient is an explicit UNSUPPORTED axis at this pin on both
query shapes, each measured in its own isolated process (no native gradient
work of any mode beforehand): the all-bound shape's unrolled module is
intractable for the EnzymeMLIR reverse pass, and the single-while traced-stream
shape currently fails the reverse pass itself (`operand #0 does not dominate
this use`); complete diagnostics are retained by the gate.
The runtime-activity result and the same-process compile observations are kept
as separate, explicitly labeled facts in
`benchmark/structured_gate_diagnostics.jl` — no order-dependence or cache
explanation is claimed. The gate pins the documented failure signatures on the
standard axes so neither a regression nor a quiet fix can hide.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.HmmDrive1Example.demo()'
```
