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
`gamma[k] = logsumexp_j(gamma[j] + mask_t·logtheta[j,k]) + emit_k(u_t, v_t)`.
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
