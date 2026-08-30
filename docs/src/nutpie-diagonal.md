# Nutpie diagonal metric adaptation

This page is an external compiler corpus derived from Adrian Seyboldt's
[`nuts-rs`](https://github.com/pymc-devs/nuts-rs/tree/97be9ab88cfaadfafd9e5f4409a3b1d5af62805a)
diagonal adaptation at revision
`97be9ab88cfaadfafd9e5f4409a3b1d5af62805a`. It demonstrates mathematical
kernels that ReactiveKernels compiles through both its native and optional
Reactant paths; it does not add a nutpie sampler API to the package.

The executable source authority is
[`examples/nutpie_diagonal_adaptation.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/examples/nutpie_diagonal_adaptation.jl).
During every docs build, the panels below read the two actual `@kernel`
definitions from that file, prepare and execute them, derive the readable
Generated kernel from `code_expr`, and render the exact selected `Plan` as the
Compute DAG. The displayed mathematics therefore cannot drift into a parallel
docs-only transcription.

## Initialization from position and gradient

Given a position vector $q$ and its gradient $g$, initialization uses the
gradient magnitude as a first diagonal scale estimate:

```math
v_i = \operatorname{clamp}\!\left(\frac{1}{|g_i|}, 10^{-20}, 10^{20}\right),
\qquad
s_i = \sqrt{v_i},
\qquad
s_i^{-1} = \sqrt{v_i^{-1}}.
```

The transformed center and inverse-scale log determinant are

```math
\mu_i = q_i + s_i^2 g_i,
\qquad
\log |J| = \sum_i \log(s_i^{-1}).
```

Non-finite initial scale estimates fall back to unit variance. The Raw input
pane shows the complete kernel definition followed by the explicit
`prepare(...; have, want)` call and invocation that produce the other two
panes.

## Running draw/gradient adaptation

The full step owns four estimators explicitly: current draw and gradient
statistics plus their background counterparts. For each accepted sample it
updates the count and old-mean differences in the same order as nuts-rs:

```math
n' = n + 1,
\qquad
\delta = x - \bar{x},
\qquad
\bar{x}' = \bar{x} + \frac{\delta}{n'},
\qquad
V' = V + \delta^2.
```

Here $V$ is deliberately nuts-rs's sum of squared *old-mean* differences, not
the conventional Welford $M_2$. The common $(n-1)^{-1}$ scale cancels in the
draw-to-gradient variance ratio.

The update ordering is part of the kernel contract:

1. Update current draw, current gradient, background draw, and background
   gradient estimators.
2. If requested, move the just-updated background statistics into current
   ownership and reset the background count.
3. Once current owns at least three samples, adapt the diagonal transform.

For a valid component, the new diagonal scale is the fourth root of the
draw/gradient variance ratio:

```math
r_i = \sqrt{\frac{V^{(q)}_i}{V^{(g)}_i}},
\qquad
s_i = \sqrt{\operatorname{clamp}(r_i, 10^{-20}, 10^{20})}.
```

Zero, infinite, or NaN ratios preserve the previous component scale, matching
upstream `fill_invalid = None`. A successful adaptation still recomputes the
center and log determinant and increments the transformation id. Rejected
draws leave all four estimators unchanged but may still re-run the transform
from already-owned statistics.

## The mathematical kernels and their compiled forms

Use the pane tabs to inspect the exact authored source, compiler-generated
Julia, and selected have→want DAG. **Compare all** places the three views side
by side.

```@eval
Main.ReactiveKernelsDocs.render_nutpie_diagonal_adaptation()
```

The physical expectations are independent of the Julia candidate. The
standalone Rust oracle
[`benchmark/nutpie_diag_oracle.rs`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/nutpie_diag_oracle.rs)
consumes explicit raw positions and gradients with no RNG, and its byte-exact
output is committed in
[`benchmark/nutpie_diag_oracle.toml`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/nutpie_diag_oracle.toml).
Native execution matches that oracle, while the isolated Reactant acceptance
test compiles the same two `PreparedKernel`s without a nutpie-specific adapter.
