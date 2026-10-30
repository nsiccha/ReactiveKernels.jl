# Declarative PPL kernel: GARCH(1, 1)

This example ports the `garch11` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posterior `garch-garch11`)
into the declarative-`@kernel` style: the model is authored inline in one source
string, the Normal endpoint is reused from the shared distribution objects, and
the constrained parameters are a plain NamedTuple. Like
[ARMA(1,1)](arma11.md) its distinctive structure is a **sequential recursion** —
here the conditional standard deviations `σₜ` — authored with the
[`scan` primitive](scan.md), which threads a carry through the series as a
retained loop.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/garch11.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/garch11.jl).
It models a scalar series `y₁,…,y_T` with mean `μ`:

```math
\begin{aligned}
\sigma_1 &= \texttt{sigma1}, \qquad
\sigma_t = \sqrt{\alpha_0 + \alpha_1 (y_{t-1} - \mu)^2 + \beta_1\, \sigma_{t-1}^2}, \\
y_t &\sim \operatorname{Normal}(\mu, \sigma_t).
\end{aligned}
```

There are no explicit priors: Stan's parametrization is improper-flat on the
constrained support, so the density is the likelihood plus the transform
Jacobian. The unconstrained vector is `(μ, u_α0, u_α1, u_β1)`. The support
transforms match Stan exactly and are constrained in declaration order, so
`β₁`'s upper bound is the *already-constrained* `α₁`:

```math
\alpha_0 = e^{u_{\alpha_0}}, \quad
\alpha_1 = \operatorname{logistic}(u_{\alpha_1}) \in (0,1), \quad
\beta_1 = (1-\alpha_1)\,\operatorname{logistic}(u_{\beta_1}) \in (0, 1-\alpha_1).
```

```text
unconstrained ─► μ, u_α0, u_α1, u_β1 ─► α0, α1, β1 (support transforms + Jacobian)
   │                                        │
sigma1 ─┬─► sigma (scan recursion) ─────────┼─► forecast_sigma (σ_{T+1})
        │   (a retained loop)               └─► pointwise ─► likelihood
y ──────┘
likelihood + log Jacobian ──► unconstrained log density
```

The conditional sd sequence `sigma`, the constrained `parameters`, and a
one-step-ahead volatility forecast `forecast_sigma` are all selectable nodes.

The panel below shows three views of this model: **Raw input** (the source), a
readable **Generated kernel**, and the **Compute DAG**.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :GARCH11Example, :GARCH11_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_garch11!,
)
```
