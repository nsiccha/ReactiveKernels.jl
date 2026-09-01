# Declarative PPL kernel: MNIST multinomial logistic

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

This example scales the Eight Schools pattern to a real dataset: a
multinomial-logistic (softmax) classifier for MNIST. It is assembled under the
same rule — the density is composed entirely from reusable, transparently
authored distribution objects, with no hand-written density code.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/mnist_logistic.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/mnist_logistic.jl).
It implements

```math
\begin{aligned}
W_{cd} &\sim \operatorname{Normal}(0, 1), \\
b_c &\sim \operatorname{Normal}(0, 1), \\
\eta_{cn} &= (W x_n + b)_c \quad (c = 1, \dots, C-1), \qquad \eta_{0n} = 0, \\
y_n &\sim \operatorname{Categorical}\bigl(\operatorname{softmax}(\eta_{\cdot n})\bigr).
\end{aligned}
```

Class 0 is the reference (its logit is fixed to 0), so the coefficients are
`W` (a `(C-1) × D` matrix) and `b` (a `C-1` vector). Every coefficient is
unconstrained, so there is no support transform and no Jacobian; the packed
sampler vector is simply `[vec(W); b]`.

## Composed from reusable distribution objects

The coefficient prior reuses the shared
[`normal`](distributions.md) object over the flattened coefficient vector, and
the per-observation likelihood reuses a `categorical_logit` object — a softmax
categorical over a logits vector — exactly as Eight Schools reuses `normal` per
observation. The `categorical_logit` object is authored the same transparent
way as every other distribution kernel (its normalizer is a numerically stable
log-sum-exp that lives inside the object, not in the model). The model source
therefore contains only the linear predictor, the reference row, and calls into
those two objects; there is no hand-written Normal or softmax formula.

The panel below shows the **Raw input** (the source), the readable **Generated
kernel** derived from the executed kernel and selected plan, and the **Compute
DAG**. The docs and tests execute this one source on a small committed
real-MNIST fixture; the benchmark below runs the same graph on the full MNIST
training split.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :MNISTLogisticExample, :MNIST_LOGISTIC_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_mnist_logistic!,
)
```

The packed unconstrained (`unconstrained`), structured (`W`, `b`), and
likelihood-only boundaries are HAVE cuts of the same graph; the coefficient
prior, pointwise likelihood, summed likelihood, and the joint density are
selectable named nodes. Asking for the prior alone prunes the observation
likelihood — the dense `W x` linear predictor never runs:

```julia
prior = prepare(model; have = :unconstrained, want = :prior)(unconstrained)
```

## Primal capability and log-density performance

The benchmark below prepares the authored graph from two starting boundaries —
`Packed unconstrained` (a sampler's flattened `[vec(W); b]` vector) and
`Structured (W, b)` — and asks for four model outcomes. The RK cells are
HAVE/WANT cuts of one graph. The manual cells are a direct handwritten Julia
control. Turing uses its native public interfaces: a fixed-transform
`LogDensityFunction` for packed parameters and `logjoint` / `logprior` /
`loglikelihood` for structured parameters. The model's likelihood is a single
`@addlogprob!` term, so Turing has no public per-observation pointwise view;
that cell stays blank rather than inventing an equivalent. Data loading, graph
preparation, and transform discovery are outside the timed region.

```@eval
Main.ReactiveKernelsDocs.render_mnist_logistic_benchmarks()
```

Each cell reports the median of ten independent BenchmarkTools minimum-time
rounds together with steady-state allocation bytes and counts. Every supported
cell must agree numerically with the handwritten control before it is timed.

Reproduce the pinned receipt from a clean detached checkout (this loads the full
MNIST training split via MLDatasets):

```sh
julia --startup-file=no benchmark/mnist_logistic_comparison.jl \
  --output=benchmark/receipts/mnist-logistic-v1.toml
julia --startup-file=no benchmark/receipts/validate_mnist_logistic.jl \
  benchmark/receipts/mnist-logistic-v1.toml
```
