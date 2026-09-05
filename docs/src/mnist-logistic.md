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
the per-observation likelihood reuses a [`categorical_logit`](distributions.md)
object — a softmax categorical over one logits vector — exactly as Eight
Schools reuses `normal` per observation. The model says this directly as
`plate(eachcol(logits), y)`: ordinary Julia executes the scalar plate, while
Reactant lowers the same authored plate generically as batched column slices
and traced gathers. The numerically stable log-sum-exp normalizer remains in
the distribution object, not in the model. There is one likelihood expression,
with no compiler-specific alternate body or private MNIST adapter.

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

## Optimized variant (vcat-free)

The idiomatic model above materializes the padded `[0; logits]` matrix on
every evaluation — exactly like the idiomatic Turing baseline. The optimized
variant lands **alongside** it rather than replacing the idiomatic comparison:
the reference class moves inside the reference-coded
[`categorical_logit_ref`](distributions.md) distribution object (it treats class
1 as an implicit zero-logit reference with a stable
`logaddexp(0, logsumexp(·))` normalizer), so the likelihood runs directly over
the `(C-1) × N` nonreference-logits matrix and the padded matrix is never
built. The two graphs agree to machine
precision on every boundary and outcome.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :MNISTLogisticExample, :MNIST_LOGISTIC_OPTIMIZED_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_mnist_logistic!,
)
```

## Primal capability and log-density performance

The complete suite uses the same ten RK configurations as Eight Schools:
ordinary native primal, nonallocating native primal, Reactant primal, native
value-and-gradient, and Reactant value-and-gradient, each with data unbound or
fixed during preparation. Fixed `X`, `y`, and `num_classes` are the public
partial-evaluation variant. With two model sources, the sampler headline is a
20-cell matrix from the packed coefficient vector to the full joint; every one
must be supported. Nonallocating AD and nonallocating Reactant remain explicit
API exclusions until those public kernel types compose.

The receipt prepares the authored graphs from two starting boundaries —
`Packed unconstrained` (a sampler's flattened `[vec(W); b]` vector) and
`Structured (W, b)` — and asks for four model outcomes. The long-form primal
receipt includes ordinary and nonallocating RK, both unbound and partially
evaluated, for both model sources. `prepare_nonallocating` runs those same
graphs through the optional MutatingFunctions destination-passing pass with
caches seeded before timing. The manual cells are a direct handwritten Julia
control. `Turing idiomatic` uses Turing's public
interfaces on the idiomatic model — a fixed-transform `LogDensityFunction`
for packed parameters and `logjoint` / `logprior` / `loglikelihood` for
structured parameters — and `Turing optimized` uses the same interfaces on
the heavier-optimized model shown under Baseline implementations below. Both
Turing likelihoods are single `@addlogprob!` terms, so Turing has no public
per-observation pointwise view; those cells stay blank rather than inventing
an equivalent. Data loading, graph preparation, and transform discovery are
outside the timed region. The presentation leads with the sampler-relevant
packed boundary and splits joint, prior, likelihood, and pointwise results into
separate plots and tables. Every table includes runtime relative to RK
idiomatic; the complete modifier × boundary inventory remains in the linked
receipt.

```@eval
Main.ReactiveKernelsDocs.render_mnist_logistic_benchmarks()
```

The exact-pin [PracticalBayes external comparator](practicalbayes.md) adds both
public PracticalBayes model styles, explicit unsupported boundaries, and the
same full raw-MNIST protocol in its separately resolved receipt.

Displayed runtimes use three significant digits and report the minimum of ten
independent BenchmarkTools minimum-time rounds — the uncontended-cost
estimator. Exact values, medians, every raw round, and the complete capability
matrix stay in the receipt. Every supported cell must agree numerically with
the handwritten control before it is timed.

## Wren-compatible PCA-40 primal comparison

The second matrix runs the same RK, handwritten Julia, and Turing public
interfaces on Wren's workload shape: the first 1,000 standard MNIST training
images projected onto the top 40 unwhitened principal components fitted on the
complete 60,000-image training split. The generated matrix is checked against
the private Wren reference CSV to roundoff; the CSV itself is not committed.

```@eval
Main.ReactiveKernelsDocs.render_mnist_logistic_wren_benchmarks()
```

This is a separate receipt, not a replacement for the full 60,000×784 result,
so both workload scales retain the identical 2×4 capability matrix and parity
gates.

### Baseline implementations

```@eval
Main.ReactiveKernelsDocs.render_mnist_logistic_baselines()
```

Reproduce the pinned receipt from a clean detached checkout (this loads the full
MNIST training split via MLDatasets):

```sh
julia --startup-file=no benchmark/mnist_logistic_comparison.jl \
  --output=benchmark/receipts/mnist-logistic-primal-v3.toml
julia --startup-file=no benchmark/receipts/validate_mnist_logistic.jl \
  benchmark/receipts/mnist-logistic-primal-v3.toml

julia --startup-file=no benchmark/mnist_logistic_comparison.jl \
  --dataset=wren-pca40 --wren-reference=/path/to/mnist.csv \
  --output=benchmark/receipts/mnist-logistic-wren-pca40-v1.toml
julia --startup-file=no benchmark/receipts/validate_mnist_logistic.jl \
  benchmark/receipts/mnist-logistic-wren-pca40-v1.toml
```
