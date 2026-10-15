# Distribution kernels

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

Distribution kernels are ordinary transparent `@kernel` graphs. A standard
family owns operations such as `logpdf`, `cdf`, and `quantile`; one shared
location-scale kernel supplies translation, scaling, standardization, and the
inverse standardization route. `Distributions.jl` appears only in tests and
benchmarks as an independent oracle—the generated kernels below do not call it.

Each panel below shows the **Raw input** (the source), a readable **Generated
kernel** derived from the executed kernel and selected plan, and its **Compute
DAG**. The exact compiled AST remains available through `code_expr`.

## One location-scale graph, several families

The complete authoring surface is below. `standard_normal`, `standard_cauchy`,
and `standard_laplace` contain only their standard-family mathematics.
`location_scale` supplies the shared transformation once. The final three lines
bind concrete, immutable kernel objects—`normal` is graph data with named ports,
not a closure hidden behind a global function.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.CONTINUOUS_SOURCE,
)
```

The simple path stays simple: `prepare(normal.logpdf)` asks for the distinguished
result of that endpoint. Inside another `@kernel`, `normal.logpdf(x)` splices the
same transparent graph into the caller; it does not insert an opaque runtime
call.

Every named boundary remains selectable. One plan can request both the density
and the CDF from the one object, and both endpoints reuse the single
`standardized(x)` node (structural CSE). The build-executed panel below asserts
that exactly one `standardized` recipe appears in the joint plan:

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.EXTRACT_JOINT_SOURCE,
)
```

`quantile(p)` follows the explicit inverse edge `inv(standardized, z)`. The graph
also contains both directions of `log_scale = log(scale)` — the reverse edge
is provisioned automatically — so a caller may start from either
representation. Explicit HAVE is authoritative:
supplying `scale` derives `log_scale`, supplying `log_scale` derives `scale`, and
supplying both recomputes and validates neither. The next panel prepares all
three routes into the same `normal.logpdf` and asserts, during the build, that
their plans differ only by the derived port:

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.HAVE_ROUTES_SOURCE,
)
```

Included child ports retain scoped names rather than colliding with the outer
result. For example, `normal[Symbol("standard.logpdf")]` addresses the standard
family term, while `normal.logpdf` addresses the adjusted location-scale
result; either can be passed to `extract`.

## One authored likelihood, either result boundary

The likelihood is ordinary kernel source. Its one `plate` node applies the
transparent distribution object pointwise, and its ordinary `return` selects
the summed result:

```julia
@kernel normal_loglik(x, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        normal(li, si).logpdf(xi)
    end
    return sum(pointwise)
end
```

`prepare(normal_loglik)` requests the distinguished return. An alternate cut,
`extract(normal_loglik; want = :pointwise)`, requests the named pointwise array;
`want = (:pointwise, :__return__)` requests both. These are three lowerings of
the same authored plate node, not parallel formulas. Total-only execution fuses
the sum into the plate loop without materializing `pointwise`; the joint cut
fills `pointwise` and accumulates the total in that same traversal.

Plate arguments use Julia broadcast semantics: scalars repeat, compatible array
dimensions zip or expand, and incompatible shapes fail before endpoint
execution. A whole array that every scalar body invocation reads is not a plate
argument: the cell closes over it.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.VECTORIZED_SOURCE,
)
```

The broadcast rules are worth making concrete. In the panel below one plate zips
the observation vector with a per-observation location, repeats the scalar scale,
and reads a whole vector by closing over it. The build checks the result
against the equivalent hand-written broadcast:

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.BROADCAST_CLOSURE_SOURCE,
)
```

Dependency-aware lowering also hoists complete invariants out of the broadcast
loop. When `location` and `scale` are declared scalars shared across the batch,
the scale-only `log_scale` term is emitted once in the loop preamble, while the
observation-dependent standardization stays in the loop at the shallowest
boundary that binds it. Through a port without a declared type, whose value can
be a scalar or an axis, the term is still emitted once, inside the loop: it runs
at the first coordinate when the value holds no axis and is reused after that.
The generated-kernel pane below is the exact plan built during this docs build;
the source asserts the hoisted `log_scale` op precedes the loop:

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.INVARIANT_HOISTING_SOURCE,
)
```

## Other standard location-scale families

Cauchy and Laplace use exactly the same `location_scale` graph. Their examples
only select the family object and prepare its `logpdf` endpoint; neither repeats
the transform or normalization plumbing.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.CAUCHY_SOURCE,
)
```

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.LAPLACE_SOURCE,
)
```

## Discrete distribution objects

`bernoulli` is the same kind of object as `normal`: `bernoulli.logpdf`,
`bernoulli.cdf`, and `bernoulli.quantile` are transparent endpoints. Its graph
contains both `p` and `logit`, so callers can supply either representation while
the log-density endpoint keeps the stable `-log1pexp(∓logit)` expression.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.DISCRETE_SOURCE,
)
```

`poisson` and `binomial` are the count families. `poisson.logpdf` accepts the
rate or its log—the canonical log link—and `binomial.logpdf` accepts a fixed
trial count `n` with either the success probability or its logit. Both guard the
observation domain to `-Inf` with a lazy branch and expose `logpdf`, `cdf` and
`ccdf`; neither exposes a `quantile`, because a discrete quantile needs an
iterative search rather than a pure straight-line endpoint. The generic `plate`
API lifts each `.logpdf` over independent counts without family-specific code.

Their `logpdf` equals the sum of two further endpoints: `lognormalizer(y)`, the
data-only term (the log binomial coefficient, or `-log(y!)`), and
`logkernel(y)`, the terms that read the parameters. `zero_inflated_binomial` has
the same pair. `zero_inflated_binomial_logit_zi(; n, logit, zi_logit)` accepts a
zero-inflation logit directly and composes the existing Binomial graph with
log-scale mixture weights. Its `logkernel`, `lognormalizer` and `logpdf`
endpoints keep finite source logits through rounded probability tails. The
ordinary `zero_inflated_binomial(; n, logit, zi)` accepts a probability,
including exact `zi=0` and `zi=1` inputs. A plate of `lognormalizer` cells
over bound counts and trials is a bind-time value, so a likelihood that sums
`logkernel` cells and adds that
plate evaluates no `loggamma` per call. Keep the two in separate plates:
summing them inside one differentiated cell trips a released-Enzyme reverse
compilation failure, which is also why `logpdf` is written out.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.POISSON_SOURCE,
)
```

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.BINOMIAL_SOURCE,
)
```

## One mixture component for a whole observed event

`bernoulli_product_mixture_logdensity` and
`binomial_product_mixture_logdensity` are function-shaped graph kernels in
`ReactiveKernelsDistributionKernels.DistributionKernelSources`. Each accepts
one observed vector, a probability matrix whose rows are observation
positions and columns are components, and one weight per column. The Binomial
version also accepts a trial-count matrix of the same shape. Each component
adds its cell log densities first; the outer mixture then computes
`logsumexp(log(weight[k]) + component_score[k])`. Thus one component governs
the complete vector. The lower-level `product_mixture_logdensity(scores,
weights)` performs the outer fold when component scores come from another
family. All three are transparent RK kernels with data-sized loops.

```julia
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    bernoulli_product_mixture_logdensity

event = Bool[true, false, true]
probabilities = [0.8 0.1; 0.2 0.7; 0.5 0.3]
weights = [0.4, 0.6]
loglik = prepare(bernoulli_product_mixture_logdensity)
loglik(event, probabilities, weights)
```

A PPL model can pass one complete response vector to `LogDensity` with this
kernel. An outer `@plate` may select several such vectors; each outer cell
then has its own marginalized component, and the inner RK plates retain the
positions within that event. The component scores stream through the outer
`logsumexp` fold, so an event allocates no score vector, and an argument that
does not vary by event (a shared probability matrix or weight vector written
in the cell) is computed once, before the outer loop.

`bernoulli_logit_product_mixture_logdensity(observed, logits, weights)` and
`binomial_logit_product_mixture_logdensity(observed, trials, logits, weights)`
take a log-odds matrix of the same shape instead of probabilities. Each cell
reads the logit route of `bernoulli(; logit)` or `binomial(; n, logit)`, so a
component defined on the logit scale stays in log space. Passing
`logistic.(eta)` to the probability laws rounds the probability first: the
absent outcome's `log1p(-p)` loses accuracy as `eta` grows and is `-Inf` from
about `eta = 37`, where the exact log mass is `-eta`. A component that can
never produce a success (a structural zero) has logit `-Inf`:

```julia
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    bernoulli_logit_product_mixture_logdensity

event = Bool[true, false, true]
logits = [0.5 -Inf; 40.0 -Inf; 2.0 -Inf]
loglik = prepare(bernoulli_logit_product_mixture_logdensity)
loglik(event, logits, [0.6, 0.4])
```

`zero_inflated_binomial_logit_product_mixture_logdensity(observed, trials,
logits, zi_logits, weights)` gives each component zero-inflated Binomial
cells. `logits[i, k]` is the success log-odds and `zi_logits[i, k]` the
log-odds of a structural zero, both read through
`zero_inflated_binomial_logit_zi(; n, logit, zi_logit)`, so neither
probability is rounded. A data-augmented community occupancy model uses it
for each species row: a member species occupies each site with occupancy
log-odds `psi` (zero-inflation log-odds `-psi`) and is then detected with
log-odds `theta` on each of `K` visits, while a non-member is never
detected (success logit `-Inf`):

```julia
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    zero_inflated_binomial_logit_product_mixture_logdensity

counts = [0, 3, 0, 1]                      # detections at four sites
psi, theta, omega = 0.8, -0.4, 0.6         # occupancy, detection, membership
loglik = prepare(zero_inflated_binomial_logit_product_mixture_logdensity)
loglik(counts, fill(18, 4, 2), hcat(fill(theta, 4), fill(-Inf, 4)),
    hcat(fill(-psi, 4), fill(-Inf, 4)), [omega, 1 - omega])
```

## Hidden Markov sequences

`hmm_logdensity(log_emissions, transition, initial)` is the forward
algorithm: the log density of one observed sequence, summed over every path of
a hidden state that moves as a Markov chain. `log_emissions[t, k]` is step
`t`'s emission log density under state `k`. `transition[i, j]` weighs a move
from state `i` to state `j`, as a row-stochastic matrix; a
`states × states × (steps - 1)` array gives the move into step `t` at index
`t - 1`. `initial[k]` weighs the first state. Weights are used as given, so
all-ones initial weights reproduce a forward pass that starts from the bare
first emission, and singleton axes broadcast: a `states × 1 × (steps - 1)`
array weighs each move by the previous state only.

`normal_hmm_logdensity(observed, locations, scales, transition, initial)` and
`exponential_hmm_logdensity(observed, scales, transition, initial)` compute
the emissions with the Normal and Exponential families and pass them to
`hmm_logdensity`. `observed` is a vector, or a `steps × streams` matrix whose
streams are independent given the state. Emission parameters broadcast over
`states × streams × steps`: a vector holds one value per state, a matrix one
per state and stream, and a third axis lets a parameter change with the step.
Exponential parameters are scales, as for `exponential`.

```julia
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal_hmm_logdensity

y = [0.3, 2.9, 3.1, 0.2, -0.1]
transition = [0.8 0.2; 0.3 0.7]
loglik = prepare(normal_hmm_logdensity)
loglik(y, [0.0, 3.0], [1.0, 0.5], transition, [0.5, 0.5])
```

The pass is one `scan` with a `history` (see [the scan page](scan.md)) over the
`(step, state)` cells: each cell reads the previous step's cells from the scan's
own output, so no step allocates a carried vector, and the derivatives are
ordinary Enzyme derivatives of these graphs, with no hand-written rule. A PPL
model passes the whole sequence to `LogDensity` with one of these kernels. For
another emission family, author a kernel that fills the `steps × states`
log-emission matrix in a plate and returns `hmm_logdensity`.

## Intrinsic CAR fields

`icar_logdensity(phi, node1, node2)` is the unit-scale intrinsic conditional
autoregressive (ICAR) log density of a node vector `phi` on an undirected graph
whose edge `e` joins nodes `node1[e]` and `node2[e]`, each edge listed once:
`-1/2 * sum((phi[node1] .- phi[node2]) .^ 2)`. It is the Gaussian Markov random
field whose precision is the graph Laplacian. The law is improper, because
adding a constant to every node of a connected component leaves it unchanged,
so it has no normalizer. A model identifies the field with a sum-to-zero
constraint or through its likelihood.

```julia
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    icar_logdensity

phi = [0.3, -0.1, 0.7, 0.2]
logprior = prepare(icar_logdensity)
logprior(phi, [1, 1, 2, 3], [2, 3, 3, 4])
```

One plate runs over the edges, and each cell reads `phi` whole, so no gathered
vector is built. The derivatives are ordinary Enzyme derivatives of this
graph. In a PPL model, `phi[1:N] ~ ICAR(node1, node2)` declares the field with
this prior.

## Positive-support and bounded families

LogNormal, Exponential, Geometric, and Uniform illustrate shapes that are not all
plain location-scale lifts. They still use the same object surface: each family
owns its supported `logpdf`, `cdf`, and `quantile` endpoints, with named
intermediate ports left available to `extract`.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.LOGNORMAL_SOURCE,
)
```

Exponential accepts scale or log scale, Geometric accepts success probability or
its logit, and Uniform accepts runtime bounds. Explicit HAVE values remain
authoritative in every case. The ordinary `plate` API lifts each `.logpdf`
endpoint over independent observations without family-specific batching code.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.EXPONENTIAL_SOURCE,
)
```

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.GEOMETRIC_SOURCE,
)
```

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.UNIFORM_SOURCE,
)
```

`gamma` and `beta` complete the continuous gallery. `gamma.logpdf` takes a shape
with the rate, its reciprocal scale, or the log rate as equivalent authoritative
HAVE routes—supplying one derives the others; `beta.logpdf` takes two positive
shapes on the unit interval. Both expose `logpdf`, `cdf`, and `quantile`; the
continuous inverse routes through the regularized incomplete-gamma / incomplete-
beta inverse directly.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.GAMMA_SOURCE,
)
```

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.BETA_SOURCE,
)
```

The scalar-family native/Reactant comparison now lives under
[Distribution kernels through Reactant](distributions-reactant.md#scalar-families),
while this page remains the source authority for the objects it measures.

## Structured families: multivariate Normal and AR(1)

Non-scalar families are method-bearing objects too: `mvnormal.logpdf` and
`ar1.logpdf` are the currently implemented endpoints. They are not scalar
`plate`s. The coordinates of a multivariate observation—and the time steps of
an autoregressive series—are coupled inside one mathematical operation.

The multivariate Normal is authored once. Ordinary repeated assignments expose
alternative graph paths that produce the same
`half_logdet_cov` and `quadratic` ports from a covariance matrix, its Cholesky
factor, a precision matrix, or its Cholesky factor. `prepare` starts at whichever
representation is in HAVE and selects only that route to `mvnormal.logpdf`;
callers do not convert everything to one privileged parametrization first.
Repeated assignments are the complete authoring surface for these alternatives.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.MVNORMAL_SOURCE,
)
```

These rows are derived from the four `PreparedKernel`s built by that source. When
you start from a full matrix the plan includes the factorization; when you start
from an already-factored form the plan skips it.

```@eval
Main.ReactiveKernelsDocs.render_mvn_parametrization_plans(
    Main.DistributionExamples.MVNORMAL_SOURCE,
)
```

The `ar1` object similarly treats the complete sequence as one value. Its
`logpdf` endpoint uses lagged residuals for an O(T) density, including the
stationary initial-state term and an explicit `-Inf` result outside
`abs(ϕ) < 1`. Scale and log scale are equivalent HAVE boundaries here as well.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.DistributionExamples.AR1_SOURCE,
)
```

Both examples then apply `replica(...; batched = :x)`: a matrix holds independent
vectors or independent series, one per column. The inner coordinate/time axis
stays coupled, while the new trailing axis runs the whole kernel once per column.

The structured native/Reactant comparison and its unsupported traced-array cells
now live under
[Distribution kernels through Reactant](distributions-reactant.md#structured-families).

## Native and Reactant benchmark evidence

The scalar-plate comparison, compiled-call amortization, and machine-readable
receipts now live together under
[Distribution kernels through Reactant](distributions-reactant.md#scalar-plate-and-call-boundary-amortization).
This page remains the sole displayed mathematical and executable source
authority for the `normal.logpdf` object and its generic plate.
