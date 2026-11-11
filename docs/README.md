# Documentation

The reader knows Julia and may know Stan or brms. The site teaches graph
authoring, preparation and model queries; statistical model construction owned
by other packages is linked from the relevant example.

## Pages

### Start

- `index.md`: install ReactiveKernels and evaluate a complete kernel on the first screen.

### Guides

- `graphs.md`: choose input and output ports and prepare the computation between them.
- `bound-data.md`: bind fixed data while keeping changing inputs explicit.
- `position-batching.md`: apply one scalar kernel to a collection of positions.
- `scan.md`: express a sequential recurrence with a carried state.
- `automatic-differentiation.md`: prepare gradients and pullbacks with explicit active inputs.
- `rkppl.md`: declare a probabilistic model, bind observations and query its density.
- `nonallocating.md`: reuse owned kernel buffers and understand returned-value lifetimes.
- `visualization.md`: inspect a live plan and export its graph or generated expression.
- `jax.md`: emit a Python/JAX module and identify the execution boundary.
- `manual-derivative-rules.md`: derive an owned primitive's AD adapters from one mathematical graph.

### Examples

- `bound-regression.md`: separate data preparation, prediction and residual evaluation in one graph.
- `eight-schools.md`: evaluate a hierarchical density and inspect its named components.
- `discrete-latents.md`: compute conditional latent probabilities in a hidden Markov model.

### Reference

- `api.md`: find exported core APIs grouped by task and type.
- `ppl-api.md`: find exported PPL authoring, binding, coordinate and query APIs.
- `distribution-api.md`: find distribution kernels and their public construction surface.
- `constraints.md`: preserve the semantic and ownership requirements for implementations and examples.

## Build

Run `julia --project=docs docs/instantiate.jl`, then
`julia --project=docs docs/make.jl`. The build executes every runnable example, checks exported
docstrings, runs the documentation lint, and fails on warnings. It writes
figures, generated views and execution receipts into the build directory.
It does not fit statistical models or rerun performance benchmarks.

Research history and reproduction commands live under `benchmark/`, outside
the site navigation. They are linked from an example's Reproduce footer when
needed. Generated assets are not committed. The shared DocsKit integration
will replace package-local infrastructure when it becomes available; this
rewrite does not copy another package's theme or widgets.
