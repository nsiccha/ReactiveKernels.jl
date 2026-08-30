# Pathfinder approximation

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

Pathfinder builds a Gaussian variational approximation along an optimizer path.
The external fixture on this page follows Algorithms 3 and 4 of [Zhang et al.
(2022)](https://jmlr.org/papers/v23/21-0889.html) and is cross-checked against
[Pathfinder.jl 0.10.7 at production revision
`dba8c9a`](https://github.com/mlcolab/Pathfinder.jl/tree/dba8c9acc25f2905078d428ddd50b5d9276c3847):
recover a safe diagonal scale, apply an inverse-BFGS covariance update, form the
local Gaussian, estimate its ELBO, and retain the best candidate.

This is compiler evidence, not a sampler API exported by `ReactiveKernels`.
The reviewed source lives in
[`benchmark/pathfinder_kernel_authoring_fixture.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/pathfinder_kernel_authoring_fixture.jl),
with an independent, source-locked Python oracle under
[`benchmark/pathfinder_oracle/`](https://github.com/nsiccha/ReactiveKernels.jl/tree/main/benchmark/pathfinder_oracle).

## The mathematics that gets transpiled

The author writes one `@kernel`, `pathfinder_candidate`. Its explicit HAVE ports
contain the optimizer position and gradient, the retained quasi-Newton pair,
the current diagonal scale, the target log density, and caller-owned standard
normal draws. Its WANT ports expose the recovered scale, curvature decision,
covariance, local mean, ELBO work, score, and output draws.

Three parts of that source are the mathematical heart:

- **Curvature safeguard and α-recovery.** A pair is accepted only when
  ``s^T z > \epsilon z^T z``. Safe denominators keep the rejected branch finite
  under eager/select-style accelerator lowering, while the selected result
  remains exactly the previous diagonal scale.
- **Inverse-BFGS covariance.** The retained ``(s,z)`` pair updates
  ``\Sigma = (I-\rho s z^T)D(I-\rho z s^T)+\rho s s^T``. A rejected pair sets
  ``\rho=0``, reducing exactly to ``D=\operatorname{diag}(\alpha)``.
- **Local Gaussian and ELBO.** The kernel computes
  ``\mu=\theta+\Sigma\nabla\log p(\theta)``, uses a Cholesky factor for explicit
  caller-supplied draws, evaluates ``\log q``, and averages
  ``\log p-\log q`` to rank the candidate.

These are the same implementation boundaries used by Pathfinder.jl's
[`gilbert_init` and `lbfgs_inverse_hessians`](https://github.com/mlcolab/Pathfinder.jl/blob/dba8c9acc25f2905078d428ddd50b5d9276c3847/src/inverse_hessian.jl),
[`fit_mvnormals` and `rand_and_logpdf`](https://github.com/mlcolab/Pathfinder.jl/blob/dba8c9acc25f2905078d428ddd50b5d9276c3847/src/mvnormal.jl), and
[`elbo_and_samples`](https://github.com/mlcolab/Pathfinder.jl/blob/dba8c9acc25f2905078d428ddd50b5d9276c3847/src/elbo.jl).
The authored compiler fixture specializes the compact L-BFGS representation to
one retained history pair and makes randomness explicit; it is mathematical
and implementation-level evidence, not a line-by-line port of Pathfinder.jl's
optimizer, tasking, or multi-history driver.

The two panels below are built from that exact reviewed fixture during the docs
build. They prepare the same authored graph with two WANT boundaries. The first
transpiles only the curvature-safe inverse-BFGS geometry; the second transpiles
the complete local-Gaussian candidate and ELBO. Open **Raw input** to see the
authored mathematical `@kernel`, **Generated kernel** for the lowered Julia, and
**Compute DAG** for the selected have→want plan.

```@eval
Main.ReactiveKernelsDocs.render_pathfinder_kernels(@__MODULE__)
```

## What remains ordinary Julia

The optimizer and RNG policy stay outside the compiled candidate. The
`run_single_path` driver walks supplied positions and gradients, feeds one
retained pair into the prepared kernel at a time, threads `alpha_next`, and
selects the largest ELBO. Standard-normal tensors are HAVE values rather than a
hidden RNG effect, so native Julia and Reactant execute the same deterministic
mathematics.

The acceptance tests compare every native observable with the independent
physical oracle and compile the full candidate through Reactant. The external
corpus records the paper digest, the pinned Pathfinder.jl revision and
source-file digests, the oracle source digest, supported backend boundary, and
minimum native-plus-Reactant acceptance without adding Pathfinder to the
exhaustive ReactiveHMC corpus.
