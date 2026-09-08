# Declarative PPL kernel: hierarchical Gaussian process

This is the posteriordb `hierarchical_gp` model (posterior
`state_wide_presidential_votes-hierarchical_gp`): a hierarchical Gaussian-process
model of state presidential vote shares (Trangucci, StanCon 2017). At **933
unconstrained dimensions** it is the largest of the GP batch and exercises the
most graph machinery at once: a Dirichlet variance decomposition through Stan's
**ILR simplex transform**, two per-year Cholesky GPs, year/state/region random
effects, and index gathers — all matching BridgeStan to machine precision on the
full real data (`N = 550`) loaded through `PosteriorDB.jl`.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/hierarchical_gp.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/hierarchical_gp.jl).
The variance `tot_var` is split by a `simplex[17]` `prop_var` into the
year/region/state/GP/error variances; each per-year GP is a sum of a long- and
short-range exponential-quadratic kernel over the year axis, applied
non-centered through the Cholesky factor; and the observation mean gathers the
random effects and the two GP matrices:

```math
\begin{aligned}
\operatorname{prop\_var} &\sim \operatorname{Dirichlet}(2\cdot\mathbf{1}_{17}), \quad
\operatorname{tot\_var} \sim \operatorname{Gamma}(3,3), \quad
\ell_\bullet \sim \operatorname{Weibull}(30, \cdot), \\
\operatorname{GP} &= \operatorname{chol}\!\big(K_{\text{long}} + K_{\text{short}} + 10^{-6} I\big)\, Z, \\
y_n &\sim \operatorname{Normal}\!\big(\mu + \text{year/state/region RE} + \operatorname{GP}[\dots],\ \sigma_{\text{err}}\big).
\end{aligned}
```

Stan's simplex transform is `softmax(sum_to_zero_constrain(·))`; the sum-to-zero
map is the linear ILR "pivot coordinates" basis (a fixed `17×16` matrix, an
in-graph node folded by `bound=`), and its exact log-Jacobian is
`Σ log(prop_var) + ½ log 17`. The positive scale parameters use the `log`
transform with the exact Jacobian. The year axis, squared distances, ILR basis,
and column-major gather indices are all in-graph shape-derived designs folded by
partial evaluation.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :HierarchicalGPExample, :HIERARCHICAL_GP_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_hierarchical_gp!,
)
```

The native primal and the plain-Enzyme reverse gradient match BridgeStan's
reference `.stan` (`propto = false`, `jacobian = true`) to machine precision on
the full 933-dimensional data.

## Reactant

The authored graph compiles and executes its **primal** through Reactant with
value parity — the ILR simplex, the dual Cholesky-factor matmuls, the reshape,
and the index gathers all lower through XLA (all data bound, only the
unconstrained draw traced).

The **compiled Reactant reverse gradient does not lower for this shape**: the two
per-year Cholesky factors hit the same upstream XLA / EnzymeMLIR adjoint gap for
`stablehlo.cholesky` as the [latent Poisson GP](gp-pois-regr.md), documented in
`reactivekernels-use` §7f. The supported AD path is the native primal plus the
native plain-Enzyme reverse gradient (both machine-precision accurate against
Stan), together with the Reactant primal.

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.HierarchicalGPExample.demo()'
```
