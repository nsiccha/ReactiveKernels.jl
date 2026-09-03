# Sum-to-zero effects and superpopulation recovery

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

A sum-to-zero parameterization removes the common-shift ambiguity between an
intercept and a vector of varying effects. The interesting part here is not a
standalone constraint API: it is how one inline model kernel exposes the
constrained effects and then recovers a draw in the original superpopulation
parameterization as an independently prepared downstream cut.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/sum_to_zero.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/sum_to_zero.jl).
It reuses the Normal and Cauchy distribution kernels introduced by the earlier
[Eight Schools example](eight-schools.md), while keeping every operation new to
this example directly inside one `@kernel model(...)` definition.

## What the constraint removes

Write an unconstrained superpopulation model as

```math
\begin{aligned}
\alpha_{\mathrm{bayes}} &\sim \operatorname{Normal}(0, s_\alpha), \\
a_{\mathrm{bayes},k} &\sim \operatorname{Normal}(0, \tau), \\
y_k &\sim \operatorname{Normal}
  (\alpha_{\mathrm{bayes}} + a_{\mathrm{bayes},k}, \sigma_k).
\end{aligned}
```

Let `m = mean(a_bayes)`. The identified representation is

```math
a_{\mathrm{s2z}} = a_{\mathrm{bayes}} - m,
\qquad
\alpha_{\mathrm{s2z}} = \alpha_{\mathrm{bayes}} + m,
\qquad
\sum_k a_{\mathrm{s2z},k} = 0.
```

This preserves every fitted location:
`α_bayes + a_bayes == α_s2z + a_s2z` componentwise.

The packed input contains `(α_s2z, log_τ, effects_free...)`, with `K - 1`
free effect coordinates. The kernel maps those coordinates into `K` zero-sum
effects with Stan's single-loop orthonormal pivot construction. It is O(K),
allocates only the result vector, and never constructs a dense basis matrix.

Because the columns of the map are orthonormal, its intrinsic log Jacobian is
zero. The source names this separately as `sum_to_zero_log_jacobian = 0.0`.
The later `subspace_normalization = log_τ` has a different role: componentwise
Normal log densities contribute `K` scale normalizers for a `K - 1`
dimensional object, so this term restores the parameter-dependent part of the
lower-dimensional prior normalization. It is not a transform Jacobian.

## The executable kernel

The panel below is built from the package's exact source authority. **Raw
input** contains the full inline `@kernel` definition and its `prepare` call;
**Generated kernel** is derived from the executed `PreparedKernel`; and
**Compute DAG** is the exact selected plan. The docs build executes the shown
call and checks its output again, so this page cannot silently drift into an
illustrative snippet.

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :SumToZeroExample, :SUM_TO_ZERO_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_sum_to_zero!,
)
```

The displayed interaction requests `parameters`, `prior`, and `likelihood`,
matching the main Eight Schools page. Both authored plates lower to scalar
accumulator loops in generated native code; pointwise buffers are absent
because no pointwise result is requested. Normal and Cauchy endpoint bodies are
inlined, there is no nested prepared-kernel dispatch, and none of the recovery
equations appear in this hot density cut.

## Prepare only the recovery

The constrained result is an ordinary named tuple:

```julia
constrain_kernel = prepare(model;
    have = :unconstrained,
    want = :parameters,
)
parameters = constrain_kernel(unconstrained)
```

The same build-executed source then prepares and calls recovery from that tuple,
without re-running the packed transform or any probability calculation:

```julia
recovery_kernel = prepare(model;
    have = (:parameters, :α_prior_sd, :reconstruction_innovation),
    want = :superpopulation,
)

recovered = recovery_kernel(parameters, 5.0, randn())
```

The caller supplies the standard-Normal innovation. Randomness therefore stays
outside the pure graph: a caller may draw afresh, replay a fixed value, or batch
the same prepared recovery kernel.

Conditional on `α_s2z` and `τ`, define

```math
v_m = \frac{\tau^2}{K}, \qquad
w = \frac{v_m}{s_\alpha^2 + v_m}, \qquad
v_c = \frac{s_\alpha^2 v_m}{s_\alpha^2 + v_m}.
```

For an innovation `z ~ Normal(0, 1)`, the kernel reconstructs

```math
\begin{aligned}
m &= w\,\alpha_{\mathrm{s2z}} + \sqrt{v_c}\,z, \\
\alpha_{\mathrm{bayes}} &= \alpha_{\mathrm{s2z}} - m, \\
a_{\mathrm{bayes}} &= a_{\mathrm{s2z}} + m.
\end{aligned}
```

The generated recovery cut contains only these scalar equations and one vector
broadcast for `a_bayes`. The transform, observations, priors, likelihood, and
both Jacobian terms are absent from that prepared function.

No benchmark is run or rendered on this page. Its executable contract is the
kernel source, the requested graph cuts, their numerical invariants, and the
generated native structure.

The transform follows the
[Stan sum-to-zero transform](https://mc-stan.org/docs/reference-manual/transforms.html#sum-to-zero-transforms),
and the recovery setup follows Sean Pinkney's
[StanCon 2026 sum-to-zero material](https://github.com/spinkney/open-talks/tree/main/sum-to-zero-stancon26).

Run the walkthrough from the repository root:

```sh
julia --project=packages -e 'using ReactiveKernelsPPLExamples; ReactiveKernelsPPLExamples.SumToZeroExample.demo()'
```
