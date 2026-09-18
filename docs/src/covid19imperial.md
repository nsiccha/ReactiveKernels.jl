# Declarative PPL kernel: Imperial covid19 renewal model

This example ports the `covid19imperial` model from
[posteriordb](https://github.com/stan-dev/posteriordb) (posteriors
`ecdc0401-covid19imperial_v2` and `ecdc0501-covid19imperial_v2`) into the same
declarative-`@kernel` style as the [ARMA(1,1) example](arma11.md). One faithful
translation serves the whole pandemic family: the bundled
`covid19imperial_v2.stan` and `covid19imperial_v3.stan` are byte-identical, and
BridgeStan confirms both expose the same 51 unconstrained parameters with
identical names — the v3 posterior JSON's extra `lockdown`/`gamma` dimension
entries are stale metadata that no shipped `.stan` declares. The graph binds the
ecdc0401 dataset at load time and stays fully data-generic, so the sibling
ecdc0501 snapshot binds the same raw ports.

The complete runnable source is
[`packages/ReactiveKernelsPPLExamples/src/covid19imperial.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsPPLExamples/src/covid19imperial.jl).
Its distinctive structure is the **nonlinear renewal recurrence** behind the
Imperial report's Rt estimation. Per country `m`, the first `N0` infections are
imputed at level `y[m]`; afterwards each day's prediction multiplies a
susceptible-depletion factor with an `Rt` driven by mitigation covariates and a
convolution of past predictions against the serial interval `SI`. Expected
deaths convolve the same predictions against the infection-fatality kernel `f`,
with Stan's day-1 special case `E_deaths[1,m] = 1e-15 * prediction[1,m]`; the
user-supplied `EpidemicStart[m]:N[m]` observation window may legitimately start
inside the imputation days, so the graph computes the full `N2 × M` grid. The
imputation-day expected deaths have an exact closed form (every convolution
term is an imputed `y[m]`: `E_deaths[i,m] = ifr[m]·y[m]·cumsum(f)[i-1,m]`), and
the renewal recurrence is irreducibly sequential (susceptible depletion feeds
back into the next prediction), so it is authored inline as ONE batched
[`scan` primitive](scan.md) over days `i = N0+1:N2` × countries whose NamedTuple
carry threads a per-country shift-register prediction buffer plus cumulative
infections:

```math
\begin{aligned}
Rt[i,m] &= \mu_m \exp(-X_m[i,:]\,\alpha), \qquad \alpha = \alpha^{\mathrm{hier}} - \tfrac{\log 1.05}{6}, \\
\mathrm{conv}_i &= \sum_{d\ge 1} \mathrm{pred}[i-d,m]\,SI[d], \qquad
c_i = c_{i-1} + \mathrm{pred}[i-1,m], \\
\mathrm{pred}[i,m] &= \frac{\mathrm{pop}_m - c_i}{\mathrm{pop}_m}\, Rt[i,m]\, \mathrm{conv}_i, \\
E_{\mathrm{deaths}}[i,m] &= \mathrm{ifr}_m \sum_{d\ge 1} \mathrm{pred}[i-d,m]\, f[d,m].
\end{aligned}
```

```text
unconstrained ──► μ, α_hier, κ, y, φ, τ, ifr_noise (exp + Σu Jacobian)
X (raw, real) ──► XX reshape ──► Rt ──► scan over i = N0+1:N2 ──► renewal E_deaths
day 1: E_deaths[1] = 1e-15 · y (own node); days 2:N0: closed form ifr·y·cumsum(f)
SI, f, pop, deaths, EpidemicStart, N (raw) ──► mask, grids, logfactorial ─┘
day 1: E_deaths[1] = 1e-15 · y (own node)
deaths ~ NegBinomial2(E_deaths, φ) on the observed grid ──► log likelihood
log prior + log Jacobian + log likelihood ──► unconstrained log density
```

All raw-data preprocessing stays **in the graph** as named all-bound nodes: the
covariate reshape `XX = reshape(permutedims(X, (2,1,3)), N2*M, P)`, the full
observed-grid mask `(i >= EpidemicStart[m]) & (i <= N[m])`, the deaths grid
(raw deaths carry `-1` placeholder rows past each country's last observed day —
the `.stan` comment says they "should be ignored"; only UNOBSERVED cells take
the finite placeholder 0, invalid observed counts are never silently clamped),
and the count log-factorial. `X` and `pop` bind as raw REAL arrays the way
Stan's data block declares them, so fractional values are legitimate inputs.
The Normal/Gamma endpoints are reused from `ReactiveKernelsDistributionKernels`;
note that its `exponential` is **scale**-parameterized while Stan's
`exponential` is **rate**-parameterized, so Stan's `exponential(0.03)` prior
enters as `exponential(1/0.03)` and the `y ~ exponential(1/τ)` family as
`exponential(τ)`.

The panel below shows the model source, generated kernel, and compute DAG:

```@eval
Main.ReactiveKernelsDocs.execute_ppl_example(
    @__MODULE__, :Covid19ImperialExample, :COVID19IMPERIAL_SOURCE;
    setup = Main.ReactiveKernelsDocs.setup_covid19imperial!,
)
```

## Reactant and AD support

The natural batched recurrence lowers through Reactant directly — primal and
reverse gradient — on the tested configuration (Julia 1.10.11, BridgeStan 2.9.0
/ Stan 2.39, Reactant 0.2.285, Enzyme 0.13.201, DifferentiationInterface
0.7.21, SpecialFunctions 2.9.0, LogExpFunctions 0.3.29). The committed
`benchmark/pandemic_gate.jl` checks the graph against the actual `.stan` via
BridgeStan (`propto=false, jacobian=true`) on reference-valid probes: native
value and native reverse-gradient axes run for every selected posterior of the
family, while the compiled Reactant primal/gradient axes are compiled once per
DISTINCT dataset (ecdc0401 and ecdc0501) rather than separately for the
byte-identical v2/v3 model names. Measured on that configuration: native value
`4.0e-16`–`5.9e-16`, native reverse gradient `1.6e-15`–`2.9e-15`,
Reactant-compiled primal `1.3e-16`–`1.8e-16`, Reactant-compiled gradient
`2.2e-15`–`2.4e-15` relative error, plus small-shape boundary controls against
actual Stan (early observation windows including day 1, and the empty renewal
tail `N2 = N0`). These are correctness receipts on the named probe scope (five
reference-valid probes per posterior for the native axes, one probe per
compiled dataset for the Reactant axes), not timing claims.

Three authoring details matter under AD and Reactant. The buffer shift is
`vcat(permutedims(pred), buffer[1:(N2-1), :])`; `vcat(transpose(...), Matrix)`
shapes fail ordinary Enzyme, the equally natural `reduce(vcat, permutedims.(eds))`
materialization of the scan outputs fails under Reactant (traced-size
typeassert in `_typed_vcat`), and `stack(eds)` is transposed (vectors become
columns), so the E_deaths grid materializes as `permutedims(reduce(hcat, eds))`.
A scan closure resolves free names in its enclosing module, not the `@kernel`
body, so `N2` rides along as an explicit `Ref` operand. And under ordinary
(unannotated) Enzyme the scan's `xs` matrix must stay purely param-derived —
concatenating a constant phase column into it stores constant memory into
differentiable state (a runtime-activity dependency) — which the exact
closed-form imputation-day block makes unnecessary; the active per-country
`ifr_noise` vector rides inside the scan carry for the same reason.
