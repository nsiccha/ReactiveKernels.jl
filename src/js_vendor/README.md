# Vendored JavaScript special functions

`ReactiveKernels.js_source` emits self-contained JavaScript modules. JavaScript
has no error or gamma functions, and Julia's own methods for them cannot be
translated: SpecialFunctions calls C libraries for `erfc`, `erfcx` and
`gamma`, and reads the bits of a `Float64` in its `erf` and `logabsgamma`. An
emitted module therefore computes these functions with
[stdlib-js](https://github.com/stdlib-js/stdlib) implementations, which this
folder carries as pinned, prebuilt files. Everything else, such as `logbeta`,
`logerfc` or RK's own distribution laws, still translates from its Julia code.

| File | Package | Algorithm it follows |
| --- | --- | --- |
| `erf.js` | `@stdlib/math-base-special-erf` | FreeBSD `s_erf.c`, as openlibm's `erf` |
| `erfc.js` | `@stdlib/math-base-special-erfc` | FreeBSD `s_erf.c`, as openlibm's `erfc` |
| `erfcx.js` | `@stdlib/math-base-special-erfcx` | the Faddeeva package, as libopenspecfun's `erfcx` |
| `gammaln.js` | `@stdlib/math-base-special-gammaln` | FreeBSD `e_lgamma_r.c`, as SpecialFunctions' `logabsgamma` |
| `gamma.js` | `@stdlib/math-base-special-gamma` | Cephes (openlibm's `tgamma` is a different algorithm) |
| `betainc.js` | `@stdlib/math-base-special-betainc` | Boost's `ibeta` (SpecialFunctions' `beta_inc` follows TOMS 708) |
| `betaincinv.js` | `@stdlib/math-base-special-betaincinv` | Boost's `ibeta_inverse` |

SpecialFunctions' `beta_inc` and `beta_inc_inv` are Julia code, but not
concretely typed throughout (a range end that is `Union{Float64,Int64}`), so
they cannot be translated either. `gamma_inc` and `gamma_inc_inv` are
translated from SpecialFunctions' Julia code instead of vendored: stdlib-js's
`gammainc` disagrees with it above a shape of about 1000 (0.1339 against
0.1586 at a = 2000, x = a − √a).

Each file is one statement defining `_rk_stdlib_<name>` and is copied into a
module only when the module uses the function. The mappings live in
`ext/ReactiveKernelsSpecialFunctionsExt.jl`; the native comparisons in
`test/test_js_source.jl`.

## The ODE integrator

`dopri.js` bundles [`@reside-ic/dopri`](https://github.com/mrc-ide/dopri-js)
(Imperial College's MRC Centre for Global Infectious Disease Analysis, MIT),
the Dormand–Prince 5(4) method native Julia runs as OrdinaryDiffEq's `DP5()`.
It defines `_rk_dopri`, whose `Dopri` class `src/js_ode.jl`'s adapter
`_rk_ode_rk45` drives when a module calls `rk_ode_rk45_tol`. MIT asks that its
permission notice travel with every copy, so the file's header carries the
licence in full (`embedLicense` in `build.mjs`).

The adapter uses these members of 1.0.1 beyond its documented interface, so a
new version needs `test/test_js_ode.jl` to pass again:

- `_step()`, which takes one accepted step, and `_t`, the time it reached, so
  that it counts accepted steps between consecutive outputs as
  `rk_ode_rk45_tol` does;
- `_history` and `_stepper.history`, the dense-output records `run` would
  otherwise collect, and `_stepper.interpolate`, which evaluates them;
- `_control.atol` and `_control.rtol`, set after construction, because the
  constructor refuses a zero tolerance that `rk_ode_rk45_tol` accepts when the
  other one is positive.

## Regenerating

Versions are pinned exactly in `package.json`, and `package-lock.json` pins
every dependency. With Node.js and npm:

```sh
cd src/js_vendor
npm ci
node build.mjs
```

`build.mjs` bundles each function with esbuild, reading JavaScript's built-in
typed arrays and property functions instead of stdlib's polyfills (`shims/`),
and writes the header with the package version and the notices of the original
implementation. It copies each package's `LICENSE` and `NOTICE` (Apache-2.0)
to `licenses/`. `node_modules/` is not committed.
