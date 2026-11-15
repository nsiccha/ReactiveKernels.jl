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

Each file is one statement defining `_rk_stdlib_<name>` and is copied into a
module only when the module uses the function. The mappings live in
`ext/ReactiveKernelsSpecialFunctionsExt.jl`; the native comparisons in
`test/test_js_source.jl`.

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
