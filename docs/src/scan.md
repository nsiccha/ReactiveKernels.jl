# Sequential recurrences with `scan`

`scan` is an opt-in `@kernel` authoring primitive for **bounded sequential
recurrences** — computations that walk a sequence in order, threading a *carry*
from one step to the next. It is the sequential counterpart to
[`plate`](compiler.md): where `plate` is a pure per-element
broadcast map (every lane independent), `scan` threads state, so step `t` sees
what step `t−1` produced.

Its purpose is Reactant lowering. A recurrence authored the obvious way — a
`for` loop that reads `x[t-1]` and writes `out[t]` element by element — evaluates
fine natively but does **not** lower through Reactant, because XLA forbids scalar
indexing of a traced array. `scan` lowers the same recurrence to a single
`stablehlo.while` carry loop: one traced loop, no unrolling, no per-step scalar
indexing. So the natural sequential form compiles under Reactant with no manual
reformulation into a vectorized closed form.

## Syntax

```julia
result = scan(xs, Ref(shared₁), Ref(shared₂), …; init = c₀) do carry, x, s₁, s₂, …
    # … compute with carry, the per-step element x, and the shared operands …
    (new_carry, output)          # the do-block must END with this 2-tuple
end
```

- **`xs`** — the sequence to scan over. It is the sole non-`Ref` positional and
  is consumed one element per step (never broadcast-invariant).
- **`init`** — seeds the threaded `carry`. Required keyword.
- **`Ref(shared)` operands** — broadcast-invariant scalars passed unchanged to
  every step, exactly like [`plate`](compiler.md)'s atomic `Ref`
  arguments.
- **The do-block** receives `(carry, x, shared...)` and must end with the 2-tuple
  `(new_carry, output)`. `scan` returns the vector `[output₁, output₂, …]` (one
  entry per element of `xs`); the final carry is internal.

The carry may be a scalar or a **`NamedTuple`** when several values must be
threaded together:

```julia
init = (; y_prev = μ, err_prev = 0.0)     # a compound carry
# … read carry.y_prev, carry.err_prev inside the step …
```

`scan` is macro sugar recognised inside a `@kernel` / `@ppl` body; it is not a
callable runtime function outside one (calling it directly throws with a pointer
to this page).

## Example: an ARMA(1,1) error recursion

The [ARMA(1,1) example](arma11.md) authors its latent one-step-ahead errors with
`scan`. The recurrence is

```math
\nu_1 = \mu + \phi\mu,\quad \nu_t = \mu + \phi\,y_{t-1} + \theta\,\varepsilon_{t-1},\quad \varepsilon_t = y_t - \nu_t,
```

which threads the previous observation and the previous error. Carry both in a
`NamedTuple`, seeded `(y₀ ≡ μ, ε₀ ≡ 0)` so the single unified step reproduces
`ν₁ = μ + φ·μ` at `t = 1`:

```julia
errors = scan(series, Ref(μ), Ref(φ), Ref(θ);
              init = (; y_prev = μ, err_prev = 0.0)) do carry, y, m, f, t
    ν = m + f * carry.y_prev + t * carry.err_prev
    e = y - ν
    ((; y_prev = y, err_prev = e), e)
end
```

`errors` is an ordinary named port: the likelihood reduces it, the forecast
reads its last entry, and a query can ask for just the errors. Under Reactant the
whole density lowers off this natural recursion; the example keeps a vectorized
closed form (`errors_closed`) purely as an independent numerical cross-check.

## A cumulative recurrence

A running maximum is a scalar-carry scan — no `NamedTuple` needed:

```julia
running_max = scan(series; init = -Inf) do carry, x
    m = max(carry, x)
    (m, m)          # carry the running max forward AND emit it each step
end
```

## Lowering and semantics

- **Native.** The generated ordered loop contains the scalar step directly,
  seeding the carry from `init`. Requesting the scan port collects its per-step
  outputs in a vector. When that port feeds only one authored `plate` with a
  selected `sum`, native preparation can run the plate cell inside the same
  carry loop and omit the intermediate scan vector. The plate's other inputs
  must be declared numeric scalars or explicit `Ref` operands. Its shared
  computations run once outside the loop. Requesting the pointwise plate port
  still returns its vector; requesting the scan port, adding another consumer,
  supplying another broadcast array, or composing multiple plates preserves
  ordinary scan materialization and broadcast shape checks.
- **Reactant.** When `xs` is a traced array, `scan` emits a `stablehlo.while`
  carry loop: the carry (scalar or `NamedTuple`) is threaded as a loop-carried
  value and the per-step outputs are written into a preallocated traced buffer
  with a dynamic-update-slice. The first step runs eagerly to fix the output
  element type; the `while` runs the rest. Native and Reactant results match to
  floating-point tolerance (see `test/test_ppl_examples_reactant.jl`).

## Limitations

- **`xs` must be non-empty** (an empty sequence throws — the output element type
  is otherwise undetermined).
- **Per-step output is a scalar** under the Reactant lowering. A non-scalar
  per-step output is a loud, reported error, never a silent mis-lowering; author
  the output as a scalar (or open an issue for the shape you need).
- **RK-macro-only.** `scan` is recognised by the `@kernel` macro; it does not
  change Reactant itself.
