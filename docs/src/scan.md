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
result = scan(xs₁, xs₂, …, Ref(shared₁), Ref(shared₂), …; init = c₀) do carry, x₁, x₂, …, s₁, s₂, …
    # … compute with carry, the per-step elements x₁, x₂, …, and the shared operands …
    (new_carry, output)          # the do-block must END with this 2-tuple
end
```

- **`xs₁, xs₂, …`** — one or more **iterated sequences**, the leading non-`Ref`
  positionals. They are advanced together in **lockstep**: step `t` receives
  `xs₁[t], xs₂[t], …`. With a single sequence this is the ordinary
  `scan(xs; …) do carry, x`. Every iterated sequence must share axes; a length
  mismatch throws `DimensionMismatch`.
- **`init`** — seeds the threaded `carry`. Required keyword.
- **`Ref(shared)` operands** — broadcast-invariant scalars passed unchanged to
  every step, exactly like [`plate`](compiler.md)'s atomic `Ref`
  arguments. Every `Ref(...)` operand must follow the iterated sequences; a bare
  (non-`Ref`) positional after a `Ref(...)` is rejected.
- **The do-block** receives `(carry, x₁, x₂, …, shared...)` and must end with the
  2-tuple `(new_carry, output)`. `scan` returns the vector `[output₁, output₂, …]`
  (one entry per step); the final carry is internal.

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

## Several co-varying sequences (lockstep)

When a step reads more than one per-step sequence — one bound-data, one
parameter-derived, or two parameter-derived — list them all as leading
positionals; they advance together. A first-order linear recurrence
`carry_t = a_t · carry_{t-1} + b_t` over two per-step sequences `a` and `b`:

```julia
seq = scan(a, b; init = 0.0) do carry, aₜ, bₜ
    next = aₜ * carry + bₜ
    (next, next)
end
```

The posteriordb `prophet` logistic-trend recurrence
`m_t = m_{t-1} + (t_change_t − m_{t-1}) · r_t`, whose `t_change` is bound data
and `r` is parameter-derived, is the same shape with two sequences:

```julia
m = scan(tchange, r; init = m0) do carry, tc, rr
    next = carry + (tc - carry) * rr
    (next, next)
end
```

Shared broadcast-invariant scalars still ride along as trailing `Ref`s:
`scan(a, b, Ref(gain); init = 0.0) do carry, aₜ, bₜ, g … end`.

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
- **Reactant.** When the iterated sequences are traced arrays, `scan` emits a
  single `stablehlo.while` carry loop over **all** of them: the carry (scalar or
  `NamedTuple`) is threaded as a loop-carried value and the per-step outputs are
  written into a preallocated traced buffer with a dynamic-update-slice. Each
  sequence is gathered inside the loop by the loop counter, so **every iterated
  sequence must be traced** — pass a data sequence as `Reactant.to_rarray(data)`
  rather than as bound host data, since a host array has no exposed counter to
  index in the `while` (a host iterated sequence is a loud, reported error). The
  first step runs eagerly to fix the output element type; the `while` runs the
  rest. Native and Reactant results match to floating-point tolerance (see
  `test/test_ppl_examples_reactant.jl`).

## Limitations

- **The iterated sequences must be non-empty and share axes** (an empty sequence
  throws — the output element type is otherwise undetermined — and a length
  mismatch between sequences throws `DimensionMismatch`).
- **Iterated sequences precede shared operands.** All bare (iterated) positionals
  come first; every `Ref(...)` shared operand follows. A non-`Ref` positional
  after a `Ref(...)` is rejected.
- **Per-step output is a scalar** under the Reactant lowering. A non-scalar
  per-step output is a loud, reported error, never a silent mis-lowering; author
  the output as a scalar (or open an issue for the shape you need).
- **RK-macro-only.** `scan` is recognised by the `@kernel` macro; it does not
  change Reactant itself.
