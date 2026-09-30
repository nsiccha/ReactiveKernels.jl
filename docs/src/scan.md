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
indexing of a traced array. On its supported traced shapes — traced 1-D
sequences, or `eachrow` over a traced matrix — `scan` lowers the same recurrence
to a single `stablehlo.while` carry loop: one traced loop, no unrolling, no
per-step scalar indexing. So the natural sequential form compiles under Reactant
with no manual reformulation into a vectorized closed form.

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
  mismatch throws `DimensionMismatch`. The sequences may be empty: no step
  runs and `scan` returns an empty vector (see [Empty sequences](#Empty-sequences)).
- **`init`** — seeds the threaded `carry`. Required keyword.
- **`Ref(shared)` operands** — broadcast-invariant scalars passed unchanged to
  every step, exactly like [`plate`](compiler.md)'s atomic `Ref`
  arguments. Every `Ref(...)` operand must follow the iterated sequences; a bare
  (non-`Ref`) positional after a `Ref(...)` is rejected.
- **The do-block** receives `(carry, x₁, x₂, …, shared...)` and must end with the
  2-tuple `(new_carry, output)`. `scan` returns the vector `[output₁, output₂, …]`
  (one entry per step); the final carry is internal.

The carry may be a scalar, a **`NamedTuple`**, or a vector (an HMM forward
pass threads its belief-state vector; see the `eachrow` shape below) when
several values must be threaded together:

```julia
init = (; y_prev = μ, err_prev = 0.0)     # a compound carry
# … read carry.y_prev, carry.err_prev inside the step …
```

`scan` is macro sugar recognised inside a `@kernel` / `@ppl` body; it is not a
callable runtime function outside one (calling it directly throws with a pointer
to this page). Like an authored `plate`, a scan is an ordinary recipe of the
graph it is written in: write it as the right-hand side of a named port, next to
any other recipes, plates and scans of that graph. It needs no kernel of its own.

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

Pass the sequences a step reads as separate positionals rather than packing
them into matrix rows. A piecewise-exact turnover recurrence
`R[t] = (R[t-1] - Rss)·exp(-c₂·dt[t]) + Rss` over midpoint concentrations and
step sizes reads both directly:

```julia
updated = scan(conc_mid, dts, Ref(pd), Ref(kin);
               init = pd.baseline) do previous, concentration, dt, parameters, input_rate
    c2 = parameters.kout * (1 + concentration /
        (parameters.theta1 * concentration + parameters.theta2))
    steady = input_rate / c2
    next = (previous - steady) * exp(-c2 * dt) + steady
    (next, next)
end
trajectory = vcat([pd.baseline], updated)
```

This form gives exactly the same result as `scan(eachrow(hcat(conc_mid, dts)), …)`
reading `row[1]` and `row[2]`, but it does not build the packed matrix. On a
3264-step schedule it allocated 52,448 B against 104,720 B for the packed form,
and took 15.8 µs against 20.2 µs. Under Reactant both forms keep one
`stablehlo.while`.

A scan over matrix rows iterates `eachrow` of the matrix — one row per step —
with whatever carry the recurrence threads (here a 2-vector belief state):

```julia
forward = scan(eachrow(scan_rows), Ref(gain); init = seed) do carry, row, g
    newg = g .* (row[1] .+ carry .* row[2]) .+ row[3]
    (newg, sum(newg))
end
```

## Empty sequences

An empty schedule needs no special case. When the iterated sequences are empty,
no step runs and `scan` returns an empty vector. The vector's element type is
the step's output type for the carry seed, one element of each sequence and the
shared operands. It is inferred without running the step, the way Julia's
`accumulate` types an empty result, and it is `Any` only when inference cannot
tell. A plate that sums the scan totals zero. In the turnover example above, an
empty schedule gives `trajectory == [pd.baseline]`. A dose-feedback recurrence
over zero doses gives `weights == Float64[]` and `total == 1.0`:

```julia
using ReactiveKernels: scan

@kernel dose_weights(mgs::Vector{Float64}, units::Matrix{Float64}, n::Int) = begin
    slots = collect(1:n)
    weights = scan(mgs, eachrow(units), Ref(slots);
            init = (; prior = zeros(Float64, n), index = 1)) do carry, mg, u, positions
        weight = mg / (1 + sum(carry.prior .* u))
        next = ifelse.(positions .== carry.index, weight, carry.prior)
        ((; prior = next, index = carry.index + 1), weight)
    end
    total::Float64 = 1.0 + sum(weights)
    return total
end
```

A traced empty sequence compiles to a program with no `stablehlo.while`. A
nonempty one keeps its single loop.

A recurrence authored in its own kernel, prepared once and called from a lazy
branch arm, still works. The arm then runs through an ordinary callable
boundary, not transparent graph splicing. Declare that arm's result type
(`weights::Vector{Float64} = if … end`). A prepared child called inside a
recipe re-enters the generic call operators the enclosing kernel is already
being inferred through, so Julia widens the nested call. Without the
declaration, every consumer of the value in a kernel that also owns a plate, a
scan or an embedded prepared kernel dispatches dynamically: inside an embedded
plate loop, once per cell.

`scan(...) do … end` is recognized only as a recipe's top-level right-hand side,
as `plate` is. Written inside a branch arm or another expression, it reaches
the runtime placeholder instead. Scan recognition also requires the actual
`ReactiveKernels.scan` binding. With `import ReactiveKernels` alone, spell the
callee `ReactiveKernels.scan`, or add `using ReactiveKernels: scan` before
using bare `scan`. A bare, unbound `scan` remains an ordinary call and raises
`UndefVarError` when executed; a typed assignment does not change this.

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
- **Reactant.** `scan` emits a single `stablehlo.while` carry loop for every
  iterated-sequence shape. The lowering is selected whenever any scan operand
  is traced (the carry seed, an iterated sequence looked through its
  `eachrow` wrapper, or a shared operand looked through its `Ref`), and every
  iterated sequence is then carried as a traced array: a 1-D sequence is
  gathered element by element by the loop counter, an `eachrow` matrix
  contributes its parent and row `i` is one traced dynamic slice (so the
  step's row arithmetic stays vector-valued whatever the row width), and a
  host (`bound=`) sequence is lifted into the traced program as a constant
  exactly as bound plate data is. Sequences of different kinds may be
  iterated together, and several lockstep sequences share the one loop. The carry — scalar, `NamedTuple`, or vector — is
  threaded as a loop-carried value, the per-step outputs are written into a
  preallocated traced buffer with a dynamic-update-slice, and the first step
  runs eagerly to seed the carry and fix the output element type (`N == 1`
  runs with an empty loop body). The emitted program is independent of the
  sequence length and of the row width, as the [core
  constraints](constraints.md) require; native and Reactant results match to
  floating-point tolerance (see `test/test_authored_scan_reactant.jl` and
  `test/test_ppl_examples_reactant.jl`). A scan whose every operand is host
  data runs the native loop as ordinary host precomputation and emits no
  program structure. A traced sequence's length is static, so an empty one
  emits no loop. Its result is an empty traced vector of the step's scalar
  output type. When a traced 1-D input sequence already has that type, it is
  forwarded; otherwise the result is a zero-sized constant. That constant works
  as an intermediate, but XLA export still rejects it when it is itself the
  compiled program's output (an upstream gap, `reactivekernels-use` §7l).

## Generated grouped recurrences

The `ReactiveKernelsPPL` example package uses a separate internal rectangular
fold for traced TGI nadir assessments. Reset flags represent unequal subject
lengths, including subjects without assessments. Its primal and reverse paths
preserve the existing output-before-update semantics.

A PK adapter is experimental and disabled for ordinary callers. It carries
compartment amounts and a concentration/AUC buffer over a flat operation table;
repeated-dose segments use a bounded binary-power loop. Joint K=1/K=3 primal
parity and bounded loop structure pass with CPU fusion enabled, but PK reverse
compilation currently fails while tracing the rectangular path's StaticArrays
matrix exponential (`TypeError: non-boolean (TracedRNumber{Bool})` in
`StaticArrays._exp` via `_pk_expm_table`, since `f396c41e`; seen on Reactant
0.2.289 with StaticArrays 1.9.22). The upstream Enzyme-JAX defects behind the
original backend failure are fixed in Reactant 0.2.289+ (RK issue #13 closed,
both standalone reproducers passing exactly). It is not a supported sampler
path.
The experiment, reproducer, and measurements are documented in
`benchmark/joint_stan_tiled/rectangular_lowering.md` in the repository.

This runtime path does not change the authored `scan` contract above. The
rectangular fold is an internal lowering boundary, not a new public
authoring function. Bound table shapes still specialize an executable; changing
the bound schedule requires preparing and compiling again.

## Limitations

- **The iterated sequences must share axes.** A length mismatch between
  sequences throws `DimensionMismatch`. Empty sequences are supported (see
  [Empty sequences](#Empty-sequences)).
- **Iterated sequences precede shared operands.** All bare (iterated) positionals
  come first; every `Ref(...)` shared operand follows. A non-`Ref` positional
  after a `Ref(...)` is rejected.
- **Per-step output is a scalar** on the Reactant `while` path. A non-scalar
  per-step output there is a loud, reported error, never a silent mis-lowering;
  author the output as a scalar (or open an issue for the shape you need).
- **A directly iterated N-D array is rejected** on the Reactant path (its
  native semantics are linear element iteration); iterate `eachrow(M)` or
  `vec(M)` explicitly.
- **RK-macro-only.** `scan` is recognised by the `@kernel` macro; it does not
  change Reactant itself.
