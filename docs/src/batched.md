# Batched log densities, for free

```@eval
Main.ReactiveKernelsDocs.render_result_assets()
```

Write the likelihood once as ordinary Julia: a `plate` computes the
per-observation values and the kernel returns their sum. The resulting graph has
three useful views without a second density formula:

- ordinary application or `prepare(normal_loglik)` returns the scalar total;
- `extract(normal_loglik; want = :pointwise)` returns the pointwise values used
  by LOO, WAIC, and PSIS; and
- `extract(normal_loglik; want = (:pointwise, :__return__))` returns both in one
  traversal.

The scalar endpoint is the canonical transparent RK `normal` kernel object
from [Distribution kernels](distributions.md). It is not
`Distributions.jl.Normal`: `Distributions.jl` remains an independent numerical
oracle and is absent from every generated compute path.

## One authored graph, three queries

This panel executes the exact source used by the nested example package. It
checks ordinary application, return-only preparation, pointwise extraction, and
the combined query against the same oracle. The generated-kernel view is the
actual return-only lowering from this docs build.

```@eval
Main.ReactiveKernelsDocs.execute_example(
    @__MODULE__, Main.BatchedExamples.BATCHED_PRIMAL_SOURCE,
)
```

The query changes the prepared boundary, not the model. Return-only lowering
has one reduction loop and no pointwise output allocation. Pointwise-only has
one loop and materializes the requested array. Asking for both still has one
loop: each pointwise value is stored and immediately added to the return.

## Broadcasting is the batching contract

`plate` follows Julia broadcasting semantics. Equal array dimensions align and
zip; singleton dimensions expand; scalars repeat. `Ref(value)` marks an
array-valued argument as one atomic value rather than a batch axis; a cell that
reads an enclosing value without passing it gets the same atomic value (see
[Plate-cell scope](#Plate-cell-scope)). Incompatible shapes raise
`DimensionMismatch`.

Whole parameter arrays can remain atomic when observation data is bound. A
kernel prepared with `bound = (; x, y)` may keep `q` as its only input and read
`q` in each plate cell as a closure, or pass it as `Ref(q)`. The same prepared
kernel accepts Reactant-traced `q`:
every cell receives the complete parameter array, while the observation
operands determine the broadcast axes and singleton expansion.

Arguments may be transparent derived expressions, not only named ports. For
example, `plate(eachcol(logits), y) do column, observed ... end` materializes
the lazy column iterator as an ordinary outer graph recipe, then runs the
authored scalar graph once per `(column, observed)` pair. The columns remain
vector-valued scalar elements; no matrix-sized pointwise buffer is introduced.

There is no separate public axis or scheduling language. The one-axis example
above is the simplest case of that contract; multidimensional inputs use the
same broadcast rules.

### Plate-cell scope

A `plate(... do` cell reads enclosing names the way a Julia closure does. The
plate's explicit non-`Ref` arguments are the only zipped axes. Any other
enclosing name the cell reads is captured whole as an atomic operand, exactly
as if it had been passed as `Ref(name)`. This covers `@kernel` signature ports,
names assigned earlier in the kernel body and, inside a nested plate, the outer
cell's arguments and locals. A captured scalar is shared by every cell. A
captured array, tuple or struct is the whole value, never its per-cell element.
Closures are the preferred way to read a whole value in every cell. An explicit
`Ref` operand stays supported and prepares the same kernel, including under
`bound=` partial evaluation and native reverse differentiation:

```julia
@kernel shifted_sum(x, d) = begin
    shifted = x .+ 1.0
    cells = plate(eachindex(d), d) do s, dd
        shifted[s] + dd          # same as plate(eachindex(d), Ref(shifted), d)
    end
    return cells
end
```

Pass an array as an explicit plate argument when its axis should zip, as in
`plate(y, mu) do observed, mean ... end`. Names assigned inside the cell are
cell-local. An authored `scan` step captures enclosing names the same way; see
[Sequential recurrences with `scan`](scan.md).

Subkernel and endpoint calls accept ordinary `f(name = value)` and
`f(; name = value)` spellings. They normalize to the same graph.

## A plate is a pure RK subgraph

The `do` block is not an opaque batch callback. ReactiveKernels lowers it to an
ordinary scalar graph, and transparent nested kernels or distribution objects
are spliced into that graph before planning. Each selected recipe therefore has
an exact transitive set of plated HAVE dependencies.

Those dependencies determine execution frequency. After Julia instantiates the
broadcast axes, native lowering places each recipe at its narrowest valid loop
boundary. For inputs shaped like `x`, `reshape(location, 1, :)`, and
`reshape(scale, 1, 1, :)`, work depending only on `scale` runs once per scale
coordinate and its scalar result is reused across the two inner dimensions.
There is still one Cartesian traversal, and no axis-sized intermediate is
created for the reused value. Each cell recipe's fused closure is inlined into
that traversal, including a cell whose body carries its own loop (a reduction
over a short inner axis such as a few doses per observation); Julia's inlining
heuristic would otherwise keep such a cell a per-coordinate function call and
recompute its loop invariants on every observation.

A reduction over a short inner axis can say what it means. The concentration
from a few doses is the sum over the doses already given,
`sum(w[j] * u[t - s[j]] for j in eachindex(s) if t > s[j]; init = 0.0)`, or,
with the unit response extended by zero before its dose,
`sum(w[j] * get(u, t - s[j], 0.0) for j in eachindex(s); init = 0.0)`. Natively
both are the authored fold. Under a tracing backend the sum over `eachindex(s)` is one
retained loop, so the program does not grow with the dose count, and the
filter and the in-range test of `get` stay lazy branches: an out-of-range lag
is never read. A filtered sum needs its `init`, because a traced condition
cannot choose which element starts the sum.

Natively the `get` spelling runs dose-outer. When the doses and the response
`u` are plate invariants (captured enclosing values, `Ref` operands, scalars,
or cell values computed from them only), the native loop makes one pass over the observations per dose, and
each observation adds that dose's term to its running sum, so every value is
the authored fold in the authored dose order, bitwise. Where the lag advances
by exactly one per observation, as `t - s[j]` does over `1:n`, each pass
splits at the window where the lag is in range: there `u` is read contiguously
without the range test, and outside it the term takes the default. That is the
shape of a hand-written accumulation over shifted slices of the response. The
step-by-one check runs at call time; for a lattice lag it compiles away, and
for stored per-observation lags (`row[j]`) it stops at the first irregular
step and that pass reads through `get`, still dose-outer. The doses and `u`
are evaluated once per call rather than once per observation, which the plate
purity contract makes equivalent; an error raised by the term can name a
different observation than the observation-outer order would reach first.

This lowering applies to an unfiltered `sum(term for j in doses; init = x)`
whose term calls `get(u, lag, default)` as an ordinary call argument (as in
`w[j] * get(...)`; not inside `?:`, `&&` or a closure), when that sum is the
cell's only per-observation value and the plate's pointwise result is
materialized. It needs a concrete sum type that the seed and every term keep,
an `Int` lag and a `Vector` response, all decided when the kernel is
compiled; the window split also needs the lag's per-observation inputs to span
the plate's axes. Otherwise, and for the filtered spelling (whose filter can
skip the lag), the observation-outer loop runs as before.
`prepare_nonallocating` evaluates a plate cell by cell through its broadcast
step and is unchanged.

A domain port needs no declared type. `observations = domain(plan)`, with
`domain` returning `1:plan.nobs` for one plan type and `eachrow(plan.rows)` for
another, is one graph for both: the native loop is scheduled from the value
each call receives, exactly as for a declared `UnitRange` or `Vector` domain.
The per-type rule belongs in a dispatching helper defined with `@traceable`,
iterated over `eachindex` of the doses:

```julia
@traceable lag(t, p::LatticePlan, j) = t - p.shifts[j]
@traceable lag(row, ::ExactPlan, j) = row[j]
# in the cell
sum(w[j] * get(u, lag(t, p, j), 0.0) for j in eachindex(w); init = 0.0)
```

Natively `lag` is the method as written. Under a tracing backend the sum is
one retained loop whose index is traced, and `@traceable` gives the helper the
same lowering of `p.shifts[j]` and `row[j]` that the term gets, so the program
is the one the inline spelling emits for either plan. An ordinary helper
cannot read at a traced index (`Scalar indexing is disallowed`).

`@traceable` is shorthand for a `ReactiveKernels.traced` method, the
implementation a kernel calls in place of the helper when a tracing backend
compiles it. Write that method by hand when the two should differ: a
hand-tuned in-place superposition loop stays the native method, and its traced
method calls a prepared plate kernel with the cell above.

```julia
superpose(plan, units, weights) = ...        # in-place loop, native only
ReactiveKernels.traced(::typeof(superpose), plan, units, weights) =
    SUPERPOSITION_CELL(plan, units, weights) # a prepared kernel
```

Purity is the plate contract, just as it is for ordinary stateless RK recipes.
RK does not inspect an opaque Julia callable to prove its implementation pure;
adding it as an ordinary recipe asserts that contract. A recipe explicitly
marked `effectful=true` is excluded by planning and therefore cannot enter a
plate. Work with observable effects belongs outside `plate`.

## Measured parity with the established plate path

The checked-in Normal receipt compares the authored return-only kernel with the
established `plate(normal.logpdf; ...)` reduction in the same process, rounds,
and data. Its native and Reactant panels now live under
[Distribution kernels through Reactant](distributions-reactant.md#batched-authored-graph-parity),
keeping this page as the one executable source authority for the authored graph.

The native hard gate requires the authored path to stay within 10% of the
established plate path for every `N ≥ 1,000`; `N = 1` is reported but excluded
from that ratio gate because timer quantization dominates such a short call.
Return-only authored execution must remain zero-byte and zero-allocation at all
six sizes. Before timing, the harness also rejects a native `similar` output or
a Reactant lowering that is not a tensorized broadcast chain consumed by
`sum`.

## Reproduce the receipt

From the repository root:

```sh
julia --startup-file=no --project=benchmark/distributions \
  benchmark/distributions/setup.jl
julia --startup-file=no --project=benchmark/distributions \
  benchmark/distributions_comparison.jl \
  --output=benchmark/receipts/distribution-logdensity-v1.toml
julia --startup-file=no benchmark/receipts/validate_distributions.jl \
  benchmark/receipts/distribution-logdensity-v1.toml
```

For a reusable pointwise output buffer, the optional MutatingFunctions
integration prepares the pointwise extraction non-allocatingly. The focused
fixture is
[`packages/ReactiveKernelsBatchingExamples/test/test_batched_nonallocating.jl`](https://github.com/nsiccha/ReactiveKernels.jl/blob/main/packages/ReactiveKernelsBatchingExamples/test/test_batched_nonallocating.jl).
