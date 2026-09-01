# Non-allocating kernels

`prepare_nonallocating` changes how a kernel runs, not what it computes. It ships
as an optional extension, so the base ReactiveKernels package keeps no hard
dependency on MutatingFunctions (which is not registered yet) and still installs
on its own on Julia 1.10.

Install the reviewed MutatingFunctions revision and load both packages to
activate the extension:

```julia
using Pkg
Pkg.add(url = "https://github.com/nsiccha/MutatingFunctions.jl",
        rev = "b353559ef3e391ae2e2d98256b6967903fdfa410")

using ReactiveKernels, MutatingFunctions
```

It picks exactly the same recipes as [`prepare`](api.md); the only difference is
that each operation writes its result into a reusable buffer the kernel keeps
(via `MutatingFunctions.apply!!`), instead of allocating a fresh result on every
call.

## Prepare, warm, reuse

```julia
using ReactiveKernels, MutatingFunctions

@kernel g(x::Vector{Float64}) = begin
    copied::Vector{Float64} = copy(x)
    reversed::Vector{Float64} = reverse(copied)
end

p = plan(g)
k = prepare_nonallocating(p)

k([1.0, 2.0, 3.0]) # first call allocates and seeds both caches
y = k([4.0, 5.0])   # later calls reuse them
@assert y == [5.0, 4.0]
```

Here `y` is storage the kernel owns, and a later call overwrites that same array.
This is not guaranteed for every operation: each cache keeps whatever its
operation first returns, so an operation that just passes an input through (like
`identity`) may hold onto a caller's array, and a plan with no recipes returns
its `have` value directly. So treat every mutable result as borrowed — it may
share memory with an input or be overwritten on the next call — and copy it if it
has to outlive that next call.

## What changes in the generated code

The ordinary generated function calls each operation directly:

```julia
copied = __ops__[1](x)
reversed = __ops__[2](copied)
```

The non-allocating version routes each call through a small helper that fills a
reusable cache:

```julia
copied = __cache_apply__(__caches__[1], __ops__[1], x)
reversed = __cache_apply__(__caches__[2], __ops__[2], copied)
```

`code_expr(k)` shows the resulting function. The recipes chosen and the order of
recipes, inputs, and outputs are all unchanged. Custom `passes` see the ordinary
function first; the cache rewrite always runs last.

## Fused authored sources decompose into destination-passing steps

An operation captured from `@kernel` source (for example
`W * transpose(X) .+ b` or `vcat(zeros(1, n), m)`) is a single fused callable,
and no in-place method can exist for an arbitrary closure. Instead of caching
such a recipe as one opaque operation, the rewrite decomposes its captured
expression into primitive steps at preparation time:

- identity-preserving wrappers (`view`, `reshape`, `transpose`, `eachcol`,
  ranges, scalar arithmetic) run inline — they never owned a buffer worth
  caching;
- broadcast materializations (dotted calls, and array `getindex` with range
  indices), `vcat`, `zeros`/`ones`, and matrix products each become their own
  destination-passing step with a typed persistent cache, guarded by exact
  shape and eltype checks so a batch-size change reseeds instead of corrupting
  a stale buffer;
- every other call becomes its own cache step, so registered `apply!!`
  coverage applies per step.

Free symbols in the captured expression resolve against the authoring module's
own `const` bindings — the exact functions the fused closure would call, never
name-based guesses. Any source shape outside this grammar (control flow, a
non-`const` global, a call through a port) keeps the whole-recipe cache step,
which preserves the original closure semantics unchanged.

One observable difference: a plate reduction such as `sum` over an authored
plate is fused into an accumulator loop by ordinary `prepare`, while the
non-allocating kernel materializes the pointwise plate into its cache and then
sums it. The materialized total is bit-exactly `sum` of the pointwise values;
against the fused accumulation, only floating-point summation association
differs.

## Allocation contract

The first call has no cache yet, so it runs the ordinary allocating operation and
stores the result. Later calls hand that stored result back for the operation to
overwrite in place.

So for a warmed-up call to allocate nothing, every operation needs an
allocation-free `apply!!` method for the actual cache and argument types.
MutatingFunctions' generic fallback keeps the right result but may still allocate
a temporary. Measure through a function barrier after warm-up:

```julia
function allocations(k, x)
    k(x)
    @allocated k(x)
end

@assert allocations(k, [1.0, 2.0, 3.0]) == 0
```

This first version is deliberately narrow:

- every selected recipe must have exactly one output, because a cache holds one
  result;
- each cache keeps whatever its operation first returns; an ordinary allocating
  operation gives fresh storage the kernel then owns, an operation that just
  passes an input through (like `identity`) may keep a caller's array, and a plan
  with no recipes returns its inputs directly;
- a prepared kernel holds this mutable state, so it is not safe to call from two
  tasks at once; prepare one kernel per independent caller;
- for a generic cache step, resizing and shape changes are supported only as far
  as the chosen `apply!!` methods support them; the decomposed destination steps
  (broadcast, `vcat`, `zeros`/`ones`, matrix product) always guard shape and
  eltype and reseed on a mismatch.

These are limits on how the kernel *runs*, not on what it computes: the graph is
unchanged, and the same `Plan` can always be given to ordinary `prepare`
instead.

## Reactive layers use owned state, not borrowed caches

`prepare_nonallocating` is meant for calling one kernel directly, over and over:
each cache is *borrowed* — it may share memory with an input and is overwritten
on the next call. The reactive layers deliberately do **not** use it.
[`ReactiveState`](online-stats.md) keeps computed, frozen, and checkpointed
values around and reuses them later, so a borrowed cache overwritten by a
subsequent call would silently corrupt one of those saved values; `get!`
therefore uses ordinary [`prepare`](api.md)-style kernels, whose results the
state owns outright.

If you want in-place updates *and* the reactive machinery, use `prepare_reactive`
→ `CompiledReactiveState`: `mutate!`/`touch!` edit the declared mutable inputs in
place, and derived values live in buffers the state owns. Ownership,
invalidation, and freeze/checkpoint all still work; for the public object/method
form over the same state machinery, see [Stateful Welford
moments](online-stats.md#stateful-welford-moments). Reach
for a `prepare_nonallocating` kernel only for direct, single-caller use, and copy
any mutable result you need to keep before the next call.

## Reproducing the extension tests

The optional extension is tested against the exact reviewed MutatingFunctions
revision used above:

```sh
julia --startup-file=no test/run_nonallocating_integration.jl
```

This creates a temporary consumer environment, installs MutatingFunctions from
the public URL at the pinned commit, develops the current ReactiveKernels tree,
and runs the allocation/API tests.
