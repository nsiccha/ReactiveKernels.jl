# Non-allocating kernels

```@eval
Main.ReactiveKernelsDocs.render_review_status(:review_pending_nonallocating)
```

`prepare_nonallocating` changes how a kernel runs, not what it computes. It ships
as an optional extension, so the base ReactiveKernels package keeps no hard
dependency on MutatingFunctions (which is not registered yet) and still installs
on its own on Julia 1.10.

Install the reviewed MutatingFunctions revision and load both packages to
activate the extension:

```julia
using Pkg
Pkg.add(url = "https://github.com/nsiccha/MutatingFunctions.jl",
        rev = "4fc41b1c7b774133ceaacc4ff3c34c67b15b87b2")

using ReactiveKernels, MutatingFunctions
```

It picks exactly the same recipes as [`prepare`](api.md); the only difference is
that an operation with an in-place form writes its result into a reusable buffer
the kernel keeps (via `MutatingFunctions.apply!!`), instead of allocating a fresh
result on every call.

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
Only operations with an in-place form keep such a buffer. Any other operation is
called directly, so its result may be one of the caller's arrays (a field read
such as `sched.xs` returns the caller's vector), and a plan with no recipes
returns its `have` value directly. So treat every mutable result as borrowed — it
may share memory with an input or be overwritten on the next call — and copy it if
it has to outlive that next call. The kernel never writes into a caller's array.

## What changes in the generated code

The ordinary generated function calls each operation directly:

```julia
copied = __ops__[1](x)
reversed = __ops__[2](copied)
```

The non-allocating version routes each call that has an in-place form through a
small helper that fills a reusable cache:

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
  postfix `'`/`.'`, ranges, scalar arithmetic) run inline — they never owned
  a buffer worth caching;
- broadcast materializations (dotted calls, and array `getindex` with range
  indices), `vcat`, `zeros`/`ones`, and matrix products each become their own
  destination-passing step with a typed persistent cache, guarded by exact
  shape and eltype checks so a batch-size change reseeds instead of corrupting
  a stale buffer;
- row/column reductions (`sum`, `prod`, `minimum`, `maximum`) with a
  *preparation-constant* `dims` keyword become destination-passing reduction
  steps (`maximum(m; dims = 2)` reuses its result buffer); a `dims` value
  read from a port or computed at runtime is outside this grammar;
- every other call with a registered `apply!!` method for its concrete cache
  and argument types becomes its own cache step; a call without one runs
  inline, as written.

Free symbols in the captured expression resolve against the authoring module's
own `const` bindings — the exact functions the fused closure would call, never
name-based guesses. Module-qualified callees (`Pkg.f`) resolve through the same
rule: every path segment must itself be a `const` module binding. Any source
shape outside this grammar (control flow, a non-`const` global, a call through
a port, keyword calls other than constant-`dims` reductions) runs as one
operation, which preserves the original closure semantics unchanged — a
`cumsum(m; dims = 2)` recipe therefore still computes correctly while re-running
its allocating twin.

One observable difference: a plate reduction such as `sum` over an authored
plate is fused into an accumulator loop by ordinary `prepare`, while the
non-allocating kernel materializes the pointwise plate into its cache and then
sums it. The materialized total is bit-exactly `sum` of the pointwise values;
against the fused accumulation, only floating-point summation association
differs.

## Which operations keep a buffer

A cache is a write destination, so the kernel keeps one only where it owns the
stored value and an in-place method fills it:

- the destination steps the decomposition emits (broadcast, gather, `vcat`,
  `zeros`/`ones`, matrix product, constant-`dims` reductions), whose first result
  is fresh storage;
- an operation with a registered `apply!!` method for its concrete cache and
  argument types (`copy`, `reverse`, `mul!`-backed `*`, …);
- authored plates and scans, which fill buffers the kernel allocates.

Every other operation is called directly and its result is never written into.
MutatingFunctions' generic fallback would copy each new result into the value the
cache first stored. When that value is one of the caller's arrays (a field read,
`eachrow` slices of a caller's matrix), the next call would overwrite the
previous call's input, and an immutable value such as a range cannot be written
at all. The fallback allocates its result before copying it, so calling the
operation directly costs no more.

## Typing from exemplar arguments

Caches and step selection are fixed when the kernel is prepared, from static
types. A HAVE port without a concrete declared type, such as an untyped schedule
or parameter record, leaves every value computed from it untyped: those steps
dispatch at run time, and a plate whose cell reads it gets a `Vector{Any}`
buffer. The ordinary kernel does not have this problem, because Julia
specializes it on the arguments of each call.

Pass example arguments after the spec to type the program from their types:

```julia
@kernel field_scale(sched, factor::Float64) = begin
    xs = sched.xs
    ys::Vector{Float64} = xs .* factor
end

args = ((; xs = [1.0, 2.0, 3.0], tag = 1), 2.0)
k = prepare_nonallocating(field_scale, args...; want = :ys)
k(args...)                              # seeds the caches
k((; xs = [4.0, 5.0], tag = 3), 0.5)   # new values, same types: no allocation
```

Pass one value per positional HAVE port, in [`inputs`](api.md) order; their
values are not used. The kernel is typed for exactly those argument types, so a
call with arguments of other types is an `ArgumentError`. Prepare one kernel per
argument-type combination you call it with.

A declared array output of an authored plate (`y::Vector{Float64} =
plate(...)`) fixes its buffer's element type with or without exemplars, as the
ordinary kernel's typed local converts the plate's result to the declared type.

## Allocation contract

The first call has no cache yet, so it runs the ordinary allocating operation and
stores the result. Later calls hand that stored result back for the operation to
overwrite in place.

So for a warmed-up call to allocate nothing, every operation needs an
allocation-free `apply!!` method for the actual cache and argument types, or a
result that needs no allocation. A directly called operation allocates its result
as the ordinary kernel does. Measure through a function barrier after warm-up:

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
- a directly called operation's result may be a caller's array, and a plan
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
