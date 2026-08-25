# Non-allocating kernels

`prepare_nonallocating` changes physical lowering, not graph semantics. It is
provided by an optional package extension because MutatingFunctions is not yet
registered. The base ReactiveKernels package therefore remains independently
resolvable on Julia 1.10.

Install the reviewed MutatingFunctions revision and load both packages to
activate the extension:

```julia
using Pkg
Pkg.add(url = "https://github.com/nsiccha/MutatingFunctions.jl",
        rev = "b353559ef3e391ae2e2d98256b6967903fdfa410")

using ReactiveKernels, MutatingFunctions
```

The extension selects exactly the same recipes as [`prepare`](api.md), then rewrites each
selected operation call through `MutatingFunctions.apply!!` and a persistent
typed cache owned by the prepared kernel.

## Prepare, warm, reuse

```julia
using ReactiveKernels, MutatingFunctions

g = Graph()
x = value!(g, :x, Vector{Float64})
copied = value!(g, :copied, Vector{Float64})
reversed = value!(g, :reversed, Vector{Float64})

add!(g, x => copied, copy)
add!(g, copied => reversed, reverse)

p = plan(g; have = (x,), want = (reversed,))
k = prepare_nonallocating(p)

k([1.0, 2.0, 3.0]) # first call allocates and seeds both caches
y = k([4.0, 5.0])   # later calls reuse them
@assert y == [5.0, 4.0]
```

For these registered allocating operations, `y` is fresh storage retained by
the kernel and a later call overwrites the same object. That ownership is not a
universal guarantee: each slot keeps whatever its operation first returns, so
an aliasing operation such as `identity` may retain caller-owned input. A plan
with no selected recipes returns its `have` value directly. Treat every mutable
result as borrowed: it may alias an input or be overwritten on the next call,
and it should be copied when it must outlive either.

## What the AST pass changes

Ordinary lowering emits calls through the positional operation tuple:

```julia
copied = __ops__[1](x)
reversed = __ops__[2](copied)
```

The final non-allocating pass keeps those operations positional and adds typed
cache slots plus an internal cache-filling helper:

```julia
copied = __cache_apply__(__caches__[1], __ops__[1], x)
reversed = __cache_apply__(__caches__[2], __ops__[2], copied)
```

`code_expr(k)` exposes the transformed expression. Planning, recipe order,
input order, and output order are unchanged. When custom `passes` are supplied,
they see the ordinary lowered expression first; the cache rewrite always runs
last.

## Allocation contract

The first call passes `nothing` to `apply!!`, so it intentionally uses the
allocating operation and records its result. On subsequent calls the cached
result is passed back to `apply!!`.

Steady-state allocation freedom therefore requires every selected operation to
have an allocation-free `apply!!` method for the actual cache and argument
types. MutatingFunctions' generic fallback preserves the result but may still
allocate an intermediate. Measure through a function barrier after warm-up:

```julia
function allocations(k, x)
    k(x)
    @allocated k(x)
end

@assert allocations(k, [1.0, 2.0, 3.0]) == 0
```

The initial integration is intentionally narrow:

- every selected recipe must have exactly one output, matching `apply!!`'s
  one-cache / one-result contract;
- slots retain the first returned object; registered allocating operations
  normally produce fresh kernel-retained storage, aliasing operations may retain
  caller inputs, and no-recipe plans return their inputs directly;
- each prepared kernel is stateful, non-reentrant, and not safe for concurrent
  invocation; prepare one kernel per independent task or caller;
- resizing and shape support follow the selected `apply!!` methods.

These are physical execution constraints only. The logical graph remains pure,
and the same `Plan` can always be passed to ordinary `prepare` instead.

## Reproducible integration gate

The default package tests remain independent of the unregistered weak
dependency. The optional extension is tested separately against the exact
reviewed MutatingFunctions revision used above:

```sh
julia --startup-file=no test/run_nonallocating_integration.jl
```

This creates a temporary consumer environment, installs MutatingFunctions from
the public URL at the pinned commit, develops the current ReactiveKernels tree,
asserts the resolved revision, and runs the allocation/API tests. The GitHub
Actions test workflow runs the same command in its own Julia 1.10 job.
