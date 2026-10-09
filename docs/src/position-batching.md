# Position batching

Position batching evaluates one scalar kernel at many parameter positions while
data shared by those positions stays atomic. This is the `vmap`-shaped complement
to a likelihood [`plate`](batched.md): a plate maps observations with one shared
parameter, while `vectorize` maps parameter positions with one shared dataset.

## One scalar source, one declared axis

Name the HAVE ports that carry positions. All other selected HAVE ports are
shared. Batched ports must agree on the trailing batch length; shared ports are
passed unchanged to every position.

```julia
@kernel normal_logpdf(x::Vector{Float64}, location::Float64,
                       logscale::Float64) = begin
    standardized = (x .- location) ./ exp(logscale)
    total::Float64 = -0.5 * sum(abs2, standardized) - length(x) * logscale -
                     0.5 * length(x) * log(2π)
    return total
end

batched = prepare_batched(normal_logpdf; batched = :x)

# `positions` has shape D × N; each column is one scalar-kernel argument.
positions = randn(6, 4)
values = batched(positions, 0.2, log(1.1))
```

`vectorize` is the concise public spelling and accepts the same arguments:

```julia
batched = vectorize(normal_logpdf; batched = :x)
```

## Shape and output contract

Batching adds one trailing axis to each named port and to each selected output:

| scalar-kernel item | batched item |
| --- | --- |
| scalar HAVE `Float64` | rank-1 array of positions |
| rank-`r` array HAVE | rank-`r + 1` array, old dimensions first |
| scalar WANT | length-`N` vector |
| rank-`r` array WANT | rank-`r + 1` array, old dimensions first |
| tuple or named-tuple HAVE/WANT | the same tree with each leaf batched |

Input and output type annotations are optional, as for a scalar kernel. A
number or an array type with a known rank checks the scalar rank plus one,
whatever its element type (`AbstractVector` checks like `Vector{Float64}`); an
untyped port, or one that leaves the rank open such as
`AbstractArray{Float64}`, derives its layout from the runtime value. Every batched array has one-based axes. For an
untyped numeric port, a vector supplies scalar positions and a matrix supplies
vector positions. Tuple and named-tuple inputs use trees of such arrays, with
one common trailing length across all leaves. Native execution also accepts a
vector of scalar records. Compiled record inputs use the tree-of-arrays form.

Outputs may be numbers, numeric arrays, or tuple/named-tuple trees of these.
Their fields, leaf element types, and shapes must agree across positions. Empty
batches need declared output types because no first position exists from which
to infer a layout; array outputs then have zero-sized original dimensions.
This empty-batch contract is native. Reactant's existing zero-sized result
export limitation also applies to a compiled empty batch; see
[issue #12](https://github.com/nsiccha/ReactiveKernels.jl/issues/12).

For example, this untyped scalar record graph is also the batched source:

```julia
@kernel response_graph(position, schedule, amount) = begin
    grid = schedule.times .^ 2
    units = position.scale .* grid
    trajectory = units .* amount
    result = (; trajectory, total=sum(trajectory))
    return result
end

positions = (; scale=[1.0, 2.0, 4.0])
schedule = (; times=[0.0, 0.5, 1.0, 2.0])
batch = vectorize(response_graph; batched=:position)
result = batch(positions, schedule, 0.5)
@assert size(result.trajectory) == (4, 3)
@assert result.total == [2.625, 5.25, 10.5]
```

Multiple batched ports may be named when independent positions advance together,
for example a position and a per-chain step size. Their trailing axes must have
equal length. `inputs`, `outputs`, `batched_ports`, and `scalar_kernel` expose
the contract without runtime Symbol lookup.

`want` selects outputs before lifting, exactly as ordinary preparation does. A
scalar `PreparedADKernel` can be lifted with `replica(ad; batched = ...)`;
native and Reactant paths then return one objective and one active-port gradient
per position.

## Fix positions once and keep another HAVE active

For a request with fixed positions and schedule but changing amounts, use two
cuts of the same graph. Binding uses ordinary scalar `prepare` before lifting:

```julia
build = vectorize(prepare(response_graph;
    have=(:position, :schedule), want=:units, bound=(; schedule));
    batched=:position)
units = build(positions)

read = vectorize(prepare(response_graph;
    have=(:units, :amount), want=:result); batched=:units)
for amount in (0.0, 0.5, -2.0, 3.0)
    @assert read(units, amount) == batch(positions, schedule, amount)
end
```

Here `units` is an authoritative supplied HAVE in the read cut, so the planner
does not select its producer or the schedule-grid recipe. The amount remains a
runtime HAVE. For a larger graph, include any additional required HAVE ports in
each cut, name all batched ports, and bind shared scalar values at preparation.
Keep the stacked positions as runtime arguments to the lifted graph; binding
them to the scalar graph would give that port the wrong layout. Refresh the
build outputs when their positions or shared inputs change. This uses the
planner's existing graph cuts and binding, with caller-owned stage values.

When shared inputs change between requests, either bind them per request or
keep them as runtime HAVE in both cuts. A repeated `prepare(...; bound=...)` of
the same graph, cut and bound ports reuses the plan and compiled code of
earlier bindings and runs only the data-only work on the new values (see
"Rebinding" on the [compiler](compiler.md) page). Runtime HAVE instead prepares
nothing per request. For this graph, the read cut above is already independent
of the schedule; reuse it with a runtime build cut:

```julia
runtime_build = vectorize(prepare(response_graph;
    have=(:position, :schedule), want=:units); batched=:position)
runtime_units = runtime_build(positions, schedule)
@assert read(runtime_units, 0.5) == batch(positions, schedule, 0.5)
```

Construct these cut objects outside the repeated request or draw loop. Compute
their units once per request; the units remain caller-owned data. Shared-only
recipes still execute once per lifted call. This reuses the prepared computation
without a process-wide cache of request values.

## Purity, allocation, and compiler parity

The lifted surface reuses the scalar graph as its mathematical authority. The
planner still performs HAVE/WANT selection and structural common-subexpression
elimination. It rejects effectful recipes: hidden mutation or state carried
between scalar calls is not a batchable contract.

Native graph lowering validates ranks and equal batch lengths, evaluates
recipes that depend only on shared ports once, then evaluates position-dependent
recipes once per position. It stacks requested outputs, so it is not the
allocation-free reducing contract of a likelihood
`plate`. The shared prefix and the per-position residual are each lowered as
`prepare` lowers a scalar kernel: an authored plate emits its fused
native loop, a scan inlines its step (and streams into a plate that consumes
it), and an embedded kernel is spliced. A plate authored inline in the batched
graph therefore costs per position what it costs in the scalar kernel, whether
it depends on a position or only on shared ports. Internal loops and lazy
branches retain the scalar semantics.

Up to canonical `d4f65756` the position driver instead called every plate
through the plate operation's per-cell fallback. On the ShinyRK superposition
plate (6529 observations, 14 doses, 32 positions per read) that allocated
3.0 MB per position and read, against 0.33 MB with the same plate in a
separately prepared child kernel, and took 6 to 9 times as long. On such a
pin, keep a position-dependent plate in its own prepared child kernel called
from the batched graph.

For compatible native numeric leaves, borrowed lowering recovers concrete final
destination types before the position loop. A recipe-free HAVE/WANT cut can
copy compatible dense numeric stacks directly, including tuple and named-tuple
trees, without creating scalar slices first. Vectors of scalar records and
custom array layouts retain projection and stacking. These internal reductions
leave the ownership, empty-batch, validation, and scalar dispatch contracts
in place. Position intermediates are described below.

For a batched dense numeric array port (`Array{T,N}` with `N > 1`), the native
driver passes a contiguous trailing-axis view when the scalar port's declared
type admits it. Undeclared ports and abstract array ports can receive views;
a concrete `Vector{Float64}` port receives a copied vector in one reusable
lane buffer. Inputs remain read-only. Pure recipe dispatch can therefore see a
`SubArray` at an eligible port. The dose-outer plate lowering accepts these
contiguous column views as gather sources.

A materialized WANT produced by an authored plate or scan similarly writes
into its column of the stacked destination when its declared type admits a
view. The first position establishes the destination's shape and element type
using one scratch buffer and one copy. Later positions write directly into
their columns; concrete array WANTs instead refill the scratch and copy it
out. A borrowed reader keeps compatible scratch buffers between calls, without
retaining destination views in its scratch cache. Pure recipes reading that
WANT can observe a dense scratch array at the first position and a `SubArray`
at later positions. Declare a concrete array type when recipe dispatch requires
that container. Declared types keep their ordinary Julia conversion semantics.

Position intermediates use the same destination protocol. A residual value
produced by an authored plate or scan, or by a recipe whose source is a
top-level dotted call (including `@.`) or an array slice such as `y[2:2:end]`
or `m[:, j]`, keeps its dense buffer in a lane slot. Later positions overwrite
that buffer when its element type and axes match, and allocate afresh
otherwise. An owning call keeps its slots for that call; a borrowed reader and
each scheduled worker keep them between calls. A WANT produced by a dotted call
or slice reuses first-position scratch and remains a dense array at every
position. Slots never leave the call, and every WANT is copied into the stacked
result before the next position runs, so values, container types and errors
are those of the scalar kernel. For example, a superposition plate whose output
is sliced, scanned, mapped with `@.` and reduced with `minimum` allocates no
per-position storage in a borrowed reader. A source recipe of any other shape,
one whose source captures a local variable, or one prepared with
`on_error = :ignore` keeps its ordinary call; such recipes, opaque functions,
concatenations and matrix products allocate as they do in the scalar kernel.
A reduction such as `minimum(rel)` reads its argument's buffer; it is not fused
into the producing loop.

By default every call owns fresh stacked output arrays, including record
leaves. Later calls cannot change retained results. To consume bounded batches
immediately on the native backend, opt into final output-buffer reuse:

```julia
borrowed = vectorize(response_graph; batched=:position, reuse=true)
first = borrowed(positions, schedule, 0.5)
published = deepcopy(first)
second = borrowed(positions, schedule, 3.0)
@assert first.trajectory === second.trajectory
@assert published == batch(positions, schedule, 0.5)
```

These outputs are **borrowed until the next call** to that prepared object,
including its views and nested leaves. An explicitly opted-in internal reducer
may consume them immediately without copying raw trajectories; finish that
reduction before calling the same object again, then publish owned compact
results. Copy or `deepcopy` arrays retained past the next call, published from a
callback, or kept in reactive state. A function return alone does not require
copying. Use a separate instance per concurrent caller; the cache is not reentrant.
Changing shapes or element types reseeds storage. Feeding a retained output back
as an input detaches its buffer so the input is not overwritten. Reuse avoids
allocation of compatible final stacked buffers and of the lane buffers and
slots above; other projections and intermediates of other recipes may still
allocate. It does not make an allocating scalar recipe allocation-free, and it
does not cache positions or unit solves.

Prepare one native borrowed template, then use `copy(template)` to create an
independent reader for each request or concurrent caller:

```julia
template = vectorize(response_graph; batched=:position, reuse=true)
reader = copy(template)
compact_totals = map((0.5, 1.0, 2.0)) do amount
    result = reader(positions, schedule, amount)
    sum(result.total) # consume borrowed arrays before the next call
end
```

Copy reuses the exact prepared scalar graph and generated callable. It creates
fresh empty buffer slots without planning, lowering, or code generation. This
works for pristine templates and objects that were already called: prior output
buffers are neither shared nor copied. Authored keyword/default signatures are
preserved. Bound data, pure recipe callables, signatures, and generated-code
metadata remain shared and read-only. Ordinary Julia specialization on a new
argument type may still occur at first invocation.

One reader can serve all sequential batches and immediate reductions within a
request. Keep its construction inside full-operation timing when measuring
request cost. Each copy follows the same shape/type reseeding and input-alias
detachment rules. Its outputs remain borrowed until its own next call, so use
`deepcopy(result)` for an owned nested result. Generic `deepcopy(template)`
recursively copies graph metadata and populated buffers; it is not the cheap
empty-instance construction contract.

## Explicit native scheduling

Ordinary native batching runs serially. When a measured workload benefits from
independent position workers, pass explicit scheduling hints:

```julia
@kernel scheduled_objective(x::Vector{Float64}, scale::Float64) = begin
    result::Float64 = scale * sum(abs2, x)
end

template = vectorize(scheduled_objective; batched=:x,
    schedule=NativeScheduling(workers=4, chunk_size=8))
reader = copy(template)
positions = reshape(collect(1.0:96.0), 3, 32)
values = reader(positions, 0.5)
@assert values == 0.5 .* vec(sum(abs2, positions; dims=1))
```

`workers` is an upper bound capped by available Julia execution threads and
runtime chunks. `chunk_size` is required and positive: it bounds the positions
stored in each worker's stacked buffers. Shared-only work runs once per call.
Workers process bounded chunks, copy them into disjoint positions of the final
result, and join before returning, including when a task fails. The final
result still contains the whole ensemble; this is not a streaming reduction.
Empty batches and calls needing only one worker use the ordinary serial driver.
All position and inner data loops keep runtime bounds.

Scheduled owning calls return fresh owned arrays. Add `reuse=true` for the
borrowed final-output lifetime described above. Both scheduled modes retain
worker buffers and are **not reentrant**: use `copy(template)` per concurrent
caller. Copies share prepared computation and bound data but start with empty
worker and final-output slots. Shapes and types reseed buffers; prior outputs
used as inputs detach. Native lane dispatch follows the ordinary batching
contract within each chunk: its first position uses scratch and subsequent
positions may use destination column views. Declare a concrete array WANT when
downstream recipe dispatch depends on that container type.

There is no automatic cost model or speedup guarantee. Measure the whole
operation, including reader construction, and account for worker storage plus
the final result. Each worker keeps its own intermediate lane slots; scheduling
does not remove the allocations of other recipes. Small or allocation-heavy work
can run slower with extra workers.

The schedule applies to native primal calls. An owning scheduled kernel compiled
with Reactant keeps the existing retained position loop; `reuse=true` keeps its
native-only boundary. Per-position derivatives use the existing scalar AD path:

```julia
using DifferentiationInterface: AutoEnzyme
using Enzyme
ad = prepare_ad(scalar_kernel(reader), AutoEnzyme(mode=Enzyme.Reverse),
    [1.0, 2.0, 3.0], 0.5; active=:x)
values, gradients = replica(ad; batched=:x)(positions, 0.5)
```

This derivative callable uses ordinary replicated scalar AD. Differentiating
the native task scheduler itself is outside this execution surface.

With Reactant, `@compile batched(positions, shared...)` lowers the same map to a
retained loop with dynamic slices and output buffers. A scalar kernel that compiles under Reactant therefore
has the same compiler requirement in vectorized form; reverse gradients have the
same requirement as the scalar prepared AD kernel. The position axis remains a
backend loop rather than one copied body per position. The loop keeps each
position's authored guards. Backend optimization must preserve selected values,
ordinary derivatives, effects and invalid-access safety, as described in the
[core constraints](constraints.md). Compile the
ordinary owning batch; `reuse=true` is a native-only borrowed-buffer surface.

The loop reads and writes each position in storage layout, with the position
axis leading, so the optimized program slices the compiled arguments directly.
With the position axis trailing, two backend rewrite patterns never finish;
see the [constraints](constraints.md) and the backend-only
`benchmark/repro_reactant_reshape_slice_rewrite.jl` reproducer. The array
layout at the call boundary is unchanged.
