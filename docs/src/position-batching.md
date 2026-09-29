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

Input and output type annotations are optional, as for a scalar kernel. A known
numeric input type checks the scalar rank plus one; an untyped port derives its
layout from the runtime value. Every batched array has one-based axes. For an
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

Prepare stable cuts once when shared inputs change between requests. Keep those
inputs as runtime HAVE in both cuts, rather than calling `prepare(...; bound=...)`
for each new value. Binding can fold data-only work, but repeated preparation
also pays planning, code generation and compilation costs. For this graph, the
read cut above is already independent of the schedule; reuse it with a runtime
build cut:

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
recipes once per position. It stacks requested outputs and copies array-valued
slices, so it is not the allocation-free reducing contract of a likelihood
`plate`. Pure authored plates, scans and embedded kernels are called as whole
scalar operations, so their shared-only recipes also execute above the position
loop. Their internal loops and lazy branches retain the scalar semantics.

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
copying. Use a
separate prepared object per concurrent caller; the cache is not reentrant.
Changing shapes or element types reseeds storage. Feeding a retained output back
as an input detaches its buffer so the input is not overwritten. Reuse avoids
allocation of compatible final stacked buffers; per-position projections and
scalar intermediate arrays may still allocate. It does not make an allocating
scalar kernel allocation-free, and it does not cache positions or unit solves.

With Reactant, `@compile batched(positions, shared...)` lowers the same map to a
retained loop with dynamic slices and output buffers. A scalar kernel that compiles under Reactant therefore
has the same compiler requirement in vectorized form; reverse gradients have the
same requirement as the scalar prepared AD kernel. The position axis remains a
backend loop rather than one copied body per position. The loop keeps each
position's lazy branches, including inactive arithmetic. Compile the
ordinary owning batch; `reuse=true` is a native-only borrowed-buffer surface.
