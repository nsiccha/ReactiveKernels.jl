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

Multiple batched ports may be named when independent positions advance together,
for example a position and a per-chain step size. Their trailing axes must have
equal length. `inputs`, `outputs`, `batched_ports`, and `scalar_kernel` expose
the contract without runtime Symbol lookup.

`want` selects outputs before lifting, exactly as ordinary preparation does. A
scalar `PreparedADKernel` can be lifted with `replica(ad; batched = ...)`;
native and Reactant paths then return one objective and one active-port gradient
per position.

## Purity, allocation, and compiler parity

The lifted surface reuses the scalar graph as its mathematical authority. The
planner still performs HAVE/WANT selection and structural CSE. It rejects
effectful recipes, and lifting also assumes pure straight-line semantics: hidden
mutation or state carried between calls is not a batchable contract.

Native execution validates ranks and equal batch lengths, evaluates independent
positions, and stacks the requested outputs. This necessarily allocates output
containers and copies array-valued slices; it is not the allocation-free
reducing contract of a likelihood `plate`. In the current typed replica
lowering, a recipe depending only on shared ports is still evaluated once per
position because it remains inside the scalar kernel. When such invariant work
dominates, precompute it once outside the position batch and pass the result as
a shared HAVE today; graph-level invariant hoisting is a separate optimization
boundary, not part of the current contract.

With Reactant, `@compile batched(positions, shared...)` lowers the same map to a
backend batch primitive. A scalar kernel that compiles under Reactant therefore
has the same compiler requirement in vectorized form; reverse gradients have the
same requirement as the scalar prepared AD kernel.
