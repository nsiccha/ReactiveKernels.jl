# Position batching

Use `vectorize` to evaluate one scalar graph at several parameter positions while sharing its data. The trailing axis identifies positions; a batched scalar becomes a vector, and a batched vector becomes a matrix with one position per column.

## Shared data

The following graph computes a response from a position-specific scale and a shared schedule. Only `position` is named as batched. The schedule therefore reaches every position unchanged.

```@example batching
using ReactiveKernels

@kernel response(position, schedule, amount) = begin
    grid = schedule.times .^ 2
    units = position.scale .* grid
    trajectory = units .* amount
    result = (; trajectory, total=sum(trajectory))
    return result
end

positions = (; scale=[1.0, 2.0, 4.0])
schedule = (; times=[0.0, 0.5, 1.0, 2.0])
batch = vectorize(response; batched=:position)
result = batch(positions, schedule, 0.5)
(; trajectories=result.trajectory, totals=result.total)
```

Tuple and named-tuple inputs keep their field structure. Each batched numeric leaf carries the same trailing length. Requested outputs are stacked in the corresponding way. This operation materializes the batch outputs; it is not a reducing likelihood plate.

## Scalar agreement

Compare each column with a call to the original scalar graph. The output displays whether every position agrees.

```@example batching
scalar = prepare(response)
agreement = [
    result.trajectory[:, i] == scalar((; scale=s), schedule, 0.5).trajectory
    for (i, s) in enumerate(positions.scale)
]
@assert all(agreement)
agreement
```

Several ports can be batched together when their positions advance in lockstep. Their trailing lengths must agree. Shared arguments have no added batch axis. A scalar type annotation fixes the expected rank; an untyped numeric port derives its layout from the supplied value.

## Prepared boundaries

Prepare a graph selection before lifting it when only an intermediate result is needed. Here the build stage computes units for the positions once, and a second stage applies a changing amount.

```@example batching
build_units = vectorize(prepare(response;
    have=(:position, :schedule), want=:units, bound=(; schedule));
    batched=:position)
units = build_units(positions)

apply_amount = vectorize(prepare(response;
    have=(:units, :amount), want=:result); batched=:units)
staged = apply_amount(units, 0.5)
@assert staged == result
(; units, agrees=staged == result)
```

The caller owns the stage values and refreshes them when the positions or schedule change. Keep the stacked positions as runtime inputs to the lifted graph; binding them to its scalar source would give the port the wrong layout.

Batching requires pure recipes. Hidden mutation or state carried from one position to another changes the semantics and is rejected. Empty batches require enough declared output information to establish a layout without a first result.

The [bound-regression example](bound-regression.md) develops the same input and output selection without a batch axis.
