# Sequential recurrences

Use `scan` to express a recurrence whose next step depends on a carried state. The graph retains a loop whose length comes from the input data.

## Carry and output

A scan step returns a pair: the next carry and the output to collect. These may differ. The following recurrence returns an exponentially decaying response to a sequence of inputs.

```@example recurrence
using ReactiveKernels

@kernel recurrence(x, decay, initial) = begin
    trajectory = scan(x; init=initial) do previous, observation
        current = decay * previous + observation
        (current, current)
    end
    final = last(trajectory)
    return trajectory
end

run = prepare(recurrence)
x = [1.0, 0.0, 0.0, 2.0]
trajectory = run(x, 0.5, 0.0)
trajectory
```

The step reads `decay` as a shared enclosing value. Bare sequence arguments are iterated together and must have matching axes. Several values can be carried in a named tuple when the recurrence needs more than one state component.

## Ordinary-loop check

The displayed agreement compares the generated recurrence with an ordinary Julia loop. The check uses the same inputs but a separate implementation of the update.

```@example recurrence
reference = let values=Float64[], previous=0.0
    for observation in x
        previous = 0.5 * previous + observation
        push!(values, previous)
    end
    values
end
@assert trajectory == reference
(; trajectory, reference, agrees=trajectory == reference)
```

A scan is an authoring construct inside a kernel, not a standalone runtime iterator. Its output is a graph value and can feed reductions, further recipes or a selected output. A query for the final result uses the same authored recurrence.

```@example recurrence
final_only = prepare(recurrence; want=:final)
(; final=final_only(x, 0.5, 0.0), collected_final=last(trajectory))
```

An empty sequence produces an empty collected output; consumers such as `last` still require nonempty data. `include_init=true` includes the initial carry in the returned trajectory. Use `@scan` for indexed recurrences with explicit seeds and reads of earlier entries; see its [reference](api.md).

The [discrete-latent example](discrete-latents.md) uses a distribution kernel whose forward computation is sequential and whose latent-state probabilities depend on the complete observation sequence.
