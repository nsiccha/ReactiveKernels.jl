module PreparedTranspilerExamples
using ReactiveKernels

@kernel accumulating_value(value; rounds=4) = begin
    advance!(increment) = begin
        for _ in 1:rounds
            value = value + increment
        end
    end
end

@kernel vector_relaxation(position; rounds=2) = begin
    squared = position .* position
    advance!(target) = begin
        for _ in 1:rounds
            position .= 0.5 .* (position .+ target)
        end
    end
end

# BEGIN scalar consumer
function scalar_example(; backend=:native)
    program = prepare_transpiled(accumulating_value, 1.0;
        backend, method=:advance!, argument=2.0, outputs=(value=:value,))
    initial = initial_transpiled_state(program)
    first = program(initial, 2.0)                 # value = 9
    continued = program(first.state, 3.0)         # value = 21
    replayed = program(initial, first.argument)  # value = 9
    (; first, continued, replayed)
end
# END scalar consumer

# BEGIN vector consumer
function vector_example(; backend=:native)
    input = [2.0, 4.0]
    program = prepare_transpiled(vector_relaxation, input;
        backend, method=:advance!, argument=zeros(2),
        outputs=(position=:position, squared=:squared))
    initial = initial_transpiled_state(program)
    first = program(initial, zeros(2))           # position = [0.5, 1.0]
    first.outputs.position .= 100.0              # edit the output snapshot
    continued = program(first.state, first.argument) # position = [0.125, 0.25]
    (; input, first, continued)
end
# END vector consumer

end
