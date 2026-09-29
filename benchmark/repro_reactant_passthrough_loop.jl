# Reactant only; no ReactiveKernels code. On Reactant 0.2.284 this retained
# copy/repeat loop fails to finish MLIR greedy rewriting within 90 seconds.
# Run under an execution bound (for example timeout -k 15 90 julia ...).
using Reactant, Test

function zero_buffer(source, dims)
    T = Reactant.unwrapped_eltype(source)
    value = Reactant.promote_to(Reactant.TracedRNumber{T}, zero(T))
    copy(Reactant.Ops.fill(value, collect(Int64, dims)))
end
function write_slot(buffer, value::Reactant.TracedRNumber, index)
    Reactant.Ops.dynamic_update_slice(buffer,
        Reactant.Ops.broadcast_in_dim(value, Int64[], Int64[1]),
        [Reactant.promote_to(Reactant.TracedRNumber{Int64}, index)])
end
function write_slot(buffer, value::Reactant.TracedRArray, index)
    Reactant.Ops.dynamic_update_slice(buffer,
        Reactant.Ops.reshape(value, Int64[size(value)..., 1]),
        [Reactant.Ops.constant(Int64(1)),
         Reactant.promote_to(Reactant.TracedRNumber{Int64}, index)])
end
function passthrough_loop(position, shared)
    count = length(position.scale)
    scale = zero_buffer(position.scale, (count,))
    curve = zero_buffer(position.curve, (size(position.curve, 1), count))
    repeated = zero_buffer(shared, (length(shared), count))
    Reactant.@allowscalar begin
        scale = write_slot(scale, position.scale[1], 1)
        curve = write_slot(curve, position.curve[:, 1], 1)
    end
    repeated = write_slot(repeated, shared, 1)
    # Count derives from input shape; keep it opaque to constant unrolling.
    limit = only(Reactant.Ops.optimization_barrier(Reactant.Ops.constant(Int64(count))))
    Reactant.@trace track_numbers=false for index in 2:limit
        Reactant.@allowscalar begin
            scale = write_slot(scale, position.scale[index], index)
            curve = write_slot(curve, position.curve[:, index], index)
        end
        repeated = write_slot(repeated, shared, index)
    end
    (; scale, curve), repeated
end

position = (; scale=Reactant.to_rarray([1.0, 2.0, 3.0]),
              curve=Reactant.to_rarray(reshape(collect(1.0:6.0), 2, 3)))
shared = Reactant.to_rarray([2.0, 5.0])
println("PASSTHROUGH_LOOP_COMPILE_BEGIN julia=", VERSION,
        " Reactant=", pkgversion(Reactant))
flush(stdout)
compiled = Reactant.@compile passthrough_loop(position, shared)
println("PASSTHROUGH_LOOP_COMPILE_DONE")
actual, repeated = compiled(position, shared)
@test Array(actual.scale) == Array(position.scale)
@test Array(actual.curve) == Array(position.curve)
@test Array(repeated) == repeat(Array(shared), 1, 3)
