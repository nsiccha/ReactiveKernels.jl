module PrecompiledPreparedConsumer

using ReactiveKernels

const WAS_PRECOMPILED = ccall(:jl_generating_output, Cint, ()) != 0

@kernel trajectory(position::Vector{Float64}, shared::Float64,
                   schedule::Vector{Float64}, dose_mgs::Vector{Float64}) = begin
    scale::Float64 = exp(shared)
    conc::Vector{Float64} = position[1] .* scale .* schedule .+ dose_mgs
    total::Float64 = sum(conc)
end

const HAVE = (:position, :shared, :schedule, :dose_mgs)
const WANT = (:conc, :total)
# The first preparation and warmup happen inside one function invocation.
# No return to top level can make newly defined context methods visible.
function build_first_batch()
    kernel = prepare_batched(trajectory; have = HAVE, batched = :position,
                             want = WANT)
    kernel, kernel(ones(2, 1), 0.0, [0.5, 1.0], [3.0, 4.0])
end
const FIRST = build_first_batch()
const SCALAR = prepare(trajectory; have = HAVE, want = WANT)
const BATCH = prepare_batched(trajectory; have = HAVE, batched = :position,
                             want = WANT)
const VECTORIZED = vectorize(trajectory; have = HAVE, batched = :position,
                            want = WANT)
const BOUND = prepare(trajectory; have = HAVE, want = WANT,
                      bound = (dose_mgs = [3.0, 4.0],))

# Exercise the body while producing the image, then use a different argument
# specialization after load so a precompiled call instance cannot mask a
# missing expression-cache entry.
const WARMED = prepare_batched(trajectory; have = HAVE, batched = :position,
                              want = :conc)
const WARMUP = WARMED(ones(2, 1), 0.0, [0.5, 1.0], [3.0, 4.0])

module Nested
using ReactiveKernels
import ..PrecompiledPreparedConsumer: trajectory, HAVE
const BATCH = prepare_batched(trajectory; have = HAVE, batched = :position,
                             want = :total)
end

end
