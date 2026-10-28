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

@kernel borrowed_curve(position, data; amount=1.0) = begin
    result = (; curve=position .* data .* amount, total=position * sum(data) * amount)
end
const BORROWED = vectorize(borrowed_curve; batched=:position, reuse=true)
const USED_BORROWED = copy(BORROWED)
const BORROWED_WARMUP = USED_BORROWED([1.0, 2.0], [3.0, 4.0]; amount=2.0)

@kernel pair_locations(x) = begin
    locations = Float64.(x)
    left = locations
    right = reshape(locations, 1, :)
    return left, right
end

@kernel pair_grid(x, scale) = begin
    left, right = pair_locations(x)
    result = plate(left, right) do a, b
        distance = abs(a - b)
        scale * exp(-distance)
    end
    return result
end
const PAIR_GRID = prepare(pair_grid)

module Nested
using ReactiveKernels
import ..PrecompiledPreparedConsumer: trajectory, HAVE
const BATCH = prepare_batched(trajectory; have = HAVE, batched = :position,
                             want = :total)
end

end
