# The optional backend consumes the native compiler's emitted program. It
# adds no sampler/model dispatch and does not load any benchmark source.
module TracedSlotCompiler
import ReactiveKernels
import Reactant
using LinearAlgebra
import Random
const RK = ReactiveKernels

include("traced_code_cache.jl")
include("traced_compile_options.jl")
include("traced_slots.jl")
include("traced_random.jl")
end
