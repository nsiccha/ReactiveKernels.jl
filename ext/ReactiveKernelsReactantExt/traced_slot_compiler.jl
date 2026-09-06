# The optional backend consumes the native compiler's emitted program. It
# adds no sampler/model dispatch and does not load any benchmark source.
module TracedSlotCompiler
import ReactiveKernels
import Reactant
using LinearAlgebra
const RK = ReactiveKernels

include("traced_slots.jl")
end
