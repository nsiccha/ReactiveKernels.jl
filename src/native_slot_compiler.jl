# Finite captured-MethodIR prototype, kept internal while its admission and
# preparation interfaces are integrated. No dependency on sampler examples.
module NativeSlotCompiler
import ..ReactiveKernels
using LinearAlgebra
import Random
const RK = ReactiveKernels

include("native_slots.jl")
include("native_slots_factory.jl")
include("transpiled_program.jl")
end
