# Runnable NUTS workflow. DifferentiationInterface and Enzyme are OPTIONAL extras —
# ReactiveKernels does not hard-depend on any AD backend — so run this under an
# environment that provides them (e.g. `julia --project=docs examples/nuts.jl`, or a
# project with DifferentiationInterface + Enzyme added), NOT the bare package env.
using LinearAlgebra
using Random
using ReactiveKernels
include(joinpath(@__DIR__, "nuts_runtime.jl"))
using .ReactiveKernelsNUTSExample
using DifferentiationInterface
import Enzyme

# A complete graph-backed NUTS workflow on the flat compiled-reactive sampler.
# `reactive_nuts_group` compiles the init/fwd/bwd Hamiltonian plus the reactive
# active-endpoint selection, energy error `dham`, and `diverged` flag into ONE
# `ReactiveProgram`; `nuts_state` returns a `CompiledNUTSState` whose transition
# runs on that program. The public model boundary is a SCALAR potential; its
# gradient is DifferentiationInterface + reverse-mode Enzyme, prepared once and
# written into the sampler's owned gradient buffer in place — no handwritten
# gradient callback on the sampled path.
const ENZYME_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

potential(position) = sum(abs2, position) / 2

dimension = 4
gradient_preparation = prepare_gradient(potential, ENZYME_BACKEND, zeros(dimension))
potential_gradient!(gradient, position) = first(value_and_gradient!(
    potential, gradient, gradient_preparation, ENZYME_BACKEND, position))

group = reactive_nuts_group(
    potential_gradient!,
    Matrix{Float64}(I, dimension, dimension),
    zeros(dimension),
    zeros(dimension),
)
sampler = nuts_state(
    group;
    rng = Xoshiro(20260825),
    step_f = partial(leapfrog!; stepsize = 0.35),
    max_depth = 7,
)
warmup = warmup!(sampler, 300)
chain = sample!(sampler, 1_000)
means = vec(sum(chain.samples; dims = 2)) ./ size(chain.samples, 2)
variances = vec(sum(abs2, chain.samples .- means; dims = 2)) ./
    (size(chain.samples, 2) - 1)

println("compiled sampler type: ", typeof(sampler).name.name)
# Introspect the ACTUAL compiled ReactiveProgram behind the group: its plan is the
# Compute DAG, and code_expr(program, handle) is the generated getter for any
# reachable derived node (here the reactive energy error `dham`).
program = reactive_program(group)
println("selected plan:")
println(explain(program.plan))
println("generated reactive energy-error (dham) getter:")
println(code_expr(program, getproperty(group.handles, :dham)))
println("sample mean: ", means)
println("sample variance: ", variances)
println("divergences: ", count(stat -> stat.diverged, chain.diagnostics))
println("adapted step size: ", warmup.final_stepsize)
println("adapted metric: ", warmup.metric)
