# Batched (vectorized) log densities as recipes, for free.
#
# A density decomposed as (1) an elementwise pointwise recipe over array-typed
# ports → (2) a `sum` reduction plans to a single straight-line vectorized
# kernel. Passing a `Vector` where a scalar went is the only thing that makes it
# "batched" — the author writes one scalar `normal_logpdf`, and the same graph
# yields the per-observation log density (`want = :per_obs`) or the total
# (`want = :logdensity`) by want-set pruning, with a single Enzyme reverse pass
# over the whole batch. No `Distributions.jl` call runs in the compute or
# gradient path; it is an independent oracle only.
#
# This module holds the build-executed, `Distributions.jl`-free source rendered
# in the docs. The zero-allocation value and gradient claims require the
# `MutatingFunctions` weak dependency and are proven separately in
# `test/test_batched_nonallocating.jl` (run through
# `test/run_nonallocating_integration.jl`); see also `docs/src/batched.md`.
module BatchedExamples

export BATCHED_SOURCE
export all_sources, evaluate_source, run

const BATCHED_SOURCE = raw"""
using Distributions
using DifferentiationInterface
import Enzyme
using Random

# One scalar pointwise log density, written once. Nothing here is
# batching-specific: batching is achieved below by passing a `Vector` for `x`.
normal_logpdf(x, μ, σ) = -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ) / σ)^2

# One graph, array-typed observation port. The pointwise density is a TYPED
# port so the elementwise recipe's operation is the bare `broadcast` (the
# broadcast-lift convention) — that is what lets the non-allocating lowering
# reuse a batch buffer (see docs/src/nonallocating.md). `σ = exp(logσ)` is a
# shared subexpression the planner emits once.
batched = @kernel batched_normal(pointwise::typeof(normal_logpdf),
                                 x::Vector{Float64}, μ::Float64,
                                 logσ::Float64) = begin
    σ::Float64 = exp(logσ)
    per_obs::Vector{Float64} = broadcast(pointwise, x, μ, σ)
    logdensity::Float64 = sum(per_obs)
end

N = 1000
x = randn(Random.MersenneTwister(1), N)
μ = 0.3
logσ = log(1.2)
σ = exp(logσ)

# Two WANT boundaries from the SAME graph. `want = :per_obs` prunes the `sum`
# reduction and returns the length-N vectorized pointwise log density (LOO/WAIC
# territory); `want = :logdensity` returns only the total. One graph, no
# duplicated arithmetic, no unnecessary work.
total_plan  = plan(batched; have = (:pointwise, :x, :μ, :logσ), want = :logdensity)
perobs_plan = plan(batched; have = (:pointwise, :x, :μ, :logσ), want = :per_obs)
total_kernel  = prepare(total_plan)
perobs_kernel = prepare(perobs_plan)

total  = total_kernel(normal_logpdf, x, μ, logσ)
per_obs = perobs_kernel(normal_logpdf, x, μ, logσ)

# Distributions.jl is an independent oracle only; it never runs in the kernel.
reference        = sum(logpdf.(Normal(μ, σ), x))
reference_perobs = logpdf.(Normal(μ, σ), x)

# Reverse-mode gradient of the total over the whole N-dimensional batch, in ONE
# pass. The batch mixes the active `x` with the constant `μ`, `σ` inside the
# broadcast, so Enzyme needs runtime activity; the closed-over kernel is Const.
analytic_gradient = @. -(x - μ) / σ^2
backend = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                       function_annotation = Enzyme.Const)
gradient = DifferentiationInterface.gradient(
    v -> total_kernel(normal_logpdf, v, μ, logσ), backend, x,
)

# want-set pruning is structural, not a runtime branch: the pruned plan holds
# strictly fewer recipes and never materializes the reduction.
total_recipes  = length(total_plan.recipes)
perobs_recipes = length(perobs_plan.recipes)

@assert total ≈ reference
@assert per_obs ≈ reference_perobs
@assert length(per_obs) == N
@assert gradient ≈ analytic_gradient
@assert perobs_recipes < total_recipes
@assert !occursin("Distributions", string(code_expr(total_kernel)))
@assert !occursin("Distributions", string(code_expr(perobs_kernel)))

docs_example = (;
    name = :batched_normal,
    origin = "native batched (vectorized) Normal log density (build executed)",
    inputs = (; pointwise = normal_logpdf, x, μ, logσ),
    kernel = total_kernel,
    output = total,
    reference,
    perobs_kernel,
    per_obs,
    reference_perobs,
    gradient,
    analytic_gradient,
    total_recipes,
    perobs_recipes,
)
"""

all_sources() = (BATCHED_SOURCE,)

function evaluate_source(source::AbstractString)
    sandbox = Module(gensym(:BatchedExample), true, true)
    Core.eval(sandbox, :(using ReactiveKernels))
    parsed = Meta.parseall(source; filename = "batched-example.jl")
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    for expression in expressions
        expression isa LineNumberNode && continue
        Core.eval(sandbox, expression)
    end
    Core.eval(sandbox, :docs_example)
end

function run(io::IO = stdout)
    artifacts = map(evaluate_source, all_sources())
    for artifact in artifacts
        println(io, artifact.name)
        println(io, "  total: ", artifact.output)
        println(io, "  reference total: ", artifact.reference)
        println(io, "  per-obs length: ", length(artifact.per_obs))
        println(io, "  max |grad - analytic|: ",
                maximum(abs, artifact.gradient .- artifact.analytic_gradient))
        println(io, "  recipes (total / per-obs): ",
                artifact.total_recipes, " / ", artifact.perobs_recipes)
    end
    artifacts
end

end # module BatchedExamples

if abspath(PROGRAM_FILE) == @__FILE__
    BatchedExamples.run()
end
