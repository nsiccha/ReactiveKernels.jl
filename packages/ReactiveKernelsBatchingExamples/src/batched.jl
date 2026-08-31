# Batched (vectorized) log densities as ordinary authored kernels.
#
# One transparent graph authors the pointwise plate and returns its sum. The
# planner can request the distinguished return, the named `pointwise` value, or
# both. Total-only lowering fuses the reduction without materializing the
# pointwise vector; both-wants lowering fills it and accumulates the return in a
# single traversal. No `Distributions.jl` call runs in the compute path; it is
# an independent oracle only.
#
# This module holds the build-executed, `Distributions.jl`-free source rendered
# in the docs. The zero-allocation value and gradient claims require the
# `MutatingFunctions` weak dependency and are proven separately in
# `test/test_batched_nonallocating.jl` (run through
# `test/run_nonallocating_integration.jl`); see also `docs/src/batched.md`.
module BatchedExamples

export BATCHED_SOURCE, BATCHED_PRIMAL_SOURCE, BATCHED_AD_SOURCE
export all_sources, evaluate_source, run

const BATCHED_PRIMAL_SOURCE = raw"""
using Distributions
using Random
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal

# The canonical RK `normal` kernel object supplies the transparent scalar
# endpoint; it is not `Distributions.jl.Normal`. The likelihood itself is
# authored once: `pointwise` remains a queryable graph value and the
# distinguished return is its sum.
@kernel normal_loglik(x, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        normal(li, si).logpdf(xi)
    end
    return sum(pointwise)
end

N = 1000
x = randn(Random.MersenneTwister(1), N)
location = 0.3
scale = 1.2

# Three WANT boundaries from the SAME authored graph. Positional application
# and the default plan request the distinguished return. Extraction selects
# the named pointwise value alone or pointwise plus return.
pointwise_spec = extract(normal_loglik; want = :pointwise)
both_spec = extract(normal_loglik; want = (:pointwise, :__return__))
total_plan = plan(normal_loglik)
pointwise_plan = plan(pointwise_spec)
both_plan = plan(both_spec)
const total_kernel = prepare(normal_loglik)
const pointwise_kernel = prepare(pointwise_spec)
const both_kernel = prepare(both_spec)

ordinary_total = normal_loglik(x, location, scale)
total = total_kernel(x, location, scale)
pointwise = pointwise_kernel(x, location, scale)
pointwise_and_total = both_kernel(x, location, scale)

# Distributions.jl is an independent oracle only; it never runs in the kernel.
reference_pointwise = logpdf.(Normal(location, scale), x)
reference = sum(reference_pointwise)

# The total-only AST has no pointwise allocation. Pointwise-only and both-wants
# each traverse the plate exactly once; the latter also accumulates the return
# during that same traversal.
total_ast = code_expr(total_kernel)
pointwise_ast = code_expr(pointwise_kernel)
both_ast = code_expr(both_kernel)

@assert ordinary_total ≈ reference
@assert total ≈ reference
@assert pointwise ≈ reference_pointwise
@assert pointwise_and_total == (pointwise, total)
@assert length(pointwise) == N
@assert !occursin("Distributions", string(code_expr(total_kernel)))
@assert !occursin("Distributions", string(code_expr(pointwise_kernel)))
@assert !occursin("Distributions", string(code_expr(both_kernel)))
@assert !occursin("similar", string(total_ast))

docs_example = (;
    name = :normal_loglik,
    origin = "authored Normal plate with pointwise and return queries (build executed)",
    inputs = (; x, location, scale),
    kernel = total_kernel,
    output = total,
    ordinary_total,
    reference,
    pointwise_kernel,
    pointwise,
    reference_pointwise,
    both_kernel,
    pointwise_and_total,
    total_plan,
    pointwise_plan,
    both_plan,
    total_ast,
    pointwise_ast,
    both_ast,
)
"""

const BATCHED_AD_SOURCE = BATCHED_PRIMAL_SOURCE * raw"""

using DifferentiationInterface
import Enzyme

# Reverse-mode gradient of the distinguished return over the whole
# N-dimensional batch in one pass. DI marks location and scale constant; the
# fused total has no active pointwise container.
analytic_gradient = @. -(x - location) / scale^2
backend = AutoEnzyme(; mode = Enzyme.Reverse)
prepared_gradient = prepare_ad(
    total_kernel, backend, x, location, scale; active = :x,
)
gradient = ad_gradient(prepared_gradient, x, location, scale)

@assert gradient ≈ analytic_gradient

docs_example = merge(docs_example, (; gradient, analytic_gradient))
"""

# Preserve the public executable example as the complete primal + AD source.
const BATCHED_SOURCE = BATCHED_AD_SOURCE

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
        println(io, "  pointwise length: ", length(artifact.pointwise))
        println(io, "  max |grad - analytic|: ",
                maximum(abs, artifact.gradient .- artifact.analytic_gradient))
        println(io, "  recipes (total / pointwise / both): ",
                length(artifact.total_plan.recipes), " / ",
                length(artifact.pointwise_plan.recipes), " / ",
                length(artifact.both_plan.recipes))
    end
    artifacts
end

end # module BatchedExamples

if abspath(PROGRAM_FILE) == @__FILE__
    BatchedExamples.run()
end
