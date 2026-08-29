module BijectorKernelExample

using ReactiveKernels
using LogExpFunctions: log1pexp

export positive_bijector, unit_interval_bijector, fused_bijector_model, demo
export BIJECTOR_KERNEL_SOURCE, BIJECTOR_DOCS_SOURCE

# These exact bytes define the runnable example and are also executed by the
# dedicated docs page. Keeping the mathematical kernels in one source authority
# prevents the displayed transform from drifting away from the tested one.
const BIJECTOR_KERNEL_SOURCE = raw"""
@kernel positive_bijector(unconstrained::Float64) = begin
    constrained::Float64 = exp(unconstrained)
    log_jacobian::Float64 = unconstrained
    return (constrained, log_jacobian)
end

@kernel unit_interval_bijector(unconstrained::Float64) = begin
    magnitude::Float64 = abs(unconstrained)
    tail::Float64 = exp(-magnitude)
    log_normalizer::Float64 = log1p(tail)

    constrained::Float64 = unconstrained >= 0 ?
        inv(1 + tail) : tail / (1 + tail)
    log_constrained::Float64 = unconstrained >= 0 ?
        -log_normalizer : unconstrained - log_normalizer
    log_complement::Float64 = unconstrained >= 0 ?
        -unconstrained - log_normalizer : -log_normalizer
    log_jacobian::Float64 = log_constrained + log_complement
    return (constrained, log_jacobian)
end

@kernel fused_bijector_model(log_scale::Float64,
                             logit_probability::Float64) = begin
    (scale::Float64, scale_log_jacobian::Float64) =
        positive_bijector(log_scale)
    (probability::Float64, probability_log_jacobian::Float64) =
        unit_interval_bijector(logit_probability)
    parameters::NamedTuple{
        (:scale, :probability),Tuple{Float64,Float64}
    } = (; scale, probability)
    log_jacobian::Float64 = scale_log_jacobian + probability_log_jacobian
    return (parameters, log_jacobian)
end
"""

Base.include_string(
    @__MODULE__, BIJECTOR_KERNEL_SOURCE, "bijector-kernel-source.jl",
)

@doc """
A positive-support bijector authored as one have/want graph.

`want = :constrained` emits the exponential, `want = :log_jacobian` is an
identity cut at the unconstrained input, and requesting both emits the
exponential once while returning that same input as the log Jacobian.
""" positive_bijector

@doc """
A numerically stable real-to-unit-interval bijector.

The decomposition is deliberately expressed as individual mathematical nodes.
The constrained-only plan stops before every logarithmic node, the
Jacobian-only plan omits the constrained value, and the joint plan shares the
absolute-value/exponential tail work. The log Jacobian stays finite for finite
inputs even when the constrained floating-point value rounds to zero or one.
""" unit_interval_bijector

@doc """
Reuse both mathematical definitions inside one fused model transform graph.

The nested calls are expanded while the graph is built. Selecting only
`parameters` prunes every Jacobian-only node, selecting only `log_jacobian`
prunes both constrained-value consumers, and selecting both shares the common
unit-interval tail calculation.
""" fused_bijector_model

const BIJECTOR_DOCS_SOURCE = string(BIJECTOR_KERNEL_SOURCE, raw"""

inputs = (log_scale = 0.7, logit_probability = -0.4)

# Three static WANT boundaries over the same fused graph.
parameters_plan = plan(fused_bijector_model; want = :parameters)
jacobian_plan = plan(fused_bijector_model; want = :log_jacobian)
joint_plan = plan(
    fused_bijector_model; want = (:parameters, :log_jacobian),
)
kernel = prepare(joint_plan)
output = kernel(inputs.log_scale, inputs.logit_probability)

# These counts make demand pruning executable documentation. The emitted joint
# kernel contains primitive recipes only: neither reusable child remains as a
# nested runtime call.
@assert length(parameters_plan.recipes) == 5
@assert length(jacobian_plan.recipes) == 7
@assert length(joint_plan.recipes) == 10
emitted = sprint(show, code_expr(joint_plan))
@assert !occursin("positive_bijector", emitted)
@assert !occursin("unit_interval_bijector", emitted)

docs_example = (;
    name = :fused_bijector_model,
    origin = "Reusable constrained-parameter transforms (build executed)",
    inputs,
    kernel,
    output,
    parameters_plan,
    jacobian_plan,
)
""")

function demo(io::IO = stdout; unconstrained::Float64 = 0.7)
    for (name, spec) in (
        (:positive, positive_bijector),
        (:unit_interval, unit_interval_bijector),
    )
        println(io, name)
        for want in (:constrained, :log_jacobian,
                     (:constrained, :log_jacobian))
            selected = plan(spec; want)
            println(io, "\nwant = ", repr(want))
            println(io, explain(selected))
            println(io, "\nvalue = ", prepare(selected)(unconstrained))
        end
        println(io)
    end
    nothing
end

end # module BijectorKernelExample

if abspath(PROGRAM_FILE) == @__FILE__
    BijectorKernelExample.demo()
end
