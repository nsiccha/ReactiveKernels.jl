# Documentation-ready views over the executable compatibility examples.
module CompatibilityArtifacts

using ReactiveKernels

if !isdefined(parentmodule(@__MODULE__), :ReactiveObjectsExamples)
    Base.include(parentmodule(@__MODULE__), joinpath(
        @__DIR__, "preexisting_reactiveobjects.jl",
    ))
end
if !isdefined(parentmodule(@__MODULE__), :ReactiveHMCExamples)
    Base.include(parentmodule(@__MODULE__), joinpath(
        @__DIR__, "preexisting_reactivehmc.jl",
    ))
end

using ..ReactiveHMCExamples
using ..ReactiveObjectsExamples

export ExampleArtifact, reactiveobjects_artifacts, reactivehmc_artifacts
export all_artifacts

const REACTIVEOBJECTS_ORIGIN =
    "ReactiveObjects.jl dev@118e73b86dcd8bb8854d1f535249b49008575a6e"
const REACTIVEHMC_ORIGIN =
    "ReactiveHMC.jl main@ca9ea4ca41924bb0e1fadc01c717e1333916aba6"

"""
    ExampleArtifact

One executable three-view documentation record. `source` is the original raw
example/call, `generated` is the actual lowered kernel expression, and `dag` is
the exact selected `Plan`. Once the visualization layer is loaded,
`visualize(artifact.dag)` renders that structured plan with HAVE/WANT/selected
colors. `output` is produced by executing `kernel` on the recorded `inputs`.
"""
struct ExampleArtifact{I,K,O}
    name::Symbol
    origin::String
    source::String
    inputs::I
    kernel::K
    output::O
    generated::Expr
    dag::Plan
end

function _artifact(name, origin, source, inputs, kernel)
    output = kernel(Tuple(inputs)...)
    ExampleArtifact(
        name, origin, source, inputs, kernel, output, code_expr(kernel),
        kernel.plan,
    )
end

function reactiveobjects_artifacts()
    chain = chain_example()
    diamond = diamond_example()
    shared = shared_example()
    ExampleArtifact[
        _artifact(
            :reactiveobjects_chain,
            REACTIVEOBJECTS_ORIGIN,
            """@reactive _demo_chain(x) = begin
    z = 2x
    w = z + 1
end""",
            (; x = 10.0),
            chain.kernel,
        ),
        _artifact(
            :reactiveobjects_diamond,
            REACTIVEOBJECTS_ORIGIN,
            """@reactive _demo_diamond(a) = begin
    b = a + 1
    c = a + 2
    d = b + c
end""",
            (; a = 10.0),
            diamond.kernel,
        ),
        _artifact(
            :reactiveobjects_shared,
            REACTIVEOBJECTS_ORIGIN,
            """@reactive _demo_shared(x) = begin
    a = x + 1
    b = 2a
    c = 3a
end""",
            (; x = 5.0),
            shared.kernel,
        ),
    ]
end

function reactivehmc_artifacts()
    pos = [0.25, -0.5]
    mom = [0.4, 0.1]
    euclidean = euclidean_examples()
    riemannian = riemannian_examples()
    softabs = softabs_examples()
    metric = euclidean.gaussian.geometry(pos).metric
    relativistic_geometry = riemannian.relativistic.geometry(pos)

    ExampleArtifact[
        _artifact(
            :euclidean_phasepoint,
            REACTIVEHMC_ORIGIN,
            "euclidean_phasepoint(pot_f, grad_f, metric, pos, mom)",
            (; pos, metric, mom),
            euclidean.gaussian.prepared.ham,
        ),
        _artifact(
            :relativistic_euclidean_phasepoint,
            REACTIVEHMC_ORIGIN,
            "relativistic_euclidean_phasepoint(pot_f, grad_f, metric, pos, mom; c, m)",
            (; pos, metric, mom),
            euclidean.relativistic.prepared.ham,
        ),
        _artifact(
            :riemannian_phasepoint,
            REACTIVEHMC_ORIGIN,
            "riemannian_phasepoint(pot_f, grad_f, metric_f, metric_grad_f, pos, mom)",
            (; pos),
            riemannian.gaussian.prepared.geometry,
        ),
        _artifact(
            :relativistic_riemannian_phasepoint,
            REACTIVEHMC_ORIGIN,
            "relativistic_riemannian_phasepoint(pot_f, grad_f, metric_f, metric_grad_f, pos, mom; c, m)",
            (;
                mom,
                chol = relativistic_geometry.chol,
                inv_metric = relativistic_geometry.inv_metric,
                metric_grad = relativistic_geometry.metric_grad,
                dpot = relativistic_geometry.dpot,
            ),
            riemannian.relativistic.prepared.dpos,
        ),
        _artifact(
            :riemannian_softabs_phasepoint,
            REACTIVEHMC_ORIGIN,
            "riemannian_softabs_phasepoint(pot_f, grad_f, premetric_f, premetric_grad_f, pos, mom; alpha)",
            (; pos),
            softabs.gaussian.prepared.geometry,
        ),
        _artifact(
            :relativistic_riemannian_softabs_phasepoint,
            REACTIVEHMC_ORIGIN,
            "relativistic_riemannian_softabs_phasepoint(pot_f, grad_f, premetric_f, premetric_grad_f, pos, mom; alpha, c, m)",
            (; pos),
            softabs.relativistic.prepared.geometry,
        ),
        _artifact(
            :leapfrog,
            REACTIVEHMC_ORIGIN,
            "leapfrog!(phasepoint; stepsize = 0.1)",
            (; pos),
            euclidean.gaussian.prepared.dpos,
        ),
        _artifact(
            :generalized_leapfrog,
            REACTIVEHMC_ORIGIN,
            "generalized_leapfrog!(phasepoint; stepsize = 0.1, n_fi_steps = 4)",
            (; pos),
            riemannian.gaussian.prepared.geometry,
        ),
        _artifact(
            :implicit_midpoint,
            REACTIVEHMC_ORIGIN,
            "implicit_midpoint!(phasepoint; stepsize = 0.1, n_fi_steps = 4)",
            (; pos),
            riemannian.gaussian.prepared.geometry,
        ),
        _artifact(
            :multistep,
            REACTIVEHMC_ORIGIN,
            "multistep(generalized_leapfrog!; n_steps = 3)(phasepoint; stepsize = 0.06, n_fi_steps = 2)",
            (; pos),
            softabs.gaussian.prepared.geometry,
        ),
    ]
end

all_artifacts() = vcat(reactiveobjects_artifacts(), reactivehmc_artifacts())

end # module CompatibilityArtifacts
