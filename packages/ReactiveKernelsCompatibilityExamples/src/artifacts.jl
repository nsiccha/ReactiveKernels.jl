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

const CHAIN_SOURCE = """@kernel graph(x::Float64) = begin
    z::Float64 = 2x
    w::Float64 = z + 1
end"""

const DIAMOND_SOURCE = """@kernel graph(a::Float64) = begin
    b::Float64 = a + 1
    c::Float64 = a + 2
    d::Float64 = b + c
end"""

const SHARED_SOURCE = """@kernel graph(x::Float64) = begin
    a::Float64 = x + 1
    b::Float64 = 2a
    c::Float64 = 3a
end"""

const EUCLIDEAN_SOURCE = """@kernel spec(pos::typeof(pos0), mom::typeof(mom0),
             metric::typeof(metric0)) = begin
    pot::typeof(pot0) = pot_f(pos)
    (pot, dpot::typeof(dpot0)) = grad_f(pos)
    chol::typeof(chol0) = cholesky(metric)
    dham_dmom::typeof(dmom0) = metric_dmom(chol, mom)
    ham::typeof(ham0) = hamiltonian(pot, chol, mom)
    return ham
end"""

const RIEMANNIAN_SOURCE = """@kernel spec(pos::typeof(pos0), mom::typeof(mom0)) = begin
    (pot::typeof(pot0), dpot::typeof(dpot0),
     metric::typeof(metric0), metric_grad::typeof(metric_grad0)) =
        metric_gradient(pos)
    chol::typeof(chol0) = cholesky(metric)
    inv_metric::typeof(inv_metric0) = Symmetric(inv(chol))
    dham_dpos::typeof(dpos0) = metric_dpos(mom, chol, inv_metric, metric_grad, dpot)
    return (pot, dpot, metric, metric_grad, chol, inv_metric, dham_dpos)
end"""

const SOFTABS_SOURCE = """@kernel spec(pos::typeof(pos0)) = begin
    (pot::typeof(pot0), dpot::typeof(dpot0),
     premetric::typeof(premetric0), premetric_grad::typeof(premetric_grad0)) =
        premetric_gradient(pos)
    (eigenvalues::typeof(eigenvalues0), eigenvectors::typeof(eigenvectors0),
     metric_eigenvalues::typeof(metric_eigenvalues0), q_inv::typeof(q_inv0),
     jacobian::typeof(jacobian0)) = softabs_geometry(premetric)
    return (pot, dpot, premetric, premetric_grad, eigenvalues, eigenvectors,
            metric_eigenvalues, q_inv, jacobian)
end"""

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
            CHAIN_SOURCE,
            (; x = 10.0),
            chain.kernel,
        ),
        _artifact(
            :reactiveobjects_diamond,
            REACTIVEOBJECTS_ORIGIN,
            DIAMOND_SOURCE,
            (; a = 10.0),
            diamond.kernel,
        ),
        _artifact(
            :reactiveobjects_shared,
            REACTIVEOBJECTS_ORIGIN,
            SHARED_SOURCE,
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
    euclidean_sources = euclidean.gaussian.sources
    relativistic_euclidean_sources = euclidean.relativistic.sources
    riemannian_sources = riemannian.gaussian.sources
    softabs_sources = softabs.gaussian.sources
    relativistic_softabs_sources = softabs.relativistic.sources

    ExampleArtifact[
        _artifact(
            :euclidean_phasepoint,
            REACTIVEHMC_ORIGIN,
            EUCLIDEAN_SOURCE,
            (;
                pot_f = euclidean_sources.pot_f,
                grad_f = euclidean_sources.grad_f,
                pos, mom, metric,
            ),
            euclidean.gaussian.prepared.ham,
        ),
        _artifact(
            :relativistic_euclidean_phasepoint,
            REACTIVEHMC_ORIGIN,
            EUCLIDEAN_SOURCE,
            (;
                pot_f = relativistic_euclidean_sources.pot_f,
                grad_f = relativistic_euclidean_sources.grad_f,
                pos, mom, metric,
            ),
            euclidean.relativistic.prepared.ham,
        ),
        _artifact(
            :riemannian_phasepoint,
            REACTIVEHMC_ORIGIN,
            RIEMANNIAN_SOURCE,
            (;
                pot_f = riemannian_sources.pot_f,
                grad_f = riemannian_sources.grad_f,
                metric_f = riemannian_sources.metric_f,
                metric_grad_f = riemannian_sources.metric_grad_f,
                metric_inverse_f = riemannian_sources.metric_inverse_f,
                pos,
            ),
            riemannian.gaussian.prepared.geometry,
        ),
        _artifact(
            :relativistic_riemannian_phasepoint,
            REACTIVEHMC_ORIGIN,
            RIEMANNIAN_SOURCE,
            (;
                mom,
                metric = relativistic_geometry.metric,
                inv_metric = relativistic_geometry.inv_metric,
                metric_grad = relativistic_geometry.metric_grad,
                dpot = relativistic_geometry.dpot,
            ),
            riemannian.relativistic.prepared.dpos,
        ),
        _artifact(
            :riemannian_softabs_phasepoint,
            REACTIVEHMC_ORIGIN,
            SOFTABS_SOURCE,
            (;
                pot_f = softabs_sources.pot_f,
                grad_f = softabs_sources.grad_f,
                premetric_f = softabs_sources.premetric_f,
                premetric_grad_f = softabs_sources.premetric_grad_f,
                softabs_geometry_f = softabs_sources.softabs_geometry_f,
                pos,
            ),
            softabs.gaussian.prepared.geometry,
        ),
        _artifact(
            :relativistic_riemannian_softabs_phasepoint,
            REACTIVEHMC_ORIGIN,
            SOFTABS_SOURCE,
            (;
                pot_f = relativistic_softabs_sources.pot_f,
                grad_f = relativistic_softabs_sources.grad_f,
                premetric_f = relativistic_softabs_sources.premetric_f,
                premetric_grad_f = relativistic_softabs_sources.premetric_grad_f,
                softabs_geometry_f =
                    relativistic_softabs_sources.softabs_geometry_f,
                pos,
            ),
            softabs.relativistic.prepared.geometry,
        ),
        _artifact(
            :leapfrog,
            REACTIVEHMC_ORIGIN,
            EUCLIDEAN_SOURCE,
            (; grad_f = euclidean_sources.grad_f, pos),
            euclidean.gaussian.prepared.dpos,
        ),
        _artifact(
            :generalized_leapfrog,
            REACTIVEHMC_ORIGIN,
            RIEMANNIAN_SOURCE,
            (;
                pot_f = riemannian_sources.pot_f,
                grad_f = riemannian_sources.grad_f,
                metric_f = riemannian_sources.metric_f,
                metric_grad_f = riemannian_sources.metric_grad_f,
                metric_inverse_f = riemannian_sources.metric_inverse_f,
                pos,
            ),
            riemannian.gaussian.prepared.geometry,
        ),
        _artifact(
            :implicit_midpoint,
            REACTIVEHMC_ORIGIN,
            RIEMANNIAN_SOURCE,
            (;
                pot_f = riemannian_sources.pot_f,
                grad_f = riemannian_sources.grad_f,
                metric_f = riemannian_sources.metric_f,
                metric_grad_f = riemannian_sources.metric_grad_f,
                metric_inverse_f = riemannian_sources.metric_inverse_f,
                pos,
            ),
            riemannian.gaussian.prepared.geometry,
        ),
        _artifact(
            :multistep,
            REACTIVEHMC_ORIGIN,
            SOFTABS_SOURCE,
            (;
                pot_f = softabs_sources.pot_f,
                grad_f = softabs_sources.grad_f,
                premetric_f = softabs_sources.premetric_f,
                premetric_grad_f = softabs_sources.premetric_grad_f,
                softabs_geometry_f = softabs_sources.softabs_geometry_f,
                pos,
            ),
            softabs.gaussian.prepared.geometry,
        ),
    ]
end

all_artifacts() = vcat(reactiveobjects_artifacts(), reactivehmc_artifacts())

end # module CompatibilityArtifacts
