# Documentation-ready views over the executable compatibility examples.
module CompatibilityArtifacts

using ReactiveKernels

if !isdefined(parentmodule(@__MODULE__), :ReactiveObjectsExamples)
    Base.include(parentmodule(@__MODULE__), joinpath(
        @__DIR__, "preexisting_reactiveobjects.jl",
    ))
end

using ..ReactiveObjectsExamples

export ExampleArtifact, reactiveobjects_artifacts
export all_artifacts

const REACTIVEOBJECTS_ORIGIN =
    "ReactiveObjects.jl dev@118e73b86dcd8bb8854d1f535249b49008575a6e"

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

all_artifacts() = reactiveobjects_artifacts()

end # module CompatibilityArtifacts
