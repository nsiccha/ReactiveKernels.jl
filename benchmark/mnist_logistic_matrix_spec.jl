module MNISTLogisticMatrixSpec

include(joinpath(@__DIR__, "model_benchmark_matrix_spec.jl"))
using .ModelBenchmarkMatrixSpec

export MNIST_MODELS, MNIST_BOUNDARIES, MNIST_OUTCOMES
export MNIST_RK_CONFIGURATIONS, MNIST_COMPARATORS
export matrix_support, headline_cells

const MNIST_MODELS = ("idiomatic", "vcat_free")
const MNIST_BOUNDARIES = ("packed_unconstrained", "structured_parameters")
const MNIST_OUTCOMES = ("joint", "prior", "likelihood", "pointwise")

# These are the public, composable RK execution configurations. Keeping the
# axes explicit avoids presenting a hand-picked collection of unrelated
# receipts as though it were one benchmark matrix.
const MNIST_RK_CONFIGURATIONS = RK_BENCHMARK_CONFIGURATIONS

# Controls use their own public interfaces. Turing has the same two model
# styles as RK (materialized softmax versus implicit-reference likelihood);
# the handwritten control is the independent numerical oracle.
const MNIST_COMPARATORS = (
    (id = "manual_primal", provider = "manual_julia",
     model = "implicit_reference", differentiation = "primal"),
    (id = "manual_ad", provider = "manual_julia",
     model = "implicit_reference", differentiation = "value_and_gradient"),
    (id = "turing_idiomatic_primal", provider = "turing",
     model = "idiomatic", differentiation = "primal"),
    (id = "turing_idiomatic_ad", provider = "turing",
     model = "idiomatic", differentiation = "value_and_gradient"),
    (id = "turing_vcat_free_primal", provider = "turing",
     model = "vcat_free", differentiation = "primal"),
    (id = "turing_vcat_free_ad", provider = "turing",
     model = "vcat_free", differentiation = "value_and_gradient"),
    (id = "practicalbayes_idiomatic_primal", provider = "practical_bayes",
     model = "idiomatic", differentiation = "primal"),
    (id = "practicalbayes_idiomatic_ad", provider = "practical_bayes",
     model = "idiomatic", differentiation = "value_and_gradient"),
    (id = "practicalbayes_vcat_free_primal", provider = "practical_bayes",
     model = "vcat_free", differentiation = "primal"),
    (id = "practicalbayes_vcat_free_ad", provider = "practical_bayes",
     model = "vcat_free", differentiation = "value_and_gradient"),
)

"""
    matrix_support(configuration, boundary, outcome)

Return `(state, reason)` for one RK matrix cell. `state` is `"supported"`,
`"not_applicable"`, or `"unsupported"`. Runtime/compiler failures discovered
by a receipt are additional evidence; this function records only deliberate
public-API limits.
"""
function matrix_support(configuration, boundary::AbstractString,
                        outcome::AbstractString)
    boundary in MNIST_BOUNDARIES ||
        return ("unsupported", "unknown input boundary")
    outcome in MNIST_OUTCOMES || return ("unsupported", "unknown outcome")

    if configuration.data == "bound" && outcome == "prior"
        return ("not_applicable",
                "the minimal prior cut has no data ports to bind")
    end
    if configuration.differentiation == "value_and_gradient"
        outcome == "pointwise" && return (
            "unsupported",
            "pointwise is vector-valued and no matched public Jacobian/VJP contract is invented",
        )
        boundary == "structured_parameters" && return (
            "unsupported",
            "the structured (W, b) boundary has two active ports and RK exposes one active AD port",
        )
    end
    ("supported", "")
end

"The sampler-relevant cells that every RK model must account for."
headline_cells() = Tuple(
    let support = matrix_support(
            configuration, "packed_unconstrained", "joint")
        (model = model, configuration = configuration.id,
         boundary = "packed_unconstrained", outcome = "joint",
         state = support[1], reason = support[2])
    end
    for model in MNIST_MODELS, configuration in MNIST_RK_CONFIGURATIONS
)

end # module MNISTLogisticMatrixSpec
