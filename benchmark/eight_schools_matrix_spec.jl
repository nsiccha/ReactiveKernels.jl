module EightSchoolsMatrixSpec

include(joinpath(@__DIR__, "model_benchmark_matrix_spec.jl"))
using .ModelBenchmarkMatrixSpec

export EIGHT_SCHOOLS_MODELS, EIGHT_SCHOOLS_BOUNDARIES, EIGHT_SCHOOLS_OUTCOMES
export EIGHT_SCHOOLS_RK_CONFIGURATIONS, EIGHT_SCHOOLS_COMPARATORS
export matrix_support, headline_cells

const EIGHT_SCHOOLS_MODELS = ("centered",)
const EIGHT_SCHOOLS_BOUNDARIES = (
    "packed_unconstrained", "constrained_parameters", "minimal_likelihood")
const EIGHT_SCHOOLS_OUTCOMES = ("joint", "prior", "likelihood", "pointwise")
const EIGHT_SCHOOLS_RK_CONFIGURATIONS = RK_BENCHMARK_CONFIGURATIONS

const EIGHT_SCHOOLS_COMPARATORS = (
    (id = "manual_primal", provider = "manual_julia",
     differentiation = "primal"),
    (id = "manual_ad", provider = "manual_julia",
     differentiation = "value_and_gradient"),
    (id = "turing_primal", provider = "turing",
     differentiation = "primal"),
    (id = "turing_ad", provider = "turing",
     differentiation = "value_and_gradient"),
    (id = "practicalbayes_primal", provider = "practical_bayes",
     differentiation = "primal"),
    (id = "practicalbayes_ad", provider = "practical_bayes",
     differentiation = "value_and_gradient"),
)

"Return the deliberate public-API state of one Eight Schools RK matrix cell."
function matrix_support(configuration, boundary::AbstractString,
                        outcome::AbstractString)
    boundary in EIGHT_SCHOOLS_BOUNDARIES ||
        return ("unsupported", "unknown input boundary")
    outcome in EIGHT_SCHOOLS_OUTCOMES ||
        return ("unsupported", "unknown outcome")

    if boundary == "minimal_likelihood" && outcome in ("joint", "prior")
        return ("unsupported",
                "joint and prior are unavailable from the minimal likelihood boundary")
    end
    if configuration.data == "bound" && outcome == "prior"
        return ("not_applicable",
                "the minimal prior cut has no data ports to bind")
    end
    if configuration.allocation == "nonallocating" &&
        boundary == "constrained_parameters"
        return (
            "unsupported",
            "the constrained NamedTuple inverse is a three-output recipe and the public nonallocating pass requires single-output recipes",
        )
    end
    if configuration.differentiation == "value_and_gradient"
        if outcome == "pointwise"
            configuration.compiler == "native" &&
                boundary != "constrained_parameters" && return ("supported", "")
            return (
                "unsupported",
                configuration.compiler == "reactant" ?
                    "the public compiled-AD surface has no compiled reverse-pullback verb; no pointwise VJP is claimed" :
                    "DifferentiationInterface/Enzyme cannot annotate the constrained NamedTuple/pointwise MixedDuplicated cross-product",
            )
        end
        boundary == "constrained_parameters" &&
            configuration.compiler == "reactant" && return (
            "unsupported",
            "native structured gradients are public, but compiled AD has no structured active-argument/result contract",
        )
    end
    ("supported", "")
end

"The sampler-relevant cells that the Eight Schools model must account for."
headline_cells() = Tuple(
    let support = matrix_support(
            configuration, "packed_unconstrained", "joint")
        (model = "centered", configuration = configuration.id,
         boundary = "packed_unconstrained", outcome = "joint",
         state = support[1], reason = support[2])
    end
    for configuration in EIGHT_SCHOOLS_RK_CONFIGURATIONS
)

end # module EightSchoolsMatrixSpec
