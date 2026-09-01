module ReactiveHMCDocsInteractions

using ReactiveKernelsNUTSExamples

"""Exact build-executed API interaction for one compatibility phase-point panel."""
function phasepoint_interaction(name::Symbol)
    definitions = Dict(
        :euclidean_phasepoint => (
            family = :euclidean,
            kinetic = :gaussian,
            prepared = :ham,
            have = (:pot_f, :grad_f, :pos, :mom, :metric),
            want = :ham,
            inputs = raw"""
inputs = (;
    pot_f = variant.sources.pot_f,
    grad_f = variant.sources.grad_f,
    pos,
    mom,
    metric = variant.geometry(pos).metric,
)""",
        ),
        :relativistic_euclidean_phasepoint => (
            family = :euclidean,
            kinetic = :relativistic,
            prepared = :ham,
            have = (:pot_f, :grad_f, :pos, :mom, :metric),
            want = :ham,
            inputs = raw"""
inputs = (;
    pot_f = variant.sources.pot_f,
    grad_f = variant.sources.grad_f,
    pos,
    mom,
    metric = variant.geometry(pos).metric,
)""",
        ),
        :riemannian_phasepoint => (
            family = :riemannian,
            kinetic = :gaussian,
            prepared = :geometry,
            have = (
                :pot_f, :grad_f, :metric_f, :metric_grad_f,
                :metric_inverse_f, :pos,
            ),
            want = (:pot, :dpot, :metric, :metric_grad, :chol, :inv_metric),
            inputs = raw"""
inputs = (;
    pot_f = variant.sources.pot_f,
    grad_f = variant.sources.grad_f,
    metric_f = variant.sources.metric_f,
    metric_grad_f = variant.sources.metric_grad_f,
    metric_inverse_f = variant.sources.metric_inverse_f,
    pos,
)""",
        ),
        :relativistic_riemannian_phasepoint => (
            family = :riemannian,
            kinetic = :relativistic,
            prepared = :dpos,
            have = (:mom, :metric, :inv_metric, :metric_grad, :dpot),
            want = :dham_dpos,
            inputs = raw"""
geometry = variant.geometry(pos)
inputs = (;
    mom,
    metric = geometry.metric,
    inv_metric = geometry.inv_metric,
    metric_grad = geometry.metric_grad,
    dpot = geometry.dpot,
)""",
        ),
        :riemannian_softabs_phasepoint => (
            family = :softabs,
            kinetic = :gaussian,
            prepared = :geometry,
            have = (
                :pot_f, :grad_f, :premetric_f, :premetric_grad_f,
                :softabs_geometry_f, :pos,
            ),
            want = (
                :pot, :dpot, :premetric, :premetric_grad, :eigenvalues,
                :eigenvectors, :metric_eigenvalues, :q_inv, :jacobian,
            ),
            inputs = raw"""
inputs = (;
    pot_f = variant.sources.pot_f,
    grad_f = variant.sources.grad_f,
    premetric_f = variant.sources.premetric_f,
    premetric_grad_f = variant.sources.premetric_grad_f,
    softabs_geometry_f = variant.sources.softabs_geometry_f,
    pos,
)""",
        ),
        :relativistic_riemannian_softabs_phasepoint => (
            family = :softabs,
            kinetic = :relativistic,
            prepared = :geometry,
            have = (
                :pot_f, :grad_f, :premetric_f, :premetric_grad_f,
                :softabs_geometry_f, :pos,
            ),
            want = (
                :pot, :dpot, :premetric, :premetric_grad, :eigenvalues,
                :eigenvectors, :metric_eigenvalues, :q_inv, :jacobian,
            ),
            inputs = raw"""
inputs = (;
    pot_f = variant.sources.pot_f,
    grad_f = variant.sources.grad_f,
    premetric_f = variant.sources.premetric_f,
    premetric_grad_f = variant.sources.premetric_grad_f,
    softabs_geometry_f = variant.sources.softabs_geometry_f,
    pos,
)""",
        ),
    )
    haskey(definitions, name) || error("unknown phase-point docs interaction: $name")
    definition = definitions[name]
    constructor = string(definition.family, "_examples")
    string(
        "using ReactiveKernelsCompatibilityExamples\n",
        "examples = ReactiveKernelsCompatibilityExamples.ReactiveHMCExamples.",
        constructor, "()\n",
        "variant = examples.", definition.kinetic, "\n",
        "pos = [0.25, -0.5]\n",
        "mom = [0.4, 0.1]\n",
        definition.inputs, "\n",
        "kernel = prepare(\n",
        "    variant.spec;\n",
        "    have = ", repr(definition.have), ",\n",
        "    want = ", repr(definition.want), ",\n",
        ")\n",
        "output = kernel(Tuple(inputs)...)\n",
        "docs_example = (;\n",
        "    name = :", name, ",\n",
        "    origin = \"package-owned phase-point constructor → prepare → call\",\n",
        "    inputs, kernel, output,\n",
        ")",
    )
end

const PATHFINDER_HAVE = (
    :logdensity,
    :position,
    :gradient,
    :alpha,
    :step,
    :gradient_delta,
    :identity,
    :elbo_standard_draws,
    :output_standard_draws,
    :curvature_tolerance,
)

const PATHFINDER_JL_HAVE = (
    :logdensity,
    :position,
    :gradient,
    :alpha,
    :history_steps,
    :history_gradient_deltas,
    :parameter_identity,
    :history_identity,
    :elbo_standard_draws,
    :output_standard_draws,
)

"""Exact build-executed API interaction for one Pathfinder WANT cut."""
function pathfinder_interaction(name::Symbol)
    if name === :pathfinder_jl_compact_history
        return string(
            "fixture = PathfinderJLKernelAuthoringFixture\n",
            "inputs = fixture.pathfinder_jl_fixture_inputs()\n",
            "kernel = prepare(\n",
            "    fixture.pathfinder_jl_compact_candidate;\n",
            "    have = ", repr(PATHFINDER_JL_HAVE), ",\n",
            "    want = fixture.PATHFINDER_JL_OUTPUTS,\n",
            ")\n",
            "output = kernel(Tuple(inputs)...)\n",
            "docs_example = (;\n",
            "    name = :pathfinder_jl_compact_history,\n",
            "    origin = \"reviewed Pathfinder.jl compact-history fixture → prepare → call\",\n",
            "    inputs, kernel, output,\n",
            ")",
        )
    end
    want = if name === :pathfinder_inverse_bfgs_geometry
        (:alpha_next, :curvature_accepted, :covariance)
    elseif name === :pathfinder_local_gaussian_and_elbo
        :PATHFINDER_OUTPUTS
    else
        error("unknown Pathfinder docs interaction: $name")
    end
    want_source = want === :PATHFINDER_OUTPUTS ?
        "fixture.PATHFINDER_OUTPUTS" : repr(want)
    string(
        "fixture = PathfinderKernelAuthoringFixture\n",
        "values = fixture.pathfinder_fixture_inputs()\n",
        "inputs = (;\n",
        "    logdensity = values.logdensity,\n",
        "    position = values.positions[:, 2],\n",
        "    gradient = values.gradients[:, 2],\n",
        "    alpha = values.initial_alpha,\n",
        "    step = values.positions[:, 2] .- values.positions[:, 1],\n",
        "    gradient_delta = values.gradients[:, 1] .- values.gradients[:, 2],\n",
        "    identity = values.identity,\n",
        "    elbo_standard_draws = values.elbo_standard_draws[:, :, 1],\n",
        "    output_standard_draws = values.output_standard_draws[:, :, 1],\n",
        "    curvature_tolerance = values.curvature_tolerance,\n",
        ")\n",
        "kernel = prepare(\n",
        "    fixture.pathfinder_candidate;\n",
        "    have = ", repr(PATHFINDER_HAVE), ",\n",
        "    want = ", want_source, ",\n",
        ")\n",
        "output = kernel(Tuple(inputs)...)\n",
        "docs_example = (;\n",
        "    name = :", name, ",\n",
        "    origin = \"reviewed Pathfinder fixture → prepare WANT cut → call\",\n",
        "    inputs, kernel, output,\n",
        ")",
    )
end

const RKE_INTERACTION = raw"""
using TOML

RK = ReactiveKernels
fixture = Main.ReactiveHMCRKEFixture
lowering = Main.ReactiveHMCRKEFunctionalLowering.lambertw_minus_one
receipt = TOML.parsefile(joinpath(
    pkgdir(ReactiveKernels), "benchmark", "receipts",
    "reactivehmc-rke-ca9-v1.toml",
))
case = first(receipt["cases"])
bindings = RK.stateful_compiler_bindings(
    lambertw = RK.pure_callable_port(
        lowering, Tuple{Float64,Int}, Float64;
        functional_lowering = lowering,
    ),
)
compiled = RK.compile_stateful(
    fixture.rke, bindings, lowering; m = case["m"], c = case["c"],
)
state = compiled(lowering; m = case["m"], c = case["c"])
input = case["q"][2]
output = RK.stateful_call(state, Val(:quantile_sq), input)
isapprox(output, case["quantile_sq"][2]; atol = 128eps(Float64), rtol = 0) ||
    error("RKE docs interaction drifted from the independent receipt")
docs_interaction = (;
    name = :relativistic_kinetic_energy,
    kind = :native_execution,
    input = (; quantile_probability = input, m = case["m"], c = case["c"]),
    output = (; squared_momentum_quantile = output),
)
"""

function integrator_interaction(name::Symbol)
    method = name === :generalized_leapfrog ? :generalized_leapfrog! :
        name === :implicit_midpoint ? :implicit_midpoint! :
        error("unknown integrator docs interaction: $name")
    case_name = name === :generalized_leapfrog ? "generalized_leapfrog" :
        "implicit_midpoint"
    string(
        "using ReactiveKernelsCompatibilityExamples\n",
        "using TOML\n\n",
        "fixture = Main.ReactiveHMCIntegratorFixture\n",
        "endpoint = ReactiveKernelsCompatibilityExamples.ReactiveHMCExamples.",
        "riemannian_examples().gaussian\n",
        "receipt = TOML.parsefile(joinpath(\n",
        "    pkgdir(ReactiveKernels), \"benchmark\", \"receipts\",\n",
        "    \"reactivehmc-integrators-ca9-v1.toml\",\n",
        "))\n",
        "case = only(filter(case -> case[\"name\"] == \"", case_name,
        "\", receipt[\"cases\"]))\n",
        "transition = ReactiveKernels.compile_state_transition(\n",
        "    endpoint.spec,\n",
        "    ReactiveKernels.partial(fixture.", method,
        "; stepsize = case[\"stepsize\"], n_fi_steps = case[\"n_fi_steps\"]),\n",
        "    values(endpoint.sources),\n",
        ")\n",
        "state = ReactiveKernels.initial_transition_state(transition)\n",
        "result = transition(state)\n",
        "output = (; pos = result.pos, mom = result.mom, ham = result.ham)\n",
        "all(isapprox(output[field], case[string(field)]; atol = 2e-15, rtol = 2e-13) ",
        "for field in (:pos, :mom, :ham)) ||\n",
        "    error(\"integrator docs interaction drifted from the independent receipt\")\n",
        "docs_interaction = (;\n",
        "    name = :", name, ", kind = :native_execution,\n",
        "    input = (; stepsize = case[\"stepsize\"], n_fi_steps = case[\"n_fi_steps\"]),\n",
        "    output,\n",
        ")",
    )
end

const STATISTICS_INTERACTION = raw"""
using TOML

RK = ReactiveKernels
fixture = Main.ReactiveHMCStatisticsFixture
receipt = TOML.parsefile(joinpath(
    pkgdir(ReactiveKernels), "benchmark", "receipts",
    "reactivehmc-statistics-ca9-v1.toml",
))
inputs = receipt["inputs"]
sources = fixture.initial_statistics_sources(inputs["dimension"], 8, 4)
names = propertynames(sources)
positional = ntuple(index -> getproperty(sources, names[index]), 13)
keyword_names = Tuple(names[14:end])
keywords = NamedTuple{keyword_names}(ntuple(
    index -> getproperty(sources, keyword_names[index]), length(keyword_names),
))
compiled = RK.compile_stateful(fixture.statistics_state, positional...; keywords...)
state = compiled(positional...; keywords...)
RK.stateful_call(
    state, Val(:reset!), inputs["reset_pos"], inputs["reset_dham_dpos"],
    inputs["reset_pot"],
) && error("statistics docs interaction overflowed while resetting")
for event in receipt["events"]
    overflow = RK.stateful_call(
        state, Val(:record_trajectory!), event["go_forward"], event["pos"],
        event["dham_dpos"], event["pot"], event["dham"],
    )
    overflow && error("statistics docs interaction overflowed")
end
sample = first(receipt["samples"])
RK.stateful_call(
    state, Val(:record_sample!), sample["init_pos"], sample["stepsize"],
    sample["diverged"],
) && error("statistics docs interaction overflowed while recording a sample")
columns = state.first:(state.first + state.count - 1)
output = (;
    positions = copy(state.positions[:, columns]),
    trajectory_indices = copy(state.idxs[columns]),
    draw = copy(state.draws[:, 1]),
    acceptance_rate = state.acc_rate[1],
)
output.positions == reduce(hcat, receipt["trajectory"]["positions"]) ||
    error("statistics docs positions drifted from the independent receipt")
output.trajectory_indices == receipt["trajectory"]["idxs"] ||
    error("statistics docs indices drifted from the independent receipt")
docs_interaction = (;
    name = :statistics_state,
    kind = :native_execution,
    input = (; reset = inputs, events = length(receipt["events"])),
    output,
)
"""

const FIXED_HMC_INSPECTION = raw"""
using TOML

fixture = Main.ReactiveHMCHMCFixture
registration = ReactiveKernels.kernel_registration(fixture.hmc_state)
methods = ReactiveKernels.method_irs(fixture.hmc_state)
receipt = TOML.parsefile(joinpath(
    pkgdir(ReactiveKernels), "benchmark", "receipts",
    "reactivehmc-hmc-ca9-v1.toml",
))
map(method -> method.id.name, methods) == (:randbernoullilog, :step!) ||
    error("fixed-HMC MethodIR boundary drifted")
all(case -> case["rng_events"] ==
    (case["diverged"] ? ["normal"] : ["normal", "exponential"]),
    receipt["cases"]) || error("fixed-HMC physical RNG-order receipt drifted")
docs_interaction = (;
    name = :fixed_step_hmc,
    kind = :fixture_receipt_inspection,
    input = (; fixture = "reactivehmc_hmc_kernel_fixture.jl", receipt = receipt["schema"]),
    output = (;
        kernel_kind = registration.kind,
        captured_methods = map(method -> (method.id.name, method.control), methods),
        physical_cases = map(case ->
            (case["name"], case["rng_events"], case["diverged"]), receipt["cases"]),
        compiler_execution_claimed = false,
    ),
)
"""

end # module ReactiveHMCDocsInteractions
