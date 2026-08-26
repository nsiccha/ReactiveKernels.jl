module ReactiveKernelsDocs

using Base64
using Documenter
using Markdown
using ReactiveKernels

struct RawHTML
    content::String
end

# DocumenterVitepress prefers image/svg+xml over text/html when both are
# showable. Wrap a rich result when the docs specifically require its
# interactive HTML surface rather than the static SVG fallback.
struct HTMLResult{T}
    content::T
end

function Base.show(io::IO, mime::MIME"text/html", result::HTMLResult)
    html = sprint(show, mime, result.content; context = io)
    # DocumenterVitepress places HTML results inside a JavaScript template
    # literal. Preserve backslashes across that boundary so embedded scripts,
    # JSON escapes, and regular expressions reach the browser intact.
    print(io, replace(html, "\\" => "\\\\"))
end

function html_result(content)
    showable(MIME"text/html"(), content) || error(
        "docs HTML result must provide a text/html display",
    )
    HTMLResult(content)
end

# Julia 1.10's stdlib Markdown predates Markdown.HTMLBlock. Teach the
# Documenter-owned MarkdownAST conversion about this docs-only block so @eval
# results can contain real HTML instead of a printed Julia value.
Documenter.MarkdownAST._convert_block(
    nodefn::Documenter.MarkdownAST.NodeFn,
    block::RawHTML,
) = nodefn(Documenter.RawNode(:html, block.content))

function _example_title(name::Symbol)
    join(uppercasefirst.(split(string(name), '_')), " ")
end

function _plain_repr(value)
    sprint(show, MIME"text/plain"(), value; context = :limit => false)
end

function _generated_source(expr::Expr)
    sprint(Base.show_unquoted, expr; context = :limit => false)
end

function _dag_html(plan::Plan)
    # The docs content column is deliberately narrow. A top-to-bottom layout
    # keeps labels readable there without changing the public API's horizontal
    # default for wider notebook and standalone surfaces.
    view = visualize(plan; orientation = :vertical)
    mime = MIME"text/html"()
    showable(mime, view) || error(
        "visualize(plan) must provide the interactive text/html docs surface",
    )
    html = sprint(show, mime, view; context = :limit => false)
    payload = base64encode(html)
    """
<ClientOnly>
  <div v-exec-scripts="'$payload'"></div>
</ClientOnly>
"""
end

function _evaluate_source(mod::Module, displayed::AbstractString)
    parsed = Meta.parseall(displayed; filename = "reactive-kernels-doc-example.jl")
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    for expression in expressions
        expression isa LineNumberNode && continue
        Core.eval(mod, expression)
    end
    nothing
end

function setup_eight_schools!(mod::Module)
    if !isdefined(mod, :EightSchoolsExample)
        Base.include(mod, joinpath(@__DIR__, "..", "examples", "eight_schools.jl"))
    end
    Core.eval(mod, :(using .EightSchoolsExample:
        EightSchoolsParameters, NewGroupPrediction,
        SchoolVector, UnconstrainedParameters, PredictionInnovations,
        EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA))
    nothing
end

function setup_linear_regression!(mod::Module)
    if !isdefined(mod, :LinearRegressionExample)
        Base.include(mod, joinpath(@__DIR__, "..", "examples", "linear_regression.jl"))
    end
    Core.eval(mod, :(using .LinearRegressionExample:
        LinearRegressionParameters, LinearPrediction,
        DataVector, UnconstrainedParameters,
        LINREG_X, LINREG_Y))
    nothing
end

function setup_beta_binomial!(mod::Module)
    if !isdefined(mod, :BetaBinomialExample)
        Base.include(mod, joinpath(@__DIR__, "..", "examples", "beta_binomial.jl"))
    end
    Core.eval(mod, :(using .BetaBinomialExample:
        BetaBinomialParameters, CountVector,
        BETA_BINOMIAL_TRIALS, BETA_BINOMIAL_SUCCESSES))
    nothing
end

function setup_poisson_gamma!(mod::Module)
    if !isdefined(mod, :PoissonGammaExample)
        Base.include(mod, joinpath(@__DIR__, "..", "examples", "poisson_gamma.jl"))
    end
    Core.eval(mod, :(using .PoissonGammaExample:
        PoissonGammaParameters, CountVector, POISSON_COUNTS))
    nothing
end

function setup_dugongs!(mod::Module)
    if !isdefined(mod, :DugongsGrowthExample)
        Base.include(mod, joinpath(@__DIR__, "..", "examples", "dugongs_growth.jl"))
    end
    Core.eval(mod, :(using .DugongsGrowthExample:
        DugongsParameters, UnconstrainedParameters, RealVector,
        DUGONGS_AGE, DUGONGS_LENGTH))
    nothing
end

function setup_arma11!(mod::Module)
    if !isdefined(mod, :ARMA11Example)
        Base.include(mod, joinpath(@__DIR__, "..", "examples", "arma11.jl"))
    end
    Core.eval(mod, :(using .ARMA11Example:
        ARMAParameters, UnconstrainedParameters, RealVector, ARMA_SERIES))
    nothing
end

function setup_gaussian_mixture!(mod::Module)
    if !isdefined(mod, :GaussianMixtureExample)
        Base.include(mod, joinpath(@__DIR__, "..", "examples", "gaussian_mixture.jl"))
    end
    Core.eval(mod, :(using .GaussianMixtureExample:
        MixtureParameters, UnconstrainedParameters, RealVector,
        MIXTURE_OBSERVATIONS))
    nothing
end

function setup_online_stats!(mod::Module)
    if !isdefined(mod, :OnlineStatsExample)
        Base.include(mod, joinpath(@__DIR__, "..", "examples", "online_stats.jl"))
    end
    Core.eval(mod, :(using Statistics))
    Core.eval(mod, :(using .OnlineStatsExample:
        MomentsAccumulator, HMCDiagnosticsAccumulator))
    nothing
end

# The ONE shared three-view UI (Raw input / Generated kernel / Compute DAG). Both
# the stateless PreparedKernel path and the ReactiveProgram path emit through this;
# there is no second renderer. `dag` is the exact `Plan` consumed by `visualize`.
function _three_pane_blocks!(blocks, title, source, generated, dag::Plan)
    push!(blocks, RawHTML("""
<h2>$(title)</h2>
<div class="rk-example" data-rk-example>
<div data-rk-pane="source">
"""))
    push!(blocks, Markdown.Code("julia", source))
    push!(blocks, RawHTML("""
</div>
<div data-rk-pane="kernel">
"""))
    push!(blocks, Markdown.Code("julia", generated))
    push!(blocks, RawHTML("""
</div>
<div data-rk-pane="dag">
$(_dag_html(dag))
</div>
</div>
"""))
    blocks
end

"""
    render_examples(artifacts) -> Markdown.MD

Render executable documentation artifacts in the standard three-view UI. Each
artifact supplies its exact raw source/call, the `code_expr` captured from the
executed `PreparedKernel`, and the selected `Plan` consumed by `visualize`.
"""
function render_examples(artifacts)
    blocks = Any[]
    for artifact in artifacts
        artifact.kernel isa PreparedKernel || error(
            "$(artifact.name) did not provide a PreparedKernel",
        )
        artifact.dag === artifact.kernel.plan || error(
            "$(artifact.name) DAG is not its PreparedKernel plan",
        )
        artifact.generated == code_expr(artifact.kernel) || error(
            "$(artifact.name) generated view is not its PreparedKernel code_expr",
        )

        source = string(
            "# Origin: ", artifact.origin, "\n",
            artifact.source, "\n\n",
            "# Executed input\n", _plain_repr(artifact.inputs), "\n\n",
            "# Actual output\n", _plain_repr(artifact.output),
        )
        generated = _generated_source(artifact.generated)
        _three_pane_blocks!(blocks, _example_title(artifact.name), source, generated, artifact.dag)
    end
    Markdown.MD(blocks)
end

# --- ReactiveProgram artifacts (the actual compiled reactive programs) ----------

# The underlying reactive object (unwrapping a nominal wrapper) and its handles.
_artifact_object(object) =
    object isa ReactiveKernels.ReactiveObject ? object : getfield(object, :object)
_artifact_handles(object) = getfield(_artifact_object(object), :handles)

# The live value read through a getter (source or derived) of the executed object.
# `invokelatest` because the displayed source may have defined fresh recipe methods
# (e.g. the DI+Enzyme potential_gradient!) into the page sandbox in a newer world.
_getter_value(object, getter::Symbol) =
    Base.invokelatest(getproperty, _artifact_object(object), getter)

_is_source(program::ReactiveProgram, handle) =
    program.sources[ReactiveKernels._slot_index(handle)]

# The full recipe/value inventory of a program's exposed handles: which names are
# HAVE sources and which are DERIVED getter nodes. The one-to-one coverage gate
# compares this against a declared expectation, so a missing/extra recipe in any of
# the five programs fails the docs build.
function _program_inventory(program::ReactiveProgram, handles)
    derived = Symbol[]; sources = Symbol[]
    for name in keys(handles)
        push!(_is_source(program, getproperty(handles, name)) ? sources : derived, name)
    end
    (; derived = sort(derived), sources = sort(sources),
       recipe_count = length(program.plan.recipes))
end

"""
    program_artifact(name, origin, source, object, getter; note="") -> NamedTuple

One-to-one docs artifact for the ACTUAL compiled `ReactiveProgram` behind a public
reactive object/wrapper. `program = reactive_program(object)` (tied to the
build-executed object), `generated` is the real `code_expr(program, handle)` of the
selected load-bearing `getter`, `dag` is that same `program.plan`, and `inventory`
is the full source/derived recipe census used by the coverage gate. `note` labels
the ordinary non-reactive orchestration (recursion, RNG, U-turn, leapfrog,
adaptation/statistics update methods) so no fake DAG is invented for it.
"""
function program_artifact(name::Symbol, origin::AbstractString,
                          source::AbstractString, object, getter::Symbol;
                          note::AbstractString = "")
    program = reactive_program(object)
    handles = _artifact_handles(object)
    handle = getproperty(handles, getter)
    (; name, origin, source, object, program, getter,
       getter_is_source = _is_source(program, handle),
       generated = code_expr(program, handle),
       output = _getter_value(object, getter),   # executed value read via the live getter
       dag = program.plan,
       inventory = _program_inventory(program, handles), note)
end

"""
    render_program_examples(artifacts) -> Markdown.MD

Render ReactiveProgram artifacts through the SAME three-view UI as
[`render_examples`](@ref). Raw input is the build-executed public constructor
source; Generated kernel is the REAL `code_expr` of the selected load-bearing getter
of the actual `reactive_program` (a derived fused getter where the program has
derived nodes; a source-slot getter for a state-only program); Compute DAG is that
same `program.plan`, which visually carries the full recipe graph. Fails if any
artifact's program/generated/DAG identity diverges from the live program.
"""
function render_program_examples(artifacts)
    blocks = Any[]
    for artifact in artifacts
        artifact.program isa ReactiveProgram || error(
            "$(artifact.name) did not provide a ReactiveProgram",
        )
        artifact.dag === artifact.program.plan || error(
            "$(artifact.name) DAG is not its reactive_program plan",
        )
        handle = getproperty(_artifact_handles(artifact.object), artifact.getter)
        artifact.generated == code_expr(artifact.program, handle) || error(
            "$(artifact.name) generated view diverges from the live code_expr($(artifact.getter))",
        )

        # Executed evidence: the live value read through the selected getter after the
        # displayed source ran must match the artifact (real build-executed example).
        _getter_value(artifact.object, artifact.getter) == artifact.output || error(
            "$(artifact.name) recorded output diverges from the live getter value",
        )

        kind = artifact.getter_is_source ?
            "state-only reactive program — the pane shows the generated source-slot " *
            "getter for `$(artifact.getter)`; its updates ($(artifact.note)) are " *
            "ordinary Julia methods over these HAVE sources" :
            "the pane shows the generated FUSED getter for the derived node " *
            "`$(artifact.getter)` — one straight-line function, no graph traversal " *
            "($(artifact.note))"
        source = string("# Origin: ", artifact.origin, "\n", artifact.source, "\n\n",
                        "# Actual output — ", artifact.getter, " (read through the ",
                        "live getter after executing the above)\n",
                        _plain_repr(artifact.output))
        generated = string("# ", kind, "\n\n", _generated_source(artifact.generated))
        _three_pane_blocks!(blocks, _example_title(artifact.name), source, generated, artifact.dag)
    end
    Markdown.MD(blocks)
end

"""
    assert_program_coverage(artifacts, expected) -> artifacts

Mechanical one-to-one coverage gate. `expected` maps each program name to its
declared `(derived, sources)` handle inventory. Errors — failing the docs build — if
the rendered set is not EXACTLY the expected program names, if any artifact's
`program`/`dag` is not the live `reactive_program(object)`/`program.plan`, if the
selected getter's `code_expr` diverges from the live program, or if any program's
actual source/derived recipe inventory diverges from its declared expectation
(so a missing or extra recipe in any of the five programs is caught).
"""
function assert_program_coverage(artifacts, expected)
    names = [a.name for a in artifacts]
    Set(names) == Set(keys(expected)) && length(names) == length(expected) || error(
        "program coverage mismatch: rendered $(sort(names)) vs expected $(sort(collect(keys(expected))))",
    )
    for a in artifacts
        a.program === reactive_program(a.object) || error(
            "$(a.name) program is not the live reactive_program(object)",
        )
        a.dag === a.program.plan || error("$(a.name) DAG is not program.plan")
        handle = getproperty(_artifact_handles(a.object), a.getter)
        a.generated == code_expr(a.program, handle) || error(
            "$(a.name) generated getter diverges from the live program",
        )
        _getter_value(a.object, a.getter) == a.output || error(
            "$(a.name) recorded output diverges from the live getter value",
        )
        want = expected[a.name]
        # Compare the exposed source/derived handle census AND the total plan recipe
        # count, so an added/removed INTERMEDIATE recipe (one not exposed as a handle)
        # is also caught — otherwise "missing OR extra recipes fail" would overclaim.
        (sort(collect(want.derived)) == a.inventory.derived &&
         sort(collect(want.sources)) == a.inventory.sources &&
         want.recipe_count == a.inventory.recipe_count) || error(
            "$(a.name) recipe inventory diverged from the declared coverage:\n" *
            "  derived: got $(a.inventory.derived)\n           want $(sort(collect(want.derived)))\n" *
            "  sources: got $(a.inventory.sources)\n           want $(sort(collect(want.sources)))\n" *
            "  recipe_count: got $(a.inventory.recipe_count) want $(want.recipe_count)",
        )
    end
    artifacts
end

"""
    execute_example(mod, code; result=:docs_example) -> Markdown.MD

Evaluate the exact displayed Julia source and render the resulting prepared
kernel through the standard three-view UI. The source must bind `result` to a
named tuple with `name`, `origin`, `inputs`, `kernel`, and `output` fields.
"""
function execute_example(mod::Module, code::AbstractString;
                         result::Symbol = :docs_example, setup = nothing)
    displayed = strip(code, '\n')
    Core.eval(mod, :(using ReactiveKernels))
    setup === nothing || setup(mod)
    _evaluate_source(mod, displayed)
    executed = Core.eval(mod, result)
    executed.kernel isa PreparedKernel || error(
        "$(executed.name) did not produce a PreparedKernel",
    )
    # The displayed source may have included fresh recipe methods into the page
    # sandbox. Re-run through the latest world so this validation exercises the
    # actual kernel without tripping Julia's world-age boundary.
    observed = Base.invokelatest(executed.kernel, Tuple(executed.inputs)...)
    isequal(observed, executed.output) || error(
        "$(executed.name) displayed output does not match a fresh kernel execution",
    )
    artifact = merge(
        executed,
        (;
            source = displayed,
            generated = code_expr(executed.kernel),
            dag = executed.kernel.plan,
        ),
    )
    render_examples((artifact,))
end

# The five compiled reactive programs of the public NUTS workflow, each defined by
# its exact build-executed public constructor source. The gradient boundary is the
# ACTUAL sampled path: a scalar potential differentiated once by DI + reverse-mode
# Enzyme into an owned buffer (as in examples/nuts.jl). `note` labels the ordinary
# non-reactive orchestration around each program.
const _FIVE_PROGRAM_SNIPPETS = (
    (name = :nuts_group,
     origin = "reactive_nuts_group (build executed)",
     getter = :dham,
     note = "the transition recursion, RNG draws, U-turn criteria, leapfrog, and " *
            "tree/proposal scratch are ordinary inferred Julia over these handles",
     source = raw"""
# The public model boundary is a SCALAR potential; its gradient goes through
# DifferentiationInterface + reverse-mode Enzyme, prepared once and written into
# the sampler's owned buffer in place (the actual sampled path — no handwritten
# gradient callback). See examples/nuts.jl for the full runnable workflow.
backend = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                     function_annotation = Enzyme.Const)
potential(q) = sum(abs2, q) / 2
dimension = 4
preparation = prepare_gradient(potential, backend, zeros(dimension))
potential_gradient!(gradient, position) = first(value_and_gradient!(
    potential, gradient, preparation, backend, position))

object = reactive_nuts_group(potential_gradient!,
    Matrix{Float64}(I, dimension, dimension), zeros(dimension), zeros(dimension))"""),
    (name = :dual_averaging,
     origin = "dual_averaging_state (build executed)",
     getter = :current,
     note = "fit! is an ordinary method advancing the accumulator sources",
     source = raw"""
object = dual_averaging_state(0.35; target = 0.8)
fit!(object, 0.9)          # ordinary method: advance the accumulator sources
fit!(object, 0.72)"""),
    (name = :welford_variance,
     origin = "welford_var (build executed)",
     getter = :var,
     note = "step! folds observations into the n/mean/var sources in place",
     source = raw"""
dimension = 4
object = welford_var(dimension)
step!(object, [0.5, -1.0, 2.0, 0.25])   # ordinary in-place source update
step!(object, [1.5, -0.5, 1.0, 0.75])"""),
    (name = :trajectory_stats,
     origin = "trajectory_stats (build executed)",
     getter = :pots,
     note = "the recorder callback and positions/gradients VIEWS are ordinary " *
            "methods; the history/index buffers are its HAVE sources",
     source = raw"""
dimension = 4
object = trajectory_stats(dimension)"""),
    (name = :sampling_stats,
     origin = "sampling_stats (build executed)",
     getter = :draws,
     note = "the per-transition callback appends to its HAVE sources; the reduced " *
            "views are ordinary methods",
     source = raw"""
dimension = 4
object = sampling_stats(trajectory_stats(dimension))"""),
)

# Declared one-to-one recipe/value inventory for each of the five programs. The
# coverage gate fails the docs build if any program's ACTUAL source/derived recipe
# census diverges from this — so a missing or extra recipe is caught mechanically.
const _FIVE_PROGRAM_INVENTORY = Dict(
    :nuts_group => (
        derived = (:active_ham, :bwd_dham_dmom, :bwd_dpot_dpos, :bwd_ham, :bwd_kin,
                   :bwd_kinetic, :bwd_pot, :bwd_valgrad, :chol_metric, :dham, :diverged,
                   :fwd_dham_dmom, :fwd_dpot_dpos, :fwd_ham, :fwd_kin, :fwd_kinetic,
                   :fwd_pot, :fwd_valgrad, :init_dham_dmom, :init_dpot_dpos, :init_ham,
                   :init_kin, :init_kinetic, :init_pot, :init_valgrad),
        sources = (:acceptance_sum, :bwd_mom, :bwd_pos, :depth, :fwd_mom, :fwd_pos,
                   :gofwd, :init_mom, :init_pos, :last_diverged, :last_energy_error,
                   :may_continue, :may_sample, :metric, :min_dham, :n_steps,
                   :potential_gradient!),
        recipe_count = 25),
    :dual_averaging => (
        derived = (:current, :final, :log_current),
        sources = (:center, :error, :iteration, :log_final, :offset,
                   :regularization_scale, :relaxation_exponent, :target),
        recipe_count = 3),
    :welford_variance => (derived = (), sources = (:mean, :n, :var), recipe_count = 0),
    :trajectory_stats => (
        derived = (),
        sources = (:count, :dhams, :dim, :first, :gradient_storage, :idxs,
                   :position_storage, :pots),
        recipe_count = 0),
    :sampling_stats => (
        derived = (),
        sources = (:acc_rate, :diverged, :draws, :full_history, :full_idxs,
                   :n_steps, :stepsizes, :trajectory),
        recipe_count = 0),
)

"""
    render_five_programs(mod) -> Markdown.MD

Build-execute the five public constructors of the compiled-reactive NUTS workflow
(NUTS group, DualAveragingState, WelfordVariance, TrajectoryStats, SamplingStats),
assert one-to-one program/getter/DAG coverage, and render each actual
`reactive_program` through the shared three-view UI. Fails the docs build if any of
the five programs lacks an artifact or diverges from its live program.
"""
function render_five_programs(mod::Module)
    Core.eval(mod, :(using ReactiveKernels, LinearAlgebra, Random,
                           DifferentiationInterface))
    Core.eval(mod, :(import Enzyme))
    artifacts = Any[]
    for snippet in _FIVE_PROGRAM_SNIPPETS
        _evaluate_source(mod, snippet.source)          # build-execute the public source
        object = Core.eval(mod, :object)               # the live object from that source
        push!(artifacts, program_artifact(snippet.name, snippet.origin,
                                          snippet.source, object, snippet.getter;
                                          note = snippet.note))
    end
    assert_program_coverage(artifacts, _FIVE_PROGRAM_INVENTORY)
    render_program_examples(artifacts)
end

end # module ReactiveKernelsDocs
