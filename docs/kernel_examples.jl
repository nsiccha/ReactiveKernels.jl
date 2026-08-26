module ReactiveKernelsDocs

using Base64
using Documenter
using LinearAlgebra
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

# Drift-proof: read the ACTUAL `@reactive ... needle(...) = begin … end` authoring
# block out of a source file, so the docs show the real kernel definition (the
# recipe math + update methods) rather than an opaque constructor call. Captures
# from the `@reactive` line carrying `needle` through the first column-0 `end`.
const _REACTIVE_NUTS_SRC = joinpath(dirname(@__DIR__), "src", "reactive_nuts.jl")
function read_reactive_block(file::AbstractString, needle::AbstractString)
    lines = readlines(file)
    start = findfirst(l -> occursin("@reactive", l) && occursin(needle, l), lines)
    start === nothing && error("no @reactive block for $(needle) in $(file)")
    stop = findnext(l -> rstrip(l) == "end", lines, start + 1)
    stop === nothing && error("unterminated @reactive block for $(needle)")
    join(lines[start:stop], "\n")
end

# The live value read through a getter (source or derived) of the executed object.
# `invokelatest` because the displayed source may have defined fresh recipe methods
# (e.g. the analytic potential_gradient!) into the page sandbox in a newer world.
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
                          note::AbstractString = "", definition::AbstractString = "")
    program = reactive_program(object)
    handles = _artifact_handles(object)
    handle = getproperty(handles, getter)
    (; name, origin, source, object, program, getter, definition,
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
        defblock = isempty(artifact.definition) ? "" : string(
            "# ─── The kernel definition — the ACTUAL @reactive authoring in\n",
            "#     src/reactive_nuts.jl (the recipe math + any update method) ───\n",
            artifact.definition, "\n\n",
            "# ─── How you construct it and interact with it ───\n")
        source = string(defblock,
                        "# Origin: ", artifact.origin, "\n", artifact.source, "\n\n",
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
# its exact build-executed public constructor source. The gradient boundary is a
# CONSUMER-SUPPLIED ANALYTIC gradient (no DI/Enzyme) so the compiled kernel, not the
# differentiation, is the visible content. `note` labels the ordinary non-reactive
# orchestration around each program.
const _FIVE_PROGRAM_SNIPPETS = (
    (name = :nuts_group,
     origin = "reactive_nuts_group (build executed)",
     def_needle = "_reactive_nuts_group_object",
     getter = :dham,
     note = "the transition recursion, RNG draws, U-turn criteria, leapfrog, and " *
            "tree/proposal scratch are ordinary inferred Julia over these handles",
     source = raw"""
# The public model boundary is a SCALAR potential with a CONSUMER-SUPPLIED ANALYTIC
# gradient — no DifferentiationInterface/Enzyme machinery — so the compiled kernel,
# not the differentiation, is the visible content. Here U(x) = ‖x‖²/2, so ∇U(x) = x.
potential_gradient!(gradient, position) =
    (copyto!(gradient, position); sum(abs2, position) / 2)
dimension = 4

object = reactive_nuts_group(potential_gradient!,
    Matrix{Float64}(I, dimension, dimension), zeros(dimension), zeros(dimension))"""),
    (name = :dual_averaging,
     origin = "dual_averaging_state (build executed)",
     def_needle = "_dual_averaging_object",
     getter = :current,
     note = "fit! is an ordinary method advancing the accumulator sources",
     source = raw"""
object = dual_averaging_state(0.35; target = 0.8)
fit!(object, 0.9)          # ordinary method: advance the accumulator sources
fit!(object, 0.72)"""),
    (name = :welford_variance,
     origin = "welford_var (build executed)",
     def_needle = "_welford_object",
     getter = :var,
     note = "step! folds observations into the n/mean/var sources in place",
     source = raw"""
dimension = 4
object = welford_var(dimension)
step!(object, [0.5, -1.0, 2.0, 0.25])   # ordinary in-place source update
step!(object, [1.5, -0.5, 1.0, 0.75])"""),
    (name = :trajectory_stats,
     origin = "trajectory_stats (build executed)",
     def_needle = "_trajectory_object",
     getter = :pots,
     note = "the recorder callback and positions/gradients VIEWS are ordinary " *
            "methods; the history/index buffers are its HAVE sources",
     source = raw"""
dimension = 4
object = trajectory_stats(dimension)"""),
    (name = :sampling_stats,
     origin = "sampling_stats (build executed)",
     def_needle = "_sampling_object",
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
    Core.eval(mod, :(using ReactiveKernels, LinearAlgebra, Random))
    artifacts = Any[]
    for snippet in _FIVE_PROGRAM_SNIPPETS
        _evaluate_source(mod, snippet.source)          # build-execute the public source
        object = Core.eval(mod, :object)               # the live object from that source
        push!(artifacts, program_artifact(snippet.name, snippet.origin,
                                          snippet.source, object, snippet.getter;
                                          note = snippet.note,
                                          definition = read_reactive_block(
                                              _REACTIVE_NUTS_SRC, snippet.def_needle)))
    end
    assert_program_coverage(artifacts, _FIVE_PROGRAM_INVENTORY)
    render_program_examples(artifacts)
end

# ===========================================================================
# Unit A — the fused stateless-composed NUTS leaf (BENCHMARK feasibility tip).
#
# This is the docs' build-executed copy of the fused leaf independently accepted
# at benchmark tip 71f37a2 (`benchmark/nuts_fused_leaf_proto.jl`, `build_leaf` /
# `prepare_leaf`). It composes the EXISTING public stateless core
#   Graph -> plan(have, want) -> lower -> _nonallocating_ast -> _prepare_nonallocating
# over the EXISTING NUTS ops (`_grad_bundle`/`_vg_*`/`_energy_error`/`_is_divergent`),
# so the numeric work is bit-identical to the reactive group's per-endpoint
# Hamiltonian — but the hot leaf is ONE straight-line schedule with no
# get!/validity/invalidation bookkeeping. It is BENCHMARK-only and BORROWED-output
# (returned arrays alias the kernel's owned caches); the production owned-slot form
# is PLANNED (see the architecture page). Keep in lockstep with the benchmark leaf;
# the coverage gate below asserts the 13-recipe / 9-have / 9-want inventory.
# ===========================================================================
const _RK = ReactiveKernels

# Leaf-only arithmetic ops — token-for-token matches to `_group_leapfrog!` so the
# fused schedule is bit-identical to the reactive leapfrog + energy error.
_lf_half(mom, stepsize, grad) = mom .- 0.5 .* stepsize .* grad   # @. mom -= 0.5*stepsize*grad
_lf_drift(pos, stepsize, vel) = pos .+ stepsize .* vel           # @. pos += stepsize*velocity
_lf_solve(chol, mom) = chol \ mom                                # velocity = M^-1 * mom
_lf_add(a, b) = a + b
_lf_dot(a, b) = dot(a, b)
# kinetic = (logdet(chol) + dot(mom, vel)) / 2, with logdet HOISTED to a persistent
# per-trajectory scalar (constant while the metric is fixed) — bit-identical to
# `_kinetic_energy`, but the straight-line leaf never recomputes logdet.
_lf_kin(logdet_chol, dotmv) = (logdet_chol + dotmv) / 2

# Leaf cache_apply: each recipe reuses its owned slot in place. Mirrors the reactive
# group's `_nuts_cache_apply`; scalars/projections stay pure.
@inline function _leaf_apply!(cache, ::typeof(_lf_half), mom, stepsize, grad)
    @. cache = mom - 0.5 * stepsize * grad
    cache
end
@inline function _leaf_apply!(cache, ::typeof(_lf_drift), pos, stepsize, vel)
    @. cache = pos + stepsize * vel
    cache
end
@inline function _leaf_apply!(cache, ::typeof(_lf_solve), chol, mom)
    copyto!(cache, mom)
    ldiv!(chol, cache)
    cache
end
@inline function _leaf_apply!(cache::_RK._ValueGradient, ::typeof(_RK._grad_bundle), pgrad, pos)
    cache.value = pgrad(cache.gradient, pos)   # THE single gradient call of the leaf
    cache
end
@inline _leaf_apply!(cache, op, args...) = op(args...)   # pure / scalar / projection

# Force a CONCRETE return type (T, not Union{Nothing,T}) so the lowered kernel's
# recipe outputs stay concretely typed — no Union propagation / boxing in warm IR.
@inline function _leaf_cache_apply(slot::Base.RefValue{Union{Nothing,T}}, op, args...) where {T}
    cached = slot[]
    val::T = cached === nothing ? op(args...) : _leaf_apply!(cached, op, args...)
    slot[] = val
    val
end

# The exact 13-recipe leaf graph. HAVE (9): {pos,mom,old_grad} change each call;
# {chol,stepsize,init_ham,threshold,pgrad,logdet_chol} are the persistent partition.
# WANT (9): the complete new endpoint state + dham + diverged.
function _build_fused_leaf(V, S, CH, PG, VG)
    g = _RK.Graph()
    pos   = _RK.value(:pos, V);         mom  = _RK.value(:mom, V)
    ograd = _RK.value(:old_grad, V);    chol = _RK.value(:chol, CH)
    ss    = _RK.value(:stepsize, S);    ih   = _RK.value(:init_ham, S)
    thr   = _RK.value(:threshold, S);   pg   = _RK.value(:pgrad, PG)
    ldc   = _RK.value(:logdet_chol, S)                     # persistent per-trajectory scalar

    half  = _RK.value(:half_mom, V);    vel1  = _RK.value(:vel1, V)
    npos  = _RK.value(:new_pos, V);     nvg   = _RK.value(:new_vg, VG)
    ngrad = _RK.value(:new_grad, V);    nval  = _RK.value(:new_value, S)
    nmom  = _RK.value(:new_mom, V);     dotmv = _RK.value(:dotmv, S)
    kin   = _RK.value(:kinetic, S);     nvel  = _RK.value(:new_vel, V)
    nham  = _RK.value(:new_ham, S);     dh    = _RK.value(:dham, S)
    dvg   = _RK.value(:diverged, Bool)

    _RK.add!(g; inputs = (mom, ss, ograd),  outputs = half,  op = _lf_half)
    _RK.add!(g; inputs = (chol, half),      outputs = vel1,  op = _lf_solve)
    _RK.add!(g; inputs = (pos, ss, vel1),   outputs = npos,  op = _lf_drift)
    _RK.add!(g; inputs = (pg, npos),        outputs = nvg,   op = _RK._grad_bundle)
    _RK.add!(g; inputs = nvg,               outputs = ngrad, op = _RK._vg_gradient)
    _RK.add!(g; inputs = nvg,               outputs = nval,  op = _RK._vg_value)
    _RK.add!(g; inputs = (half, ss, ngrad), outputs = nmom,  op = _lf_half)
    _RK.add!(g; inputs = (chol, nmom),      outputs = nvel,  op = _lf_solve)
    _RK.add!(g; inputs = (nmom, nvel),      outputs = dotmv, op = _lf_dot)
    _RK.add!(g; inputs = (ldc, dotmv),      outputs = kin,   op = _lf_kin)
    _RK.add!(g; inputs = (nval, kin),       outputs = nham,  op = _lf_add)
    _RK.add!(g; inputs = (ih, nham),        outputs = dh,    op = _RK._energy_error)
    _RK.add!(g; inputs = (dh, thr),         outputs = dvg,   op = _RK._is_divergent)

    have = (pos, mom, ograd, chol, ss, ih, thr, pg, ldc)
    want = (npos, nmom, ngrad, nvel, nham, dh, dvg, nval, kin)
    g, have, want
end

function _prepare_fused_leaf(metric, pos0, pgrad)
    V = typeof(pos0); S = eltype(pos0)
    CH = typeof(cholesky(metric)); PG = typeof(pgrad); VG = _RK._ValueGradient{S,V}
    g, have, want = _build_fused_leaf(V, S, CH, PG, VG)
    p = _RK.plan(g; have = have, want = want)
    ast = _RK._nonallocating_ast(_RK.lower(p))
    kernel = _RK._prepare_nonallocating(p, ast, _leaf_cache_apply)
    (; g, have, want, plan = p, kernel)
end

# --- non-vacuous coverage gate for Unit A ----------------------------------
const _FUSED_LEAF_HAVE = (:pos, :mom, :old_grad, :chol, :stepsize, :init_ham,
                          :threshold, :pgrad, :logdet_chol)
const _FUSED_LEAF_WANT = (:new_pos, :new_mom, :new_grad, :new_vel, :new_ham,
                          :dham, :diverged, :new_value, :kinetic)
const _FUSED_LEAF_RECIPES = 13

"""
    assert_fused_leaf_coverage(built; have, want, recipes) -> built

Mechanical inventory gate for the fused leaf. Errors — failing the docs build — if
the planned program's HAVE/WANT boundary or its total recipe count diverges from the
expected benchmark inventory (a missing or extra recipe is caught). The expectation
is passed explicitly (defaulting to the declared benchmark inventory) so the gate is
demonstrably NON-VACUOUS: a tampered expectation is rejected.
"""
function assert_fused_leaf_coverage(built; have = _FUSED_LEAF_HAVE,
                                    want = _FUSED_LEAF_WANT, recipes = _FUSED_LEAF_RECIPES)
    got_have = Tuple(v.name for v in built.plan.have)
    got_want = Tuple(v.name for v in built.plan.want)
    Set(got_have) == Set(have) && length(got_have) == length(have) || error(
        "fused leaf HAVE diverged: got $(got_have) vs $(have)")
    Set(got_want) == Set(want) && length(got_want) == length(want) || error(
        "fused leaf WANT diverged: got $(got_want) vs $(want)")
    length(built.plan.recipes) == recipes || error(
        "fused leaf recipe count diverged: got $(length(built.plan.recipes)) want $(recipes)")
    built.kernel.plan === built.plan || error("fused leaf kernel plan is not the built plan")
    built
end

# The consumer analytic scalar-potential boundary + the persistent-partition seed,
# evaluated in the page sandbox exactly as the benchmark sets it up.
const _FUSED_LEAF_SETUP = raw"""
using LinearAlgebra, Random

# The model boundary is a SCALAR potential with a CONSUMER-SUPPLIED ANALYTIC gradient
# — no DifferentiationInterface/Enzyme machinery — so the KERNEL, not the
# differentiation, is the visible content. Here U(x) = ‖x‖²/2, so ∇U(x) = x;
# potential_gradient! fills the gradient buffer in place and returns the potential.
potential_gradient!(gradient, position) =
    (copyto!(gradient, position); sum(abs2, position) / 2)
dimension = 4

# Persistent per-trajectory partition: metric factor, its log-determinant, and the
# fixed reference energy `init_ham` — all constant while the metric/stepsize is fixed.
metric = Matrix{Float64}(I, dimension, dimension)
chol = cholesky(metric)
logdet_chol = logdet(chol)
stepsize = 0.35
threshold = -1000.0
Random.seed!(20260826)
position = randn(dimension)
momentum = randn(dimension)
old_gradient = similar(position)
init_potential = potential_gradient!(old_gradient, position)   # ∇U(pos), returns U(pos)
velocity0 = chol \ momentum
init_ham = init_potential + (logdet_chol + dot(momentum, velocity0)) / 2
"""

# The readable Raw-input pane: the exact fused-leaf construction shown to the reader
# (mirrors `_build_fused_leaf`/`_prepare_fused_leaf` above, which is what executes).
const _FUSED_LEAF_SOURCE = raw"""
# Reuse the EXISTING NUTS ops so the numeric work is identical to the reactive group.
using ReactiveKernels, LinearAlgebra
const RK = ReactiveKernels

# Leaf-only arithmetic — token-for-token matches to `_group_leapfrog!`.
_lf_half(mom, stepsize, grad) = mom .- 0.5 .* stepsize .* grad   # @. mom -= 0.5*stepsize*grad
_lf_drift(pos, stepsize, vel) = pos .+ stepsize .* vel           # @. pos += stepsize*velocity
_lf_solve(chol, mom) = chol \ mom                                # velocity = M^-1 * mom
_lf_dot(a, b) = dot(a, b)
_lf_kin(logdet_chol, dotmv) = (logdet_chol + dotmv) / 2          # logdet HOISTED out of the leaf
_lf_add(a, b) = a + b

# HAVE ports (9): {pos,mom,old_grad} change each call; the rest are the persistent
# per-trajectory partition. `pgrad` is the consumer's analytic potential_gradient!.
g = RK.Graph()
pos = RK.value(:pos, V);   mom = RK.value(:mom, V);   old_grad = RK.value(:old_grad, V)
chol = RK.value(:chol, CH); stepsize = RK.value(:stepsize, S); init_ham = RK.value(:init_ham, S)
threshold = RK.value(:threshold, S); pgrad = RK.value(:pgrad, PG); logdet_chol = RK.value(:logdet_chol, S)

# Intermediate + WANT values, then 13 single-output recipes. RK._grad_bundle is the
# ONE gradient call of the leaf; RK._vg_gradient/_vg_value are borrowed projections.
half = RK.value(:half_mom, V); vel1 = RK.value(:vel1, V); new_pos = RK.value(:new_pos, V)
new_vg = RK.value(:new_vg, VG); new_grad = RK.value(:new_grad, V); new_val = RK.value(:new_value, S)
new_mom = RK.value(:new_mom, V); new_vel = RK.value(:new_vel, V); dotmv = RK.value(:dotmv, S)
kinetic = RK.value(:kinetic, S); new_ham = RK.value(:new_ham, S); dham = RK.value(:dham, S)
diverged = RK.value(:diverged, Bool)

RK.add!(g; inputs = (mom, stepsize, old_grad),  outputs = half,     op = _lf_half)
RK.add!(g; inputs = (chol, half),               outputs = vel1,     op = _lf_solve)
RK.add!(g; inputs = (pos, stepsize, vel1),      outputs = new_pos,  op = _lf_drift)
RK.add!(g; inputs = (pgrad, new_pos),           outputs = new_vg,   op = RK._grad_bundle)
RK.add!(g; inputs = new_vg,                     outputs = new_grad, op = RK._vg_gradient)
RK.add!(g; inputs = new_vg,                     outputs = new_val,  op = RK._vg_value)
RK.add!(g; inputs = (half, stepsize, new_grad), outputs = new_mom,  op = _lf_half)
RK.add!(g; inputs = (chol, new_mom),            outputs = new_vel,  op = _lf_solve)
RK.add!(g; inputs = (new_mom, new_vel),         outputs = dotmv,    op = _lf_dot)
RK.add!(g; inputs = (logdet_chol, dotmv),       outputs = kinetic,  op = _lf_kin)
RK.add!(g; inputs = (new_val, kinetic),         outputs = new_ham,  op = _lf_add)
RK.add!(g; inputs = (init_ham, new_ham),        outputs = dham,     op = RK._energy_error)
RK.add!(g; inputs = (dham, threshold),          outputs = diverged, op = RK._is_divergent)

# Plan the have -> want query, then lower to a straight-line NON-ALLOCATING kernel
# whose recipes reuse owned caches in place (a hand-written cache policy; the
# production form consumes poc's compile_update owned-slot primitive instead).
have = (pos, mom, old_grad, chol, stepsize, init_ham, threshold, pgrad, logdet_chol)
want = (new_pos, new_mom, new_grad, new_vel, new_ham, dham, diverged, new_val, kinetic)
plan   = RK.plan(g; have = have, want = want)
kernel = RK._prepare_nonallocating(plan, RK._nonallocating_ast(RK.lower(plan)), _leaf_cache_apply)
"""

"""
    render_fused_leaf(mod) -> Markdown.MD

Build-execute the fused NUTS leaf (Unit A) at the consumer analytic
scalar-potential boundary, assert its 13-recipe coverage, run one leaf, and render
the actual program through the shared three-view UI. **Generated kernel** is the real
`code_expr` of the fused non-allocating schedule; **Compute DAG** is `plan`. Fails the
docs build if the inventory diverges or the generated view is not the live kernel's.
"""
function render_fused_leaf(mod::Module)
    _evaluate_source(mod, _FUSED_LEAF_SETUP)     # build-execute the analytic gradient boundary
    metric = Core.eval(mod, :metric)
    pos0   = Core.eval(mod, :position)
    mom0   = Core.eval(mod, :momentum)
    ograd0 = Core.eval(mod, :old_gradient)
    pgrad! = Core.eval(mod, :potential_gradient!)
    chol   = Core.eval(mod, :chol)
    ss     = Core.eval(mod, :stepsize)
    ih     = Core.eval(mod, :init_ham)
    thr    = Core.eval(mod, :threshold)
    ldc    = Core.eval(mod, :logdet_chol)

    built = _prepare_fused_leaf(metric, pos0, pgrad!)
    assert_fused_leaf_coverage(built)

    named_inputs = (; pos = pos0, mom = mom0, old_grad = ograd0, chol = chol,
                    stepsize = ss, init_ham = ih, threshold = thr,
                    pgrad = pgrad!, logdet_chol = ldc)
    # Positional call in the kernel's actual input order.
    input_tuple = Tuple(getproperty(named_inputs, v.name) for v in built.plan.have)
    raw = Base.invokelatest(built.kernel, input_tuple...)
    # Borrowed arrays alias the kernel's owned caches; snapshot before display.
    output = map(x -> x isa AbstractArray ? copy(x) : x, raw)
    named_output = NamedTuple{Tuple(v.name for v in built.plan.want)}(output)
    # Sanity on the executed evidence (real build-executed leaf).
    (named_output.diverged isa Bool && isfinite(named_output.dham) &&
     all(isfinite, named_output.new_pos)) || error(
        "fused leaf produced a non-finite / malformed result")

    generated = code_expr(built.kernel)
    generated == built.kernel.ast || error("fused leaf generated view is not the live kernel code_expr")

    source = string(
        "# Origin: fused NUTS leaf — benchmark feasibility tip 71f37a2 (build executed)\n",
        strip(_FUSED_LEAF_SOURCE, '\n'), "\n\n",
        "# Executed input (D=4, identity metric, seed 20260826)\n",
        _plain_repr(named_inputs), "\n\n",
        "# Actual output — one fused leaf (arrays are BORROWED: they alias the\n",
        "# kernel's owned caches and are valid until the next call)\n",
        _plain_repr(named_output),
    )
    kernel_view = string(
        "# The fused NON-ALLOCATING schedule: one straight-line function over the\n",
        "# recipe caches (`__caches__`) through the injected cache policy\n",
        "# (`__cache_apply__`) — NO get!/validity/invalidation in the hot leaf.\n\n",
        _generated_source(generated),
    )
    _three_pane_blocks!(Any[], "Unit A — the fused leaf", source, kernel_view, built.plan) |>
        Markdown.MD
end

# Build provenance — the exact commit the docs were build-executed from, so a reader
# can tie every generated pane and inventory to a source SHA (drift-proofing).
function build_commit()
    try
        strip(read(`git -C $(dirname(@__DIR__)) rev-parse --short=12 HEAD`, String))
    catch
        "unavailable"
    end
end

"""
    render_build_commit() -> Markdown.MD

A one-line build-provenance stamp naming the exact commit the page was
build-executed from.
"""
function render_build_commit()
    sha = build_commit()
    Markdown.MD(Markdown.Paragraph(Any[
        "Every generated pane, DAG, and inventory below was build-executed from commit ",
        Markdown.Code(sha), ".",
    ]))
end

# The reviewed source-faithful `@kernel` NUTS authoring fixture, shown as a
# NON-EXECUTABLE, SOURCE-ONLY target. Read DRIFT-PROOF from the repo file at build
# time — NOT executed, NO generated-kernel/Compute-DAG pane, and NO parity, allocation,
# performance, or production claim (the stateful `@kernel` lowering is mid-implementation:
# construction-only). Sourced from the reviewed fixture at 725ac9b (ReactiveHMC.jl-faithful).
const _AUTHORING_FIXTURE_PATH =
    joinpath(dirname(@__DIR__), "benchmark", "nuts_kernel_authoring_fixture.jl")

"""
    render_authoring_fixture() -> Markdown.MD

Render the reviewed source-faithful `@kernel` NUTS authoring fixture as a plain code
block, read drift-proof from `benchmark/nuts_kernel_authoring_fixture.jl` at build time.
It is NOT executed, no generated-kernel/Compute-DAG pane is produced, and no
parity/allocation/performance/production claim is made — this is the illustrative
reviewed *source* surface (compiler lowering in progress), not an executable sampler.
"""
function render_authoring_fixture()
    src = read(_AUTHORING_FIXTURE_PATH, String)
    Markdown.MD(Any[Markdown.Code("julia", rstrip(src))])
end

end # module ReactiveKernelsDocs
