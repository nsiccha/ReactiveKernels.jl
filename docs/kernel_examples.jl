module ReactiveKernelsDocs

using Base64
using Documenter
using LinearAlgebra
using Markdown
using ReactiveKernels
using TOML

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

function _readable_generated_source(expr::Expr, context, label)
    readable = ReactiveKernels._readable_expr(expr, context)
    occursin(r"__ops__\[\d+\]", string(readable)) && error(
        "$label readable generated view retained an opaque operation slot",
    )
    occursin(r"\boperation\(", string(readable)) && error(
        "$label readable generated view retained an unnamed operation",
    )
    _generated_source(readable)
end

# --- eval-throughput visualization ----------------------------------------
# A build-executed chart of the static eval-throughput receipt: grouped
# horizontal bars on a log time axis, faceted by mode, one bar per single-call
# latency series.
# The bars are read straight from the receipt TOML, so the picture cannot drift
# from the numbers. `RK + Reactant` and `Turing native` share the axis; Turing +
# Reactant is unsupported and omitted.
function _eval_throughput_receipt()
    joinpath(pkgdir(ReactiveKernels), "benchmark", "receipts",
             "eval-throughput-v1.toml")
end

function _fmt_ns(v)
    v >= 1000 ? string(round(v / 1000; digits = 1), " µs") :
                string(round(Int, v), " ns")
end

function eval_throughput_chart(; receipt = _eval_throughput_receipt())
    data = TOML.parsefile(receipt)
    measurements = data["measurements"]
    modes = ("primal", "gradient", "gq")
    sizes = (16, 256, 4096)
    # label, implementation, variant, color (chosen to read on light AND dark).
    series = (
        ("RK native", "reactivekernels", "native", "#0a9d6b"),
        ("RK + Reactant", "reactivekernels", "reactant", "#e08a1e"),
        ("Turing native", "turing", "native", "#7b6cd9"),
    )
    med(impl, variant, mode, size) = begin
        i = findfirst(m -> m["implementation"] == impl &&
                           m["variant"] == variant &&
                           m["mode"] == mode && m["size"] == size, measurements)
        i === nothing ? nothing : float(measurements[i]["median_ns"])
    end

    # Log time axis shared across facets.
    vmin, vmax = 10.0, 32_000.0
    lg(v) = log10(clamp(v, vmin, vmax))
    xL, xR = 132.0, 686.0                     # plot area in the 720-wide viewBox
    xpix(v) = xL + (lg(v) - lg(vmin)) / (lg(vmax) - lg(vmin)) * (xR - xL)

    barh, bargap, groupgap = 13.0, 5.0, 16.0
    facet_title_h, facet_gap = 26.0, 20.0
    rows_per_facet = length(sizes) * length(series)
    facet_body_h = length(sizes) *
        (length(series) * barh + (length(series) - 1) * bargap) +
        (length(sizes) - 1) * groupgap
    facet_h = facet_title_h + facet_body_h
    top = 44.0                                # legend band
    total_h = top + length(modes) * facet_h + (length(modes) - 1) * facet_gap + 24

    io = IOBuffer()
    print(io, """<svg viewBox="0 0 720 $(round(Int, total_h))" width="100%" """,
        """role="img" aria-label="Single-call evaluation latency: median per-call time """,
        """by mode, size and implementation" style="font: 12px/1.3 var(--vp-font-family-base, system-ui); color: var(--vp-c-text-1, currentColor);">""")

    # Legend.
    lx = xL
    for (label, _, _, color) in series
        print(io, """<rect x="$(round(lx;digits=1))" y="14" width="12" height="12" rx="2" fill="$color"/>""")
        print(io, """<text x="$(round(lx+17;digits=1))" y="24" fill="currentColor">$label</text>""")
        lx += 90 + 8 * length(label)
    end

    # Log gridlines + tick labels (10, 100, 1k, 10k).
    for (tick, tlab) in ((10, "10 ns"), (100, "100 ns"), (1000, "1 µs"),
                         (10_000, "10 µs"))
        gx = round(xpix(tick); digits = 1)
        print(io, """<line x1="$gx" y1="$(round(top;digits=1))" x2="$gx" """,
            """y2="$(round(top + length(modes)*facet_h + (length(modes)-1)*facet_gap; digits=1))" """,
            """stroke="var(--vp-c-divider, #ccc)" stroke-width="1" opacity="0.5"/>""")
        print(io, """<text x="$gx" y="$(round(top-6;digits=1))" text-anchor="middle" """,
            """fill="var(--vp-c-text-2, currentColor)" font-size="10">$tlab</text>""")
    end

    y = top
    for mode in modes
        print(io, """<text x="0" y="$(round(y+16;digits=1))" font-weight="600" """,
            """fill="currentColor">$mode</text>""")
        yb = y + facet_title_h
        for size in sizes
            print(io, """<text x="0" y="$(round(yb + (length(series)*barh + (length(series)-1)*bargap)/2 + 4; digits=1))" """,
                """fill="var(--vp-c-text-2, currentColor)" font-size="11">n = $size</text>""")
            for (label, impl, variant, color) in series
                v = med(impl, variant, mode, size)
                if v === nothing
                    yb += barh + bargap
                    continue
                end
                w = max(xpix(v) - xL, 1.0)
                print(io, """<rect x="$(round(xL;digits=1))" y="$(round(yb;digits=1))" """,
                    """width="$(round(w;digits=1))" height="$(round(barh;digits=1))" rx="2" """,
                    """fill="$color"><title>$mode, n=$size, $label: $(_fmt_ns(v))</title></rect>""")
                lx2 = xL + w + 5
                print(io, """<text x="$(round(lx2;digits=1))" y="$(round(yb+barh-2;digits=1))" """,
                    """font-size="10.5" fill="var(--vp-c-text-2, currentColor)">$(_fmt_ns(v))</text>""")
                yb += barh + bargap
            end
            yb += groupgap - bargap
        end
        y += facet_h + facet_gap
    end
    print(io, "</svg>")
    # Wrap in a Markdown.MD so Documenter's `@eval` renders the RawHTML block
    # through the registered MarkdownAST conversion (same path as the PPL panels).
    Markdown.MD(Any[RawHTML(String(take!(io)))])
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
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: EightSchoolsExample))
    end
    # Bind the observations and the shared distribution KernelSpecs consumed by
    # the displayed PPL assembly; no model-specific evaluator is injected.
    Core.eval(mod, :(using .EightSchoolsExample:
        EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA,
        NORMAL_LOGDENSITY, CAUCHY_LOGDENSITY))
    nothing
end

function setup_linear_regression!(mod::Module)
    if !isdefined(mod, :LinearRegressionExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: LinearRegressionExample))
    end
    Core.eval(mod, :(using .LinearRegressionExample:
        LinearRegressionParameters, LinearPrediction,
        DataVector, UnconstrainedParameters,
        LINREG_X, LINREG_Y,
        split_unconstrained, positive_scale, assemble_parameters,
        log_abs_det_jacobian, log_prior, pointwise_log_likelihood,
        sum_log_likelihood, total_log_density, predict_new))
    nothing
end

function setup_beta_binomial!(mod::Module)
    if !isdefined(mod, :BetaBinomialExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: BetaBinomialExample))
    end
    Core.eval(mod, :(using .BetaBinomialExample:
        BetaBinomialParameters, CountVector,
        BETA_BINOMIAL_TRIALS, BETA_BINOMIAL_SUCCESSES,
        logistic, assemble_parameters, log_abs_det_jacobian,
        log_prior, pointwise_log_likelihood, sum_log_likelihood,
        total_log_density, expected_successes))
    nothing
end

function setup_poisson_gamma!(mod::Module)
    if !isdefined(mod, :PoissonGammaExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: PoissonGammaExample))
    end
    Core.eval(mod, :(using .PoissonGammaExample:
        PoissonGammaParameters, CountVector, POISSON_COUNTS,
        positive_rate, assemble_parameters, log_abs_det_jacobian,
        log_prior, pointwise_log_likelihood, sum_log_likelihood,
        total_log_density, expected_count))
    nothing
end

function setup_dugongs!(mod::Module)
    if !isdefined(mod, :DugongsGrowthExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: DugongsGrowthExample))
    end
    Core.eval(mod, :(using .DugongsGrowthExample:
        DugongsParameters, UnconstrainedParameters, RealVector,
        DUGONGS_AGE, DUGONGS_LENGTH,
        split_unconstrained, bounded_lambda, sd_from_log_precision,
        assemble_parameters, log_abs_det_jacobian, log_prior,
        pointwise_log_likelihood, sum_log_likelihood,
        fused_log_likelihood, total_log_density, predicted_length))
    nothing
end

function setup_arma11!(mod::Module)
    if !isdefined(mod, :ARMA11Example)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: ARMA11Example))
    end
    Core.eval(mod, :(using .ARMA11Example:
        ARMAParameters, UnconstrainedParameters, RealVector, ARMA_SERIES,
        split_unconstrained, positive_scale, assemble_parameters,
        log_abs_det_jacobian, arma_errors, log_prior,
        pointwise_log_likelihood, sum_log_likelihood,
        total_log_density, one_step_forecast))
    nothing
end

function setup_gaussian_mixture!(mod::Module)
    if !isdefined(mod, :GaussianMixtureExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: GaussianMixtureExample))
    end
    Core.eval(mod, :(using .GaussianMixtureExample:
        MixtureParameters, UnconstrainedParameters, RealVector,
        MIXTURE_OBSERVATIONS, split_unconstrained, ordered_means,
        exp_scale, logistic, assemble_parameters, log_abs_det_jacobian,
        log_prior, pointwise_log_likelihood, sum_log_likelihood,
        fused_log_likelihood, total_log_density, component1_responsibility))
    nothing
end

function setup_online_stats!(mod::Module)
    if !isdefined(mod, :OnlineStatsExample)
        Base.include(mod, joinpath(@__DIR__, "..", "examples", "online_stats.jl"))
    end
    Core.eval(mod, :(using Statistics))
    Core.eval(mod, :(using Main.ReactiveKernelsNUTSExample: NUTSDiagnostics))
    Core.eval(mod, :(using .OnlineStatsExample:
        MomentsAccumulator, HMCDiagnosticsAccumulator))
    nothing
end

function _source_between(source::AbstractString, start_marker::AbstractString,
                         stop_marker::AbstractString)
    start = findfirst(start_marker, source)
    start === nothing && error("docs source marker is missing: $start_marker")
    search_from = nextind(source, last(start))
    stop = findnext(stop_marker, source, search_from)
    stop === nothing && error("docs source stop marker is missing: $stop_marker")
    duplicate = findnext(start_marker, source, search_from)
    duplicate === nothing || error("docs source marker is ambiguous: $start_marker")
    strip(source[first(start):prevind(source, first(stop))], '\n')
end

function _nutpie_kernel_sources()
    path = joinpath(pkgdir(ReactiveKernels), "examples",
                    "nutpie_diagonal_adaptation.jl")
    source = read(path, String)
    initialize = _source_between(
        source,
        "@kernel nutpie_diagonal_initialize(",
        "# One functional adaptation step.",
    )
    adaptation = _source_between(
        source,
        "@kernel nutpie_diagonal_adaptation(",
        "const INITIALIZE_INPUTS",
    )
    (; path, initialize, adaptation)
end

function _port_call_source(kernel_name::AbstractString, spec_name::AbstractString,
                           inputs, outputs)
    input_lines = join(("        " * string(name) for name in inputs), ",\n")
    string(
        kernel_name, " = prepare(", spec_name, ";\n",
        "    have = ", repr(inputs), ",\n",
        "    want = ", repr(outputs), ",\n",
        ")\n",
        "result = ", kernel_name, "(\n", input_lines, ",\n)",
    )
end

"""
    render_nutpie_diagonal_adaptation() -> Markdown.MD

Read the two real nutpie diagonal-adaptation `@kernel` definitions from their
example-owned source, execute their native prepared forms, and render the live
generated kernels and exact plans through the shared three-pane docs UI.
"""
function render_nutpie_diagonal_adaptation()
    example = Main.NutpieDiagonalAdaptationExample
    sources = _nutpie_kernel_sources()

    initialize_inputs = (;
        position = copy(example.ORACLE_INPUTS.init_position),
        gradient = copy(example.ORACLE_INPUTS.init_gradient),
    )
    initialize_output = Base.invokelatest(
        example.initialize_kernel, Tuple(initialize_inputs)...)
    initialize_source = string(
        sources.initialize, "\n\n",
        _port_call_source(
            "initialize_kernel", "nutpie_diagonal_initialize",
            example.INITIALIZE_INPUTS, example.INITIALIZE_OUTPUTS,
        ),
    )

    initial = example.initial_state(
        example.ORACLE_INPUTS.init_position,
        example.ORACLE_INPUTS.init_gradient,
    )
    first_step = example.ORACLE_INPUTS.steps[1]
    state = example.advance(
        initial, first_step.position, first_step.gradient;
        is_good = first_step.is_good,
        switch_now = first_step.switch_now,
        adapt_now = first_step.adapt_now,
    )
    step = example.ORACLE_INPUTS.steps[2]
    adaptation_values = (
        state.draw_mean, state.draw_variance,
        state.grad_mean, state.grad_variance, state.count,
        state.background_draw_mean, state.background_draw_variance,
        state.background_grad_mean, state.background_grad_variance,
        state.background_count, state.stds, state.inv_stds,
        state.transformation_mean, state.logdet, state.transformation_id,
        step.position, step.gradient,
        step.is_good, step.switch_now, step.adapt_now,
    )
    adaptation_inputs = NamedTuple{example.ADAPTATION_INPUTS}(adaptation_values)
    adaptation_output = Base.invokelatest(
        example.adaptation_kernel, Tuple(adaptation_inputs)...)
    adaptation_source = string(
        sources.adaptation, "\n\n",
        _port_call_source(
            "adaptation_kernel", "nutpie_diagonal_adaptation",
            example.ADAPTATION_INPUTS, example.ADAPTATION_OUTPUTS,
        ),
    )

    artifacts = (
        (;
            name = :nutpie_diagonal_initialize,
            origin = relpath(sources.path, pkgdir(ReactiveKernels)),
            source = initialize_source,
            inputs = initialize_inputs,
            kernel = example.initialize_kernel,
            output = initialize_output,
            generated = code_expr(example.initialize_kernel),
            dag = example.initialize_kernel.plan,
        ),
        (;
            name = :nutpie_diagonal_adaptation,
            origin = relpath(sources.path, pkgdir(ReactiveKernels)),
            source = adaptation_source,
            inputs = adaptation_inputs,
            kernel = example.adaptation_kernel,
            output = adaptation_output,
            generated = code_expr(example.adaptation_kernel),
            dag = example.adaptation_kernel.plan,
        ),
    )
    render_examples(artifacts)
end

# The ONE shared three-view UI (Raw input / Generated kernel / Compute DAG). Both
# the stateless PreparedKernel path and the ReactiveProgram path emit through this;
# there is no second renderer. `generated` is a display-only readable copy of the
# compiled AST; `dag` is the exact `Plan` consumed by `visualize`.
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
artifact supplies its exact raw source/call, the raw `code_expr` captured from
the executed `PreparedKernel`, and the selected `Plan` consumed by `visualize`.
The Generated kernel pane derives a readable, display-only expression from the
raw AST and that plan without changing either artifact.
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
        generated = _readable_generated_source(
            artifact.generated, artifact.kernel, artifact.name,
        )
        _three_pane_blocks!(blocks, _example_title(artifact.name), source, generated, artifact.dag)
    end
    Markdown.MD(blocks)
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
    rendered = render_examples((artifact,))
    _record_ppl_execution!(executed.name)
    rendered
end

# Pathfinder is an external compiler-acceptance artifact, so the documentation
# loads its reviewed benchmark fixture instead of maintaining a second copy of
# the mathematics.  The raw pane is extracted from that same file through the
# full prepare boundary; the generated panes and DAGs come from the actual
# PreparedKernels constructed during the docs build.
const _PATHFINDER_FIXTURE_PATH = normpath(joinpath(
    @__DIR__, "..", "benchmark", "pathfinder_kernel_authoring_fixture.jl",
))

const _PATHFINDER_HAVE = (
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

function _pathfinder_authored_source()
    source = replace(
        read(_PATHFINDER_FIXTURE_PATH, String),
        "\r\n" => "\n",
        "\r" => "\n",
    )
    start_marker = "@kernel pathfinder_candidate("
    stop_marker = "\n\"\"\"Run the same prepared candidate kernel"
    start = findfirst(start_marker, source)
    stop = findfirst(stop_marker, source)
    start === nothing && error("Pathfinder fixture lost its authored @kernel")
    stop === nothing && error("Pathfinder fixture lost its prepare boundary")
    first(start) < first(stop) || error("Pathfinder fixture source markers are reversed")
    strip(source[first(start):(first(stop) - 1)], '\n')
end

function _pathfinder_inputs(fixture::Module)
    # The fixture is included from this same Documenter `@eval` call. Cross the
    # resulting world-age boundary explicitly before reading its fresh methods.
    values = Base.invokelatest(fixture.pathfinder_fixture_inputs)
    (;
        logdensity = values.logdensity,
        position = values.positions[:, 2],
        gradient = values.gradients[:, 2],
        alpha = values.initial_alpha,
        step = values.positions[:, 2] .- values.positions[:, 1],
        gradient_delta = values.gradients[:, 1] .- values.gradients[:, 2],
        identity = values.identity,
        elbo_standard_draws = values.elbo_standard_draws[:, :, 1],
        output_standard_draws = values.output_standard_draws[:, :, 1],
        curvature_tolerance = values.curvature_tolerance,
    )
end

"""
    render_pathfinder_kernels(mod) -> Markdown.MD

Load the exact external Pathfinder fixture and render two compiled cuts of its
single authored mathematical graph: the curvature-safe inverse-BFGS geometry,
and the complete local-Gaussian/ELBO candidate.  Both panels execute during the
docs build and share the standard Raw input / Generated kernel / Compute DAG UI.
"""
function render_pathfinder_kernels(mod::Module)
    if !isdefined(mod, :PathfinderKernelAuthoringFixture)
        Base.include(mod, _PATHFINDER_FIXTURE_PATH)
    end
    fixture = getfield(mod, :PathfinderKernelAuthoringFixture)
    fixture isa Module || error("Pathfinder fixture did not define its module")

    inputs = _pathfinder_inputs(fixture)
    authored_source = _pathfinder_authored_source()
    geometry_kernel = Base.invokelatest(
        prepare,
        fixture.pathfinder_candidate;
        have = _PATHFINDER_HAVE,
        want = (:alpha_next, :curvature_accepted, :covariance),
    )
    candidate_kernel = fixture.PATHFINDER_CANDIDATE

    geometry_output = Base.invokelatest(geometry_kernel, Tuple(inputs)...)
    candidate_output = Base.invokelatest(candidate_kernel, Tuple(inputs)...)
    artifacts = (
        (;
            name = :pathfinder_inverse_bfgs_geometry,
            origin = "Algorithms 3/4 from the reviewed Pathfinder fixture; geometry WANT cut",
            inputs,
            kernel = geometry_kernel,
            output = geometry_output,
            source = authored_source,
            generated = code_expr(geometry_kernel),
            dag = geometry_kernel.plan,
        ),
        (;
            name = :pathfinder_local_gaussian_and_elbo,
            origin = "Algorithms 3/4 from the reviewed Pathfinder fixture; full candidate WANT cut",
            inputs,
            kernel = candidate_kernel,
            output = candidate_output,
            source = authored_source,
            generated = code_expr(candidate_kernel),
            dag = candidate_kernel.plan,
        ),
    )
    render_examples(artifacts)
end

# A docs-scoped stateless extraction of the Euclidean phasepoint recurrence from
# benchmark/nuts_kernel_authoring_fixture.jl. The complete eight-spec fixture
# remains byte-locked in nuts.md; this smaller executable authority exists so the
# selected Hamiltonian work can use the same PreparedKernel three-pane renderer
# as every other stateless example.
const NUTS_PHASEPOINT_SOURCE = raw"""
using LinearAlgebra

docs_nuts_potential(position) = sum(abs2, position) / 2
docs_nuts_potential_gradient(position) =
    (docs_nuts_potential(position), copy(position))

@kernel docs_nuts_phasepoint(position, momentum, metric) = begin
    potential = docs_nuts_potential(position)
    potential, potential_gradient = docs_nuts_potential_gradient(position)
    metric_cholesky = cholesky(metric)
    momentum_gradient = metric_cholesky \ momentum
    kinetic = oftype(potential, 0.5) *
        (@node(logdet(metric_cholesky)) + dot(momentum, momentum_gradient))
    hamiltonian = potential + kinetic
    hamiltonian_position_gradient = potential_gradient
    hamiltonian_momentum_gradient = momentum_gradient
end

inputs = (
    position = [0.25, -0.5],
    momentum = [0.4, 0.2],
    metric = Diagonal([1.0, 2.0]),
)
kernel = prepare(
    docs_nuts_phasepoint;
    have = (:position, :momentum, :metric),
    want = (
        :hamiltonian,
        :hamiltonian_position_gradient,
        :hamiltonian_momentum_gradient,
    ),
)
output = kernel(inputs.position, inputs.momentum, inputs.metric)

docs_example = (;
    name = :NUTS_phasepoint_Hamiltonian,
    origin = "Euclidean phasepoint recurrence from the sealed NUTS fixture",
    inputs,
    kernel,
    output,
)
"""

"""
    render_nuts_phasepoint(mod) -> Markdown.MD

Build and execute the representative stateless phasepoint Hamiltonian from the
sealed NUTS fixture, then render its selected plan through the standard shared
Raw input / Generated kernel / Compute DAG UI.
"""
render_nuts_phasepoint(mod::Module) = execute_example(mod, NUTS_PHASEPOINT_SOURCE)

# The FULL compiled NUTS kernel: reactive_nuts_group compiles the per-transition
# init/fwd/bwd Hamiltonian work into ONE flat ReactiveProgram whose plan is the
# 67-node Compute DAG. Unlike NUTS_PHASEPOINT_SOURCE (a stateless single-endpoint
# extraction), this is the reactive three-endpoint group program. A trivial
# analytic gradient and unit metric only instantiate the graph; its topology is
# independent of the potential, so no AD backend is needed for the build.
const NUTS_COMPILED_KERNEL_SOURCE = raw"""
using LinearAlgebra

# potential_gradient!(g, x) writes g in place and returns the SCALAR potential —
# the reactive_nuts_group contract (see examples/nuts.jl). A trivial analytic
# gradient suffices to build the graph; the compiled topology does not depend on it.
docs_nuts_potential_gradient!(gradient, position) =
    (gradient .= position; sum(abs2, position) / 2)

dimension = 4
# reactive_nuts_group is example-owned (loaded via ReactiveKernelsNUTSExample);
# it compiles init/fwd/bwd Hamiltonian + dham + diverged into one ReactiveProgram.
group = ReactiveKernels.reactive_nuts_group(
    docs_nuts_potential_gradient!,
    Matrix(1.0 * I, dimension, dimension),
    zeros(dimension),
    zeros(dimension),
)
program = reactive_program(group)

# Raw generated artifact: the fused reactive `:dham` (energy-error) getter — one
# representative compiled getter, NOT a whole-program listing. The renderer
# turns its positional operation slots into readable recipe expressions.
generated_dham_getter = code_expr(program, getproperty(group.handles, :dham))
# Compute DAG pane: the exact reactive_program(group).plan.
compiled_plan = program.plan
"""

"""
    render_nuts_compiled_kernel_dag(mod) -> Markdown.MD

Build the FULL compiled NUTS kernel — the single flat `ReactiveProgram` that
`reactive_nuts_group` compiles the per-transition Hamiltonian work into — and
render it through the standard shared Raw input / Generated kernel / Compute DAG
UI. The source is build-executed: the group and its `ReactiveProgram` are
constructed while the docs build runs, and the Compute DAG is that exact
`reactive_program(group).plan`. The Generated pane is the fused `:dham` getter, a
representative reactive getter rather than a whole-program listing.
"""
function render_nuts_compiled_kernel_dag(mod::Module)
    source = strip(NUTS_COMPILED_KERNEL_SOURCE, '\n')
    Core.eval(mod, :(using ReactiveKernels))
    _evaluate_source(mod, source)
    program = Core.eval(mod, :program)
    group = Core.eval(mod, :group)
    program isa ReactiveKernels.ReactiveProgram || error(
        "render_nuts_compiled_kernel_dag: source did not build a ReactiveProgram",
    )
    raw_generated = code_expr(program, getproperty(group.handles, :dham))
    generated = _readable_generated_source(
        raw_generated, program.plan, "render_nuts_compiled_kernel_dag",
    )
    blocks = Any[]
    _three_pane_blocks!(
        blocks, "Full compiled NUTS kernel — reactive group program",
        source, generated, program.plan)
    Markdown.MD(blocks)
end

"""
    execute_ppl_example(mod, owner, source; setup) -> Markdown.MD

Load a PPL example module into the page sandbox, resolve its exported source
authority only after setup, then execute those exact bytes through
[`execute_example`](@ref). Symbols keep the docs call independent of evaluation
order while making the owning example and source binding explicit.
"""
function execute_ppl_example(mod::Module, owner::Symbol, source::Symbol;
                             setup, result::Symbol = :docs_example)
    setup(mod)
    isdefined(mod, owner) || error("PPL docs setup did not define module $owner")
    owner_module = getfield(mod, owner)
    isdefined(owner_module, source) || error("$owner does not define source $source")
    code = getfield(owner_module, source)
    code isa AbstractString || error("$owner.$source is not source text")
    execute_example(mod, code; result, setup = nothing)
end

const EXPECTED_PPL_EXAMPLES = (
    :eight_schools_extraction,
    :linear_regression_density,
    :beta_binomial_density,
    :poisson_gamma_density,
    :dugongs_density,
    :arma11_density,
    :gaussian_mixture_density,
)
const _PPL_EXECUTION_COUNTS = Dict(name => 0 for name in EXPECTED_PPL_EXAMPLES)

function _record_ppl_execution!(name::Symbol)
    haskey(_PPL_EXECUTION_COUNTS, name) || return nothing
    _PPL_EXECUTION_COUNTS[name] += 1
    nothing
end

function assert_ppl_examples_executed!()
    failures = String[]
    for name in EXPECTED_PPL_EXAMPLES
        count = _PPL_EXECUTION_COUNTS[name]
        count == 1 || push!(failures, "$name executed $count times (expected exactly once)")
    end
    isempty(failures) || error(
        "PPL documentation execution gate failed:\n" * join(failures, "\n"),
    )
    nothing
end

# Benchmark and result rendering lives in result_views.jl.

# The NUTS `@kernel` authoring surface is embedded STATICALLY in docs/src/nuts.md as a plain ```julia
# fenced block (the exact bytes of benchmark/nuts_kernel_authoring_fixture.jl). It is intentionally NOT
# rendered by a build-time `@eval` here: it is a deliberately static publication
# surface, byte-locked LOUDLY to benchmark/nuts_kernel_authoring_fixture.jl by
# test/test_nuts_docs_fixture.jl. PPL walkthrough panels use the separate
# build-execution gate above, and `eval_block` errors are fatal in docs/make.jl.

# WALNUTS-D is a smaller external compiler fixture whose page emphasizes a few
# mathematical/control slices and still offers its complete source.  Read those
# bytes directly from the authoritative fixture: the page must never drift into
# hand-transcribed pseudocode.  docs/make.jl includes the fixture before loading
# the page, so this display also requires the real @kernel definitions to parse.
function _walnuts_fixture_source()
    isdefined(Main, :WalnutsKernelAuthoringFixture) || error(
        "WALNUTS-D authoring fixture was not loaded by docs/make.jl",
    )
    fixture = getfield(Main, :WalnutsKernelAuthoringFixture)
    irs = ReactiveKernels.method_irs(getfield(fixture, :walnuts_state))
    length(irs) == 15 && all(ir -> ir.ok, irs) || error(
        "WALNUTS-D docs source no longer has its admitted 15-method MethodIR",
    )
    path = joinpath(pkgdir(ReactiveKernels), "benchmark",
                    "walnuts_kernel_authoring_fixture.jl")
    isfile(path) || error("WALNUTS-D authoring fixture is missing: $path")
    replace(read(path, String), "\r\n" => "\n", "\r" => "\n")
end

function _walnuts_source_between(source, first_line, next_line; drop_last = false)
    lines = split(source, '\n'; keepempty = true)
    first_index = findfirst(line -> startswith(line, first_line), lines)
    first_index === nothing && error(
        "WALNUTS-D docs source start anchor vanished: $first_line",
    )
    next_index = findnext(line -> startswith(line, next_line), lines,
                          first_index + 1)
    next_index === nothing && error(
        "WALNUTS-D docs source end anchor vanished: $next_line",
    )
    selected = collect(lines[first_index:(next_index - 1)])
    while !isempty(selected) && isempty(last(selected))
        pop!(selected)
    end
    drop_last && !isempty(selected) && pop!(selected)
    rstrip(join(selected, "\n"))
end

function render_walnuts_source(section::Symbol)
    source = _walnuts_fixture_source()
    snippet = if section === :state
        _walnuts_source_between(
            source,
            "@kernel walnuts_state(init; step_f, macro_time,",
            "    finiteorneginf(x) = begin",
        )
    elseif section === :macro_step
        _walnuts_source_between(
            source,
            "    macro_step!(ep) = begin",
            "    next_direction!(stream) = begin",
        )
    elseif section === :nuts_step
        _walnuts_source_between(
            source,
            "    step!(directions, exponentials) = begin",
            "    flip!(depth) = if depth > 1",
        )
    elseif section === :nuts_leaf
        _walnuts_source_between(
            source,
            "    start!(ep, depth, exponentials) = if depth == 1",
            "# All stochastic inputs are explicit values.",
            drop_last = true,
        )
    elseif section === :entry
        _walnuts_source_between(
            source,
            "@kernel walnuts!!(state; momentum, directions, exponentials)",
            "end # module WalnutsKernelAuthoringFixture",
        )
    else
        error("unknown WALNUTS-D docs source section: $section")
    end
    Markdown.MD(Any[Markdown.Code("julia", snippet)])
end

render_walnuts_complete_source() =
    Markdown.MD(Any[Markdown.Code("julia", rstrip(_walnuts_fixture_source()))])

end # module ReactiveKernelsDocs
