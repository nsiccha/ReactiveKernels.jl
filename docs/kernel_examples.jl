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
        EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA,
        split_unconstrained, positive_scale, assemble_parameters,
        log_abs_det_jacobian, log_prior, plated_loglik,
        total_log_density, predict_new_group))
    nothing
end

function setup_linear_regression!(mod::Module)
    if !isdefined(mod, :LinearRegressionExample)
        Base.include(mod, joinpath(@__DIR__, "..", "examples", "linear_regression.jl"))
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
        Base.include(mod, joinpath(@__DIR__, "..", "examples", "beta_binomial.jl"))
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
        Base.include(mod, joinpath(@__DIR__, "..", "examples", "poisson_gamma.jl"))
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
        Base.include(mod, joinpath(@__DIR__, "..", "examples", "dugongs_growth.jl"))
    end
    Core.eval(mod, :(using .DugongsGrowthExample:
        DugongsParameters, UnconstrainedParameters, RealVector,
        DUGONGS_AGE, DUGONGS_LENGTH,
        split_unconstrained, bounded_lambda, sd_from_log_precision,
        assemble_parameters, log_abs_det_jacobian, log_prior,
        pointwise_log_likelihood, sum_log_likelihood,
        total_log_density, predicted_length))
    nothing
end

function setup_arma11!(mod::Module)
    if !isdefined(mod, :ARMA11Example)
        Base.include(mod, joinpath(@__DIR__, "..", "examples", "arma11.jl"))
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
        Base.include(mod, joinpath(@__DIR__, "..", "examples", "gaussian_mixture.jl"))
    end
    Core.eval(mod, :(using .GaussianMixtureExample:
        MixtureParameters, UnconstrainedParameters, RealVector,
        MIXTURE_OBSERVATIONS, split_unconstrained, ordered_means,
        exp_scale, logistic, assemble_parameters, log_abs_det_jacobian,
        log_prior, pointwise_log_likelihood, sum_log_likelihood,
        total_log_density, component1_responsibility))
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
    :eight_schools_density,
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

end # module ReactiveKernelsDocs
