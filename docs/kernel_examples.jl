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

# ---- source-synced `@kernel` NUTS authoring surface (source-only, NOT executed) --------------------
# The current production NUTS authoring surface is authored as eight method-bearing `@kernel`
# specifications. Their stateful lowering (inline update methods + `__self__`) is mid-implementation on
# the syntax/poc/hmc lanes and is NOT evaluable by today's function-shaped `@kernel`, so this page renders
# the EXACT source text drift-proof at build time — it is NOT build-executed, and no generated-kernel /
# Compute-DAG pane, parity, allocation, or performance claim is made. The file is byte-synced from the
# canonical fixture commit below; if it drifts or is truncated the structural gate fails the build.
const _AUTHORING_FIXTURE_PATH =
    joinpath(dirname(@__DIR__), "benchmark", "nuts_kernel_authoring_fixture.jl")

# Canonical origin of the synced source (benchmark/nuts_kernel_authoring_fixture.jl). Labelled on the page
# so a reader can pin the exact reviewed surface; the docs copy is byte-identical to this commit's blob.
const _AUTHORING_FIXTURE_ORIGIN_SHA = "ccb35d3"

# The eight method-bearing `@kernel` specifications the production surface must expose.
const _AUTHORING_FIXTURE_KERNELS = (
    "euclidean_phasepoint", "leapfrog!", "refresh_momentum!!", "nuts_stats!",
    "nuts_state", "nuts!!", "dual_averaging_state", "welford_var",
)

# The seven public `@rk_*` effect declarations the compiler schedules the helpers by. Their presence is
# part of the surface contract; a kernel-name check alone is too weak.
const _AUTHORING_FIXTURE_RK_DECLS = (
    "@rk_pure finiteorneginf", "@rk_pure min1exp", "@rk_borrows badd",
    "@rk_rng randbernoullilog", "@rk_pure logswapprob", "@rk_pure compute_criterion",
    "@rk_pure smooth",
)

# Call/type spellings that MUST NOT appear in the executable authoring. `@reactive` must be gone (the
# `@kernel` surface is the only public macro); no `Ref(`/`RefValue(`/`Ref{` may be constructed (the state
# is implicit-field, no-Ref). The fixture's PROSE comments mention "Ref"/"RefValue"/"no-Ref", so the gate
# matches the code spellings (trailing `(` / `{`) rather than the bare word, which would false-positive.
const _AUTHORING_FIXTURE_FORBIDDEN = ("@reactive", "Ref(", "RefValue(", "Ref{")

"""
    render_authoring_fixture() -> Markdown.MD

Render the source-synced eight-`@kernel` NUTS authoring surface as a plain Julia code block, read
drift-proof from `benchmark/nuts_kernel_authoring_fixture.jl` at build time. It is NOT executed: no
generated-kernel / Compute-DAG pane is produced, and no parity / allocation / performance / production
claim is made. This is the exact reviewed *source* (compiler lowering in progress), byte-synced from the
canonical fixture commit `$(_AUTHORING_FIXTURE_ORIGIN_SHA)`.

The render is refused (fails the docs build) unless the synced source still exposes all eight
`@kernel`s, all seven `@rk_*` effect declarations, and contains no `@reactive` and no
`Ref(`/`RefValue(`/`Ref{` in its executable authoring — a stronger gate than name occurrence alone.
"""
function render_authoring_fixture()
    src = read(_AUTHORING_FIXTURE_PATH, String)
    for kernel in _AUTHORING_FIXTURE_KERNELS
        occursin("@kernel $(kernel)(", src) || error(
            "authoring fixture drift: `@kernel $(kernel)(...)` missing from " *
            "$(_AUTHORING_FIXTURE_PATH); the synced source is stale or truncated.",
        )
    end
    for decl in _AUTHORING_FIXTURE_RK_DECLS
        occursin(decl, src) || error(
            "authoring fixture drift: effect declaration `$(decl)` missing from " *
            "$(_AUTHORING_FIXTURE_PATH); the seven `@rk_*` declarations are part of the surface contract.",
        )
    end
    for forbidden in _AUTHORING_FIXTURE_FORBIDDEN
        occursin(forbidden, src) && error(
            "authoring fixture regression: forbidden token `$(forbidden)` appears in " *
            "$(_AUTHORING_FIXTURE_PATH); `@kernel` is the only public macro and the authoring is no-Ref.",
        )
    end
    Markdown.MD(Any[Markdown.Code("julia", rstrip(src))])
end

end # module ReactiveKernelsDocs
