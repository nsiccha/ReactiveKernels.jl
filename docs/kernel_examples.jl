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
    # Constrained parameters and predictions are plain NamedTuples, so there are
    # no custom types to import — only the model's data.
    Core.eval(mod, :(using .EightSchoolsExample:
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

const _DISTRIBUTION_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "distribution-logdensity-v1.toml",
)

function _benchmark_time(ns)
    value = Float64(ns)
    value < 1_000 && return string(round(value; digits = 1), " ns")
    value < 1_000_000 && return string(round(value / 1_000; digits = 2), " μs")
    string(round(value / 1_000_000; digits = 2), " ms")
end

function _benchmark_bytes(bytes)
    value = Int(bytes)
    value < 1_024 && return "$value B"
    value < 1_048_576 && return string(round(value / 1_024; digits = 1), " KiB")
    string(round(value / 1_048_576; digits = 2), " MiB")
end

_benchmark_time_cell(row, name) = haskey(row, name) ?
    _benchmark_time(row[name]["median_ns"]) : "unsupported"

function _benchmark_allocation_cell(row, name)
    haskey(row, name) || return "unsupported"
    measurement = row[name]
    string(_benchmark_bytes(measurement["median_bytes"]), " / ",
           Int(measurement["median_allocs"]), " alloc")
end

"""
    render_distribution_benchmarks() -> Markdown.MD

Render the checked-in Normal log-density benchmark receipt. The docs build
refuses a dirty/unpinned receipt or one on which the RK and ProbabilityMeasures
Reactant compatibility gates did not pass.
"""
function render_distribution_benchmarks()
    receipt = TOML.parsefile(_DISTRIBUTION_RECEIPT_PATH)
    get(receipt, "schema", "") == "distribution-logdensity-v1" || error(
        "unexpected distribution benchmark receipt schema",
    )
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false || error(
        "distribution benchmark receipt was produced from a dirty RK tree",
    )
    support = receipt["support"]
    get(support, "rk_reactant", false) || error(
        "distribution benchmark receipt does not accept the RK Reactant path",
    )
    get(support, "probability_measures_reactant", false) || error(
        "distribution benchmark receipt does not accept ProbabilityMeasures + Reactant",
    )
    protocol = receipt["protocol"]
    get(protocol, "reactant_sync", false) || error("Reactant receipt is not synchronous")
    get(protocol, "reactant_transfers_included", true) && error(
        "Reactant receipt includes host/device transfers",
    )

    timing_rows = Vector{Any}[
        Any[
            "N", "RK native", "Distributions native", "ProbabilityMeasures native",
            "RK + Reactant", "Distributions + Reactant", "ProbabilityMeasures + Reactant",
        ],
    ]
    allocation_rows = Vector{Any}[
        Any[
            "N", "RK native", "Distributions native", "ProbabilityMeasures native",
            "RK + Reactant", "ProbabilityMeasures + Reactant",
        ],
    ]
    for row in receipt["measurements"]
        push!(timing_rows, Any[
            string(Int(row["n"])),
            _benchmark_time_cell(row, "rk_native"),
            _benchmark_time_cell(row, "distributions_native"),
            _benchmark_time_cell(row, "probability_measures_native"),
            _benchmark_time_cell(row, "rk_reactant"),
            _benchmark_time_cell(row, "distributions_reactant"),
            _benchmark_time_cell(row, "probability_measures_reactant"),
        ])
        push!(allocation_rows, Any[
            string(Int(row["n"])),
            _benchmark_allocation_cell(row, "rk_native"),
            _benchmark_allocation_cell(row, "distributions_native"),
            _benchmark_allocation_cell(row, "probability_measures_native"),
            _benchmark_allocation_cell(row, "rk_reactant"),
            _benchmark_allocation_cell(row, "probability_measures_reactant"),
        ])
    end

    largest = last(receipt["measurements"])
    ratio(numerator, denominator) = round(
        largest[numerator]["median_ns"] / largest[denominator]["median_ns"];
        digits = 2,
    )
    crossover = first(
        row for row in receipt["measurements"]
        if row["rk_reactant"]["median_ns"] < row["rk_native"]["median_ns"]
    )
    largest_n = Int(largest["n"])
    crossover_n = Int(crossover["n"])
    sha = first(String(pins["reactivekernels_sha"]), 10)
    pm_sha = first(String(pins["probability_measures_sha"]), 10)
    distributions_version = pins["distributions_version"]
    reactant_version = pins["reactant_version"]
    julia_version = pins["julia_version"]
    cpu = receipt["environment"]["cpu"]
    distributions_ratio = ratio("distributions_native", "rk_native")
    probability_measures_ratio = ratio("probability_measures_native", "rk_native")
    reactant_ratio = ratio("rk_native", "rk_reactant")
    reactant_pm_ratio = ratio("probability_measures_reactant", "rk_reactant")

    summary = "At N=$largest_n, native RK is " *
              "$distributions_ratio× faster than Distributions and " *
              "$probability_measures_ratio× " *
              "faster than ProbabilityMeasures. RK + Reactant is " *
              "$reactant_ratio× faster than native RK and " *
              "$reactant_pm_ratio× faster " *
              "than ProbabilityMeasures + Reactant. In the sampled sizes, Reactant first " *
              "beats native RK at N=$crossover_n."
    provenance = "Receipt pins: RK `$sha`; ProbabilityMeasures `$pm_sha`; " *
                 "Distributions $distributions_version; Reactant $reactant_version; " *
                 "Julia $julia_version; $cpu."

    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        Markdown.Table(timing_rows, fill(:r, 7)),
        Markdown.Paragraph(Any[Markdown.Bold("Allocation receipts")]),
        Markdown.Table(allocation_rows, fill(:r, 6)),
        Markdown.Paragraph(Any[Markdown.Italic(provenance)]),
    ])
end

# The NUTS `@kernel` authoring surface is embedded STATICALLY in docs/src/nuts.md as a plain ```julia
# fenced block (the exact bytes of benchmark/nuts_kernel_authoring_fixture.jl). It is intentionally NOT
# rendered by a build-time `@eval` here: makedocs runs with `warnonly = true`, so a throwing render block
# would be swallowed and the kernel source would silently vanish from the page (this happened once). A
# static fence cannot fail to render. The byte-for-byte match between nuts.md and the fixture is enforced
# LOUDLY by test/test_nuts_docs_fixture.jl, not by build-time evaluation.

end # module ReactiveKernelsDocs
