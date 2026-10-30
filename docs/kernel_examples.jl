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

# Vitepress/Shiki amplifies a single long generated-code line into several MiB
# of highlighted HTML/JS. Preserve the executed artifact and its exact output in
# the build gate, but give readers a bounded generated preview rather than
# highlighting megabytes of inlined constants.
function _display_code(label::AbstractString, code::AbstractString;
                       max_bytes::Int = 64 * 1024)
    sizeof(code) <= max_bytes && return code
    lines = split(code, '\n'; keepempty = true)
    head_lines = first(lines, 24)
    tail_lines = last(lines, 12)
    while sizeof(join(vcat(head_lines, ["…"], tail_lines), '\n')) > max_bytes &&
          length(head_lines) > 4 && length(tail_lines) > 2
        if length(head_lines) > length(tail_lines)
            head_lines = head_lines[1:(end - 1)]
        else
            tail_lines = tail_lines[2:end]
        end
    end
    omitted_bytes = sizeof(code) -
        sizeof(join(vcat(head_lines, tail_lines), '\n'))
    return join(vcat(
        head_lines,
        [
            "# … display bounded for docs build memory …",
            "# $(length(lines)) lines total; about $(Base.format_bytes(sizeof(code))) total,",
            "# about $(Base.format_bytes(max(omitted_bytes, 0))) omitted here.",
        ],
        tail_lines,
    ), '\n')
end

function _generated_source(expr::Expr, context, label)
    expr == code_expr(context) || error("$label generated expression changed")
    sprint(print, Base.remove_linenums!(deepcopy(expr)))
end

# --- eval-throughput visualization ----------------------------------------
# A build-executed focused comparison of the static eval-throughput receipt:
# one relative-runtime plot and compact baseline table per operation.
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
    return _render_eval_throughput_focused(receipt)
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
    # Bind only the observations. The displayed PPL assembly imports and uses
    # the shared distribution objects directly; no helper evaluator, factor,
    # or separately prepared plate is injected.
    Core.eval(mod, :(using .EightSchoolsExample:
        EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA))
    nothing
end

function setup_sum_to_zero!(mod::Module)
    if !isdefined(mod, :SumToZeroExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: SumToZeroExample))
    end
    # The authored source reuses the Eight Schools observations while keeping
    # every new transform and reconstruction equation inline in its one model.
    Core.eval(mod, :(using ReactiveKernelsPPLExamples.EightSchoolsExample:
        EIGHT_SCHOOLS_Y, EIGHT_SCHOOLS_SIGMA))
    nothing
end

function setup_mnist_logistic!(mod::Module)
    if !isdefined(mod, :MNISTLogisticExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: MNISTLogisticExample))
    end
    # Bind only the data. The displayed PPL assembly imports and reuses the
    # shared `normal` and `categorical_logit` distribution objects directly.
    Core.eval(mod, :(using .MNISTLogisticExample:
        MNIST_LOGISTIC_X, MNIST_LOGISTIC_Y, NUM_CLASSES))
    nothing
end

function setup_linear_regression!(mod::Module)
    if !isdefined(mod, :LinearRegressionExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: LinearRegressionExample))
    end
    # Bind only the data. The displayed PPL assembly imports and reuses the
    # shared `normal` distribution object directly; no helper evaluator, factor,
    # or separately prepared density/plate is injected.
    Core.eval(mod, :(using .LinearRegressionExample:
        LINREG_X, LINREG_Y))
    nothing
end

function setup_beta_binomial!(mod::Module)
    if !isdefined(mod, :BetaBinomialExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: BetaBinomialExample))
    end
    # Bind only the data. The displayed PPL assembly imports and reuses the
    # shared `beta` and `binomial` distribution objects directly.
    Core.eval(mod, :(using .BetaBinomialExample:
        BETA_BINOMIAL_TRIALS, BETA_BINOMIAL_SUCCESSES))
    nothing
end

function setup_poisson_gamma!(mod::Module)
    if !isdefined(mod, :PoissonGammaExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: PoissonGammaExample))
    end
    # Bind only the data. The displayed PPL assembly imports and reuses the
    # shared `gamma` and `poisson` distribution objects directly.
    Core.eval(mod, :(using .PoissonGammaExample: POISSON_COUNTS))
    nothing
end

function setup_dugongs!(mod::Module)
    if !isdefined(mod, :DugongsGrowthExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: DugongsGrowthExample))
    end
    # Bind only the data. The displayed PPL assembly imports and reuses the
    # shared `normal`, `uniform`, and `gamma` distribution objects directly.
    Core.eval(mod, :(using .DugongsGrowthExample: DUGONGS_AGE, DUGONGS_LENGTH))
    nothing
end

function setup_arma11!(mod::Module)
    if !isdefined(mod, :ARMA11Example)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: ARMA11Example))
    end
    # Bind only the data. The displayed PPL assembly imports and reuses the
    # shared `normal` and `cauchy` distribution objects directly and authors the
    # error recursion inline; no helper evaluator is injected.
    Core.eval(mod, :(using .ARMA11Example: ARMA_SERIES))
    nothing
end

function setup_covid19imperial!(mod::Module)
    if !isdefined(mod, :Covid19ImperialExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: Covid19ImperialExample))
    end
    # Bind only the RAW posteriordb arrays; the displayed PPL assembly derives
    # every model-specific preprocessing step (covariate reshape, observed-grid
    # mask, deaths grid, count log-factorial) as named in-graph nodes and reuses
    # the shared `normal`, `gamma`, and `exponential` endpoints directly.
    Core.eval(mod, :(using .Covid19ImperialExample: COVID19IMPERIAL_X,
        COVID19IMPERIAL_EPIDEMICSTART, COVID19IMPERIAL_N, COVID19IMPERIAL_DEATHS,
        COVID19IMPERIAL_SI, COVID19IMPERIAL_F, COVID19IMPERIAL_POP,
        COVID19IMPERIAL_M, COVID19IMPERIAL_P, COVID19IMPERIAL_N0,
        COVID19IMPERIAL_N2))
    nothing
end

function setup_garch11!(mod::Module)
    if !isdefined(mod, :GARCH11Example)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: GARCH11Example))
    end
    Core.eval(mod, :(using .GARCH11Example: GARCH11_Y, GARCH11_SIGMA1))
    nothing
end

function setup_hmm_example!(mod::Module)
    if !isdefined(mod, :HmmExampleExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: HmmExampleExample))
    end
    Core.eval(mod, :(using .HmmExampleExample: HMM_EXAMPLE_Y, HMM_EXAMPLE_K))
    nothing
end

function setup_hmm_gaussian!(mod::Module)
    if !isdefined(mod, :HmmGaussianExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: HmmGaussianExample))
    end
    Core.eval(mod, :(using .HmmGaussianExample: HMM_GAUSSIAN_Y, HMM_GAUSSIAN_K))
    nothing
end

function setup_iohmm_reg!(mod::Module)
    if !isdefined(mod, :IohmmRegExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: IohmmRegExample))
    end
    Core.eval(mod, :(using .IohmmRegExample: IOHMM_REG_Y, IOHMM_REG_U, IOHMM_REG_K))
    nothing
end

function setup_mvnormal_regression!(mod::Module)
    if !isdefined(mod, :MVNormalRegressionExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: MVNormalRegressionExample))
    end
    # Bind only the data referenced by the displayed covariance-path query. The
    # displayed PPL assembly imports and reuses the shared `normal` and
    # `mvnormal` distribution objects directly.
    Core.eval(mod, :(using .MVNormalRegressionExample:
        MVREG_X, MVREG_Y, MVREG_COVARIANCE))
    nothing
end

function setup_bound_regression!(mod::Module)
    if !isdefined(mod, :BoundRegressionExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: BoundRegressionExample))
    end
    # Bind only the data. The displayed PPL assembly imports and reuses the
    # shared `normal` distribution object directly and authors the
    # standardization prefix inline.
    Core.eval(mod, :(using .BoundRegressionExample: BOUND_RAW_X, BOUND_Y))
    nothing
end

function setup_diamonds!(mod::Module)
    if !isdefined(mod, :DiamondsExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: DiamondsExample))
    end
    # Bind only the data. The displayed PPL assembly imports the shared `normal`
    # and `student_t` endpoints itself and authors the brms centered-design
    # prefix inline; binding the raw `X` port hoists it.
    Core.eval(mod, :(using .DiamondsExample: DIAMONDS_X, DIAMONDS_Y, DIAMONDS_PRIOR_ONLY))
    nothing
end

function setup_normal_mixture_k!(mod::Module)
    if !isdefined(mod, :NormalMixtureKExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: NormalMixtureKExample))
    end
    # Bind only the data and the component count K. The displayed PPL assembly
    # imports the shared `normal` endpoint itself and authors the inverse-ILR
    # simplex transform and the marginalized K-way mixture likelihood inline.
    Core.eval(mod, :(using .NormalMixtureKExample: NORMAL_MIXTURE_K_Y, NORMAL_MIXTURE_K_K))
    nothing
end

function setup_hmm_drive_1!(mod::Module)
    if !isdefined(mod, :HmmDrive1Example)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: HmmDrive1Example))
    end
    # Bind only the raw data. The displayed PPL assembly imports and reuses the
    # shared `normal` and `dirichlet` distribution objects directly and authors
    # the forward-algorithm `scan` inline; no helper evaluator is injected.
    Core.eval(mod, :(using .HmmDrive1Example:
        HMM_DRIVE_1_U, HMM_DRIVE_1_V, HMM_DRIVE_1_ALPHA, HMM_DRIVE_1_TAU,
        HMM_DRIVE_1_RHO))
    nothing
end

function setup_dogs_nonhierarchical!(mod::Module)
    if !isdefined(mod, :DogsNonhierarchicalExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: DogsNonhierarchicalExample))
    end
    # Bind only the raw y matrix. The displayed PPL assembly imports the shared
    # `normal` and `bernoulli` endpoints itself and derives the running-count
    # design (the strict-upper-triangular operator, prev_shock = y·C, …) in-graph.
    Core.eval(mod, :(using .DogsNonhierarchicalExample: DOGS_NH_Y))
    nothing
end

function setup_logistic_regression_rhs!(mod::Module)
    if !isdefined(mod, :LogisticRegressionRHSExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: LogisticRegressionRHSExample))
    end
    # Bind the data and the fixed horseshoe hyper-scalars. The displayed PPL
    # assembly imports the shared endpoints itself and authors the regularized
    # horseshoe (lambda_tilde, the exp support transforms) inline.
    Core.eval(mod, :(using .LogisticRegressionRHSExample:
        LOGISTIC_RHS_X, LOGISTIC_RHS_Y, LOGISTIC_RHS_HYPER))
    nothing
end

function setup_lda!(mod::Module)
    if !isdefined(mod, :LDAExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: LDAExample))
    end
    # Bind only the raw data (the rendered default is three_men1-ldaK2, whose
    # transformed-data ones-vector priors are LDA_ALPHA/LDA_BETA). The displayed
    # PPL assembly imports the shared `dirichlet` endpoint itself and authors the
    # in-graph inverse-ILR simplex transform and the marginalized likelihood inline.
    Core.eval(mod, :(using .LDAExample: LDA_DOC, LDA_W, LDA_ALPHA, LDA_BETA, LDA_M))
    nothing
end

function setup_nn_rbm!(mod::Module)
    if !isdefined(mod, :NNRBMExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: NNRBMExample))
    end
    # Bind only the data and the fixed hidden-unit count (the rendered default is
    # mnist_100-nn_rbm1bJ10). The displayed PPL assembly imports the shared
    # `normal`/`inverse_gamma`/`categorical_logit` endpoints itself and authors the
    # tanh hidden layer and reference-coded softmax inline.
    Core.eval(mod, :(using .NNRBMExample: NN_RBM_X, NN_RBM_Y, NN_RBM_K, NN_RBM_J))
    nothing
end

function setup_gp_regr!(mod::Module)
    if !isdefined(mod, :GPRegrExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: GPRegrExample))
    end
    # Bind only the real data the displayed source references; the PPL assembly
    # imports and reuses the shared `normal` and `gamma` endpoints directly.
    Core.eval(mod, :(using .GPRegrExample: GP_REGR_X, GP_REGR_Y))
    nothing
end

function setup_gp_pois_regr!(mod::Module)
    if !isdefined(mod, :GPPoisRegrExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: GPPoisRegrExample))
    end
    Core.eval(mod, :(using .GPPoisRegrExample: GP_POIS_X, GP_POIS_K))
    nothing
end

function setup_accel_gp!(mod::Module)
    if !isdefined(mod, :AccelGPExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: AccelGPExample))
    end
    Core.eval(mod, :(using .AccelGPExample:
        ACCEL_GP_Y, ACCEL_GP_XGP, ACCEL_GP_SLAMBDA,
        ACCEL_GP_XGP_SIGMA, ACCEL_GP_SLAMBDA_SIGMA, ACCEL_GP_PRIOR_ONLY))
    nothing
end

function setup_hierarchical_gp!(mod::Module)
    if !isdefined(mod, :HierarchicalGPExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: HierarchicalGPExample))
    end
    Core.eval(mod, :(using .HierarchicalGPExample:
        HGP_Y, HGP_YEAR_IND, HGP_STATE_IND, HGP_REGION_IND, HGP_STATE_REGION_IND,
        HGP_N_YEARS, HGP_N_REGIONS, HGP_N_STATES, HGP_N_YEARS_OBS))
    nothing
end

function setup_losscurve_sislob!(mod::Module)
    if !isdefined(mod, :LosscurveSislobExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: LosscurveSislobExample))
    end
    # Bind only the data. The displayed PPL assembly imports the shared `normal`
    # and `lognormal` endpoints itself and authors the flag-selected growth
    # factor and per-datum mean in-graph.
    Core.eval(mod, :(using .LosscurveSislobExample:
        LOSSCURVE_GROWTHMODEL_ID, LOSSCURVE_COHORT_ID, LOSSCURVE_T_IDX,
        LOSSCURVE_T_VALUE, LOSSCURVE_PREMIUM, LOSSCURVE_LOSS))
    nothing
end

function setup_accel_splines!(mod::Module)
    if !isdefined(mod, :AccelSplinesExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: AccelSplinesExample))
    end
    # Bind only the response and the four design/basis matrices plus the
    # prior_only flag; the displayed PPL assembly imports the shared `normal`
    # and `student_t` endpoints itself and authors the linear predictors in-graph.
    Core.eval(mod, :(using .AccelSplinesExample:
        ACCEL_Y, ACCEL_XS, ACCEL_ZS_1_1, ACCEL_XS_SIGMA, ACCEL_ZS_SIGMA_1_1,
        ACCEL_PRIOR_ONLY))
    nothing
end

function setup_state_space_stochastic!(mod::Module)
    if !isdefined(mod, :StateSpaceStochasticExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: StateSpaceStochasticExample))
    end
    # Bind only the raw series y, x, w; the displayed PPL assembly imports the
    # shared `normal` and `student_t` endpoints itself and derives the bounded
    # level transform, positive_ordered scales and seasonal window sum in-graph.
    Core.eval(mod, :(using .StateSpaceStochasticExample:
        STATE_SPACE_Y, STATE_SPACE_X, STATE_SPACE_W))
    nothing
end

function setup_prophet!(mod::Module)
    if !isdefined(mod, :ProphetExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: ProphetExample))
    end
    # Bind the raw time/changepoint/design data for the LINEAR trend; the
    # displayed PPL assembly imports the shared `normal` and `laplace` endpoints
    # itself and derives the changepoint incidence matrix and trend in-graph.
    Core.eval(mod, :(using .ProphetExample:
        PROPHET_T, PROPHET_T_CHANGE, PROPHET_X, PROPHET_SIGMAS, PROPHET_TAU,
        PROPHET_S_A, PROPHET_S_M, PROPHET_Y))
    nothing
end

function setup_irt_2pl!(mod::Module)
    if !isdefined(mod, :Irt2plExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: Irt2plExample))
    end
    # Bind only the raw I×J response matrix. The displayed PPL assembly imports
    # the shared endpoints itself and forms the per-cell linear predictor as an
    # in-graph broadcast over the item/person axes (no external index).
    Core.eval(mod, :(using .Irt2plExample: IRT_2PL_Y))
    nothing
end

function setup_2pl_latent_reg_irt!(mod::Module)
    if !isdefined(mod, :TwoplLatentRegIrtExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: TwoplLatentRegIrtExample))
    end
    # Bind the raw long-form ii/jj/y, the covariate matrix W, and the item count.
    # The displayed PPL assembly derives the covariate design (obtain_adjustments)
    # and the sum-to-zero difficulty map in-graph.
    Core.eval(mod, :(using .TwoplLatentRegIrtExample:
        TWOPL_LR_II, TWOPL_LR_JJ, TWOPL_LR_Y, TWOPL_LR_W, TWOPL_LR_I))
    nothing
end

function setup_hier_2pl!(mod::Module)
    if !isdefined(mod, :Hier2plExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: Hier2plExample))
    end
    # Bind the raw long-form ii/jj/y and the item/person counts. The displayed
    # PPL assembly forms the cholesky_factor_corr transform, the analytic LKJ(4)
    # density, and the fused multi_normal_cholesky item prior in-graph.
    Core.eval(mod, :(using .Hier2plExample:
        HIER_2PL_II, HIER_2PL_JJ, HIER_2PL_Y, HIER_2PL_I, HIER_2PL_J))
    nothing
end

function setup_gpcm_latent_reg_irt!(mod::Module)
    if !isdefined(mod, :GpcmLatentRegIrtExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: GpcmLatentRegIrtExample))
    end
    # Bind the raw long-form ii/jj/y (ordinal), the covariate matrix W, and the
    # item count. The displayed PPL assembly derives the ragged per-item category
    # structure, the covariate design, and the sum-to-zero map in-graph.
    Core.eval(mod, :(using .GpcmLatentRegIrtExample:
        GPCM_LR_II, GPCM_LR_JJ, GPCM_LR_Y, GPCM_LR_W, GPCM_LR_I))
    nothing
end

function setup_grsm_latent_reg_irt!(mod::Module)
    if !isdefined(mod, :GrsmLatentRegIrtExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: GrsmLatentRegIrtExample))
    end
    # Bind the raw long-form ii/jj/y (ordinal), the covariate matrix W, and the
    # item count. The displayed PPL assembly derives the category count, the
    # covariate design, and both sum-to-zero maps in-graph.
    Core.eval(mod, :(using .GrsmLatentRegIrtExample:
        GRSM_LR_II, GRSM_LR_JJ, GRSM_LR_Y, GRSM_LR_W, GRSM_LR_I))
    nothing
end

function setup_kronecker_gp!(mod::Module)
    if !isdefined(mod, :KroneckerGpExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: KroneckerGpExample))
    end
    # Bind the grid locations x1 and the observation matrix y; the squared-
    # distance matrix, both margin eigendecompositions, and the LKJ-Cholesky
    # transform are derived in-graph.
    Core.eval(mod, :(using .KroneckerGpExample: KRON_X1, KRON_Y))
    nothing
end

function setup_glmm1!(mod::Module)
    if !isdefined(mod, :GLMM1ModelExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: GLMM1ModelExample))
    end
    # Bind only the raw counts + site indices; the displayed PPL assembly imports
    # the shared normal/poisson endpoints itself and gathers alpha[obssite] in-graph.
    Core.eval(mod, :(using .GLMM1ModelExample: GLMM1_OBS, GLMM1_OBSSITE, GLMM1_NSITE))
    nothing
end

function setup_bym2_offset_only!(mod::Module)
    if !isdefined(mod, :Bym2OffsetOnlyExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: Bym2OffsetOnlyExample))
    end
    # Bind only the raw adjacency/counts/exposure/scaling; log_E, convolved_re and
    # the ICAR phi[node1]/phi[node2] gathers are derived in-graph.
    Core.eval(mod, :(using .Bym2OffsetOnlyExample:
        BYM2_NODE1, BYM2_NODE2, BYM2_Y, BYM2_E, BYM2_SCALING_FACTOR))
    nothing
end

function setup_bones!(mod::Module)
    if !isdefined(mod, :BonesModelExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: BonesModelExample))
    end
    # Bind only the raw grade/gamma/delta/ncat block; grid coordinates and the
    # ragged cut selection are all derived in-graph.
    Core.eval(mod, :(using .BonesModelExample: BONES_GRADE, BONES_GAMMA, BONES_DELTA, BONES_NCAT))
    nothing
end

function setup_multi_occupancy!(mod::Module)
    if !isdefined(mod, :MultiOccupancyExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: MultiOccupancyExample))
    end
    # Bind only the raw n×J detection matrix + dims; the flat counts, species
    # coordinate and binomial normalizer are all derived in-graph.
    Core.eval(mod, :(using .MultiOccupancyExample:
        MULTI_OCC_X, MULTI_OCC_N, MULTI_OCC_J, MULTI_OCC_K, MULTI_OCC_S))
    nothing
end

function setup_lotka_volterra!(mod::Module)
    if !isdefined(mod, :LotkaVolterraExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: LotkaVolterraExample))
    end
    Core.eval(mod, :(using .LotkaVolterraExample:
        LOTKA_TS, LOTKA_Y_INIT, LOTKA_Y))
    nothing
end

function setup_sir!(mod::Module)
    if !isdefined(mod, :SIRExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: SIRExample))
    end
    Core.eval(mod, :(using .SIRExample:
        SIR_T, SIR_Y0, SIR_STOI_HAT, SIR_B_HAT))
    nothing
end

function setup_soil_incubation!(mod::Module)
    if !isdefined(mod, :SoilIncubationExample)
        Core.eval(mod, :(using ReactiveKernelsPPLExamples: SoilIncubationExample))
    end
    Core.eval(mod, :(using .SoilIncubationExample:
        SOIL_T0, SOIL_TOTAL_C_T0, SOIL_TS, SOIL_ECO2MEAN))
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

_html_escape(value) = replace(
    string(value), '&' => "&amp;", '<' => "&lt;", '>' => "&gt;",
    '"' => "&quot;",
)

# The ONE shared three-view UI (Raw input / Generated kernel / Compute DAG). Both
# the stateless PreparedKernel path and the ReactiveProgram path emit through this;
# there is no second renderer. `generated` is the compiled AST itself, the code
# that runs (`code_expr`); `dag` is the exact `Plan` consumed by `visualize`.
function _three_pane_blocks!(blocks, artifact_id, title, source, generated, dag::Plan)
    stable_id = _html_escape("example:" * string(artifact_id))
    push!(blocks, RawHTML("""
<h2>$(title)</h2>
<div class="rk-example" data-rk-example data-rk-artifact-id="$stable_id"
     data-rk-artifact-kind="example-panel">
<div data-rk-pane="source">
"""))
    push!(blocks, Markdown.Code("julia", _display_code("source", source)))
    push!(blocks, RawHTML("""
</div>
<div data-rk-pane="kernel">
"""))
    push!(blocks, Markdown.Code("julia", _display_code("generated", generated)))
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
The Generated kernel pane prints that expression as it is, with only its
line-number nodes removed: the generated code is the readable code.
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
        generated = _generated_source(
            artifact.generated, artifact.kernel, artifact.name,
        )
        _three_pane_blocks!(
            blocks, artifact.name, _example_title(artifact.name),
            source, generated, artifact.dag,
        )
    end
    Markdown.MD(blocks)
end

function _manual_derivative_rule_sources()
    path = joinpath(
        pkgdir(ReactiveKernels), "examples", "manual_derivative_rule.jl",
    )
    source = replace(read(path, String), "\r\n" => "\n", "\r" => "\n")
    graph = _source_between(
        source,
        "@kernel matvec_rule(",
        "# -- END DOCS: pure mathematical derivative rule --",
    )
    rule = _source_between(
        source,
        "const matvec = derivative_rule(",
        "# -- END DOCS: generated rule --",
    )
    (; path, graph, rule)
end

"""
    render_manual_derivative_rule_cuts() -> Markdown.MD

Render and execute the primal, forward, and reverse HAVE→WANT cuts of the one
source-owned manual derivative graph. Each cut uses the standard raw source,
generated kernel, and compute-DAG documentation surface.
"""
function render_manual_derivative_rule_cuts()
    example = Main.ManualDerivativeRuleExample
    sources = _manual_derivative_rule_sources()
    inputs = example.EXAMPLE_INPUTS
    cuts = (
        (;
            name = :manual_rule_primal,
            kernel = example.matvec_primal,
            have = example.PRIMAL_HAVE,
            want = example.PRIMAL_WANT,
        ),
        (;
            name = :manual_rule_forward,
            kernel = example.matvec_forward,
            have = example.FORWARD_HAVE,
            want = example.FORWARD_WANT,
        ),
        (;
            name = :manual_rule_reverse,
            kernel = example.matvec_reverse,
            have = example.REVERSE_HAVE,
            want = example.REVERSE_WANT,
        ),
    )
    origin = relpath(sources.path, pkgdir(ReactiveKernels))
    artifacts = map(cuts) do cut
        selected_inputs = NamedTuple{cut.have}(
            Tuple(getproperty(inputs, name) for name in cut.have),
        )
        output = Base.invokelatest(cut.kernel, Tuple(selected_inputs)...)
        (;
            name = cut.name,
            origin,
            source = string(
                sources.graph, "\n\n",
                _port_call_source(
                    "selected", "matvec_rule", cut.have, cut.want,
                ),
            ),
            inputs = selected_inputs,
            kernel = cut.kernel,
            output,
            generated = code_expr(cut.kernel),
            dag = cut.kernel.plan,
        )
    end
    render_examples(artifacts)
end

"""Render the example's exact rule graph and its `derivative_rule` declaration."""
function render_manual_derivative_rule_source()
    sources = _manual_derivative_rule_sources()
    Markdown.MD(Any[Markdown.Code(
        "julia", string(sources.graph, "\n\n", sources.rule),
    )])
end

"""
    execute_example(mod, code; result=:docs_example) -> Markdown.MD

Evaluate the exact displayed Julia source and render the resulting prepared
kernel through the standard three-view UI. The source must bind `result` to a
named tuple with `name`, `origin`, `inputs`, `kernel`, and `output` fields.
"""
function execute_example(mod::Module, code::AbstractString;
                         result::Symbol = :docs_example, setup = nothing,
                         gate::Bool = false)
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
    gate && _record_ppl_execution!(executed.name)
    rendered
end

"""
    render_inert_source(source) -> Markdown.MD

Render `source` as a static, non-executed Julia code block. Used for a
reverse-mode AD example whose executable authority is the package test suite, so
the docs build displays it as source and carries no Enzyme/LLVM autodiff
toolchain (user decision `00jueci`).
"""
render_inert_source(source::AbstractString) =
    Markdown.MD(Markdown.Code("julia", strip(source, '\n')))

# The public Normal observation example is the same source replayed in native
# acceptance. All three panels come from the kernel executed in this sandbox.
function render_nested_plate_example(mod::Module)
    fixture = joinpath(pkgdir(ReactiveKernels), "test", "fixtures", "nested_plates.jl")
    isdefined(mod, :NestedPlates) || Base.include(mod, fixture)
    execute_example(mod, getfield(mod, :NestedPlates).DOCS_SOURCE)
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
    execute_example(mod, code; result, setup = nothing, gate = true)
end

const EXPECTED_PPL_EXAMPLES = (
    :eight_schools_extraction,
    :sum_to_zero_logdensity,
    :linear_regression_density,
    :beta_binomial_density,
    :poisson_gamma_density,
    :dugongs_density,
    :arma11_density,
    :garch11_density,
    :hmm_example_density,
    :hmm_gaussian_density,
    :iohmm_reg_density,
    :mnist_logistic_density,
    :mnist_logistic_optimized_density,
    :mvnormal_regression_density,
    :bound_regression_density,
    :diamonds_posterior,
    :normal_mixture_k_posterior,
    :dogs_nonhierarchical_posterior,
    :logistic_regression_rhs_posterior,
    :covid19imperial_density,
    :lda_density,
    :nn_rbm_density,
    :gp_regr_posterior,
    :gp_pois_regr_posterior,
    :accel_gp_posterior,
    :hierarchical_gp_posterior,
    :losscurve_sislob_posterior,
    :accel_splines_posterior,
    :state_space_stochastic_posterior,
    :prophet_posterior,
    :irt_2pl_posterior,
    :two_pl_latent_reg_irt_posterior,
    :hier_2pl_posterior,
    :gpcm_latent_reg_irt_posterior,
    :grsm_latent_reg_irt_posterior,
    :kronecker_gp_posterior,
    :glmm1_model_posterior,
    :bym2_offset_only_posterior,
    :bones_model_posterior,
    :multi_occupancy_posterior,
    :lotka_volterra_posterior,
    :sir_posterior,
    :soil_incubation_posterior,
    :hmm_drive_1_density,
)
const _PPL_EXECUTION_COUNTS = Dict(name => 0 for name in EXPECTED_PPL_EXAMPLES)

function _record_ppl_execution!(name::Symbol)
    # execute_example is shared by PPL and non-PPL executable docs; only PPL
    # names participate in the exact-once posterior-walkthrough gate.
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

end # module ReactiveKernelsDocs
