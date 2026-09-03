using AlgebraOfVega: BarPlot, Scatter, ScatterLines, config, data, mapping, scales,
    sorter, symlog, vdraw, visual
using Base64
using HTMX: h
using HTMXObjects: sortable_table, sortable_table_js
using Markdown
using ReactiveKernels
using TOML

const _DISTRIBUTION_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "distribution-logdensity-v1.toml",
)
const _STRUCTURED_DISTRIBUTION_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts",
    "structured-distribution-logdensity-v1.toml",
)
const _SCALAR_GALLERY_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts",
    "scalar-distribution-gallery-v1.toml",
)
const _DISTRIBUTION_GRADIENT_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "distribution-gradient-v1.toml",
)
const _EIGHT_SCHOOLS_PRIMAL_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "eight-schools-primal-v2.toml",
)
const _EIGHT_SCHOOLS_AD_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "eight-schools-ad-v2.toml",
)
const _SUM_TO_ZERO_NATIVE_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "sum-to-zero-native-v1.toml",
)
const _SUM_TO_ZERO_REACTANT_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "sum-to-zero-reactant-v1.toml",
)
const _MNIST_LOGISTIC_AD_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "mnist-logistic-ad-v2.toml",
)
const _MNIST_LOGISTIC_AD_WREN_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts",
    "mnist-logistic-ad-wren-pca40-v1.toml",
)
const _NUTS_G7_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "nuts-g7-v1.toml",
)
const _NUTS_REACTANT_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "nuts-reactant-v1.toml",
)
const _EIGHT_SCHOOLS_REACTANT_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "eight-schools-reactant-v2.toml",
)
const _EIGHT_SCHOOLS_REACTANT_AD_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts",
    "eight-schools-reactant-ad-v2.toml",
)

_node_html(node) = sprint(show, MIME"text/html"(), node; context = :limit => false)

_html_attribute(value) = replace(
    string(value), '&' => "&amp;", '"' => "&quot;", '<' => "&lt;", '>' => "&gt;",
)

function _interactive_block(node; class = "rk-aov-panel",
                            artifact_id::AbstractString,
                            artifact_kind::AbstractString = "aov-panel")
    payload = base64encode(_node_html(node))
    stable_id = _html_attribute(artifact_id)
    stable_kind = _html_attribute(artifact_kind)
    stable_payload = _html_attribute(payload)
    RawHTML("""
<div class="$class" data-rk-artifact-id="$stable_id"
     data-rk-artifact-kind="$stable_kind" data-rk-exec-payload="$stable_payload">
  <ClientOnly>
    <div v-exec-scripts="'$payload'"></div>
  </ClientOnly>
</div>
""")
end

_static_block(node) = RawHTML(_node_html(node))

"""Install HTMXObjects' sorting runtime once on a page that renders result tables."""
function render_result_assets()
    Markdown.MD(Any[_interactive_block(
        sortable_table_js(); class = "rk-result-assets",
        artifact_id = "assets:result-assets", artifact_kind = "result-assets",
    )])
end

const _REVIEW_STATUS_CONTENT = Dict(
    :frozen_ppl => (
        state = "frozen",
        icon = "❄",
        label = "Frozen model guide",
        title = "Outside the current PPL review set",
        body = "This walkthrough is preserved as executable example material, but it has not received the current migration and review pass. Eight Schools and MNIST are the reviewed PPL paths; treat claims on this page as provisional until it is revisited.",
    ),
    :frozen_sampling => (
        state = "frozen",
        icon = "❄",
        label = "Frozen sampling guide",
        title = "Experimental and awaiting re-review",
        body = "This sampling material is retained as historical or experimental evidence outside the current reviewed set. NUTS and WALNUTS remain source-only in the docs build; do not infer a current supported sampler or compiler capability from this page.",
    ),
    :frozen_bijectors => (
        state = "frozen",
        icon = "❄",
        label = "Frozen guide",
        title = "Preserved, but not recently reviewed",
        body = "This bijector guide remains available as implementation documentation, but it has not received the current focused review. Treat it as provisional rather than as the front-line supported contract.",
    ),
    :review_pending_nonallocating => (
        state = "review-pending",
        icon = "◌",
        label = "Review pending",
        title = "Implementation retained; guide awaiting review",
        body = "The non-allocating implementation and its tests remain in place, but this guide has not yet received the current focused review. This label records documentation review status, not a known runtime failure.",
    ),
)

"""Render a visible, source-controlled review-state banner for a docs page."""
function render_review_status(kind::Symbol)
    haskey(_REVIEW_STATUS_CONTENT, kind) ||
        error("unknown documentation review status: $(repr(kind))")
    status = _REVIEW_STATUS_CONTENT[kind]
    node = h.aside(;
        class = "rk-review-state rk-review-state--$(status.state)",
        data_rk_review_state = status.state,
        aria_label = "Documentation review status",
    )(
        h.div(status.icon; class = "rk-review-state__icon", aria_hidden = "true"),
        h.div(
            h.div(
                h.span(status.label; class = "rk-review-state__badge"),
                h.span("Documentation status"; class = "rk-review-state__eyebrow"),
                class = "rk-review-state__heading",
            ),
            h.strong(status.title; class = "rk-review-state__title"),
            h.p(status.body),
            class = "rk-review-state__content",
        ),
    )
    Markdown.MD(Any[_static_block(node)])
end

_default_cell(value, _) = string(value)
_default_sort(value, _) = value === missing ? nothing : value

function _column(key::Symbol, label::AbstractString;
                 format = _default_cell, sort = _default_sort)
    (; key, label = String(label), format, sort)
end

function _result_cell(row, column)
    value = getproperty(row, column.key)
    content = column.format(value, row)
    sort_value = column.sort(value, row)
    isnothing(sort_value) ? h.td(content) :
        h.td(content; data_sort_value = string(sort_value))
end

function _result_table(rows, columns; id::AbstractString, title = "Exact values",
                       note = "Select a column heading to sort.")
    body = [h.tr((_result_cell(row, column) for column in columns)...) for row in rows]
    table = sortable_table(
        [column.label for column in columns], body;
        id = String(id), download = false, class = "rk-result-table",
    )
    _static_block(h.section(;
        class = "rk-result-table-section",
        data_rk_artifact_id = "table:" * String(id),
        data_rk_artifact_kind = "sortable-table",
    )(
        h.h3(title),
        h.p(note; class = "rk-result-table-note"),
        h.div(table; class = "rk-sortable-table-wrap"),
    ))
end

function _plot_block(spec; id::AbstractString, title::AbstractString,
                     description::AbstractString = "")
    caption = isempty(description) ? h.strong(title) :
        h.div(h.strong(title), h.span(description; class = "rk-result-caption-detail"))
    node = h.figure(; class = "rk-result-figure")(
        h.figcaption(caption),
        vdraw(spec; id = String(id)),
    )
    _interactive_block(node; artifact_id = "plot:" * String(id))
end

const _TIMING_BACKENDS = (
    ("rk_native", "RK shared object"),
    ("rk_direct_native", "RK one-off control"),
    ("distributions_native", "Distributions native"),
    ("probability_measures_native", "ProbabilityMeasures native"),
    ("rk_reactant", "RK + Reactant"),
    ("distributions_reactant", "Distributions + Reactant"),
    ("probability_measures_reactant", "ProbabilityMeasures + Reactant"),
)

const _ALLOCATION_BACKENDS = (
    ("rk_native", "RK shared object"),
    ("rk_direct_native", "RK one-off control"),
    ("distributions_native", "Distributions native"),
    ("probability_measures_native", "ProbabilityMeasures native"),
    ("rk_reactant", "RK + Reactant"),
    ("probability_measures_reactant", "ProbabilityMeasures + Reactant"),
)

function _timing_rows(measurements; family = _ -> "", backends = _TIMING_BACKENDS)
    rows = NamedTuple[]
    for measurement in measurements
        for (key, label) in backends
            supported = haskey(measurement, key)
            push!(rows, (;
                family = family(measurement),
                n = Int(measurement["n"]),
                backend = label,
                median_ns = supported ? Float64(measurement[key]["median_ns"]) : missing,
                status = supported ? "measured" : "unsupported",
            ))
        end
    end
    rows
end

function _allocation_rows(measurements; family = _ -> "",
                          backends = _ALLOCATION_BACKENDS)
    rows = NamedTuple[]
    for measurement in measurements
        for (key, label) in backends
            supported = haskey(measurement, key)
            push!(rows, (;
                family = family(measurement),
                n = Int(measurement["n"]),
                backend = label,
                median_bytes = supported ? Int(measurement[key]["median_bytes"]) : missing,
                median_allocs = supported ? Int(measurement[key]["median_allocs"]) : missing,
                status = supported ? "measured" : "unsupported",
            ))
        end
    end
    rows
end

_unsupported(value, formatter) = value === missing ?
    h.span("unsupported"; class = "rk-result-unsupported") : formatter(value)

function _timing_columns(; include_family = false)
    columns = Any[]
    include_family && push!(columns, _column(:family, "Family"))
    append!(columns, (
        _column(:n, "N"),
        _column(:backend, "Backend"),
        _column(:median_ns, "Median runtime";
            format = (value, _) -> _unsupported(value, ns -> string(ns, " ns"))),
        _column(:status, "Status";
            format = (value, _) -> value == "measured" ? value :
                h.span(value; class = "rk-result-unsupported")),
    ))
    columns
end

function _allocation_columns(; include_family = false)
    columns = Any[]
    include_family && push!(columns, _column(:family, "Family"))
    append!(columns, (
        _column(:n, "N"),
        _column(:backend, "Backend"),
        _column(:median_bytes, "Median bytes";
            format = (value, _) -> _unsupported(value, bytes -> string(bytes, " B"))),
        _column(:median_allocs, "Median allocations";
            format = (value, _) -> _unsupported(value, string)),
        _column(:status, "Status";
            format = (value, _) -> value == "measured" ? value :
                h.span(value; class = "rk-result-unsupported")),
    ))
    columns
end

function _timing_plot(rows; id, title)
    supported = filter(row -> row.median_ns !== missing, rows)
    spec = data(supported) *
        mapping(:n => "N", :median_ns => "Median runtime (ns)";
                color = :backend => "Backend") *
        visual(ScatterLines) *
        config(
            height = 300,
            scales = scales(X = (; scale = log10), Y = (; scale = log10)),
        )
    _plot_block(spec; id, title,
        description = "Both axes are logarithmic; points are checked-in receipt medians.")
end

function _allocation_plot(rows; id, title)
    supported = filter(row -> row.median_bytes !== missing, rows)
    spec = data(supported) *
        mapping(:n => "N", :median_bytes => "Median allocated bytes";
                color = :backend => "Backend") *
        visual(ScatterLines) *
        config(
            height = 300,
            scales = scales(X = (; scale = log10), Y = (; scale = symlog)),
        )
    _plot_block(spec; id, title,
        description = "The symlog byte axis preserves genuine zero-allocation measurements.")
end

"""Render the build-executed MVN plans as two AoV comparisons and one sortable table."""
function render_mvn_parametrization_plans(source::AbstractString)
    sandbox = Module(gensym(:MVNParametrizationPlans), true, true)
    Core.eval(sandbox, :(using ReactiveKernels))
    _evaluate_source(sandbox, strip(source, '\n'))
    artifact = Core.eval(sandbox, :docs_example)
    kernels = artifact.kernels
    expected = (;
        covariance = ((:x, :μ, :covariance), "Covariance Σ"),
        cholesky = ((:x, :μ, :chol), "Covariance Cholesky L"),
        precision = ((:x, :μ, :precision), "Precision Ω"),
        precision_cholesky =
            ((:x, :μ, :precision_chol), "Precision Cholesky Q"),
    )
    propertynames(kernels) == propertynames(expected) ||
        error("unexpected MVN parametrization inventory")

    rows = NamedTuple[]
    for name in propertynames(expected)
        kernel = getproperty(kernels, name)
        kernel isa PreparedKernel || error("$name did not produce a PreparedKernel")
        have, label = getproperty(expected, name)
        observed_have = Tuple(value.name for value in inputs(kernel))
        observed_have == have ||
            error("$name selected HAVE $observed_have; expected $have")
        push!(rows, (;
            representation = label,
            have = join(string.(have), ", "),
            selected_recipes = length(kernel.plan.recipes),
            plan_cost = kernel.plan.cost,
        ))
    end

    recipe_spec = data(rows) *
        mapping(:representation => "Available representation",
                :selected_recipes => "Selected recipes") *
        visual(BarPlot) * config(height = 260, scales = scales(Y = (; zero = true)))
    cost_spec = data(rows) *
        mapping(:representation => "Available representation", :plan_cost => "Plan cost") *
        visual(BarPlot) * config(height = 260, scales = scales(Y = (; zero = true)))
    columns = (
        _column(:representation, "Available representation"),
        _column(:have, "HAVE boundary";
            format = (value, _) -> h.code(value)),
        _column(:selected_recipes, "Selected recipes"),
        _column(:plan_cost, "Plan cost"),
    )
    Markdown.MD(Any[
        _plot_block(recipe_spec; id = "mvn-selected-recipes",
            title = "Selected work by available representation"),
        _plot_block(cost_spec; id = "mvn-plan-cost",
            title = "Planner cost by available representation"),
        _result_table(rows, columns; id = "mvn-plan-table", title = "Exact plan data"),
    ])
end

"""Render the checked-in Normal benchmark receipt as AoV plots plus sortable data."""
function render_distribution_benchmarks()
    return _render_distribution_benchmarks_focused()
    receipt = TOML.parsefile(_DISTRIBUTION_RECEIPT_PATH)
    get(receipt, "schema", "") == "distribution-logdensity-v1" ||
        error("unexpected distribution benchmark receipt schema")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("distribution benchmark receipt was produced from a dirty RK tree")
    support = receipt["support"]
    get(support, "rk_reactant", false) ||
        error("distribution benchmark receipt does not accept the RK Reactant path")
    get(support, "probability_measures_reactant", false) || error(
        "distribution benchmark receipt does not accept ProbabilityMeasures + Reactant",
    )
    protocol = receipt["protocol"]
    get(protocol, "reactant_sync", false) || error("Reactant receipt is not synchronous")
    get(protocol, "reactant_transfers_included", true) &&
        error("Reactant receipt includes host/device transfers")

    measurements = receipt["measurements"]
    timing_rows = _timing_rows(measurements)
    allocation_rows = _allocation_rows(measurements)

    largest = last(measurements)
    ratio(numerator, denominator) = round(
        largest[numerator]["median_ns"] / largest[denominator]["median_ns"];
        digits = 2,
    )
    crossover = first(row for row in measurements
        if row["rk_reactant"]["median_ns"] < row["rk_native"]["median_ns"])
    largest_n = Int(largest["n"])
    crossover_n = Int(crossover["n"])
    shared_control_ratio = round(
        largest["rk_native"]["median_ns"] /
        largest["rk_direct_native"]["median_ns"];
        digits = 2,
    )
    summary = "At N=$largest_n, native RK is " *
        "$(ratio("distributions_native", "rk_native"))× faster than Distributions and " *
        "$(ratio("probability_measures_native", "rk_native"))× faster than " *
        "ProbabilityMeasures. The shared-object/one-off-control runtime ratio is " *
        "$shared_control_ratio× (1× is identical). " *
        "RK + Reactant is " *
        "$(ratio("rk_native", "rk_reactant"))× faster than native RK and " *
        "$(ratio("probability_measures_reactant", "rk_reactant"))× faster than " *
        "ProbabilityMeasures + Reactant. In the sampled sizes, Reactant first " *
        "beats native RK at N=$crossover_n."
    provenance = "Receipt pins: RK `$(first(String(pins["reactivekernels_sha"]), 10))`; " *
        "ProbabilityMeasures `$(first(String(pins["probability_measures_sha"]), 10))`; " *
        "Distributions $(pins["distributions_version"]); Reactant " *
        "$(pins["reactant_version"]); Julia $(pins["julia_version"]); " *
        "$(receipt["environment"]["cpu"])."

    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _timing_plot(timing_rows; id = "normal-runtime", title = "Normal log-density runtime"),
        _result_table(timing_rows, _timing_columns(); id = "normal-runtime-table",
            title = "Exact runtime values"),
        _allocation_plot(allocation_rows; id = "normal-allocations",
            title = "Normal log-density allocation"),
        _result_table(allocation_rows, _allocation_columns();
            id = "normal-allocation-table", title = "Exact allocation values"),
        Markdown.Paragraph(Any[Markdown.Italic(Any[provenance])]),
    ])
end

const _DISTRIBUTION_GRADIENT_SURFACES = (
    ("returned_gradient", "Gradient result — ad_gradient"),
    ("caller_owned_gradient",
     "Value + caller-owned gradient — ad_value_and_gradient!"),
)

const _DISTRIBUTION_GRADIENT_FAMILY_LABELS = Dict(
    "normal" => "Normal plate",
    "cauchy_location_scale" => "Cauchy",
    "laplace_location_scale" => "Laplace",
    "bernoulli_logit" => "Bernoulli",
    "lognormal_logscale" => "LogNormal",
    "exponential_logscale" => "Exponential",
    "geometric_logit" => "Geometric",
    "uniform_bounded" => "Uniform",
    "mvnormal_cholesky" => "MVN (Cholesky)",
)

const _DISTRIBUTION_GRADIENT_GROUP_LABELS = Dict(
    "normal_plate" => "Normal plate",
    "scalar_gallery" => "Scalar gallery",
    "structured" => "Structured",
)

function _distribution_gradient_rows(measurements)
    rows = NamedTuple[]
    for measurement in measurements
        family = _DISTRIBUTION_GRADIENT_FAMILY_LABELS[measurement["family"]]
        for (key, method) in _DISTRIBUTION_GRADIENT_SURFACES
            haskey(measurement, key) || continue
            result = measurement[key]
            push!(rows, (;
                group_key = String(measurement["group"]),
                group = _DISTRIBUTION_GRADIENT_GROUP_LABELS[measurement["group"]],
                family,
                n = Int(measurement["n"]),
                active = String(measurement["active"]),
                method,
                series = string(family, " / ", method),
                median_ns = Float64(result["median_ns"]),
                median_bytes = Int(result["median_bytes"]),
                median_allocs = Int(result["median_allocs"]),
            ))
        end
    end
    rows
end

function _distribution_gradient_plot(rows, group_key, metric;
                                     id::AbstractString, title::AbstractString)
    selected = filter(row -> row.group_key == group_key, rows)
    color_key = group_key == "scalar_gallery" ? :series : :method
    label = metric === :median_ns ? "Median runtime (ns)" : "Median allocated bytes"
    scale = metric === :median_ns ? log10 : symlog
    spec = data(selected) *
        mapping(:n => "N", metric => label; color = color_key => "Call surface") *
        visual(ScatterLines) *
        config(
            height = group_key == "scalar_gallery" ? 380 : 300,
            scales = scales(X = (; scale = log10), Y = (; scale)),
        )
    description = metric === :median_ns ?
        "Both axes are logarithmic; preparation is excluded." :
        "The symlog byte axis preserves genuine zero-allocation measurements."
    _plot_block(spec; id, title, description)
end

"""Render prepared distribution-gradient timing and allocation evidence."""
function render_distribution_gradient_benchmarks()
    return _render_distribution_gradient_benchmarks_focused()
    receipt = TOML.parsefile(_DISTRIBUTION_GRADIENT_RECEIPT_PATH)
    get(receipt, "schema", "") == "distribution-gradient-v1" ||
        error("unexpected distribution-gradient receipt schema")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("distribution-gradient receipt was produced from a dirty RK tree")
    protocol = receipt["protocol"]
    get(protocol, "preparation_in_timed_region", true) == false ||
        error("distribution-gradient receipt includes preparation")
    get(protocol, "returned_surface", "") == "ad_gradient" ||
        error("distribution-gradient returned surface drifted")
    get(protocol, "caller_owned_surface", "") == "ad_value_and_gradient!" ||
        error("distribution-gradient caller-owned surface drifted")
    Int(get(protocol, "rounds", 0)) >= 5 ||
        error("distribution-gradient receipt lacks five raw rounds")

    source_receipt_paths = Dict(
        "normal" => _DISTRIBUTION_RECEIPT_PATH,
        "scalar_gallery" => _SCALAR_GALLERY_RECEIPT_PATH,
        "structured" => _STRUCTURED_DISTRIBUTION_RECEIPT_PATH,
    )
    for (name, path) in source_receipt_paths
        source = TOML.parsefile(path)
        recorded = receipt["source_receipts"][name]
        get(recorded, "schema", "") == source["schema"] ||
            error("distribution-gradient $name source schema drifted")
        get(recorded, "generated_at", "") == source["generated_at"] ||
            error("distribution-gradient $name source generation drifted")
        get(recorded, "reactivekernels_sha", "") ==
            source["pins"]["reactivekernels_sha"] ||
            error("distribution-gradient $name source pin drifted")
    end

    measurements = receipt["measurements"]
    length(measurements) == 24 ||
        error("distribution-gradient receipt must contain 24 case/size rows")
    all(row -> Float64(row["max_relative_error"]) <= 1e-10, measurements) ||
        error("distribution-gradient receipt exceeds analytic parity tolerance")
    rows = _distribution_gradient_rows(measurements)

    steady_measurement(row) = row[row["active_kind"] == "vector" ?
        "caller_owned_gradient" : "returned_gradient"]
    zero_cases = count(measurements) do row
        measurement = steady_measurement(row)
        Int(measurement["median_bytes"]) == 0 &&
            Int(measurement["median_allocs"]) == 0
    end
    zero_cases == 20 ||
        error("distribution-gradient receipt must retain 20 zero-allocation cases")
    structured_bytes = [
        Int(row["caller_owned_gradient"]["median_bytes"])
        for row in measurements if row["group"] == "structured"
    ]
    summary = "Across $(length(measurements)) distribution case/size combinations, " *
        "$zero_cases have a zero-byte steady-state gradient path: every Normal " *
        "plate and scalar-gallery row. The four Cholesky-MVN rows retain " *
        "$(minimum(structured_bytes))–$(maximum(structured_bytes)) B of internal " *
        "work even when the gradient destination is caller-owned."
    provenance = "Receipt pins: RK `$(first(String(pins["reactivekernels_sha"]), 10))`; " *
        "DifferentiationInterface $(pins["differentiationinterface_version"]); " *
        "Enzyme $(pins["enzyme_version"]); Julia $(pins["julia_version"]); " *
        "$(receipt["environment"]["cpu"])."

    timing_columns = (
        _column(:group, "Corpus"),
        _column(:family, "Family"),
        _column(:n, "N"),
        _column(:active, "Active port"; format = (value, _) -> h.code(value)),
        _column(:method, "Call surface"),
        _column(:median_ns, "Median runtime";
            format = (value, _) -> _fmt_ns(value)),
    )
    allocation_columns = (
        _column(:group, "Corpus"),
        _column(:family, "Family"),
        _column(:n, "N"),
        _column(:active, "Active port"; format = (value, _) -> h.code(value)),
        _column(:method, "Call surface"),
        _column(:median_bytes, "Median bytes";
            format = (value, _) -> string(value, " B")),
        _column(:median_allocs, "Median allocations"),
    )

    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _distribution_gradient_plot(rows, "normal_plate", :median_ns;
            id = "ad-normal-runtime", title = "Normal gradient runtime"),
        _distribution_gradient_plot(rows, "normal_plate", :median_bytes;
            id = "ad-normal-allocation", title = "Normal gradient allocation"),
        _distribution_gradient_plot(rows, "scalar_gallery", :median_ns;
            id = "ad-scalar-runtime", title = "Scalar-gallery gradient runtime"),
        _distribution_gradient_plot(rows, "scalar_gallery", :median_bytes;
            id = "ad-scalar-allocation", title = "Scalar-gallery gradient allocation"),
        _distribution_gradient_plot(rows, "structured", :median_ns;
            id = "ad-mvn-runtime", title = "MVN gradient runtime"),
        _distribution_gradient_plot(rows, "structured", :median_bytes;
            id = "ad-mvn-allocation", title = "MVN gradient allocation"),
        _result_table(rows, timing_columns; id = "distribution-gradient-runtime-table",
            title = "Exact distribution-gradient runtimes"),
        _result_table(rows, allocation_columns;
            id = "distribution-gradient-allocation-table",
            title = "Exact distribution-gradient allocations"),
        Markdown.Paragraph(Any[Markdown.Italic(Any[provenance])]),
    ])
end

"""Render the replicated Reactant call boundary as total and per-evaluation time."""
function render_distribution_amortization()
    return _render_distribution_amortization_focused()
    receipt = TOML.parsefile(_DISTRIBUTION_RECEIPT_PATH)
    get(receipt, "schema", "") == "distribution-logdensity-v1" ||
        error("unexpected distribution benchmark receipt schema")
    measurements = receipt["measurements"]
    direct = only(row for row in measurements if Int(row["n"]) == 1)
    direct_ns = Float64(direct["rk_reactant"]["median_ns"])
    rows = map(receipt["reactant_amortization"]) do row
        measurement = row["rk_reactant_replicated"]
        per_evaluation = Float64(row["median_ns_per_evaluation"])
        (;
            replicas = Int(row["replicas"]),
            total_ns = Float64(measurement["median_ns"]),
            per_evaluation_ns = per_evaluation,
            host_bytes = Int(measurement["median_bytes"]),
            host_allocs = Int(measurement["median_allocs"]),
            speedup = direct_ns / per_evaluation,
        )
    end
    largest = last(rows)
    summary = "The direct one-observation compiled call takes $(_fmt_ns(direct_ns)). " *
        "With $(largest.replicas) independent observations in one replica call, " *
        "the normalized cost is $(_fmt_ns(largest.per_evaluation_ns)) " *
        "($(round(largest.speedup; digits = 1))× lower), while the entire call " *
        "still has $(largest.host_allocs) host allocations / $(largest.host_bytes) B."
    columns = (
        _column(:replicas, "Independent evaluations"),
        _column(:total_ns, "Whole-call runtime";
            format = (value, _) -> _fmt_ns(value)),
        _column(:per_evaluation_ns, "Runtime / evaluation";
            format = (value, _) -> _fmt_ns(value)),
        _column(:host_allocs, "Host allocations"),
        _column(:host_bytes, "Host bytes";
            format = (value, _) -> string(value, " B")),
        _column(:speedup, "Direct ÷ batched";
            format = (value, _) -> string(round(value; digits = 1), "×")),
    )
    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _result_table(rows, columns; id = "normal-reactant-amortization",
            title = "Reactant invocation-cost amortization"),
    ])
end

"""Render PPL replica throughput separately from the single-call latency chart."""
function render_eval_throughput_amortization()
    return _render_eval_throughput_amortization_focused()
    receipt = TOML.parsefile(_eval_throughput_receipt())
    replicas = Int(receipt["protocol"]["replicas"])
    measurements = receipt["measurements"]
    rows = NamedTuple[]
    for n in (16, 256, 4096), mode in ("primal", "gradient", "gq")
        direct = only(row for row in measurements
            if Int(row["size"]) == n && row["mode"] == mode &&
               row["implementation"] == "reactivekernels" &&
               row["variant"] == "reactant")
        replicated = only(row for row in measurements
            if Int(row["size"]) == n && row["mode"] == mode &&
               row["implementation"] == "reactivekernels" &&
               row["variant"] == "reactant_replicated")
        push!(rows, (;
            n,
            mode,
            replicas = Int(replicated["batch_size"]),
            direct_ns = Float64(direct["median_ns"]),
            batch_ns = Float64(replicated["median_batch_ns"]),
            per_evaluation_ns = Float64(replicated["median_ns"]),
            speedup = Float64(direct["median_ns"]) /
                Float64(replicated["median_ns"]),
        ))
    end
    small = filter(row -> row.n == 16, rows)
    small_speedups = getproperty.(small, :speedup)
    summary = "At n=16, one $replicas-position call lowers normalized Reactant " *
        "cost by $(round(minimum(small_speedups); digits = 1))–" *
        "$(round(maximum(small_speedups); digits = 1))× across the three modes."
    columns = (
        _column(:n, "Position size"),
        _column(:mode, "Mode"),
        _column(:replicas, "Positions / call"),
        _column(:direct_ns, "Direct call"; format = (value, _) -> _fmt_ns(value)),
        _column(:batch_ns, "Whole batch"; format = (value, _) -> _fmt_ns(value)),
        _column(:per_evaluation_ns, "Batch / position";
            format = (value, _) -> _fmt_ns(value)),
        _column(:speedup, "Direct ÷ batched";
            format = (value, _) -> string(round(value; digits = 1), "×")),
    )
    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _result_table(rows, columns; id = "eval-reactant-amortization",
            title = "Replicated PPL evaluation throughput"),
    ])
end

"""Render the scalar distribution gallery receipt as per-family plots and one table."""
function render_scalar_gallery_benchmarks()
    return _render_scalar_gallery_benchmarks_focused()
    receipt = TOML.parsefile(_SCALAR_GALLERY_RECEIPT_PATH)
    get(receipt, "schema", "") == "scalar-distribution-gallery-v1" ||
        error("unexpected scalar gallery benchmark receipt schema")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("scalar gallery receipt was produced from a dirty RK tree")
    protocol = receipt["protocol"]
    get(protocol, "reactant_sync", false) ||
        error("scalar gallery Reactant receipt is not synchronous")
    get(protocol, "reactant_transfers_included", true) &&
        error("scalar gallery Reactant receipt includes host/device transfers")
    families = (
        "cauchy_location_scale",
        "laplace_location_scale",
        "bernoulli_logit",
        "lognormal_logscale",
        "exponential_logscale",
        "geometric_logit",
        "uniform_bounded",
    )
    Tuple(protocol["families"]) == families ||
        error("unexpected scalar gallery family inventory")
    support = receipt["support"]
    support_errors = receipt["support_errors"]
    for family in families
        family_support = support[family]
        get(family_support, "rk_reactant", false) ||
            error("$family does not accept the RK Reactant path")
        get(family_support, "probability_measures_reactant", false) ||
            error("$family does not accept the ProbabilityMeasures Reactant path")
        if !get(family_support, "distributions_reactant", false)
            isempty(get(support_errors[family], "distributions_reactant", "")) &&
                error("$family Distributions Reactant path lacks a diagnostic")
        end
    end

    family_label = Dict(
        "cauchy_location_scale" => "Cauchy",
        "laplace_location_scale" => "Laplace",
        "bernoulli_logit" => "Bernoulli",
        "lognormal_logscale" => "LogNormal",
        "exponential_logscale" => "Exponential",
        "geometric_logit" => "Geometric",
        "uniform_bounded" => "Uniform",
    )
    measurements = receipt["measurements"]
    rows = _timing_rows(measurements;
        family = measurement -> family_label[measurement["family"]])
    largest_n = maximum(Int(row["n"]) for row in measurements)
    ratio(row, numerator, denominator) = round(
        row[numerator]["median_ns"] / row[denominator]["median_ns"];
        digits = 2,
    )
    summaries = String[]
    for family in families
        row = only(filter(candidate -> candidate["family"] == family &&
            Int(candidate["n"]) == largest_n, measurements))
        push!(summaries,
            "$(family_label[family]): Distributions/RK native " *
            "$(ratio(row, "distributions_native", "rk_native"))×, " *
            "ProbabilityMeasures/RK native " *
            "$(ratio(row, "probability_measures_native", "rk_native"))×, " *
            "ProbabilityMeasures+Reactant/RK+Reactant " *
            "$(ratio(row, "probability_measures_reactant", "rk_reactant"))×")
    end
    summary = "At N=$largest_n, measured runtime ratios (comparison/RK) are: " *
        join(summaries, "; ") * "."
    provenance = "Receipt pins: RK `$(first(String(pins["reactivekernels_sha"]), 10))`; " *
        "ProbabilityMeasures `$(first(String(pins["probability_measures_sha"]), 10))`; " *
        "Distributions $(pins["distributions_version"]); Reactant " *
        "$(pins["reactant_version"]); Julia $(pins["julia_version"]); " *
        "$(receipt["environment"]["cpu"])."

    blocks = Any[Markdown.Paragraph(Any[summary])]
    for family in families
        label = family_label[family]
        family_rows = filter(row -> row.family == label, rows)
        push!(blocks, _timing_plot(family_rows;
            id = "scalar-$(lowercase(label))-runtime", title = "$label runtime"))
    end
    push!(blocks,
        _result_table(rows, _timing_columns(; include_family = true);
            id = "scalar-runtime-table", title = "Exact scalar-gallery runtimes"),
        Markdown.Paragraph(Any[Markdown.Italic(Any[provenance])]),
    )
    Markdown.MD(blocks)
end

"""Render the structured distribution receipt as per-family plots and one table."""
function render_structured_distribution_benchmarks()
    return _render_structured_distribution_benchmarks_focused()
    receipt = TOML.parsefile(_STRUCTURED_DISTRIBUTION_RECEIPT_PATH)
    get(receipt, "schema", "") == "structured-distribution-logdensity-v1" ||
        error("unexpected structured distribution benchmark receipt schema")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("structured distribution receipt was produced from a dirty RK tree")
    protocol = receipt["protocol"]
    get(protocol, "reactant_sync", false) ||
        error("structured Reactant receipt is not synchronous")
    get(protocol, "reactant_transfers_included", true) &&
        error("structured Reactant receipt includes host/device transfers")
    Tuple(get(protocol, "mvn_have_boundaries", String[])) ==
        ("covariance", "cholesky", "precision", "precision_cholesky") ||
        error("structured receipt does not cover every MVN HAVE boundary")
    get(protocol, "mvn_all_boundaries_native_and_reactant_accepted", false) ||
        error("structured receipt lacks all-boundary native/Reactant acceptance")
    support = receipt["support"]
    support_errors = receipt["support_errors"]
    for family in ("mvnormal_cholesky",)
        family_support = support[family]
        get(family_support, "rk_reactant", false) ||
            error("$family does not accept the RK Reactant path")
        for library in ("distributions_reactant", "probability_measures_reactant")
            get(family_support, library, false) && continue
            isempty(get(support_errors[family], library, "")) &&
                error("$family $library is unsupported without a diagnostic")
        end
    end

    family_label = Dict("mvnormal_cholesky" => "MVN (Cholesky)")
    measurements = receipt["measurements"]
    rows = _timing_rows(measurements;
        family = measurement -> family_label[measurement["family"]])
    largest = last(measurements)
    n = Int(largest["n"])
    dist_ratio = round(
        largest["distributions_native"]["median_ns"] /
        largest["rk_native"]["median_ns"];
        digits = 2,
    )
    pm_ratio = round(
        largest["probability_measures_native"]["median_ns"] /
        largest["rk_native"]["median_ns"];
        digits = 2,
    )
    summary = "At N=$n, measured full-MVN runtime ratios (comparison/RK) are " *
        "Distributions $dist_ratio× and ProbabilityMeasures $pm_ratio×. " *
        "Full-MVN Reactant cells for both comparison libraries are unsupported; " *
        "the RK MVN kernel compiles."
    provenance = "Receipt pins: RK `$(first(String(pins["reactivekernels_sha"]), 10))`; " *
        "ProbabilityMeasures `$(first(String(pins["probability_measures_sha"]), 10))`; " *
        "Distributions $(pins["distributions_version"]); Reactant " *
        "$(pins["reactant_version"]); Julia $(pins["julia_version"]); " *
        "$(receipt["environment"]["cpu"])."

    blocks = Any[Markdown.Paragraph(Any[summary])]
    for label in ("MVN (Cholesky)",)
        family_rows = filter(row -> row.family == label, rows)
        push!(blocks, _timing_plot(family_rows;
            id = "structured-mvn-runtime", title = "$label runtime"))
    end
    push!(blocks,
        _result_table(rows, _timing_columns(; include_family = true);
            id = "structured-runtime-table", title = "Exact structured-kernel runtimes"),
        Markdown.Paragraph(Any[Markdown.Italic(Any[provenance])]),
    )
    Markdown.MD(blocks)
end

const _BATCHED_TIMING_BACKENDS = (
    ("rk_native", "Legacy plate / native"),
    ("rk_authored_native", "Authored return / native"),
    ("rk_reactant", "Legacy plate / Reactant"),
    ("rk_authored_reactant", "Authored return / Reactant"),
)

const _BATCHED_ALLOCATION_BACKENDS = _BATCHED_TIMING_BACKENDS

"""Render authored/legacy plate parity from the checked-in Normal receipt."""
function render_batched_benchmarks()
    return _render_batched_benchmarks_focused()
    receipt = TOML.parsefile(_DISTRIBUTION_RECEIPT_PATH)
    get(receipt, "schema", "") == "distribution-logdensity-v1" ||
        error("unexpected batched benchmark receipt schema")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("batched benchmark receipt was produced from a dirty RK tree")
    support = receipt["support"]
    get(support, "rk_reactant", false) ||
        error("batched receipt does not accept the legacy RK Reactant path")
    get(support, "rk_authored_reactant", false) ||
        error("batched receipt does not accept the authored RK Reactant path")
    protocol = receipt["protocol"]
    get(protocol, "reactant_sync", false) ||
        error("batched Reactant receipt is not synchronous")
    get(protocol, "reactant_transfers_included", true) &&
        error("batched Reactant receipt includes host/device transfers")
    get(protocol, "rk_authored_native_lowering", "") ==
        "one reduction traversal with no similar/pointwise output" ||
        error("batched receipt lacks the authored native fusion check")
    get(protocol, "rk_authored_reactant_lowering", "") ==
        "tensorized broadcast chain consumed by Base.sum; no host loop or similar" ||
        error("batched receipt lacks the authored Reactant fusion check")

    measurements = receipt["measurements"]
    Tuple(Int(row["n"]) for row in measurements) ==
        (1, 1_000, 10_000, 30_000, 100_000, 1_000_000) ||
        error("batched receipt does not use the published six-size series")
    all(haskey(row, key) for row in measurements
        for (key, _) in _BATCHED_TIMING_BACKENDS) ||
        error("batched receipt is missing an authored/legacy measurement")
    timing_rows = _timing_rows(measurements; backends = _BATCHED_TIMING_BACKENDS)
    allocation_rows = _allocation_rows(
        measurements; backends = _BATCHED_ALLOCATION_BACKENDS,
    )

    ratio(row, numerator, denominator) =
        Float64(row[numerator]["median_ns"]) /
        Float64(row[denominator]["median_ns"])
    gated_rows = filter(row -> Int(row["n"]) > 1, measurements)
    maximum_native_ratio = maximum(
        ratio(row, "rk_authored_native", "rk_native") for row in gated_rows
    )
    largest = last(measurements)
    largest_n = Int(largest["n"])
    largest_native_ratio = ratio(largest, "rk_authored_native", "rk_native")
    largest_reactant_ratio = ratio(largest, "rk_authored_reactant", "rk_reactant")
    authored_native_bytes = unique(
        Int(row["rk_authored_native"]["median_bytes"]) for row in measurements
    )
    authored_native_allocs = unique(
        Int(row["rk_authored_native"]["median_allocs"]) for row in measurements
    )
    authored_native_bytes == [0] ||
        error("authored native return is not zero-byte across all sizes")
    authored_native_allocs == [0] ||
        error("authored native return is not zero-allocation across all sizes")

    summary = "The authored return-only kernel remains within " *
        "$(round(maximum_native_ratio; digits = 2))× of the legacy plate across " *
        "the gated N ≥ 1,000 rows (the validator requires ≤ 1.10×). At " *
        "N=$largest_n, authored/legacy is " *
        "$(round(largest_native_ratio; digits = 2))× natively and " *
        "$(round(largest_reactant_ratio; digits = 2))× under Reactant. " *
        "The authored native return is zero-byte and zero-allocation at every " *
        "measured size; Reactant uses the same tensorized broadcast-to-sum " *
        "shape without a pointwise output or host loop."
    provenance = "Receipt pin: ReactiveKernels " *
        "`$(pins["reactivekernels_sha"])`; Julia $(pins["julia_version"]); " *
        "Reactant $(pins["reactant_version"]); $(receipt["environment"]["cpu"]). " *
        "Raw per-round timings, bytes, and allocation counts are retained in " *
        "benchmark/receipts/distribution-logdensity-v1.toml."

    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _timing_plot(timing_rows; id = "batched-authored-runtime",
            title = "Authored return versus legacy plate runtime"),
        _result_table(timing_rows, _timing_columns();
            id = "batched-authored-runtime-table", title = "Exact runtime values"),
        _allocation_plot(allocation_rows; id = "batched-authored-allocation",
            title = "Authored return versus legacy plate allocation"),
        _result_table(allocation_rows, _allocation_columns();
            id = "batched-authored-allocation-table",
            title = "Exact allocation values"),
        Markdown.Paragraph(Any[Markdown.Italic(Any[provenance])]),
    ])
end

const _EIGHT_SCHOOLS_BOUNDARIES = (
    "packed_unconstrained", "constrained_parameters", "minimal_likelihood",
)
const _EIGHT_SCHOOLS_OUTCOMES = ("joint", "prior", "likelihood", "pointwise")
const _EIGHT_SCHOOLS_BACKENDS = ("rk_native", "manual_julia", "turing_native")

function _eight_schools_supported(boundary, outcome, backend)
    boundary != "minimal_likelihood" && return true
    outcome in ("likelihood", "pointwise") && backend in ("rk_native", "manual_julia")
end

function _eight_schools_measurement(row, backend)
    haskey(row, backend) || return missing
    result = row[backend]
    (;
        median_ns = Float64(result["median_ns"]),
        median_bytes = Int(result["median_bytes"]),
        median_allocs = Int(result["median_allocs"]),
    )
end

function _eight_schools_cell(value, _)
    value === missing && return ""
    "$(_fmt_ns(value.median_ns)); $(value.median_bytes) B; " *
    "$(value.median_allocs) alloc"
end

function _longform_result(row)
    get(row, "state", "") == "supported" || return nothing
    result = get(row, "result", nothing)
    result isa AbstractDict || error(
        "supported long-form benchmark row lacks a result: " *
        "$(get(row, "configuration", "unknown")) / " *
        "$(get(row, "boundary", "unknown")) / $(get(row, "outcome", "unknown"))")
    result
end

function _longform_optional_metric(result, key, convert)
    result === nothing && return missing
    haskey(result, key) || return missing
    convert(result[key])
end

function _longform_metric(value, row, formatter)
    value !== missing && return formatter(value)
    label = row.status == "measured" ? "not recorded" : "—"
    h.span(label; class = "rk-result-unsupported")
end

function _render_longform_model_receipt(path, expected_schema;
                                        id, title, note)
    receipt = TOML.parsefile(path)
    get(receipt, "schema", "") == expected_schema ||
        error("unexpected benchmark receipt schema at $path")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("benchmark receipt was produced from a dirty checkout")
    sha = String(get(pins, "reactivekernels_sha", ""))
    occursin(r"^[0-9a-f]{40}$", sha) ||
        error("benchmark receipt lacks an exact ReactiveKernels SHA")

    measurements = receipt["measurements"]
    rows = map(measurements) do row
        result = _longform_result(row)
        runtime = result === nothing ? missing : Float64(
            haskey(result, "min_ns") ? result["min_ns"] : result["median_ns"])
        bytes = _longform_optional_metric(result, "median_bytes", Int)
        allocations = _longform_optional_metric(result, "median_allocs", Int)
        state = String(get(row, "state", "unsupported"))
        status = state == "supported" ? "measured" :
            state * ": " * String(get(row, "reason", "no public cell"))
        (;
            provider = String(row["provider"]),
            model = String(row["model"]),
            configuration = String(row["configuration"]),
            boundary = String(row["boundary"]),
            outcome = String(row["outcome"]),
            runtime,
            bytes,
            allocations,
            status,
        )
    end
    measured = count(row -> row.runtime !== missing, rows)
    count(row -> row.status == "measured", rows) == measured || error(
        "long-form receipt contains a supported row without timing evidence")
    summary = "The receipt accounts for $(length(rows)) long-form matrix rows; " *
        "$measured are measured and $(length(rows) - measured) are explicit " *
        "N/A or unsupported public-API cells."
    columns = (
        _column(:provider, "Provider"),
        _column(:model, "Model"),
        _column(:configuration, "Configuration"),
        _column(:boundary, "Input boundary"),
        _column(:outcome, "Requested output"),
        _column(:runtime, "Steady-state runtime";
            format = (value, row) -> _longform_metric(value, row, _fmt_ns)),
        _column(:bytes, "Allocated bytes";
            format = (value, row) -> _longform_metric(
                value, row, bytes -> string(bytes, " B"))),
        _column(:allocations, "Allocations";
            format = (value, row) -> _longform_metric(value, row, string)),
        _column(:status, "State";
            format = (value, _) -> value == "measured" ? value :
                h.span(value; class = "rk-result-unsupported")),
    )
    provenance = "Receipt `$(basename(path))`; RK `$sha`; " *
        "Julia $(pins["julia_version"]); $(receipt["environment"]["cpu"])."
    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _result_table(rows, columns; id, title, note),
        Markdown.Paragraph(Any[Markdown.Italic(Any[provenance])]),
    ])
end

include(joinpath(@__DIR__, "mnist_result_views.jl"))
include(joinpath(@__DIR__, "all_benchmark_views.jl"))

"""Render the pinned native and Reactant sum-to-zero hot-loop comparisons."""
function render_sum_to_zero_benchmarks()
    native = _allbench_checked_receipt(
        _SUM_TO_ZERO_NATIVE_RECEIPT_PATH, "sum-to-zero-hotloop-v1")
    reactant = _allbench_checked_receipt(
        _SUM_TO_ZERO_REACTANT_RECEIPT_PATH, "sum-to-zero-reactant-v1")
    for receipt in (native, reactant)
        protocol = receipt["protocol"]
        get(protocol, "hot_loop_excludes_recovery", false) ||
            error("sum-to-zero benchmark allowed recovery into the hot loop")
        get(protocol, "setup_in_timed_region", false) == false ||
            error("sum-to-zero setup entered steady-state timing")
        get(protocol, "preparation_in_timed_region", false) == false ||
            error("sum-to-zero preparation entered steady-state timing")
    end
    reactant_protocol = reactant["protocol"]
    get(reactant_protocol, "reactant_sync", false) ||
        error("sum-to-zero Reactant receipt is not synchronous")
    get(reactant_protocol, "reactant_transfers_in_timed_region", true) == false ||
        error("sum-to-zero Reactant timing includes transfers")
    get(reactant_protocol, "reactant_compile_time_in_timed_region", true) == false ||
        error("sum-to-zero Reactant timing includes compilation")
    get(reactant_protocol, "reactant_readback_in_timed_region", true) == false ||
        error("sum-to-zero Reactant timing includes readback")

    panels = (
        ("Native primal", native, (
            ("manual_primal", "Manual Julia", "Manual Julia"),
            ("rk_primal_native", "ReactiveKernels", "ReactiveKernels"),
            ("turing_primal", "Turing", "Turing"),
        ), "manual_primal", "Manual Julia"),
        ("Native value + gradient", native, (
            ("manual_ad", "Manual Julia + Enzyme", "Manual Julia"),
            ("rk_ad_native", "ReactiveKernels + Enzyme", "ReactiveKernels"),
            ("turing_ad", "Turing + Enzyme", "Turing"),
        ), "manual_ad", "Manual Julia + Enzyme"),
        ("Reactant primal", reactant, (
            ("manual_primal_reactant", "Manual Julia + Reactant", "Manual Julia"),
            ("rk_primal_reactant", "ReactiveKernels + Reactant", "ReactiveKernels"),
        ), "manual_primal_reactant", "Manual Julia + Reactant"),
        ("Reactant value + gradient", reactant, (
            ("manual_ad_reactant", "Manual Julia + Reactant AD", "Manual Julia"),
            ("rk_ad_reactant", "ReactiveKernels + Reactant AD", "ReactiveKernels"),
        ), "manual_ad_reactant", "Manual Julia + Reactant AD"),
    )

    rows = NamedTuple[]
    for (panel, receipt, entries, baseline_configuration, baseline_label) in panels
        supported = Dict(
            String(row["configuration"]) => row
            for row in receipt["measurements"] if row["state"] == "supported")
        baseline = supported[baseline_configuration]["result"]
        baseline_ns = Float64(baseline["median_ns"])
        for (configuration, label, implementation) in entries
            result = supported[configuration]["result"]
            runtime_ns = Float64(result["median_ns"])
            push!(rows, (;
                panel,
                series = label,
                implementation,
                runtime_ns,
                baseline_ns,
                baseline_label,
                relative_time = runtime_ns / baseline_ns,
                median_bytes = haskey(result, "median_bytes") ?
                    Int(result["median_bytes"]) : missing,
                median_allocs = haskey(result, "median_allocs") ?
                    Int(result["median_allocs"]) : missing,
            ))
        end
    end

    native_pins = native["pins"]
    reactant_pins = reactant["pins"]
    summary = "The sampler hot loop uses the packed unconstrained posterior " *
        "with observations, observation scales, and the intercept prior scale " *
        "bound before timing. Each panel is normalized to its optimized manual " *
        "Julia control. Recovery is a separate prepared cut and is absent from " *
        "every timed operation."
    provenance = "Receipts `$(basename(_SUM_TO_ZERO_NATIVE_RECEIPT_PATH))` and " *
        "`$(basename(_SUM_TO_ZERO_REACTANT_RECEIPT_PATH))`; native RK " *
        "`$(native_pins["reactivekernels_sha"])`; Reactant RK " *
        "`$(reactant_pins["reactivekernels_sha"])`; Reactant " *
        "$(reactant_pins["reactant_version"]); Turing " *
        "$(native_pins["turing_version"]); Julia " *
        "$(native_pins["julia_version"]); $(native["environment"]["cpu"])."
    _allbench_snapshot_sections(rows;
        id_prefix = "sum-to-zero-hotloop",
        profile_label = "Sum-to-zero hot loop",
        summary,
        provenance,
        panel_order = first.(panels),
    )
end

_eight_schools_sort(value, _) = value === missing ? nothing : value.median_ns

"""Render focused primal Eight Schools comparisons from the complete receipt."""
function render_eight_schools_primal_benchmarks()
    return _render_eight_schools_longform_focused(
        _EIGHT_SCHOOLS_PRIMAL_RECEIPT_PATH, "eight-schools-primal-v2";
        mode = :primal, id_prefix = "eight-schools-primal",
        profile_label = "Eight Schools primal")
    receipt = TOML.parsefile(_EIGHT_SCHOOLS_PRIMAL_RECEIPT_PATH)
    get(receipt, "schema", "") == "eight-schools-primal-v1" ||
        error("unexpected Eight Schools primal benchmark receipt schema")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("Eight Schools receipt was produced from a dirty RK tree")
    occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")) ||
        error("Eight Schools receipt lacks an exact ReactiveKernels SHA")

    protocol = receipt["protocol"]
    Tuple(String.(protocol["input_boundaries"])) == _EIGHT_SCHOOLS_BOUNDARIES ||
        error("Eight Schools input-boundary inventory changed")
    Tuple(String.(protocol["outcomes"])) == _EIGHT_SCHOOLS_OUTCOMES ||
        error("Eight Schools outcome inventory changed")
    Int(get(protocol, "rounds", 0)) >= 10 ||
        error("Eight Schools receipt has fewer than ten timing rounds")
    get(protocol, "setup_in_timed_region", true) == false ||
        error("Eight Schools setup entered the timed region")
    get(protocol, "preparation_in_timed_region", true) == false ||
        error("Eight Schools preparation entered the timed region")
    get(protocol, "gradients_included", true) == false ||
        error("gradient measurements do not belong in the primal receipt")
    get(protocol, "generated_predictions_included", true) == false ||
        error("generated predictions do not belong in the primal receipt")
    get(protocol, "unsupported_cells_omitted", false) == true ||
        error("unsupported Eight Schools cells must remain omitted")

    measurements = receipt["measurements"]
    length(measurements) ==
        length(_EIGHT_SCHOOLS_BOUNDARIES) * length(_EIGHT_SCHOOLS_OUTCOMES) ||
        error("Eight Schools receipt is not a complete 3×4 matrix")
    indexed = Dict((String(row["boundary"]), String(row["outcome"])) => row
                   for row in measurements)
    length(indexed) == length(measurements) ||
        error("Eight Schools receipt contains duplicate matrix rows")

    boundary_labels = Dict(
        "packed_unconstrained" => "Packed unconstrained",
        "constrained_parameters" => "Constrained parameters",
        "minimal_likelihood" => "Likelihood inputs only",
    )
    outcome_labels = Dict(
        "joint" => "Joint",
        "prior" => "Prior",
        "likelihood" => "Likelihood",
        "pointwise" => "Pointwise likelihood",
    )
    rows = NamedTuple[]
    for boundary in _EIGHT_SCHOOLS_BOUNDARIES, outcome in _EIGHT_SCHOOLS_OUTCOMES
        row = get(indexed, (boundary, outcome), nothing)
        isnothing(row) && error("missing Eight Schools row: $boundary / $outcome")
        for backend in _EIGHT_SCHOOLS_BACKENDS
            haskey(row, backend) == _eight_schools_supported(boundary, outcome, backend) ||
                error("unexpected support for $boundary / $outcome / $backend")
            haskey(row, backend) || continue
            result = row[backend]
            length(result["times_ns"]) >= 10 ||
                error("insufficient timing rounds for $boundary / $outcome / $backend")
        end
        push!(rows, (;
            boundary = boundary_labels[boundary],
            outcome = outcome_labels[outcome],
            rk_native = _eight_schools_measurement(row, "rk_native"),
            manual_julia = _eight_schools_measurement(row, "manual_julia"),
            turing_native = _eight_schools_measurement(row, "turing_native"),
        ))
    end

    turing_rows = filter(row -> row.turing_native !== missing, rows)
    rk_faster_than_turing = count(
        row -> row.rk_native.median_ns < row.turing_native.median_ns,
        turing_rows,
    )
    turing_ratios = [
        row.turing_native.median_ns / row.rk_native.median_ns for row in turing_rows
    ]
    manual_rows = filter(row -> row.manual_julia !== missing, rows)
    manual_ratios = [
        row.manual_julia.median_ns / row.rk_native.median_ns for row in manual_rows
    ]
    summary = "RK is faster than Turing in $rk_faster_than_turing/" *
        "$(length(turing_rows)) matched native cells; Turing/RK runtime ranges from " *
        "$(round(minimum(turing_ratios); digits = 2))× to " *
        "$(round(maximum(turing_ratios); digits = 2))×. Against the handwritten " *
        "Julia control, manual/RK ranges from " *
        "$(round(minimum(manual_ratios); digits = 2))× to " *
        "$(round(maximum(manual_ratios); digits = 2))× (1× is parity)."
    columns = (
        _column(:boundary, "Starting boundary"),
        _column(:outcome, "Requested outcome"),
        _column(:rk_native, "RK native";
            format = _eight_schools_cell, sort = _eight_schools_sort),
        _column(:manual_julia, "Manual Julia";
            format = _eight_schools_cell, sort = _eight_schools_sort),
        _column(:turing_native, "Turing native";
            format = _eight_schools_cell, sort = _eight_schools_sort),
    )
    provenance = "Receipt pin: ReactiveKernels `$(pins["reactivekernels_sha"])`; " *
        "Julia $(pins["julia_version"]); Turing $(pins["turing_version"]); " *
        "DynamicPPL $(pins["dynamicppl_version"]); " *
        "$(receipt["environment"]["cpu"]). Raw rounds are retained in " *
        "benchmark/receipts/eight-schools-primal-v1.toml."

    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _result_table(rows, columns;
            id = "eight-schools-primal-matrix",
            title = "Primal boundary × outcome matrix",
            note = "Each measured cell is median runtime; bytes; allocations. " *
                   "A blank cell means that backend has no matching public boundary."),
        Markdown.Paragraph(Any[Markdown.Italic(Any[provenance])]),
    ])
end

const _MNIST_LOGISTIC_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "mnist-logistic-primal-v3.toml")
const _MNIST_LOGISTIC_WREN_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts",
    "mnist-logistic-wren-pca40-v1.toml")
const _MNIST_LOGISTIC_BOUNDARIES = ("packed_unconstrained", "structured_parameters")
const _MNIST_LOGISTIC_OUTCOMES = ("joint", "prior", "likelihood", "pointwise")
const _MNIST_LOGISTIC_V1_BACKENDS =
    ("rk_native", "manual_julia", "turing_native")

# The model's likelihood is a single `@addlogprob!` term, so Turing exposes no
# public per-observation pointwise view; that cell is omitted, not synthesized.
_mnist_logistic_supported(boundary, outcome, backend) =
    !(backend == "turing_native" && outcome == "pointwise")

# Headline estimator for the ms-scale matmul cells: the minimum of per-round
# minimums (uncontended cost). A shared-host load episode spanning over half
# the rounds corrupts a median; the receipt retains medians and raw rounds.
function _mnist_logistic_measurement(row, backend)
    haskey(row, backend) || return missing
    result = row[backend]
    (;
        median_ns = Float64(result["min_ns"]),
        median_bytes = Int(result["median_bytes"]),
        median_allocs = Int(result["median_allocs"]),
    )
end

function _mnist_logistic_profile(protocol)
    profile = String(get(protocol, "dataset_profile", "full-raw"))
    if profile == "full-raw"
        Int(get(protocol, "num_features", 0)) == 784 ||
            error("full-raw MNIST receipt must use 784 features")
        return (; profile, label = "Full raw MNIST", id_suffix = "")
    end
    profile == "wren-pca40" || error("unknown MNIST dataset profile: $profile")
    Int(get(protocol, "num_observations", 0)) == 1000 ||
        error("Wren-compatible MNIST receipt must use 1000 observations")
    Int(get(protocol, "num_features", 0)) == 40 ||
        error("Wren-compatible MNIST receipt must use 40 PCA features")
    Int(get(protocol, "pca_fit_observations", 0)) == 60000 ||
        error("Wren-compatible PCA must be fitted on all 60000 training images")
    get(protocol, "pca_centered", false) == true ||
        error("Wren-compatible PCA must center the training matrix")
    get(protocol, "pca_whitened", true) == false ||
        error("Wren-compatible PCA must remain unwhitened")
    get(protocol, "wren_reference_checked", false) == true ||
        error("Wren-compatible receipt must verify the copied reference CSV")
    (; profile, label = "Wren PCA-40", id_suffix = "-wren-pca40")
end

"""Render the checked-in MNIST multinomial-logistic boundary/outcome matrix."""
function render_mnist_logistic_benchmarks()
    _render_mnist_longform_focused(
        _MNIST_LOGISTIC_RECEIPT_PATH, "mnist-logistic-primal-v3";
        mode = :primal,
        id_prefix = "mnist-logistic-full",
        profile_label = "Full raw MNIST primal")
end

function _render_mnist_logistic_v1_benchmarks(receipt_path::AbstractString)
    return _render_mnist_v1_focused(receipt_path;
        mode = :primal,
        id_prefix = "mnist-logistic-wren-pca40",
        profile_label = "Wren PCA-40 primal")
    receipt = TOML.parsefile(receipt_path)
    get(receipt, "schema", "") == "mnist-logistic-v1" ||
        error("unexpected MNIST logistic benchmark receipt schema")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("MNIST logistic receipt was produced from a dirty RK tree")
    occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")) ||
        error("MNIST logistic receipt lacks an exact ReactiveKernels SHA")

    protocol = receipt["protocol"]
    Tuple(String.(protocol["input_boundaries"])) == _MNIST_LOGISTIC_BOUNDARIES ||
        error("MNIST logistic input-boundary inventory changed")
    Tuple(String.(protocol["outcomes"])) == _MNIST_LOGISTIC_OUTCOMES ||
        error("MNIST logistic outcome inventory changed")
    Int(get(protocol, "rounds", 0)) >= 10 ||
        error("MNIST logistic receipt has fewer than ten timing rounds")
    profile = _mnist_logistic_profile(protocol)
    get(protocol, "setup_in_timed_region", true) == false ||
        error("MNIST logistic setup entered the timed region")
    get(protocol, "preparation_in_timed_region", true) == false ||
        error("MNIST logistic preparation entered the timed region")
    get(protocol, "turing_pointwise_supported", true) == false ||
        error("Turing has no public pointwise view for the @addlogprob! likelihood")

    measurements = receipt["measurements"]
    length(measurements) ==
        length(_MNIST_LOGISTIC_BOUNDARIES) * length(_MNIST_LOGISTIC_OUTCOMES) ||
        error("MNIST logistic receipt is not a complete 2×4 matrix")
    indexed = Dict((String(row["boundary"]), String(row["outcome"])) => row
                   for row in measurements)

    boundary_labels = Dict(
        "packed_unconstrained" => "Packed unconstrained",
        "structured_parameters" => "Structured (W, b)")
    outcome_labels = Dict(
        "joint" => "Joint", "prior" => "Prior",
        "likelihood" => "Likelihood", "pointwise" => "Pointwise likelihood")
    rows = NamedTuple[]
    for boundary in _MNIST_LOGISTIC_BOUNDARIES, outcome in _MNIST_LOGISTIC_OUTCOMES
        row = get(indexed, (boundary, outcome), nothing)
        isnothing(row) && error("missing MNIST logistic row: $boundary / $outcome")
        for backend in _MNIST_LOGISTIC_V1_BACKENDS
            haskey(row, backend) == _mnist_logistic_supported(boundary, outcome, backend) ||
                error("unexpected support for $boundary / $outcome / $backend")
            haskey(row, backend) || continue
            length(row[backend]["times_ns"]) >= 10 ||
                error("insufficient timing rounds for $boundary / $outcome / $backend")
        end
        push!(rows, (;
            boundary = boundary_labels[boundary],
            outcome = outcome_labels[outcome],
            rk_native = _mnist_logistic_measurement(row, "rk_native"),
            manual_julia = _mnist_logistic_measurement(row, "manual_julia"),
            turing_native = _mnist_logistic_measurement(row, "turing_native")))
    end

    turing_rows = filter(row -> row.turing_native !== missing, rows)
    rk_faster_than_turing = count(
        row -> row.rk_native.median_ns < row.turing_native.median_ns, turing_rows)
    turing_ratios =
        [row.turing_native.median_ns / row.rk_native.median_ns for row in turing_rows]
    manual_ratios =
        [row.manual_julia.median_ns / row.rk_native.median_ns for row in rows]
    pointwise_only = length(rows) - length(turing_rows)
    summary = "$(profile.label). RK is faster than Turing in $rk_faster_than_turing/" *
        "$(length(turing_rows)) matched native cells; Turing/RK runtime ranges from " *
        "$(round(minimum(turing_ratios); digits = 2))× to " *
        "$(round(maximum(turing_ratios); digits = 2))×. Against the handwritten Julia " *
        "control, manual/RK ranges from $(round(minimum(manual_ratios); digits = 2))× " *
        "to $(round(maximum(manual_ratios); digits = 2))× (1× is parity). RK also " *
        "exposes per-observation pointwise log likelihoods in $pointwise_only cells " *
        "where Turing's single `@addlogprob!` term has no public view."
    columns = (
        _column(:boundary, "Starting boundary"),
        _column(:outcome, "Requested outcome"),
        _column(:rk_native, "RK native";
            format = _eight_schools_cell, sort = _eight_schools_sort),
        _column(:manual_julia, "Manual Julia";
            format = _eight_schools_cell, sort = _eight_schools_sort),
        _column(:turing_native, "Turing native";
            format = _eight_schools_cell, sort = _eight_schools_sort))
    provenance = "Receipt pin: ReactiveKernels `$(pins["reactivekernels_sha"])`; " *
        "Julia $(pins["julia_version"]); Turing $(pins["turing_version"]); " *
        "DynamicPPL $(pins["dynamicppl_version"]); MLDatasets $(pins["mldatasets_version"]); " *
        "$(receipt["environment"]["cpu"]). $(protocol["num_observations"]) MNIST images × " *
        "$(protocol["num_features"]) features × $(protocol["num_classes"]) classes. " *
        "Raw rounds are retained in benchmark/receipts/$(basename(receipt_path))."

    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _result_table(rows, columns;
            id = "mnist-logistic-matrix$(profile.id_suffix)",
            title = "$(profile.label): MNIST logistic boundary × outcome matrix",
            note = "Each measured cell is minimum (uncontended) runtime; bytes; " *
                   "allocations. " *
                   "A blank cell means that backend has no matching public boundary."),
        Markdown.Paragraph(Any[Markdown.Italic(Any[provenance])]),
    ])
end


render_mnist_logistic_wren_benchmarks() =
    _render_mnist_logistic_v1_benchmarks(_MNIST_LOGISTIC_WREN_RECEIPT_PATH)

const _MNIST_LOGISTIC_AD_IMPLEMENTATIONS = (
    ("rk_native", "ReactiveKernels"),
    ("manual_enzyme", "Manual Julia control"),
    ("turing_enzyme", "Turing / DynamicPPL"),
)
const _MNIST_LOGISTIC_AD_SUPPORTED = Set((
    ("packed_unconstrained", "joint"),
    ("packed_unconstrained", "prior"),
    ("packed_unconstrained", "likelihood"),
))

function _mnist_logistic_ad_plot(rows, metric;
                                 id::AbstractString, title::AbstractString)
    label = metric === :min_ns ?
        "Minimum uncontended value + gradient runtime (ns)" :
        "Median allocated bytes"
    scale = metric === :min_ns ? log10 : sqrt
    zero = metric !== :min_ns
    cell_order = unique(getproperty.(rows, :plot_cell))
    spec = data(rows) *
        mapping(
            :plot_cell => sorter(cell_order) => "Packed scalar outcome",
            metric => label;
            color = :implementation => "Implementation",
            dodge_x = :implementation,
        ) *
        visual(Scatter, markersize = 100) *
        config(height = 520, scales = scales(Y = (; scale, zero)))
    description = metric === :min_ns ?
        "Preparation and first execution are excluded; the runtime axis is logarithmic." :
        "The square-root byte axis preserves the genuine zero-allocation prior paths."
    _plot_block(spec; id, title, description)
end

function _mnist_logistic_ad_cell(value, _)
    value === missing && return ""
    "$(_mnist_logistic_ad_time(value.min_ns)); $(value.median_bytes) B; " *
    "$(value.median_allocs) alloc"
end

_mnist_logistic_ad_sort(value, _) =
    value === missing ? nothing : value.min_ns

_mnist_logistic_ad_time(value) = value >= 1e6 ?
    string(round(value / 1e6; digits = 2), " ms") : _fmt_ns(value)

function _mnist_logistic_ad_measurement(row, backend)
    haskey(row, backend) || return missing
    result = row[backend]
    (;
        min_ns = Float64(result["min_ns"]),
        median_bytes = Int(result["median_bytes"]),
        median_allocs = Int(result["median_allocs"]),
    )
end

"""Render the exact full-data MNIST value-and-gradient benchmark matrix."""
function render_mnist_logistic_ad_benchmarks()
    _render_mnist_longform_focused(
        _MNIST_LOGISTIC_AD_RECEIPT_PATH, "mnist-logistic-ad-v2";
        mode = :ad,
        id_prefix = "mnist-logistic-ad-full",
        profile_label = "Full raw MNIST value + gradient")
end

"""Render a legacy v1 MNIST value-and-gradient benchmark matrix."""
function _render_mnist_logistic_ad_v1_benchmarks(receipt_path::AbstractString)
    return _render_mnist_v1_focused(receipt_path;
        mode = :ad,
        id_prefix = "mnist-logistic-ad-wren-pca40",
        profile_label = "Wren PCA-40 value + gradient")
    receipt = TOML.parsefile(receipt_path)
    get(receipt, "schema", "") == "mnist-logistic-ad-v1" ||
        error("unexpected MNIST logistic AD receipt schema")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("MNIST logistic AD receipt was produced from a dirty tree")
    occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")) ||
        error("MNIST logistic AD receipt lacks an exact candidate SHA")
    model_source = pins["model_source"]
    get(model_source, "path", "") ==
        "packages/ReactiveKernelsPPLExamples/src/mnist_logistic.jl" ||
        error("MNIST logistic AD model authority drifted")
    occursin(r"^[0-9a-f]{40}$", get(model_source, "commit", "")) ||
        error("MNIST logistic AD model authority lacks a published SHA")

    protocol = receipt["protocol"]
    profile = _mnist_logistic_profile(protocol)
    Tuple(String.(protocol["input_boundaries"])) == _MNIST_LOGISTIC_BOUNDARIES ||
        error("MNIST logistic AD input-boundary inventory changed")
    Tuple(String.(protocol["outcomes"])) == _MNIST_LOGISTIC_OUTCOMES ||
        error("MNIST logistic AD outcome inventory changed")
    expected_active_parameter_count =
        9 * Int(get(protocol, "num_features", 0)) + 9
    Int(get(protocol, "active_parameter_count", 0)) ==
        expected_active_parameter_count ||
        error("MNIST logistic AD active-vector length drifted")
    get(protocol, "pointwise_jacobian_or_vjp_invented", true) == false ||
        error("MNIST logistic AD receipt invented a pointwise Jacobian/VJP")
    get(protocol, "structured_multi_active_boundary_invented", true) == false ||
        error("MNIST logistic AD receipt invented a two-active-port boundary")
    get(protocol, "preparation_in_timed_region", true) == false ||
        error("MNIST logistic AD steady-state timing includes preparation")
    get(protocol, "first_execution_in_steady_state_region", true) == false ||
        error("MNIST logistic AD steady-state timing includes first execution")
    Int(get(protocol, "rounds", 0)) >= 10 ||
        error("MNIST logistic AD receipt lacks ten raw rounds")

    measurements = receipt["measurements"]
    length(measurements) ==
        length(_MNIST_LOGISTIC_BOUNDARIES) * length(_MNIST_LOGISTIC_OUTCOMES) ||
        error("MNIST logistic AD receipt is not a complete 2×4 matrix")
    indexed = Dict((String(row["boundary"]), String(row["outcome"])) => row
                   for row in measurements)
    length(indexed) == length(measurements) ||
        error("MNIST logistic AD receipt contains duplicate matrix rows")

    boundary_labels = Dict(
        "packed_unconstrained" => "Packed unconstrained",
        "structured_parameters" => "Structured (W, b)",
    )
    outcome_labels = Dict(
        "joint" => "Joint",
        "prior" => "Prior",
        "likelihood" => "Likelihood",
        "pointwise" => "Pointwise likelihood",
    )
    matrix_rows = NamedTuple[]
    plot_rows = NamedTuple[]
    setup_rows = NamedTuple[]
    for boundary in _MNIST_LOGISTIC_BOUNDARIES,
        outcome in _MNIST_LOGISTIC_OUTCOMES
        row = get(indexed, (boundary, outcome), nothing)
        isnothing(row) && error("missing MNIST logistic AD row: $boundary / $outcome")
        expected_support = (boundary, outcome) in _MNIST_LOGISTIC_AD_SUPPORTED
        get(row, "supported", false) == expected_support ||
            error("MNIST logistic AD support drifted for $boundary / $outcome")
        cell = string(boundary_labels[boundary], " / ", outcome_labels[outcome])
        cell_values = Dict{String,Any}()
        for (key, implementation) in _MNIST_LOGISTIC_AD_IMPLEMENTATIONS
            cell_values[key] = _mnist_logistic_ad_measurement(row, key)
            haskey(row, key) || continue
            result = row[key]
            length(result["times_ns"]) >= 10 ||
                error("insufficient MNIST logistic AD rounds for $cell / $implementation")
            Float64(result["min_ns"]) == minimum(Float64.(result["times_ns"])) ||
                error("MNIST logistic AD minimum drifted for $cell / $implementation")
            push!(plot_rows, (;
                cell, plot_cell = outcome_labels[outcome],
                implementation,
                min_ns = Float64(result["min_ns"]),
                median_bytes = Int(result["median_bytes"]),
            ))
            push!(setup_rows, (;
                cell,
                implementation,
                preparation_seconds = Float64(result["preparation_seconds"]),
                preparation_bytes = Int(result["preparation_bytes"]),
                first_execution_seconds = Float64(result["first_execution_seconds"]),
                first_execution_bytes = Int(result["first_execution_bytes"]),
            ))
        end
        push!(matrix_rows, (;
            boundary = boundary_labels[boundary],
            outcome = outcome_labels[outcome],
            active = expected_support ? String(row["active_port"]) : "",
            status = expected_support ? "measured" : String(row["unsupported_reason"]),
            rk_native = cell_values["rk_native"],
            manual_enzyme = cell_values["manual_enzyme"],
            turing_enzyme = cell_values["turing_enzyme"],
        ))
    end

    supported_rows = Dict(String(row["outcome"]) => row for row in measurements
                          if get(row, "supported", false))
    function outcome_summary(outcome)
        row = supported_rows[outcome]
        rk = row["rk_native"]
        manual = row["manual_enzyme"]
        turing = row["turing_enzyme"]
        "$(_mnist_logistic_ad_time(rk["min_ns"])) RK, " *
        "$(_mnist_logistic_ad_time(manual["min_ns"])) manual, " *
        "$(_mnist_logistic_ad_time(turing["min_ns"])) Turing"
    end
    prior = supported_rows["prior"]
    Int(prior["rk_native"]["median_bytes"]) == 0 ||
        error("MNIST logistic RK prior lost its zero-allocation path")
    Int(prior["manual_enzyme"]["median_bytes"]) == 0 ||
        error("MNIST logistic manual prior lost its zero-allocation path")
    maximum_error = maximum(
        Float64(row[key]["gradient_max_abs_error"])
        for row in values(supported_rows)
        for (key, _) in _MNIST_LOGISTIC_AD_IMPLEMENTATIONS)
    summary = "On $(profile.label), the packed joint takes " *
        outcome_summary("joint") * "; likelihood takes " *
        outcome_summary("likelihood") * "; and the pruned prior takes " *
        outcome_summary("prior") * ". RK and the manual control allocate zero " *
        "bytes for the prior; joint and likelihood honestly retain their dense " *
        "linear-predictor/reverse-pass storage. All nine measured gradients match " *
        "the independent analytic score (maximum absolute error " *
        "$(round(maximum_error; sigdigits = 3)))."

    matrix_columns = (
        _column(:boundary, "Starting boundary"),
        _column(:outcome, "Requested outcome"),
        _column(:active, "Active port";
            format = (value, _) -> isempty(value) ? "" : h.code(value)),
        _column(:rk_native, "RK + Enzyme";
            format = _mnist_logistic_ad_cell, sort = _mnist_logistic_ad_sort),
        _column(:manual_enzyme, "Manual + Enzyme";
            format = _mnist_logistic_ad_cell, sort = _mnist_logistic_ad_sort),
        _column(:turing_enzyme, "Turing + Enzyme";
            format = _mnist_logistic_ad_cell, sort = _mnist_logistic_ad_sort),
        _column(:status, "Status";
            format = (value, _) -> value == "measured" ? value :
                h.span(value; class = "rk-result-unsupported")),
    )
    setup_columns = (
        _column(:cell, "Packed scalar outcome"),
        _column(:implementation, "Implementation"),
        _column(:preparation_seconds, "Preparation";
            format = (value, _) -> string(round(value; sigdigits = 4), " s")),
        _column(:preparation_bytes, "Preparation bytes";
            format = (value, _) -> string(value, " B")),
        _column(:first_execution_seconds, "First execution";
            format = (value, _) -> string(round(value; sigdigits = 4), " s")),
        _column(:first_execution_bytes, "First-execution bytes";
            format = (value, _) -> string(value, " B")),
    )
    provenance = "Receipt pin: RK `$(pins["reactivekernels_sha"])`; published " *
        "model authority `$(model_source["commit"])`; " *
        "DifferentiationInterface $(pins["differentiationinterface_version"]); " *
        "Enzyme $(pins["enzyme_version"]); Turing $(pins["turing_version"]); " *
        "Julia $(pins["julia_version"]); $(receipt["environment"]["cpu"]). " *
        "$(protocol["num_observations"]) images × $(protocol["num_features"]) " *
        "features; $(protocol["active_parameter_count"]) active coefficients."

    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _mnist_logistic_ad_plot(plot_rows, :min_ns;
            id = "mnist-logistic-ad-runtime$(profile.id_suffix)",
            title = "$(profile.label): MNIST value-and-gradient runtime"),
        _mnist_logistic_ad_plot(plot_rows, :median_bytes;
            id = "mnist-logistic-ad-allocation$(profile.id_suffix)",
            title = "$(profile.label): MNIST value-and-gradient allocation"),
        _result_table(matrix_rows, matrix_columns;
            id = "mnist-logistic-ad-matrix$(profile.id_suffix)",
            title = "$(profile.label): MNIST AD boundary × outcome matrix",
            note = "Measured cells show minimum uncontended value-and-gradient " *
                   "runtime; median bytes; median allocations. Unsupported cells " *
                   "retain their exact public-boundary reason."),
        _result_table(setup_rows, setup_columns;
            id = "mnist-logistic-ad-setup$(profile.id_suffix)",
            title = "$(profile.label): preparation and first execution",
            note = "These one-time costs are retained separately and excluded " *
                   "from the steady-state plots and matrix."),
        Markdown.Paragraph(Any[Markdown.Italic(Any[provenance])]),
    ])
end


render_mnist_logistic_ad_wren_benchmarks() =
    _render_mnist_logistic_ad_v1_benchmarks(
        _MNIST_LOGISTIC_AD_WREN_RECEIPT_PATH)
# ---- benchmark baseline sources ------------------------------------------------
# The comparison pages must show the exact Turing / handwritten baselines the
# benchmark harnesses execute. The displayed code is extracted verbatim at
# docs-build time from the executed `_body.jl` file, between explicit
# `# DOCS-BASELINE-BEGIN: <name>` / `# DOCS-BASELINE-END: <name>` markers, so a
# docs-only copy cannot drift from what actually ran; a missing or empty region
# fails the warning-fatal build.

const _BENCHMARK_SOURCE_DIR = normpath(joinpath(@__DIR__, "..", "benchmark"))

function _benchmark_baseline_region(file::AbstractString, name::AbstractString)
    path = joinpath(_BENCHMARK_SOURCE_DIR, file)
    lines = readlines(path)
    begin_line = "# DOCS-BASELINE-BEGIN: $(name)"
    end_line = "# DOCS-BASELINE-END: $(name)"
    starts = findall(==(begin_line), lines)
    stops = findall(==(end_line), lines)
    (length(starts), length(stops)) == (1, 1) || error(
        "expected exactly one $(begin_line) / $(end_line) pair in $(path)")
    only(starts) < only(stops) || error(
        "baseline markers for $(name) are out of order in $(path)")
    region = strip(join(lines[(only(starts) + 1):(only(stops) - 1)], "\n"))
    isempty(region) && error("baseline region $(name) in $(path) is empty")
    region
end

function _render_benchmark_baselines(intro::AbstractString, file::AbstractString,
                                     sections::Tuple)
    blocks = Any[Markdown.Paragraph(Any[intro])]
    for (title, name) in sections
        push!(blocks, Markdown.Paragraph(Any[Markdown.Bold(Any[title])]))
        push!(blocks, Markdown.Code("julia", _benchmark_baseline_region(file, name)))
    end
    Markdown.MD(blocks)
end

"""Render the Turing and handwritten baselines the MNIST benchmark executes."""
render_mnist_logistic_baselines() = _render_benchmark_baselines(
    "These are the baseline implementations the comparison cells above " *
    "execute, shown verbatim from `benchmark/mnist_logistic_comparison_body.jl` " *
    "(extracted from the executed file at docs-build time).",
    "mnist_logistic_comparison_body.jl",
    (("Turing baseline", "turing"),
     ("Turing optimized baseline", "turing-optimized"),
     ("Handwritten Julia control", "manual")))

"""Render the Turing and handwritten baselines the primal Eight Schools benchmark executes."""
render_eight_schools_primal_baselines() = _render_benchmark_baselines(
    "These are the baseline implementations the comparison cells above " *
    "execute, shown verbatim from " *
    "`benchmark/eight_schools_primal_comparison_body.jl` (extracted from the " *
    "executed file at docs-build time).",
    "eight_schools_primal_comparison_body.jl",
    (("Turing baseline", "turing"),
     ("Handwritten Julia control", "manual")))

"""Render the exact Turing and handwritten sum-to-zero benchmark controls."""
render_sum_to_zero_baselines() = _render_benchmark_baselines(
    "These are the model implementations used by the sum-to-zero comparison, " *
    "shown verbatim from `benchmark/sum_to_zero_comparison_body.jl`. The " *
    "native AD cells differentiate these same definitions; recovery remains " *
    "outside every timed sampler hot loop.",
    "sum_to_zero_comparison_body.jl",
    (("Turing baseline", "turing"),
     ("Optimized handwritten Julia control", "manual")))

"""Render the Turing baseline the evaluation-throughput benchmark executes."""
render_eval_throughput_baselines() = _render_benchmark_baselines(
    "This is the Turing baseline the comparison above executes, shown verbatim " *
    "from `benchmark/eval_throughput_comparison_body.jl` (extracted from the " *
    "executed file at docs-build time).",
    "eval_throughput_comparison_body.jl",
    (("Turing baseline", "turing"),))

"""Render the baselines the MNIST AD comparison differentiates."""
render_mnist_logistic_ad_baselines() = _render_benchmark_baselines(
    "The AD comparison differentiates the same baseline implementations the " *
    "primal MNIST benchmark executes (the AD harness includes " *
    "`mnist_logistic_comparison_body.jl` for the exact definitions), shown " *
    "verbatim from that file at docs-build time; the AD cells differentiate " *
    "these definitions with Enzyme through DifferentiationInterface " *
    "`AutoEnzyme` backends.",
    "mnist_logistic_comparison_body.jl",
    (("Turing baseline", "turing"),
     ("Handwritten Julia control", "manual")))

"""Render the baselines the Eight Schools AD comparison differentiates."""
render_eight_schools_ad_baselines() = _render_benchmark_baselines(
    "The AD comparison differentiates the same baseline implementations the " *
    "primal Eight Schools benchmark executes (the AD harness includes " *
    "`eight_schools_primal_comparison_body.jl`), shown verbatim from that file " *
    "at docs-build time; the AD cells differentiate these definitions with " *
    "Enzyme through DifferentiationInterface `AutoEnzyme` backends.",
    "eight_schools_primal_comparison_body.jl",
    (("Turing baseline", "turing"),
     ("Handwritten Julia control", "manual")))

const _EIGHT_SCHOOLS_AD_IMPLEMENTATIONS = (
    ("rk_native", "ReactiveKernels"),
    ("manual_enzyme", "Manual Julia control"),
    ("turing_enzyme", "Turing / DynamicPPL"),
)
const _EIGHT_SCHOOLS_AD_SUPPORTED = Set((
    ("packed_unconstrained", "joint"),
    ("packed_unconstrained", "prior"),
    ("packed_unconstrained", "likelihood"),
    ("packed_unconstrained", "pointwise"),
    ("constrained_parameters", "joint"),
    ("constrained_parameters", "prior"),
    ("constrained_parameters", "likelihood"),
    ("minimal_likelihood", "likelihood"),
    ("minimal_likelihood", "pointwise"),
))

function _eight_schools_ad_plot(rows, metric;
                                id::AbstractString, title::AbstractString)
    label = metric === :median_ns ? "Median reverse-AD runtime (ns)" :
        "Median allocated bytes"
    scale = metric === :median_ns ? log10 : sqrt
    zero = metric !== :median_ns
    cell_order = unique(getproperty.(rows, :plot_cell))
    spec = data(rows) *
        mapping(
            :plot_cell => sorter(cell_order) => "Reverse-AD matrix cell",
            metric => label;
            color = :implementation => "Implementation",
            dodge_x = :implementation,
        ) *
        visual(Scatter, markersize = 100) *
        config(height = 520, scales = scales(Y = (; scale, zero)))
    description = (metric === :median_ns ?
        "Preparation and first execution are excluded; the runtime axis is logarithmic." :
        "The square-root byte axis retains the genuine zero-allocation scalar RK measurements.") *
        " Cell labels abbreviate packed unconstrained (PU), constrained parameters (CP), " *
        "and likelihood-only (LO)."
    _plot_block(spec; id, title, description)
end

function _eight_schools_ad_cell(value, _)
    value === missing && return ""
    "$(_fmt_ns(value.median_ns)); $(value.median_bytes) B; " *
    "$(value.median_allocs) alloc"
end

_eight_schools_ad_sort(value, _) =
    value === missing ? nothing : value.median_ns

"""Render focused Eight Schools AD comparisons from the complete receipt."""
function render_eight_schools_ad_benchmarks()
    return _render_eight_schools_longform_focused(
        _EIGHT_SCHOOLS_AD_RECEIPT_PATH, "eight-schools-ad-v2";
        mode = :ad, id_prefix = "eight-schools-ad",
        profile_label = "Eight Schools value + gradient")
    receipt = TOML.parsefile(_EIGHT_SCHOOLS_AD_RECEIPT_PATH)
    get(receipt, "schema", "") == "eight-schools-ad-v1" ||
        error("unexpected Eight Schools AD receipt schema")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("Eight Schools AD receipt was produced from a dirty tree")
    occursin(r"^[0-9a-f]{40}$", get(pins, "reactivekernels_sha", "")) ||
        error("Eight Schools AD receipt lacks an exact candidate SHA")
    model_source = pins["model_source"]
    get(model_source, "path", "") ==
        "packages/ReactiveKernelsPPLExamples/src/eight_schools.jl" ||
        error("Eight Schools AD model authority drifted")
    occursin(r"^[0-9a-f]{40}$", get(model_source, "commit", "")) ||
        error("Eight Schools AD model authority lacks a published SHA")

    protocol = receipt["protocol"]
    Tuple(String.(protocol["input_boundaries"])) == _EIGHT_SCHOOLS_BOUNDARIES ||
        error("Eight Schools AD input-boundary inventory changed")
    Tuple(String.(protocol["outcomes"])) == _EIGHT_SCHOOLS_OUTCOMES ||
        error("Eight Schools AD outcome inventory changed")
    get(protocol, "pointwise_jacobian_or_vjp_invented", true) == false ||
        error("Eight Schools AD receipt invented a pointwise Jacobian/VJP")
    get(protocol, "preparation_in_timed_region", true) == false ||
        error("Eight Schools AD steady-state timing includes preparation")
    get(protocol, "first_execution_in_steady_state_region", true) == false ||
        error("Eight Schools AD steady-state timing includes first execution")
    Int(get(protocol, "rounds", 0)) >= 10 ||
        error("Eight Schools AD receipt lacks ten raw rounds")

    measurements = receipt["measurements"]
    length(measurements) ==
        length(_EIGHT_SCHOOLS_BOUNDARIES) * length(_EIGHT_SCHOOLS_OUTCOMES) ||
        error("Eight Schools AD receipt is not a complete 3×4 matrix")
    indexed = Dict((String(row["boundary"]), String(row["outcome"])) => row
                   for row in measurements)
    length(indexed) == length(measurements) ||
        error("Eight Schools AD receipt contains duplicate matrix rows")

    boundary_labels = Dict(
        "packed_unconstrained" => "Packed unconstrained",
        "constrained_parameters" => "Constrained parameters",
        "minimal_likelihood" => "Likelihood inputs only",
    )
    outcome_labels = Dict(
        "joint" => "Joint",
        "prior" => "Prior",
        "likelihood" => "Likelihood",
        "pointwise" => "Pointwise likelihood",
    )
    plot_boundary_labels = Dict(
        "packed_unconstrained" => "PU",
        "constrained_parameters" => "CP",
        "minimal_likelihood" => "LO",
    )
    plot_outcome_labels = Dict(
        "joint" => "Joint",
        "prior" => "Prior",
        "likelihood" => "Likelihood",
        "pointwise" => "Pointwise",
    )
    matrix_rows = NamedTuple[]
    plot_rows = NamedTuple[]
    setup_rows = NamedTuple[]
    for boundary in _EIGHT_SCHOOLS_BOUNDARIES,
        outcome in _EIGHT_SCHOOLS_OUTCOMES
        row = get(indexed, (boundary, outcome), nothing)
        isnothing(row) && error("missing Eight Schools AD row: $boundary / $outcome")
        expected_support = (boundary, outcome) in _EIGHT_SCHOOLS_AD_SUPPORTED
        get(row, "supported", false) == expected_support ||
            error("Eight Schools AD support drifted for $boundary / $outcome")
        cell = string(boundary_labels[boundary], " / ", outcome_labels[outcome])
        values = Dict{String,Any}()
        for (key, implementation) in _EIGHT_SCHOOLS_AD_IMPLEMENTATIONS
            values[key] = haskey(row, key) ? _eight_schools_measurement(row, key) : missing
            haskey(row, key) || continue
            result = row[key]
            length(result["times_ns"]) >= 10 ||
                error("insufficient Eight Schools AD rounds for $cell / $implementation")
            push!(plot_rows, (;
                cell,
                plot_cell = string(
                    plot_boundary_labels[boundary], " · ", plot_outcome_labels[outcome],
                ),
                implementation,
                median_ns = Float64(result["median_ns"]),
                median_bytes = Int(result["median_bytes"]),
            ))
            push!(setup_rows, (;
                cell,
                implementation,
                preparation_seconds = Float64(result["preparation_seconds"]),
                preparation_bytes = Int(result["preparation_bytes"]),
                first_execution_seconds =
                    Float64(result["first_execution_seconds"]),
                first_execution_bytes = Int(result["first_execution_bytes"]),
            ))
        end
        push!(matrix_rows, (;
            boundary = boundary_labels[boundary],
            outcome = outcome_labels[outcome],
            active = expected_support ? String(row["active_port"]) : "",
            status = expected_support ? "measured" : String(row["unsupported_reason"]),
            rk_native = values["rk_native"],
            manual_enzyme = values["manual_enzyme"],
            turing_enzyme = values["turing_enzyme"],
        ))
    end

    rk_rows = [row["rk_native"] for row in measurements if get(row, "supported", false)]
    zero_rk = count(row -> Int(row["median_bytes"]) == 0 &&
                           Int(row["median_allocs"]) == 0, rk_rows)
    zero_rk == 4 || error("Eight Schools AD receipt lost a zero-allocation RK cell")
    turing_rows = filter(row -> row.implementation == "Turing / DynamicPPL", plot_rows)
    turing_bytes = getproperty.(turing_rows, :median_bytes)
    summary = "The public RK boundary now measures seven scalar gradients and " *
        "two pointwise reverse pullbacks. Four array-backed scalar cells retain " *
        "a zero-byte, zero-allocation steady-state path; structured NamedTuple " *
        "sensitivities use DI's honest nonmutating result. The three matched " *
        "DynamicPPL scalar cells allocate $(minimum(turing_bytes))–" *
        "$(maximum(turing_bytes)) B. Only the structured-input × pointwise-output " *
        "cross-product remains an Enzyme MixedDuplicated limitation."

    matrix_columns = (
        _column(:boundary, "Starting boundary"),
        _column(:outcome, "Requested outcome"),
        _column(:active, "Active port";
            format = (value, _) -> isempty(value) ? "" : h.code(value)),
        _column(:rk_native, "RK + Enzyme";
            format = _eight_schools_ad_cell, sort = _eight_schools_ad_sort),
        _column(:manual_enzyme, "Manual + Enzyme";
            format = _eight_schools_ad_cell, sort = _eight_schools_ad_sort),
        _column(:turing_enzyme, "Turing + Enzyme";
            format = _eight_schools_ad_cell, sort = _eight_schools_ad_sort),
        _column(:status, "Status";
            format = (value, _) -> value == "measured" ? value :
                h.span(value; class = "rk-result-unsupported")),
    )
    setup_columns = (
        _column(:cell, "Reverse-AD matrix cell"),
        _column(:implementation, "Implementation"),
        _column(:preparation_seconds, "Preparation";
            format = (value, _) -> string(round(value; sigdigits = 4), " s")),
        _column(:preparation_bytes, "Preparation bytes";
            format = (value, _) -> string(value, " B")),
        _column(:first_execution_seconds, "First execution";
            format = (value, _) -> string(round(value; sigdigits = 4), " s")),
        _column(:first_execution_bytes, "First-execution bytes";
            format = (value, _) -> string(value, " B")),
    )
    provenance = "Receipt pin: RK `$(pins["reactivekernels_sha"])`; published " *
        "model authority `$(model_source["commit"])`; " *
        "DifferentiationInterface $(pins["differentiationinterface_version"]); " *
        "Enzyme $(pins["enzyme_version"]); Turing $(pins["turing_version"]); " *
        "Julia $(pins["julia_version"]); $(receipt["environment"]["cpu"])."

    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _eight_schools_ad_plot(plot_rows, :median_ns;
            id = "eight-schools-ad-runtime",
            title = "Eight Schools reverse-AD runtime"),
        _eight_schools_ad_plot(plot_rows, :median_bytes;
            id = "eight-schools-ad-allocation",
            title = "Eight Schools reverse-AD allocation"),
        _result_table(matrix_rows, matrix_columns;
            id = "eight-schools-ad-matrix",
            title = "AD boundary × outcome matrix",
            note = "Scalar cells measure value-and-gradient; pointwise cells " *
                   "measure value-and-VJP for the receipt's fixed output " *
                   "cotangent. Unsupported cells retain their exact reason."),
        _result_table(setup_rows, setup_columns;
            id = "eight-schools-ad-setup",
            title = "Preparation, compilation, and first execution",
            note = "These one-time costs are retained separately and excluded " *
                   "from the steady-state plots and matrix."),
        Markdown.Paragraph(Any[Markdown.Italic(Any[provenance])]),
    ])
end

function _definition_grid(rows; class = "rk-definition-grid")
    children = Any[]
    for row in rows
        push!(children, h.dt(row.term), h.dd(row.description))
    end
    h.dl(children...; class)
end

"""Render the DAG appearance key as an accessible definition list."""
function render_visualization_legend()
    rows = [
        (term = "Green value", description = "HAVE boundary; supplied by the caller"),
        (term = "Orange value", description = "WANT boundary; returned by the kernel"),
        (term = "Grey recipe", description = "Registered recipe on a full Graph"),
        (term = "Blue recipe", description = "Selected computation"),
        (term = "Dashed grey recipe",
         description = "Reachable alternative that was not selected"),
        (term = "Dashed red recipe",
         description = "Effectful recipe, visible on a full Graph but never selectable"),
    ]
    Markdown.MD(Any[_static_block(_definition_grid(rows; class = "rk-dag-legend"))])
end

function _comparison_cards(rows; class = "rk-comparison-grid")
    cards = [
        h.article(; class = "rk-comparison-card")(
            h.h3(row.title),
            h.dl(
                h.dt(row.first_label), h.dd(row.first_value),
                h.dt(row.second_label), h.dd(row.second_value),
            ),
        )
        for row in rows
    ]
    h.div(cards...; class)
end

"""Render visualization output choices as comparison cards rather than a matrix."""
function render_visualization_surfaces()
    rows = [
        (title = "Interactive docs inspection", first_label = "Surface",
         first_value = "Cytoscape.js + ELK HTML", second_label = "Tradeoff",
         second_value = "Libraries are pinned and bundled into VitePress; fit, pan, zoom, focus, and structural inspection are available."),
        (title = "Interactive notebook or IDE inspection", first_label = "Surface",
         first_value = "Cytoscape.js + ELK HTML", second_label = "Tradeoff",
         second_value = "Pinned packages load on demand; the embedded SVG remains usable offline."),
        (title = "Files that render anywhere", first_label = "Surface",
         first_value = h.code("save_visualization(\"graph.svg\", x)"),
         second_label = "Tradeoff", second_value = "Static rather than interactive."),
        (title = "Standalone artifact", first_label = "Surface",
         first_value = h.code("save_visualization(\"graph.html\", x)"),
         second_label = "Tradeoff",
         second_value = "Opens directly with embedded SVG and loads pinned interactive libraries when online."),
        (title = "Large or publication-oriented layout", first_label = "Surface",
         first_value = h.code("dot_source or .dot / .gv export"),
         second_label = "Tradeoff",
         second_value = "A downstream Graphviz renderer is needed; ReactiveKernels does not impose its binary or artifact footprint on every user."),
    ]
    Markdown.MD(Any[_static_block(_comparison_cards(rows))])
end

function _status_cards(rows; class = "rk-status-grid")
    cards = [h.article(; class = "rk-status-card")(
        h.h3(row.title), row.body,
    ) for row in rows]
    h.div(cards...; class)
end

"""Render the NUTS evidence state as readable status cards."""
function render_nuts_status()
    rows = [
        (title = "Source contract",
         body = h.p(
             h.strong("Landed as external compiler evidence. "),
             h.code("pot_f"), " and ", h.code("grad_f"),
             " are alternative producers of ", h.code("pot"),
             "; the planner selects the needed recipe while retaining both read-only callable authorities by identity.",
         )),
        (title = "Helper effect authority",
         body = h.p(
             h.strong("Macro-free. "),
             "Hot helpers are visible arithmetic/control or captured sibling ", h.code("@kernel"),
             " methods; ordinary unregistered helpers reject. The former ", h.code("@rk_pure"),
             " / ", h.code("@rk_borrows"), " / ", h.code("@rk_rng"),
             " declarations no longer exist. ", h.code("@node"),
             " is unrelated and remains supported.",
         )),
        (title = "All eight source specifications",
         body = h.p(
             h.strong("Verified on main. "),
             "The compiler constructs and runs the complete external fixture; its sealed certificate records ",
             h.code("mode = production"),
             ", which describes compiler evidence rather than package API status.",
         )),
        (title = "Executable leapfrog (leaf scope)",
         body = h.p(
             h.strong("Verified. "),
             "Analytic F32/F64; normal gradient Δ1, ", h.code("@inferred"),
             ", exact 0 B; dirty-produced recovery analytic; dirty-source reject.",
         )),
        (title = "Sealed fixture nuts!!",
         body = h.p(
             h.strong("Verified external exemplar. "),
             "Sealed registry-free native recursion; ", h.code("nuts!!(state; rng) === state"),
             " (same object, fixed type), exact 0 B on the compiler acceptance path.",
         )),
        (title = "Compiled-reactive nuts_state / CompiledNUTSState",
         body = h.p(
             h.strong("External example only. "),
             "It uses compiled-reactive Hamiltonian dependencies with ordinary inferred Julia recursion and proposal scratch; it is neither loaded nor exported by ReactiveKernels and is not the sealed fixture artifact.",
         )),
        (title = "Adaptive transition through Reactant",
         body = h.p(
             h.strong("Compiled and measured. "),
             "The accepted full-depth transition lowers to one data-dependent ",
             h.code("stablehlo.while"),
             ". Its separate matched native/Reactant receipt reports synchronous CPU wall time, work-normalized leapfrog throughput, and compile latency without claiming a speedup.",
         )),
        (title = "End-to-end sampling time and ESS",
         body = h.p(
             h.strong("Not measured. "),
             "The native G7 receipt is work-normalized inner-loop throughput. The Reactant receipt measures matched full-transition execution but excludes adaptation, retained draws, transfers, and ESS; neither supports a time-to-effective-sample claim.",
         )),
    ]
    Markdown.MD(Any[_static_block(_status_cards(rows))])
end

"""Render the frozen G7 work-normalized NUTS receipt as one plot/table source."""
function render_nuts_g7_benchmark()
    return _render_nuts_g7_benchmark_focused()
    receipt = TOML.parsefile(_NUTS_G7_RECEIPT_PATH)
    get(receipt, "schema", "") == "nuts-g7-v1" ||
        error("unexpected NUTS G7 benchmark receipt schema")
    verdict = receipt["verdict"]
    get(verdict, "gate_clears", false) || error("NUTS G7 receipt does not clear its gate")
    get(verdict, "rk_work_exact", false) || error("NUTS G7 RK work is not exact")
    get(verdict, "nuts_work_exact", false) ||
        error("NUTS G7 NUTS.jl work is not exact")

    sampler_order = ("RK", "NUTS.jl", "DHMC", "AHMC")
    medians = receipt["medians"]
    work = receipt["work"]
    rk_median = Float64(medians["RK"])
    rows = [(;
        sampler,
        median_steps_per_s = Float64(medians[sampler]),
        grads = Int(work[sampler]["grads"]),
        steps = Int(work[sampler]["steps"]),
        rk_ratio = round(rk_median / Float64(medians[sampler]); digits = 2),
    ) for sampler in sampler_order]

    spec = data(rows) *
        mapping(
            :sampler => sorter(collect(sampler_order)) => "Sampler",
            :median_steps_per_s => "Median leapfrog steps/s",
        ) *
        visual(BarPlot) *
        config(height = 280, scales = scales(Y = (; zero = true)))
    columns = (
        _column(:sampler, "Sampler"),
        _column(:median_steps_per_s, "Median leapfrog steps/s"),
        _column(:grads, "Gradients"),
        _column(:steps, "Leapfrog steps"),
        _column(:rk_ratio, "RK ÷ sampler";
            format = (value, _) -> string(value, "×")),
    )

    env = receipt["env"]
    pins = receipt["pins"]
    summary = "RK is $(rows[4].rk_ratio)× AdvancedHMC and " *
        "$(rows[3].rk_ratio)× DynamicHMC throughput; it is " *
        "$(rows[2].rk_ratio)× the reported nsiccha/NUTS.jl reference."
    caption = "Frozen $(env["target"]), dimension $(env["dimension"]), " *
        "unit diagonal mass, step size $(env["stepsize"]), one shared " *
        "DifferentiationInterface + Enzyme gradient, and matched RNG. " *
        "This isolates framework overhead from the model gradient; no ESS or wall-time claim."
    provenance = h.p(; class = "rk-result-provenance")(
        "Source: ",
        h.a(h.code("benchmark/receipts/nuts-g7-v1.toml");
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/nuts-g7-v1.toml"),
        ". Pins: AdvancedHMC $(pins["advancedhmc"]), DynamicHMC $(pins["dynamichmc"]), " *
        "NUTS.jl $(first(String(pins["nuts_jl_sha"]), 7)), Julia $(pins["julia_version"]).",
    )

    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _plot_block(spec; id = "nuts-g7-throughput",
            title = "Work-normalized leapfrog throughput", description = caption),
        _result_table(rows, columns; id = "nuts-g7-table",
            title = "Exact G7 receipt values",
            note = "Select a column heading to sort. RK ÷ sampler is RK's median divided by that sampler's median."),
        _static_block(provenance),
    ])
end

"""Render the frozen matched native/Reactant adaptive-NUTS receipt."""
function render_nuts_reactant_benchmark()
    return _render_nuts_reactant_benchmark_focused()
    receipt = TOML.parsefile(_NUTS_REACTANT_RECEIPT_PATH)
    get(receipt, "schema", "") == "nuts-reactant-v1" ||
        error("unexpected Reactant NUTS benchmark receipt schema")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("Reactant NUTS receipt was produced from a dirty checkout")
    protocol = receipt["protocol"]
    get(protocol, "reactant_sync", false) ||
        error("Reactant NUTS receipt is not synchronous")
    get(protocol, "max_depth", 0) == 10 ||
        error("Reactant NUTS receipt does not exercise max_depth=10")
    acceptance = receipt["acceptance"]
    get(acceptance,
        "per_transition_tolerance_bounded_phase_diagnostic_parity", false) ||
        error("Reactant NUTS receipt lacks tolerance-bounded floating parity")
    get(acceptance, "exact_control_and_random_consumption_parity", false) ||
        error("Reactant NUTS receipt lacks exact control/random-consumption parity")
    get(acceptance, "matched_control_flow_corpus", false) ||
        error("Reactant NUTS receipt is not a matched-control corpus")
    get(acceptance, "parity_screening_reported", false) ||
        error("Reactant NUTS receipt does not report parity screening")
    get(acceptance, "all_overflow_flags_zero", false) ||
        error("Reactant NUTS receipt has a capacity overflow")
    get(acceptance, "stablehlo_while_count", 0) == 1 ||
        error("Reactant NUTS receipt does not contain exactly one stablehlo.while")

    medians = receipt["medians"]
    native_ms = Float64(medians["native_transition_ms"])
    reactant_ms = Float64(medians["reactant_transition_ms"])
    native_steps = Float64(medians["native_steps_per_second"])
    reactant_steps = Float64(medians["reactant_steps_per_second"])
    time_ratio = Float64(medians["reactant_over_native_transition_time"])
    throughput_ratio = Float64(medians["reactant_over_native_steps_per_second"])
    rows = [
        (backend = "Native source compiler", transition_ms = native_ms,
         steps_per_second = native_steps, relative_time = 1.0),
        (backend = "Reactant (CPU)", transition_ms = reactant_ms,
         steps_per_second = reactant_steps, relative_time = time_ratio),
    ]
    spec = data(rows) *
        mapping(:backend => "Backend", :transition_ms => "Median transition time (ms)") *
        visual(Scatter, markersize = 120) *
        config(height = 280, scales = scales(Y = (; scale = log10, zero = false)))
    columns = (
        _column(:backend, "Backend"),
        _column(:transition_ms, "Median transition";
            format = (value, _) -> string(round(value; sigdigits=4), " ms")),
        _column(:steps_per_second, "Leapfrog steps/s";
            format = (value, _) -> string(round(Int, value))),
        _column(:relative_time, "Time ÷ native";
            format = (value, _) -> string(round(value; digits=1), "×")),
    )

    compilation = receipt["compilation"]
    compile_rows = [
        (stage = "Native source compile + first Julia JIT",
         seconds = Float64(compilation["native_seconds"])),
        (stage = "Native first execution",
         seconds = Float64(compilation["native_first_execution_seconds"])),
        (stage = "Reactant host lowering",
         seconds = Float64(compilation["reactant_lower_seconds"])),
        (stage = "Reactant XLA compile",
         seconds = Float64(compilation["reactant_xla_seconds"])),
        (stage = "Reactant first synchronous execution",
         seconds = Float64(compilation["reactant_first_execution_seconds"])),
    ]
    compile_columns = (
        _column(:stage, "Stage"),
        _column(:seconds, "Elapsed";
            format = (value, _) -> string(round(value; sigdigits=4), " s")),
    )
    summary = "On this frozen CPU receipt, Reactant took " *
        "$(round(time_ratio; digits=1))× the native per-transition time " *
        "($(round(reactant_ms; sigdigits=4)) ms vs $(round(native_ms; sigdigits=4)) ms) " *
        "and delivered $(round(reactant_steps; sigdigits=4)) vs " *
        "$(round(native_steps; sigdigits=4)) leapfrog steps/s " *
        "($(round(throughput_ratio; sigdigits=4))× native). This is a measured slowdown, not a speedup claim or an inherent limit on batched/loop-amortized execution."
    caption = "Frozen $(protocol["target"]), $(protocol["metric"]), step size " *
        "$(protocol["stepsize"]), max depth $(protocol["max_depth"]), " *
        "$(protocol["rounds"]) rounds × $(protocol["transitions_per_round"]) independent full transitions from matched starting states. " *
        "Both arms consume identical pre-generated random bundles; $(protocol["candidate_bundles_rejected"]) of $(protocol["candidate_bundles_examined"]) deterministic candidates were excluded outside timing after backend-sensitive transition parity mismatches. Floating phase/diagnostic values use atol=$(protocol["floating_parity_absolute_tolerance"]) and rtol=$(protocol["floating_parity_relative_tolerance"]); controls and random consumption match exactly. Timings exclude compilation, corpus screening, state setup, transfers, RNG generation, rebundling, and readback."
    provenance = h.p(; class = "rk-result-provenance")(
        "Sources: ",
        h.a(h.code("benchmark/receipts/nuts-reactant-v1.toml");
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/nuts-reactant-v1.toml"),
        " generated by ",
        h.a(h.code("benchmark/nuts_reactant_comparison.jl");
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/nuts_reactant_comparison.jl"),
        ". Pins: Reactant $(pins["reactant_version"]), Reactant_jll $(pins["reactant_jll_version"]), Julia $(pins["julia_version"]), RK $(first(String(pins["reactivekernels_sha"]), 7)).",
    )

    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _plot_block(spec; id = "nuts-reactant-transition-time",
            title = "Matched adaptive-NUTS transition time", description = caption),
        _result_table(rows, columns; id = "nuts-reactant-table",
            title = "Receipt medians for the matched-control corpus",
            note = "This is a matched-control microbenchmark, not sampler throughput. The logarithmic plot keeps both measured arms visible. Lower transition time and higher leapfrog steps/s are better."),
        _result_table(compile_rows, compile_columns; id = "nuts-reactant-compile-table",
            title = "Compilation and first-call costs",
            note = "Compile order is native source compiler, Reactant lowering, then Reactant XLA. These values are reported separately and are not in steady-state timing."),
        _static_block(provenance),
    ])
end

"""Render focused native-RK/Reactant primal comparisons from the full receipt."""
function render_eight_schools_reactant_benchmark()
    return _render_eight_schools_reactant_longform_focused(
        _EIGHT_SCHOOLS_REACTANT_RECEIPT_PATH, "eight-schools-reactant-v2";
        mode = :primal, id_prefix = "eight-schools-reactant",
        profile_label = "Eight Schools Reactant primal")
    receipt = TOML.parsefile(_EIGHT_SCHOOLS_REACTANT_RECEIPT_PATH)
    get(receipt, "schema", "") == "eight-schools-reactant-v1" ||
        error("unexpected Eight Schools Reactant benchmark receipt schema")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("Eight Schools Reactant receipt was produced from a dirty checkout")
    protocol = receipt["protocol"]
    get(protocol, "source_reused", false) ||
        error("Eight Schools Reactant receipt does not reuse the authored source")
    get(protocol, "reactant_sync", false) ||
        error("Eight Schools Reactant receipt is not synchronous")
    get(protocol, "gradients_included", true) &&
        error("Eight Schools Reactant receipt unexpectedly contains gradients")
    Tuple(String.(protocol["input_boundaries"])) == _EIGHT_SCHOOLS_BOUNDARIES ||
        error("unexpected Eight Schools input-boundary matrix")
    Tuple(String.(protocol["outcomes"])) == _EIGHT_SCHOOLS_OUTCOMES ||
        error("unexpected Eight Schools output matrix")

    boundary_labels = Dict(
        "packed_unconstrained" => "Packed unconstrained",
        "constrained_parameters" => "Constrained parameters",
        "minimal_likelihood" => "Likelihood inputs only",
    )
    rows = map(receipt["measurements"]) do measurement
        native_supported = get(measurement, "rk_native_supported", false)
        reactant_supported = get(measurement, "rk_reactant_supported", false)
        native_ns = native_supported ?
            Float64(measurement["rk_native"]["median_ns"]) : missing
        reactant_ns = reactant_supported ?
            Float64(measurement["rk_reactant"]["median_ns"]) : missing
        diagnostic = reactant_supported ? "measured" :
            get(measurement, "rk_reactant_error", "unsupported")
        (;
            boundary = boundary_labels[measurement["boundary"]],
            outcome = measurement["outcome"],
            native_ns,
            reactant_ns,
            relative_time = reactant_supported ? reactant_ns / native_ns : missing,
            compiler_result = diagnostic,
        )
    end
    measured = filter(row -> row.reactant_ns !== missing, rows)
    ratios = Float64[row.relative_time for row in measured]
    unsupported = count(row -> row.native_ns !== missing && row.reactant_ns === missing, rows)
    summary = isempty(ratios) ?
        "No Reactant matrix cell compiled in this receipt." :
        "Reactant compiled $(length(measured)) of the 10 model-defined primal cells; " *
        "$unsupported retain their compiler diagnostics. Across compiled cells, " *
        "steady-state Reactant/native time ranges from " *
        "$(round(minimum(ratios); digits = 2))× to " *
        "$(round(maximum(ratios); digits = 2))× on this CPU receipt."
    timing_columns = (
        _column(:boundary, "Input boundary"),
        _column(:outcome, "Requested output"),
        _column(:native_ns, "Native RK";
            format = (value, _) -> _unsupported(value, _fmt_ns)),
        _column(:reactant_ns, "Reactant";
            format = (value, _) -> _unsupported(value, _fmt_ns)),
        _column(:relative_time, "Reactant ÷ native";
            format = (value, _) -> _unsupported(
                value, ratio -> string(round(ratio; digits = 2), "×"))),
        _column(:compiler_result, "Compiler result";
            format = (value, _) -> value == "measured" ? value :
                h.span(value; class = "rk-result-unsupported")),
    )

    setup_rows = map(filter(
        measurement -> get(measurement, "rk_native_supported", false),
        receipt["measurements"],
    )) do measurement
        supported = get(measurement, "rk_reactant_supported", false)
        (;
            boundary = boundary_labels[measurement["boundary"]],
            outcome = measurement["outcome"],
            transfer_seconds = Float64(measurement["reactant_transfer_seconds"]),
            compile_seconds = Float64(measurement["reactant_compile_seconds"]),
            first_seconds = supported ?
                Float64(measurement["reactant_first_execution_seconds"]) : missing,
            result = supported ? "compiled" : "unsupported",
        )
    end
    seconds(value) = string(round(value; sigdigits = 4), " s")
    setup_columns = (
        _column(:boundary, "Input boundary"),
        _column(:outcome, "Requested output"),
        _column(:transfer_seconds, "Input transfer";
            format = (value, _) -> seconds(value)),
        _column(:compile_seconds, "Compile attempt";
            format = (value, _) -> seconds(value)),
        _column(:first_seconds, "First synchronous call";
            format = (value, _) -> _unsupported(value, seconds)),
        _column(:result, "Result";
            format = (value, _) -> value == "compiled" ? value :
                h.span(value; class = "rk-result-unsupported")),
    )

    setup = receipt["setup"]
    setup_summary = "The fresh environment took " *
        "$(seconds(Float64(setup["environment_seconds"]))) to resolve/install, " *
        "package precompilation took " *
        "$(seconds(Float64(setup["package_precompile_seconds"]))), and preparing " *
        "all native HAVE/WANT kernels took " *
        "$(seconds(Float64(setup["kernel_preparation_seconds"]))). None of these, " *
        "the per-cell transfers, compilation, first calls, or result readback is " *
        "inside the steady-state timings."
    provenance = h.p(; class = "rk-result-provenance")(
        "Sources: ",
        h.a(h.code("benchmark/receipts/eight-schools-reactant-v1.toml");
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/eight-schools-reactant-v1.toml"),
        " generated by ",
        h.a(h.code("benchmark/eight_schools_reactant_comparison.jl");
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/eight_schools_reactant_comparison.jl"),
        ". Exact model authority: ",
        h.a(h.code(String(pins["source_authority_path"]));
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/$(pins["source_authority_path"])"),
        ". Pins: source blob $(first(String(pins["source_authority_blob"]), 10)), " *
        "Reactant $(pins["reactant_version"]), Julia $(pins["julia_version"]), " *
        "RK $(first(String(pins["reactivekernels_sha"]), 10)).",
    )

    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _result_table(rows, timing_columns; id = "eight-schools-reactant-matrix",
            title = "Native RK / Reactant steady-state matrix",
            note = "Every row is one cell of the matched 3×4 primal matrix. Unsupported cells remain visible with their recorded diagnostic."),
        Markdown.Paragraph(Any[setup_summary]),
        _result_table(setup_rows, setup_columns;
            id = "eight-schools-reactant-setup",
            title = "Setup, compilation, and first-call costs",
            note = "These costs are reported separately and excluded from steady-state timing."),
        _static_block(provenance),
    ])
end

"""Render focused Reactant-compiled-AD comparisons from the full receipt."""
function render_eight_schools_reactant_ad_benchmark()
    return _render_eight_schools_reactant_longform_focused(
        _EIGHT_SCHOOLS_REACTANT_AD_RECEIPT_PATH,
        "eight-schools-reactant-ad-v2";
        mode = :ad, id_prefix = "eight-schools-reactant-ad",
        profile_label = "Eight Schools Reactant value + gradient")
    receipt = TOML.parsefile(_EIGHT_SCHOOLS_REACTANT_AD_RECEIPT_PATH)
    get(receipt, "schema", "") == "eight-schools-reactant-ad-v1" ||
        error("unexpected Eight Schools Reactant-AD benchmark receipt schema")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("Eight Schools Reactant-AD receipt was produced from a dirty checkout")
    protocol = receipt["protocol"]
    get(protocol, "source_reused", false) ||
        error("Eight Schools Reactant-AD receipt does not reuse the authored source")
    get(protocol, "reactant_sync", false) ||
        error("Eight Schools Reactant-AD receipt is not synchronous")
    get(protocol, "gradient_operation", "") ==
        "value and gradient for scalar WANTs; value and reverse pullback for pointwise WANTs" ||
        error("Eight Schools Reactant-AD reverse-operation contract drifted")
    get(protocol, "compiled_pullback_exposed", true) == false ||
        error("Eight Schools Reactant-AD receipt claims a compiled pullback")
    get(protocol, "compiled_structured_active_exposed", true) == false ||
        error("Eight Schools Reactant-AD receipt claims structured tracing")
    Tuple(String.(protocol["input_boundaries"])) == _EIGHT_SCHOOLS_BOUNDARIES ||
        error("unexpected Eight Schools input-boundary matrix")
    Tuple(String.(protocol["outcomes"])) == _EIGHT_SCHOOLS_OUTCOMES ||
        error("unexpected Eight Schools output matrix")

    boundary_labels = Dict(
        "packed_unconstrained" => "Packed unconstrained",
        "constrained_parameters" => "Constrained parameters",
        "minimal_likelihood" => "Likelihood inputs only",
    )
    rows = map(receipt["measurements"]) do measurement
        native_supported = get(measurement, "rk_native_ad_supported", false)
        reactant_supported = get(measurement, "rk_reactant_ad_supported", false)
        native_ns = native_supported ?
            Float64(measurement["rk_native_ad"]["median_ns"]) : missing
        reactant_ns = reactant_supported ?
            Float64(measurement["rk_reactant_ad"]["median_ns"]) : missing
        diagnostic = reactant_supported ? "measured" :
            get(measurement, "rk_reactant_ad_error",
                get(measurement, "rk_native_ad_error", "unsupported"))
        (;
            boundary = boundary_labels[measurement["boundary"]],
            outcome = measurement["outcome"],
            operation = get(measurement, "operation", "—"),
            active = get(measurement, "active_port", "—"),
            native_ns,
            reactant_ns,
            relative_time = reactant_supported ? reactant_ns / native_ns : missing,
            gradient_error = reactant_supported ?
                Float64(measurement["max_abs_error"]) : missing,
            compiler_result = diagnostic,
        )
    end
    measured = filter(row -> row.reactant_ns !== missing, rows)
    ratios = Float64[row.relative_time for row in measured]
    native_cells = count(row -> row.native_ns !== missing, rows)
    native_gradients = count(row -> row.native_ns !== missing &&
        row.operation == "value and gradient", rows)
    native_pullbacks = count(row -> row.native_ns !== missing &&
        row.operation == "value and pullback", rows)
    native_only = count(
        row -> row.native_ns !== missing && row.reactant_ns === missing, rows)
    summary = isempty(ratios) ?
        "No Reactant-AD matrix cell compiled in this receipt." :
        "Native RK AD measures $native_cells cells: $native_gradients scalar " *
        "gradients and $native_pullbacks fixed-cotangent pointwise pullbacks. " *
        "Reactant compiled $(length(measured)) array-backed scalar gradients at " *
        "exact parity; $native_only native cells remain compiled-AD unsupported. " *
        "The receipt does not claim compiled structured gradients or pullbacks, " *
        "and packed joint/prior retain their real primal compiler diagnostics. " *
        "Across compiled cells, " *
        "steady-state Reactant/native AD time ranges from " *
        "$(round(minimum(ratios); digits = 2))× to " *
        "$(round(maximum(ratios); digits = 2))× on this CPU receipt."
    timing_columns = (
        _column(:boundary, "Input boundary"),
        _column(:outcome, "Requested output"),
        _column(:operation, "Native operation"),
        _column(:active, "Active port"),
        _column(:native_ns, "Native RK reverse AD";
            format = (value, _) -> _unsupported(value, _fmt_ns)),
        _column(:reactant_ns, "Reactant AD";
            format = (value, _) -> _unsupported(value, _fmt_ns)),
        _column(:relative_time, "Reactant ÷ native";
            format = (value, _) -> _unsupported(
                value, ratio -> string(round(ratio; digits = 2), "×"))),
        _column(:gradient_error, "Gradient max-abs-error";
            format = (value, _) -> _unsupported(
                value, err -> string(round(err; sigdigits = 2)))),
        _column(:compiler_result, "Compiler result";
            format = (value, _) -> value == "measured" ? value :
                h.span(value; class = "rk-result-unsupported")),
    )

    setup_rows = map(filter(
        measurement -> get(measurement, "rk_native_ad_supported", false),
        receipt["measurements"],
    )) do measurement
        supported = get(measurement, "rk_reactant_ad_supported", false)
        attempted = haskey(measurement, "reactant_ad_compile_seconds")
        (;
            boundary = boundary_labels[measurement["boundary"]],
            outcome = measurement["outcome"],
            ad_prep_seconds = Float64(measurement["ad_preparation_seconds"]),
            transfer_seconds = attempted ?
                Float64(measurement["reactant_transfer_seconds"]) : missing,
            compile_seconds = attempted ?
                Float64(measurement["reactant_ad_compile_seconds"]) : missing,
            first_seconds = supported ?
                Float64(measurement["reactant_first_execution_seconds"]) : missing,
            result = supported ? "compiled" :
                attempted ? "compile rejected" : "not a public compiled surface",
        )
    end
    seconds(value) = string(round(value; sigdigits = 4), " s")
    setup_columns = (
        _column(:boundary, "Input boundary"),
        _column(:outcome, "Requested output"),
        _column(:ad_prep_seconds, "AD preparation";
            format = (value, _) -> seconds(value)),
        _column(:transfer_seconds, "Input transfer";
            format = (value, _) -> _unsupported(value, seconds)),
        _column(:compile_seconds, "AD compile attempt";
            format = (value, _) -> _unsupported(value, seconds)),
        _column(:first_seconds, "First synchronous call";
            format = (value, _) -> _unsupported(value, seconds)),
        _column(:result, "Result";
            format = (value, _) -> value == "compiled" ? value :
                h.span(value; class = "rk-result-unsupported")),
    )

    setup = receipt["setup"]
    setup_summary = "The fresh environment took " *
        "$(seconds(Float64(setup["environment_seconds"]))) to resolve/install, " *
        "package precompilation took " *
        "$(seconds(Float64(setup["package_precompile_seconds"]))), and building the " *
        "model graph took " *
        "$(seconds(Float64(setup["kernel_preparation_seconds"]))). None of these, " *
        "the per-cell AD preparation, input transfers, gradient compilation, " *
        "first calls, or result readback is inside the steady-state timings."
    provenance = h.p(; class = "rk-result-provenance")(
        "Sources: ",
        h.a(h.code("benchmark/receipts/eight-schools-reactant-ad-v1.toml");
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/eight-schools-reactant-ad-v1.toml"),
        " generated by ",
        h.a(h.code("benchmark/eight_schools_reactant_ad_comparison.jl");
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/eight_schools_reactant_ad_comparison.jl"),
        ", consuming native ", h.code("ad_value_and_gradient[!]"), " and ",
        h.code("ad_value_and_pullback[!]"), " plus the public compiled scalar verb ",
        h.code("compile_ad_value_and_gradient"),
        ". It reuses the same derivative matrix as the ",
        h.a(h.code("eight-schools-ad-v1.toml");
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/eight-schools-ad-v1.toml"),
        " AD receipt. Exact model authority: ",
        h.a(h.code(String(pins["source_authority_path"]));
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/$(pins["source_authority_path"])"),
        ". Pins: source blob $(first(String(pins["source_authority_blob"]), 10)), " *
        "Reactant $(pins["reactant_version"]), Enzyme $(pins["enzyme_version"]), " *
        "Julia $(pins["julia_version"]), RK $(first(String(pins["reactivekernels_sha"]), 10)).",
    )

    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _result_table(rows, timing_columns; id = "eight-schools-reactant-ad-matrix",
            title = "Native RK AD / Reactant-compiled AD steady-state matrix",
            note = "Scalar cells measure native value-and-gradient; pointwise cells measure native value-and-VJP for the fixed receipt cotangent. Reactant claims only the public compiled scalar value-and-gradient subset; no compiled pullback or structured tracing surface is implied."),
        Markdown.Paragraph(Any[setup_summary]),
        _result_table(setup_rows, setup_columns;
            id = "eight-schools-reactant-ad-setup",
            title = "AD preparation, compilation, and first-call costs",
            note = "These costs are reported separately and excluded from steady-state timing."),
        _static_block(provenance),
    ])
end

const _MNIST_REACTANT_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "mnist-reactant-v2.toml")
const _MNIST_REACTANT_WREN_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts",
    "mnist-reactant-wren-pca40-v1.toml")

"""Render the checked-in MNIST native-RK/Reactant receipt as sortable data."""
function render_mnist_reactant_benchmark(
    receipt_path::AbstractString = _MNIST_REACTANT_RECEIPT_PATH,
)
    receipt = TOML.parsefile(receipt_path)
    if get(receipt, "schema", "") == "mnist-reactant-v2"
        return _render_mnist_reactant_longform_focused(
            receipt_path, "mnist-reactant-v2";
            mode = :primal,
            id_prefix = "mnist-reactant-full",
            profile_label = "Full raw MNIST Reactant primal")
    end
    get(receipt, "schema", "") == "mnist-reactant-v1" ||
        error("unexpected MNIST Reactant benchmark receipt schema")
    return _render_mnist_reactant_v1_focused(receipt_path;
        mode = :primal,
        id_prefix = "mnist-reactant-matrix-wren-pca40",
        profile_label = "Wren PCA-40 Reactant primal")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("MNIST Reactant receipt was produced from a dirty checkout")
    protocol = receipt["protocol"]
    get(protocol, "source_reused", false) ||
        error("MNIST Reactant receipt does not reuse the authored source")
    get(protocol, "reactant_sync", false) ||
        error("MNIST Reactant receipt is not synchronous")
    get(protocol, "gradients_included", true) &&
        error("MNIST Reactant receipt unexpectedly contains gradients")
    get(protocol, "partial_evaluation_enabled", true) &&
        error("MNIST Reactant receipt unexpectedly binds the full dataset")
    Tuple(String.(get(protocol, "runtime_data_ports", String[]))) ==
        ("X", "y", "num_classes") ||
        error("MNIST Reactant receipt lost the runtime dataset boundary")
    Tuple(String.(protocol["input_boundaries"])) == _MNIST_LOGISTIC_BOUNDARIES ||
        error("unexpected MNIST input-boundary matrix")
    Tuple(String.(protocol["outcomes"])) == _MNIST_LOGISTIC_OUTCOMES ||
        error("unexpected MNIST output matrix")
    dataset_profile = String(get(protocol, "dataset_profile", "full-raw"))
    profile_label = dataset_profile == "wren-pca40" ?
        "Wren PCA-40" : "Full raw MNIST"
    id_suffix = dataset_profile == "wren-pca40" ? "-wren-pca40" : ""
    receipt_name = basename(receipt_path)

    boundary_labels = Dict(
        "packed_unconstrained" => "Packed unconstrained",
        "structured_parameters" => "Structured (W, b)",
    )
    # Same headline estimator as the matched primal receipt: the minimum of
    # per-round BenchmarkTools minimums (uncontended cost).
    rows = map(receipt["measurements"]) do measurement
        reactant_supported = get(measurement, "rk_reactant_supported", false)
        native_ns = Float64(measurement["rk_native"]["min_ns"])
        reactant_ns = reactant_supported ?
            Float64(measurement["rk_reactant"]["min_ns"]) : missing
        diagnostic = reactant_supported ? "measured" :
            get(measurement, "rk_reactant_error", "unsupported")
        (;
            boundary = boundary_labels[measurement["boundary"]],
            outcome = measurement["outcome"],
            native_ns,
            reactant_ns,
            relative_time = reactant_supported ? reactant_ns / native_ns : missing,
            compiler_result = diagnostic,
        )
    end
    measured = filter(row -> row.reactant_ns !== missing, rows)
    ratios = Float64[row.relative_time for row in measured]
    unsupported = count(row -> row.reactant_ns === missing, rows)
    packed_joint = only(filter(rows) do row
        row.boundary == "Packed unconstrained" && row.outcome == "joint"
    end)
    packed_joint.reactant_ns === missing &&
        error("MNIST packed full joint is not compiled in the receipt")
    joint_speedup = packed_joint.native_ns / packed_joint.reactant_ns
    joint_comparison = joint_speedup >= 1 ?
        "$(round(joint_speedup; digits = 2))× faster" :
        "$(round(inv(joint_speedup); digits = 2))× native time"
    summary = isempty(ratios) ?
        "No Reactant matrix cell compiled in this receipt." :
        "$profile_label. Packed full joint: Reactant " *
        "$(_fmt_ns(packed_joint.reactant_ns)) " *
        "versus native RK $(_fmt_ns(packed_joint.native_ns)) " *
        "($joint_comparison). Reactant compiled $(length(measured)) of the 8 " *
        "primal matrix cells; $unsupported retain compiler diagnostics. " *
        "Across compiled cells, " *
        "steady-state Reactant/native time ranges from " *
        "$(round(minimum(ratios); digits = 2))× to " *
        "$(round(maximum(ratios); digits = 2))× on this CPU receipt."
    timing_columns = (
        _column(:boundary, "Input boundary"),
        _column(:outcome, "Requested output"),
        _column(:native_ns, "Native RK";
            format = (value, _) -> _unsupported(value, _fmt_ns)),
        _column(:reactant_ns, "Reactant";
            format = (value, _) -> _unsupported(value, _fmt_ns)),
        _column(:relative_time, "Reactant ÷ native";
            format = (value, _) -> _unsupported(
                value, ratio -> string(round(ratio; digits = 2), "×"))),
        _column(:compiler_result, "Compiler result";
            format = (value, _) -> value == "measured" ? value :
                h.span(value; class = "rk-result-unsupported")),
    )

    setup_rows = map(receipt["measurements"]) do measurement
        supported = get(measurement, "rk_reactant_supported", false)
        (;
            boundary = boundary_labels[measurement["boundary"]],
            outcome = measurement["outcome"],
            transfer_seconds = Float64(measurement["reactant_transfer_seconds"]),
            compile_seconds = Float64(measurement["reactant_compile_seconds"]),
            first_seconds = supported ?
                Float64(measurement["reactant_first_execution_seconds"]) : missing,
            result = supported ? "compiled" : "unsupported",
        )
    end
    seconds(value) = string(round(value; sigdigits = 4), " s")
    setup_columns = (
        _column(:boundary, "Input boundary"),
        _column(:outcome, "Requested output"),
        _column(:transfer_seconds, "Input transfer";
            format = (value, _) -> seconds(value)),
        _column(:compile_seconds, "Compile attempt";
            format = (value, _) -> seconds(value)),
        _column(:first_seconds, "First synchronous call";
            format = (value, _) -> _unsupported(value, seconds)),
        _column(:result, "Result";
            format = (value, _) -> value == "compiled" ? value :
                h.span(value; class = "rk-result-unsupported")),
    )

    setup = receipt["setup"]
    setup_summary = "The fresh environment took " *
        "$(seconds(Float64(setup["environment_seconds"]))) to resolve/install, " *
        "package precompilation took " *
        "$(seconds(Float64(setup["package_precompile_seconds"]))), preparing " *
        "the $(protocol["num_observations"])×$(protocol["num_features"]) " *
        "$profile_label matrix took " *
        "$(seconds(Float64(setup["data_load_seconds"]))), and preparing all " *
        "native HAVE/WANT kernels took " *
        "$(seconds(Float64(setup["kernel_preparation_seconds"]))). None of " *
        "these, the per-cell transfers, compilation, first calls, or result " *
        "readback is inside the steady-state timings."
    provenance = h.p(; class = "rk-result-provenance")(
        "Sources: ",
        h.a(h.code("benchmark/receipts/$receipt_name");
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/$receipt_name"),
        " generated by ",
        h.a(h.code("benchmark/mnist_reactant_comparison.jl");
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/mnist_reactant_comparison.jl"),
        ". Exact model authority: ",
        h.a(h.code(String(pins["source_authority_path"]));
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/$(pins["source_authority_path"])"),
        ". Pins: source blob $(first(String(pins["source_authority_blob"]), 10)), " *
        "Reactant $(pins["reactant_version"]), Julia $(pins["julia_version"]), " *
        "RK $(first(String(pins["reactivekernels_sha"]), 10)).",
    )

    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _result_table(rows, timing_columns;
            id = "mnist-reactant-matrix$id_suffix",
            title = "Native RK / Reactant steady-state matrix",
            note = "Every row is one cell of the matched 2×4 primal matrix, using the same runtime data boundary; each measured cell is the minimum of per-round BenchmarkTools minimums. Unsupported cells remain visible with their recorded diagnostic."),
        Markdown.Paragraph(Any[setup_summary]),
        _result_table(setup_rows, setup_columns;
            id = "mnist-reactant-setup$id_suffix",
            title = "Setup, compilation, and first-call costs",
            note = "These costs are reported separately and excluded from steady-state timing."),
        _static_block(provenance),
    ])
end

render_mnist_reactant_wren_benchmark() =
    render_mnist_reactant_benchmark(_MNIST_REACTANT_WREN_RECEIPT_PATH)

const _MNIST_REACTANT_AD_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "mnist-reactant-ad-v2.toml")
const _MNIST_REACTANT_AD_WREN_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts",
    "mnist-reactant-ad-wren-pca40-v1.toml")

"""Render the checked-in MNIST Reactant-compiled-AD receipt as sortable data."""
function render_mnist_reactant_ad_benchmark(
    receipt_path::AbstractString = _MNIST_REACTANT_AD_RECEIPT_PATH,
)
    receipt = TOML.parsefile(receipt_path)
    if get(receipt, "schema", "") == "mnist-reactant-ad-v2"
        return _render_mnist_reactant_longform_focused(
            receipt_path, "mnist-reactant-ad-v2";
            mode = :ad,
            id_prefix = "mnist-reactant-ad-full",
            profile_label = "Full raw MNIST Reactant value + gradient")
    end
    get(receipt, "schema", "") == "mnist-reactant-ad-v1" ||
        error("unexpected MNIST Reactant-AD benchmark receipt schema")
    return _render_mnist_reactant_v1_focused(receipt_path;
        mode = :ad,
        id_prefix = "mnist-reactant-ad-matrix-wren-pca40",
        profile_label = "Wren PCA-40 Reactant value + gradient")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("MNIST Reactant-AD receipt was produced from a dirty checkout")
    protocol = receipt["protocol"]
    get(protocol, "source_reused", false) ||
        error("MNIST Reactant-AD receipt does not reuse the authored source")
    get(protocol, "reactant_sync", false) ||
        error("MNIST Reactant-AD receipt is not synchronous")
    get(protocol, "rk_reactant_ad_surface", "") ==
        "prepare_ad + compile_ad_value_and_gradient" ||
        error("MNIST Reactant-AD receipt does not consume the first-class verb")
    get(protocol, "partial_evaluation_enabled", true) &&
        error("MNIST Reactant-AD receipt unexpectedly binds the full dataset")
    Tuple(String.(get(protocol, "runtime_data_ports", String[]))) ==
        ("X", "y", "num_classes") ||
        error("MNIST Reactant-AD receipt lost the runtime dataset boundary")
    Tuple(String.(protocol["input_boundaries"])) == _MNIST_LOGISTIC_BOUNDARIES ||
        error("unexpected MNIST AD input-boundary matrix")
    Tuple(String.(protocol["outcomes"])) == _MNIST_LOGISTIC_OUTCOMES ||
        error("unexpected MNIST AD output matrix")
    dataset_profile = String(get(protocol, "dataset_profile", "full-raw"))
    profile_label = dataset_profile == "wren-pca40" ?
        "Wren PCA-40" : "Full raw MNIST"
    id_suffix = dataset_profile == "wren-pca40" ? "-wren-pca40" : ""
    receipt_name = basename(receipt_path)

    boundary_labels = Dict(
        "packed_unconstrained" => "Packed unconstrained",
        "structured_parameters" => "Structured (W, b)",
    )
    # Same headline estimator as the matched AD receipt: the minimum of
    # per-round BenchmarkTools minimums (uncontended cost).
    rows = map(receipt["measurements"]) do measurement
        native_supported = get(measurement, "rk_native_ad_supported", false)
        reactant_supported = get(measurement, "rk_reactant_ad_supported", false)
        native_ns = native_supported ?
            Float64(measurement["rk_native_ad"]["min_ns"]) : missing
        reactant_ns = reactant_supported ?
            Float64(measurement["rk_reactant_ad"]["min_ns"]) : missing
        diagnostic = reactant_supported ? "measured" :
            get(measurement, "rk_reactant_ad_error", "unsupported")
        (;
            boundary = boundary_labels[measurement["boundary"]],
            outcome = measurement["outcome"],
            native_ns,
            reactant_ns,
            relative_time = reactant_supported ? reactant_ns / native_ns : missing,
            compiler_result = diagnostic,
        )
    end
    measured = filter(row -> row.reactant_ns !== missing, rows)
    ratios = Float64[row.relative_time for row in measured]
    frontier = count(row -> row.native_ns !== missing &&
        row.reactant_ns === missing, rows)
    packed_joint = only(filter(rows) do row
        row.boundary == "Packed unconstrained" && row.outcome == "joint"
    end)
    packed_joint.reactant_ns === missing &&
        error("MNIST packed full-joint AD is not compiled in the receipt")
    joint_speedup = packed_joint.native_ns / packed_joint.reactant_ns
    joint_comparison = joint_speedup >= 1 ?
        "$(round(joint_speedup; digits = 2))× faster" :
        "$(round(inv(joint_speedup); digits = 2))× native time"
    summary = isempty(ratios) ?
        "No Reactant-compiled gradient cell compiled in this receipt." :
        "$profile_label. Packed full-joint value-and-gradient: Reactant " *
        "$(_fmt_ns(packed_joint.reactant_ns)) versus native RK AD " *
        "$(_fmt_ns(packed_joint.native_ns)) ($joint_comparison). " *
        "Reactant-compiled AD covers $(length(measured)) of the 3 native-AD " *
        "scalar cells; $frontier keep native AD but no compiled gradient. " *
        "Across compiled " *
        "cells, steady-state Reactant/native value-and-gradient time ranges " *
        "from $(round(minimum(ratios); digits = 2))× to " *
        "$(round(maximum(ratios); digits = 2))× on this CPU receipt."
    timing_columns = (
        _column(:boundary, "Input boundary"),
        _column(:outcome, "Requested output"),
        _column(:native_ns, "Native RK AD";
            format = (value, _) -> _unsupported(value, _fmt_ns)),
        _column(:reactant_ns, "Reactant-compiled AD";
            format = (value, _) -> _unsupported(value, _fmt_ns)),
        _column(:relative_time, "Reactant ÷ native";
            format = (value, _) -> _unsupported(
                value, ratio -> string(round(ratio; digits = 2), "×"))),
        _column(:compiler_result, "Compiler result";
            format = (value, _) -> value == "measured" ? value :
                h.span(value; class = "rk-result-unsupported")),
    )

    setup_rows = map(filter(
        measurement -> get(measurement, "rk_native_ad_supported", false),
        receipt["measurements"],
    )) do measurement
        supported = get(measurement, "rk_reactant_ad_supported", false)
        (;
            boundary = boundary_labels[measurement["boundary"]],
            outcome = measurement["outcome"],
            prepare_seconds = Float64(measurement["ad_preparation_seconds"]),
            transfer_seconds = Float64(measurement["reactant_transfer_seconds"]),
            compile_seconds = Float64(measurement["reactant_ad_compile_seconds"]),
            first_seconds = supported ?
                Float64(measurement["reactant_first_execution_seconds"]) : missing,
            result = supported ? "compiled" : "unsupported",
        )
    end
    seconds(value) = string(round(value; sigdigits = 4), " s")
    setup_columns = (
        _column(:boundary, "Input boundary"),
        _column(:outcome, "Requested output"),
        _column(:prepare_seconds, "AD preparation";
            format = (value, _) -> seconds(value)),
        _column(:transfer_seconds, "Input transfer";
            format = (value, _) -> seconds(value)),
        _column(:compile_seconds, "Compile attempt";
            format = (value, _) -> seconds(value)),
        _column(:first_seconds, "First synchronous call";
            format = (value, _) -> _unsupported(value, seconds)),
        _column(:result, "Result";
            format = (value, _) -> value == "compiled" ? value :
                h.span(value; class = "rk-result-unsupported")),
    )

    setup = receipt["setup"]
    setup_summary = "The fresh environment took " *
        "$(seconds(Float64(setup["environment_seconds"]))) to resolve/install, " *
        "package precompilation took " *
        "$(seconds(Float64(setup["package_precompile_seconds"]))), preparing " *
        "the $(protocol["num_observations"])×$(protocol["num_features"]) " *
        "$profile_label matrix took " *
        "$(seconds(Float64(setup["data_load_seconds"]))), and building the " *
        "model graph took " *
        "$(seconds(Float64(setup["kernel_preparation_seconds"]))). None of " *
        "these, per-cell AD preparation, transfers, gradient compilation, " *
        "first calls, or result readback is inside the steady-state timings."
    provenance = h.p(; class = "rk-result-provenance")(
        "Sources: ",
        h.a(h.code("benchmark/receipts/$receipt_name");
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/$receipt_name"),
        " generated by ",
        h.a(h.code("benchmark/mnist_reactant_ad_comparison.jl");
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/mnist_reactant_ad_comparison.jl"),
        ". Exact model authority: ",
        h.a(h.code(String(pins["source_authority_path"]));
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/$(pins["source_authority_path"])"),
        ". Pins: source blob $(first(String(pins["source_authority_blob"]), 10)), " *
        "Reactant $(pins["reactant_version"]), Enzyme $(pins["enzyme_version"]), " *
        "Julia $(pins["julia_version"]), " *
        "RK $(first(String(pins["reactivekernels_sha"]), 10)).",
    )

    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _result_table(rows, timing_columns;
            id = "mnist-reactant-ad-matrix$id_suffix",
            title = "Native RK AD / Reactant-compiled AD steady-state matrix",
            note = "Value-and-gradient per differentiable scalar cell, using the same runtime data boundary; each measured cell is the minimum of per-round BenchmarkTools minimums. Non-scalar (pointwise) and two-active-port structured cells stay unsupported with their recorded reason."),
        Markdown.Paragraph(Any[setup_summary]),
        _result_table(setup_rows, setup_columns;
            id = "mnist-reactant-ad-setup$id_suffix",
            title = "AD preparation, compilation, and first-call costs",
            note = "These costs are reported separately and excluded from steady-state timing."),
        _static_block(provenance),
    ])
end

render_mnist_reactant_ad_wren_benchmark() =
    render_mnist_reactant_ad_benchmark(_MNIST_REACTANT_AD_WREN_RECEIPT_PATH)

"""Render the public compiler surfaces as an ordered pipeline."""
function render_compiler_api_map()
    rows = [
        (stage = "Author a dataflow graph", surface = "@kernel, Graph, Value, Recipe, add!, merge, compose",
         result = "KernelSpec or Graph metadata", runtime = "None; authoring executes no recipe body"),
        (stage = "Select work", surface = "plan, explain, inputs, outputs",
         result = "Inspectable Plan", runtime = "None after preparation"),
        (stage = "Emit a stateless kernel", surface = "lower, transform, compile, prepare, prepare!, code_expr",
         result = "PreparedKernel", runtime = "Only selected recipe calls and their value flow"),
        (stage = "Lift scalar work", surface = "plate, lower_batched, replica",
         result = "Batched or replicated callable", runtime = "One native loop/map or a backend tensor program"),
        (stage = "Reuse recipe storage", surface = "prepare_nonallocating",
         result = "NonAllocatingKernel", runtime = "Selected calls plus cache mutation; no planner"),
        (stage = "Plan demands dynamically", surface = "ReactiveState, get!, materialize!, freeze!, checkpoint",
         result = "Open-ended incremental state", runtime = "Planning and provenance bookkeeping at each demand"),
        (stage = "Compile a closed state graph", surface = "prepare_reactive, ReactiveProgram, CompiledReactiveState",
         result = "Typed slots and generated getters", runtime = "Validity checks, required recipe calls, and invalidation worklists"),
        (stage = "Compile a free state transition", surface = "compile_state_transition, initial_transition_state, partial",
         result = "CompiledStateTransition", runtime = "Authored writes, statically unrolled control, and demand-driven derived-field repairs"),
        (stage = "Compile a bounded stateful method", surface = "compile_stateful, stateful_compiler_bindings, functionalize_stateful, stateful_snapshot",
         result = "Typed native state plus a backend-neutral functional transition", runtime = "Source-ordered structured control, canonical nested-state repairs, typed effects, and explicit overflow"),
        (stage = "Inspect the selected graph", surface = "visualize, dot_source, save_visualization",
         result = "Plan/DAG view", runtime = "No effect on compilation"),
    ]
    cards = [
        h.li(; class = "rk-pipeline-step")(
            h.article(
                h.h3(row.stage),
                h.dl(
                    h.dt("Public surface"), h.dd(h.code(row.surface)),
                    h.dt("Result"), h.dd(row.result),
                    h.dt("Runtime work that remains"), h.dd(row.runtime),
                ),
            ),
        )
        for row in rows
    ]
    Markdown.MD(Any[_static_block(h.ol(cards...; class = "rk-pipeline"))])
end

"""Render the definitive compiler support matrix as an HTMXObjects sortable table."""
function render_compiler_capabilities()
    rows = [
        (capability = "Exact selection among alternative producers", supported = "Yes", boundary = "Finite acyclic graph; additive non-negative declared costs; exponential worst case"),
        (capability = "Multi-output stateless recipes", supported = "Yes", boundary = "One call, ordered tuple outputs; selected-owner rules discard collateral duplicates"),
        (capability = "Arbitrary Julia inside a stateless recipe", supported = "Yes, as opaque execution", boundary = "No internal effect, algebra, allocation, or control-flow analysis"),
        (capability = "Graph structural CSE", supported = "Yes, opt-in", boundary = "Same non-null key, canonical inputs, output arity/types, non-effectful only"),
        (capability = "Algebraic/symbolic optimization", supported = "No", boundary = "No commutativity, reassociation, constant folding, symbolic differentiation, or cost measurement"),
        (capability = "Straight-line native preparation", supported = "Yes", boundary = "Recipe operations still obey ordinary Julia behavior"),
        (capability = "User AST passes", supported = "Yes", boundary = "User is responsible for semantic preservation"),
        (capability = "Multiple WANTS", supported = "Yes", boundary = "Returned in request order"),
        (capability = "Stateless signature defaults/keywords", supported = "Yes", boundary = "Fixed arguments; no positional/keyword splats"),
        (capability = "Plate map/reduce or collect", supported = "Yes", boundary = "Single-output recipes and one batch-dependent scalar WANT; no scan/fold/parallel tree reduction"),
        (capability = "Whole-kernel replica axis", supported = "Yes", boundary = "Number/array ports and outputs; native map/stack may allocate"),
        (capability = "Open-ended reactive demands", supported = "Yes", boundary = "Planning/provenance work remains at each demand"),
        (capability = "Fixed compiled reactive state", supported = "Yes", boundary = "Graph and boundary frozen; mutable instance is not thread-safe"),
        (capability = "In-place cache reuse", supported = "Optional", boundary = "Single-output operation support and runtime types determine allocation behavior"),
        (capability = "Reactive object field methods", supported = "Yes, declared grammar", boundary = "No let, exceptions, comprehensions/generators, closures, do, or nested functions inside rewritten methods"),
        (capability = "Captured method branches/loops/recursion", supported = "Yes, compiler subset", boundary = "Exact source representation, ownership, effects, and overloads required"),
        (capability = "Functional free state transition", supported = "Yes, narrow public subset", boundary = "Direct owned writes; exact map/copy/destructuring; bound numeric controls; captured finite Colon loops are statically unrolled"),
        (capability = "Fixed-capacity structural container ABI", supported = "Yes, compiler subset", boundary = "Prototype-bound recursive topology, exact primitive leaves, static authorities and owned aliases; validated read/write/copy/swap/select; no resizing or arbitrary elements"),
        (capability = "Recursive functional state-machine SCC", supported = "No", boundary = "Finite containers and bounded control are admitted first; recursive SCC lowering rejects explicitly at the current frontier"),
        (capability = "Infer effects from arbitrary Julia methods", supported = "No", boundary = "Source capture plus exact registrations only; no Julia IR/inference analysis"),
        (capability = "Generic effectful operation fallback", supported = "No", boundary = "Opaque or specialization-unsafe calls reject before executable lowering"),
        (capability = "Runtime graph mutation after preparation", supported = "No", boundary = "Cached stateless entries version out; compiled reactive programs throw version errors"),
        (capability = "Reentrant/concurrent stateful kernel instance", supported = "No", boundary = "Use one prepared mutable/cache-owning instance per independent caller"),
        (capability = "PPL/model semantics", supported = "No compiler feature", boundary = "Examples build log densities and samplers from ordinary recipes; there is no trace/address language"),
        (capability = "NUTS, log-density, or PPL domain API", supported = "No", boundary = "These live in external compilation examples and are not loaded or exported by RK"),
        (capability = "General dynamic scheduling in a hot kernel", supported = "No", boundary = "Dynamic planning exists only in ReactiveState demand orchestration"),
        (capability = "General Julia compiler replacement", supported = "No", boundary = "Unsupported captured shapes reject; ordinary opaque stateless recipes remain ordinary Julia"),
    ]
    support_format = function(value, _)
        class = startswith(value, "Yes") ? "rk-support-yes" :
            startswith(value, "No") ? "rk-support-no" : "rk-support-partial"
        h.span(value; class = "rk-support $class")
    end
    columns = (
        _column(:capability, "Capability"),
        _column(:supported, "Supported"; format = support_format),
        _column(:boundary, "Boundary or rejection"),
    )
    Markdown.MD(Any[_result_table(rows, columns;
        id = "compiler-capability-table", title = "Definitive support matrix")])
end

const _PROBPROG_MCMC_RECEIPT_PATH = joinpath(
    @__DIR__, "..", "benchmark", "receipts", "probprog-mcmc-v1.toml")

const _PROBPROG_MCMC_HARNESS_LABELS = (
    ("probprog_nuts", "ProbProg compiled NUTS (Reactant)"),
    ("advancedhmc_nuts", "AdvancedHMC NUTS over the RK density"),
    ("turing_nuts", "Turing NUTS (attested twin model)"),
)

const _PROBPROG_MCMC_MODEL_LABELS = Dict(
    "eight_schools" => "Eight Schools (10-dim packed posterior)",
    "mnist_logistic_wren_pca40" =>
        "MNIST logistic, Wren PCA-40 (369-dim packed joint)",
)

function _probprog_mcmc_receipt()
    receipt = TOML.parsefile(_PROBPROG_MCMC_RECEIPT_PATH)
    get(receipt, "schema", "") == "probprog-mcmc-v1" ||
        error("unexpected ProbProg MCMC receipt schema")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("ProbProg MCMC receipt was produced from a dirty checkout")
    receipt
end

_probprog_mcmc_row(receipt, model, harness) = only(filter(
    row -> row["model"] == model && row["harness"] == harness,
    receipt["measurements"]))

"""Render the checked-in ProbProg/AdvancedHMC/Turing NUTS sampling receipt."""
function render_probprog_mcmc_benchmark()
    receipt = _probprog_mcmc_receipt()
    pins = receipt["pins"]
    protocol = receipt["protocol"]
    num_warmup = Int(protocol["num_warmup"])
    num_samples = Int(protocol["num_samples"])
    transitions = num_warmup + num_samples

    seconds(value) = string(round(Float64(value); sigdigits = 4), " s")
    per_second(value) = string(round(Float64(value); sigdigits = 4), " /s")
    ess(value) = string(round(Float64(value); digits = 1))

    rows = NamedTuple[]
    for (model, model_label) in sort!(collect(_PROBPROG_MCMC_MODEL_LABELS))
        for (harness, harness_label) in _PROBPROG_MCMC_HARNESS_LABELS
            row = _probprog_mcmc_row(receipt, model, harness)
            measured = get(row, "state", "") == "measured"
            sampling_seconds = measured ?
                Float64(row["sampling_seconds"]) : missing
            push!(rows, (;
                model = model_label,
                harness = harness_label,
                warm = measured ?
                    Float64(row["compile_or_warm_start_seconds"]) : missing,
                sampling = sampling_seconds,
                rate = measured ? transitions / sampling_seconds : missing,
                divergences = measured ? Int(row["divergences"]) : missing,
                min_ess = measured ? Float64(row["min_ess"]) : missing,
                median_ess = measured ? Float64(row["median_ess"]) : missing,
                min_ess_rate = measured ?
                    Float64(row["min_ess"]) / sampling_seconds : missing,
                status = measured ? "measured" :
                    get(row, "reason", "unsupported"),
            ))
        end
    end

    columns = (
        _column(:model, "Model"),
        _column(:harness, "Sampler harness"),
        _column(:warm, "Compile / JIT warm start";
            format = (value, _) -> _unsupported(value, seconds)),
        _column(:sampling, "Warmup + sampling";
            format = (value, _) -> _unsupported(value, seconds)),
        _column(:rate, "Transitions / s";
            format = (value, _) -> _unsupported(value, per_second)),
        _column(:divergences, "Divergences";
            format = (value, _) -> _unsupported(value, string)),
        _column(:min_ess, "Min ESS";
            format = (value, _) -> _unsupported(value, ess)),
        _column(:median_ess, "Median ESS";
            format = (value, _) -> _unsupported(value, ess)),
        _column(:min_ess_rate, "Min ESS / s";
            format = (value, _) -> _unsupported(value, per_second)),
        _column(:status, "Compiler result";
            format = (value, _) -> value == "measured" ? value :
                h.span(value; class = "rk-result-unsupported")),
    )

    moment(value) = string(round(Float64(value); sigdigits = 4))
    measured_row(model, harness) = begin
        row = _probprog_mcmc_row(receipt, model, harness)
        get(row, "state", "") == "measured" ? row : nothing
    end
    eight_schools_rows = [(;
        harness = harness_label,
        mean_mu = Float64(row["posterior_mean_mu"]),
        sd_mu = Float64(row["posterior_sd_mu"]),
        mean_tau = Float64(row["posterior_mean_tau"]),
    ) for (harness, harness_label) in _PROBPROG_MCMC_HARNESS_LABELS
      for row in (measured_row("eight_schools", harness),) if row !== nothing]
    eight_schools_columns = (
        _column(:harness, "Sampler harness"),
        _column(:mean_mu, "E[μ]"; format = (value, _) -> moment(value)),
        _column(:sd_mu, "sd[μ]"; format = (value, _) -> moment(value)),
        _column(:mean_tau, "E[τ]"; format = (value, _) -> moment(value)),
    )
    mnist_rows = [(;
        harness = harness_label,
        mean_w = Float64(row["posterior_mean_first_coefficient"]),
        sd_w = Float64(row["posterior_sd_first_coefficient"]),
    ) for (harness, harness_label) in _PROBPROG_MCMC_HARNESS_LABELS
      for row in (measured_row("mnist_logistic_wren_pca40", harness),)
      if row !== nothing]
    mnist_columns = (
        _column(:harness, "Sampler harness"),
        _column(:mean_w, "E[W[1,1]]"; format = (value, _) -> moment(value)),
        _column(:sd_w, "sd[W[1,1]]"; format = (value, _) -> moment(value)),
    )

    parity_gaps = Float64[]
    rejected_probprog = String[]
    for (model, model_label) in sort!(collect(_PROBPROG_MCMC_MODEL_LABELS))
        row = _probprog_mcmc_row(receipt, model, "probprog_nuts")
        if get(row, "state", "") == "measured"
            push!(parity_gaps, Float64(row["density_parity_max_abs"]))
        else
            push!(rejected_probprog, model_label)
        end
    end
    isempty(parity_gaps) &&
        error("no measured ProbProg cell in the receipt; the page premise is gone")
    summary = "Every measured harness ran $num_warmup warmup plus " *
        "$num_samples retained adaptive NUTS transitions from one matched " *
        "initial position per model (seed $(protocol["seed"])). ProbProg's " *
        "per-draw log densities matched the native RK kernel to at most " *
        "$(round(maximum(parity_gaps); sigdigits = 3)) across every retained " *
        "draw of every measured cell; sampler trajectories are compared " *
        "statistically, never exactly." *
        (isempty(rejected_probprog) ? "" :
         " The ProbProg compile is a recorded rejection for: " *
         join(rejected_probprog, "; ") *
         " — its diagnostic stays visible in the table.")

    provenance = h.p(; class = "rk-result-provenance")(
        "Sources: ",
        h.a(h.code("benchmark/receipts/probprog-mcmc-v1.toml");
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/receipts/probprog-mcmc-v1.toml"),
        " generated by ",
        h.a(h.code("benchmark/probprog_mcmc_comparison.jl");
            href = "https://github.com/nsiccha/ReactiveKernels.jl/blob/main/benchmark/probprog_mcmc_comparison.jl"),
        ". Pins: Reactant $(pins["reactant_version"]), " *
        "AdvancedHMC $(pins["advancedhmc_version"]), " *
        "Turing $(pins["turing_version"]), Enzyme $(pins["enzyme_version"]), " *
        "Julia $(pins["julia_version"]), " *
        "RK $(first(String(pins["reactivekernels_sha"]), 10)).",
    )

    plot_rows = [(;
        harness = row.harness,
        model = row.model,
        min_ess_rate = row.min_ess_rate,
    ) for row in rows if row.min_ess_rate !== missing]
    harness_order = unique(getproperty.(plot_rows, :harness))
    ess_rate_plot = _plot_block(
        data(plot_rows) *
        mapping(
            :harness => sorter(harness_order) => "Sampler harness",
            :min_ess_rate => "Min ESS / s";
            color = :model => "Model",
        ) *
        visual(Scatter, markersize = 120) *
        config(height = 280, scales = scales(Y = (; scale = log10)));
        id = "probprog-mcmc-ess-rate",
        title = "Minimum effective samples per second",
        description = "Per measured harness and model; the rate axis is " *
            "logarithmic. The rejected ProbProg MNIST cell has no rate and " *
            "appears only in the table.")

    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        ess_rate_plot,
        _result_table(rows, columns; id = "probprog-mcmc-performance",
            title = "NUTS sampling performance and diagnostics",
            note = "One wall-clock run per harness covering warmup plus " *
                "retained draws. Compilation (ProbProg: XLA; native harnesses: " *
                "Julia JIT warm start) is the separate first column and is " *
                "outside the sampling time."),
        _result_table(eight_schools_rows, eight_schools_columns;
            id = "probprog-mcmc-eight-schools-moments",
            title = "Eight Schools posterior agreement",
            note = "Constrained-scale posterior summaries computed from each " *
                "harness's packed draws."),
        _result_table(mnist_rows, mnist_columns;
            id = "probprog-mcmc-mnist-moments",
            title = "MNIST posterior agreement",
            note = "The first coefficient of the packed coefficient vector, " *
                "per harness."),
        _static_block(provenance),
    ])
end

"""Render the sampling harnesses and Turing twins the ProbProg comparison executes."""
render_probprog_mcmc_baselines() = _render_benchmark_baselines(
    "These are the exact sampling harnesses and Turing twin models the " *
    "comparison executes, shown verbatim from " *
    "`benchmark/probprog_mcmc_comparison_body.jl` (extracted from the " *
    "executed file at docs-build time).",
    "probprog_mcmc_comparison_body.jl",
    (("ProbProg compiled NUTS harness", "probprog"),
     ("AdvancedHMC harness over the RK density", "advancedhmc"),
     ("Turing NUTS harness", "turing-sampling"),
     ("Turing Eight Schools twin", "turing-eight-schools"),
     ("Turing MNIST twin", "turing-mnist")))
