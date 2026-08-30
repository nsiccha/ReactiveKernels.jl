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
const _NUTS_G7_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "nuts-g7-v1.toml",
)
const _NUTS_REACTANT_RECEIPT_PATH = joinpath(
    dirname(@__DIR__), "benchmark", "receipts", "nuts-reactant-v1.toml",
)

_node_html(node) = sprint(show, MIME"text/html"(), node; context = :limit => false)

function _interactive_block(node; class = "rk-aov-panel")
    payload = base64encode(_node_html(node))
    RawHTML("""
<ClientOnly>
  <div class="$class" v-exec-scripts="'$payload'"></div>
</ClientOnly>
""")
end

_static_block(node) = RawHTML(_node_html(node))

"""Install HTMXObjects' sorting runtime once on a page that renders result tables."""
function render_result_assets()
    Markdown.MD(Any[_interactive_block(sortable_table_js(); class = "rk-result-assets")])
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
    _static_block(h.section(; class = "rk-result-table-section")(
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
    _interactive_block(node)
end

const _TIMING_BACKENDS = (
    ("rk_native", "RK native"),
    ("distributions_native", "Distributions native"),
    ("probability_measures_native", "ProbabilityMeasures native"),
    ("rk_reactant", "RK + Reactant"),
    ("distributions_reactant", "Distributions + Reactant"),
    ("probability_measures_reactant", "ProbabilityMeasures + Reactant"),
)

const _ALLOCATION_BACKENDS = (
    ("rk_native", "RK native"),
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
    summary = "At N=$largest_n, native RK is " *
        "$(ratio("distributions_native", "rk_native"))× faster than Distributions and " *
        "$(ratio("probability_measures_native", "rk_native"))× faster than " *
        "ProbabilityMeasures. RK + Reactant is " *
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

"""Render the replicated Reactant call boundary as total and per-evaluation time."""
function render_distribution_amortization()
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
    families = ("exponential_logscale", "geometric_logit", "uniform_bounded")
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
    for family in ("mvnormal_cholesky", "stationary_ar1")
        family_support = support[family]
        get(family_support, "rk_reactant", false) ||
            error("$family does not accept the RK Reactant path")
        for library in ("distributions_reactant", "probability_measures_reactant")
            get(family_support, library, false) && continue
            isempty(get(support_errors[family], library, "")) &&
                error("$family $library is unsupported without a diagnostic")
        end
    end

    family_label = Dict(
        "mvnormal_cholesky" => "MVN (Cholesky)",
        "stationary_ar1" => "AR(1)",
    )
    measurements = receipt["measurements"]
    rows = _timing_rows(measurements;
        family = measurement -> family_label[measurement["family"]])
    largest_ar = last(filter(row -> row["family"] == "stationary_ar1", measurements))
    n = Int(largest_ar["n"])
    ar_dist_ratio = round(
        largest_ar["distributions_native"]["median_ns"] /
        largest_ar["rk_native"]["median_ns"];
        digits = 2,
    )
    ar_pm_ratio = round(
        largest_ar["probability_measures_native"]["median_ns"] /
        largest_ar["rk_native"]["median_ns"];
        digits = 2,
    )
    summary = "At N=$n, the authored O(T) AR(1) kernel is " *
        "$ar_dist_ratio× faster than the equivalent dense Distributions MvNormal and " *
        "$ar_pm_ratio× faster than the ProbabilityMeasures MvNormal on this machine. " *
        "Full-MVN Reactant cells for both comparison libraries are unsupported; " *
        "RK compiles both structured kernels."
    provenance = "Receipt pins: RK `$(first(String(pins["reactivekernels_sha"]), 10))`; " *
        "ProbabilityMeasures `$(first(String(pins["probability_measures_sha"]), 10))`; " *
        "Distributions $(pins["distributions_version"]); Reactant " *
        "$(pins["reactant_version"]); Julia $(pins["julia_version"]); " *
        "$(receipt["environment"]["cpu"])."

    blocks = Any[Markdown.Paragraph(Any[summary])]
    for label in ("MVN (Cholesky)", "AR(1)")
        family_rows = filter(row -> row.family == label, rows)
        plot_id = label == "AR(1)" ? "structured-ar1-runtime" : "structured-mvn-runtime"
        push!(blocks, _timing_plot(family_rows; id = plot_id, title = "$label runtime"))
    end
    push!(blocks,
        _result_table(rows, _timing_columns(; include_family = true);
            id = "structured-runtime-table", title = "Exact structured-kernel runtimes"),
        Markdown.Paragraph(Any[Markdown.Italic(Any[provenance])]),
    )
    Markdown.MD(blocks)
end

"""Render the batched allocation comparison from one table-shaped source."""
function render_batched_allocations()
    rows = [
        (path = "prepare", want = "logdensity", bytes = 8128,
         note = "one per-observation vector"),
        (path = "prepare_nonallocating", want = "logdensity", bytes = 0,
         note = "reused batch buffer"),
        (path = "prepare_nonallocating", want = "per_obs", bytes = 0,
         note = "reused returned buffer"),
    ]
    plot_rows = [(;
        configuration = string(
            row.path == "prepare_nonallocating" ? "nonallocating" : row.path,
            " / ", row.want,
        ),
        bytes = row.bytes,
    ) for row in rows]
    spec = data(plot_rows) *
        mapping(:configuration => "Execution path / WANT", :bytes => "Bytes (N = 1000)") *
        visual(Scatter; markersize = 80) * config(height = 280)
    columns = (
        _column(:path, "Path"; format = (value, _) -> h.code(value)),
        _column(:want, "WANT"; format = (value, _) -> h.code(value)),
        _column(:bytes, "Steady-state bytes (N = 1000)";
            format = (value, _) -> string(value, " B")),
        _column(:note, "Meaning"),
    )
    Markdown.MD(Any[
        _plot_block(spec; id = "batched-steady-state-bytes",
            title = "Steady-state allocation by execution path",
            description = "Points on the baseline are true zero-byte measurements."),
        _result_table(rows, columns; id = "batched-allocation-table",
            title = "Exact allocation values"),
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
        visual(BarPlot) *
        config(height = 280, scales = scales(Y = (; scale = log10)))
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
        (capability = "Infer effects from arbitrary Julia methods", supported = "No", boundary = "Source capture plus exact registrations only; no Julia IR/inference analysis"),
        (capability = "Generic effectful operation fallback", supported = "No", boundary = "Opaque or specialization-unsafe calls reject before executable lowering"),
        (capability = "Runtime graph mutation after preparation", supported = "No", boundary = "Cached stateless entries version out; compiled reactive programs throw version errors"),
        (capability = "Reentrant/concurrent stateful kernel instance", supported = "No", boundary = "Use one prepared mutable/cache-owning instance per independent caller"),
        (capability = "Automatic differentiation", supported = "No compiler feature", boundary = "AD may be an ordinary recipe/backend concern; the compiler does not choose or prove it"),
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
