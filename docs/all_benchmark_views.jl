const _ALLBENCH_OUTCOME_LABELS = Dict(
    "joint" => "Joint density",
    "prior" => "Prior density",
    "likelihood" => "Summed likelihood",
    "pointwise" => "Pointwise likelihood",
)

_allbench_compact_number(value) = _mnist_compact_number(value)
_allbench_duration(value) = _mnist_focused_duration(value)
_allbench_bytes(value) = _mnist_focused_bytes(value)
_allbench_interpretation(value) = _mnist_relative_interpretation(value)

_allbench_optional(value, formatter) =
    value === missing ? h.span("—"; class = "rk-result-unsupported") : formatter(value)

function _allbench_slug(value)
    slug = lowercase(replace(String(value), r"[^A-Za-z0-9]+" => "-"))
    strip(slug, '-')
end

function _allbench_checked_receipt(path, schema; synchronous = false)
    receipt = TOML.parsefile(path)
    get(receipt, "schema", "") == schema ||
        error("unexpected benchmark receipt schema at $path")
    pins = get(receipt, "pins", Dict{String,Any}())
    haskey(pins, "reactivekernels_dirty") &&
        get(pins, "reactivekernels_dirty", true) != false &&
        error("benchmark receipt was produced from a dirty checkout")
    if synchronous
        protocol = receipt["protocol"]
        get(protocol, "reactant_sync", false) ||
            error("Reactant benchmark receipt is not synchronous")
        get(protocol, "reactant_transfers_included", false) &&
            error("Reactant benchmark receipt includes transfers in timing")
    end
    receipt
end

function _allbench_comparison_table(rows; id, title, baseline_note,
                                    leading_columns = (), extra_columns = (),
                                    runtime_label = "Runtime",
                                    baseline_runtime_label = "Baseline runtime")
    columns = Any[leading_columns...]
    push!(columns,
        _column(:series, "Implementation / configuration"),
        _column(:runtime_ns, runtime_label;
            format = (value, _) -> _allbench_duration(value)),
        _column(:baseline_label, "Baseline"),
        _column(:baseline_ns, baseline_runtime_label;
            format = (value, _) -> _allbench_duration(value)),
        _column(:relative_time, "Runtime ÷ baseline";
            format = (value, _) -> string(round(value; digits = 2), "×")),
        _column(:relative_time, "Interpretation";
            format = (value, _) -> _allbench_interpretation(value)),
    )
    append!(columns, extra_columns)
    if any(row -> row.median_bytes !== missing, rows)
        push!(columns, _column(:median_bytes, "Allocated";
            format = (value, _) -> _allbench_optional(value, _allbench_bytes)))
    end
    if any(row -> row.median_allocs !== missing, rows)
        push!(columns, _column(:median_allocs, "Allocations";
            format = (value, _) -> _allbench_optional(value, string)))
    end
    _result_table(rows, Tuple(columns); id, title,
        note = "$baseline_note. Display values use three significant digits; " *
               "the checked-in receipt retains exact values and raw rounds.")
end

function _allbench_snapshot_plot(rows; id, title, baseline_note)
    series_order = unique(getproperty.(rows, :series))
    spec = data(rows) *
        mapping(
            :series => sorter(series_order) => "Implementation / configuration",
            :relative_time => "Runtime ÷ baseline";
            color = :implementation => "Implementation",
        ) *
        visual(Scatter, markersize = 120) *
        config(height = 280, scales = scales(Y = (; scale = log10)))
    _plot_block(spec; id, title,
        description = "$baseline_note; 1.00× is parity and lower is faster. " *
                      "The ratio axis is logarithmic.")
end

function _allbench_series_plot(rows; id, title, baseline_note,
                               x_key = :n, x_label = "N", log_x = true)
    mapping_spec = mapping(
        x_key => x_label,
        :relative_time => "Runtime ÷ baseline";
        color = :series => "Implementation / configuration",
    )
    scale_config = log_x ?
        scales(X = (; scale = log10), Y = (; scale = log10)) :
        scales(Y = (; scale = log10))
    spec = data(rows) * mapping_spec * visual(ScatterLines) *
        config(height = 300, scales = scale_config)
    _plot_block(spec; id, title,
        description = "$baseline_note; 1.00× is parity and lower is faster. " *
                      "The ratio axis is logarithmic.")
end

function _allbench_series_sections(rows; id_prefix, summary, provenance,
                                   panel_order = unique(getproperty.(rows, :panel)),
                                   x_key = :n, x_label = "N", log_x = true,
                                   leading_columns = (), extra_columns = (),
                                   runtime_label = "Runtime",
                                   baseline_runtime_label = "Baseline runtime")
    blocks = Any[Markdown.Paragraph(Any[summary])]
    for panel in panel_order
        selected = filter(row -> row.panel == panel, rows)
        isempty(selected) && continue
        baseline_note = "Baseline: matched row named in the table"
        slug = _allbench_slug(panel)
        push!(blocks,
            _allbench_series_plot(selected;
                id = "$id_prefix-$slug-relative",
                title = "$panel — relative runtime", baseline_note,
                x_key, x_label, log_x),
            _allbench_comparison_table(selected;
                id = "$id_prefix-$slug-table", title = "$panel — comparison",
                baseline_note, leading_columns, extra_columns,
                runtime_label, baseline_runtime_label),
        )
    end
    push!(blocks, Markdown.Paragraph(Any[Markdown.Italic(Any[provenance])]))
    Markdown.MD(blocks)
end

function _allbench_snapshot_sections(rows; id_prefix, profile_label,
                                     summary, provenance, panel_order)
    blocks = Any[Markdown.Paragraph(Any[summary])]
    for panel in panel_order
        selected = filter(row -> row.panel == panel, rows)
        isempty(selected) && continue
        baseline_labels = unique(getproperty.(selected, :baseline_label))
        baseline_note = length(baseline_labels) == 1 ?
            "Baseline: $(only(baseline_labels))" :
            "Baseline: the matched row named in the table"
        slug = _allbench_slug(panel)
        push!(blocks,
            _allbench_snapshot_plot(selected;
                id = "$id_prefix-$slug-relative",
                title = "$profile_label — $panel relative runtime",
                baseline_note),
            _allbench_comparison_table(selected;
                id = "$id_prefix-$slug-table",
                title = "$profile_label — $panel", baseline_note),
        )
    end
    push!(blocks, Markdown.Paragraph(Any[Markdown.Italic(Any[provenance])]))
    Markdown.MD(blocks)
end

function _allbench_push_measurement_rows!(rows, measurement, panel,
                                          entries, baseline_key,
                                          baseline_label)
    haskey(measurement, baseline_key) || return rows
    baseline = measurement[baseline_key]
    baseline_ns = Float64(baseline["median_ns"])
    n = Int(measurement["n"])
    for (key, series, implementation) in entries
        haskey(measurement, key) || continue
        result = measurement[key]
        runtime_ns = Float64(result["median_ns"])
        push!(rows, (;
            panel,
            n,
            series,
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
    rows
end

const _ALLBENCH_NATIVE_DISTRIBUTIONS = (
    ("rk_native", "RK shared object", "ReactiveKernels"),
    ("rk_direct_native", "RK one-off control", "ReactiveKernels"),
    ("distributions_native", "Distributions", "Distributions"),
    ("probability_measures_native", "ProbabilityMeasures", "ProbabilityMeasures"),
)

const _ALLBENCH_REACTANT_DISTRIBUTIONS = (
    ("rk_reactant", "RK + Reactant", "ReactiveKernels"),
    ("distributions_reactant", "Distributions + Reactant", "Distributions"),
    ("probability_measures_reactant", "ProbabilityMeasures + Reactant", "ProbabilityMeasures"),
)

function _render_distribution_benchmarks_focused()
    receipt = _allbench_checked_receipt(
        _DISTRIBUTION_RECEIPT_PATH, "distribution-logdensity-v1";
        synchronous = true)
    rows = NamedTuple[]
    for measurement in receipt["measurements"]
        _allbench_push_measurement_rows!(rows, measurement, "Native Julia",
            _ALLBENCH_NATIVE_DISTRIBUTIONS, "rk_native", "RK shared object")
        _allbench_push_measurement_rows!(rows, measurement, "Reactant CPU",
            _ALLBENCH_REACTANT_DISTRIBUTIONS, "rk_reactant", "RK + Reactant")
    end
    pins = receipt["pins"]
    summary = "Native and compiled comparisons are separated so each row uses " *
        "an execution-matched RK baseline. Absolute timings and allocations stay " *
        "beside the normalized ratio rather than in a second exhaustive table."
    provenance = "Receipt `$(basename(_DISTRIBUTION_RECEIPT_PATH))`; RK " *
        "`$(pins["reactivekernels_sha"])`; Reactant $(pins["reactant_version"]); " *
        "Julia $(pins["julia_version"]); $(receipt["environment"]["cpu"])."
    _allbench_series_sections(rows;
        id_prefix = "normal-logdensity", summary, provenance,
        panel_order = ("Native Julia", "Reactant CPU"),
        leading_columns = (_column(:n, "N"),))
end

function _allbench_distribution_family_rows(receipt, family_label)
    rows = NamedTuple[]
    for measurement in receipt["measurements"]
        panel = family_label[String(measurement["family"])]
        _allbench_push_measurement_rows!(rows, measurement, panel,
            _ALLBENCH_NATIVE_DISTRIBUTIONS, "rk_native", "RK native")
        _allbench_push_measurement_rows!(rows, measurement, panel,
            _ALLBENCH_REACTANT_DISTRIBUTIONS, "rk_reactant", "RK + Reactant")
    end
    rows
end

function _render_scalar_gallery_benchmarks_focused()
    receipt = _allbench_checked_receipt(
        _SCALAR_GALLERY_RECEIPT_PATH, "scalar-distribution-gallery-v1";
        synchronous = true)
    family_label = Dict(
        "cauchy_location_scale" => "Cauchy",
        "laplace_location_scale" => "Laplace",
        "bernoulli_logit" => "Bernoulli",
        "lognormal_logscale" => "LogNormal",
        "exponential_logscale" => "Exponential",
        "geometric_logit" => "Geometric",
        "uniform_bounded" => "Uniform",
    )
    rows = _allbench_distribution_family_rows(receipt, family_label)
    pins = receipt["pins"]
    summary = "Each scalar family now has its own plot and compact table. " *
        "Native rows use RK native as baseline; compiled rows use RK + Reactant."
    provenance = "Receipt `$(basename(_SCALAR_GALLERY_RECEIPT_PATH))`; RK " *
        "`$(pins["reactivekernels_sha"])`; Reactant $(pins["reactant_version"]); " *
        "Julia $(pins["julia_version"]); $(receipt["environment"]["cpu"])."
    panel_order = Tuple(family_label[key] for key in (
        "cauchy_location_scale", "laplace_location_scale", "bernoulli_logit",
        "lognormal_logscale", "exponential_logscale", "geometric_logit",
        "uniform_bounded"))
    _allbench_series_sections(rows;
        id_prefix = "scalar-distributions", summary, provenance, panel_order,
        leading_columns = (_column(:n, "N"),))
end

function _render_structured_distribution_benchmarks_focused()
    receipt = _allbench_checked_receipt(
        _STRUCTURED_DISTRIBUTION_RECEIPT_PATH,
        "structured-distribution-logdensity-v1"; synchronous = true)
    rows = _allbench_distribution_family_rows(
        receipt, Dict("mvnormal_cholesky" => "MVN (Cholesky)"))
    pins = receipt["pins"]
    summary = "The structured comparison uses RK native and RK + Reactant as " *
        "separate matched baselines; unsupported compiled library cells remain " *
        "in the receipt rather than appearing as fabricated timings."
    provenance = "Receipt `$(basename(_STRUCTURED_DISTRIBUTION_RECEIPT_PATH))`; " *
        "RK `$(pins["reactivekernels_sha"])`; Reactant " *
        "$(pins["reactant_version"]); Julia $(pins["julia_version"]); " *
        "$(receipt["environment"]["cpu"])."
    _allbench_series_sections(rows;
        id_prefix = "structured-distributions", summary, provenance,
        panel_order = ("MVN (Cholesky)",),
        leading_columns = (_column(:n, "N"),))
end

function _render_batched_benchmarks_focused()
    receipt = _allbench_checked_receipt(
        _DISTRIBUTION_RECEIPT_PATH, "distribution-logdensity-v1";
        synchronous = true)
    rows = NamedTuple[]
    native_entries = (
        ("rk_native", "Legacy plate", "Legacy plate"),
        ("rk_authored_native", "Authored return", "Authored return"),
    )
    reactant_entries = (
        ("rk_reactant", "Legacy plate", "Legacy plate"),
        ("rk_authored_reactant", "Authored return", "Authored return"),
    )
    for measurement in receipt["measurements"]
        _allbench_push_measurement_rows!(rows, measurement, "Native Julia",
            native_entries, "rk_native", "legacy plate / native")
        _allbench_push_measurement_rows!(rows, measurement, "Reactant CPU",
            reactant_entries, "rk_reactant", "legacy plate / Reactant")
    end
    pins = receipt["pins"]
    summary = "Authored-return parity is separated into native and Reactant " *
        "profiles. Every size reports authored runtime relative to the matching " *
        "legacy plate plus compact allocation evidence."
    provenance = "Receipt `$(basename(_DISTRIBUTION_RECEIPT_PATH))`; RK " *
        "`$(pins["reactivekernels_sha"])`; Reactant $(pins["reactant_version"]); " *
        "Julia $(pins["julia_version"]); $(receipt["environment"]["cpu"])."
    _allbench_series_sections(rows;
        id_prefix = "batched-authored", summary, provenance,
        panel_order = ("Native Julia", "Reactant CPU"),
        leading_columns = (_column(:n, "N"),))
end

function _render_distribution_gradient_benchmarks_focused()
    receipt = _allbench_checked_receipt(
        _DISTRIBUTION_GRADIENT_RECEIPT_PATH, "distribution-gradient-v1")
    protocol = receipt["protocol"]
    get(protocol, "preparation_in_timed_region", true) == false ||
        error("distribution-gradient receipt includes preparation")
    family_label = _DISTRIBUTION_GRADIENT_FAMILY_LABELS
    rows = NamedTuple[]
    for measurement in receipt["measurements"]
        haskey(measurement, "returned_gradient") || continue
        baseline = measurement["returned_gradient"]
        baseline_ns = Float64(baseline["median_ns"])
        panel = family_label[String(measurement["family"])]
        for (key, series) in _DISTRIBUTION_GRADIENT_SURFACES
            haskey(measurement, key) || continue
            result = measurement[key]
            runtime_ns = Float64(result["median_ns"])
            push!(rows, (;
                panel,
                n = Int(measurement["n"]),
                series,
                implementation = series,
                runtime_ns,
                baseline_ns,
                baseline_label = "ad_gradient for the same family and N",
                relative_time = runtime_ns / baseline_ns,
                median_bytes = Int(result["median_bytes"]),
                median_allocs = Int(result["median_allocs"]),
            ))
        end
    end
    panels = (
        "Normal plate", "Cauchy", "Laplace", "Bernoulli", "LogNormal",
        "Exponential", "Geometric", "Uniform", "MVN (Cholesky)",
    )
    pins = receipt["pins"]
    summary = "Each distribution family now isolates the returned-gradient and " *
        "caller-owned value+gradient surfaces. Runtime is normalized to " *
        "`ad_gradient` at the same family and size; bytes and allocation counts " *
        "remain beside the timing comparison."
    provenance = "Receipt `$(basename(_DISTRIBUTION_GRADIENT_RECEIPT_PATH))`; " *
        "RK `$(pins["reactivekernels_sha"])`; Enzyme " *
        "$(pins["enzyme_version"]); Julia $(pins["julia_version"]); " *
        "$(receipt["environment"]["cpu"])."
    _allbench_series_sections(rows;
        id_prefix = "distribution-gradient", summary, provenance,
        panel_order = panels, leading_columns = (_column(:n, "N"),))
end

function _render_distribution_amortization_focused()
    receipt = _allbench_checked_receipt(
        _DISTRIBUTION_RECEIPT_PATH, "distribution-logdensity-v1";
        synchronous = true)
    direct = only(row for row in receipt["measurements"] if Int(row["n"]) == 1)
    baseline_ns = Float64(direct["rk_reactant"]["median_ns"])
    rows = map(receipt["reactant_amortization"]) do row
        result = row["rk_reactant_replicated"]
        runtime_ns = Float64(row["median_ns_per_evaluation"])
        (;
            panel = "Reactant replica amortization",
            replicas = Int(row["replicas"]),
            series = "Batched / evaluation",
            implementation = "Reactant replica",
            runtime_ns,
            baseline_ns,
            baseline_label = "one direct synchronous compiled evaluation",
            relative_time = runtime_ns / baseline_ns,
            whole_call_ns = Float64(result["median_ns"]),
            median_bytes = Int(result["median_bytes"]),
            median_allocs = Int(result["median_allocs"]),
        )
    end
    pins = receipt["pins"]
    summary = "The replica view now plots normalized per-evaluation cost and " *
        "keeps whole-call cost and host allocation evidence in the same table."
    provenance = "Receipt `$(basename(_DISTRIBUTION_RECEIPT_PATH))`; RK " *
        "`$(pins["reactivekernels_sha"])`; Reactant $(pins["reactant_version"]); " *
        "Julia $(pins["julia_version"]); $(receipt["environment"]["cpu"])."
    _allbench_series_sections(rows;
        id_prefix = "normal-reactant-amortization", summary, provenance,
        panel_order = ("Reactant replica amortization",),
        x_key = :replicas, x_label = "Independent evaluations / call",
        leading_columns = (_column(:replicas, "Evaluations / call"),),
        extra_columns = (_column(:whole_call_ns, "Whole-call runtime";
            format = (value, _) -> _allbench_duration(value)),),
        runtime_label = "Runtime / evaluation")
end

function _render_eval_throughput_focused(path)
    receipt = TOML.parsefile(path)
    get(receipt, "schema", "") == "eval-throughput-v1" ||
        error("unexpected evaluation-throughput receipt schema")
    protocol = receipt["protocol"]
    get(protocol, "reactant_sync", false) ||
        error("evaluation-throughput Reactant rows are not synchronous")
    get(protocol, "reactant_transfers_included", true) &&
        error("evaluation-throughput timing includes transfers")
    measurements = receipt["measurements"]
    practicalbayes = _practicalbayes_receipt()
    practicalbayes_measurements = practicalbayes["eval_measurements"]
    mode_label = Dict(
        "primal" => "Primal log density",
        "gradient" => "Value + gradient",
        "gq" => "Pointwise generated quantities",
    )
    entries = (
        ("reactivekernels", "native", "RK native", "ReactiveKernels"),
        ("reactivekernels", "reactant", "RK + Reactant", "ReactiveKernels"),
        ("turing", "native", "Turing native", "Turing"),
    )
    rows = NamedTuple[]
    for mode in ("primal", "gradient", "gq"), n in (16, 256, 4096)
        baseline = only(row for row in measurements
            if row["implementation"] == "reactivekernels" &&
               row["variant"] == "native" && row["mode"] == mode &&
               Int(row["size"]) == n)
        baseline_ns = Float64(baseline["median_ns"])
        for (implementation_key, variant, series, implementation) in entries
            candidates = filter(row ->
                row["implementation"] == implementation_key &&
                row["variant"] == variant && row["mode"] == mode &&
                Int(row["size"]) == n, measurements)
            isempty(candidates) && continue
            result = only(candidates)
            runtime_ns = Float64(result["median_ns"])
            push!(rows, (;
                panel = mode_label[mode],
                n,
                series,
                implementation,
                runtime_ns,
                baseline_ns,
                baseline_label = "RK native at the same mode and N",
                relative_time = runtime_ns / baseline_ns,
                median_bytes = missing,
                median_allocs = missing,
            ))
        end
        candidates = filter(row ->
            row["implementation"] == "practical_bayes" &&
            row["variant"] == "native" && row["mode"] == mode &&
            Int(row["size"]) == n && row["state"] == "supported",
            practicalbayes_measurements)
        if !isempty(candidates)
            result = only(candidates)["result"]
            runtime_ns = Float64(result["min_ns"])
            push!(rows, (;
                panel = mode_label[mode],
                n,
                series = "PracticalBayes native",
                implementation = "PracticalBayes",
                runtime_ns,
                baseline_ns,
                baseline_label = "RK native at the same mode and N",
                relative_time = runtime_ns / baseline_ns,
                median_bytes = missing,
                median_allocs = missing,
            ))
        end
    end
    pins = receipt["pins"]
    summary = "Primal, value+gradient, and generated-quantity latency are " *
        "separate comparisons. Every point is normalized to RK native at the " *
        "same mode and position size; unsupported Turing+Reactant and " *
        "PracticalBayes+Reactant cells remain omitted."
    practicalbayes_pins = practicalbayes["pins"]
    provenance = "Receipt `$(basename(path))`; RK " *
        "`$(pins["reactivekernels_sha"])`; Turing $(pins["turing_version"]); " *
        "Reactant $(pins["reactant_version"]); Julia $(pins["julia_version"]); " *
        "$(receipt["environment"]["cpu"]). PracticalBayes " *
        "$(practicalbayes_pins["practicalbayes_version"]) rows come from " *
        "`$(basename(_PRACTICALBAYES_RECEIPT_PATH))`, resolved separately " *
        "because of its incompatible Bijectors pin."
    _allbench_series_sections(rows;
        id_prefix = "eval-throughput", summary, provenance,
        panel_order = ("Primal log density", "Value + gradient",
                       "Pointwise generated quantities"),
        leading_columns = (_column(:n, "Position size"),))
end

function _render_eval_throughput_amortization_focused()
    receipt = TOML.parsefile(_eval_throughput_receipt())
    get(receipt, "schema", "") == "eval-throughput-v1" ||
        error("unexpected evaluation-throughput receipt schema")
    replicas = Int(receipt["protocol"]["replicas"])
    measurements = receipt["measurements"]
    mode_label = Dict(
        "primal" => "Primal log density",
        "gradient" => "Value + gradient",
        "gq" => "Pointwise generated quantities",
    )
    rows = NamedTuple[]
    for mode in ("primal", "gradient", "gq"), n in (16, 256, 4096)
        direct = only(row for row in measurements
            if Int(row["size"]) == n && row["mode"] == mode &&
               row["implementation"] == "reactivekernels" &&
               row["variant"] == "reactant")
        replicated = only(row for row in measurements
            if Int(row["size"]) == n && row["mode"] == mode &&
               row["implementation"] == "reactivekernels" &&
               row["variant"] == "reactant_replicated")
        baseline_ns = Float64(direct["median_ns"])
        runtime_ns = Float64(replicated["median_ns"])
        push!(rows, (;
            panel = mode_label[mode],
            n,
            series = "$replicas-position batch / position",
            implementation = "Reactant replica",
            runtime_ns,
            baseline_ns,
            baseline_label = "direct Reactant call at the same mode and N",
            relative_time = runtime_ns / baseline_ns,
            batch_ns = Float64(replicated["median_batch_ns"]),
            median_bytes = missing,
            median_allocs = missing,
        ))
    end
    pins = receipt["pins"]
    summary = "Replica throughput is split by operation. The plot shows " *
        "per-position batch cost relative to the matched direct Reactant call; " *
        "the table keeps the whole-batch measurement explicit."
    provenance = "Receipt `$(basename(_eval_throughput_receipt()))`; RK " *
        "`$(pins["reactivekernels_sha"])`; Reactant $(pins["reactant_version"]); " *
        "Julia $(pins["julia_version"]); $(receipt["environment"]["cpu"])."
    _allbench_series_sections(rows;
        id_prefix = "eval-throughput-amortization", summary, provenance,
        panel_order = ("Primal log density", "Value + gradient",
                       "Pointwise generated quantities"),
        leading_columns = (_column(:n, "Position size"),),
        extra_columns = (_column(:batch_ns, "Whole-batch runtime";
            format = (value, _) -> _allbench_duration(value)),),
        runtime_label = "Batch / position")
end

const _ALLBENCH_EIGHT_SCHOOLS_OUTCOMES = (
    "joint", "prior", "likelihood", "pointwise",
)

function _render_eight_schools_longform_focused(path, schema;
                                                mode, id_prefix,
                                                profile_label)
    receipt = _allbench_checked_receipt(path, schema)
    protocol = receipt["protocol"]
    Tuple(String.(protocol["input_boundaries"])) == _EIGHT_SCHOOLS_BOUNDARIES ||
        error("Eight Schools input-boundary inventory changed")
    Tuple(String.(protocol["outcomes"])) == _EIGHT_SCHOOLS_OUTCOMES ||
        error("Eight Schools outcome inventory changed")
    get(protocol, "preparation_in_timed_region", true) == false ||
        error("Eight Schools preparation entered steady-state timing")

    entries = mode == :primal ? (
        ("rk", "primal_native", "ReactiveKernels", "ReactiveKernels"),
        ("manual_julia", "manual_primal", "Manual Julia", "Manual Julia"),
        ("turing", "turing_primal", "Turing", "Turing"),
    ) : (
        ("rk", "ad_native", "ReactiveKernels + Enzyme", "ReactiveKernels"),
        ("manual_julia", "manual_ad", "Manual Julia + Enzyme", "Manual Julia"),
        ("turing", "turing_ad", "Turing + Enzyme", "Turing"),
    )
    measurements = receipt["measurements"]
    practicalbayes = _practicalbayes_receipt()
    practicalbayes_measurements = practicalbayes["model_measurements"]
    models = unique(String(row["model"]) for row in measurements)
    rows = NamedTuple[]
    for model in models, outcome in _ALLBENCH_EIGHT_SCHOOLS_OUTCOMES
        baseline_candidates = filter(row ->
            row["model"] == model && row["boundary"] == "packed_unconstrained" &&
            row["outcome"] == outcome && row["provider"] == "rk" &&
            row["configuration"] == (mode == :primal ? "primal_native" : "ad_native") &&
            row["state"] == "supported", measurements)
        isempty(baseline_candidates) && continue
        baseline = only(baseline_candidates)
        baseline_ns = _mnist_focused_runtime(baseline["result"])
        for (provider, configuration, label, implementation) in entries
            candidates = filter(row ->
                row["model"] == model && row["boundary"] == "packed_unconstrained" &&
                row["outcome"] == outcome && row["provider"] == provider &&
                row["configuration"] == configuration &&
                row["state"] == "supported", measurements)
            isempty(candidates) && continue
            result = only(candidates)["result"]
            runtime_ns = _mnist_focused_runtime(result)
            push!(rows, (;
                panel = _ALLBENCH_OUTCOME_LABELS[outcome],
                model = titlecase(replace(model, '_' => ' ')),
                series = "$label — $(titlecase(replace(model, '_' => ' ')))",
                implementation,
                runtime_ns,
                baseline_ns,
                baseline_label = "ReactiveKernels / $(titlecase(replace(model, '_' => ' ')))",
                relative_time = runtime_ns / baseline_ns,
                median_bytes = Int(result["median_bytes"]),
                median_allocs = Int(result["median_allocs"]),
            ))
        end
        practicalbayes_configuration =
            mode == :primal ? "practicalbayes_primal" : "practicalbayes_ad"
        candidates = filter(row ->
            row["workload"] == "eight_schools" && row["model"] == model &&
            row["boundary"] == "packed_unconstrained" &&
            row["outcome"] == outcome &&
            row["configuration"] == practicalbayes_configuration &&
            row["state"] == "supported", practicalbayes_measurements)
        if !isempty(candidates)
            result = only(candidates)["result"]
            runtime_ns = Float64(result["median_ns"])
            model_label = titlecase(replace(model, '_' => ' '))
            push!(rows, (;
                panel = _ALLBENCH_OUTCOME_LABELS[outcome],
                model = model_label,
                series = "PracticalBayes — $model_label",
                implementation = "PracticalBayes",
                runtime_ns,
                baseline_ns,
                baseline_label = "ReactiveKernels / $model_label",
                relative_time = runtime_ns / baseline_ns,
                median_bytes = Int(result["median_bytes"]),
                median_allocs = Int(result["median_allocs"]),
            ))
        end
    end

    pins = receipt["pins"]
    operation = mode == :primal ? "primal evaluation" : "value + gradient"
    summary = "The headline comparison uses the shared packed-unconstrained " *
        "boundary and separates joint, prior, summed-likelihood, and pointwise " *
        "outcomes. Each model is normalized to its matching ReactiveKernels " *
        "$operation baseline. The complete boundary/configuration matrix, " *
        "unsupported cells, setup costs, and raw rounds remain in the receipt."
    practicalbayes_pins = practicalbayes["pins"]
    provenance = "Receipt `$(basename(path))`; RK `$(pins["reactivekernels_sha"])`; " *
        "Turing $(get(pins, "turing_version", "not applicable")); Julia " *
        "$(pins["julia_version"]); $(receipt["environment"]["cpu"]). " *
        "PracticalBayes $(practicalbayes_pins["practicalbayes_version"]) " *
        "rows come from `$(basename(_PRACTICALBAYES_RECEIPT_PATH))`, resolved " *
        "separately because of its incompatible Bijectors pin."
    _allbench_snapshot_sections(rows;
        id_prefix, profile_label, summary, provenance,
        panel_order = Tuple(_ALLBENCH_OUTCOME_LABELS[key]
            for key in _ALLBENCH_EIGHT_SCHOOLS_OUTCOMES))
end

function _render_eight_schools_reactant_longform_focused(path, schema;
                                                         mode, id_prefix,
                                                         profile_label)
    receipt = _allbench_checked_receipt(path, schema)
    protocol = receipt["protocol"]
    get(protocol, "source_reused", false) ||
        error("Eight Schools Reactant receipt does not reuse the authored source")
    get(protocol, "reactant_sync", false) ||
        error("Eight Schools Reactant receipt is not synchronous")
    get(protocol, "reactant_transfers_in_timed_region", true) == false ||
        error("Eight Schools Reactant timing includes transfers")
    get(protocol, "reactant_compile_time_in_timed_region", true) == false ||
        error("Eight Schools Reactant timing includes compilation")
    get(protocol, "reactant_readback_in_timed_region", true) == false ||
        error("Eight Schools Reactant timing includes readback")

    configurations = mode == :primal ? (
        "primal_reactant" => "Runtime inputs",
        "primal_reactant_bound" => "Bound inputs",
    ) : (
        "ad_reactant" => "Runtime inputs",
        "ad_reactant_bound" => "Bound inputs",
    )
    rows = NamedTuple[]
    for measurement in receipt["measurements"]
        measurement["boundary"] == "packed_unconstrained" || continue
        measurement["state"] == "supported" || continue
        label = get(Dict(configurations), String(measurement["configuration"]), nothing)
        isnothing(label) && continue
        result = measurement["result"]
        native = measurement["native_control"]
        runtime_ns = _mnist_focused_runtime(result)
        baseline_ns = _mnist_focused_runtime(native)
        model = titlecase(replace(String(measurement["model"]), '_' => ' '))
        outcome = String(measurement["outcome"])
        push!(rows, (;
            panel = _ALLBENCH_OUTCOME_LABELS[outcome],
            model,
            series = "$label — $model",
            implementation = "ReactiveKernels + Reactant",
            runtime_ns,
            baseline_ns,
            baseline_label = "matched native RK / $(lowercase(label)) / $model",
            relative_time = runtime_ns / baseline_ns,
            median_bytes = missing,
            median_allocs = missing,
        ))
    end

    pins = receipt["pins"]
    operation = mode == :primal ? "primal" : "value + gradient"
    summary = "Synchronous Reactant $operation timings are separated by outcome " *
        "and normalized to the embedded native control for the same model and " *
        "binding configuration. Compilation, transfers, first execution, and " *
        "readback are excluded; the full capability matrix and compiler " *
        "diagnostics remain in the receipt."
    provenance = "Receipt `$(basename(path))`; RK `$(pins["reactivekernels_sha"])`; " *
        "Reactant $(pins["reactant_version"]); Julia $(pins["julia_version"]); " *
        "$(receipt["environment"]["cpu"])."
    _allbench_snapshot_sections(rows;
        id_prefix, profile_label, summary, provenance,
        panel_order = Tuple(_ALLBENCH_OUTCOME_LABELS[key]
            for key in _ALLBENCH_EIGHT_SCHOOLS_OUTCOMES))
end

function _render_nuts_g7_benchmark_focused()
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
    rk_steps = Float64(medians["RK"])
    rows = [(;
        panel = "G7 work-normalized throughput",
        series = sampler,
        sampler,
        implementation = sampler,
        steps_per_second = Float64(medians[sampler]),
        baseline_steps_per_second = rk_steps,
        baseline_label = "RK",
        relative_time = rk_steps / Float64(medians[sampler]),
        grads = Int(work[sampler]["grads"]),
        steps = Int(work[sampler]["steps"]),
        median_bytes = missing,
        median_allocs = missing,
    ) for sampler in sampler_order]

    baseline_note = "Baseline: RK work-normalized time per leapfrog step"
    plot_rows = [(; row..., runtime_ns = row.relative_time,
        baseline_ns = 1.0) for row in rows]
    columns = (
        _column(:sampler, "Sampler"),
        _column(:steps_per_second, "Median leapfrog steps/s";
            format = (value, _) -> _allbench_compact_number(value)),
        _column(:baseline_steps_per_second, "RK baseline steps/s";
            format = (value, _) -> _allbench_compact_number(value)),
        _column(:relative_time, "Time / leapfrog step ÷ RK";
            format = (value, _) -> string(round(value; digits = 2), "×")),
        _column(:relative_time, "Interpretation";
            format = (value, _) -> _allbench_interpretation(value)),
        _column(:grads, "Gradients"),
        _column(:steps, "Leapfrog steps"),
    )
    env = receipt["env"]
    pins = receipt["pins"]
    summary = "The comparison is normalized by completed leapfrog work rather " *
        "than raw transition time. A ratio below 1× means the implementation " *
        "executes each leapfrog step faster than RK; exact work counts remain " *
        "beside the compact three-significant-digit throughput."
    caption = "Frozen $(env["target"]), dimension $(env["dimension"]), " *
        "unit diagonal mass, step size $(env["stepsize"]), and one shared " *
        "DifferentiationInterface + Enzyme gradient. No ESS claim is made."
    provenance = "Receipt `$(basename(_NUTS_G7_RECEIPT_PATH))`; RK " *
        "`$(pins["reactivekernels_sha"])`; NUTS.jl " *
        "`$(pins["nuts_jl_sha"])`; Julia $(pins["julia_version"])."
    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _allbench_snapshot_plot(plot_rows;
            id = "nuts-g7-relative", title = "G7 work-normalized relative cost",
            baseline_note = "$baseline_note. $caption"),
        _result_table(rows, columns; id = "nuts-g7-table",
            title = "G7 work-normalized comparison",
            note = "$baseline_note. Display values use three significant digits; " *
                   "the receipt retains exact medians and raw rounds."),
        Markdown.Paragraph(Any[Markdown.Italic(Any[provenance])]),
    ])
end

function _render_nuts_reactant_benchmark_focused()
    receipt = _allbench_checked_receipt(
        _NUTS_REACTANT_RECEIPT_PATH, "nuts-reactant-v1")
    protocol = receipt["protocol"]
    get(protocol, "reactant_sync", false) ||
        error("Reactant NUTS receipt is not synchronous")
    get(protocol, "host_device_transfers_in_steady_state", true) == false ||
        error("Reactant NUTS receipt includes transfers")
    acceptance = receipt["acceptance"]
    for key in ("per_transition_tolerance_bounded_phase_diagnostic_parity",
                "exact_control_and_random_consumption_parity",
                "matched_control_flow_corpus", "parity_screening_reported",
                "all_overflow_flags_zero")
        get(acceptance, key, false) ||
            error("Reactant NUTS receipt failed acceptance: $key")
    end
    get(acceptance, "stablehlo_while_count", 0) == 1 ||
        error("Reactant NUTS receipt does not contain one stablehlo.while")

    medians = receipt["medians"]
    native_ns = Float64(medians["native_transition_ms"]) * 1e6
    reactant_ns = Float64(medians["reactant_transition_ms"]) * 1e6
    native_steps = Float64(medians["native_steps_per_second"])
    reactant_steps = Float64(medians["reactant_steps_per_second"])
    rows = [
        (panel = "Matched transition", series = "Native source compiler",
         implementation = "ReactiveKernels", runtime_ns = native_ns,
         baseline_ns = native_ns, baseline_label = "native source compiler",
         relative_time = 1.0, steps_per_second = native_steps,
         median_bytes = missing, median_allocs = missing),
        (panel = "Matched transition", series = "Reactant (CPU)",
         implementation = "ReactiveKernels + Reactant", runtime_ns = reactant_ns,
         baseline_ns = native_ns, baseline_label = "native source compiler",
         relative_time = reactant_ns / native_ns,
         steps_per_second = reactant_steps,
         median_bytes = missing, median_allocs = missing),
    ]
    compilation = receipt["compilation"]
    compile_rows = [
        (stage = "Native source compile + first Julia JIT",
         runtime_ns = Float64(compilation["native_seconds"]) * 1e9),
        (stage = "Native first execution",
         runtime_ns = Float64(compilation["native_first_execution_seconds"]) * 1e9),
        (stage = "Reactant host lowering",
         runtime_ns = Float64(compilation["reactant_lower_seconds"]) * 1e9),
        (stage = "Reactant XLA compile",
         runtime_ns = Float64(compilation["reactant_xla_seconds"]) * 1e9),
        (stage = "Reactant first synchronous execution",
         runtime_ns = Float64(compilation["reactant_first_execution_seconds"]) * 1e9),
    ]
    compile_columns = (
        _column(:stage, "Stage"),
        _column(:runtime_ns, "Elapsed";
            format = (value, _) -> _allbench_duration(value)),
    )
    pins = receipt["pins"]
    summary = "The matched-control transition comparison reports absolute " *
        "latency, native-relative latency, and leapfrog throughput together. " *
        "Both paths use the same authored transition and deterministic accepted " *
        "corpus; this is not an end-to-end ESS benchmark."
    provenance = "Receipt `$(basename(_NUTS_REACTANT_RECEIPT_PATH))`; RK " *
        "`$(pins["reactivekernels_sha"])`; Reactant " *
        "$(pins["reactant_version"]); Julia $(pins["julia_version"])."
    Markdown.MD(Any[
        Markdown.Paragraph(Any[summary]),
        _allbench_snapshot_plot(rows;
            id = "nuts-reactant-transition-relative",
            title = "Matched adaptive-NUTS transition relative cost",
            baseline_note = "Baseline: native source compiler"),
        _allbench_comparison_table(rows;
            id = "nuts-reactant-table",
            title = "Matched-control transition comparison",
            baseline_note = "Baseline: native source compiler",
            extra_columns = (_column(:steps_per_second, "Leapfrog steps/s";
                format = (value, _) -> _allbench_compact_number(value)),)),
        _result_table(compile_rows, compile_columns;
            id = "nuts-reactant-compile-table",
            title = "Compilation and first-call costs",
            note = "Reported separately from steady-state timing; display values " *
                   "use three significant digits."),
        Markdown.Paragraph(Any[Markdown.Italic(Any[provenance])]),
    ])
end
