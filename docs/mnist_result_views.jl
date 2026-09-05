const _MNIST_FOCUSED_OUTCOMES = ("joint", "prior", "likelihood", "pointwise")
const _MNIST_FOCUSED_OUTCOME_LABELS = Dict(
    "joint" => "Joint density",
    "prior" => "Prior density",
    "likelihood" => "Summed likelihood",
    "pointwise" => "Pointwise likelihood",
)

function _mnist_compact_number(value)
    rounded = round(Float64(value); sigdigits = 3)
    isinteger(rounded) ? string(Int(rounded)) : string(rounded)
end

function _mnist_focused_duration(value)
    ns = Float64(value)
    amount, unit = ns >= 1e9 ? (ns / 1e9, "s") :
        ns >= 1e6 ? (ns / 1e6, "ms") :
        ns >= 1e3 ? (ns / 1e3, "µs") : (ns, "ns")
    string(_mnist_compact_number(amount), " ", unit)
end

function _mnist_focused_bytes(value)
    bytes = Int(value)
    bytes == 0 && return "0 B"
    amount, unit = bytes >= 1024^2 ? (bytes / 1024^2, "MiB") :
        bytes >= 1024 ? (bytes / 1024, "KiB") : (bytes, "B")
    string(_mnist_compact_number(amount), " ", unit)
end

_mnist_focused_runtime(result) = Float64(
    haskey(result, "min_ns") ? result["min_ns"] : result["median_ns"])

function _mnist_relative_interpretation(relative_time)
    isapprox(relative_time, 1.0; atol = 1e-12) && return "baseline"
    isapprox(relative_time, 1.0; atol = 0.005) && return "≈ parity"
    relative_time < 1 && return string(
        round(inv(relative_time); digits = 2), "× faster")
    string(round(relative_time; digits = 2), "× slower")
end

function _mnist_focused_plot(rows; id, title, baseline_note)
    series_order = getproperty.(rows, :series)
    spec = data(rows) *
        mapping(
            :series => sorter(series_order) => "Implementation",
            :relative_time => "Runtime ÷ baseline";
            color = :implementation => "Implementation",
        ) *
        visual(Scatter, markersize = 120) *
        config(height = 280, scales = scales(Y = (; scale = log10)))
    _plot_block(spec; id, title,
        description = "One outcome only. $baseline_note; 1.00× is parity and lower is faster. " *
                      "The ratio axis is logarithmic so large slowdowns do not hide near-parity results.")
end

function _mnist_focused_table(rows; id, title, baseline_note)
    columns = (
        _column(:series, "Implementation / configuration"),
        _column(:runtime_ns, "Runtime";
            format = (value, _) -> _mnist_focused_duration(value)),
        _column(:baseline_ns, "Baseline runtime";
            format = (value, _) -> _mnist_focused_duration(value)),
        _column(:relative_time, "Runtime ÷ baseline";
            format = (value, _) -> string(round(value; digits = 2), "×")),
        _column(:relative_time, "Interpretation";
            format = (value, _) -> _mnist_relative_interpretation(value)),
        _column(:median_bytes, "Allocated";
            format = (value, _) -> _mnist_focused_bytes(value)),
        _column(:median_allocs, "Allocations"),
    )
    _result_table(rows, columns; id, title,
        note = "$baseline_note. Display values use three significant digits; " *
               "the receipt retains exact values and all raw rounds.")
end

function _render_mnist_focused_rows(rows; id_prefix, profile_label,
                                    summary, provenance)
    blocks = Any[Markdown.Paragraph(Any[summary])]
    for outcome in _MNIST_FOCUSED_OUTCOMES
        selected = filter(row -> row.outcome == outcome, rows)
        isempty(selected) && continue
        label = _MNIST_FOCUSED_OUTCOME_LABELS[outcome]
        baseline_labels = unique(getproperty.(selected, :baseline_label))
        length(baseline_labels) == 1 || error(
            "MNIST focused comparison mixed baseline labels for $outcome")
        baseline_note = "Baseline: $(only(baseline_labels))"
        push!(blocks,
            _mnist_focused_plot(selected;
                id = "$id_prefix-$outcome-relative",
                title = "$profile_label — $label relative runtime",
                baseline_note),
            _mnist_focused_table(selected;
                id = "$id_prefix-$outcome-table",
                title = "$profile_label — $label",
                baseline_note),
        )
    end
    push!(blocks, Markdown.Paragraph(Any[Markdown.Italic(Any[provenance])]))
    Markdown.MD(blocks)
end

function _mnist_longform_primary_label(row, mode)
    provider = String(row["provider"])
    model = String(row["model"])
    configuration = String(row["configuration"])
    native = mode == :primal ? "primal_native" : "ad_native"
    manual = mode == :primal ? "manual_primal" : "manual_ad"
    turing_idiomatic = mode == :primal ?
        "turing_idiomatic_primal" : "turing_idiomatic_ad"
    turing_vcat_free = mode == :primal ?
        "turing_vcat_free_primal" : "turing_vcat_free_ad"
    provider == "rk" && configuration == native && model == "idiomatic" &&
        return ("RK idiomatic", "RK")
    provider == "rk" && configuration == native && model == "vcat_free" &&
        return ("RK vcat-free", "RK")
    provider == "manual_julia" && configuration == manual &&
        return ("Manual Julia", "Manual Julia")
    provider == "turing" && configuration == turing_idiomatic &&
        return ("Turing idiomatic", "Turing")
    provider == "turing" && configuration == turing_vcat_free &&
        return ("Turing vcat-free", "Turing")
    nothing
end

function _render_mnist_longform_focused(path, expected_schema;
                                        mode, id_prefix, profile_label)
    receipt = TOML.parsefile(path)
    get(receipt, "schema", "") == expected_schema ||
        error("unexpected MNIST focused receipt schema at $path")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("MNIST focused receipt was produced from a dirty checkout")
    primary = filter(receipt["measurements"]) do row
        String(row["boundary"]) == "packed_unconstrained" &&
            _mnist_longform_primary_label(row, mode) !== nothing
    end
    practicalbayes = _practicalbayes_receipt()
    practicalbayes_measurements = practicalbayes["model_measurements"]
    rows = NamedTuple[]
    for outcome in _MNIST_FOCUSED_OUTCOMES
        candidates = filter(row -> String(row["outcome"]) == outcome, primary)
        baseline_row = only(filter(candidates) do row
            label = _mnist_longform_primary_label(row, mode)
            label !== nothing && first(label) == "RK idiomatic"
        end)
        baseline_result = _longform_result(baseline_row)
        baseline_result === nothing && continue
        baseline_ns = _mnist_focused_runtime(baseline_result)
        for row in candidates
            result = _longform_result(row)
            result === nothing && continue
            series, implementation = _mnist_longform_primary_label(row, mode)
            runtime_ns = _mnist_focused_runtime(result)
            push!(rows, (;
                outcome,
                series,
                implementation,
                runtime_ns,
                baseline_ns,
                baseline_label = "RK idiomatic on the packed sampler boundary",
                relative_time = runtime_ns / baseline_ns,
                median_bytes = Int(result["median_bytes"]),
                median_allocs = Int(result["median_allocs"]),
            ))
        end
        differentiation = mode == :primal ? "primal" : "value_and_gradient"
        for model in ("idiomatic", "vcat_free")
            practicalbayes_rows = filter(row ->
                row["workload"] == "mnist_logistic" && row["model"] == model &&
                row["boundary"] == "packed_unconstrained" &&
                row["outcome"] == outcome &&
                row["differentiation"] == differentiation &&
                row["state"] == "supported", practicalbayes_measurements)
            isempty(practicalbayes_rows) && continue
            result = only(practicalbayes_rows)["result"]
            runtime_ns = _mnist_focused_runtime(result)
            model_label = replace(model, '_' => '-')
            push!(rows, (;
                outcome,
                series = "PracticalBayes $model_label",
                implementation = "PracticalBayes",
                runtime_ns,
                baseline_ns,
                baseline_label = "RK idiomatic on the packed sampler boundary",
                relative_time = runtime_ns / baseline_ns,
                median_bytes = Int(result["median_bytes"]),
                median_allocs = Int(result["median_allocs"]),
            ))
        end
    end
    measured_total = count(
        row -> get(row, "state", "") == "supported", receipt["measurements"])
    practicalbayes_measured = count(row ->
        get(row, "state", "") == "supported",
        practicalbayes_measurements)
    summary = "$profile_label. The plots and tables below isolate one outcome " *
        "at a time on the packed sampler boundary. They show the five directly " *
        "comparable RK/manual/Turing implementations plus every supported " *
        "PracticalBayes model adapter against RK idiomatic; " *
        "the receipt retains all $measured_total measured modifier/boundary cells " *
        "plus every explicit N/A or unsupported row. The separately resolved " *
        "PracticalBayes receipt retains $practicalbayes_measured supported cells " *
        "and its explicit public-API limitations."
    practicalbayes_pins = practicalbayes["pins"]
    provenance = "Receipt `$(basename(path))`; RK `$(pins["reactivekernels_sha"])`; " *
        "Julia $(pins["julia_version"]); $(receipt["environment"]["cpu"]). " *
        "PracticalBayes $(practicalbayes_pins["practicalbayes_version"]) rows " *
        "come from `$(basename(_PRACTICALBAYES_RECEIPT_PATH))`, resolved " *
        "separately because of its incompatible Bijectors pin."
    _render_mnist_focused_rows(rows; id_prefix, profile_label, summary, provenance)
end

function _render_mnist_v1_focused(path; mode, id_prefix, profile_label)
    receipt = TOML.parsefile(path)
    expected_schema = mode == :primal ? "mnist-logistic-v1" : "mnist-logistic-ad-v1"
    get(receipt, "schema", "") == expected_schema ||
        error("unexpected MNIST v1 focused receipt schema")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("MNIST v1 focused receipt was produced from a dirty checkout")
    packed = filter(row -> String(row["boundary"]) == "packed_unconstrained",
                    receipt["measurements"])
    implementations = mode == :primal ? (
        ("rk_native", "RK", "RK"),
        ("manual_julia", "Manual Julia", "Manual Julia"),
        ("turing_native", "Turing", "Turing"),
    ) : (
        ("rk_native", "RK + Enzyme", "RK + Enzyme"),
        ("manual_enzyme", "Manual + Enzyme", "Manual + Enzyme"),
        ("turing_enzyme", "Turing + Enzyme", "Turing + Enzyme"),
    )
    rows = NamedTuple[]
    for measurement in packed
        outcome = String(measurement["outcome"])
        haskey(measurement, "rk_native") || continue
        baseline_result = measurement["rk_native"]
        baseline_ns = _mnist_focused_runtime(baseline_result)
        for (key, series, implementation) in implementations
            haskey(measurement, key) || continue
            result = measurement[key]
            runtime_ns = _mnist_focused_runtime(result)
            push!(rows, (;
                outcome,
                series,
                implementation,
                runtime_ns,
                baseline_ns,
                baseline_label = mode == :primal ?
                    "RK on the packed sampler boundary" :
                    "RK + Enzyme on the packed sampler boundary",
                relative_time = runtime_ns / baseline_ns,
                median_bytes = Int(result["median_bytes"]),
                median_allocs = Int(result["median_allocs"]),
            ))
        end
    end
    summary = "$profile_label. Each outcome has its own relative-runtime plot " *
        "and compact table; setup, preparation, and first execution stay outside " *
        "steady-state timing and remain in the receipt."
    provenance = "Receipt `$(basename(path))`; RK `$(pins["reactivekernels_sha"])`; " *
        "Julia $(pins["julia_version"]); $(receipt["environment"]["cpu"])."
    _render_mnist_focused_rows(rows; id_prefix, profile_label, summary, provenance)
end

function _render_mnist_reactant_longform_focused(path, expected_schema;
                                                 mode, id_prefix, profile_label)
    receipt = TOML.parsefile(path)
    get(receipt, "schema", "") == expected_schema ||
        error("unexpected MNIST Reactant focused receipt schema")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("MNIST Reactant focused receipt was produced from a dirty checkout")
    rows = NamedTuple[]
    for measurement in receipt["measurements"]
        String(measurement["boundary"]) == "packed_unconstrained" || continue
        result = _longform_result(measurement)
        result === nothing && continue
        native = get(measurement, "native_control", nothing)
        native isa AbstractDict || error("MNIST Reactant row lacks native control")
        model = String(measurement["model"]) == "vcat_free" ?
            "vcat-free" : "idiomatic"
        configuration = String(measurement["configuration"])
        data_mode = occursin("_bound", configuration) ? "data bound" : "runtime data"
        runtime_ns = _mnist_focused_runtime(result)
        baseline_ns = _mnist_focused_runtime(native)
        push!(rows, (;
            outcome = String(measurement["outcome"]),
            series = "$(uppercasefirst(model)) · $data_mode",
            implementation = mode == :primal ? "Reactant" : "Reactant compiled AD",
            runtime_ns,
            baseline_ns,
            baseline_label = "the matched native RK twin for each configuration",
            relative_time = runtime_ns / baseline_ns,
            median_bytes = Int(result["median_bytes"]),
            median_allocs = Int(result["median_allocs"]),
        ))
    end
    summary = "$profile_label. Every plot isolates one outcome and compares " *
        "synchronous Reactant CPU time with that row's matched native RK twin. " *
        "Compilation, transfer, first call, and readback are excluded."
    provenance = "Receipt `$(basename(path))`; RK `$(pins["reactivekernels_sha"])`; " *
        "Reactant $(pins["reactant_version"]); Julia $(pins["julia_version"]); " *
        "$(receipt["environment"]["cpu"])."
    _render_mnist_focused_rows(rows; id_prefix, profile_label, summary, provenance)
end

function _render_mnist_reactant_v1_focused(path;
                                           mode, id_prefix, profile_label)
    receipt = TOML.parsefile(path)
    expected_schema = mode == :primal ? "mnist-reactant-v1" : "mnist-reactant-ad-v1"
    get(receipt, "schema", "") == expected_schema ||
        error("unexpected MNIST Reactant v1 focused receipt schema")
    pins = receipt["pins"]
    get(pins, "reactivekernels_dirty", true) == false ||
        error("MNIST Reactant v1 focused receipt was produced from a dirty checkout")
    native_key = mode == :primal ? "rk_native" : "rk_native_ad"
    reactant_key = mode == :primal ? "rk_reactant" : "rk_reactant_ad"
    support_key = mode == :primal ? "rk_reactant_supported" : "rk_reactant_ad_supported"
    rows = NamedTuple[]
    for measurement in receipt["measurements"]
        String(measurement["boundary"]) == "packed_unconstrained" || continue
        get(measurement, support_key, false) || continue
        native = measurement[native_key]
        result = measurement[reactant_key]
        runtime_ns = _mnist_focused_runtime(result)
        baseline_ns = _mnist_focused_runtime(native)
        push!(rows, (;
            outcome = String(measurement["outcome"]),
            series = mode == :primal ? "Reactant" : "Reactant compiled AD",
            implementation = mode == :primal ? "Reactant" : "Reactant compiled AD",
            runtime_ns,
            baseline_ns,
            baseline_label = mode == :primal ? "native RK" : "native RK + Enzyme",
            relative_time = runtime_ns / baseline_ns,
            median_bytes = Int(result["median_bytes"]),
            median_allocs = Int(result["median_allocs"]),
        ))
    end
    summary = "$profile_label. Each outcome is shown separately; Reactant timing " *
        "is synchronous and compared directly with the matched native receipt."
    provenance = "Receipt `$(basename(path))`; RK `$(pins["reactivekernels_sha"])`; " *
        "Reactant $(pins["reactant_version"]); Julia $(pins["julia_version"]); " *
        "$(receipt["environment"]["cpu"])."
    _render_mnist_focused_rows(rows; id_prefix, profile_label, summary, provenance)
end
