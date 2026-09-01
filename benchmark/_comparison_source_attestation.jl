module ComparisonSourceAttestation

export comparator_source_matches_current_delta, COMPARATOR_SOURCE_CURRENT_DELTA
export eight_schools_model_source_preserves_published_authority
export EIGHT_SCHOOLS_MODEL_SOURCE_CURRENT_DELTA
export mnist_model_source_preserves_published_authority
export MNIST_COMPARATOR_SOURCE_CURRENT_DELTA, MNIST_MODEL_SOURCE_CURRENT_DELTA

const COMPARATOR_SOURCE_CURRENT_DELTA =
    "long-form native/bound/nonallocating matrix around byte-preserved Turing/manual baselines plus terminal definition-only guard"
const EIGHT_SCHOOLS_MODEL_SOURCE_CURRENT_DELTA =
    "published centered hierarchy and every public boundary are preserved; additive scalar-index-free packed extraction and a single-output Jacobian recipe enable Reactant and nonallocating configurations"
const MNIST_COMPARATOR_SOURCE_CURRENT_DELTA =
    "additive two-model native/bound/nonallocating matrix; published Turing/manual AD baselines unchanged; documentation markers plus terminal definition-only include guard"
const MNIST_MODEL_SOURCE_CURRENT_DELTA =
    "published idiomatic and optimized models retain their math and boundaries; each categorical recipe keeps the exact authored scalar plate for native/nonallocating execution and adds an equivalent explicit tensor body for array compilers"
const _DOCS_BASELINE_MARKERS = (
    "# DOCS-BASELINE-BEGIN: turing",
    "# DOCS-BASELINE-END: turing",
    "# DOCS-BASELINE-BEGIN: manual",
    "# DOCS-BASELINE-END: manual",
)
const _MNIST_DOCS_BASELINE_MARKERS = (
    "# DOCS-BASELINE-BEGIN: turing",
    "# DOCS-BASELINE-END: turing",
    "# DOCS-BASELINE-BEGIN: turing-optimized",
    "# DOCS-BASELINE-END: turing-optimized",
    "# DOCS-BASELINE-BEGIN: manual",
    "# DOCS-BASELINE-END: manual",
)

_normalized_text(text) =
    replace(String(text), "\r\n" => "\n", "\r" => "\n")

function _single_marked_body(text, name)
    begin_marker = "# DOCS-BASELINE-BEGIN: $name\n"
    end_marker = "# DOCS-BASELINE-END: $name\n"
    length(findall(begin_marker, text)) == 1 || return nothing
    length(findall(end_marker, text)) == 1 || return nothing
    after_begin = split(text, begin_marker; limit = 2)[2]
    split(after_begin, end_marker; limit = 2)[1]
end

function _single_delimited_section(text, begin_marker, end_marker)
    length(findall(begin_marker, text)) == 1 || return nothing
    after_begin = split(text, begin_marker; limit = 2)[2]
    isempty(findall(end_marker, after_begin)) && return nothing
    begin_marker * split(after_begin, end_marker; limit = 2)[1] * end_marker
end

function _single_raw_body(text, begin_marker)
    length(findall(begin_marker, text)) == 1 || return nothing
    after_begin = split(text, begin_marker; limit = 2)[2]
    end_marker = "\n\"\"\""
    isempty(findall(end_marker, after_begin)) && return nothing
    split(after_begin, end_marker; limit = 2)[1]
end

function eight_schools_model_source_preserves_published_authority(
        current, published)
    current = _normalized_text(current)
    published = _normalized_text(published)
    replacements = (
        (
            "    # One-element reductions retain the ordinary packed-vector boundary while\n" *
            "    # avoiding scalar indexing when the same prepared kernel is traced as a\n" *
            "    # Reactant tensor program. Native Julia specializes these constant slices.\n" *
            "    μ::Float64 = sum(view(unconstrained, 1:1))\n" *
            "    log_τ::Float64 = sum(view(unconstrained, 2:2))\n",
            "    μ::Float64 = unconstrained[1]\n" *
            "    log_τ::Float64 = unconstrained[2]\n",
        ),
        (
            "    # The Jacobian is also available alone. That single-output path lets scalar\n" *
            "    # packed-density queries use the public nonallocating preparation pass,\n" *
            "    # whose recipes are deliberately single-output. Asking for BOTH parameters\n" *
            "    # and the Jacobian can still select the joint producer below.\n" *
            "    log_jacobian::Float64 = log_τ\n\n",
            "",
        ),
        (
            "    # first for a constrain-only query and the joint producer when both outputs\n" *
            "    # are requested together.\n",
            "    # first for a constrain-only query and the second whenever the Jacobian —\n" *
            "    # hence the unconstrained posterior — is wanted.\n",
        ),
    )
    transformed = current
    for (new, old) in replacements
        length(findall(new, transformed)) == 1 || return false
        transformed = replace(transformed, new => old; count = 1)
    end
    transformed == published
end

function mnist_model_source_preserves_published_authority(current, published)
    current = _normalized_text(current)
    published = _normalized_text(published)
    current_source = _single_raw_body(
        current,
        "const MNIST_LOGISTIC_SOURCE = " *
        "CATEGORICAL_LOGIT_COLUMNS_KERNEL_SOURCE * raw\"\"\"\n",
    )
    published_source = _single_raw_body(
        published, "const MNIST_LOGISTIC_SOURCE = raw\"\"\"\n")
    current_source !== nothing && published_source !== nothing || return false

    replacements = (
        ("    normal\n",
         "    normal, categorical_logit\n"),
        ("    # prepended as the first row. The batched `categorical_logit_columns`\n" *
         "    # distribution kernel returns the pointwise log densities in one\n" *
         "    # tensor-friendly gather.\n" *
         "    nonreference_logits::Matrix{Float64} = W * transpose(X) .+ b\n" *
         "    logits::Matrix{Float64} =\n" *
         "        vcat(zeros(1, size(nonreference_logits, 2)), nonreference_logits)\n" *
         "    pointwise::Vector{Float64} =\n" *
         "        _categorical_logit_columns_kernel(logits, y)\n",
         "    # prepended as the first row. Each observation's likelihood then reuses the\n" *
         "    # `categorical_logit` distribution object over that observation's logits\n" *
         "    # column — exactly as Eight Schools reuses `normal` per observation.\n" *
         "    nonreference_logits = W * transpose(X) .+ b\n" *
         "    logits = vcat(zeros(1, size(nonreference_logits, 2)), nonreference_logits)\n" *
         "    pointwise = plate(eachcol(logits), y) do observation_logits, observed_class\n" *
         "        categorical_logit(observation_logits).logpdf(observed_class)\n" *
         "    end\n"),
        ("# Pointwise likelihoods and their total are alternate cuts through the same\n" *
         "# batched distribution-object result.\n",
         "# Pointwise likelihoods and their total are alternate cuts through the same\n" *
         "# authored plate; a total-only query fuses the sum without materializing the\n" *
         "# per-observation vector.\n"),
        ("    normal_object = normal,\n)",
         "    normal_object = normal,\n" *
         "    categorical_logit_object = categorical_logit,\n)"),
    )
    transformed = current_source
    for (new, old) in replacements
        length(findall(new, transformed)) == 1 || return false
        transformed = replace(transformed, new => old; count = 1)
    end
    transformed == published_source || return false

    for optimized_anchor in (
            "const MNIST_LOGISTIC_OPTIMIZED_SOURCE =\n" *
            "    CATEGORICAL_LOGIT_REF_COLUMNS_KERNEL_SOURCE * raw\"\"\"",
            "_categorical_logit_ref_columns_kernel(nonreference_logits, y)",
            "build_mnist_logistic_optimized_graph() =",
        )
        length(findall(optimized_anchor, current)) == 1 || return false
    end

    for (kernel_source_anchor, count) in (
            "CATEGORICAL_LOGIT_COLUMNS_KERNEL_SOURCE," => 1,
            "const MNIST_LOGISTIC_SOURCE = " *
                "CATEGORICAL_LOGIT_COLUMNS_KERNEL_SOURCE * raw\"\"\"" => 1,
            "CATEGORICAL_LOGIT_REF_COLUMNS_KERNEL_SOURCE" => 2,
        )
        length(findall(kernel_source_anchor, current)) == count || return false
    end

    for authority in (
            "_evaluate_ppl_source(MNIST_LOGISTIC_SOURCE, @__MODULE__; bindings = (\n        :MNIST_LOGISTIC_X, :MNIST_LOGISTIC_Y, :NUM_CLASSES,\n    ))",
            "build_mnist_logistic_graph() = compose(_MNIST_LOGISTIC_GRAPH_TEMPLATE[])",
        )
        length(findall(authority, current)) == 1 || return false
        length(findall(authority, published)) == 1 || return false
    end
    true
end

function comparator_source_matches_current_delta(
        current, published, definition_only_guard)
    current = _normalized_text(current)
    marker_lines = filter(
        line -> startswith(line, "# DOCS-BASELINE-"), split(current, '\n'))
    marker_lines in (collect(_DOCS_BASELINE_MARKERS),
                     collect(_MNIST_DOCS_BASELINE_MARKERS)) || return false
    endswith(current, definition_only_guard) || return false
    length(findall(definition_only_guard, current)) == 1 || return false
    published = _normalized_text(published)
    for name in ("turing", "manual")
        body = _single_marked_body(current, name)
        body !== nothing && !isempty(body) || return false
        length(findall(body, published)) == 1 || return false
    end
    if marker_lines == collect(_MNIST_DOCS_BASELINE_MARKERS)
        optimized = _single_marked_body(current, "turing-optimized")
        optimized !== nothing && !isempty(optimized) || return false
    end
    true
end

end # module ComparisonSourceAttestation
