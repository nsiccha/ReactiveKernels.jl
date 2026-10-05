# Design-shape analysis: terms + bound plan data → widths, labels, levels.
#
# Shared by layout assignment (offsets), preprocessing (matrices), and the
# generator (coefficient identity). Assumes `validate_plan` passed; residual
# checks here are loud defense in depth. Coefficient labels follow the pinned
# v2 scheme; factor levels come from the plan's LevelMap (full-rank over
# exactly the mapped values).

"""
    DesignBlock(kind, column, addressee, width, labels, levels, elements)

One term's design contribution. `labels` are coefficient labels in column
order (`:Intercept` for intercepts, the column name for continuous and
monotonic terms, `col_level` over mapped levels for factors, empty for
offsets and beta-free summands). `levels` is meaningful for factors only
(the map's evaluated values). `column` is the data column except for
spline/hsgp summands (the basis id), monotonic blocks (the increments
key naming the contrast recipe), and matrix blocks (the matrix name).
`elements` is meaningful for matrix blocks only (the matrix columns in
order, `nothing` at intercept positions — per-element prior addresses
and intercept flags for consumers that fan out).
"""
struct DesignBlock
    kind::TermKind
    column::Union{Nothing,Symbol}
    addressee::Symbol
    width::Int
    labels::Vector{Symbol}
    levels::Vector
    elements::Vector{Union{Nothing,Symbol}}
end

# Non-matrix blocks carry no elements.
DesignBlock(kind::TermKind, column::Union{Nothing,Symbol}, addressee::Symbol,
    width::Int, labels::Vector{Symbol}, levels::Vector) =
    DesignBlock(kind, column, addressee, width, labels, levels,
        Union{Nothing,Symbol}[])

"""Full design shape of one predictor: ordered blocks + total width."""
struct DesignShape
    predictor::Symbol
    blocks::Vector{DesignBlock}
    width::Int
end

"""
    design_shape(pred, columns; levelmaps, matrices) -> DesignShape

Analyze one predictor's terms against raw columns. Factor blocks read
their levels from `levelmaps` (keyed by predictor name + column);
columns stay the source for continuous/offset terms. Matrix blocks read
their columns from `matrices` (keyed by the term's matrix name).
"""
function design_shape(pred::PredictorSpec, columns::AbstractDict{Symbol};
        levelmaps::Vector{LevelMap} = LevelMap[],
        matrices::Vector{DesignMatrix} = DesignMatrix[])
    blocks = DesignBlock[]
    for t in pred.terms
        push!(blocks, _term_block(t, columns, pred.label, pred.name, levelmaps,
            matrices))
    end
    labels = Symbol[]
    for b in blocks
        append!(labels, b.labels)
    end
    (_parameter_terms(pred) || length(unique(labels)) == length(labels)) ||
        throw(ContractValidationError("[$(pred.label)] duplicate coefficient labels"))
    width = sum(b.width for b in blocks; init = 0)
    return DesignShape(pred.name, blocks, width)
end

function _term_block(t::TermSpec, columns, label, pname, levelmaps, matrices)
    if t.kind === InterceptTerm
        return DesignBlock(InterceptTerm, nothing, t.addressee, 1, [:Intercept], [])
    elseif t.kind === ContinuousTerm
        col = only(t.columns)
        return DesignBlock(ContinuousTerm, col, t.addressee, 1, [col], [])
    elseif t.kind === OffsetTerm
        return DesignBlock(OffsetTerm, only(t.columns), t.addressee, 0, Symbol[], [])
    elseif t.kind === LatentTerm
        # The latent VECTOR is the whole linear predictor (identity design);
        # its coefficients live in the PlateParameter layout block, so this
        # term contributes no design width.
        return DesignBlock(LatentTerm, only(t.columns), t.addressee, 0, Symbol[], [])
    elseif t.kind === ScanSummandTerm
        # A scan summand is a direct `state .* coef` expression over the
        # in-graph recurrence state and a scalar sampled coefficient (or
        # the bare state when `coef === nothing`) — no design-matrix width
        # (the state is sampled, not data). The state rides in `column`;
        # the generator reads the TERMS for the full (scan_id, coef) key,
        # which does not fit one Symbol.
        return DesignBlock(ScanSummandTerm, t.options.scan_id,
            t.addressee, 0, Symbol[], [])
    elseif t.kind === FactorTerm
        col = only(t.columns)
        m = _find_levelmap(levelmaps, pname, col)
        m === nothing && throw(ContractValidationError(
            "[$label] factor term over $col has no LevelMap " *
            "(validate_levelmaps should have caught this)"))
        isempty(m.values) &&
            !isempty(only(_eval_levelmaps(LevelMap[m], columns)).values) &&
            throw(ContractValidationError(
            "[$label] LevelMap for $col has no evaluated values " *
            "(bind_data fills these)"))
        labels = Symbol[Symbol(string(col) * "_" * string(level)) for level in m.values]
        return DesignBlock(FactorTerm, col, t.addressee, length(labels), labels,
            collect(m.values))
    elseif t.kind === MatrixTerm
        # A matrix term splices `X * view(coef, ...)` over its design
        # matrix: width K with per-element labels matching the affine
        # spelling (`:Intercept` at intercept positions, the column
        # otherwise — twins report identically). `column` carries the
        # matrix name (the recipe part); `elements` the matrix columns
        # for per-element prior consumers.
        i = findfirst(m -> m.name === t.options.matrix, matrices)
        i === nothing && throw(ContractValidationError(
            "[$label] matrix term addresses :$(t.options.matrix), which " *
            "has no DesignMatrix (validate_predictors should have caught " *
            "this)"))
        m = matrices[i]
        labels = Symbol[e === nothing ? :Intercept : e for e in m.columns]
        return DesignBlock(MatrixTerm, m.name, t.addressee, length(m.columns),
            labels, [], copy(m.columns))
    elseif t.kind === ComposedTerm
        # A composed term evaluates an elementwise combination tree over
        # sub-predictor LP nodes and scalar nodes in-graph — no
        # design-matrix width (coefficients live in the sub-predictors
        # and scalars). The generator reads the TERMS for the tree.
        return DesignBlock(ComposedTerm, nothing, t.addressee, 0, Symbol[],
            [])
    else
        throw(ContractValidationError("[$label] term kind $(t.kind) has no design rule"))
    end
end

"""
    coefficient_prior_specs(shape, priors) -> Vector{PopulationPrior}

Per-coefficient [`PopulationPrior`](@ref) rows in design-column order (the
shared expansion core: a factor addressee fans out to its whole block, a
matrix block looks up each element addressee in turn).
[`coefficient_priors`](@ref) projects locations/scales; the generator
reads families from the same walk.
"""
function coefficient_prior_specs(shape::DesignShape,
        priors::Vector{PopulationPrior})
    by_addressee = Dict{Symbol,PopulationPrior}()
    for pr in priors
        pr.predictor === shape.predictor || continue
        by_addressee[pr.addressee] = pr
    end
    out = PopulationPrior[]
    for b in shape.blocks
        b.width == 0 && continue
        if b.kind === MatrixTerm
            for (e, lab) in zip(b.elements, b.labels)
                addr = e === nothing ? :Intercept : e
                haskey(by_addressee, addr) || throw(
                    ContractValidationError(
                        "[$(shape.predictor)] no prior for addressee " *
                        "$addr (matrix $(b.addressee) element $lab)"),
                )
                push!(out, by_addressee[addr])
            end
            continue
        end
        haskey(by_addressee, b.addressee) || throw(
            ContractValidationError("[$(shape.predictor)] no prior for addressee " *
                                    "$(b.addressee)"),
        )
        append!(out, fill(by_addressee[b.addressee], b.width))
    end
    return out
end

"""
    coefficient_priors(shape, priors) -> (locations, scales)

Expand per-addressee [`PopulationPrior`](@ref)s to per-coefficient location
and scale vectors in design-column order (via
[`coefficient_prior_specs`](@ref)). A factor addressee fans out to its
whole contrast block (one shared prior); a matrix block looks up each
element addressee in turn (per-column priors).
"""
function coefficient_priors(shape::DesignShape, priors::Vector{PopulationPrior})
    specs = coefficient_prior_specs(shape, priors)
    return ([Float64(s.location) for s in specs],
        [Float64(s.scale) for s in specs])
end
