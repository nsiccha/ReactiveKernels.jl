# Design-shape analysis: terms + bound plan data → widths, labels, levels.
#
# Shared by layout assignment (offsets), preprocessing (matrices), and the
# generator (coefficient identity). Assumes `validate_plan` passed; residual
# checks here are loud defense in depth. Coefficient labels follow the pinned
# v2 scheme; factor levels come from the plan's LevelMap (full-rank over
# exactly the mapped values).

"""
    DesignBlock(kind, column, addressee, width, labels, levels)

One term's design contribution. `labels` are coefficient labels in column
order (`:Intercept` for intercepts, the column name for continuous terms,
`col_level` over mapped levels for factors, empty for offsets). `levels`
is meaningful for factors only (the map's evaluated values).
"""
struct DesignBlock
    kind::TermKind
    column::Union{Nothing,Symbol}
    addressee::Symbol
    width::Int
    labels::Vector{Symbol}
    levels::Vector
end

"""Full design shape of one predictor: ordered blocks + total width."""
struct DesignShape
    predictor::Symbol
    blocks::Vector{DesignBlock}
    width::Int
end

"""
    design_shape(pred, columns; levelmaps) -> DesignShape

Analyze one predictor's terms against raw columns. Factor blocks read
their levels from `levelmaps` (keyed by predictor name + column);
columns stay the source for continuous/offset terms.
"""
function design_shape(pred::PredictorSpec, columns::Dict{Symbol,AbstractVector};
        levelmaps::Vector{LevelMap} = LevelMap[])
    blocks = DesignBlock[]
    for t in pred.terms
        push!(blocks, _term_block(t, columns, pred.label, pred.name, levelmaps))
    end
    labels = Symbol[]
    for b in blocks
        append!(labels, b.labels)
    end
    length(unique(labels)) == length(labels) ||
        throw(ContractValidationError("[$(pred.label)] duplicate coefficient labels"))
    width = sum(b.width for b in blocks; init = 0)
    return DesignShape(pred.name, blocks, width)
end

function _term_block(t::TermSpec, columns, label, pname, levelmaps)
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
    elseif t.kind === SplineSummandTerm
        # A spline summand is a direct `X*b + Z*(sd*z)` expression over
        # materialized basis columns and SplineVector layout blocks — no
        # design-matrix width. The basis id rides in `column` so the
        # generator can resolve the blocks without re-reading terms.
        return DesignBlock(SplineSummandTerm, t.options.spline_id,
            t.addressee, 0, Symbol[], [])
    elseif t.kind === FactorTerm
        col = only(t.columns)
        m = _find_levelmap(levelmaps, pname, col)
        m === nothing && throw(ContractValidationError(
            "[$label] factor term over $col has no LevelMap " *
            "(validate_levelmaps should have caught this)"))
        isempty(m.values) && throw(ContractValidationError(
            "[$label] LevelMap for $col has no evaluated values " *
            "(bind_data fills these)"))
        labels = [Symbol(string(col) * "_" * string(level)) for level in m.values]
        return DesignBlock(FactorTerm, col, t.addressee, length(labels), labels,
            collect(m.values))
    else
        throw(ContractValidationError("[$label] term kind $(t.kind) has no design rule"))
    end
end

"""
    coefficient_priors(shape, priors) -> (locations, scales)

Expand per-addressee [`PopulationPrior`](@ref)s to per-coefficient location
and scale vectors in design-column order. A factor addressee fans out to its
whole contrast block (one shared Normal).
"""
function coefficient_priors(shape::DesignShape, priors::Vector{PopulationPrior})
    by_addressee = Dict{Symbol,PopulationPrior}()
    for pr in priors
        pr.predictor === shape.predictor || continue
        by_addressee[pr.addressee] = pr
    end
    locations = Float64[]
    scales = Float64[]
    for b in shape.blocks
        b.width == 0 && continue
        haskey(by_addressee, b.addressee) || throw(
            ContractValidationError("[$(shape.predictor)] no prior for addressee " *
                                    "$(b.addressee)"),
        )
        pr = by_addressee[b.addressee]
        append!(locations, fill(Float64(pr.location), b.width))
        append!(scales, fill(Float64(pr.scale), b.width))
    end
    return (locations, scales)
end
