# Design-shape analysis: terms + raw columns → widths, labels, levels.
#
# Shared by layout assignment (offsets), preprocessing (matrices), and the
# generator (coefficient identity). Assumes `validate_plan` passed; residual
# checks here are loud defense in depth. Coefficient labels follow the pinned
# v2 scheme; BRM contrasts stay positional on the emitter side with the
# position↔label map in the ext/parity code.

"""
    DesignBlock(kind, column, addressee, width, labels, levels, ref)

One term's design contribution. `labels` are coefficient labels in column
order (`:Intercept` for intercepts, the column name for continuous terms,
`col_level` over non-ref sort-ordered levels for factors, empty for
offsets). `levels`/`ref` are meaningful for factors only.
"""
struct DesignBlock
    kind::TermKind
    column::Union{Nothing,Symbol}
    addressee::Symbol
    width::Int
    labels::Vector{Symbol}
    levels::Vector
    ref::Int
end

"""Full design shape of one predictor: ordered blocks + total width."""
struct DesignShape
    predictor::Symbol
    blocks::Vector{DesignBlock}
    width::Int
end

"""
    design_shape(pred, columns) -> DesignShape

Analyze one predictor's terms against raw columns. Factor levels are
`sort(unique(col))` (pinned ordering); contrasts run over non-ref levels in
sort order with labels `Symbol(col_level)`.
"""
function design_shape(pred::PredictorSpec, columns::Dict{Symbol,AbstractVector})
    blocks = DesignBlock[]
    for t in pred.terms
        push!(blocks, _term_block(t, columns, pred.label))
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

function _term_block(t::TermSpec, columns, label)
    if t.kind === InterceptTerm
        return DesignBlock(InterceptTerm, nothing, t.addressee, 1, [:Intercept], [], 0)
    elseif t.kind === ContinuousTerm
        col = only(t.columns)
        return DesignBlock(ContinuousTerm, col, t.addressee, 1, [col], [], 0)
    elseif t.kind === OffsetTerm
        return DesignBlock(OffsetTerm, only(t.columns), t.addressee, 0, Symbol[], [], 0)
    elseif t.kind === FactorTerm
        col = only(t.columns)
        levels = _grouping_levels(columns[col])
        ref = t.options.ref
        labels = Symbol[]
        for (i, level) in enumerate(levels)
            i == ref && continue
            push!(labels, Symbol(string(col) * "_" * string(level)))
        end
        return DesignBlock(FactorTerm, col, t.addressee, length(labels), labels,
            levels, ref)
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
