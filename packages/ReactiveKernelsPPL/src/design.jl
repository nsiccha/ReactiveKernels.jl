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
    length(unique(labels)) == length(labels) ||
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
    elseif t.kind === MonotonicTerm
        # A monotonic (mo) column: width 1 with a free coefficient, labeled
        # by its index column (the continuous precedent). The contrast is
        # parameter-derived, so it never enters the data-only design
        # matrix — the generator splices it per-block against its
        # coefficient coordinate. `column` carries the increments key (the
        # spline-basis-id precedent), which names the contrast recipe.
        col = only(t.columns)
        return DesignBlock(MonotonicTerm, t.options.increments, t.addressee,
            1, [col], [])
    elseif t.kind === MonotonicSummandTerm
        # A monotonic summand (mo1): a direct beta-free contrast splice —
        # no design-matrix width. `column` carries the increments key, as
        # for the column shape.
        return DesignBlock(MonotonicSummandTerm, t.options.increments,
            t.addressee, 0, Symbol[], [])
    elseif t.kind === SplineSummandTerm
        # A spline summand is a direct `X*b + Z*(sd*z)` expression over
        # materialized basis columns and SplineVector layout blocks — no
        # design-matrix width. The basis id rides in `column` so the
        # generator can resolve the blocks without re-reading terms.
        return DesignBlock(SplineSummandTerm, t.options.spline_id,
            t.addressee, 0, Symbol[], [])
    elseif t.kind === HSGPSummandTerm
        # An HSGP summand is a direct `PHI * (sqrt_spd .* beta)` expression
        # over in-graph basis columns and the term's layout blocks (Stage
        # B) — no design-matrix width. The basis id rides in `column` so
        # the generator can resolve the basis without re-reading terms.
        return DesignBlock(HSGPSummandTerm, t.options.hsgp_id,
            t.addressee, 0, Symbol[], [])
    elseif t.kind === ScanSummandTerm
        # A scan summand is a direct `state .* coef` expression over the
        # in-graph recurrence state and a scalar sampled coefficient — no
        # design-matrix width (the state is sampled, not data). The state
        # rides in `column`; the generator reads the TERMS for the full
        # (scan_id, coef) key, which does not fit one Symbol.
        return DesignBlock(ScanSummandTerm, t.options.scan_id,
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
    elseif t.kind === VaryingEffectTerm
        # A varying effect is a direct `r` expression over the group
        # index and the draws block's draws (SB's
        # `r_<target>_<suffix>` summand) — no design-matrix width.
        # `column` carries the grouping column (the encoder input); the
        # generator reads the TERMS for the draws label.
        return DesignBlock(VaryingEffectTerm, only(t.columns), t.addressee,
            0, Symbol[], [])
    elseif t.kind === MatrixTerm
        # A matrix term splices `X * view(coef, ...)` over its design
        # matrix: width K with per-element labels matching the affine
        # spelling (`:Intercept` at intercept positions, the column
        # otherwise — twins report identically). `column` carries the
        # matrix name (the recipe part); `elements` the matrix columns
        # for per-element consumers (priors, R2D2).
        i = findfirst(m -> m.name === t.options.matrix, matrices)
        i === nothing && throw(ContractValidationError(
            "[$label] matrix term addresses :$(t.options.matrix), which " *
            "has no DesignMatrix (validate_predictors should have caught " *
            "this)"))
        m = matrices[i]
        labels = Symbol[e === nothing ? :Intercept : e for e in m.columns]
        return DesignBlock(MatrixTerm, m.name, t.addressee, length(m.columns),
            labels, [], copy(m.columns))
    else
        throw(ContractValidationError("[$label] term kind $(t.kind) has no design rule"))
    end
end

"""
    coefficient_priors(shape, priors) -> (locations, scales)

Expand per-addressee [`PopulationPrior`](@ref)s to per-coefficient location
and scale vectors in design-column order. A factor addressee fans out to its
whole contrast block (one shared Normal); a matrix block looks up each
element addressee in turn (per-column Normals).
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
        if b.kind === MatrixTerm
            for (e, lab) in zip(b.elements, b.labels)
                addr = e === nothing ? :Intercept : e
                haskey(by_addressee, addr) || throw(
                    ContractValidationError(
                        "[$(shape.predictor)] no prior for addressee " *
                        "$addr (matrix $(b.addressee) element $lab)"),
                )
                pr = by_addressee[addr]
                push!(locations, Float64(pr.location))
                push!(scales, Float64(pr.scale))
            end
            continue
        end
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

"""
    r2d2_column_scales(shape, columns, overrides) -> (share, fallback, loc, varx)

Bound data-only R2D2 share composition (SB `_sb_emit_r2d2_popefs!`
mirror): per-design-column `share_idx` (0 = fallback), fallback
scales, locations, and column variances, all in design-column order.
The intercept is always share 0 (SB default `(loc, fallback) =
(0.0, 1.0)` unless overridden); every other data column takes the
next share unless its addressee carries an explicit-Normal override.
Factor blocks fan out per dummy (variances via the
`brm_cat_variances` `m*(n-m)/(n*(n-1))` formula in level order);
continuous columns take the sample variance (N−1, the Stan
`variance()` normalization). Width-0 blocks (offsets, effects,
summands) contribute nothing. Monotonic columns fail closed: their
contrast is parameter-derived, so no data variance exists (no SB
precedent in the flat mirror).
"""
function r2d2_column_scales(shape::DesignShape,
        columns::AbstractDict{Symbol},
        overrides::Dict{Symbol,Tuple{Float64,Float64}})
    share = Int[]
    fallback = Float64[]
    loc = Float64[]
    varx = Float64[]
    next_share = 1
    for b in shape.blocks
        b.width == 0 && continue
        if b.kind === MonotonicTerm
            throw(ContractValidationError(
                "[$(shape.predictor)] R2D2 over a monotonic column is " *
                "not in the flat slice (the mo contrast is " *
                "parameter-derived, so no data variance exists)"))
        end
        if b.kind === InterceptTerm
            # Always share 0 (SB); the override, if stated, supplies
            # loc/scale. No data variance exists (column is nothing) —
            # unused at share 0 either way.
            oloc, ofb = get(overrides, b.addressee, (0.0, 1.0))
            push!(share, 0)
            push!(fallback, ofb)
            push!(loc, oloc)
            push!(varx, 0.0)
            continue
        end
        if b.kind === MatrixTerm
            # Per-element composition: intercept positions take share 0
            # (the intercept arm), other positions an explicit-Normal
            # override by element addressee or the next share. Variances
            # mirror the continuous rule per data column (0.0 at
            # intercepts, unused at share 0 either way).
            for e in b.elements
                if e === nothing
                    oloc, ofb = get(overrides, :Intercept, (0.0, 1.0))
                    push!(share, 0)
                    push!(fallback, ofb)
                    push!(loc, oloc)
                    push!(varx, 0.0)
                elseif haskey(overrides, e)
                    oloc, ofb = overrides[e]
                    push!(share, 0)
                    push!(fallback, ofb)
                    push!(loc, oloc)
                    push!(varx, _r2d2_sample_variance(columns[e]))
                else
                    push!(share, next_share)
                    next_share += 1
                    push!(fallback, 1.0)
                    push!(loc, 0.0)
                    push!(varx, _r2d2_sample_variance(columns[e]))
                end
            end
            continue
        end
        if haskey(overrides, b.addressee)
            oloc, ofb = overrides[b.addressee]
            append!(share, fill(0, b.width))
            append!(fallback, fill(ofb, b.width))
            append!(loc, fill(oloc, b.width))
            append!(varx, _r2d2_block_variances(b, columns))
            continue
        end
        for _ in 1:b.width
            push!(share, next_share)
            next_share += 1
        end
        append!(fallback, fill(1.0, b.width))
        append!(loc, fill(0.0, b.width))
        append!(varx, _r2d2_block_variances(b, columns))
    end
    return (share, fallback, loc, varx)
end

function _r2d2_block_variances(b::DesignBlock, columns::AbstractDict{Symbol})
    if b.kind === FactorTerm
        col = columns[b.column]
        n = length(col)
        return Float64[
            _r2d2_dummy_variance(col, lvl, n) for lvl in b.levels]
    end
    col = columns[b.column]
    return Float64[_r2d2_sample_variance(col)]
end

# Sample variance, N−1 normalization (Stan `variance()`).
function _r2d2_sample_variance(col::AbstractVector)
    n = length(col)
    n < 2 && return NaN
    m = sum(col) / n
    return sum((x - m)^2 for x in col) / (n - 1)
end

# Dummy variance without materializing the dummy (SB
# `brm_cat_variances`: m*(n-m)/(n*(n-1)) over the level codes).
function _r2d2_dummy_variance(col::AbstractVector, lvl, n::Int)
    n < 2 && return NaN
    m = count(==(lvl), col)
    return Float64(m * (n - m) / (n * (n - 1)))
end
