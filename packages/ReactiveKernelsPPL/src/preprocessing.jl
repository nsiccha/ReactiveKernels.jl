# In-graph preprocessing recipes (D5a): named data-only nodes.
#
# Slice 1 builds one design matrix per predictor plus one offset vector
# (when offset terms exist) from bound raw columns. All recipe names carry
# the reserved `_ppl_` prefix, so the contract's reserved-prefix validation
# guarantees they collide with nothing. Inputs are all bound, so `bound=`
# partial evaluation folds every recipe to a compile-time constant —
# exactly the brm_hsgp.jl basis pattern. Assumes `validate_plan` passed.

"""Design-matrix recipe name for a predictor (`mu` → `_ppl_design_mu`)."""
design_name(predictor::Symbol) = Symbol(:_ppl_design_, predictor)

"""Offset-vector recipe name for a predictor (`mu` → `_ppl_offset_mu`)."""
offset_name(predictor::Symbol) = Symbol(:_ppl_offset_, predictor)

"""
    design_recipe(shape, n_obs) -> Union{Nothing,Expr}

`_ppl_design_<pred> = Float64.(hcat(<blocks…>))`, or `nothing` for a
width-0 (offset-only) predictor. Blocks: intercept → `ones(n)`, continuous
→ the bare column, factor → treatment contrasts over sort-ordered levels.
"""
function design_recipe(shape::DesignShape, n_obs::Int)
    shape.width == 0 && return nothing
    # Continuous blocks contribute their bare column (a bound port), not a
    # wrapped Expr.
    parts = Any[]
    for b in shape.blocks
        if b.kind === InterceptTerm
            push!(parts, :(ones($n_obs)))
        elseif b.kind === ContinuousTerm
            push!(parts, b.column)
        elseif b.kind === FactorTerm
            push!(parts, _contrast_expr(b))
        end
    end
    matrix = Expr(:call, :hcat, parts...)
    name = design_name(shape.predictor)
    return :($name = Float64.($matrix))
end

"""
    offset_recipe(shape) -> Union{Nothing,Expr}

`_ppl_offset_<pred> = col1 + col2 + …` over offset-term columns, or
`nothing` when the predictor has no offset terms.
"""
function offset_recipe(shape::DesignShape)
    cols = Symbol[]
    for b in shape.blocks
        b.kind === OffsetTerm && push!(cols, b.column)
    end
    isempty(cols) && return nothing
    total = foldl((a, c) -> :($a + $c), cols)
    name = offset_name(shape.predictor)
    return :($name = $total)
end

"""
    preprocessing_recipes(plan) -> Vector{Expr}

All data-only recipes for a plan, in `plan.predictors` order: each
predictor's design matrix (unless width-0) then its offset vector (unless
absent).
"""
function preprocessing_recipes(plan::StructuralPlan)
    stmts = Expr[]
    for pred in plan.predictors
        shape = design_shape(pred, plan.columns)
        recipe = design_recipe(shape, plan.n_obs)
        recipe !== nothing && push!(stmts, recipe)
        off = offset_recipe(shape)
        off !== nothing && push!(stmts, off)
    end
    return stmts
end

# Treatment contrasts over sort-ordered non-ref levels:
# `Float64.(g .== permutedims([l1, l2, …]))`.
function _contrast_expr(b::DesignBlock)
    @assert b.kind === FactorTerm
    nonref = [lvl for (i, lvl) in enumerate(b.levels) if i != b.ref]
    lvlvec = Expr(:vect, (_level_literal(lvl) for lvl in nonref)...)
    # Build `g .== permutedims(lvlvec)` via quasiquote for stable lowering.
    g = b.column
    perms = :(permutedims($lvlvec))
    return :(Float64.($g .== $perms))
end

# Grouping levels embed as literals; Symbols need QuoteNode (a bare Symbol
# in an Expr would resolve as a variable). Anything else is loud.
_level_literal(lvl::Union{Number,String,Bool,Char}) = lvl
_level_literal(lvl::Symbol) = QuoteNode(lvl)
_level_literal(lvl) = throw(
    ContractValidationError("[preprocessing] grouping level $(repr(lvl)) " *
                            "is not literal-embeddable (numeric/string/symbol only)"),
)
