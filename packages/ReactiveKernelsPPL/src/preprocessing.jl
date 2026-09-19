# In-graph preprocessing recipes (D5a): named recipe nodes (design/offset
# recipes are data-only; monotonic contrasts read layout elements).
#
# Slice 1 builds one design matrix per predictor plus one offset vector
# (when offset terms exist) from bound raw columns, plus one monotonic
# contrast per `mo`/`mo1` term (the only parameter-dependent recipes —
# everything else folds under `bound=`). All recipe names carry the
# reserved `_ppl_` prefix, so the contract's reserved-prefix validation
# guarantees they collide with nothing. Design/offset inputs are all
# bound, so `bound=` partial evaluation folds those recipes to
# compile-time constants — exactly the brm_hsgp.jl basis pattern.
# Assumes `validate_plan` passed.

using LinearAlgebra: Diagonal, Symmetric, eigen, norm, nullspace

# Spline fits (verbatim port of BRM `src/preparation_basis.jl`, `_brm_` →
# `_rk_` + `ContractValidationError`): same op order, so bit-identical
# bases on identical input (verified by a /tmp differential, not committed
# — BRM is not a test dep). Eigen/nullspace/quantile are inexpressible in
# the kernel graph, so the fit runs at BIND (host, full LAPACK) — the exact
# Stan transformed-data mirror — and materializes basis COLUMNS as bound
# vectors; the graph sees only vector ops. Fit/apply stay split (slice-2
# replay reuses the fit object); density+gradient-only binds call both.
_rk_spline_fail(msg) = throw(ContractValidationError("[spline] " * msg))

function _rk_tps_kernel(x::AbstractVector{<:Real}, centers::AbstractVector{<:Real})
    E = Matrix{Float64}(undef, length(x), length(centers))
    for j in eachindex(centers), i in eachindex(x)
        E[i, j] = abs(Float64(x[i]) - Float64(centers[j]))^3 / 12
    end
    E
end

function _rk_fit_spline(x::AbstractVector{<:Real}; k::Int=10)
    k > 2 || _rk_spline_fail("`s(x)` needs basis dimension k > 2 (got $k)")
    xs = collect(Float64, x)
    all(isfinite, xs) || _rk_spline_fail("`s(x)` requires finite numeric data")
    length(unique(xs)) >= k || _rk_spline_fail(
        "`s(x)` needs at least $k unique x values for the default " *
        "thin-plate basis (got $(length(unique(xs))))")

    shift = sum(xs) / length(xs)
    centers = xs .- shift
    E = _rk_tps_kernel(centers, centers)
    eig_E = eigen(Symmetric(E))
    keep = sortperm(abs.(eig_E.values); rev=true)[1:k]
    U = eig_E.vectors[:, keep]
    D = eig_E.values[keep]

    T = hcat(ones(Float64, length(xs)), centers)
    Z = nullspace(transpose(T) * U)
    size(Z, 2) == k - 2 || _rk_spline_fail(
        "`s(x)` could not isolate the two-dimensional TPS null space")

    S = Symmetric(transpose(Z) * Diagonal(D) * Z)
    eig_S = eigen(S)
    penalty_scale = maximum(abs, eig_S.values)
    penalty_scale > 0 || _rk_spline_fail("`s(x)` produced a zero range-space penalty")
    tol = penalty_scale * eps(Float64) * 100
    minimum(eig_S.values) >= -tol || _rk_spline_fail(
        "`s(x)` produced a non-positive range-space penalty")
    penalty_values = max.(eig_S.values, tol)
    penalty_whitener = eig_S.vectors * Diagonal(inv.(sqrt.(penalty_values)))
    range_projection = U * Z * penalty_whitener

    (; shift, centers, range_projection, k)
end

function _rk_apply_spline(fit, x::AbstractVector{<:Real})
    xs = collect(Float64, x)
    all(isfinite, xs) || _rk_spline_fail("`s(x)` requires finite numeric data")
    centered = xs .- fit.shift
    Xnull = hcat(ones(Float64, length(xs)), centered)
    Zpen = _rk_tps_kernel(centered, fit.centers) * fit.range_projection
    Xnull, Zpen
end

_rk_spline_basis_tps(x::AbstractVector{<:Real}; k::Int=10) =
    _rk_apply_spline(_rk_fit_spline(x; k), x)

function _rk_type7_knots(x::AbstractVector{<:Real}, k::Int)
    values = sort!(unique(collect(Float64, x)))
    length(values) >= k || _rk_spline_fail(
        "`t2` margin needs at least $k unique values (got $(length(values)))")
    n = length(values)
    knots = Vector{Float64}(undef, k)
    for i in 1:k
        pos = 1 + (n - 1) * (i - 1) / (k - 1)
        lo = clamp(floor(Int, pos), 1, n)
        hi = clamp(ceil(Int, pos), 1, n)
        weight = pos - lo
        knots[i] = (1 - weight) * values[lo] + weight * values[hi]
    end
    all(diff(knots) .> 0) || _rk_spline_fail(
        "`t2` margin produced non-distinct cubic-regression-spline knots")
    knots
end

function _rk_cr_second_derivative_map(knots::AbstractVector{<:Real})
    k = length(knots)
    h = diff(knots)
    all(h .> 0) || _rk_spline_fail("`t2` cubic-regression-spline knots must increase")
    D = zeros(Float64, k - 2, k)
    B = zeros(Float64, k - 2, k - 2)
    for i in 1:(k - 2)
        D[i, i] = inv(h[i])
        D[i, i + 1] = -inv(h[i]) - inv(h[i + 1])
        D[i, i + 2] = inv(h[i + 1])
        B[i, i] = (h[i] + h[i + 1]) / 3
        if i < k - 2
            B[i, i + 1] = h[i + 1] / 6
            B[i + 1, i] = B[i, i + 1]
        end
    end
    interior = B \ D
    F = zeros(Float64, k, k)
    F[2:(k - 1), :] .= interior
    F, transpose(D) * interior
end

function _rk_cr_basis(knots, F, x::AbstractVector{<:Real})
    k = length(knots)
    X = zeros(Float64, length(x), k)
    for (i, raw_x) in enumerate(x)
        xi = Float64(raw_x)
        if xi < knots[1]
            h = knots[2] - knots[1]
            xik = xi - knots[1]
            cjm = -xik * h / 3
            cjp = -xik * h / 6
            for q in 1:k
                X[i, q] = cjm * F[1, q] + cjp * F[2, q]
            end
            X[i, 1] += 1 - xik / h
            X[i, 2] += xik / h
        elseif xi > knots[k]
            h = knots[k] - knots[k - 1]
            xik = xi - knots[k]
            cjm = xik * h / 6
            cjp = xik * h / 3
            for q in 1:k
                X[i, q] = cjm * F[k - 1, q] + cjp * F[k, q]
            end
            X[i, k - 1] -= xik / h
            X[i, k] += 1 + xik / h
        else
            j = clamp(searchsortedlast(knots, xi), 1, k - 1)
            h = knots[j + 1] - knots[j]
            ajm = knots[j + 1] - xi
            ajp = xi - knots[j]
            cjm = ajm * (ajm * ajm / h - h) / 6
            cjp = ajp * (ajp * ajp / h - h) / 6
            for q in 1:k
                X[i, q] = cjm * F[j, q] + cjp * F[j + 1, q]
            end
            X[i, j] += ajm / h
            X[i, j + 1] += ajp / h
        end
    end
    X
end

function _rk_fit_cr_spline(x::AbstractVector{<:Real}; k::Int=5)
    k > 2 || _rk_spline_fail("`t2` basis dimensions must be integers greater than 2 (got $k)")
    xs = collect(Float64, x)
    isempty(xs) && _rk_spline_fail("`t2` cannot use an empty margin")
    all(isfinite, xs) || _rk_spline_fail("`t2` margins require finite numeric data")
    shift = sum(xs) / length(xs)
    scale = maximum(xs) - minimum(xs)
    scale > 0 || _rk_spline_fail("`t2` margin is degenerate (all values equal)")
    normalized = (xs .- shift) ./ scale
    knots = _rk_type7_knots(normalized, k)
    F, penalty = _rk_cr_second_derivative_map(knots)

    eig_penalty = eigen(Symmetric(penalty))
    order = sortperm(eig_penalty.values; rev=true)
    keep = order[1:(k - 2)]
    penalty_scale = maximum(abs, eig_penalty.values)
    tol = penalty_scale * eps(Float64) * 100
    minimum(eig_penalty.values) >= -tol || _rk_spline_fail(
        "`t2` cubic-regression-spline penalty is not positive semidefinite")
    minimum(eig_penalty.values[keep]) > tol || _rk_spline_fail(
        "`t2` could not isolate the two-dimensional marginal null space")
    range_projection = eig_penalty.vectors[:, keep] *
                       Diagonal(inv.(sqrt.(eig_penalty.values[keep])))

    null_const_scale = inv(sqrt(length(xs)))
    slope_norm = norm(normalized)
    slope_norm > 0 || _rk_spline_fail("`t2` margin has a zero linear null-space norm")

    (; shift, scale, knots, F, range_projection, null_const_scale, slope_norm, k)
end

function _rk_apply_cr_spline(fit, x::AbstractVector{<:Real})
    xs = collect(Float64, x)
    all(isfinite, xs) || _rk_spline_fail("`t2` margins require finite numeric data")
    normalized = (xs .- fit.shift) ./ fit.scale
    Xnull = hcat(fill(fit.null_const_scale, length(xs)),
                 normalized ./ fit.slope_norm)
    range = _rk_cr_basis(fit.knots, fit.F, normalized) * fit.range_projection
    Xnull, range
end

function _rk_row_tensor(A::AbstractMatrix, B::AbstractMatrix)
    size(A, 1) == size(B, 1) || _rk_spline_fail(
        "`t2` marginal basis row counts differ ($(size(A, 1)) vs $(size(B, 1)))")
    out = Matrix{Float64}(undef, size(A, 1), size(A, 2) * size(B, 2))
    for i in axes(out, 1), a in axes(A, 2), b in axes(B, 2)
        out[i, (a - 1) * size(B, 2) + b] = A[i, a] * B[i, b]
    end
    out
end

function _rk_t2_raw_blocks(margins, x, z)
    N1, R1 = _rk_apply_cr_spline(margins[1], x)
    N2, R2 = _rk_apply_cr_spline(margins[2], z)
    NN = _rk_row_tensor(N1, N2)
    (fixed=Matrix(NN[:, 2:end]), rr=_rk_row_tensor(R1, R2),
     rn=_rk_row_tensor(R1, N2), nr=_rk_row_tensor(N1, R2))
end

_rk_block_center(A::AbstractMatrix) = vec(sum(A; dims=1)) ./ size(A, 1)
_rk_center_block(A::AbstractMatrix, center) = A .- reshape(center, 1, :)

function _rk_fit_t2(x::AbstractVector{<:Real}, z::AbstractVector{<:Real};
                    k::Tuple{Int,Int}=(5, 5))
    length(x) == length(z) || _rk_spline_fail(
        "`t2(x, z)` margins must have equal lengths ($(length(x)) vs $(length(z)))")
    margins = (_rk_fit_cr_spline(x; k=k[1]), _rk_fit_cr_spline(z; k=k[2]))
    raw = _rk_t2_raw_blocks(margins, x, z)
    fixed_center = _rk_block_center(raw.fixed)
    (; margins, fixed_center, k)
end

function _rk_apply_t2(fit, x::AbstractVector{<:Real}, z::AbstractVector{<:Real})
    length(x) == length(z) || _rk_spline_fail(
        "`t2(x, z)` margins must have equal lengths ($(length(x)) vs $(length(z)))")
    raw = _rk_t2_raw_blocks(fit.margins, x, z)
    (_rk_center_block(raw.fixed, fit.fixed_center), raw.rr, raw.rn, raw.nr)
end

"""Design-matrix recipe name for a predictor (`mu` → `_ppl_design_mu`)."""
design_name(predictor::Symbol) = Symbol(:_ppl_design_, predictor)

"""Offset-vector recipe name for a predictor (`mu` → `_ppl_offset_mu`)."""
offset_name(predictor::Symbol) = Symbol(:_ppl_offset_, predictor)

"""Monotonic-contrast recipe name for an increments simplex (`s` → `_ppl_mo_s`)."""
monotonic_name(increments::Symbol) = Symbol(:_ppl_mo_, increments)

# Cumulative-contrast temporary for level j (`_ppl_mo_cum_s_1 == 0.0`,
# `_ppl_mo_cum_s_j = cum_(j-1) + s_(j-1)`): the in-graph `cumsum([0; incr])`.
_monotonic_cum_name(increments::Symbol, j::Int) =
    Symbol(:_ppl_mo_cum_, increments, :_, j)

"""
    monotonic_recipe(increments, idx, K) -> Vector{Expr}

The SB `_sb_mo` level-gather recipe for one increments simplex:
`cumsum([0; incr])[idx]` from the bound index column `idx` (integer codes
1..K), where the K−1 increments are the `_ppl_v_` layout elements of the
`increments` simplex. Cumulative level temps mirror the `cumsum`
(`_ppl_mo_cum_s_1 = 0.0`, each later level adds one increment); the gather
is an indicator sum over levels 2..K (level 1 contributes 0 by
construction), the vectorized ranef-encoder shape — dotted ops only, so
the Enzyme reverse pass sees no new surface. Emitted before the
predictor's design recipe; the LP splice (per-block `mo` term or direct
`mo1` summand) reads the `_ppl_mo_<s>` contrast.
"""
function monotonic_recipe(increments::Symbol, idx::Symbol, K::Int)
    K >= 2 || throw(ContractValidationError(
        "[preprocessing] monotonic levels K=$K < 2 " *
        "(K=1 degenerates emitter-side)"))
    stmts = Expr[]
    push!(stmts, :($(_monotonic_cum_name(increments, 1))::Float64 = 0.0))
    for j in 2:K
        cum = _monotonic_cum_name(increments, j)
        prev = _monotonic_cum_name(increments, j - 1)
        push!(stmts, :($cum::Float64 =
            $prev + $(_vector_elt_name(increments, j - 1))))
    end
    parts = Any[Expr(:call, :.*,
        Expr(:call, :.==, idx, j), _monotonic_cum_name(increments, j))
        for j in 2:K]
    gather = foldl((a, c) -> Expr(:call, :.+, a, c), parts)
    push!(stmts, Expr(:(=), monotonic_name(increments), gather))
    return stmts
end

"""
    design_recipe(shape, n_obs) -> Union{Nothing,Expr}

`_ppl_design_<pred> = Float64.(hcat(<blocks…>))`, or `nothing` for a
predictor with no static (data-only) blocks. Blocks: intercept →
`ones(n)`, continuous → the bare column, factor → full-rank dummies over
mapped levels. Monotonic blocks contribute no part (their contrast is
parameter-derived — `hcat` cannot mix data with symbolic columns under
the Enzyme reverse pass — so the generator splices them per-block
against their coefficient coordinate instead).
"""
function design_recipe(shape::DesignShape, n_obs::Int)
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
    isempty(parts) && return nothing
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

All preprocessing recipes for a plan, in `plan.predictors` order: each
predictor's monotonic contrasts (one level-gather recipe per `mo`/`mo1`
term — the only parameter-dependent recipes — before anything that reads
them), then its design matrix (unless it has no static blocks, or the
predictor splices `mo` columns per-block — the fused matvec is that
recipe's only reader, so emitting it there would be dead code) and its
offset vector (unless absent).
"""
function preprocessing_recipes(plan::StructuralPlan)
    stmts = Expr[]
    for pred in plan.predictors
        shape = design_shape(pred, plan.columns; levelmaps = plan.levelmaps)
        for t in pred.terms
            (t.kind === MonotonicTerm || t.kind === MonotonicSummandTerm) ||
                continue
            append!(stmts, monotonic_recipe(t.options.increments,
                only(t.columns), _monotonic_K(plan, t)))
        end
        if !any(b -> b.kind === MonotonicTerm, shape.blocks)
            recipe = design_recipe(shape, plan.n_obs)
            recipe !== nothing && push!(stmts, recipe)
        end
        off = offset_recipe(shape)
        off !== nothing && push!(stmts, off)
    end
    return stmts
end

# Level count K for a monotonic term: one more than its linked increments
# simplex's bound size (K−1 increments). The plan is bound here, so the
# size is concrete; a residual `nothing` is loud defense in depth.
function _monotonic_K(plan::StructuralPlan, t::TermSpec)
    i = findfirst(p -> p.name === t.options.increments,
        plan.vector_parameters)
    i === nothing && throw(ContractValidationError(
        "[preprocessing] monotonic increments $(t.options.increments) " *
        "is not a vector parameter"))
    m = plan.vector_parameters[i].size
    m === nothing && throw(ContractValidationError(
        "[preprocessing] monotonic increments $(t.options.increments) " *
        "has unresolved size (bind_data infers it)"))
    return m + 1
end

# Full-rank dummies over mapped levels:
# `Float64.(g .== permutedims([l1, l2, …]))`.
function _contrast_expr(b::DesignBlock)
    @assert b.kind === FactorTerm
    lvlvec = Expr(:vect, (_level_literal(lvl) for lvl in b.levels)...)
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
