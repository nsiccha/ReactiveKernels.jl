# Program generator: StructuralPlan → self-contained `@kernel` program.
#
# Emission order per program: layout transforms (unconstrained →
# constrained) → scalar + derived assignments in topo order → preprocessing
# recipes (data-only, folded by `bound=`) → linear predictors →
# per-response plate likelihoods over distribution-kernel endpoints → prior
# terms → canonical `prior`/`likelihood`/`log_jacobian`/`posterior` nodes.
# Recipes sit after assignments because design/offset blocks may reference
# derived-column locals; folding is input-driven, so position changes
# nothing for data-only recipes. Plates compile to allocation-free loops
# with shared work hoisted; constraining stays hand-rolled (no bijectors).
# The `@kernel` def is evaluated in the dedicated `PPLGeneratedModels` scope
# (counter-suffixed binding per build).

"""
    build_kernel(plan) -> (; spec, layout)

Validate, assign layout, emit, and evaluate a self-contained `@kernel`
program for `plan`. `spec` is the `KernelSpec` (callable after `prepare`
with `have=(:unconstrained, data…)`); `layout` is its
[`LayoutTable`](@ref) (R10 read API for the sampler side).
"""
function build_kernel(plan::StructuralPlan)
    validate_plan(plan)
    isbound(plan) || throw(ContractValidationError(
        "[generator] build_kernel requires a bound plan (bind_data first)"))
    layout = assign_layout(plan)
    def = kernel_expr(plan, layout)
    spec = _eval_kernel_def(def)
    return (; spec, layout)
end

"""
    kernel_expr(plan, layout; name=:ppl_model) -> Expr

The `@kernel` definition expression (`Expr(:(=), signature, body)`).
Pure (no eval): the generator tests inspect and evaluate it.
"""
function kernel_expr(plan::StructuralPlan, layout::LayoutTable; name::Symbol = :ppl_model)
    validate_plan(plan)
    isbound(plan) || throw(ContractValidationError(
        "[generator] kernel_expr requires a bound plan (bind_data first)"))
    stmts = Expr[]
    for e in layout.entries
        append!(stmts, transform_statements(e))
    end
    append!(stmts, _assignment_statements(plan))
    append!(stmts, preprocessing_recipes(plan))
    append!(stmts, _varying_statements(plan))
    append!(stmts, _hsgp_basis_statements(plan))
    append!(stmts, _scan_reconstruction_statements(plan, layout))
    append!(stmts, _predictor_statements(plan))
    append!(stmts, _likelihood_statements(plan))
    append!(stmts, _prior_statements(plan, layout))
    push!(stmts, _log_jacobian_statement(layout))
    push!(stmts, :(posterior::Float64 = prior + likelihood + log_jacobian))
    push!(stmts, :(return posterior))
    sig = Expr(:call, name, :(unconstrained::Vector{Float64}),
        (_data_arg(colname, col) for (colname, col) in _ordered_columns(plan))...)
    return Expr(:(=), sig, Expr(:block, stmts...))
end

_ordered_columns(plan::StructuralPlan) =
    sort!(collect(plan.columns); by = first)

_data_arg(name::Symbol, col::AbstractVector) =
    Expr(:(::), name, Vector{eltype(col)})

# Dedicated eval scope for generated models. The `using` lines resolve via
# this package's own Project (by file location), so generated code loads in
# ANY consumer session with no LOAD_PATH dependence. One counter-suffixed
# binding per build; slice-1 scale makes interning harmless.
module PPLGeneratedModels
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources:
    normal, bernoulli, poisson, cauchy, exponential, gamma, lognormal,
    beta, inverse_gamma, binomial, negative_binomial2,
    gp_exp_quad_cov, gp_chol_latent
using SpecialFunctions: erfc, loggamma
# Selective (explicit imports win over any re-export chain, so no `using`
# ambiguity if ReactiveKernels ever exports these too): the only Statistics
# names in the assignment allowlist.
using Statistics: mean, std, var
using LinearAlgebra: dot
using LogExpFunctions: log1pexp
# Bijector objects the generated program splices (constrained-parameter
# transforms); imported from the enclosing module so the emitted
# `positive_bijector()` / `unit_bijector()` calls resolve.
import ..positive_bijector, ..unit_bijector
# In-model grouping encoder (`_ppl_gidx_<group>` nodes call it with the
# raw column + literal declared levels).
import .._declared_codes
end

const _MODEL_COUNTER = Ref(0)

function _eval_kernel_def(def::Expr)
    _MODEL_COUNTER[] += 1
    name = Symbol(:ppl_model_, _MODEL_COUNTER[])
    sig = def.args[1]
    renamed = Expr(:(=), Expr(:call, name, sig.args[2:end]...), def.args[2])
    call =
        Expr(:macrocall, Symbol("@kernel"), LineNumberNode(1, :generator), renamed)
    Core.eval(PPLGeneratedModels, call)
    return Core.eval(PPLGeneratedModels, name)
end

# Scalar + derived assignments in topo order (params already constrained
# above, so every scalar name resolves; derived columns resolve as locals
# for the recipes below). Unannotated: Int temporaries (e.g. `length`)
# must not meet a Float64 assertion.
function _assignment_statements(plan::StructuralPlan)
    by_name = Dict{Symbol,Any}(a.name => a for a in plan.assignments)
    for d in plan.derived
        by_name[d.name] = d
    end
    return Expr[:($(name) = $(by_name[name].expr))
        for name in topological_order(plan) if haskey(by_name, name)]
end

_lp_name(pred::PredictorSpec) = Symbol(:_ppl_lp_, pred.name)

function _predictor_statements(plan::StructuralPlan)
    stmts = Expr[]
    for pred in plan.predictors
        shape = design_shape(pred, plan.columns; levelmaps = plan.levelmaps)
        lp = _lp_name(pred)
        terms = Any[]
        if any(b -> b.kind === MonotonicTerm, shape.blocks)
            append!(terms, _mo_block_terms(plan, shape))
        elseif shape.width > 0
            push!(terms, :($(design_name(pred.name)) * $(block_name(pred.name))))
        end
        if any(b -> b.kind === OffsetTerm, shape.blocks)
            push!(terms, offset_name(pred.name))
        end
        # A monotonic summand (mo1) contributes its contrast directly —
        # beta-free, the offset-arm shape with a parameter-derived column.
        for t in pred.terms
            t.kind === MonotonicSummandTerm &&
                push!(terms, monotonic_name(t.options.increments))
        end
        # A latent term contributes the per-cell latent VECTOR directly
        # (identity design): `lp = theta` on its own, or added to fixed-effect
        # design/offset terms for a random-intercept-plus-covariates predictor.
        for b in shape.blocks
            b.kind === LatentTerm && push!(terms, b.column)
        end
        # A spline summand contributes its basis's direct summand expression
        # (SB's `X*b + Z*(sd*z)` shape over materialized basis columns and
        # SplineVector layout blocks).
        for b in shape.blocks
            b.kind === SplineSummandTerm &&
                push!(terms, _spline_summand_expr(plan, b.column))
        end
        # An HSGP summand contributes its basis's direct `PHI * w`
        # expression (SB `_sb_hsgp`'s `PHI * (sqrt_spd .* beta_raw)`,
        # evaluated in-graph by `_hsgp_basis_statements`).
        for b in shape.blocks
            b.kind === HSGPSummandTerm &&
                push!(terms, _hsgp_summand_expr(plan, b.column))
        end
        # A varying effect contributes its draws block's direct `r`
        # expression (SB's `r_<target>_<suffix>` summand), resolved from
        # the TERMS — the draws label does not fit a design block.
        for t in pred.terms
            t.kind === VaryingEffectTerm &&
                push!(terms, _varying_effect_expr(plan, pred, t))
        end
        # A scan summand contributes its state's direct scaled expression
        # (`state .* coef`, SB's `ar` latent path with its free beta),
        # resolved from the TERMS like any summand.
        for t in pred.terms
            t.kind === ScanSummandTerm &&
                push!(terms, _scan_summand_expr(plan, pred, t))
        end
        # Degenerate (e.g. single-level-factor-only) predictors carry a scalar
        # zero LP, which broadcasts everywhere a vector LP would.
        rhs = isempty(terms) ? :(0.0) : foldl((a, b) -> :($a + $b), terms)
        push!(stmts, :($lp = $rhs))
    end
    return stmts
end

# Scalar coefficient-coordinate read (`sum(view(coef, k:k))`, the
# `coordinate_read` shape over a coefficient block rather than the packed
# vector).
_coef_coord(coef::Symbol, k::Int) = :(sum(view($coef, $k:$k)))

# Per-block LP terms for a predictor with `mo` columns. The fused
# `design * coef` matvec cannot cover a monotonic block — its contrast
# column is parameter-derived, and `hcat` cannot mix data with symbolic
# columns under the Enzyme reverse pass — so each coefficient-carrying
# block splices against its own coefficient coordinates (positions follow
# design order, the layout block's own order): intercept/continuous/
# monotonic blocks scale one column by one coordinate, factor blocks keep
# the data-matrix × coefficient-slice matvec. Predictors without `mo`
# keep the fused form above, untouched.
function _mo_block_terms(plan::StructuralPlan, shape::DesignShape)
    coef = block_name(shape.predictor)
    terms = Any[]
    k = 1
    for b in shape.blocks
        if b.kind === InterceptTerm
            push!(terms, Expr(:call, :.*, Expr(:call, :ones, plan.n_obs),
                _coef_coord(coef, k)))
            k += 1
        elseif b.kind === ContinuousTerm
            push!(terms, :($(b.column) .* $(_coef_coord(coef, k))))
            k += 1
        elseif b.kind === FactorTerm
            w = b.width
            push!(terms, :($(_contrast_expr(b)) *
                $(:(view($coef, $k:$(k + w - 1))))))
            k += w
        elseif b.kind === MonotonicTerm
            push!(terms,
                :($(monotonic_name(b.column)) .* $(_coef_coord(coef, k))))
            k += 1
        end
    end
    return terms
end

# One basis's direct summand as a scaled-column sum (SB `_sb_s_generic` /
# `_sb_t2_generic`): fixed blocks `X[j] .* b[j]`, pen blocks
# `Z[j] .* (sd[k] * r[j])`, all joined with `.+`. Reads the BOUND basis's
# materialized columns (bind asserted widths) and the `_spline_block_roles`
# vector names (the contract's single source — no re-derivation here).
function _spline_summand_expr(plan::StructuralPlan, id::Symbol)
    i = findfirst(b -> b.id === id, plan.spline_bases)
    i === nothing && throw(ContractValidationError(
        "[generator] spline summand addresses unknown basis :$id"))
    sb = plan.spline_bases[i]
    byblock = Dict{Symbol,SplineBasisBlock}(b.name => b for b in sb.blocks)
    roles, sd = _spline_block_roles(sb.id, sb.kind, sb.k)
    parts = Any[]
    for (block, coef, sdidx) in roles
        haskey(byblock, block) || throw(ContractValidationError(
            "[generator] spline :$id basis is missing block :$block"))
        cols = byblock[block].columns
        isempty(cols) && throw(ContractValidationError(
            "[generator] spline :$id block :$block has no materialized " *
            "columns (bind_data fills these)"))
        for (j, c) in enumerate(cols)
            cel = Expr(:call, :.*, c, Expr(:ref, coef, j))
            if sdidx !== nothing
                scaled = Expr(:call, :*, Expr(:ref, sd, sdidx),
                    Expr(:ref, coef, j))
                cel = Expr(:call, :.*, c, scaled)
            end
            push!(parts, cel)
        end
    end
    return foldl((a, b) -> :($a .+ $b), parts)
end

# `sqrt(2π)` verbatim from SB `brm_hsgp_sqrt_spd` (the spectral scale).
const _HSGP_SQRT2PI = 2.5066282746310002

# In-graph HSGP node names for one basis (all `_ppl_`-hygienic): per-axis
# trig columns, tensor-product columns, the `hcat` basis matrix, the
# spectral scale, per-basis `sqrt_spd` scalars, their `vect`, the
# spectral weights, and the predictor summand.
_hsgp_ax_name(id::Symbol, j::Int, k::Int) = Symbol(:_ppl_hsgp_, id, :_ax, j, :_k, k)
_hsgp_phi_name(id::Symbol, b::Int) = Symbol(:_ppl_hsgp_, id, :_phi_, b)
_hsgp_PHI_name(id::Symbol) = Symbol(:_ppl_hsgp_, id, :_PHI)
_hsgp_sscale_name(id::Symbol) = Symbol(:_ppl_hsgp_, id, :_sscale)
_hsgp_s_name(id::Symbol, b::Int) = Symbol(:_ppl_hsgp_, id, :_s_, b)
_hsgp_S_name(id::Symbol) = Symbol(:_ppl_hsgp_, id, :_S)
_hsgp_w_name(id::Symbol) = Symbol(:_ppl_hsgp_, id, :_w)
_hsgp_sum_name(id::Symbol) = Symbol(:_ppl_hsgp_, id)

# One basis's in-graph evaluation (SB `_brm_apply_hsgp` /
# `brm_hsgp_sqrt_spd` / `_sb_hsgp`, SB op order throughout): per-axis 1D
# trig columns from the frozen bind fits (`(mu, L)` literals), their
# tensor-product columns in `CartesianIndices(K)` order, the `hcat` basis
# matrix, unrolled `sqrt_spd` scalars over the sampled `(rho, sigma)`,
# and the spec-literal matmul summand `PHI * (S .* beta)`. The basis
# columns are data-only (bound-folded, the `design_recipe` precedent);
# the spectral weights stay symbolic. Runs before the predictors (the
# summand node is the LP splice); the priors stay in `_prior_statements`
# (order-free).
function _hsgp_basis_statements(plan::StructuralPlan)
    stmts = Expr[]
    for hb in plan.hsgp_bases
        length(hb.fits) == length(hb.axes) || throw(ContractValidationError(
            "[generator] hsgp :$(hb.id): fits not filled at bind " *
            "(bind_data fills one (mu, L) per axis)"))
        append!(stmts, _hsgp_basis_stmts(hb))
    end
    return stmts
end

function _hsgp_basis_stmts(hb::HSGPBasis)
    id = hb.id
    d = length(hb.axes)
    stmts = Expr[]
    # Per-axis 1D columns: `PHI[i,k] = inv_sqrt_L * sin(lam_sqrt[k] *
    # (x[i] - mu + L))` (SB `_brm_apply_hsgp`, element order verbatim).
    # `lam[k]` is SB's `lambda` literal, `lam_sqrt[k]` its `sqrt`.
    for (j, axis) in enumerate(hb.axes)
        mu, L = hb.fits[j]
        mu, L = Float64(mu), Float64(L)
        inv_sqrt_L = 1.0 / sqrt(L)
        for k in 1:hb.K[j]
            lam_sqrt = sqrt((k * pi / (2.0 * L))^2)
            col = _hsgp_ax_name(id, j, k)
            push!(stmts, :($col =
                $inv_sqrt_L .* sin.($lam_sqrt .* ($axis .- $mu .+ $L))))
        end
    end
    # Tensor-product columns in `CartesianIndices(K)` order (SB's
    # `enumerate(CartesianIndices(K))`): one axis reuses its column.
    midcs = collect(CartesianIndices(Tuple(hb.K)))
    phis = Symbol[]
    for (b, I) in enumerate(midcs)
        if d == 1
            push!(phis, _hsgp_ax_name(id, 1, I[1]))
        else
            phi = _hsgp_phi_name(id, b)
            cols = [_hsgp_ax_name(id, j, I[j]) for j in 1:d]
            push!(stmts, :($phi = $(foldl((a, c) -> :($a .* $c), cols))))
            push!(phis, phi)
        end
    end
    push!(stmts, :($(_hsgp_PHI_name(id)) = hcat($(phis...))))
    # Spectral weights (SB `brm_hsgp_sqrt_spd`): `scale = sigma *
    # prod(sqrt(rho_j * sqrt(2π)))`, `s[b] = scale * exp(-0.25 *
    # sum(rho_j^2 * omega2[b,j]))` — left-assoc folds, SB order. Iso
    # shares one rho across axes; `omega2` is the frozen `lambda`
    # literal above.
    names = _hsgp_names(hb)
    rhos = hb.iso ? fill(names.rhos[1], d) : names.rhos
    factors = Any[names.sigma]
    for j in 1:d
        push!(factors, :(sqrt($(rhos[j]) * $_HSGP_SQRT2PI)))
    end
    sscale = _hsgp_sscale_name(id)
    push!(stmts, :($sscale::Float64 = $(foldl((a, c) -> :($a * $c), factors))))
    snames = Symbol[]
    for (b, I) in enumerate(midcs)
        terms = Any[]
        for j in 1:d
            lam = (I[j] * pi / (2.0 * hb.fits[j][2]))^2
            push!(terms, :($(rhos[j]) * $(rhos[j]) * $lam))
        end
        expsum = foldl((a, c) -> :($a + $c), terms)
        s = _hsgp_s_name(id, b)
        push!(stmts, :($s::Float64 = $sscale * exp(-0.25 * $expsum)))
        push!(snames, s)
    end
    S = _hsgp_S_name(id)
    push!(stmts, :($S = $(Expr(:vect, snames...))))
    w = _hsgp_w_name(id)
    push!(stmts, :($w = $S .* $(names.beta)))
    push!(stmts, :($(_hsgp_sum_name(id)) = $(_hsgp_PHI_name(id)) * $w))
    return stmts
end

# One HSGP summand's direct expression: the basis's precomputed
# `_hsgp_basis_statements` node (resolved from the design block's basis
# id; the lookup below is loud defense in depth).
function _hsgp_summand_expr(plan::StructuralPlan, id::Symbol)
    any(hb -> hb.id === id, plan.hsgp_bases) || throw(ContractValidationError(
        "[generator] hsgp summand addresses unknown basis :$id"))
    return _hsgp_sum_name(id)
end

# One scan summand's direct expression (`state .* coef`, explicit dotted
# form): the in-graph recurrence state scaled by its sampled scalar
# coefficient. Both names resolve from the term's options (validated
# up front; the lookups below are loud defense in depth).
function _scan_summand_expr(plan::StructuralPlan, pred::PredictorSpec, t::TermSpec)
    o = t.options
    any(s -> s.state === o.scan_id, plan.scans) || throw(ContractValidationError(
        "[generator] scan summand in predictor $(pred.name) addresses " *
        "unknown scan :$(o.scan_id)"))
    any(p -> p.name === o.coef, plan.parameters) || throw(ContractValidationError(
        "[generator] scan summand coef :$(o.coef) is not a sampled parameter"))
    return Expr(:call, :.*, o.scan_id, o.coef)
end

# Group-index encoder nodes, one per grouped column (`_ppl_gidx_<group>`):
# an in-model `_declared_codes` call over the draws' DECLARED levels
# (bind-known — filled or emitter-provided — so the order agrees with
# validation by construction; never sorted). Data-only, hence
# bound-folded; strings are native-only, exactly like factor contrasts.
# K=1 and correlated draws on the same group share one encoder
# (per-group dedup; same-group levels agreement is validated).
function _varying_statements(plan::StructuralPlan)
    stmts = Expr[]
    groups = Symbol[]
    for d in plan.varying_draws
        d.group in groups && continue
        push!(groups, d.group)
        d.levels === nothing && throw(ContractValidationError(
            "[generator] internal: draws $(d.label) has no declared " *
            "levels (validate_plan proves this)"))
        lvlvec = Expr(:vect,
            (_level_literal(lv) for lv in d.levels)...)
        push!(stmts, Expr(:(=), Symbol(:_ppl_gidx_, d.group),
            Expr(:call, :_declared_codes, d.group, lvlvec)))
    end
    return stmts
end

# Term-to-draws join by label, plus the (draws, target) slice (unique
# by validation). The term carries the draws label; the slice carries
# the explicit column range — the generator never re-derives ranges.
function _slice_draws(plan::StructuralPlan, pred::PredictorSpec,
        t::TermSpec)
    i = findfirst(d -> d.label === t.options.draws, plan.varying_draws)
    i === nothing && throw(ContractValidationError(
        "[generator] effect term addresses unknown draws " *
        "($(t.options.draws))"))
    d = plan.varying_draws[i]
    si = findfirst(s -> s.draws === d.label && s.target === pred.name,
        plan.varying_slices)
    si === nothing && throw(ContractValidationError(
        "[generator] internal: effect term of draws $(d.label) in " *
        "predictor $(pred.name) has no slice (validate_plan proves this)"))
    return d, plan.varying_slices[si]
end

# One varying margin's Z as an rvalue: bare columns stay bare (raw
# ports and derived locals alike); dummies compare against the
# bind-known level value. `:ones` never reaches emission (an intercept
# needs no Z multiply, and slope1 margins are never `:ones` by kind
# dispatch).
function _varying_z_expr(z::VaryingZRecipe)
    z.kind === :column && return z.column
    z.kind === :dummy &&
        return Expr(:call, :.==, z.column, _level_literal(z.level))
    throw(ContractValidationError(
        "[generator] internal: ones-Z reached effect emission"))
end

# One varying draws block's direct `r` summand, SB-literal (SB's K=1
# intercept/slope math and association order — no `b` node, the draws
# stay implicit): intercept `exp(log_scale) * xi[idx]`, slope
# `tau * (xi[idx] .* Z)`. Correlated draws take the K² implicit-draws
# arm below (this slice's columns only).
function _varying_effect_expr(plan::StructuralPlan, pred::PredictorSpec,
        t::TermSpec)
    d, s = _slice_draws(plan, pred, t)
    d.kind === :correlated &&
        return _varying_corr_effect_expr(plan, d, s)
    scale, xi = _varying_k1_names(d)
    gathered = Expr(:ref, xi, Symbol(:_ppl_gidx_, d.group))
    if d.kind === :intercept1
        return Expr(:call, :*, Expr(:call, :exp, scale), gathered)
    end
    Z = _varying_z_expr(only(d.margins).z)
    return Expr(:call, :*, scale, Expr(:call, :.*, gathered, Z))
end

# One correlated draws block's direct `r` summand for one slice
# (SB `rows_dot_product(Z, b[idx,cols])` with the draws implicit — the
# no-`b`-node precedent): per slice margin j,
# `Z_j .* sum_s (tau[j]*L[j,s]) .* z_flat[s + (gidx-1)*K]` over
# `s in 1:j` (L lower-triangular — the `s > j` terms are structural
# zeros, never emitted). `:ones` Z drops the factor (multiply by 1).
# K, the slice range, and the `s` bound are all static; tau reads are
# scalar refs (the coefficient-block precedent) and L reads the named
# `_ppl_rl_` scalars from the layout edges.
function _varying_corr_effect_expr(plan::StructuralPlan, d::VaryingDraws,
        s::VaryingSlice)
    cols = s.columns
    K = length(d.margins)
    L, tau, z = _varying_corr_names(d)
    gidx = Symbol(:_ppl_gidx_, d.group)
    parts = Any[]
    for j in cols
        m = d.margins[j]
        inner = Any[]
        for q in 1:j
            A = :($(Expr(:ref, tau, j)) * $(_rl_name(L, j, q)))
            idx = :($q .+ ($gidx .- 1) .* $K)
            push!(inner, :($A .* $(Expr(:ref, z, idx))))
        end
        sj = foldl((a, c) -> :($a .+ $c), inner)
        if m.z.kind === :ones
            push!(parts, sj)
        else
            push!(parts, :($(_varying_z_expr(m.z)) .* $sj))
        end
    end
    return foldl((a, c) -> :($a .+ $c), parts)
end

_lik_name(label::Symbol) = Symbol(:_ppl_lik_, label)

function _likelihood_statements(plan::StructuralPlan)
    stmts = Expr[]
    terms = Any[]
    for r in plan.responses
        append!(stmts, _response_likelihood_stmts(r, plan))
        push!(terms, _lik_name(r.label))
    end
    for kp in plan.kernel_plates
        kstmts, kterm = _kernel_plate_likelihood(kp, plan)
        append!(stmts, kstmts)
        push!(terms, kterm)
    end
    joint = foldl((a, b) -> :($a + $b), terms; init = :(0.0))
    push!(stmts, :(likelihood::Float64 = $joint))
    return stmts
end

# Panel-kernel likelihood (flat codegen): panel-v1 cell bodies are
# elementwise, so the subject map dissolves into flat vector ops over the
# `n_sub*T` block (numerically identical to a per-subject loop for the
# admitted subset): slice params rewrite to flat column refs (scalar
# slices to their bind-time T-block expansions), cell assignments emit as
# flat statements, and the single Gaussian obs lowers as a flat plate
# reusing the plate-sum machinery. The collected name aliases its flat
# value (future generated quantities read it).
function _kernel_plate_likelihood(kp::KernelPlate, plan::StructuralPlan)
    isbound(plan) ||
        throw(ContractValidationError("[generator] kernel plates lower " *
              "from a bound plan (bind_data first)"))
    kp.subjects isa Int ||
        throw(ContractValidationError("[generator] kernel plate " *
              "`$(kp.result)` subjects unresolved (bind_data with dims first)"))
    flatmap = _kernel_flatmap(kp)
    stmts = Expr[]
    for (nm, ex) in _canonicalize_kernel_assignments(kp)
        push!(stmts, :($nm = $(_rewrite_kernel_refs(ex, flatmap))))
    end
    obs = kp.obs
    obs.family === GaussianFam ||
        throw(ContractValidationError("[generator] kernel plate " *
              "`$(kp.result)` obs family $(obs.family) has no emitter " *
              "(panel v1: Gaussian only)"))
    inputs = Any[flatmap[obs.response]]
    rv = _dovar(1)
    # Location threads as input 2 when symbolic (the `_ppl_lp_` precedent:
    # computed flat locals ride as plate inputs); literals inline.
    locv = _thread_ref!(inputs, _kernel_obs_ref(obs.location, flatmap))
    sref = _thread_ref!(inputs, _kernel_obs_ref(obs.scale, flatmap))
    cell = :(normal($locv, $sref).logpdf($rv))
    klabel = Symbol(:kernel_, kp.result)
    append!(stmts, _plate_sum_stmts(_pw_name(klabel), _lik_name(klabel),
        inputs, cell))
    collected = haskey(flatmap, kp.collected) ? flatmap[kp.collected] : kp.collected
    collected === kp.result ||
        push!(stmts, :($(kp.result) = $collected))
    return stmts, _lik_name(klabel)
end

# Slice params to flat refs: vector slices ride their flat T-blocked
# column; scalar slices ride the bind-time T-block expansion (or the raw
# column in all-scalar models, where no expansion exists).
function _kernel_flatmap(kp::KernelPlate)
    flatmap = Dict{Symbol,Symbol}()
    for (col, param, kind) in kp.slices
        kind in (:vector, :scalar) ||
            throw(ContractValidationError("[generator] kernel plate " *
                  "`$(kp.result)` slice `$param` kind unresolved " *
                  "(bind_data first)"))
        flatmap[param] =
            (kind === :vector || kp.timepoints === nothing) ? col :
            _kexp_name(kp.result, col)
    end
    return flatmap
end

# Obs location/scale through the flatmap (slice params only); cell
# locals, globals, and literals pass to `_thread_ref!` unchanged.
_kernel_obs_ref(ref, flatmap::Dict{Symbol,Symbol}) =
    ref isa Symbol && haskey(flatmap, ref) ? flatmap[ref] : ref

function _rewrite_kernel_refs(ex, flatmap::Dict{Symbol,Symbol})
    ex isa Symbol && return get(flatmap, ex, ex)
    ex isa Expr || return ex
    return Expr(ex.head, (_rewrite_kernel_refs(a, flatmap) for a in ex.args)...)
end

# One plate likelihood per response (pointwise plate + scalar sum node).
# Triples 2 and 3 (Bernoulli-logit) lower identically; the triple only
# selects the form. Branches are explicit per family; the else is a
# fail-closed guard for enum members without an emitter (never silent).
function _response_likelihood_stmts(r::LikelihoodSpec, plan::StructuralPlan)
    node = _lik_name(r.label)
    pw = _pw_name(r.label)
    if r.family === GaussianFam
        return _gaussian_plate_stmts(r, plan, node, pw)
    elseif r.family === BernoulliLogitFam
        # Base GLM case (no evidence, no weights, no literal range): fused
        # whole-vector reduction. Ranged responses stay on the plate path
        # (the cover rule makes them whole-column today, but the fused sum
        # must never silently outgrow a future partial range).
        r.evidence.kind === :none && r.weights === nothing &&
            r.range === nothing &&
            return _bernoulli_wholevec_stmts(r, plan, node)
        return _bernoulli_plate_stmts(r, plan, node, pw)
    elseif r.family === PoissonLogFam
        # Base GLM case (no evidence, no weights, no literal range): fused
        # whole-vector reduction (faster native + Reactant; the per-cell
        # plate handles evidence/weights/ranges).
        r.evidence.kind === :none && r.weights === nothing &&
            r.range === nothing &&
            return _poisson_wholevec_stmts(r, plan, node)
        return _poisson_plate_stmts(r, plan, node, pw)
    elseif r.family === BinomialLogitFam
        return _binomial_plate_stmts(r, plan, node, pw)
    elseif r.family === NegativeBinomial2Fam
        return _nb2_plate_stmts(r, plan, node, pw)
    elseif r.family === GammaLogFam
        return _gamma_plate_stmts(r, plan, node, pw)
    elseif r.family === BernoulliProbitFam
        return _bernoulli_probit_plate_stmts(r, plan, node, pw)
    elseif r.family === BernoulliCloglogFam
        return _bernoulli_cloglog_plate_stmts(r, plan, node, pw)
    elseif r.family === BinomialProbitFam
        return _binomial_probit_plate_stmts(r, plan, node, pw)
    elseif r.family === BinomialCloglogFam
        return _binomial_cloglog_plate_stmts(r, plan, node, pw)
    elseif r.family === BetaLogitFam
        return _beta_plate_stmts(r, plan, node, pw)
    elseif r.family === CategoricalLogitFam
        return _categorical_plate_stmts(r, plan, node, pw)
    elseif r.family === OrderedLogisticFam || r.family === OrdinalFam
        return _ordinal_plate_stmts(r, plan, node, pw)
    elseif r.family === MultinomialFam
        return _multinomial_plate_stmts(r, plan, node, pw)
    elseif r.family === CategoricalFam
        return _categorical_plain_plate_stmts(r, plan, node, pw)
    elseif r.family === MvNormalCholeskyFam
        return _mvn_cholesky_plate_stmts(r, plan, node, pw)
    else
        throw(ContractValidationError(
            "[generator] response family $(r.family) has no emitter"))
    end
end

_predictor(plan::StructuralPlan, name::Symbol) =
    only(p for p in plan.predictors if p.name === name)

# The per-observation location node feeding a response's likelihood plate: a
# scan-state or per-cell (plate) latent vector fed directly (its own name —
# the layout view), or a linear predictor's `_ppl_lp_<name>` node otherwise.
function _location_node(r::LikelihoodSpec, plan::StructuralPlan)
    any(s -> s.state === r.predictor, plan.scans) && return r.predictor
    _is_plate_param(plan, r.predictor) && return r.predictor
    return _lp_name(_predictor(plan, r.predictor))
end

_dovar(i::Int) = Symbol(:_ppl_c, i)
_pw_name(label::Symbol) = Symbol(:_ppl_pw_, label)

# `pointwise = plate(inputs...) do dovars...; cell; end` + scalar sum node.
# A plate must be a whole recipe RHS (never nested under `sum`), and the
# do-block body carries a LineNumberNode or the cell types as Any. Response
# `y` and predictor `lp` are always inputs 1-2 (`_ppl_c1/_ppl_c2`).
function _plate_sum_stmts(pointwise::Symbol, node::Symbol, inputs::Vector{Any}, cell::Expr)
    dovars = [_dovar(i) for i in eachindex(inputs)]
    body = Expr(:block, LineNumberNode(0, :generator), cell)
    lambda = Expr(:(->), Expr(:tuple, dovars...), body)
    doex = Expr(:do, Expr(:call, :plate, inputs...), lambda)
    return Expr[:($pointwise = $doex), :($node::Float64 = sum($pointwise))]
end

# Thread a Symbol ref as a plate input (returning its do-var); Real
# literals inline (`as_int` for Poisson bounds — validated integer-valued).
function _thread_ref!(inputs::Vector{Any}, ref, as_int::Bool = false)
    if ref isa Symbol
        push!(inputs, ref)
        return _dovar(length(inputs))
    end
    return as_int ? Int(ref) : Float64(ref)
end

# Thread Symbol bounds (do-vars), inline Real bounds; nothing stays nothing.
function _thread_bounds!(inputs::Vector{Any}, ev::ResponseEvidence, as_int::Bool)
    lb = ev.lower === nothing ? nothing : _thread_ref!(inputs, ev.lower, as_int)
    ub = ev.upper === nothing ? nothing : _thread_ref!(inputs, ev.upper, as_int)
    return (lb, ub)
end

function _gaussian_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _location_node(r, plan)
    pre = Expr[]
    sarg = _scale_plate_arg(r, plan, pre)
    inputs = Any[y, lp]
    yv, lpv = _dovar(1), _dovar(2)
    sref = _thread_ref!(inputs, sarg)
    lb, ub = _thread_bounds!(inputs, r.evidence, false)
    base = :(normal($lpv, $sref).logpdf($yv))
    cell = _gaussian_cell(r.evidence.kind, base, yv, lb, ub, lpv, sref)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

function _gaussian_cell(kind::Symbol, base::Expr, yv::Symbol, lb, ub, lpv::Symbol, sref)
    kind === :none && return base
    nccdf(b) = :(normal($lpv, $sref).cdf($b))
    if kind === :truncated
        corr = if lb === nothing && ub === nothing
            return base
        elseif lb === nothing
            :(log($(nccdf(ub))))
        elseif ub === nothing
            :(log(1.0 - $(nccdf(lb))))
        else
            :(log($(nccdf(ub)) - $(nccdf(lb))))
        end
        return :($base - $corr)
    elseif kind === :censored
        if lb === nothing && ub === nothing
            return base
        elseif lb === nothing
            return :(ifelse($yv > $ub, log1p(-$(nccdf(ub))), $base))
        elseif ub === nothing
            return :(ifelse($yv < $lb, log($(nccdf(lb))), $base))
        else
            return :(ifelse($yv < $lb, log($(nccdf(lb))),
                ifelse($yv > $ub, log1p(-$(nccdf(ub))), $base)))
        end
    else # :interval_censored
        return :(log($(nccdf(ub)) - $(nccdf(yv))))
    end
end

function _bernoulli_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    inputs = Any[y, lp]
    yv, etav = _dovar(1), _dovar(2)
    col = plan.columns[y]
    # Validated Bool-or-0/1-Int; the endpoint takes Bool.
    yref = eltype(col) === Bool ? yv : :($yv != 0)
    cell = :(bernoulli(; logit = $etav).logpdf($yref))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return _plate_sum_stmts(pw, node, inputs, cell)
end

# Fused whole-vector Poisson-log likelihood (base case: no evidence, no
# weights). Value-identical to `Σ poisson(; log_rate=ηᵢ).logpdf(yᵢ)` for the
# contract's nonnegative-integer `y`: `Σ yᵢ·ηᵢ − Σ exp(ηᵢ) − C`, with
# `C = Σ loggamma(yᵢ+1)` baked at generation from the bound response (data-only,
# never on the gradient tape). `_ppl_yf_<label> = Float64.(y)` is a NAMED
# recipe so `bound=` folds it to a constant Float vector (no per-eval alloc)
# AND gives the fused `dot` a Float operand (Reactant `dot_general` type match).
# `sum(exp, η)` reduces without materialising the intermediate. Gradient stays
# ordinary Enzyme/Reactant AD — no analytic adjoint.
_yfloat_name(label::Symbol) = Symbol(:_ppl_yf_, label)

# Fused whole-vector Bernoulli-logit likelihood (base case). Logit-form log-mass
# `Σ yᵢ·ηᵢ − Σ log1pexp(ηᵢ)` — no data-only normalizer, a plain fused reduction.
# Value-identical to `Σ bernoulli(; logit=ηᵢ).logpdf(yᵢ)`; same folded-`_ppl_yf`
# and `sum(f, x)` treatment as the Poisson form. Gradient stays ordinary AD.
function _bernoulli_wholevec_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    yf = _yfloat_name(r.label)
    return Expr[
        :($yf = Float64.($y)),
        :($node::Float64 = dot($yf, $lp) - sum(log1pexp, $lp)),
    ]
end

function _poisson_wholevec_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    ycol = plan.columns[y]
    cterm = sum(SpecialFunctions.loggamma(Float64(v) + 1.0) for v in ycol)
    yf = _yfloat_name(r.label)
    return Expr[
        :($yf = Float64.($y)),
        :($node::Float64 = dot($yf, $lp) - sum(exp, $lp) - $cterm),
    ]
end

function _poisson_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    inputs = Any[y, lp]
    yv, etav = _dovar(1), _dovar(2)
    lb, ub = _thread_bounds!(inputs, r.evidence, true)
    base = :(poisson(; log_rate = $etav).logpdf($yv))
    cell = _poisson_cell(r.evidence.kind, base, yv, lb, ub, etav)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return _plate_sum_stmts(pw, node, inputs, cell)
end

# Lower-side cdf argument for the inclusive discrete cdf: the mass below
# lb is F(lb - 1), so truncated/censored low arms and interval cells shift
# their lower argument by one. Int literals fold; do-vars convert via Int
# (cdf takes Int, which also hardens non-Int Integer columns). The kernel's
# `observed >= 0` guard maps -1 to 0.0, so no clamp is needed.
_poisson_below(b::Int) = b - 1
_poisson_below(b::Symbol) = :(Int($b) - 1)

function _poisson_cell(kind::Symbol, base::Expr, yv::Symbol, lb, ub, etav::Symbol)
    kind === :none && return base
    pcdf(b) = :(poisson(; log_rate = $etav).cdf($b))
    if kind === :truncated
        corr = if lb === nothing && ub === nothing
            return base
        elseif lb === nothing
            :(log($(pcdf(ub))))
        elseif ub === nothing
            :(log(1.0 - $(pcdf(_poisson_below(lb)))))
        else
            :(log($(pcdf(ub)) - $(pcdf(_poisson_below(lb)))))
        end
        return :($base - $corr)
    elseif kind === :censored
        if lb === nothing && ub === nothing
            return base
        elseif lb === nothing
            return :(ifelse($yv > $ub, log1p(-$(pcdf(ub))), $base))
        elseif ub === nothing
            return :(ifelse($yv < $lb, log($(pcdf(_poisson_below(lb)))), $base))
        else
            return :(ifelse($yv < $lb, log($(pcdf(_poisson_below(lb)))),
                ifelse($yv > $ub, log1p(-$(pcdf(ub))), $base)))
        end
    else # :interval_censored
        return :(log($(pcdf(ub)) - $(pcdf(_poisson_below(yv)))))
    end
end

function _binomial_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    inputs = Any[y, lp]
    yv, etav = _dovar(1), _dovar(2)
    nref = _thread_ref!(inputs, r.trials, true)
    # All-keyword: the object constructor cannot mix positional and named
    # owner bindings (matches the `:observed/:n/:logit` HAVE ports).
    cell = :(binomial(; n = $nref, logit = $etav).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return _plate_sum_stmts(pw, node, inputs, cell)
end

# Mean/rate vectors are precomputed statements (like `_ppl_lp_*`): plate
# cells take plain do-vars — a computed `exp` constructor arg miscompiles
# the Enzyme pullback (NB2 eta-gradient, found by test).
_mu_name(label::Symbol) = Symbol(:_ppl_mu_, label)
_rate_name(label::Symbol) = Symbol(:_ppl_rate_, label)

function _nb2_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    mu = _mu_name(r.label)
    pre = Expr[:($mu = exp.($lp))]
    sarg = _scale_plate_arg(r, plan, pre)
    inputs = Any[y, mu]
    yv, muv = _dovar(1), _dovar(2)
    phiref = _thread_ref!(inputs, sarg)
    cell = :(negative_binomial2($muv, $phiref).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

function _gamma_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    pre = Expr[]
    sarg = _scale_plate_arg(r, plan, pre)
    # Surface is Distributions-SCALE `Gamma(alpha, mu/alpha)`; the kernel
    # takes rate, so the boundary inverts (same as the sampled-gamma prior).
    av = sarg isa Symbol ? sarg : Float64(sarg)
    rate = _rate_name(r.label)
    push!(pre, :($rate = $av ./ exp.($lp)))
    inputs = Any[y, rate]
    yv, ratev = _dovar(1), _dovar(2)
    aref = _thread_ref!(inputs, sarg)
    cell = :(gamma($aref, $ratev).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# The scale argument a likelihood plate threads per cell: scalar scales
# (parameter, assignment, literal, raw data column) pass through untouched;
# a predictor-fed scale binds its constrained vector once
# (`_ppl_sc_<label>`, the `_ppl_mu_`/`_ppl_rate_` precompute precedent —
# the link inverts here, never inside the cell) and the plate iterates
# the node. The logit inversion reuses the Beta plate's inlined
# `1 ./ (1 .+ exp.(-lp))` spelling (no new imports, Enzyme-safe).
_sc_name(label::Symbol) = Symbol(:_ppl_sc_, label)

function _scale_plate_arg(r::LikelihoodSpec, plan::StructuralPlan, pre::Vector{Expr})
    s = r.scale
    s isa ScalePredictorRef || return s
    pred = _predictor(plan, s.predictor)
    lp = _lp_name(pred)
    sc = _sc_name(r.label)
    rhs = if s.link === IdentityLink
        lp
    elseif s.link === LogLink
        :(exp.($lp))
    elseif s.link === LogitLink
        :(1 ./ (1 .+ exp.(-$lp)))
    else
        throw(ContractValidationError(
            "[generator] scale predictor link $(s.link) has no inversion " *
            "(admitted: identity, log, logit)"))
    end
    # The annotation is load-bearing for AD, not decoration: it proves the
    # plate input `:axis` statically, so `prepare` lowers the straight-line
    # plate form. Unannotated (metadata-`Any`) vector inputs lower with the
    # runtime `_authored_plate_is_axis` / `_plate_dependency_changed` guards,
    # whose form defeats Enzyme's static-activity analysis on some endpoint
    # bodies (NB2, found by test: silently wrong gradients). `AbstractVector`
    # is eltype-free so integer offset-only LPs still match. The LP is
    # always a vector here: codegen entry points validate first (empty
    # predictors rejected) and gate HSGP (the only termless-at-emission
    # shape), so every scale predictor contributes a vector summand.
    push!(pre, :($sc::AbstractVector = $rhs))
    return sc
end

_prob_name(label::Symbol) = Symbol(:_ppl_p_, label)
_logitp_name(label::Symbol) = Symbol(:_ppl_logitp_, label)
_shape_a_name(label::Symbol) = Symbol(:_ppl_a_, label)
_shape_b_name(label::Symbol) = Symbol(:_ppl_b_, label)

# Bernoulli probit: Phi precompute as pure Base arithmetic (Gamma-pre
# pattern). Phi is 0.5*erfc(-z/sqrt(2)), exactly the `standard_normal.cdf`
# formula — inlined rather than broadcast through the endpoint object
# because Enzyme cannot differentiate the object-broadcast (runtime
# activity on the const kernel object). Positional-p cell (primary form).
function _bernoulli_probit_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    p = _prob_name(r.label)
    pre = :($p = 0.5 .* erfc.(-$lp ./ sqrt(2)))
    inputs = Any[y, p]
    yv, pv = _dovar(1), _dovar(2)
    col = plan.columns[y]
    yref = eltype(col) === Bool ? yv : :($yv != 0)
    cell = :(bernoulli($pv).logpdf($yref))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre, _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Bernoulli cloglog: pure-arithmetic p precompute, positional-p cell.
function _bernoulli_cloglog_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    p = _prob_name(r.label)
    pre = :($p = 1 .- exp.(-exp.($lp)))
    inputs = Any[y, p]
    yv, pv = _dovar(1), _dovar(2)
    col = plan.columns[y]
    yref = eltype(col) === Bool ? yv : :($yv != 0)
    cell = :(bernoulli($pv).logpdf($yref))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre, _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Binomial probit/cloglog: precompute p, then logit(p), and reuse the
# proven logit route — the binomial kernel's p port is unverified, while
# the (:n, :logit) route is what slice-1 Binomial emits.
function _binomial_probit_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    p = _prob_name(r.label)
    logitp = _logitp_name(r.label)
    inputs = Any[y, logitp]
    yv, lpv = _dovar(1), _dovar(2)
    nref = _thread_ref!(inputs, r.trials, true)
    cell = :(binomial(; n = $nref, logit = $lpv).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[:($p = 0.5 .* erfc.(-$lp ./ sqrt(2))),
        :($logitp = log.($p) .- log1p.(-$p)),
        _plate_sum_stmts(pw, node, inputs, cell)...]
end

function _binomial_cloglog_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    p = _prob_name(r.label)
    logitp = _logitp_name(r.label)
    inputs = Any[y, logitp]
    yv, lpv = _dovar(1), _dovar(2)
    nref = _thread_ref!(inputs, r.trials, true)
    cell = :(binomial(; n = $nref, logit = $lpv).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[:($p = 1 .- exp.(-exp.($lp))),
        :($logitp = log.($p) .- log1p.(-$p)),
        _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Beta mean-concentration: mu/a/b precomputes (Gamma-pre pattern), kappa
# by name (Symbol) or inlined (literal); cell needs only (y, a, b), so
# kappa is never a plate input. Positional beta cell (primary form).
function _beta_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    k = r.scale isa Symbol ? r.scale : Float64(r.scale)
    mu = _mu_name(r.label)
    a = _shape_a_name(r.label)
    b = _shape_b_name(r.label)
    inputs = Any[y, a, b]
    yv, avv, bvv = _dovar(1), _dovar(2), _dovar(3)
    cell = :(beta($avv, $bvv).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[:($mu = 1 ./ (1 .+ exp.(-$lp))),
        :($a = $mu .* $k),
        :($b = (1 .- $mu) .* $k),
        _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Reference-coded multi-logit categorical (SB `CategoricalLogit`): K−1
# linear predictors supply the non-reference logits; class 1 is the
# implicit zero reference. The cell is the `categorical_logit_ref` math
# in scalar form (per-row logit vectors would need matrix assembly):
# the observed term selects by `y` (ifelse chain) and the normalizer is
# a max-shifted log-sum-exp over (0, etas...) — linear-size in K (a
# nested logaddexp chain would double nodes per level; the max chain is
# re-embedded per term, so K² worst case — K is small).
function _categorical_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    preds = [r.predictor; r.extra_predictors...]
    lps = [_lp_name(_predictor(plan, q)) for q in preds]
    inputs = Any[r.response, lps...]
    yv = _dovar(1)
    etas = [_dovar(i) for i in 2:length(inputs)]
    K = length(preds) + 1
    obs = :(0.0)
    for j in K:-1:2
        obs = :(ifelse($yv == $j, $(etas[j-1]), $obs))
    end
    m = :(0.0)
    for e in etas
        m = :(max($m, $e))
    end
    sumexp = :(exp(0.0 - $m))
    for e in etas
        sumexp = :($sumexp + exp($e - $m))
    end
    cell = :($obs - ($m + log($sumexp)))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return _plate_sum_stmts(pw, node, inputs, cell)
end

# Stable log-logistic (Stan `log_inv_logit`): `z ≥ 0` takes
# `-log1p(exp(-z))`, `z < 0` takes `z - log1p(exp(z))` — no overflow
# either tail. Base ops only (transparent to the reverse pass).
_log_inv_logit(z) = :(ifelse($z >= 0.0, -log1p(exp(-$z)), $z - log1p(exp($z))))

# Ordinal link log-CDF / log-CCDF over a scalar z (SB `brm_ordinal_logcdf`
# / `brm_ordinal_logccdf`). Probit uses the slice-1 erfc treatment
# (`0.5*erfc(∓z/√2)` under a log — accurate in moderate ranges; extreme
# tails round, the accepted slice-1 probit caveat).
function _ordinal_logF(link::LinkFunction, z)
    link === LogitLink && return _log_inv_logit(z)
    link === ProbitLink && return :(log(0.5 * erfc(-$z / sqrt(2))))
    return :(log(-expm1(-exp($z))))
end
function _ordinal_logCC(link::LinkFunction, z)
    link === LogitLink && return _log_inv_logit(:(-$z))
    link === ProbitLink && return :(log(0.5 * erfc($z / sqrt(2))))
    return :(-exp($z))
end

# Stable log-difference of log-probs (a ≥ b): `a + log1p(-exp(b - a))`.
_log_diff_exp(a, b) = :($a + log1p(-exp($b - $a)))

# Per-threshold effect column name (stage j of response `label`).
_eff_name(label::Symbol, j::Int) = Symbol(:_ppl_eff_, label, :_, j)

# Modeled-scale precompute name (response `label`).
_disc_name(label::Symbol) = Symbol(:_ppl_disc_, label)

# Ordinal latent scale as a plate do-var: absent inlines 1.0 (the
# 3-positional form — byte-identical emission), a literal inlines, a data
# column threads raw, and a log-link predictor threads its `exp`
# precompute (structural positivity — the Poisson `exp.(lp)` precedent).
function _ordinal_scale_ref!(inputs::Vector{Any}, prests::Vector{Expr},
        r::LikelihoodSpec, plan::StructuralPlan)
    d = r.discrimination
    d === nothing && return 1.0
    if d isa Symbol && any(p -> p.name === d, plan.predictors)
        push!(inputs, _disc_pre!(prests, r, plan, d))
        return _dovar(length(inputs))
    end
    return _thread_ref!(inputs, d)
end

# Modeled-scale column: `exp` over the scale predictor's lp node (the
# predictor statements run before the likelihood, so the node exists;
# validation proved the link is LogLink). Explicit dotted form, evaluated
# once and threaded like the stage-effect columns.
function _disc_pre!(prests::Vector{Expr}, r::LikelihoodSpec,
        plan::StructuralPlan, sname::Symbol)
    lp = _lp_name(_predictor(plan, sname))
    name = _disc_name(r.label)
    push!(prests, :($name = exp.($lp)))
    return name
end

# Ordered response plate (OrderedLogistic + Ordinal, one uniform path:
# OrderedLogistic is cumulative-logit with d = 1 and no threshold
# effects). Threshold scalars thread as broadcast plate inputs; the cell
# is fully static in structure/link/K. K=1 lowers to a zero cell (SB's
# zero-information likelihood).
function _ordinal_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    K = r.n_levels
    K === nothing && throw(ContractValidationError(
        "[generator] ordered response $(r.label) has unresolved n_levels " *
        "(bind_data infers it)"))
    inputs = Any[y, lp]
    yv, etav = _dovar(1), _dovar(2)
    prests = Expr[]
    dref = _ordinal_scale_ref!(inputs, prests, r, plan)
    tnames = [_vector_elt_name(r.thresholds, i) for i in 1:K-1]
    trefs = [_thread_ref!(inputs, t) for t in tnames]
    erefs = Any[]
    if !isempty(r.threshold_columns)
        for j in 1:K-1
            push!(inputs, _eff_pre!(prests, r, plan, j))
            push!(erefs, _dovar(length(inputs)))
        end
    end
    structure = r.family === OrderedLogisticFam ? :cumulative : r.ordinal_structure
    cell = structure === :cumulative ?
        _ordinal_cumulative_cell(r.link, K, yv, etav, dref, trefs) :
        _ordinal_stopping_cell(r.link, K, yv, etav, dref, trefs, erefs)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[prests..., _plate_sum_stmts(pw, node, inputs, cell)...]
end

# Stage-j threshold-effect column (StoppingRatio per_threshold): the
# broadcast sum `Σ_c col_c .* β[j,c]` over the raw design columns and
# the packed coefficient scalars (stage-major: stage j occupies
# `(j-1)*p+1 .. j*p`). Emits the precompute statement and returns its
# name for threading. Data-only columns fold under `bound=`; the
# broadcast itself is the Gamma-pre pattern.
function _eff_pre!(prests::Vector{Expr}, r::LikelihoodSpec, plan::StructuralPlan, j::Int)
    p = length(r.threshold_columns)
    coefs = r.threshold_coefs
    terms = Any[]
    for (c, col) in enumerate(r.threshold_columns)
        b = _vector_elt_name(coefs, (j - 1) * p + c)
        push!(terms, :($col .* $b))
    end
    rhs = foldl((a, b) -> :($a .+ $b), terms)
    name = _eff_name(r.label, j)
    push!(prests, :($name = $rhs))
    return name
end

# Cumulative cell: `y == 1` takes logF, `y == K` takes logCC, interior
# levels take the stable log-difference. Thresholds are ordered
# (constrained), so hi ≥ lo and the difference is well-defined.
function _ordinal_cumulative_cell(link::LinkFunction, K::Int, yv::Symbol,
        etav::Symbol, dref, trefs::Vector)
    # K=1 is SB's zero-information likelihood; the cell stays a real
    # Expr over the (integer) response do-var (`:(0.0)` would quote to
    # a bare Float64, which the plate builder does not take).
    K == 1 && return :(0.0 * $yv)
    first = _ordinal_logF(link, :($dref * ($(trefs[1]) - $etav)))
    last = _ordinal_logCC(link, :($dref * ($(trefs[K-1]) - $etav)))
    K == 2 && return :(ifelse($yv == 1, $first, $last))
    mid = :(0.0)
    for y in K-1:-1:2
        hi = _ordinal_logF(link, :($dref * ($(trefs[y]) - $etav)))
        lo = _ordinal_logF(link, :($dref * ($(trefs[y-1]) - $etav)))
        mid = :(ifelse($yv == $y, $(_log_diff_exp(hi, lo)), $mid))
    end
    return :(ifelse($yv == 1, $first, ifelse($yv == $K, $last, $mid)))
end

# Stopping-ratio cell (SB `brm_ordinal` structure 2): stage j contributes
# logCC below `y`, logF at `y`, nothing above. `erefs` holds the
# per-row effect do-vars (empty without per_threshold — effects are 0).
function _ordinal_stopping_cell(link::LinkFunction, K::Int, yv::Symbol,
        etav::Symbol, dref, trefs::Vector, erefs::Vector)
    # K=1 runs zero stages (SB's zero-information likelihood); see the
    # cumulative K=1 note on quoting a real Expr.
    K == 1 && return :(0.0 * $yv)
    terms = Any[]
    for j in 1:K-1
        eff = isempty(erefs) ? 0.0 : erefs[j]
        z = :($dref * ($(trefs[j]) - $etav - $eff))
        push!(terms, :(ifelse($yv > $j, $(_ordinal_logCC(link, z)),
            ifelse($yv == $j, $(_ordinal_logF(link, z)), 0.0))))
    end
    return foldl((a, b) -> :($a + $b), terms)
end

# Shared-simplex multinomial (SB `brm_multinomial` vector[K] method): the
# count matrix crosses as K raw columns; the simplex parameter threads
# as K broadcast scalars. The cell is Stan's `multinomial_lpmf` in
# scalar form — `lgamma(N+1) − Σ lgamma(c+1) + Σ c*log(p)` — with the
# `0*log(0) = 0` convention guarded per term (Stan treats a zero count
# at a zero probability as 0, not NaN). A literal N folds its
# `lgamma(N+1)` host-side (exact same value, computed once).
function _multinomial_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    counts = [r.response; r.count_columns...]
    K = length(counts)
    inputs = Any[counts...]
    cvs = [_dovar(i) for i in 1:K]
    nref = _thread_ref!(inputs, r.trials, true)
    prefs = [_thread_ref!(inputs, _vector_elt_name(r.predictor, i)) for i in 1:K]
    lfact = r.trials isa Int ? loggamma(r.trials + 1.0) : :(loggamma($nref + 1.0))
    cell = :($lfact)
    for (cv, pv) in zip(cvs, prefs)
        cell = :($cell - loggamma($cv + 1.0) +
            ifelse($cv == 0, 0.0, $cv * log($pv)))
    end
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return _plate_sum_stmts(pw, node, inputs, cell)
end

# Plain categorical over shared-simplex probabilities (Stan
# `categorical_lpmf`): the cell selects `log(p[y])` by the observed
# level (ifelse chain). K=1 lowers to `log(1.0) = 0` uniformly.
function _categorical_plain_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    K = r.n_levels
    K === nothing && throw(ContractValidationError(
        "[generator] categorical response $(r.label) has unresolved n_levels " *
        "(bind_data infers it)"))
    inputs = Any[r.response]
    yv = _dovar(1)
    prefs = [_thread_ref!(inputs, _vector_elt_name(r.predictor, i)) for i in 1:K]
    cell = :(log($(prefs[1])))
    for j in 2:K
        cell = :(ifelse($yv == $j, log($(prefs[j])), $cell))
    end
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return _plate_sum_stmts(pw, node, inputs, cell)
end

# Joint correlated-outcomes likelihood (SB's per-row
# `multi_normal_cholesky(mean_row, L)` with
# `L = diag_pre_multiply(scales, L_corr)`): one plate over the K outcome
# columns + K mean LPs sums the per-row density. The row cell is scalar
# forward substitution over threaded do-vars (the multinomial-cell shape —
# no matrix, no triangular solve in the cell, so the tensorized plate
# traces; a core-`mvnormal` splice was probed and fails Reactant primal
# inside the plate — the in-cell `\` hits scalar indexing in Reactant's
# `generic_trimatdiv!`, a primal gap distinct from the §7f gradient gap).
# The L entries materialize as `_ppl_mvn_Le_` scalars (row-scaled layout
# temps) and thread as shared scalar plate inputs.
function _mvn_cholesky_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan,
        node::Symbol, pw::Symbol)
    outcomes = [r.response; r.extra_responses...]
    preds = [r.predictor; r.extra_predictors...]
    K = length(outcomes)
    si = findfirst(p -> p.name === r.factor_scales, plan.vector_parameters)
    ci = findfirst(p -> p.name === r.factor_corr, plan.vector_parameters)
    (si === nothing || ci === nothing) && throw(ContractValidationError(
        "[generator] joint response $(r.label) factor pieces unresolved " *
        "(validate_plan links them)"))
    sc, cr = plan.vector_parameters[si], plan.vector_parameters[ci]
    (sc.size == K && cr.size == K) || throw(ContractValidationError(
        "[generator] joint response $(r.label) factor sizes disagree " *
        "with the $K outcomes (validate_plan checks this)"))
    stmts = Expr[]
    # L[i,j] = scales[i] * L_corr[i,j] (j ≤ i), one scalar per lower entry.
    for i in 1:K, j in 1:i
        push!(stmts, :($(_mvn_L_entry(r.label, i, j))::Float64 =
            $(_vector_elt_name(sc.name, i)) * $(_rl_name(cr.name, i, j))))
    end
    lps = [_lp_name(_predictor(plan, q)) for q in preds]
    inputs = Any[outcomes...; lps...]
    yvs = [_dovar(i) for i in 1:K]
    mvs = [_dovar(K + i) for i in 1:K]
    Ld = Dict{Tuple{Int,Int},Symbol}()
    for i in 1:K, j in 1:i
        Ld[(i, j)] = _thread_ref!(inputs, _mvn_L_entry(r.label, i, j))
    end
    cell = _mvn_row_cell(K, yvs, mvs, Ld)
    append!(stmts, _plate_sum_stmts(pw, node, inputs, cell))
    return stmts
end

# In-graph L-entry name for a joint response (`_ppl_mvn_Le_<label>_<i>_<j>`,
# j ≤ i). All `_ppl_`-hygienic.
_mvn_L_entry(label::Symbol, i::Int, j::Int) =
    Symbol(:_ppl_mvn_Le_, label, :_, i, :_, j)

# One joint row's log-density as a single scalar expression: residuals,
# forward substitution (`z[i] = (d[i] − Σ L[i,j]·z[j]) / L[i,i]`, inlined —
# K is small), quadratic form, and the row constant. Pointwise-pure: each
# lane evaluates its row's full `multi_normal_cholesky` log-density.
function _mvn_row_cell(K::Int, yvs::Vector{Symbol}, mvs::Vector{Symbol},
        Ld::Dict{Tuple{Int,Int},Symbol})
    ds = [:( $(yvs[i]) - $(mvs[i]) ) for i in 1:K]
    zs = Any[]
    for i in 1:K
        num = ds[i]
        for j in 1:i-1
            num = :( $num - $(Ld[(i, j)]) * $(zs[j]) )
        end
        push!(zs, :( ($num) / $(Ld[(i, i)]) ))
    end
    quad = foldl((a, z) -> :( $a + $z * $z ), zs; init = :(0.0))
    logdet = foldl((a, i) -> :( $a + log($(Ld[(i, i)])) ), 1:K; init = :(0.0))
    row_const = -0.5 * K * log(2 * pi)
    return :( $row_const - $logdet - 0.5 * $quad )
end

# R2D2 prior bindings: the location vector stays a literal (SB
# `beta_loc`), while the scale vector unrolls per design column —
# literal fallbacks at share 0, `sqrt(phi[k]*R2*tau^2/varx[j])`
# expressions over the in-graph simplex elements + scalars otherwise
# (SB `brm_r2d2_scale`, unrolled — the share map is static data, so no
# data-dependent branching enters the graph). All radicands are
# positive by construction (simplex/logistic/exp transforms +
# validated varx). The consuming plate-sum shape is unchanged.
function _r2d2_prior_stmts(rp::R2D2Prior, shape::DesignShape,
        columns::Dict{Symbol,AbstractVector}, mut::Symbol, sdt::Symbol)
    share, fallback, loc, varx =
        r2d2_column_scales(shape, columns, rp.overrides)
    elts = Any[]
    for j in eachindex(share)
        if share[j] == 0
            push!(elts, fallback[j])
            continue
        end
        ph = _vector_elt_name(rp.phi, share[j])
        r2 = rp.r2
        t2 = rp.tau isa Symbol ? :($(rp.tau) * $(rp.tau)) :
            Float64(rp.tau)^2
        push!(elts, :(sqrt($ph * $r2 * $t2 / $(varx[j]))))
    end
    return Any[:($mut = Float64[$(loc...)]), :($sdt = Float64[$(elts...)])]
end

_r2d2_for(plan::StructuralPlan, pred::Symbol) = begin
    for rp in plan.r2d2_priors
        rp.predictor === pred && return rp
    end
    return nothing
end

function _prior_statements(plan::StructuralPlan, layout::LayoutTable)
    stmts = Expr[]
    terms = Any[]
    for pred in plan.predictors
        shape = design_shape(pred, plan.columns; levelmaps = plan.levelmaps)
        shape.width == 0 && continue
        node = Symbol(:_ppl_prior_, pred.name)
        pw = Symbol(:_ppl_pw_prior_, pred.name)
        mut = Symbol(:_ppl_prmu_, pred.name)
        sdt = Symbol(:_ppl_prsd_, pred.name)
        rp = _r2d2_for(plan, pred.name)
        if rp === nothing
            loc, sca = coefficient_priors(shape, plan.population_priors)
            push!(stmts, :($mut = Float64[$(loc...)]))
            push!(stmts, :($sdt = Float64[$(sca...)]))
        else
            push!(stmts, _r2d2_prior_stmts(rp, shape, plan.columns, mut, sdt)...)
        end
        coef = block_name(pred.name)
        cv, mv, sv = _dovar(1), _dovar(2), _dovar(3)
        cell = :(normal($mv, $sv).logpdf($cv))
        append!(stmts, _plate_sum_stmts(pw, node, Any[coef, mut, sdt], cell))
        push!(terms, node)
    end
    for p in plan.parameters
        node = Symbol(:_ppl_prior_, p.name)
        push!(stmts, :($node::Float64 = $(_sampled_prior_expr(p))))
        push!(terms, node)
    end
    # Leveled vector latents (cutpoints/thresholds/simplexes/coefficient
    # packs, joint-factor scales/Cholesky): unrolled scalar sums over the
    # `_ppl_v_` layout elements (bound plans carry concrete sizes). Empty
    # packs contribute 0.0.
    for p in plan.vector_parameters
        node = Symbol(:_ppl_prior_, p.name)
        push!(stmts, :($node::Float64 = $(_vector_prior_expr(p))))
        push!(terms, node)
    end
    # Per-cell latent (plate) parameters: one plate over the block, summing the
    # shared-prior log-density across cells (the same plate-sum shape as a
    # population-prior coefficient block, generalized to any standard family).
    # Every value the cell reads is threaded as a plate PORT (the latent vector
    # plus each scalar prior arg) — captured free names are rejected by the
    # `@kernel` plate expander, exactly as the Gaussian-likelihood scale is
    # threaded.
    for p in plan.plate_parameters
        _vector_prior_stmts!(stmts, terms, p.name, p.family, p.args,
            p.support_override)
    end
    # Spline coefficient vectors: the same plate-prior shape (broadcast the
    # shared prior over cells). `b_fixed` is flat — a 0.0 node, mirroring a
    # scalar flat parameter (never a plate: a vacuous cell would leave the
    # do-var unread).
    for v in plan.spline_vectors
        if v.family === :flat
            node = Symbol(:_ppl_prior_, v.name)
            push!(stmts, :($node::Float64 = 0.0))
            push!(terms, node)
            continue
        end
        _vector_prior_stmts!(stmts, terms, v.name, v.family, v.args,
            v.support_override)
    end
    # Varying draws. K=1: the scalar scale prior plus the
    # standardized `xi` plate (shared vector-prior helper). Correlated:
    # the LKJ node plus the `tau`/`z_flat` plates (shared vector-prior
    # helper). `tau` emits WITHOUT the thin layer's `+log(2)` half
    # renormalizer in both: SB's `std_normal(; lower=0)` is Stan
    # lower-bound kernel semantics (exp Jacobian only, no truncation
    # normalizer). User-facing `HalfNormal` priors keep the
    # proper-half convention; draws-internal `tau` follows SB.
    for d in plan.varying_draws
        if d.kind === :correlated
            L, tau, z = _varying_corr_names(d)
            lnode = Symbol(:_ppl_prior_, L)
            push!(stmts, :($lnode::Float64 = $(_lkj_prior_expr(d))))
            push!(terms, lnode)
            _vector_prior_stmts!(stmts, terms, tau, :normal,
                (arg1 = 0, arg2 = 1), nothing)
            _vector_prior_stmts!(stmts, terms, z, :normal,
                (arg1 = 0, arg2 = 1), nothing)
            continue
        end
        scale, xi = _varying_k1_names(d)
        snode = Symbol(:_ppl_prior_, scale)
        scell = _family_logpdf_expr(:normal, Any[0, 1], scale)
        push!(stmts, :($snode::Float64 = $scell))
        push!(terms, snode)
        _vector_prior_stmts!(stmts, terms, xi, :normal, (arg1 = 0, arg2 = 1),
            nothing)
    end
    # HSGP bases (SB `_sb_hsgp`/`_sb_hsgp_aniso`): the length scales and
    # marginal scale as scalar `lognormal(0, 1)` nodes plus the
    # standardized `beta_raw` plate (shared vector-prior helper). The
    # floored rhos emit WITHOUT a truncation normalizer: SB's
    # `lognormal(0,1; lower=rho_lower)` is Stan lower-bound kernel
    # semantics (offset-exp Jacobian only — the varying-`tau` precedent).
    for hb in plan.hsgp_bases
        names = _hsgp_names(hb)
        for rho in names.rhos
            node = Symbol(:_ppl_prior_, rho)
            cell = _family_logpdf_expr(:lognormal, Any[0, 1], rho)
            push!(stmts, :($node::Float64 = $cell))
            push!(terms, node)
        end
        snode = Symbol(:_ppl_prior_, names.sigma)
        scell = _family_logpdf_expr(:lognormal, Any[0, 1], names.sigma)
        push!(stmts, :($snode::Float64 = $scell))
        push!(terms, snode)
        _vector_prior_stmts!(stmts, terms, names.beta, :normal,
            (arg1 = 0, arg2 = 1), nothing)
    end
    # Sequential-recurrence (scan) latents: setup + recurrence density.
    scanstmts, scannodes = _scan_prior_statements(plan, layout)
    append!(stmts, scanstmts)
    append!(terms, scannodes)
    joint = foldl((a, b) -> :($a + $b), terms; init = :(0.0))
    push!(stmts, :(prior::Float64 = $joint))
    return stmts
end

# Shared LKJ-prior node body (Stan `lkj_corr_cholesky_lpdf` op order,
# preserved verbatim from the host `lkj_corr_cholesky_logpdf`: constant
# literal first, then per-diagonal terms in row order — `(K-i)*log(L[i,i])`
# at `eta == 1.0`, else `a*log + b*log` with emission-time coefficients).
# Reads the named `_ppl_rl_` diagonal scalars — no matrix materializes, and
# the scalar-only form keeps the native Enzyme reverse pass on the same
# straight-line shape as every other prior (the core `lkj_corr_cholesky`
# object's `(1:K)` range broadcasts fail Enzyme reverse, so the splice
# stays out of generated code). K=1 is the `0.0` literal (Stan's K=1 LKJ
# term is ±0.0 — no diagonal, zero constant).
function _lkj_prior_terms(L::Symbol, K::Int, eta::Float64)
    K == 1 && return :(0.0)
    terms = Any[lkj_logconst(K, eta)]
    if eta == 1.0
        for i in 2:K
            push!(terms, :($(K - i) * log($(_rl_name(L, i, i)))))
        end
    else
        bcoef = 2 * eta - 2
        for i in 2:K
            k = i - 2
            push!(terms, :($(K - 1 - k - 1) * log($(_rl_name(L, i, i))) +
                $bcoef * log($(_rl_name(L, i, i)))))
        end
    end
    return foldl((a, c) -> :($a + $c), terms)
end

# One correlated draws block's LKJ prior node (names/sizes from the draws).
function _lkj_prior_expr(d::VaryingDraws)
    return _lkj_prior_terms(_varying_corr_names(d)[1], length(d.margins),
        d.lkj_eta)
end

# One plate over a latent VECTOR (a plate parameter or a spline vector),
# summing the shared-prior log-density across cells. Every value the cell
# reads is threaded as a plate PORT (the vector plus each scalar prior
# arg) — captured free names are rejected by the `@kernel` plate
# expander, exactly as the Gaussian-likelihood scale is threaded.
function _vector_prior_stmts!(stmts::Vector{Expr}, terms::Vector{Any},
        name::Symbol, family::Symbol, args::NamedTuple,
        support::SupportOverride)
    node = Symbol(:_ppl_prior_, name)
    pw = Symbol(:_ppl_pw_prior_, name)
    inputs = Any[name]
    tv = _dovar(1)
    argvals = Any[_thread_ref!(inputs, v) for v in values(args)]
    cell = _family_logpdf_expr(family, argvals, tv)
    corr = _support_correction(support, argvals)
    corr === nothing || (cell = :($cell + $corr))
    append!(stmts, _plate_sum_stmts(pw, node, inputs, cell))
    push!(terms, node)
    return nothing
end

# Scalar prior log-density per family via distribution-kernel endpoints
# (Distributions.jl semantics). Args are literals (inlined) or
# parameter/assignment refs (body locals, fine in scalar position).
# Half-Normal/half-Cauchy (the only overrides) renormalize by exactly
# +log(2) (symmetry at 0). `gamma` takes rate, so the contract's scale
# inverts.
# Shared `<family>(remapped args…).logpdf(x)` for a variate expression `x` (a
# scalar parameter name, a plate do-var, or a scan setup/recurrence read). Args
# are literals (inlined) or parameter/assignment/threaded refs. Shared by scalar
# priors, per-cell plate priors, and the scan density. `gamma` takes rate, so
# the contract's scale inverts.
function _family_logpdf_expr(family::Symbol, a, x)
    if family === :normal
        mu, s = a
        :(normal($mu, $s).logpdf($x))
    elseif family === :cauchy
        mu, s = a
        :(cauchy($mu, $s).logpdf($x))
    elseif family === :exponential
        (th,) = a
        :(exponential($th).logpdf($x))
    elseif family === :gamma
        al, th = a
        :(gamma($al, 1 / $th).logpdf($x))
    elseif family === :lognormal
        mu, s = a
        :(lognormal($mu, $s).logpdf($x))
    elseif family === :beta
        al, be = a
        :(beta($al, $be).logpdf($x))
    elseif family === :inverse_gamma
        al, th = a
        :(inverse_gamma($al, $th).logpdf($x))
    else # :flat
        :(0.0)
    end
end

# The additive support-override correction for a prior log-density (or `nothing`
# for no override): `:positive` (half-Normal/half-Cauchy) renormalizes by exactly
# +log(2) (symmetry at literal 0); `(:interval, lo, hi)` (a truncated Normal)
# renormalizes by -log(cdf(hi) - cdf(lo)) at any location, where `argvals` are the
# family's (mu, s) argument expressions (literals/refs for a scalar prior, or
# per-cell do-vars for a plate prior — the CDF endpoints thread identically).
function _support_correction(ov::SupportOverride, argvals)
    ov === nothing && return nothing
    if ov isa Tuple  # (:interval, lo, hi) — truncated Normal
        lo, hi = ov[2], ov[3]
        mu, s = argvals[1], argvals[2]
        return :(-log(normal($mu, $s).cdf($hi) - normal($mu, $s).cdf($lo)))
    end
    return :(log(2))  # :positive half
end

# Scalar prior log-density per family via distribution-kernel endpoints
# (Distributions.jl semantics). The support override adds the +log(2) half or
# the -log(cdf(hi)-cdf(lo)) truncated-interval renormalization (`_support_correction`).
function _sampled_prior_expr(p::SampledParameter)
    argvals = [v for v in values(p.args)]
    base = _family_logpdf_expr(p.family, argvals, p.name)
    corr = _support_correction(p.support_override, argvals)
    corr === nothing && return base
    return :($base + $corr)
end

# Vector-latent prior over the `_ppl_v_` layout elements: elementwise
# Normal for threshold/coefficient packs (Stan `ordered`/`vector`
# semantics — no factorial normalizer, matching `_BRMThresholdPrior`),
# Dirichlet for simplexes (Stan `dirichlet_lpdf`: the log-multivariate-Beta
# normalizer folds host-side — data-only — plus Σ (α−1)·log(s)),
# elementwise Exponential for joint-factor scales (a literal scale inlines;
# a sampled hyperparameter resolves as a body local, the scalar-prior
# shape), and the shared LKJ node for joint Cholesky factors.
function _vector_prior_expr(p::VectorParameter)
    m = p.size
    m === nothing && throw(ContractValidationError(
        "[generator] vector parameter $(p.name) has unresolved size " *
        "(bind_data infers it)"))
    if p.family === :cholesky_corr_lkj
        return _lkj_prior_terms(p.name, m, Float64(p.args.arg1))
    end
    elts = [_vector_elt_name(p.name, i) for i in 1:m]
    if p.family === :simplex_dirichlet
        alpha = Vector{Float64}(p.args.arg1)
        normalizer = loggamma(sum(alpha)) - sum(loggamma, alpha)
        terms = Any[normalizer]
        for (a, s) in zip(alpha, elts)
            push!(terms, :($(a - 1.0) * log($s)))
        end
        return foldl((x, y) -> :($x + $y), terms)
    end
    if p.family === :positive_exponential
        th = p.args.arg1
        theta = th isa Symbol ? th : Float64(th)
        terms = Any[:(exponential($theta).logpdf($t)) for t in elts]
        return foldl((x, y) -> :($x + $y), terms; init = :(0.0))
    end
    if p.family === :ordered_normal || p.family === :vector_normal
        mu, s = Float64(p.args.arg1), Float64(p.args.arg2)
        terms = Any[:(normal($mu, $s).logpdf($t)) for t in elts]
        return foldl((x, y) -> :($x + $y), terms; init = :(0.0))
    end
    throw(ContractValidationError(
        "[generator] vector parameter $(p.name) family $(p.family) " *
        "has no scalar prior form"))
end

# --- Sequential-recurrence (scan) density (slice 1: CENTERED) ---

# Distinct backward lags `state[loopvar-j]` read across a step's dist args.
function _scan_step_lags(step::ScanStep, state::Symbol, loopvar::Symbol)
    lags = Set{Int}()
    walk(ex) = begin
        ex isa Expr || return
        if ex.head === :ref && length(ex.args) == 2 && ex.args[1] === state
            idx = ex.args[2]
            if idx isa Expr && idx.head === :call && length(idx.args) == 3 &&
               idx.args[1] === :- && idx.args[2] === loopvar && idx.args[3] isa Int
                push!(lags, idx.args[3])
                return
            end
        end
        foreach(walk, ex.args)
    end
    step.args === nothing || foreach(walk, step.args)
    return sort!(collect(lags))
end

# Replace each `state[loopvar-j]` lag read with its aligned-slice do-var.
function _subst_scan_lags(ex, state::Symbol, loopvar::Symbol, dovar::Dict{Int,Symbol})
    ex isa Expr || return ex
    if ex.head === :ref && length(ex.args) == 2 && ex.args[1] === state
        idx = ex.args[2]
        if idx isa Expr && idx.head === :call && length(idx.args) == 3 &&
           idx.args[1] === :- && idx.args[2] === loopvar && idx.args[3] isa Int
            return dovar[idx.args[3]]
        end
    end
    return Expr(ex.head,
        (_subst_scan_lags(a, state, loopvar, dovar) for a in ex.args)...)
end

# Value symbols in a (lag-substituted) recurrence arg that must be threaded as
# plate inputs: captured scalars (params/assignments). Excludes call heads and
# the already-substituted lag do-vars (`_ppl_c…`).
function _scan_cell_caps!(caps::Vector{Symbol}, ex)
    if ex isa Symbol
        (startswith(string(ex), "_ppl_c") || ex in caps) || push!(caps, ex)
        return nothing
    end
    ex isa Expr || return nothing
    args = ex.head === :call ? ex.args[2:end] : ex.args
    for a in args
        _scan_cell_caps!(caps, a)
    end
    return nothing
end

# Replace symbols per `map` (leaves call heads alone — they never appear in the
# capture map).
function _subst_syms(ex, map::Dict{Symbol,Symbol})
    ex isa Symbol && return get(map, ex, ex)
    ex isa Expr || return ex
    return Expr(ex.head, (_subst_syms(a, map) for a in ex.args)...)
end

# --- Non-centered scan reconstruction (AR(1) slice) ---

# v1 admission for a non-centered scan: exactly one `Normal(0, 1)` seed (the
# first innovation), exactly two steps (one non-indexed `Normal(0, 1)`
# innovation sample + the indexed deterministic carry write), lag 1.
# Returns (innovation_symbol, carry_write_expr); anything else throws
# `ContractValidationError` naming the admitted shape.
function _scan_noncentered_form(s::ScanSpec)
    where = "[generator] scan $(s.state)"
    _scan_stdnormal_seed(s) || throw(ContractValidationError(
        "$where: non-centered v1 needs exactly one `Normal(0, 1)` seed " *
        "(the first innovation)"))
    length(s.step) == 2 || throw(ContractValidationError(
        "$where: non-centered v1 needs exactly one innovation sample + " *
        "one carry write, got $(length(s.step)) steps"))
    first, second = s.step[1], s.step[2]
    (first.kind === :sample && !first.indexed) || throw(ContractValidationError(
        "$where: the innovation sample must precede the carry write " *
        "(`eps ~ Normal(0, 1)` then `$(s.state)[$(s.loopvar)] = ...`)"))
    _is_unit_normal(first.family, first.args) || throw(ContractValidationError(
        "$where: the innovation `$(first.target)` must be `Normal(0, 1)` in v1"))
    (second.kind === :assign && second.indexed) || throw(ContractValidationError(
        "$where: the second step must be the deterministic carry write " *
        "`$(s.state)[$(s.loopvar)] = ...`"))
    s.maxlag == 1 || throw(ContractValidationError(
        "$where: non-centered AR(p > 1) needs a tuple carry (planned)"))
    return first.target, second.expr
end

_scan_stdnormal_seed(s::ScanSpec) =
    length(s.setup) == 1 && _is_unit_normal(s.setup[1].family, s.setup[1].args)

_is_unit_normal(family, args) =
    family === :normal && length(args) == 2 &&
    args[1] isa Real && args[1] == 0 && args[2] isa Real && args[2] == 1

# Translate a v1 carry-write RHS into the `scan(...)` step body: the lag-1
# carried read becomes the carry do-var, the bare innovation becomes the
# step-element do-var, and every other leaf must be a scalar parameter or
# assignment name (threaded as a `Ref`). Returns (translated_expr,
# sorted_refs). The contract never inspects step RHSs, so these leaves are
# the only screen for hand-built plans — everything else fails closed.
function _scan_step_body(s::ScanSpec, eps::Symbol, ex, scalars)
    refs = Set{Symbol}()
    body = _scan_translate_step(ex, s, eps, :_ppl_carry, :_ppl_elem, refs,
        scalars)
    return body, sort!(collect(refs))
end

function _scan_translate_step(ex, s::ScanSpec, eps::Symbol, carry::Symbol,
        elem::Symbol, refs::Set{Symbol}, scalars)
    if ex isa Symbol
        ex === eps && return elem
        ex === s.loopvar && throw(ContractValidationError(
            "[generator] scan $(s.state): the carry write uses the loop index " *
            "`$(s.loopvar)` directly — not supported in v1"))
        ex === s.state && throw(ContractValidationError(
            "[generator] scan $(s.state): bare read of the carried state — " *
            "read the backward lag `$(s.state)[$(s.loopvar) - 1]`"))
        ex in scalars || throw(ContractValidationError(
            "[generator] scan $(s.state): step leaf `$(ex)` is not a scalar " *
            "parameter or assignment (data-varying steps are planned)"))
        push!(refs, ex)
        return ex
    end
    ex isa Expr || return ex
    if ex.head === :ref && length(ex.args) == 2 && ex.args[1] === s.state
        _scan_assert_lag1(ex.args[2], s)
        return carry
    end
    if ex.head === :ref
        throw(ContractValidationError(
            "[generator] scan $(s.state): indexed read `$(ex.args[1])[...]` " *
            "in the carry write — v1 threads only the carried lag-1"))
    end
    args = ex.head === :call ? ex.args[2:end] : ex.args
    newargs = [_scan_translate_step(a, s, eps, carry, elem, refs, scalars)
        for a in args]
    return ex.head === :call ?
        Expr(:call, ex.args[1], newargs...) : Expr(ex.head, newargs...)
end

function _scan_assert_lag1(idx, s::ScanSpec)
    idx isa Expr && idx.head === :call && length(idx.args) == 3 &&
        idx.args[1] === :- && idx.args[2] === s.loopvar &&
        idx.args[3] isa Int && idx.args[3] == 1 && return nothing
    throw(ContractValidationError(
        "[generator] scan $(s.state): the carry write must read exactly " *
        "`$(s.state)[$(s.loopvar) - 1]` in v1"))
end

# Reconstruction statements for every non-centered scan, in plan order:
# the seed innovation heads the state, the tail folds over the rest —
#   `rest = scan(view(z, 2:T), Ref(params)...; init = z[1]) do ... end`
#   `state = vcat(z[1], rest)`
# The split is load-bearing: a uniform scan from a zero carry would scale
# the seed (`h[1] = s*z[1]` under `phi*carry + s*eps`), contradicting the
# declared `Normal(0, 1)` seed — while the split reproduces SB's
# `ar1_recurse` (`u[1] = eps[1]`) under ANY step formula. Runs before
# predictors/likelihood (both may read the state); the innovation prior
# stays in `_scan_prior_statements` (order-free).
function _scan_reconstruction_statements(plan::StructuralPlan,
        layout::LayoutTable)
    stmts = Expr[]
    scalars = _union_names(plan)
    for s in plan.scans
        _is_noncentered_scan(s) || continue
        eps, expr = _scan_noncentered_form(s)
        zname = _scan_innovation_name(s)
        entry = only(e for e in layout.entries
            if e.kind === :scan && e.name === zname)
        T = entry.size
        body, refs = _scan_step_body(s, eps, expr, scalars)
        next = :_ppl_next
        lambda = Expr(:->, Expr(:tuple, :_ppl_carry, :_ppl_elem, refs...),
            Expr(:block, :($next = $body), :(($next, $next))))
        kw = Expr(:parameters, Expr(:kw, :init, :($(zname)[1])))
        call = Expr(:call, :scan, kw, :(view($zname, 2:$T)),
            (:(Ref($r)) for r in refs)...)
        rest = Symbol(:_ppl_scan_rest_, s.state)
        push!(stmts, :($rest = $(Expr(:do, call, lambda))))
        push!(stmts, :($(s.state) = vcat($(zname)[1], $rest)))
    end
    return stmts
end

# The non-centered innovation prior: the iid `Normal(0, 1)` plate-vector
# prior shape over the `_ppl_scan_z_<state>` slice (one cell per step, same
# `_plate_sum_stmts` reduction the `PlateParameter` path uses), totalled
# under the scan-flavored `_ppl_scan_<state>` node.
function _scan_noncentered_prior!(stmts::Vector{Expr}, nodes::Vector{Symbol},
        s::ScanSpec)
    _scan_noncentered_form(s)    # fail closed unless the v1 shape holds
    zname = _scan_innovation_name(s)
    cell = _family_logpdf_expr(:normal, Any[0, 1], _dovar(1))
    node = Symbol(:_ppl_scan_, s.state)
    pw = Symbol(:_ppl_scan_pw_, s.state)
    append!(stmts, _plate_sum_stmts(pw, node, Any[zname], cell))
    push!(nodes, node)
    return nothing
end

# Prior-density statements for every scan plus the total-node names to add to
# the prior sum. Centered: the recurrence body is exactly one indexed `~` of
# the carried state (`state[t] ~ dist`); the density is the seed term(s) plus
# a plate over aligned lagged slices (the recurrence factorizes given the
# state). Non-centered: the iid-innovation prior over the `_ppl_scan_z_`
# slice (the plate-vector prior shape), with the state reconstructed in
# `_scan_reconstruction_statements`.
function _scan_prior_statements(plan::StructuralPlan, layout::LayoutTable)
    stmts = Expr[]
    nodes = Symbol[]
    for s in plan.scans
        if _is_noncentered_scan(s)
            _scan_noncentered_prior!(stmts, nodes, s)
            continue
        end
        (length(s.step) == 1 && s.step[1].kind === :sample &&
         s.step[1].indexed && s.step[1].target === s.state) || throw(
            ContractValidationError("[generator] scan $(s.state): only centered " *
                "recurrences (`$(s.state)[$(s.loopvar)] ~ dist`) and v1 " *
                "non-centered recurrences (one innovation sample + one carry " *
                "write) emit"))
        step = s.step[1]
        entry = only(e for e in layout.entries
                     if e.kind === :scan && e.name === s.state)
        T = entry.size
        m = length(s.setup)
        terms = Any[]
        for (k, f) in enumerate(s.setup)
            seed = Symbol(:_ppl_scan_seed_, s.state, :_, k)
            push!(stmts, :($seed::Float64 =
                $(_family_logpdf_expr(f.family, f.args, :($(s.state)[$k])))))
            push!(terms, seed)
        end
        lags = _scan_step_lags(step, s.state, s.loopvar)
        inputs = Any[:(view($(s.state), $(m + 1):$T))]     # hcur → do-var _ppl_c1
        dovar = Dict{Int,Symbol}()
        for (i, j) in enumerate(lags)
            push!(inputs, :(view($(s.state), $(m + 1 - j):$(T - j))))
            dovar[j] = _dovar(i + 1)
        end
        # Substitute the lag reads, then thread the recurrence's captured scalars
        # (params/assignments) as explicit plate inputs — RK requires a plate
        # cell's distribution args to be caller ports, not lexical captures.
        lagargs = [_subst_scan_lags(a, s.state, s.loopvar, dovar) for a in step.args]
        caps = Symbol[]
        for a in lagargs
            _scan_cell_caps!(caps, a)
        end
        s.loopvar in caps && throw(ContractValidationError(
            "[generator] scan $(s.state): the recurrence uses the loop index " *
            "`$(s.loopvar)` directly — not supported in slice 1"))
        capmap = Dict{Symbol,Symbol}()
        for c in caps
            push!(inputs, c)
            capmap[c] = _dovar(length(inputs))
        end
        cellargs = [_subst_syms(a, capmap) for a in lagargs]
        cell = _family_logpdf_expr(step.family, cellargs, _dovar(1))
        recnode = Symbol(:_ppl_scan_rec_, s.state)
        recpw = Symbol(:_ppl_scan_pw_, s.state)
        append!(stmts, _plate_sum_stmts(recpw, recnode, inputs, cell))
        push!(terms, recnode)
        total = Symbol(:_ppl_scan_, s.state)
        push!(stmts, :($total::Float64 = $(foldl((a, b) -> :($a + $b), terms))))
        push!(nodes, total)
    end
    return stmts, nodes
end

function _log_jacobian_statement(layout::LayoutTable)
    terms = Any[]
    for e in layout.entries
        t = jacobian_term(e)
        t === nothing || push!(terms, t)
    end
    jac = isempty(terms) ? :(0.0) : foldl((a, b) -> :($a + $b), terms)
    return :(log_jacobian::Float64 = $jac)
end

