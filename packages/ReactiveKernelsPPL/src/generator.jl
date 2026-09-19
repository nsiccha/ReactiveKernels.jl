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
    isempty(plan.ranef_buckets) || throw(ContractValidationError(
        "[generator] ranef codegen is not Stage A (K=1 geometry lands in " *
        "Stage B, LKJ-correlated in Stage C) — refusing to silently drop " *
        "$(length(plan.ranef_buckets)) bucket(s)"))
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
    beta, inverse_gamma, binomial, negative_binomial2
using SpecialFunctions: erfc
# Selective (explicit imports win over any re-export chain, so no `using`
# ambiguity if ReactiveKernels ever exports these too): the only Statistics
# names in the assignment allowlist.
using Statistics: mean, std, var
using LinearAlgebra: dot
# Bijector objects the generated program splices (constrained-parameter
# transforms); imported from the enclosing module so the emitted
# `positive_bijector()` / `unit_bijector()` calls resolve.
import ..positive_bijector, ..unit_bijector
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
        if shape.width > 0
            push!(terms, :($(design_name(pred.name)) * $(block_name(pred.name))))
        end
        if any(b -> b.kind === OffsetTerm, shape.blocks)
            push!(terms, offset_name(pred.name))
        end
        # A latent term contributes the per-cell latent VECTOR directly
        # (identity design): `lp = theta` on its own, or added to fixed-effect
        # design/offset terms for a random-intercept-plus-covariates predictor.
        for b in shape.blocks
            b.kind === LatentTerm && push!(terms, b.column)
        end
        # Degenerate (e.g. single-level-factor-only) predictors carry a scalar
        # zero LP, which broadcasts everywhere a vector LP would.
        rhs = isempty(terms) ? :(0.0) : foldl((a, b) -> :($a + $b), terms)
        push!(stmts, :($lp = $rhs))
    end
    return stmts
end

_lik_name(label::Symbol) = Symbol(:_ppl_lik_, label)

function _likelihood_statements(plan::StructuralPlan)
    stmts = Expr[]
    terms = Any[]
    for r in plan.responses
        append!(stmts, _response_likelihood_stmts(r, plan))
        push!(terms, _lik_name(r.label))
    end
    joint = foldl((a, b) -> :($a + $b), terms; init = :(0.0))
    push!(stmts, :(likelihood::Float64 = $joint))
    return stmts
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
        return _bernoulli_plate_stmts(r, plan, node, pw)
    elseif r.family === PoissonLogFam
        # Base GLM case (no evidence, no weights): fused whole-vector reduction
        # (faster native + Reactant; the per-cell plate handles evidence/weights).
        r.evidence.kind === :none && r.weights === nothing &&
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
    else
        throw(ContractValidationError(
            "[generator] response family $(r.family) has no emitter"))
    end
end

_predictor(plan::StructuralPlan, name::Symbol) =
    only(p for p in plan.predictors if p.name === name)

# The per-observation location node feeding a response's likelihood plate: a
# scan-state latent vector fed directly (its own name — the layout view), or a
# linear predictor's `_ppl_lp_<name>` node otherwise.
function _location_node(r::LikelihoodSpec, plan::StructuralPlan)
    any(s -> s.state === r.predictor, plan.scans) && return r.predictor
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
    inputs = Any[y, lp]
    yv, lpv = _dovar(1), _dovar(2)
    sref = _thread_ref!(inputs, r.scale)
    lb, ub = _thread_bounds!(inputs, r.evidence, false)
    base = :(normal($lpv, $sref).logpdf($yv))
    cell = _gaussian_cell(r.evidence.kind, base, yv, lb, ub, lpv, sref)
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return _plate_sum_stmts(pw, node, inputs, cell)
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
    pre = :($mu = exp.($lp))
    inputs = Any[y, mu]
    yv, muv = _dovar(1), _dovar(2)
    phiref = _thread_ref!(inputs, r.scale)
    cell = :(negative_binomial2($muv, $phiref).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre, _plate_sum_stmts(pw, node, inputs, cell)...]
end

function _gamma_plate_stmts(r::LikelihoodSpec, plan::StructuralPlan, node::Symbol, pw::Symbol)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    # Surface is Distributions-SCALE `Gamma(alpha, mu/alpha)`; the kernel
    # takes rate, so the boundary inverts (same as the sampled-gamma prior).
    av = r.scale isa Symbol ? r.scale : Float64(r.scale)
    rate = _rate_name(r.label)
    pre = :($rate = $av ./ exp.($lp))
    inputs = Any[y, rate]
    yv, ratev = _dovar(1), _dovar(2)
    aref = _thread_ref!(inputs, r.scale)
    cell = :(gamma($aref, $ratev).logpdf($yv))
    if r.weights !== nothing
        wv = _thread_ref!(inputs, r.weights)
        cell = :($wv * $cell)
    end
    return Expr[pre, _plate_sum_stmts(pw, node, inputs, cell)...]
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

function _prior_statements(plan::StructuralPlan, layout::LayoutTable)
    stmts = Expr[]
    terms = Any[]
    for pred in plan.predictors
        shape = design_shape(pred, plan.columns; levelmaps = plan.levelmaps)
        shape.width == 0 && continue
        node = Symbol(:_ppl_prior_, pred.name)
        pw = Symbol(:_ppl_pw_prior_, pred.name)
        loc, sca = coefficient_priors(shape, plan.population_priors)
        mut = Symbol(:_ppl_prmu_, pred.name)
        sdt = Symbol(:_ppl_prsd_, pred.name)
        push!(stmts, :($mut = Float64[$(loc...)]))
        push!(stmts, :($sdt = Float64[$(sca...)]))
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
    # Per-cell latent (plate) parameters: one plate over the block, summing the
    # shared-prior log-density across cells (the same plate-sum shape as a
    # population-prior coefficient block, generalized to any standard family).
    # Every value the cell reads is threaded as a plate PORT (the latent vector
    # plus each scalar prior arg) — captured free names are rejected by the
    # `@kernel` plate expander, exactly as the Gaussian-likelihood scale is
    # threaded.
    for p in plan.plate_parameters
        node = Symbol(:_ppl_prior_, p.name)
        pw = Symbol(:_ppl_pw_prior_, p.name)
        inputs = Any[p.name]
        tv = _dovar(1)
        argvals = Any[_thread_ref!(inputs, v) for v in values(p.args)]
        cell = _family_logpdf_expr(p.family, argvals, tv)
        corr = _support_correction(p.support_override, argvals)
        corr === nothing || (cell = :($cell + $corr))
        append!(stmts, _plate_sum_stmts(pw, node, inputs, cell))
        push!(terms, node)
    end
    # Sequential-recurrence (scan) latents: setup + recurrence density.
    scanstmts, scannodes = _scan_prior_statements(plan, layout)
    append!(stmts, scanstmts)
    append!(terms, scannodes)
    joint = foldl((a, b) -> :($a + $b), terms; init = :(0.0))
    push!(stmts, :(prior::Float64 = $joint))
    return stmts
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

# Prior-density statements for every scan plus the total-node names to add to
# the prior sum. Slice-1 CENTERED only: the recurrence body is exactly one
# indexed `~` of the carried state (`state[t] ~ dist`); the density is the seed
# term(s) plus a plate over aligned lagged slices (the recurrence factorizes
# given the state). Non-centered (reconstruction via RK-core `scan(...)`) is
# a follow-up.
function _scan_prior_statements(plan::StructuralPlan, layout::LayoutTable)
    stmts = Expr[]
    nodes = Symbol[]
    for s in plan.scans
        (length(s.step) == 1 && s.step[1].kind === :sample &&
         s.step[1].indexed && s.step[1].target === s.state) || throw(
            ContractValidationError("[generator] scan $(s.state): only centered " *
                "recurrences (`$(s.state)[$(s.loopvar)] ~ dist`) emit in slice 1 " *
                "(non-centered reconstruction is planned)"))
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

