# Program generator: StructuralPlan → self-contained `@kernel` program.
#
# Emission order per program: preprocessing recipes (data-only, folded by
# `bound=`) → layout transforms (unconstrained → constrained) → scalar
# assignments in topo order → linear predictors → per-response fused
# likelihoods → prior terms → canonical `prior`/`likelihood`/`log_jacobian`/
# `posterior` nodes. Likelihoods are whole-vector fused forms (§4a), never
# per-cell plates. The `@kernel` def is evaluated in the dedicated
# `PPLGeneratedModels` scope (counter-suffixed binding per build).

"""
    build_kernel(plan) -> (; spec, layout)

Validate, assign layout, emit, and evaluate a self-contained `@kernel`
program for `plan`. `spec` is the `KernelSpec` (callable after `prepare`
with `have=(:unconstrained, data…)`); `layout` is its
[`LayoutTable`](@ref) (R10 read API for the sampler side).
"""
function build_kernel(plan::StructuralPlan)
    validate_plan(plan)
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
    stmts = Expr[]
    append!(stmts, preprocessing_recipes(plan))
    for e in layout.entries
        append!(stmts, transform_statements(e))
    end
    append!(stmts, _assignment_statements(plan))
    append!(stmts, _predictor_statements(plan))
    append!(stmts, _likelihood_statements(plan))
    append!(stmts, _prior_statements(plan))
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
using ReactiveKernels, SpecialFunctions, LogExpFunctions
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

# Scalar assignments in topo order (params already constrained above, so
# every scalar name resolves). Unannotated: Int temporaries (e.g. `length`)
# must not meet a Float64 assertion.
function _assignment_statements(plan::StructuralPlan)
    by_name = Dict{Symbol,AssignmentSpec}(a.name => a for a in plan.assignments)
    return Expr[:($(name) = $(by_name[name].expr))
        for name in topological_order(plan) if haskey(by_name, name)]
end

_lp_name(pred::PredictorSpec) = Symbol(:_ppl_lp_, pred.name)

function _predictor_statements(plan::StructuralPlan)
    stmts = Expr[]
    for pred in plan.predictors
        shape = design_shape(pred, plan.columns)
        lp = _lp_name(pred)
        terms = Any[]
        if shape.width > 0
            push!(terms, :($(design_name(pred.name)) * $(block_name(pred.name))))
        end
        if any(b -> b.kind === OffsetTerm, shape.blocks)
            push!(terms, offset_name(pred.name))
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

# One fused whole-vector likelihood node per response (plus a shared lambda
# node for Poisson+evidence). Triples 2 and 3 (Bernoulli-logit) lower
# identically; the triple only selects the form.
function _response_likelihood_stmts(r::LikelihoodSpec, plan::StructuralPlan)
    node = _lik_name(r.label)
    y = r.response
    lp = _lp_name(_predictor(plan, r.predictor))
    if r.family === GaussianFam
        return Expr[:($node::Float64 = $(_gaussian_body(r, y, lp)))]
    elseif r.family === BernoulliLogitFam
        return Expr[:($node::Float64 = $(_bernoulli_body(r, y, lp)))]
    else
        return _poisson_stmts(r, node, y, lp)
    end
end

_predictor(plan::StructuralPlan, name::Symbol) =
    only(p for p in plan.predictors if p.name === name)

_scale_expr(r::LikelihoodSpec) = r.scale # Symbol ref or Real literal, both spliceable

function _gaussian_body(r::LikelihoodSpec, y::Symbol, lp::Symbol)
    s = _scale_expr(r)
    base = if r.weights === nothing
        :(-0.5 * (length($y) * log(2 * pi) + 2 * length($y) * log($s) +
            sum(($y .- $lp) .^ 2) / $s ^ 2))
    else
        w = r.weights
        :(-0.5 * (sum($w) * log(2 * pi) + 2 * sum($w) * log($s) +
            sum($w .* ($y .- $lp) .^ 2) / $s ^ 2))
    end
    kind = r.evidence.kind
    kind === :none && return base
    kind === :truncated &&
        return :($base - sum($(_weighted_cells(r, _gaussian_trunc_cells(r, y, lp, s)))))
    kind === :censored &&
        return :(sum($(_weighted_cells(r, _gaussian_cens_cells(r, y, lp, s)))))
    return :(sum($(_weighted_cells(r, _gaussian_ival_cells(r, y, lp, s)))))
end

# Φ(bound) vector over rows via Φ(z) = 0.5*erfc(-z/√2), or nothing when that
# side is unbounded. Bounds cross as literals or columns; both splice
# uniformly. Missing sides are statically omitted (the generator knows).
function _phi_expr(bound, y::Symbol, lp::Symbol, s)
    bound === nothing && return nothing
    b = _bound_value(bound)
    return :(0.5 .* erfc.((.-($b .- $lp)) ./ $s ./ sqrt(2)))
end

_bound_value(bound::Real) = Float64(bound)
_bound_value(bound::Symbol) = bound

_weighted_cells(r::LikelihoodSpec, cells) =
    r.weights === nothing ? cells : :($(r.weights) .* $cells)

function _gaussian_trunc_cells(r::LikelihoodSpec, y::Symbol, lp::Symbol, s)
    ev = r.evidence
    phi_lb = _phi_expr(ev.lower, y, lp, s)
    phi_ub = _phi_expr(ev.upper, y, lp, s)
    phi_lb === nothing && phi_ub === nothing && return :(0.0)
    phi_lb === nothing && return :(log.($phi_ub))
    phi_ub === nothing && return :(log.(1.0 .- $phi_lb))
    return :(log.($phi_ub .- $phi_lb))
end

_gaussian_base_cell(y::Symbol, lp::Symbol, s) =
    :(-0.5 * log(2 * pi) .- log($s) .- 0.5 .* (($y .- $lp) ./ $s) .^ 2)

function _gaussian_cens_cells(r::LikelihoodSpec, y::Symbol, lp::Symbol, s)
    ev = r.evidence
    base_cell = _gaussian_base_cell(y, lp, s)
    phi_lb = _phi_expr(ev.lower, y, lp, s)
    phi_ub = _phi_expr(ev.upper, y, lp, s)
    phi_lb === nothing && phi_ub === nothing && return base_cell
    phi_lb === nothing &&
        return :(ifelse.($y .> $(_bound_value(ev.upper)), log1p.(-$phi_ub), $base_cell))
    phi_ub === nothing &&
        return :(ifelse.($y .< $(_bound_value(ev.lower)), log.($phi_lb), $base_cell))
    lb = _bound_value(ev.lower)
    ub = _bound_value(ev.upper)
    return :(ifelse.($y .< $lb, log.($phi_lb),
        ifelse.($y .> $ub, log1p.(-$phi_ub), $base_cell)))
end

function _gaussian_ival_cells(r::LikelihoodSpec, y::Symbol, lp::Symbol, s)
    # Response is the lower endpoint; contribution = log(Φ_u - Φ_d).
    phi_d = :(0.5 .* erfc.((.-($y .- $lp)) ./ $s ./ sqrt(2)))
    phi_u = _phi_expr(r.evidence.upper, y, lp, s)
    return :(log.($phi_u .- $phi_d))
end

function _bernoulli_body(r::LikelihoodSpec, y::Symbol, lp::Symbol)
    cells = :($y .* $lp .- log1pexp.($lp))
    r.weights === nothing && return :(sum($cells))
    return :(sum($(r.weights) .* $cells))
end

function _poisson_stmts(r::LikelihoodSpec, node::Symbol, y::Symbol, lp::Symbol)
    base_cell = :($y .* $lp .- exp.($lp) .- loggamma.($y .+ 1.0))
    base = r.weights === nothing ? :(sum($base_cell)) :
        :(sum($(r.weights) .* $base_cell))
    r.evidence.kind === :none &&
        return Expr[:($node::Float64 = $base)]
    lam = Symbol(:_ppl_lambda_, r.label)
    lam_stmt = :($lam = exp.($lp))
    body = if r.evidence.kind === :truncated
        :($base - sum($(_weighted_cells(r, _poisson_trunc_cells(r, y, lam)))))
    elseif r.evidence.kind === :interval_censored
        :($(_sum_expr(r, _poisson_ival_cells(r, y, lam))))
    else
        :($(_sum_expr(r, _poisson_cens_cells(r, y, lp, lam))))
    end
    return Expr[lam_stmt, :($node::Float64 = $body)]
end

_sum_expr(r::LikelihoodSpec, cells) = :(sum($(_weighted_cells(r, cells))))

# Poisson CDF via the regularized upper gamma (DistributionKernels
# precedent): P(k,λ) = last(gamma_inc(k+1, λ)), guarded for k < 0.
function _poisson_cdf_expr(k, lam::Symbol)
    return :(ifelse.($k .< 0, 0.0,
        last.(gamma_inc.(max.($k, 0) .+ 1.0, $lam))))
end

function _poisson_trunc_cells(r::LikelihoodSpec, y::Symbol, lam::Symbol)
    ev = r.evidence
    pu = ev.upper === nothing ? nothing :
        _poisson_cdf_expr(_bound_value(ev.upper), lam)
    pl = ev.lower === nothing ? nothing :
        _poisson_cdf_expr(_bound_value(ev.lower), lam)
    pu === nothing && pl === nothing && return :(0.0)
    pu === nothing && return :(log.(1.0 .- $pl))
    pl === nothing && return :(log.($pu))
    return :(log.($pu .- $pl))
end

function _poisson_ival_cells(r::LikelihoodSpec, y::Symbol, lam::Symbol)
    pu = _poisson_cdf_expr(_bound_value(r.evidence.upper), lam)
    pd = _poisson_cdf_expr(y, lam)
    return :(log.($pu .- $pd))
end

function _poisson_cens_cells(r::LikelihoodSpec, y::Symbol, lp::Symbol, lam::Symbol)
    ev = r.evidence
    base_cell = :($y .* $lp .- exp.($lp) .- loggamma.($y .+ 1.0))
    pu = ev.upper === nothing ? nothing :
        _poisson_cdf_expr(_bound_value(ev.upper), lam)
    pl = ev.lower === nothing ? nothing :
        _poisson_cdf_expr(_bound_value(ev.lower), lam)
    pu === nothing && pl === nothing && return base_cell
    pu === nothing &&
        return :(ifelse.($y .< $(_bound_value(ev.lower)), log.($pl), $base_cell))
    pl === nothing &&
        return :(ifelse.($y .> $(_bound_value(ev.upper)), log1p.(-$pu), $base_cell))
    lb = _bound_value(ev.lower)
    ub = _bound_value(ev.upper)
    return :(ifelse.($y .< $lb, log.($pl),
        ifelse.($y .> $ub, log1p.(-$pu), $base_cell)))
end

function _prior_statements(plan::StructuralPlan)
    stmts = Expr[]
    terms = Any[]
    for pred in plan.predictors
        shape = design_shape(pred, plan.columns)
        shape.width == 0 && continue
        node = Symbol(:_ppl_prior_, pred.name)
        loc, sca = coefficient_priors(shape, plan.population_priors)
        locvec = Expr(:vect, loc...)
        scavec = Expr(:vect, sca...)
        coef = block_name(pred.name)
        push!(stmts,
            :($node::Float64 = -0.5 * (length($coef) * log(2 * pi) +
                2 * sum(log.($scavec)) +
                sum((($coef .- $locvec) ./ $scavec) .^ 2))))
        push!(terms, node)
    end
    for p in plan.parameters
        node = Symbol(:_ppl_prior_, p.name)
        push!(stmts, :($node::Float64 = $(_sampled_prior_expr(p))))
        push!(terms, node)
    end
    joint = foldl((a, b) -> :($a + $b), terms; init = :(0.0))
    push!(stmts, :(prior::Float64 = $joint))
    return stmts
end

# Scalar prior log-density per family (Distributions.jl semantics). Args are
# literals (baked as-is) or parameter/assignment refs (emitted earlier in
# topo order). Half-Normal/half-Cauchy (the only overrides) renormalize by
# exactly +log(2) (symmetry at 0).
function _sampled_prior_expr(p::SampledParameter)
    x = p.name
    a = [v for v in values(p.args)]
    base = if p.family === :normal
        mu, s = a
        :(-0.5 * log(2 * pi) - log($s) - 0.5 * (($x - $mu) / $s) ^ 2)
    elseif p.family === :cauchy
        mu, s = a
        :(-log(pi) - log($s) - log(1 + (($x - $mu) / $s) ^ 2))
    elseif p.family === :exponential
        (th,) = a
        :(-log($th) - $x / $th)
    elseif p.family === :gamma
        al, th = a
        :(-loggamma($al) - $al * log($th) + ($al - 1) * log($x) - $x / $th)
    elseif p.family === :lognormal
        mu, s = a
        :(-log($x) - 0.5 * log(2 * pi) - log($s) - 0.5 * ((log($x) - $mu) / $s) ^ 2)
    elseif p.family === :beta
        al, be = a
        :(-(loggamma($al) + loggamma($be) - loggamma($al + $be)) +
            ($al - 1) * log($x) + ($be - 1) * log1p(-$x))
    elseif p.family === :inverse_gamma
        al, th = a
        :($al * log($th) - loggamma($al) - ($al + 1) * log($x) - $th / $x)
    else # :flat
        :(0.0)
    end
    p.support_override === nothing && return base
    return :($base + log(2))
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

