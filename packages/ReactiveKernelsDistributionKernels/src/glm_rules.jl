# Analytic reverse adjoints for the fused GLM scalar likelihoods — Stan's
# `partials_propagator` analog. Each rule's forward pass runs one fused
# `@turbo` loop (value + tape cache, SLEEF transcendentals); the reverse
# reads the tape back with ZERO transcendentals in a second `@turbo` loop.
# A generic Enzyme reverse instead recomputes the transcendental per cell
# (measured 1.4–1.8× slower than Stan on bernoulli gradients), so these
# rules are the entire gradient story. Rule bodies are never
# differentiated themselves, which is what makes `@turbo` safe here; the
# scalar `H` primals stay plain so the generic-AD fallback (and Reactant
# tracing) never sees LoopVectorization IR.
#
# Forward-mode rules are deliberately absent: with no forward rule defined
# at all, Enzyme uses generic forward AD (correct, just unoptimized), and
# reverse is the hot path. Reverse combos are another story: a missing
# method does NOT fall back — Enzyme throws `custom_rule_method_error`
# (observed, not assumed). So every meaningful activity combo gets a
# method (eta/beta Const-or-Duplicated × family-param Const-or-active,
# data always Const); only meaningless combos (active data: y/X/N
# Duplicated) still throw, which is the honest failure for a meaningless
# query. Scalar actives arrive as `Active{Float64}` on the real kernel
# path (scalars have no shadow) — never bare `Duplicated{Float64}` in
# practice — so scalar slots take the Union of both (belt and braces;
# bodies are identical since both return the partial by value).

function EnzymeCore.EnzymeRules.augmented_primal(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_bernoulli_logit)},
        ::Type{<:EnzymeCore.Active},
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Duplicated)
    yv = y.val
    etav = eta.val
    n = _glm_bernoulli_checklengths(yv, etav)
    exp_m = Vector{Float64}(undef, n)
    v = 0.0
    @turbo for i in 1:n
        yt = (2 * yv[i] - 1) * etav[i]
        em = exp(-yt)
        exp_m[i] = em
        v += _glm_bernoulli_cell_bl(yt, em)
    end
    primal = EnzymeCore.EnzymeRules.needs_primal(config) ? v : nothing
    return EnzymeCore.EnzymeRules.AugmentedReturn(primal, nothing, exp_m)
end

function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_bernoulli_logit)},
        dret::EnzymeCore.Active, tape,
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Duplicated)
    exp_m = tape
    yv = y.val
    ed = eta.dval
    n = length(ed)
    (length(yv) == n && length(exp_m) == n) ||
        throw(DimensionMismatch("bernoulli_logit_glm adjoint: y/tape length mismatch"))
    d = dret.val
    @turbo for i in 1:n
        ed[i] = ed[i] + d * _glm_bernoulli_theta_bl(2 * yv[i] - 1, exp_m[i])
    end
    return (nothing, nothing)
end

function EnzymeCore.EnzymeRules.augmented_primal(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_poisson_log)},
        ::Type{<:EnzymeCore.Active},
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Duplicated,
        cterm::EnzymeCore.Const{Float64})
    yv = y.val
    etav = eta.val
    n = _glm_poisson_checklengths(yv, etav)
    exp_t = Vector{Float64}(undef, n)
    v = -cterm.val
    @turbo for i in 1:n
        et = exp(etav[i])
        exp_t[i] = et
        v += _glm_poisson_cell(yv[i], etav[i], et)
    end
    primal = EnzymeCore.EnzymeRules.needs_primal(config) ? v : nothing
    return EnzymeCore.EnzymeRules.AugmentedReturn(primal, nothing, exp_t)
end

function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_poisson_log)},
        dret::EnzymeCore.Active, exp_t,
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Duplicated,
        cterm::EnzymeCore.Const{Float64})
    yv = y.val
    ed = eta.dval
    n = length(ed)
    (length(yv) == n && length(exp_t) == n) ||
        throw(DimensionMismatch("poisson_log_glm adjoint: y/tape length mismatch"))
    d = dret.val
    @turbo for i in 1:n
        ed[i] = ed[i] + d * _glm_poisson_theta(yv[i], exp_t[i])
    end
    return (nothing, nothing, nothing)
end

# Normal forward: one method — sigma is only read here, so abstract
# `Annotation` covers Const and Duplicated alike (both expose `.val`).
function EnzymeCore.EnzymeRules.augmented_primal(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_normal_id)},
        ::Type{<:EnzymeCore.Active},
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Annotation,
        sigma::EnzymeCore.Annotation)
    yv = y.val
    etav = eta.val
    n = _glm_normal_checklengths(yv, etav)
    s = sigma.val
    w = 1.0 / (s * s)
    c = -log(s) - 0.5 * log(2π)
    v = 0.0
    @turbo for i in 1:n
        v += _glm_normal_cell(yv[i] - etav[i], w, c)
    end
    primal = EnzymeCore.EnzymeRules.needs_primal(config) ? v : nothing
    return EnzymeCore.EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

# Normal reverse, bound sigma: recompute residuals, accumulate eta only.
function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_normal_id)},
        dret::EnzymeCore.Active, tape,
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Duplicated,
        sigma::EnzymeCore.Const{Float64})
    yv = y.val
    ed = eta.dval
    n = length(ed)
    length(yv) == n ||
        throw(DimensionMismatch("normal_id_glm adjoint: y/eta length mismatch"))
    w = 1.0 / (sigma.val * sigma.val)
    d = dret.val
    @turbo for i in 1:n
        ed[i] = ed[i] + d * _glm_normal_theta(yv[i] - eta.val[i], w)
    end
    return (nothing, nothing, nothing)
end

# Normal reverse, sampled sigma: additionally return dℓ/dσ. A Float64
# shadow is immutable, so unlike the vector case it cannot accumulate in
# place — the convention is returning the contribution, which Enzyme adds
# to the caller's shadow (validated by the sigma-active FD test).
function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_normal_id)},
        dret::EnzymeCore.Active, tape,
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Duplicated,
        sigma::Union{EnzymeCore.Active{Float64}, EnzymeCore.Duplicated{Float64}})
    yv = y.val
    etav = eta.val
    ed = eta.dval
    n = length(ed)
    length(yv) == n ||
        throw(DimensionMismatch("normal_id_glm adjoint: y/eta length mismatch"))
    s = sigma.val
    w = 1.0 / (s * s)
    d = dret.val
    r2sum = 0.0
    @turbo for i in 1:n
        r = yv[i] - etav[i]
        ed[i] = ed[i] + d * _glm_normal_theta(r, w)
        r2sum += r * r
    end
    return (nothing, nothing, d * (r2sum * w / s - n / s))
end

# Normal reverse, bound eta + sampled sigma (beta bound — the Gibbs
# shape): sigma partial only.
function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_normal_id)},
        dret::EnzymeCore.Active, tape,
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Const{<:AbstractVector},
        sigma::Union{EnzymeCore.Active{Float64}, EnzymeCore.Duplicated{Float64}})
    yv = y.val
    etav = eta.val
    n = length(etav)
    length(yv) == n ||
        throw(DimensionMismatch("normal_id_glm adjoint: y/eta length mismatch"))
    s = sigma.val
    w = 1.0 / (s * s)
    d = dret.val
    r2sum = 0.0
    @turbo for i in 1:n
        r = yv[i] - etav[i]
        r2sum += r * r
    end
    return (nothing, nothing, d * (r2sum * w / s - n / s))
end

# Fused normal rules: the matvec lives inside the ruled call, so no
# n-vector Enzyme shadow on eta is ever allocated (see `_fused` note).
function EnzymeCore.EnzymeRules.augmented_primal(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_normal_id_fused)},
        ::Type{<:EnzymeCore.Active},
        y::EnzymeCore.Const{<:AbstractVector}, X::EnzymeCore.Const{<:AbstractMatrix},
        beta::EnzymeCore.Annotation, sigma::EnzymeCore.Annotation)
    yv = y.val
    Xv = X.val
    n = length(yv)
    size(Xv, 1) == n || throw(DimensionMismatch(
        "normal_id_glm fused: X has $(size(Xv, 1)) rows for $n observations"))
    eta_t = Xv * beta.val
    s = sigma.val
    w = 1.0 / (s * s)
    c = -log(s) - 0.5 * log(2π)
    v = 0.0
    @turbo for i in 1:n
        v += _glm_normal_cell(yv[i] - eta_t[i], w, c)
    end
    primal = EnzymeCore.EnzymeRules.needs_primal(config) ? v : nothing
    return EnzymeCore.EnzymeRules.AugmentedReturn(primal, nothing, eta_t)
end

function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_normal_id_fused)},
        dret::EnzymeCore.Active, eta_t,
        y::EnzymeCore.Const{<:AbstractVector}, X::EnzymeCore.Const{<:AbstractMatrix},
        beta::EnzymeCore.Duplicated, sigma::EnzymeCore.Const{Float64})
    yv = y.val
    Xv = X.val
    n = length(yv)
    (length(eta_t) == n && size(Xv, 1) == n) ||
        throw(DimensionMismatch("normal_id_glm fused adjoint: shape mismatch"))
    w = 1.0 / (sigma.val * sigma.val)
    d = dret.val
    theta = Vector{Float64}(undef, n)
    @turbo for i in 1:n
        theta[i] = d * _glm_normal_theta(yv[i] - eta_t[i], w)
    end
    mul!(beta.dval, Xv', theta, 1.0, 1.0)
    return (nothing, nothing, nothing, nothing)
end

function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_normal_id_fused)},
        dret::EnzymeCore.Active, eta_t,
        y::EnzymeCore.Const{<:AbstractVector}, X::EnzymeCore.Const{<:AbstractMatrix},
        beta::EnzymeCore.Duplicated, sigma::Union{EnzymeCore.Active{Float64}, EnzymeCore.Duplicated{Float64}})
    yv = y.val
    Xv = X.val
    n = length(yv)
    (length(eta_t) == n && size(Xv, 1) == n) ||
        throw(DimensionMismatch("normal_id_glm fused adjoint: shape mismatch"))
    s = sigma.val
    w = 1.0 / (s * s)
    d = dret.val
    theta = Vector{Float64}(undef, n)
    r2sum = 0.0
    @turbo for i in 1:n
        r = yv[i] - eta_t[i]
        theta[i] = d * _glm_normal_theta(r, w)
        r2sum += r * r
    end
    mul!(beta.dval, Xv', theta, 1.0, 1.0)
    return (nothing, nothing, nothing, d * (r2sum * w / s - n / s))
end

# Fused reverse, bound beta + sampled sigma: sigma partial only, off the
# cached eta (no theta vector — nothing accumulates into beta).
function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_normal_id_fused)},
        dret::EnzymeCore.Active, eta_t,
        y::EnzymeCore.Const{<:AbstractVector}, X::EnzymeCore.Const{<:AbstractMatrix},
        beta::EnzymeCore.Const{<:AbstractVector}, sigma::Union{EnzymeCore.Active{Float64}, EnzymeCore.Duplicated{Float64}})
    yv = y.val
    n = length(yv)
    length(eta_t) == n ||
        throw(DimensionMismatch("normal_id_glm fused adjoint: shape mismatch"))
    s = sigma.val
    w = 1.0 / (s * s)
    d = dret.val
    r2sum = 0.0
    @turbo for i in 1:n
        r = yv[i] - eta_t[i]
        r2sum += r * r
    end
    return (nothing, nothing, nothing, d * (r2sum * w / s - n / s))
end

# Binomial rule: forward caches `p = logistic(eta)` (one exp) while the
# value uses the branchless log1pexp twin (robust at all eta, no cutoff);
# reverse is `y - N*p` with zero transcendentals. Stan's reverse
# recomputes an exp per cell, so this is the whole gradient story again.
function EnzymeCore.EnzymeRules.augmented_primal(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_binomial_logit)},
        ::Type{<:EnzymeCore.Active},
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Duplicated,
        ntrials::EnzymeCore.Const{<:AbstractVector}, lch::EnzymeCore.Const{<:AbstractVector})
    yv = y.val
    etav = eta.val
    nv = ntrials.val
    lcv = lch.val
    n = _glm_binomial_checklengths(yv, etav)
    p = Vector{Float64}(undef, n)
    v = 0.0
    @turbo for i in 1:n
        e = etav[i]
        pi = 1.0 / (1.0 + exp(-e))
        p[i] = pi
        v += _glm_binomial_cell(yv[i], e, nv[i], _glm_log1pexp_bl(e), lcv[i])
    end
    primal = EnzymeCore.EnzymeRules.needs_primal(config) ? v : nothing
    return EnzymeCore.EnzymeRules.AugmentedReturn(primal, nothing, p)
end

function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_binomial_logit)},
        dret::EnzymeCore.Active, p,
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Duplicated,
        ntrials::EnzymeCore.Const{<:AbstractVector}, lch::EnzymeCore.Const{<:AbstractVector})
    yv = y.val
    ed = eta.dval
    nv = ntrials.val
    n = length(p)
    d = dret.val
    @turbo for i in 1:n
        ed[i] = ed[i] + d * _glm_binomial_theta(yv[i], nv[i], p[i])
    end
    return (nothing, nothing, nothing, nothing)
end

# Negbin rule: both lgamma vectors arrive precomputed (folded when
# bound), so the forward is a fused `@turbo` loop (value + mu cache,
# SIMD exp/log1p); the eta reverse is a pure-arithmetic `@turbo` axpy
# off that cache (Stan recomputes a vectorized exp), and the sampled-phi
# reverse adds a scalar direct-partial loop plus propagation into lgp's
# shadow (the graph owns the lgp-temp path and adds its digamma share
# itself — bundling it here would double-count). One augmented serves
# every live activity combo.
function EnzymeCore.EnzymeRules.augmented_primal(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_negbin2_log)},
        ::Type{<:EnzymeCore.Active},
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Annotation,
        phi::EnzymeCore.Annotation, lg_y1::EnzymeCore.Const{<:AbstractVector},
        lg_yp::EnzymeCore.Annotation)
    yv = y.val
    etav = eta.val
    n = _glm_negbin2_checklengths(yv, etav)
    pv = phi.val
    s = log(pv)
    lg_phi = loggamma(pv)
    lgy1v = lg_y1.val
    lgypv = lg_yp.val
    er_cache = Vector{Float64}(undef, n)
    v = 0.0
    @turbo for i in 1:n
        e = etav[i]
        er = exp(e)
        er_cache[i] = er
        v += _glm_negbin2_cell(yv[i], e, pv, s, lg_phi, lgy1v[i], lgypv[i], er)
    end
    primal = EnzymeCore.EnzymeRules.needs_primal(config) ? v : nothing
    return EnzymeCore.EnzymeRules.AugmentedReturn(primal, nothing, er_cache)
end

# Negbin reverse, bound phi: recompute nothing, accumulate eta only.
function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_negbin2_log)},
        dret::EnzymeCore.Active, er_cache,
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Duplicated,
        phi::EnzymeCore.Const{Float64}, lg_y1::EnzymeCore.Const{<:AbstractVector},
        lg_yp::EnzymeCore.Const{<:AbstractVector})
    yv = y.val
    ed = eta.dval
    n = length(er_cache)
    pv = phi.val
    d = dret.val
    @turbo for i in 1:n
        ed[i] = ed[i] + d * _glm_negbin2_theta(yv[i], pv, er_cache[i])
    end
    return (nothing, nothing, nothing, nothing, nothing)
end

# Negbin reverse, sampled phi: the `@turbo` eta axpy plus the direct phi
# partial by value (immutable Float64 shadow convention, as for sigma)
# plus propagation into lgp's shadow (unit coefficient).
function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_negbin2_log)},
        dret::EnzymeCore.Active, er_cache,
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Duplicated,
        phi::Union{EnzymeCore.Active{Float64}, EnzymeCore.Duplicated{Float64}}, lg_y1::EnzymeCore.Const{<:AbstractVector},
        lg_yp::EnzymeCore.Duplicated)
    yv = y.val
    ed = eta.dval
    n = length(er_cache)
    pv = phi.val
    dg_phi = digamma(pv)
    lgpd = lg_yp.dval
    d = dret.val
    @turbo for i in 1:n
        ed[i] = ed[i] + d * _glm_negbin2_theta(yv[i], pv, er_cache[i])
    end
    dp = 0.0
    @inbounds for i in 1:n
        dp += _glm_negbin2_dphi(yv[i], pv, er_cache[i], dg_phi)
        lgpd[i] += d
    end
    return (nothing, nothing, d * dp, nothing, nothing)
end

# Negbin reverse, bound eta + sampled phi: phi loop only.
function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_negbin2_log)},
        dret::EnzymeCore.Active, er_cache,
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Const{<:AbstractVector},
        phi::Union{EnzymeCore.Active{Float64}, EnzymeCore.Duplicated{Float64}}, lg_y1::EnzymeCore.Const{<:AbstractVector},
        lg_yp::EnzymeCore.Duplicated)
    yv = y.val
    n = length(er_cache)
    pv = phi.val
    dg_phi = digamma(pv)
    lgpd = lg_yp.dval
    d = dret.val
    dp = 0.0
    @inbounds for i in 1:n
        dp += _glm_negbin2_dphi(yv[i], pv, er_cache[i], dg_phi)
        lgpd[i] += d
    end
    return (nothing, nothing, d * dp, nothing, nothing)
end

# Fused negbin rules: the matvec lives inside the ruled call, so no
# n-vector Enzyme shadow on eta is ever allocated (the eta-entry tax
# Stan never pays — see the normal fused note). The tape is the mu
# cache alone (theta and dphi both read it; eta itself is never
# needed back).
function EnzymeCore.EnzymeRules.augmented_primal(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_negbin2_log_fused)},
        ::Type{<:EnzymeCore.Active},
        y::EnzymeCore.Const{<:AbstractVector}, X::EnzymeCore.Const{<:AbstractMatrix},
        beta::EnzymeCore.Annotation, phi::EnzymeCore.Annotation,
        lg_y1::EnzymeCore.Const{<:AbstractVector}, lg_yp::EnzymeCore.Annotation)
    yv = y.val
    Xv = X.val
    n = length(yv)
    size(Xv, 1) == n || throw(DimensionMismatch(
        "neg_binomial_2_log_glm fused: X has $(size(Xv, 1)) rows for $n observations"))
    eta_t = Xv * beta.val
    pv = phi.val
    s = log(pv)
    lg_phi = loggamma(pv)
    lgy1v = lg_y1.val
    lgypv = lg_yp.val
    er_cache = Vector{Float64}(undef, n)
    v = 0.0
    @turbo for i in 1:n
        e = eta_t[i]
        er = exp(e)
        er_cache[i] = er
        v += _glm_negbin2_cell(yv[i], e, pv, s, lg_phi, lgy1v[i], lgypv[i], er)
    end
    primal = EnzymeCore.EnzymeRules.needs_primal(config) ? v : nothing
    return EnzymeCore.EnzymeRules.AugmentedReturn(primal, nothing, er_cache)
end

function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_negbin2_log_fused)},
        dret::EnzymeCore.Active, er_cache,
        y::EnzymeCore.Const{<:AbstractVector}, X::EnzymeCore.Const{<:AbstractMatrix},
        beta::EnzymeCore.Duplicated, phi::EnzymeCore.Const{Float64},
        lg_y1::EnzymeCore.Const{<:AbstractVector}, lg_yp::EnzymeCore.Const{<:AbstractVector})
    yv = y.val
    Xv = X.val
    n = length(yv)
    (length(er_cache) == n && size(Xv, 1) == n) ||
        throw(DimensionMismatch("neg_binomial_2_log_glm fused adjoint: shape mismatch"))
    pv = phi.val
    d = dret.val
    theta = Vector{Float64}(undef, n)
    @turbo for i in 1:n
        theta[i] = d * _glm_negbin2_theta(yv[i], pv, er_cache[i])
    end
    mul!(beta.dval, Xv', theta, 1.0, 1.0)
    return (nothing, nothing, nothing, nothing, nothing, nothing)
end

function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_negbin2_log_fused)},
        dret::EnzymeCore.Active, er_cache,
        y::EnzymeCore.Const{<:AbstractVector}, X::EnzymeCore.Const{<:AbstractMatrix},
        beta::EnzymeCore.Duplicated, phi::Union{EnzymeCore.Active{Float64}, EnzymeCore.Duplicated{Float64}},
        lg_y1::EnzymeCore.Const{<:AbstractVector}, lg_yp::EnzymeCore.Duplicated)
    yv = y.val
    Xv = X.val
    n = length(yv)
    (length(er_cache) == n && size(Xv, 1) == n) ||
        throw(DimensionMismatch("neg_binomial_2_log_glm fused adjoint: shape mismatch"))
    pv = phi.val
    dg_phi = digamma(pv)
    lgpd = lg_yp.dval
    d = dret.val
    theta = Vector{Float64}(undef, n)
    @turbo for i in 1:n
        theta[i] = d * _glm_negbin2_theta(yv[i], pv, er_cache[i])
    end
    mul!(beta.dval, Xv', theta, 1.0, 1.0)
    dp = 0.0
    @inbounds for i in 1:n
        dp += _glm_negbin2_dphi(yv[i], pv, er_cache[i], dg_phi)
        lgpd[i] += d
    end
    return (nothing, nothing, nothing, d * dp, nothing, nothing)
end

function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_negbin2_log_fused)},
        dret::EnzymeCore.Active, er_cache,
        y::EnzymeCore.Const{<:AbstractVector}, X::EnzymeCore.Const{<:AbstractMatrix},
        beta::EnzymeCore.Const{<:AbstractVector}, phi::Union{EnzymeCore.Active{Float64}, EnzymeCore.Duplicated{Float64}},
        lg_y1::EnzymeCore.Const{<:AbstractVector}, lg_yp::EnzymeCore.Duplicated)
    yv = y.val
    n = length(yv)
    length(er_cache) == n ||
        throw(DimensionMismatch("neg_binomial_2_log_glm fused adjoint: shape mismatch"))
    pv = phi.val
    dg_phi = digamma(pv)
    lgpd = lg_yp.dval
    d = dret.val
    dp = 0.0
    @inbounds for i in 1:n
        dp += _glm_negbin2_dphi(yv[i], pv, er_cache[i], dg_phi)
        lgpd[i] += d
    end
    return (nothing, nothing, nothing, d * dp, nothing, nothing)
end

# Categorical rule: scalar rowwise forward (dynamic-C inner loops, no
# `@turbo`; data-indexed, no `@inbounds`) caching the softmax matrix;
# flat `@turbo` axpy off the tape plus a bounds-checked scatter at the
# observed codes in reverse. Single method: beta and alpha both flow
# through eta, so no extra activity entries are needed.
function EnzymeCore.EnzymeRules.augmented_primal(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_categorical_logit)},
        ::Type{<:EnzymeCore.Active},
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Duplicated)
    yv = y.val
    etav = eta.val
    (n, C) = _glm_categorical_checkshapes(yv, etav)
    P = Matrix{Float64}(undef, n, C)
    v = 0.0
    for i in 1:n
        yi = yv[i]
        m = etav[i, 1]
        for c in 2:C
            e = etav[i, c]
            e > m && (m = e)
        end
        s = 0.0
        for c in 1:C
            raw = exp(etav[i, c] - m)
            P[i, c] = raw
            s += raw
        end
        v += etav[i, yi] - m - log(s)
        for c in 1:C
            P[i, c] /= s
        end
    end
    primal = EnzymeCore.EnzymeRules.needs_primal(config) ? v : nothing
    return EnzymeCore.EnzymeRules.AugmentedReturn(primal, nothing, P)
end

function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_categorical_logit)},
        dret::EnzymeCore.Active, P,
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Duplicated)
    yv = y.val
    ed = eta.dval
    size(ed) == size(P) ||
        throw(DimensionMismatch("categorical_logit_glm adjoint: eta/tape shape mismatch"))
    n = size(ed, 1)
    d = dret.val
    @turbo for idx in 1:length(ed)
        ed[idx] = ed[idx] - d * P[idx]
    end
    for i in 1:n
        ed[i, yv[i]] = ed[i, yv[i]] + d
    end
    return (nothing, nothing)
end

# Fused categorical rules: the matvec lives inside the ruled call, so no
# n×C Enzyme shadow on eta is ever allocated (the eta-entry's scattered
# RMW into that shadow measured 1.7x behind Stan). The reverse is all
# `@turbo`: fused dots replace the skinny BLAS GEMM (packing overhead
# measured 6.7x at n=200K), predicated passes replace the grouped
# scatter-RMW (2.4x), and the tape is already normalized so there is no
# explicit -softmax form pass. Forward and reverse both match Stan's
# scalar-exp floor; the reverse is ~3x leaner than Stan's.
function EnzymeCore.EnzymeRules.augmented_primal(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_categorical_logit_fused)},
        ::Type{<:EnzymeCore.Active},
        y::EnzymeCore.Const{<:AbstractVector}, X::EnzymeCore.Const{<:AbstractMatrix},
        B::EnzymeCore.Annotation, a::EnzymeCore.Annotation)
    yv = y.val
    Xv = X.val
    Bv = B.val
    av = a.val
    n = length(yv)
    size(Xv, 1) == n || throw(DimensionMismatch(
        "categorical_logit_glm fused: X has $(size(Xv, 1)) rows for $n observations"))
    C = length(av)
    eta = Matrix{Float64}(undef, n, C)
    mul!(eta, Xv, Bv)
    eta .+= av'
    P = Matrix{Float64}(undef, n, C)
    maxv = Vector{Float64}(undef, n)
    S = Vector{Float64}(undef, n)
    _glm_cat_colmax!(maxv, eta, n, C)
    fill!(S, 0.0)
    for c in 1:C
        @inbounds @simd for i in 1:n
            r = exp(eta[i, c] - maxv[i])
            P[i, c] = r
            S[i] += r
        end
    end
    @inbounds @simd for i in 1:n
        S[i] = log(S[i])
    end
    v = _glm_cat_softmax_value(eta, yv, maxv, S, n)
    # Normalize by re-summing P (the S buffer is free again: no second
    # tape) with one reciprocal per row instead of C divides each.
    fill!(S, 0.0)
    for c in 1:C
        @inbounds @simd for i in 1:n
            S[i] += P[i, c]
        end
    end
    @inbounds @simd for i in 1:n
        S[i] = inv(S[i])
    end
    for c in 1:C
        @inbounds @simd for i in 1:n
            P[i, c] *= S[i]
        end
    end
    primal = EnzymeCore.EnzymeRules.needs_primal(config) ? v : nothing
    return EnzymeCore.EnzymeRules.AugmentedReturn(primal, nothing, P)
end

function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_categorical_logit_fused)},
        dret::EnzymeCore.Active, P,
        y::EnzymeCore.Const{<:AbstractVector}, X::EnzymeCore.Const{<:AbstractMatrix},
        B::EnzymeCore.Duplicated, a::EnzymeCore.Duplicated)
    yv = y.val
    Xv = X.val
    Bd = B.dval
    ad = a.dval
    n = length(yv)
    C = length(ad)
    (size(P) == (n, C) && size(Bd, 2) == C) ||
        throw(DimensionMismatch("categorical_logit_glm fused adjoint: shape mismatch"))
    K = size(Xv, 2)
    d = dret.val
    # Predicated grouped sums: K*C vectorized passes replace the scalar
    # scatter-RMW (measured 2.4x faster at n=200K); counts fold in free.
    G = zeros(K, C)
    counts = zeros(C)
    for c in 1:C, k in 1:K
        s = 0.0
        @turbo for i in 1:n
            s += Xv[i, k] * (yv[i] == c)
        end
        G[k, c] = s
    end
    for c in 1:C
        s = 0.0
        @turbo for i in 1:n
            s += (yv[i] == c)
        end
        counts[c] = s
    end
    for c in 1:C
        s = 0.0
        @turbo for i in 1:n
            s += P[i, c]
        end
        ad[c] += d * (counts[c] - s)
    end
    # Fused dots replace the skinny BLAS GEMM (no pack overhead: 6.7x).
    acc = zeros(K, C)
    @turbo for i in 1:n, c in 1:C, k in 1:K
        acc[k, c] += Xv[i, k] * P[i, c]
    end
    Bd .+= d .* (G .- acc)
    return (nothing, nothing, nothing, nothing)
end

function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_categorical_logit_fused)},
        dret::EnzymeCore.Active, P,
        y::EnzymeCore.Const{<:AbstractVector}, X::EnzymeCore.Const{<:AbstractMatrix},
        B::EnzymeCore.Duplicated, a::EnzymeCore.Const{<:AbstractVector})
    yv = y.val
    Xv = X.val
    Bd = B.dval
    n = length(yv)
    C = size(Bd, 2)
    size(P) == (n, C) ||
        throw(DimensionMismatch("categorical_logit_glm fused adjoint: shape mismatch"))
    K = size(Xv, 2)
    d = dret.val
    G = zeros(K, C)
    for c in 1:C, k in 1:K
        s = 0.0
        @turbo for i in 1:n
            s += Xv[i, k] * (yv[i] == c)
        end
        G[k, c] = s
    end
    acc = zeros(K, C)
    @turbo for i in 1:n, c in 1:C, k in 1:K
        acc[k, c] += Xv[i, k] * P[i, c]
    end
    Bd .+= d .* (G .- acc)
    return (nothing, nothing, nothing, nothing)
end

function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_categorical_logit_fused)},
        dret::EnzymeCore.Active, P,
        y::EnzymeCore.Const{<:AbstractVector}, X::EnzymeCore.Const{<:AbstractMatrix},
        B::EnzymeCore.Const{<:AbstractMatrix}, a::EnzymeCore.Duplicated)
    yv = y.val
    ad = a.dval
    n = length(yv)
    C = length(ad)
    size(P) == (n, C) ||
        throw(DimensionMismatch("categorical_logit_glm fused adjoint: shape mismatch"))
    d = dret.val
    counts = zeros(C)
    for c in 1:C
        s = 0.0
        @turbo for i in 1:n
            s += (yv[i] == c)
        end
        counts[c] = s
    end
    for c in 1:C
        s = 0.0
        @turbo for i in 1:n
            s += P[i, c]
        end
        ad[c] += d * (counts[c] - s)
    end
    return (nothing, nothing, nothing, nothing)
end

# Ordered rule: scalar forward (data-indexed, no `@inbounds`) caching
# the (a, b) cumulative-logit pairs; the eta reverse is a
# pure-arithmetic `@turbo` axpy of a + b - 1, and the sampled-cuts
# reverse adds a bounds-checked scatter that skips zero denominators
# (the cell value is -Inf there, so the partial is moot — but it must
# stay non-NaN). One augmented serves both cuts activities.
function EnzymeCore.EnzymeRules.augmented_primal(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_ordered_logistic)},
        ::Type{<:EnzymeCore.Active},
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Annotation,
        cuts::EnzymeCore.Annotation)
    yv = y.val
    etav = eta.val
    n = _glm_ordered_checklengths(yv, etav)
    cp = [-Inf; cuts.val; Inf]
    A = Vector{Float64}(undef, n)
    B = Vector{Float64}(undef, n)
    v = 0.0
    for i in 1:n
        yi = yv[i]
        e = etav[i]
        a = LogExpFunctions.logistic(cp[yi] - e)
        b = LogExpFunctions.logistic(cp[yi + 1] - e)
        A[i] = a
        B[i] = b
        v += _glm_ordered_cell(a, b)
    end
    primal = EnzymeCore.EnzymeRules.needs_primal(config) ? v : nothing
    return EnzymeCore.EnzymeRules.AugmentedReturn(primal, nothing, (A, B))
end

# Ordered reverse, bound cuts: eta axpy only.
function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_ordered_logistic)},
        dret::EnzymeCore.Active, tape,
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Duplicated,
        cuts::EnzymeCore.Const{<:AbstractVector})
    A = tape[1]
    B = tape[2]
    ed = eta.dval
    n = length(ed)
    length(A) == n ||
        throw(DimensionMismatch("ordered_logistic_glm adjoint: eta/tape length mismatch"))
    d = dret.val
    @turbo for i in 1:n
        ed[i] = ed[i] + d * _glm_ordered_theta(A[i], B[i])
    end
    return (nothing, nothing, nothing)
end

# Ordered reverse, sampled cuts: eta axpy plus the cuts scatter. A
# Duplicated vector accumulates in place (unlike the immutable Float64
# scalar convention), so nothing is returned by value.
function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_ordered_logistic)},
        dret::EnzymeCore.Active, tape,
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Duplicated,
        cuts::EnzymeCore.Duplicated)
    A = tape[1]
    B = tape[2]
    yv = y.val
    ed = eta.dval
    cd = cuts.dval
    n = length(ed)
    length(A) == n ||
        throw(DimensionMismatch("ordered_logistic_glm adjoint: eta/tape length mismatch"))
    C = length(cd) + 1
    d = dret.val
    @turbo for i in 1:n
        ed[i] = ed[i] + d * _glm_ordered_theta(A[i], B[i])
    end
    for i in 1:n
        yi = yv[i]
        a = A[i]
        b = B[i]
        den = b - a
        # Exact-0.0 skip: the cell value is -Inf here, so the partial is
        # moot — but 0/0 would poison the whole gradient with NaN.
        den == 0 && continue
        if yi > 1
            cd[yi - 1] = cd[yi - 1] + d * (-a * (1 - a) / den)
        end
        if yi < C
            cd[yi] = cd[yi] + d * (b * (1 - b) / den)
        end
    end
    return (nothing, nothing, nothing)
end

# Ordered reverse, bound eta + sampled cuts: scatter only.
function EnzymeCore.EnzymeRules.reverse(
        config::EnzymeCore.EnzymeRules.RevConfigWidth{1},
        func::EnzymeCore.Const{typeof(_glm_ordered_logistic)},
        dret::EnzymeCore.Active, tape,
        y::EnzymeCore.Const{<:AbstractVector}, eta::EnzymeCore.Const{<:AbstractVector},
        cuts::EnzymeCore.Duplicated)
    A = tape[1]
    B = tape[2]
    yv = y.val
    cd = cuts.dval
    n = length(A)
    C = length(cd) + 1
    d = dret.val
    for i in 1:n
        yi = yv[i]
        a = A[i]
        b = B[i]
        den = b - a
        den == 0 && continue
        if yi > 1
            cd[yi - 1] = cd[yi - 1] + d * (-a * (1 - a) / den)
        end
        if yi < C
            cd[yi] = cd[yi] + d * (b * (1 - b) / den)
        end
    end
    return (nothing, nothing, nothing)
end
