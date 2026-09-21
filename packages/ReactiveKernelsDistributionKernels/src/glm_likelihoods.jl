# Fused whole-vector GLM likelihoods (plain Julia, white-box tested).
#
# Each family shares one scalar cell formula and one scalar theta
# (dℓ/dη) formula across its single-pass fused loops — no near-copies:
# `H` (the ruled scalar entry), `_cells` (`pointwise`), `_score`
# (`score`), plus the rule's own forward/reverse loops over branchless
# twins of the same formulas (see the `_bl` note below). The loops
# differ only in what they store per cell, never in math. All formulas
# are Stan's fused glm forms (single-exp reuse with cutoff-20 tails),
# so values match Stan's *_glm to floating error.
#
# Only the scalar `H` entries are `@noinline`: the Enzyme rule matches the
# call, and inlining it would silently disable the analytic adjoint (the
# ops-containment test in test/test_glm.jl guards this).

# ---------------- bernoulli_logit ----------------
#
# Single-pass fused loops throughout: one exp per cell, the exp cached for
# the reverse (the tape is `exp_m` only — branch tests read `em` magnitude,
# never a stored `ytheta`). The interior cell takes the sign split
# (`-log1p(em)` for `yt >= 0`, `yt - log1p(1/em)` below): both arms keep
# `log1p` on small args, while a branchless `-log1p(em)` pays scalar-libm's
# slow large-arg path half the time (measured +7ns/cell). Tail branches are
# taken rarely enough to predict out.

const _GLM_EM_LO = exp(-20.0) # em below this ⟺ ytheta above the cutoff
const _GLM_EM_HI = exp(20.0) # em above this ⟺ ytheta below the cutoff

@inline function _glm_bernoulli_cell(yt::Float64, em::Float64)
    if yt > 20.0
        return -em
    elseif yt < -20.0
        return yt
    elseif yt >= 0.0
        return -log1p(em)
    else
        return yt - log1p(1 / em)
    end
end

@inline function _glm_bernoulli_theta(s::Real, em::Float64)
    if em < _GLM_EM_LO
        return s * em
    elseif em > _GLM_EM_HI
        return s * 1.0
    else
        return s * em / (em + 1.0)
    end
end

# Branchless twins of the cell/theta formulas for the `@turbo` loops in the
# Enzyme rules (same cutoff semantics, same edges; rounding differs ≤2ulp
# from the sign-split scalar twins — far below every test tolerance, and the
# AD tests assert rule-forward/H-primal value consistency directly).
# Scalar-libm's slow large-arg `log1p` path is why the scalar loops need the
# sign split; SLEEF has no slow path, so the SIMD loops take the plain form.
# Deliberately UNTYPED arguments: LoopVectorization's `can_turbo` gate
# probes callees with integer `Vec`s, which never match a `::Float64`
# annotation — annotated twins silently disable SIMD (measured: both rule
# loops fell back to scalar until the annotations came off).
@inline function _glm_bernoulli_cell_bl(yt, em)
    ifelse(yt > 20.0, -em, ifelse(yt < -20.0, yt, -log1p(em)))
end

@inline function _glm_bernoulli_theta_bl(s, em)
    ifelse(em < _GLM_EM_LO, s * em,
        ifelse(em > _GLM_EM_HI, s * 1.0, s * em / (em + 1.0)))
end

function _glm_bernoulli_checklengths(y::AbstractVector, eta::AbstractVector)
    n = length(eta)
    length(y) == n ||
        throw(DimensionMismatch("bernoulli_logit_glm: y has $(length(y)) entries, eta has $n"))
    return n
end

@noinline function _glm_bernoulli_logit(y::AbstractVector, eta::AbstractVector)
    n = _glm_bernoulli_checklengths(y, eta)
    v = 0.0
    @inbounds for i in 1:n
        yt = (2 * y[i] - 1) * eta[i]
        v += _glm_bernoulli_cell(yt, exp(-yt))
    end
    return v
end

@inline function _glm_bernoulli_logit_cells(y::AbstractVector, eta::AbstractVector)
    n = _glm_bernoulli_checklengths(y, eta)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        yt = (2 * y[i] - 1) * eta[i]
        out[i] = _glm_bernoulli_cell(yt, exp(-yt))
    end
    return out
end

@inline function _glm_bernoulli_logit_score(y::AbstractVector, eta::AbstractVector)
    n = _glm_bernoulli_checklengths(y, eta)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        out[i] = _glm_bernoulli_theta(2 * y[i] - 1, exp(-(2 * y[i] - 1) * eta[i]))
    end
    return out
end

# ---------------- poisson_log ----------------
#
# Stan's form with no cutoff (Stan has none either): the cache is
# `exp.(eta)`, the value is `dot(y,eta) - sum(cache) - cterm`, and the
# theta is `y - cache`. The data-only `cterm`/`lg` terms are graph
# recipes over the bound response (see the object source), so `bound=`
# folds them to constants — they never enter the differentiated code.

# Untyped for the same `can_turbo` integer-Vec probe reason as the
# bernoulli `_bl` twins (these feed `@turbo` loops in the poisson rule).
@inline _glm_poisson_cell(y, e, et) = y * e - et
@inline _glm_poisson_theta(y, et) = y - et

function _glm_poisson_checklengths(y::AbstractVector, eta::AbstractVector)
    n = length(eta)
    length(y) == n ||
        throw(DimensionMismatch("poisson_log_glm: y has $(length(y)) entries, eta has $n"))
    return n
end

@noinline function _glm_poisson_log(
        y::AbstractVector, eta::AbstractVector, cterm::Float64)
    n = _glm_poisson_checklengths(y, eta)
    v = -cterm
    @inbounds for i in 1:n
        v += _glm_poisson_cell(y[i], eta[i], exp(eta[i]))
    end
    return v
end

@inline function _glm_poisson_log_cells(
        y::AbstractVector, eta::AbstractVector, lg::AbstractVector)
    n = _glm_poisson_checklengths(y, eta)
    length(lg) == n || throw(DimensionMismatch(
        "poisson_log_glm: lg has $(length(lg)) entries, eta has $n"))
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        out[i] = _glm_poisson_cell(y[i], eta[i], exp(eta[i])) - lg[i]
    end
    return out
end

@inline function _glm_poisson_log_score(y::AbstractVector, eta::AbstractVector)
    n = _glm_poisson_checklengths(y, eta)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        out[i] = _glm_poisson_theta(y[i], exp(eta[i]))
    end
    return out
end

# ---------------- normal_id ----------------
#
# Identity link, scalar sigma (v1; heteroskedastic sigma is a fast-follow
# dual route). No transcendentals in the reverse at all: theta is
# `(y-eta)/sigma^2`, so the rule recomputes the residual from its inputs
# and carries an EMPTY tape — not even Stan avoids its theta temporary.
# The `-0.5*log2pi` rides per-cell, so no observation count ever enters.

# Untyped: these feed `@turbo` loops (see the `can_turbo` note above).
@inline _glm_normal_cell(r, w, c) = -0.5 * r * r * w + c
@inline _glm_normal_theta(r, w) = r * w

function _glm_normal_checklengths(y::AbstractVector, eta::AbstractVector)
    n = length(eta)
    length(y) == n ||
        throw(DimensionMismatch("normal_id_glm: y has $(length(y)) entries, eta has $n"))
    return n
end

@noinline function _glm_normal_id(y::AbstractVector, eta::AbstractVector, sigma::Float64)
    n = _glm_normal_checklengths(y, eta)
    w = 1.0 / (sigma * sigma)
    c = -log(sigma) - 0.5 * log(2π)
    v = 0.0
    @inbounds for i in 1:n
        v += _glm_normal_cell(y[i] - eta[i], w, c)
    end
    return v
end

# Fused entry: (X, beta) straight to the scalar, for the object's `logpdf`
# endpoint. The rule on THIS (not on the eta entry) is what the object
# differentiates: keeping the matvec inside the ruled call avoids the
# n-vector Enzyme shadow on eta (alloc + zero + read-modify-write traffic
# Stan never pays — measured 1.35x behind Stan on the n=200K gradient
# through the eta entry). External-eta consumers (thin-layer predictors)
# use `_glm_normal_id`, whose rule stays for them.
@noinline function _glm_normal_id_fused(
        y::AbstractVector, X::AbstractMatrix, beta::AbstractVector, sigma::Float64)
    return _glm_normal_id(y, X * beta, sigma)
end

@inline function _glm_normal_id_cells(
        y::AbstractVector, eta::AbstractVector, sigma::Float64)
    n = _glm_normal_checklengths(y, eta)
    w = 1.0 / (sigma * sigma)
    c = -log(sigma) - 0.5 * log(2π)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        out[i] = _glm_normal_cell(y[i] - eta[i], w, c)
    end
    return out
end

@inline function _glm_normal_id_score(
        y::AbstractVector, eta::AbstractVector, sigma::Float64)
    n = _glm_normal_checklengths(y, eta)
    w = 1.0 / (sigma * sigma)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        out[i] = _glm_normal_theta(y[i] - eta[i], w)
    end
    return out
end

# ---------------- binomial_logit ----------------
#
# Value in log1pexp form (`y*eta - N*log1pexp(eta)`, exact at all eta, no
# cutoff needed); cache is `p = logistic(eta)` (one exp); theta is
# `y - N*p` with zero transcendentals. Stan's equivalent does ~2
# log1pexp forward plus an exp in reverse — strictly more. The rounding
# of `p` to 0/1 at extreme eta is harmless (theta stays correct; the
# value never touches `p`). The `lchoose` vector arrives precomputed
# (bound data folds it; see the object source).

# Untyped: these feed `@turbo` loops (see the `can_turbo` note above).
@inline _glm_binomial_cell(y, e, n, lp, lc) = y * e - n * lp + lc
@inline _glm_binomial_theta(y, n, p) = y - n * p

# Branchless `log1pexp` twin for `@turbo` (the library spelling branches on
# the sign of `e`, which kills vectorization). `max(e,0) + log1p(exp(-|e|))`
# is the same operations as `log1pexp` on each side, so bit-identical; used
# by the scalar loops too so ruled and unruled values agree exactly.
@inline _glm_log1pexp_bl(e) = max(e, 0.0) + log1p(exp(-abs(e)))

function _glm_binomial_checklengths(y::AbstractVector, eta::AbstractVector)
    n = length(eta)
    length(y) == n ||
        throw(DimensionMismatch("binomial_logit_glm: y has $(length(y)) entries, eta has $n"))
    return n
end

@noinline function _glm_binomial_logit(y::AbstractVector, eta::AbstractVector,
        ntrials::AbstractVector, lch::AbstractVector)
    n = _glm_binomial_checklengths(y, eta)
    v = 0.0
    @inbounds for i in 1:n
        e = eta[i]
        v += _glm_binomial_cell(y[i], e, ntrials[i], _glm_log1pexp_bl(e), lch[i])
    end
    return v
end

@inline function _glm_binomial_logit_cells(y::AbstractVector, eta::AbstractVector,
        ntrials::AbstractVector, lch::AbstractVector)
    n = _glm_binomial_checklengths(y, eta)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        e = eta[i]
        out[i] = _glm_binomial_cell(y[i], e, ntrials[i], _glm_log1pexp_bl(e), lch[i])
    end
    return out
end

@inline function _glm_binomial_logit_score(y::AbstractVector, eta::AbstractVector,
        ntrials::AbstractVector)
    n = _glm_binomial_checklengths(y, eta)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        out[i] = _glm_binomial_theta(y[i], ntrials[i], 1.0 / (1.0 + exp(-eta[i])))
    end
    return out
end

# ---------------- neg_binomial_2_log ----------------
#
# NB2 in log-mean form: cell = y*(eta - log(phi)) - (y+phi)*log1p(mu/phi)
# + (lgamma(y+phi) - lgamma(y+1) - lgamma(phi)). Both lgamma vectors
# arrive precomputed (endpoint temps over ports, so bound data folds
# them — including lgamma(y+phi) when phi is bound, which is exactly
# what Stan's propto=true skips and the naive version recomputes per
# call at 1.5x). With no lgamma left inside, the ruled forward is
# `@turbo` (SIMD exp/log1p); the scalar primals below stay plain for
# Reactant (see the header note in glm_rules.jl). The sampled-phi
# reverse still pays digamma per cell, exactly like Stan. Reciprocal
# forms (`phi/mu + 1` denominators) keep the adjoints finite at extreme
# eta (no NaN from Inf/Inf).

# Untyped: `_glm_negbin2_cell` and `_glm_negbin2_theta` feed `@turbo`
# loops (see the `can_turbo` note above).
@inline _glm_negbin2_cell(y, e, phi, s, lg_phi, lg_y1, lg_yp, er) =
    y * (e - s) - (y + phi) * log1p(er / phi) + (lg_yp - lg_y1 - lg_phi)
@inline _glm_negbin2_theta(y, phi, er) = y - (y + phi) / (phi / er + 1)
# Direct-only: the +digamma(y+phi) term lives on the lgp-temp path,
# which the graph differentiates itself (the rule propagates dℓ/dlgp
# into lgp's shadow; see the Dup-phi methods). Bundling it here would
# double-count it under sampled phi.
@inline _glm_negbin2_dphi(y, phi, er, dg_phi) =
    -y / phi - log1p(er / phi) + (y + phi) / (phi * (phi / er + 1)) - dg_phi

function _glm_negbin2_checklengths(y::AbstractVector, eta::AbstractVector)
    n = length(eta)
    length(y) == n ||
        throw(DimensionMismatch("neg_binomial_2_log_glm: y has $(length(y)) entries, eta has $n"))
    return n
end

@noinline function _glm_negbin2_log(y::AbstractVector, eta::AbstractVector,
        phi::Float64, lg_y1::AbstractVector, lg_yp::AbstractVector)
    n = _glm_negbin2_checklengths(y, eta)
    s = log(phi)
    lg_phi = loggamma(phi)
    v = 0.0
    @inbounds for i in 1:n
        e = eta[i]
        er = exp(e)
        v += _glm_negbin2_cell(y[i], e, phi, s, lg_phi, lg_y1[i], lg_yp[i], er)
    end
    return v
end

@noinline function _glm_negbin2_log_fused(y::AbstractVector, X::AbstractMatrix,
        beta::AbstractVector, phi::Float64,
        lg_y1::AbstractVector, lg_yp::AbstractVector)
    n = length(y)
    size(X, 1) == n || throw(DimensionMismatch(
        "neg_binomial_2_log_glm fused: X has $(size(X, 1)) rows for $n observations"))
    eta_t = X * beta
    s = log(phi)
    lg_phi = loggamma(phi)
    v = 0.0
    @inbounds for i in 1:n
        e = eta_t[i]
        er = exp(e)
        v += _glm_negbin2_cell(y[i], e, phi, s, lg_phi, lg_y1[i], lg_yp[i], er)
    end
    return v
end

@inline function _glm_negbin2_log_cells(y::AbstractVector, eta::AbstractVector,
        phi::Float64, lg_y1::AbstractVector, lg_yp::AbstractVector)
    n = _glm_negbin2_checklengths(y, eta)
    s = log(phi)
    lg_phi = loggamma(phi)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        e = eta[i]
        out[i] = _glm_negbin2_cell(y[i], e, phi, s, lg_phi, lg_y1[i], lg_yp[i], exp(e))
    end
    return out
end

@inline function _glm_negbin2_log_score(y::AbstractVector, eta::AbstractVector,
        phi::Float64)
    n = _glm_negbin2_checklengths(y, eta)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        out[i] = _glm_negbin2_theta(y[i], phi, exp(eta[i]))
    end
    return out
end

# ---------------- categorical_logit ----------------
#
# Rowwise softmax cross-entropy; no data-only normalizer (like
# bernoulli). The rule's forward caches the full softmax matrix as the
# tape (n×C doubles — less than Stan's autodiff stack for this op); the
# eta reverse is then a flat pure-arithmetic axpy plus a scatter of +d
# at the observed codes. Stan recomputes n×C exps in reverse — that gap
# is the whole story. Loops touching y-as-index run WITHOUT @inbounds:
# an out-of-range code must BoundsError, never corrupt.

function _glm_categorical_checkshapes(y::AbstractVector, eta::AbstractMatrix)
    n = size(eta, 1)
    length(y) == n ||
        throw(DimensionMismatch("categorical_logit_glm: y has $(length(y)) entries, eta has $n rows"))
    return (n, size(eta, 2))
end

@noinline function _glm_categorical_logit(y::AbstractVector, eta::AbstractMatrix)
    (n, C) = _glm_categorical_checkshapes(y, eta)
    v = 0.0
    for i in 1:n
        yi = y[i]
        m = eta[i, 1]
        for c in 2:C
            e = eta[i, c]
            e > m && (m = e)
        end
        s = 0.0
        for c in 1:C
            s += exp(eta[i, c] - m)
        end
        v += eta[i, yi] - m - log(s)
    end
    return v
end

@inline function _glm_categorical_logit_cells(y::AbstractVector, eta::AbstractMatrix)
    (n, C) = _glm_categorical_checkshapes(y, eta)
    out = Vector{Float64}(undef, n)
    for i in 1:n
        yi = y[i]
        m = eta[i, 1]
        for c in 2:C
            e = eta[i, c]
            e > m && (m = e)
        end
        s = 0.0
        for c in 1:C
            s += exp(eta[i, c] - m)
        end
        out[i] = eta[i, yi] - m - log(s)
    end
    return out
end

@inline function _glm_categorical_logit_score(y::AbstractVector, eta::AbstractMatrix)
    (n, C) = _glm_categorical_checkshapes(y, eta)
    out = Matrix{Float64}(undef, n, C)
    for i in 1:n
        yi = y[i]
        m = eta[i, 1]
        for c in 2:C
            e = eta[i, c]
            e > m && (m = e)
        end
        s = 0.0
        for c in 1:C
            s += exp(eta[i, c] - m)
        end
        for c in 1:C
            out[i, c] = (c == yi) - exp(eta[i, c] - m) / s
        end
    end
    return out
end

# Column-softmax helpers: LLVM SVML-vectorizes simple column loops
# (8-wide exp/log); the rowwise nest only gets partial vectorization
# (measured ~2x slower at n=200K). Summation order matches the rowwise
# form exactly (c = 1..C), so values are bit-identical.
@inline function _glm_cat_colmax!(buf::AbstractVector, eta::AbstractMatrix, n::Int, C::Int)
    @inbounds for i in 1:n
        buf[i] = eta[i, 1]
    end
    for c in 2:C
        @inbounds @simd for i in 1:n
            e = eta[i, c]
            buf[i] = ifelse(e > buf[i], e, buf[i])
        end
    end
    buf
end

@inline function _glm_cat_softmax_value(eta::AbstractMatrix, y::AbstractVector,
        maxv::AbstractVector, logS::AbstractVector, n::Int)
    v = 0.0
    @inbounds for i in 1:n
        v += eta[i, y[i]] - maxv[i] - logS[i]
    end
    v
end

# Fused entry: (X, beta, alpha) in, summed log-density out. Plain NN
# GEMM plus a column-form softmax (a C×n-transposed variant measured
# slower: the transposed GEMM reads X strided, dwarfing the
# sequential-loop gain; the rowwise nest vectorizes only partially).
@noinline function _glm_categorical_logit_fused(y::AbstractVector, X::AbstractMatrix,
        B::AbstractMatrix, a::AbstractVector)
    n = length(y)
    size(X, 1) == n || throw(DimensionMismatch(
        "categorical_logit_glm fused: X has $(size(X, 1)) rows for $n observations"))
    size(X, 2) == size(B, 1) || throw(DimensionMismatch(
        "categorical_logit_glm fused: X has $(size(X, 2)) cols for $(size(B, 1)) beta rows"))
    C = length(a)
    size(B, 2) == C || throw(DimensionMismatch(
        "categorical_logit_glm fused: beta has $(size(B, 2)) cols for $C classes"))
    eta = Matrix{Float64}(undef, n, C)
    mul!(eta, X, B)
    eta .+= a'
    maxv = Vector{Float64}(undef, n)
    S = Vector{Float64}(undef, n)
    _glm_cat_colmax!(maxv, eta, n, C)
    fill!(S, 0.0)
    for c in 1:C
        @inbounds @simd for i in 1:n
            S[i] += exp(eta[i, c] - maxv[i])
        end
    end
    @inbounds @simd for i in 1:n
        S[i] = log(S[i])
    end
    return _glm_cat_softmax_value(eta, y, maxv, S, n)
end

# ---------------- ordered_logistic ----------------
#
# Cumulative-logit cells from a padded cut vector (`[-Inf; cuts; Inf]`
# kills the edge branches): cell = log(F(hi - e) - F(lo - e)). The eta
# adjoint collapses exactly — (a(1-a) - b(1-b))/(b-a) = a + b - 1 (no
# division, no cutoff, exact at all eta; the tests check it against the
# direct form). The cuts adjoint scatters per-obs pairs and skips zero
# denominators (the cell value is -Inf there, so the partial is moot —
# but it must stay non-NaN). The forward caches (a, b); Stan recomputes
# 2 exps per cell in reverse. Probability-space throughout: exact for
# |eta - c| below ~700 (past that the value saturates to -Inf while the
# score stays ideal-exact — a log-space form would cost ~2x the
# transcendentals for a region that never occurs). Data-indexed loops
# skip @inbounds (bad codes BoundsError, never corrupt).

# Untyped: `_glm_ordered_theta` feeds a `@turbo` loop (see the
# `can_turbo` note above).
@inline _glm_ordered_cell(a, b) = log(b - a)
@inline _glm_ordered_theta(a, b) = a + b - 1

function _glm_ordered_checklengths(y::AbstractVector, eta::AbstractVector)
    n = length(eta)
    length(y) == n ||
        throw(DimensionMismatch("ordered_logistic_glm: y has $(length(y)) entries, eta has $n"))
    return n
end

@noinline function _glm_ordered_logistic(y::AbstractVector, eta::AbstractVector,
        cuts::AbstractVector)
    n = _glm_ordered_checklengths(y, eta)
    cp = [-Inf; cuts; Inf]
    v = 0.0
    for i in 1:n
        yi = y[i]
        e = eta[i]
        a = LogExpFunctions.logistic(cp[yi] - e)
        b = LogExpFunctions.logistic(cp[yi + 1] - e)
        v += _glm_ordered_cell(a, b)
    end
    return v
end

@inline function _glm_ordered_logistic_cells(y::AbstractVector, eta::AbstractVector,
        cuts::AbstractVector)
    n = _glm_ordered_checklengths(y, eta)
    cp = [-Inf; cuts; Inf]
    out = Vector{Float64}(undef, n)
    for i in 1:n
        yi = y[i]
        e = eta[i]
        a = LogExpFunctions.logistic(cp[yi] - e)
        b = LogExpFunctions.logistic(cp[yi + 1] - e)
        out[i] = _glm_ordered_cell(a, b)
    end
    return out
end

@inline function _glm_ordered_logistic_score(y::AbstractVector, eta::AbstractVector,
        cuts::AbstractVector)
    n = _glm_ordered_checklengths(y, eta)
    cp = [-Inf; cuts; Inf]
    out = Vector{Float64}(undef, n)
    for i in 1:n
        yi = y[i]
        e = eta[i]
        a = LogExpFunctions.logistic(cp[yi] - e)
        b = LogExpFunctions.logistic(cp[yi + 1] - e)
        out[i] = _glm_ordered_theta(a, b)
    end
    return out
end

# IWLS weights (dE[y]/deta)^2/Var(y) via the survival sums E[y] = 1 +
# sum S_k, E[y^2] = 1 + sum (2k+1) S_k with S_k = P(Y > k). Zero
# variance (all mass on one class, or m2 - m1^2 rounding) maps to weight
# 0 (no information) rather than NaN.
@inline function _glm_ordered_weights(eta::AbstractVector, cuts::AbstractVector)
    n = length(eta)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        e = eta[i]
        m1 = 1.0
        m2 = 1.0
        dm = 0.0
        for k in 1:length(cuts)
            s = LogExpFunctions.logistic(e - cuts[k])
            m1 += s
            m2 += (2k + 1) * s
            dm += s * (1 - s)
        end
        V = m2 - m1 * m1
        out[i] = V > 0 ? (dm * dm) / V : 0.0
    end
    return out
end
