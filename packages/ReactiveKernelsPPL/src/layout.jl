# Packed layout + transforms (D5b thin-layer-owned).
#
# Constrained-parameter transforms are provided by the reusable bijector
# library (`bijectors.jl`, decision 0l3dsru): the host path runs the bijector's
# prepared endpoints and the in-graph generator splices the same endpoints, so
# in-graph and host evaluation agree by construction (structural), not by
# hand-kept duplication. A SCALAR sampled entry splices the endpoints directly;
# a per-cell (plate) block maps the same scalar endpoints over its cell view via
# the `plate` primitive (host + graph both). `:identity` (real support) is a
# genuine no-op and stays a direct read in both paths.

"""
    support_of(family, override) -> Symbol

Inferred unconstrained support (`:real`/`:positive`/`:unit`) for a sampled
family plus optional `:positive` override (half-Normal/half-Cauchy style).
Loud on unknown families and inapplicable overrides.
"""
function support_of(family::Symbol, override::SupportOverride)
    haskey(SAMPLED_SUPPORT, family) ||
        throw(ContractValidationError("[layout] sampled family $family unknown"))
    inferred = SAMPLED_SUPPORT[family]
    override === nothing && return inferred
    if override isa Tuple
        override[1] === :interval || throw(ContractValidationError(
            "[layout] tuple support override must be (:interval, lo, hi), got $override"))
        inferred === :real || throw(ContractValidationError(
            "[layout] :interval override needs a real-support family"))
        return :interval
    end
    override === :positive || throw(
        ContractValidationError("[layout] support override must be :positive, got $override"),
    )
    inferred === :real || throw(
        ContractValidationError("[layout] :positive override needs a real-support family"),
    )
    return :positive
end

# The transform kind and (for :interval) the constrained bounds a layout entry
# needs, from a parameter's family + support override.
function _entry_transform(family::Symbol, override::SupportOverride)
    support = support_of(family, override)
    if support === :interval
        return (:interval, Float64(override[2]), Float64(override[3]))
    end
    transform =
        support === :real ? :identity :
        support === :positive ? :exp : :logistic
    return (transform, NaN, NaN)
end

"""One packed slice: a coefficient block, a scalar latent, a scan vector
latent, a per-cell latent block, a leveled vector latent (cutpoints,
thresholds, simplex), a spline coefficient-vector block, a ranef vector
block (K=1 `xi`, correlated `tau`/`z_flat`), a correlated-ranef LKJ
Cholesky factor, or an HSGP coefficient-vector block (`beta_raw`). `lo`
is the constrained lower bound of an `:interval`/`:floored` transform
(`:interval` also sets `hi`; `NaN` otherwise)."""
struct LayoutEntry
    kind::Symbol # :coefficient | :sampled | :scan | :plate | :vector | :spline | :ranef | :ranef_corr | :hsgp
    predictor::Union{Nothing,Symbol}
    name::Symbol # block name (`mu_coef`), parameter name, or scan-state name
    labels::Vector{Symbol} # per-coordinate labels (length == size)
    offset::Int # 1-based packed offset
    size::Int
    transform::Symbol # :identity | :exp | :logistic | :interval | :floored | :ordered | :simplex | :lkj
    lo::Float64 # :interval/:floored lower bound (else NaN)
    hi::Float64 # :interval upper bound (else NaN)
end
# Non-interval entries omit the bounds.
LayoutEntry(kind::Symbol, predictor::Union{Nothing,Symbol}, name::Symbol,
    labels::Vector{Symbol}, offset::Int, size::Int, transform::Symbol) =
    LayoutEntry(kind, predictor, name, labels, offset, size, transform, NaN, NaN)

"""Packed unconstrained layout: ordered entries + total dimension."""
struct LayoutTable
    entries::Vector{LayoutEntry}
    total::Int
end

"""
    assign_layout(plan) -> LayoutTable

Assign packed coordinates: coefficient blocks in `plan.predictors` order,
then sampled parameters, plate parameters, and vector parameters in plan
order (pinned, deterministic). Assumes `validate_plan` passed.
"""
function assign_layout(plan::StructuralPlan)
    isbound(plan) || throw(ContractValidationError(
        "[layout] assign_layout requires a bound plan (bind_data first)"))
    entries = LayoutEntry[]
    offset = 1
    for pred in plan.predictors
        shape = design_shape(pred, plan.columns; levelmaps = plan.levelmaps)
        labels = Symbol[]
        for b in shape.blocks
            append!(labels, b.labels)
        end
        push!(entries,
            LayoutEntry(:coefficient, pred.name, block_name(pred.name), labels,
                offset, shape.width, :identity))
        offset += shape.width
    end
    for p in plan.parameters
        transform, lo, hi = _entry_transform(p.family, p.support_override)
        push!(entries,
            LayoutEntry(:sampled, nothing, p.name, [p.name], offset, 1, transform,
                lo, hi))
        offset += 1
    end
    for p in plan.plate_parameters
        transform, lo, hi = _entry_transform(p.family, p.support_override)
        size = p.range === nothing ? plan.n_obs : length(p.range)
        push!(entries,
            LayoutEntry(:plate, nothing, p.name, [p.name], offset, size, transform,
                lo, hi))
        offset += size
    end
    for p in plan.vector_parameters
        p.size === nothing && throw(ContractValidationError(
            "[layout] vector parameter $(p.name) has unresolved size " *
            "(bind_data infers it from the linked response or " *
            "monotonic term)"))
        transform = p.family === :ordered_normal ? :ordered :
            p.family === :simplex_dirichlet ? :simplex : :identity
        # `size` is the PACKED (unconstrained) length: a simplex packs
        # K−1 stick-breaking logits (a 1-simplex packs zero — Stan's
        # deterministic `[1.0]`); threshold vectors pack their size.
        # Constrained length recovers as `_vector_constrained_size`.
        packed = transform === :simplex ? p.size - 1 : p.size
        labels = [Symbol(string(p.name) * "." * string(i)) for i in 1:packed]
        push!(entries,
            LayoutEntry(:vector, nothing, p.name, labels, offset, packed,
                transform))
        offset += packed
    end
    # Spline coefficient vectors: one contiguous block per SplineVector
    # (plate-shaped; priors broadcast over cells). Width is static from k.
    for v in plan.spline_vectors
        transform, lo, hi = _entry_transform(v.family, v.support_override)
        push!(entries,
            LayoutEntry(:spline, nothing, v.name, [v.name], offset, v.width,
                transform, lo, hi))
        offset += v.width
    end
    # Ranef buckets in plan order. K=1 (Stage B): the scalar scale
    # (`:sampled`, real for `log_scale`, exp for `tau` — the exp
    # Jacobian is Stan's lower-bound kernel term, no renormalizer)
    # plus the standardized G-vector `xi` (`:ranef`, plate-shaped
    # identity block; G from bind levels). Correlated (Stage C, SB
    # declaration order L/tau/z): the LKJ Cholesky factor (`:ranef_corr`
    # packing K*(K-1)/2 thetas — K=1 packs zero and constrains to
    # `[1.0]`), the marginal-scale K-vector `tau` (`:ranef` with `:exp`
    # — the same Stan kernel semantics as the Stage-B scalar), and the
    # standardized `z_flat` (`:ranef` identity, K*G column-major).
    for b in plan.ranef_buckets
        if b.kind === :correlated
            K = length(b.margins)
            L, tau, z = _ranef_corr_names(b)
            P = K * (K - 1) ÷ 2
            labels = [Symbol(string(L) * "." * string(i)) for i in 1:P]
            push!(entries,
                LayoutEntry(:ranef_corr, nothing, L, labels, offset, P,
                    :lkj))
            offset += P
            push!(entries,
                LayoutEntry(:ranef, nothing, tau, [tau], offset, K,
                    :exp))
            offset += K
            G = length(_grouping_levels(plan.columns[b.group]))
            push!(entries,
                LayoutEntry(:ranef, nothing, z, [z], offset, K * G,
                    :identity))
            offset += K * G
            continue
        end
        scale, xi = _ranef_k1_names(b)
        transform = b.kind === :intercept1 ? :identity : :exp
        push!(entries,
            LayoutEntry(:sampled, nothing, scale, [scale], offset, 1,
                transform))
        offset += 1
        G = length(_grouping_levels(plan.columns[b.group]))
        push!(entries,
            LayoutEntry(:ranef, nothing, xi, [xi], offset, G, :identity))
        offset += G
    end
    # Sequential-recurrence latents: one identity array slice per scan. The
    # length T is the loop bound — a literal Int, or `n_obs` when the bound is a
    # data length name (the canonical observation-indexed state-space case).
    # v1 latents have real support (identity transform ⇒ no Jacobian). A
    # centered slice holds the carried state itself; a non-centered slice
    # holds the iid innovations under the `_ppl_scan_z_<state>` name while
    # the state name binds the emitter's `scan(...)` reconstruction.
    for s in plan.scans
        T = s.hi isa Int ? s.hi : plan.n_obs
        T >= s.lo || throw(ContractValidationError(
            "[layout] scan $(s.state) length $(T) < loop start $(s.lo) — " *
            "the recurrence must run at least once"))
        labels = Symbol[Symbol(i) for i in 1:T]
        nm = _is_noncentered_scan(s) ? _scan_innovation_name(s) : s.state
        push!(entries,
            LayoutEntry(:scan, nothing, nm, labels, offset, T, :identity))
        offset += T
    end
    # HSGP bases in plan order, SB `_sb_hsgp` declaration order per basis
    # (rho, sigma, beta): length scales as `:sampled` scalars on the
    # parameterized `:floored` support (`x = lo + exp(u)`, logjac `u` —
    # Stan lower-bound kernel semantics, no truncation normalizer), the
    # marginal scale as a plain `:exp` scalar, and the standardized
    # M-vector `beta_raw` as one `:hsgp` identity block (the
    # spline-vector shape). A zero floor (K=1, unbounded) routes to
    # `:exp`, bit-identical to `:floored` at `lo == 0.0`.
    for hb in plan.hsgp_bases
        length(hb.fits) == length(hb.axes) || throw(ContractValidationError(
            "[layout] hsgp :$(hb.id): fits not filled at bind " *
            "(bind_data fills one (mu, L) per axis)"))
        names = _hsgp_names(hb)
        for (rho, fl) in zip(names.rhos, _hsgp_floors(hb.K, hb.fits, hb.iso))
            if fl == 0.0
                push!(entries, LayoutEntry(:sampled, nothing, rho, [rho],
                    offset, 1, :exp))
            else
                push!(entries, LayoutEntry(:sampled, nothing, rho, [rho],
                    offset, 1, :floored, fl, NaN))
            end
            offset += 1
        end
        push!(entries, LayoutEntry(:sampled, nothing, names.sigma,
            [names.sigma], offset, 1, :exp))
        offset += 1
        M = _hsgp_n_basis(hb)
        push!(entries, LayoutEntry(:hsgp, nothing, names.beta, [names.beta],
            offset, M, :identity))
        offset += M
    end
    return LayoutTable(entries, offset - 1)
end

"""Constrained length of a `:vector` entry: simplex packs K−1 logits
for K probabilities; threshold vectors pack their size."""
_vector_constrained_size(e::LayoutEntry) =
    e.transform === :simplex ? e.size + 1 : e.size

# Scalar logistic (Stan `inv_logit`) and logit, shared by the host
# simplex edges and their in-graph twins below.
_ordered_sigma(x::Float64) = 1.0 / (1.0 + exp(-x))
_simplex_logit(z::Float64) = log(z) - log1p(-z)

"""
    ordered_constrain(u) -> Vector{Float64}

Host-side ordered transform (Stan `ordered`): `y[1] = u[1]`,
`y[i] = y[i-1] + exp(u[i])`. The in-graph twin unrolls the identical
scalar chain.
"""
function ordered_constrain(u::AbstractVector{<:Real})
    y = Vector{Float64}(undef, length(u))
    for (i, x) in enumerate(u)
        y[i] = i == 1 ? Float64(x) : y[i-1] + exp(Float64(x))
    end
    return y
end

"""Inverse of [`ordered_constrain`](@ref)."""
function ordered_unconstrain(y::AbstractVector{<:Real})
    u = Vector{Float64}(undef, length(y))
    prev = 0.0
    for (i, v) in enumerate(y)
        u[i] = i == 1 ? Float64(v) : log(Float64(v) - prev)
        prev = Float64(v)
    end
    return u
end

"""Log-Jacobian of [`ordered_constrain`](@ref): `Σ u[2:end]`."""
function ordered_logjac(u::AbstractVector{<:Real})
    total = 0.0
    for (i, x) in enumerate(u)
        i > 1 && (total += Float64(x))
    end
    return total
end

"""
    simplex_constrain(u) -> Vector{Float64}

Host-side stick-breaking simplex transform (thin-layer-owned
parameterization — NOT Stan's isometric-log-ratio `simplex_constrain`;
the middle layer owns layout+transforms, the LKJ-factor precedent):
`z[j] = σ(u[j] + log(K-j))`, `s[j] = remaining[j]*z[j]`,
`s[K] = remaining[K]`. Empty input constrains to `[1.0]` (the
deterministic 1-simplex, as in Stan). The in-graph twin unrolls the
identical scalar chain.
"""
function simplex_constrain(u::AbstractVector{<:Real})
    K = length(u) + 1
    s = Vector{Float64}(undef, K)
    K == 1 && (s[1] = 1.0; return s)
    remaining = 1.0
    for (j, x) in enumerate(u)
        z = _ordered_sigma(Float64(x) + log(K - j))
        s[j] = remaining * z
        remaining -= s[j]
    end
    s[K] = remaining
    return s
end

"""Inverse of [`simplex_constrain`](@ref) (stick-breaking, not Stan's ILR
`simplex_free`)."""
function simplex_unconstrain(s::AbstractVector{<:Real})
    K = length(s)
    u = Vector{Float64}(undef, max(K - 1, 0))
    K <= 1 && return u
    remaining = 1.0
    for (j, v) in enumerate(s)
        j >= K && break
        z = Float64(v) / remaining
        u[j] = _simplex_logit(z) - log(K - j)
        remaining -= Float64(v)
    end
    return u
end

"""Log-Jacobian of [`simplex_constrain`](@ref):
`Σ [log(remaining) + log(z) + log1p(-z)]` over the K−1 breaks."""
function simplex_logjac(u::AbstractVector{<:Real})
    K = length(u) + 1
    K == 1 && return 0.0
    total = 0.0
    remaining = 1.0
    for (j, x) in enumerate(u)
        z = _ordered_sigma(Float64(x) + log(K - j))
        total += log(remaining) + log(z) + log1p(-z)
        remaining -= remaining * z
    end
    return total
end

"""
    _lkj_dim(packed) -> Int

Margin count K from a packed LKJ-theta length (`packed == K*(K-1)/2`;
0 ⟺ K=1). Loud on non-triangular lengths.
"""
function _lkj_dim(packed::Int)
    packed == 0 && return 1
    packed > 0 || throw(ContractValidationError(
        "[layout] LKJ packed length $packed is negative"))
    K = (1 + isqrt(1 + 8 * packed)) ÷ 2
    K * (K - 1) ÷ 2 == packed ||
        throw(ContractValidationError("[layout] LKJ packed length $packed " *
              "is not triangular (want K*(K-1)/2)"))
    return K
end

"""
    lkj_chol_constrain(u, K) -> Matrix{Float64}

Host-side LKJ Cholesky-factor transform (thin-layer-owned
parameterization — NOT Stan's partial-correlation vine; contract:
the middle layer owns layout+transforms). Row 1 is `[1, 0, ...]`;
row `i >= 2` is a unit vector from `i-1` logistic angles
`theta = pi*sigma(u)` packed row-major:
`L[i,j] = cos(theta_j) * prod(sin(theta[1:j-1]))`,
`L[i,i] = prod(sin(theta))`. Above-diagonal stays zero. The in-graph
twin unrolls the identical scalar chain (left-assoc products), so
host and graph agree bit-for-bit. K=1 constrains `[]` to `[1.0]`.
"""
function lkj_chol_constrain(u::AbstractVector{<:Real}, K::Int)
    K >= 1 || throw(ContractValidationError(
        "[layout] LKJ margin count K=$K < 1"))
    length(u) == K * (K - 1) ÷ 2 || throw(ContractValidationError(
        "[layout] LKJ packed length $(length(u)) ≠ $K*$(K-1)/2"))
    L = zeros(Float64, K, K)
    L[1, 1] = 1.0
    p = 0
    for i in 2:K
        th = Vector{Float64}(undef, i - 1)
        for j in 1:i-1
            p += 1
            s = 1.0 / (1.0 + exp(-Float64(u[p])))
            th[j] = pi * s
        end
        for j in 1:i-1
            v = cos(th[j])
            for m in 1:j-1
                v *= sin(th[m])
            end
            L[i, j] = v
        end
        d = 1.0
        for m in 1:i-1
            d *= sin(th[m])
        end
        L[i, i] = d
    end
    return L
end

"""Inverse of [`lkj_chol_constrain`](@ref): hyperspherical inversion
per row (`theta_j = atan(hypot(row[j+1:i]), row[j])`), then
`u = log(theta) - log(pi - theta)`. Loud outside the hemisphere
(non-positive diagonal) and on non-square input."""
function lkj_chol_unconstrain(L::AbstractMatrix{<:Real}, K::Int)
    size(L) == (K, K) || throw(ContractValidationError(
        "[layout] LKJ factor size $(size(L)) ≠ ($K, $K)"))
    u = Vector{Float64}(undef, K * (K - 1) ÷ 2)
    p = 0
    for i in 2:K
        Float64(L[i, i]) > 0 || throw(ContractValidationError(
            "[layout] LKJ factor row $i leaves the hemisphere " *
            "(non-positive diagonal — not a Cholesky factor)"))
        for j in 1:i-1
            p += 1
            rest = sqrt(sum(Float64(L[i, m])^2 for m in j+1:i))
            th = atan(rest, Float64(L[i, j]))
            u[p] = log(th) - log(pi - th)
        end
    end
    return u
end

"""Log-Jacobian of [`lkj_chol_constrain`](@ref): per-angle Gram factor
`(i-j)*log(sin theta)` (the correlation-matrix volume element the
Stan-verbatim `lkj_corr_cholesky_logpdf` is a density against — one
more log-sin per angle than the hyperspherical sphere-volume
exponent) + logistic `log(pi) + log(s) + log1p(-s)`, summed
row-major. The in-graph twin unrolls the identical sum over the
named theta/sigma temps, so host and graph agree bit-for-bit."""
function lkj_chol_logjac(u::AbstractVector{<:Real}, K::Int)
    length(u) == K * (K - 1) ÷ 2 || throw(ContractValidationError(
        "[layout] LKJ packed length $(length(u)) ≠ $K*$(K-1)/2"))
    total = 0.0
    p = 0
    for i in 2:K, j in 1:i-1
        p += 1
        x = Float64(u[p])
        s = 1.0 / (1.0 + exp(-x))
        th = pi * s
        total += (i - j) * log(sin(th)) + log(pi) + log(s) + log1p(-s)
    end
    return total
end

"""
    lkj_logconst(K, eta) -> Float64

LKJ normalizing constant: verbatim port of Stan's `do_lkj_constant`
(Lewandowski–Kurowicka–Joe 2009, theorem 5), INCLUDING the
`eta == 1.0` fast branch — the joint parity case compares against
Stan's `lkj_corr_cholesky_lpdf` with the same branch taken, so the
branch dispatch is load-bearing, not cosmetic. K=1 is ±0.0 (== 0.0
either way — the eta==1.0 branch yields -0.0, exactly as Stan does).
"""
function lkj_logconst(K::Int, eta::Real)
    e = Float64(eta)
    Km1 = K - 1
    if e == 1.0
        denom = 0.0
        for k in 1:Km1÷2
            denom += loggamma(2.0 * k)
        end
        constant = -denom
        if K % 2 == 1
            constant -= 0.25 * (K * K - 1) * log(pi) -
                0.25 * (Km1 * Km1) * log(2.0) -
                Km1 * loggamma(0.5 * (K + 1))
        else
            constant -= 0.25 * K * (K - 2) * log(pi) +
                0.25 * (3 * K * K - 4 * K) * log(2.0) +
                K * loggamma(0.5 * K) - Km1 * loggamma(Float64(K))
        end
        return constant
    end
    constant = Km1 * loggamma(e + 0.5 * Km1)
    for k in 1:Km1
        constant -= 0.5 * k * log(pi) + loggamma(e + 0.5 * (Km1 - k))
    end
    return constant
end

"""
    lkj_corr_cholesky_logpdf(L, eta) -> Float64

Host-side LKJ Cholesky log-density (Stan `lkj_corr_cholesky_lpdf`,
propto=false): diagonal sum over rows 2..K plus
[`lkj_logconst`](@ref). Stan's op order is preserved verbatim
(per-diagonal `(Km1-k-1)*ld + (2*eta-2)*ld`; the `eta == 1.0`
single-term branch) so joint parity against Stan holds to 1ulp.
K=1 is exactly 0.0.
"""
function lkj_corr_cholesky_logpdf(L::AbstractMatrix{<:Real}, eta::Real)
    K = size(L, 1)
    size(L, 2) == K || throw(ContractValidationError(
        "[layout] LKJ factor is not square: $(size(L))"))
    e = Float64(eta)
    lp = lkj_logconst(K, e)
    if e == 1.0
        for k in 0:K-2
            lp += (K - 1 - k - 1) * log(Float64(L[k+2, k+2]))
        end
        return lp
    end
    for k in 0:K-2
        ld = log(Float64(L[k+2, k+2]))
        lp += (K - 1 - k - 1) * ld + (2 * e - 2) * ld
    end
    return lp
end

"""
    coordinate_names(layout) -> Vector{Symbol}

Flat per-coordinate names (R10 read API): coefficients qualified
`Symbol("predictor.coef")`, sampled parameters bare. Length == total.
"""
function coordinate_names(layout::LayoutTable)
    names = Symbol[]
    for e in layout.entries
        if e.kind === :coefficient
            for label in e.labels
                push!(names, Symbol(string(e.predictor) * "." * string(label)))
            end
        elseif e.kind === :plate || e.kind === :spline || e.kind === :ranef ||
               e.kind === :hsgp
            for i in 1:e.size
                push!(names, Symbol(string(e.name) * "." * string(i)))
            end
        elseif e.kind === :vector
            append!(names, e.labels)
        elseif e.kind === :ranef_corr
            append!(names, e.labels)
        elseif e.kind === :scan
            for label in e.labels
                push!(names, Symbol(string(e.name) * "." * string(label)))
            end
        else
            push!(names, e.name)
        end
    end
    return names
end

"""
    constrain(layout, unconstrained) -> NamedTuple

Host-side constrain: packed vector → `(predictor => Vector, param => scalar,
…)`. A correlated-ranef bucket contributes its Cholesky factor `L` (K×K),
its scale vector `tau`, its standardized draws `z_flat`, plus the DERIVED
correlated draws `b_<suffix>` (G×K, SB `(diag_pre_multiply(tau,L)*z)'`
— output mapping, ignored by [`unconstrain`](@ref)). For testing,
output mapping, and future prediction; the generator emits the
in-graph equivalent.
"""
function constrain(layout::LayoutTable, u::AbstractVector{<:Real})
    length(u) == layout.total ||
        throw(ContractValidationError("[layout] unconstrained length $(length(u)) ≠ $(layout.total)"))
    pairs = Pair{Symbol,Any}[]
    for e in layout.entries
        seg = u[e.offset:(e.offset + e.size - 1)]
        if e.kind === :coefficient
            push!(pairs, e.predictor => Vector{Float64}(seg))
        elseif e.kind === :plate || e.kind === :spline || e.kind === :ranef ||
               e.kind === :hsgp
            v = [_constrain_elt(e, Float64(x)) for x in seg]
            push!(pairs, e.name => v)
        elseif e.kind === :scan
            push!(pairs, e.name => Vector{Float64}(seg))
        elseif e.kind === :vector
            push!(pairs, e.name => _vector_constrain(e, seg))
        elseif e.kind === :ranef_corr
            push!(pairs, e.name =>
                lkj_chol_constrain(Vector{Float64}(seg), _lkj_dim(e.size)))
        else
            v = _constrain_elt(e, Float64(only(seg)))
            push!(pairs, e.name => v)
        end
    end
    for (bname, b) in _ranef_corr_draws(layout, u)
        push!(pairs, bname => b)
    end
    return NamedTuple{Tuple(first.(pairs))}(Tuple(last.(pairs)))
end

# Derived correlated draws per `:ranef_corr` entry: `b_<suffix>` (G×K),
# SB `(diag_pre_multiply(tau,L)*z)'` with `z_flat` in SB column-major
# order. Sibling entries are found by the canonical `_ranef_corr_names`
# spelling (`L_<s>` → `tau_<s>` / `z_flat_<s>`), never by adjacency,
# so entry-order changes cannot miswire it.
function _ranef_corr_draws(layout::LayoutTable, u::AbstractVector{<:Real})
    out = Pair{Symbol,Matrix{Float64}}[]
    byname = Dict{Symbol,LayoutEntry}(e.name => e for e in layout.entries)
    for e in layout.entries
        e.kind === :ranef_corr || continue
        sfx = string(e.name)[3:end]
        tau_e = get(byname, Symbol("tau_", sfx), nothing)
        z_e = get(byname, Symbol("z_flat_", sfx), nothing)
        (tau_e === nothing || z_e === nothing) && throw(ContractValidationError(
            "[layout] LKJ entry $(e.name) has no tau/z_flat siblings " *
            "(assign_layout always emits the triple)"))
        K = _lkj_dim(e.size)
        tau = [_constrain_elt(tau_e, Float64(x)) for x in
            u[tau_e.offset:(tau_e.offset + tau_e.size - 1)]]
        zf = [Float64(x) for x in u[z_e.offset:(z_e.offset + z_e.size - 1)]]
        length(zf) == K * (length(zf) ÷ K) || throw(ContractValidationError(
            "[layout] z_flat length $(length(zf)) is not a multiple of K=$K"))
        G = length(zf) ÷ K
        L = lkj_chol_constrain(
            Vector{Float64}(u[e.offset:(e.offset + e.size - 1)]), K)
        zmat = reshape(zf, K, G)
        push!(out, Symbol("b_", sfx) => Matrix((Diagonal(tau) * L * zmat)'))
    end
    return out
end

"""
    unconstrain(layout, constrained) -> Vector{Float64}

Inverse of [`constrain`](@ref): named values → packed unconstrained vector.
"""
function unconstrain(layout::LayoutTable, nt::NamedTuple)
    u = Vector{Float64}(undef, layout.total)
    for e in layout.entries
        if e.kind === :coefficient
            haskey(nt, e.predictor) || throw(
                ContractValidationError("[layout] missing predictor $(e.predictor)"),
            )
            v = nt[e.predictor]
            length(v) == e.size || throw(
                ContractValidationError("[layout] predictor $(e.predictor) length mismatch"),
            )
            u[e.offset:(e.offset + e.size - 1)] .= Float64.(v)
        elseif e.kind === :plate || e.kind === :spline || e.kind === :ranef ||
               e.kind === :hsgp
            what = e.kind === :plate ? "plate parameter" :
                e.kind === :spline ? "spline vector" :
                e.kind === :ranef ? "ranef vector" : "hsgp vector"
            haskey(nt, e.name) || throw(
                ContractValidationError("[layout] missing $what $(e.name)"),
            )
            v = nt[e.name]
            length(v) == e.size || throw(
                ContractValidationError("[layout] $what $(e.name) length mismatch"),
            )
            for (k, x) in enumerate(v)
                u[e.offset + k - 1] = _unconstrain_elt(e, Float64(x))
            end
        elseif e.kind === :scan
            haskey(nt, e.name) || throw(
                ContractValidationError("[layout] missing scan state $(e.name)"),
            )
            v = nt[e.name]
            length(v) == e.size || throw(
                ContractValidationError("[layout] scan state $(e.name) length mismatch"),
            )
            u[e.offset:(e.offset + e.size - 1)] .= Float64.(v)
        elseif e.kind === :vector
            haskey(nt, e.name) || throw(
                ContractValidationError("[layout] missing vector parameter $(e.name)"),
            )
            v = nt[e.name]
            want = _vector_constrained_size(e)
            length(v) == want || throw(
                ContractValidationError("[layout] vector parameter $(e.name) length mismatch"),
            )
            u[e.offset:(e.offset + e.size - 1)] .= _vector_unconstrain(e, v)
        elseif e.kind === :ranef_corr
            haskey(nt, e.name) || throw(
                ContractValidationError("[layout] missing LKJ factor $(e.name)"),
            )
            v = nt[e.name]
            v isa AbstractMatrix || throw(
                ContractValidationError("[layout] LKJ factor $(e.name) " *
                      "must be a K×K matrix, got $(typeof(v))"),
            )
            u[e.offset:(e.offset + e.size - 1)] .=
                lkj_chol_unconstrain(v, _lkj_dim(e.size))
        else
            haskey(nt, e.name) || throw(
                ContractValidationError("[layout] missing parameter $(e.name)"),
            )
            u[e.offset] = _unconstrain_elt(e, Float64(nt[e.name]))
        end
    end
    return u
end

"""
    logjac(layout, unconstrained) -> Float64

Host-side total log-Jacobian, using the identical operations as the
in-graph terms (positive: the unconstrained value; unit:
`log(x) + log1p(-x)` from the constrained value).
"""
function logjac(layout::LayoutTable, u::AbstractVector{<:Real})
    length(u) == layout.total ||
        throw(ContractValidationError("[layout] unconstrained length $(length(u)) ≠ $(layout.total)"))
    total = 0.0
    for e in layout.entries
        e.transform === :identity && continue
        seg = u[e.offset:(e.offset + e.size - 1)]
        if e.kind === :vector
            # Vector Jacobians couple coordinates (ordered sums, simplex
            # stick-breaking) — entry-level, never per-coordinate.
            total += _vector_logjac(e, seg)
            continue
        end
        if e.kind === :ranef_corr
            # LKJ thetas couple through the hyperspherical rows —
            # entry-level, never per-coordinate.
            total += lkj_chol_logjac(Vector{Float64}(seg), _lkj_dim(e.size))
            continue
        end
        for v in seg
            total += _logjac_elt(e, Float64(v))
        end
    end
    return total
end

# Entry-level vector edges (host side). The in-graph twins in
# `_vector_transform_statements`/`jacobian_term` unroll the IDENTICAL
# scalar chains, so host and graph agree bit-for-bit.
_vector_constrain(e::LayoutEntry, seg) =
    e.transform === :identity ? Vector{Float64}(seg) :
    e.transform === :ordered ? ordered_constrain(seg) :
    simplex_constrain(seg)
_vector_unconstrain(e::LayoutEntry, v) =
    e.transform === :identity ? Vector{Float64}(v) :
    e.transform === :ordered ? ordered_unconstrain(v) :
    simplex_unconstrain(v)
_vector_logjac(e::LayoutEntry, seg) =
    e.transform === :ordered ? ordered_logjac(seg) : simplex_logjac(seg)

# Host transform values route through the bijector library (the single source
# of truth shared with the in-graph splices); `:identity` is a genuine no-op.
_constrain_value(transform::Symbol, u) =
    transform === :identity ? u : _prepared_endpoint(transform, :constrain)(u)
_unconstrain_value(transform::Symbol, x) =
    transform === :identity ? x : _prepared_endpoint(transform, :unconstrain)(x)
_logjac_value(transform::Symbol, u) =
    transform === :identity ? 0.0 : _prepared_endpoint(transform, :logjac)(u)

# Entry-level constrain/unconstrain/log-Jacobian. `:exp`/`:logistic` route
# through the bijector library (`_constrain_value(transform::Symbol, …)`, the
# single source of truth shared with the in-graph splices). An `:interval`
# transform is PARAMETERIZED by per-entry bounds, so it does not fit the
# Symbol-keyed parameterless bijector registry (like `:identity`, it lives
# outside it): it maps ℝ → (lo, hi) via an affine-logistic (`lo + (hi-lo)·σ(u)`)
# with log-Jacobian `log(x-lo) + log(hi-x) - log(hi-lo)` in the CONSTRAINED
# value. `:floored` is likewise parameterized (Stan's lower-bound kernel
# ℝ → (lo, ∞), `lo + exp(u)`, log-Jacobian the bare `u` — no truncation
# normalizer). The in-graph interval/floored edges (below) use the IDENTICAL
# operations, so host and graph agree bit-for-bit.
function _constrain_elt(e::LayoutEntry, u)
    e.transform === :interval &&
        return e.lo + (e.hi - e.lo) / (1 + exp(-u))
    e.transform === :floored && return e.lo + exp(u)
    return _constrain_value(e.transform, u)
end
function _unconstrain_elt(e::LayoutEntry, x)
    e.transform === :interval && return log(x - e.lo) - log(e.hi - x)
    e.transform === :floored && return log(x - e.lo)
    return _unconstrain_value(e.transform, x)
end
function _logjac_elt(e::LayoutEntry, u)
    e.transform === :interval || e.transform === :floored ||
        return _logjac_value(e.transform, u)
    e.transform === :floored && return u
    x = e.lo + (e.hi - e.lo) / (1 + exp(-u))
    return log(x - e.lo) + log(e.hi - x) - log(e.hi - e.lo)
end

"""
    coordinate_read(offset) -> Expr

Scalar packed-coordinate read (`sum(view(unconstrained, o:o))`,
allocation-free, `@ppl` shape).
"""
coordinate_read(offset::Int) = :(sum(view(unconstrained, $offset:$offset)))

"""
    block_read(offset, width) -> Expr

Packed-slice read (`view(unconstrained, lo:hi)`, `@ppl` vector shape).
"""
block_read(offset::Int, width::Int) =
    :(view(unconstrained, $offset:$(offset + width - 1)))

"""
    transform_statements(entry) -> Vector{Expr}

In-graph constrain edge(s) for one layout entry. Coefficient/scan slices read a
`view`; a `:identity` sampled entry reads its scalar coordinate; a constrained
sampled entry (`:exp`/`:logistic`) splices the bijector's `constrain` endpoint,
which the planner inlines.
"""
function transform_statements(e::LayoutEntry)
    if e.kind === :coefficient || e.kind === :scan
        # both are identity array slices read into `e.name` (a coefficient
        # block name, a scan-state slice, or a non-centered scan's
        # `_ppl_scan_z_<state>` innovation slice the emitter reconstructs from)
        lo = e.offset
        hi = e.offset + e.size - 1
        return Expr[:($(e.name)::AbstractVector{Float64} =
            view(unconstrained, $lo:$hi))]
    end
    if e.kind === :plate || e.kind === :spline || e.kind === :ranef ||
       e.kind === :hsgp
        # Spline vectors ride the plate transform path (block + scalar
        # endpoints); the contract pins their supports to real/positive,
        # so the :interval arm below is unreachable for them. Ranef
        # vectors ride it too (`xi`/`z_flat` identity, `tau` exp), as do
        # HSGP coefficient vectors (`beta_raw`, identity only).
        return _plate_transform_statements(e)
    end
    if e.kind === :vector
        return _vector_transform_statements(e)
    end
    if e.kind === :ranef_corr
        return _ranef_corr_transform_statements(e)
    end
    o = e.offset
    coord = coordinate_read(o)
    e.transform === :identity && return Expr[:($(e.name)::Float64 = $coord)]
    if e.transform === :interval
        # An :interval transform is parameterized by per-entry bounds, so it is
        # not in the Symbol-keyed (parameterless) bijector registry — its
        # forward + inverse edges are hand-rolled here, with the IDENTICAL math
        # to the host `_*_elt` path so in-graph and host agree bit-for-bit.
        u = Symbol(:_ppl_int_, e.name)
        blo, bhi = e.lo, e.hi
        return Expr[
            :($u::Float64 = $coord),
            :($(e.name)::Float64 = $blo + ($bhi - $blo) / (1 + exp(-$u))),
            :($u::Float64 = log($(e.name) - $blo) - log($bhi - $(e.name))),
        ]
    end
    if e.transform === :floored
        # Parameterized like :interval (the per-entry floor is not in the
        # registry): forward + inverse edges hand-rolled with the IDENTICAL
        # math to the host `_*_elt` path (`lo + exp(u)` / `log(x - lo)`).
        u = Symbol(:_ppl_fl_, e.name)
        blo = e.lo
        return Expr[
            :($u::Float64 = $coord),
            :($(e.name)::Float64 = $blo + exp($u)),
            :($u::Float64 = log($(e.name) - $blo)),
        ]
    end
    # Constrained supports splice the bijector's `constrain` endpoint; the
    # planner inlines it (no runtime call survives) and shares `coord` with the
    # Jacobian term via structural CSE.
    bij = _bijector_name(e.transform)
    return Expr[:($(e.name)::Float64 = $(bij)().constrain($coord))]
end

# The per-cell log-Jacobian vector name a plate block's `jacobian_term` sums.
_plate_logjac_name(name::Symbol) = Symbol(:_ppl_ljcells_, name)

# `plate(view) do cell; body; end` as an Expr (an allocation-free per-cell loop).
function _plate_map(view_read, cell::Symbol, body)
    lambda = Expr(:(->), Expr(:tuple, cell),
        Expr(:block, LineNumberNode(0, :layout), body))
    return Expr(:do, Expr(:call, :plate, view_read), lambda)
end

# Per-cell latent (plate) block: map the SCALAR bijector endpoints over the
# block view via the `plate` primitive — the same library the scalar and host
# paths use, so in-graph and host agree by construction rather than by a
# hand-kept broadcast. `constrain` yields the constrained cell vector; a
# companion `logjac` plate supplies the per-cell Jacobian this block's
# `jacobian_term` sums (pruned by have→want when the Jacobian is not wanted).
# `:identity` (real support) is a direct view read — no transform, no Jacobian.
function _plate_transform_statements(e::LayoutEntry)
    lo = e.offset
    hi = e.offset + e.size - 1
    view_read = :(view(unconstrained, $lo:$hi))
    e.transform === :identity &&
        return Expr[:($(e.name)::AbstractVector{Float64} = $view_read)]
    if e.transform === :interval
        # Parameterized bounds ⇒ not in the (parameterless) bijector registry;
        # hand-rolled broadcast edges over the block view, identical math to the
        # host `_*_elt` path so in-graph and host agree bit-for-bit. Its
        # `jacobian_term` sums the same expression (below), so no companion
        # `logjac` plate is emitted.
        u = Symbol(:_ppl_int_, e.name)
        blo, bhi = e.lo, e.hi
        return Expr[
            :($u::AbstractVector{Float64} = $view_read),
            :($(e.name)::AbstractVector{Float64} =
                $blo .+ ($bhi - $blo) ./ (1 .+ exp.(-$u))),
            :($u::AbstractVector{Float64} =
                log.($(e.name) .- $blo) .- log.($bhi .- $(e.name))),
        ]
    end
    bij = _bijector_name(e.transform)
    cell = Symbol(:_ppl_cell_, e.name)
    ljcells = _plate_logjac_name(e.name)
    return Expr[
        :($(e.name)::AbstractVector{Float64} =
            $(_plate_map(view_read, cell, :($(bij)().constrain($cell))))),
        :($(ljcells)::AbstractVector{Float64} =
            $(_plate_map(view_read, cell, :($(bij)().logjac($cell))))),
    ]
end

# Constrained-element / stick-breaking-temporary names for a `:vector`
# entry: `_ppl_v_<name>_<i>` (elements, shared with the generator's prior
# and likelihood cells), `_ppl_vz_<name>_<j>` (break fractions),
# `_ppl_vr_<name>_<j>` (remainders). All `_ppl_`-hygienic.
_vector_elt_name(name::Symbol, i::Int) = Symbol(:_ppl_v_, name, :_, i)
_vector_elt(e::LayoutEntry, i::Int) = _vector_elt_name(e.name, i)
_vector_z(e::LayoutEntry, j::Int) = Symbol(:_ppl_vz_, e.name, :_, j)
_vector_r(e::LayoutEntry, j::Int) = Symbol(:_ppl_vr_, e.name, :_, j)

# Leveled vector edges: scalar-unrolled twins of the host
# `ordered_constrain`/`simplex_constrain` (IDENTICAL scalar ops in the
# IDENTICAL order, so in-graph and host agree bit-for-bit). No vector
# ever materializes in-graph: downstream prior/likelihood cells read the
# `_ppl_v_` scalars directly (fully transparent to the planner and the
# reverse pass — no new Enzyme surface). Empty threshold vectors emit no
# statements; a 1-simplex emits its constant `[1.0]`.
function _vector_transform_statements(e::LayoutEntry)
    if e.transform === :identity
        return Expr[:($(_vector_elt(e, i))::Float64 =
            $(coordinate_read(e.offset + i - 1))) for i in 1:e.size]
    end
    if e.transform === :ordered
        stmts = Expr[]
        for i in 1:e.size
            t = _vector_elt(e, i)
            if i == 1
                push!(stmts, :($t::Float64 = $(coordinate_read(e.offset))))
            else
                prev = _vector_elt(e, i - 1)
                push!(stmts, :($t::Float64 =
                    $prev + exp($(coordinate_read(e.offset + i - 1)))))
            end
        end
        return stmts
    end
    # :simplex — stick-breaking, unrolled (thin-layer-owned, not Stan's ILR).
    K = e.size + 1
    stmts = Expr[]
    K == 1 && return Expr[:($(_vector_elt(e, 1))::Float64 = 1.0)]
    push!(stmts, :($(_vector_r(e, 1))::Float64 = 1.0))
    for j in 1:K-1
        z = _vector_z(e, j)
        s = _vector_elt(e, j)
        r = _vector_r(e, j)
        logK = log(K - j)
        push!(stmts, :($z::Float64 =
            1.0 / (1.0 + exp(-($(coordinate_read(e.offset + j - 1)) + $logK)))))
        push!(stmts, :($s::Float64 = $r * $z))
        push!(stmts, :($(_vector_r(e, j + 1))::Float64 = $r - $s))
    end
    push!(stmts, :($(_vector_elt(e, K))::Float64 = $(_vector_r(e, K))))
    return stmts
end

# Constrained Cholesky entries / logistic-sigma + theta temps for a
# `:ranef_corr` entry: `_ppl_rl_<L>_<i>_<j>` (lower triangle incl.
# diagonal; downstream gather/prior cells read these scalars directly),
# `_ppl_rsg_<L>_<i>_<j>` (sigma(u)), `_ppl_rth_<L>_<i>_<j>` (theta).
# All `_ppl_`-hygienic.
_rl_name(L::Symbol, i::Int, j::Int) = Symbol(:_ppl_rl_, L, :_, i, :_, j)
_rsg_name(L::Symbol, i::Int, j::Int) = Symbol(:_ppl_rsg_, L, :_, i, :_, j)
_rth_name(L::Symbol, i::Int, j::Int) = Symbol(:_ppl_rth_, L, :_, i, :_, j)

# LKJ Cholesky edges: scalar-unrolled twin of the host
# `lkj_chol_constrain` (IDENTICAL scalar ops in the IDENTICAL order —
# left-assoc products, `pi`/`log(pi)` precomputed host-side literals —
# so in-graph and host agree bit-for-bit). No matrix ever
# materializes: the gather reads the `_ppl_rl_` scalars directly
# (fully transparent to the planner and the reverse pass — no new
# Enzyme surface). K=1 emits its constant `[1.0]` edge only.
function _ranef_corr_transform_statements(e::LayoutEntry)
    K = _lkj_dim(e.size)
    L = e.name
    stmts = Expr[:($(_rl_name(L, 1, 1))::Float64 = 1.0)]
    PI = Float64(pi)
    p = 0
    for i in 2:K
        for j in 1:i-1
            p += 1
            coord = coordinate_read(e.offset + p - 1)
            s = _rsg_name(L, i, j)
            t = _rth_name(L, i, j)
            push!(stmts, :($s::Float64 = 1.0 / (1.0 + exp(-$coord))))
            push!(stmts, :($t::Float64 = $PI * $s))
        end
        for j in 1:i-1
            factors = Any[:(cos($(_rth_name(L, i, j))))]
            for m in 1:j-1
                push!(factors, :(sin($(_rth_name(L, i, m)))))
            end
            prod = foldl((a, b) -> :($a * $b), factors)
            push!(stmts, :($(_rl_name(L, i, j))::Float64 = $prod))
        end
        dfactors = Any[1.0]
        for m in 1:i-1
            push!(dfactors, :(sin($(_rth_name(L, i, m)))))
        end
        dprod = foldl((a, b) -> :($a * $b), dfactors)
        push!(stmts, :($(_rl_name(L, i, i))::Float64 = $dprod))
    end
    return stmts
end

"""
    jacobian_term(entry) -> Union{Nothing,Expr}

This entry's log-Jacobian contribution (`nothing` for identity). A scalar
constrained support splices the bijector's `logjac` endpoint over the same
coordinate the `constrain` edge reads (shared via structural CSE); a per-cell
latent (plate) block sums its companion `logjac` plate (`_plate_logjac_name`);
a leveled vector entry sums its unrolled twin of the host
`ordered_logjac`/`simplex_logjac` (shared coordinates via CSE); an LKJ
entry sums its unrolled twin of the host `lkj_chol_logjac` (named
theta/sigma temps, shared with the constrain edges).
"""
function jacobian_term(e::LayoutEntry)
    e.transform === :identity && return nothing
    if e.kind === :coefficient
        throw(ContractValidationError("[layout] non-identity coefficient block"))
    end
    if e.kind === :ranef_corr
        K = _lkj_dim(e.size)
        K == 1 && return nothing
        L = e.name
        LOGPI = log(Float64(pi))
        terms = Any[]
        for i in 2:K, j in 1:i-1
            s = _rsg_name(L, i, j)
            t = _rth_name(L, i, j)
            push!(terms, :($(i - j) * log(sin($t)) + $LOGPI + log($s) +
                log1p(-$s)))
        end
        return foldl((a, b) -> :($a + $b), terms)
    end
    if e.kind === :vector
        if e.transform === :ordered
            e.size < 2 && return nothing
            terms = Any[coordinate_read(e.offset + i - 1) for i in 2:e.size]
            return foldl((a, b) -> :($a + $b), terms)
        end
        # :simplex — Σ [log(r) + log(z) + log1p(-z)] over the
        # breaks, reading the `_vector_transform_statements` temps.
        e.size < 1 && return nothing
        terms = Any[:(log($(_vector_r(e, j))) + log($(_vector_z(e, j))) +
            log1p(-$(_vector_z(e, j)))) for j in 1:e.size]
        return foldl((a, b) -> :($a + $b), terms)
    end
    if e.kind === :plate || e.kind === :spline || e.kind === :ranef ||
       e.kind === :hsgp
        # Interval plates hand-roll the per-cell Jacobian sum (parameterized
        # bounds, no companion `logjac` plate); every registry transform sums
        # its companion `logjac` plate from `_plate_transform_statements`.
        # Spline vectors share the shape (their supports never reach
        # :interval, but the arm stays correct if that ever changes), as do
        # ranef vectors (`xi`/`z_flat` identity, `tau` exp) and hsgp
        # vectors (`beta_raw` identity). Anything else is loud (a companion
        # plate that was never emitted must never sum silently).
        if e.transform === :interval
            blo, bhi = e.lo, e.hi
            return :(sum(log.($(e.name) .- $blo) .+ log.($bhi .- $(e.name)) .-
                         log($bhi - $blo)))
        end
        e.transform === :exp || e.transform === :logistic ||
            throw(ContractValidationError("[layout] $(e.kind) block " *
                  "$(e.name) has no Jacobian rule for transform " *
                  "$(e.transform)"))
        return :(sum($(_plate_logjac_name(e.name))))
    end
    if e.transform === :interval
        # Parameterized bounds ⇒ hand-rolled (not in the bijector registry);
        # uses the constrained `e.name` from the interval constrain edge.
        blo, bhi = e.lo, e.hi
        return :(log($(e.name) - $blo) + log($bhi - $(e.name)) - log($bhi - $blo))
    end
    if e.transform === :floored
        # Stan's lower-bound kernel: the bare unconstrained coordinate
        # (shared with the constrain edge via structural CSE) — no
        # truncation normalizer.
        return coordinate_read(e.offset)
    end
    bij = _bijector_name(e.transform)
    return :($(bij)().logjac($(coordinate_read(e.offset))))
end
