# A design with one group-specific coefficient and any shared columns has
# disjoint local columns. Project each shared column and the response off the
# local column in its group, then QR-factor the common residual design once:
#
#   RSS = shared_RSS(theta) +
#         sum_g (norms[g]*(alpha[g]-alpha_hat[g]) + cross[g,:]'*(theta-theta_hat))^2
#
# norms holds local column norms. Each row of cross is the shared design's
# projection on that unit column. Group IDs and column counts stay data.
struct _LSMixedStatistics
    shared::_LSStatistics
    groups::Vector{Int}
    min_group::Int
    max_group::Int
    consecutive::Bool
    norms::Vector{Float64}
    cross::Matrix{Float64}
    hi::Vector{Float64}
    lo::Vector{Float64}
    rows::Int
    usable::Bool
end

function _ls_mixed_unusable(n)
    _LSMixedStatistics(_ls_unusable(0, 0, Int[], n), Int[], 0, -1, false, Float64[],
        zeros(0, 0), Float64[], Float64[], n, false)
end

# Compensated division. The numerator and denominator are dot products of
# original columns, and their ratio is reused for every row in the group.
@inline function _ls_ratio(ah, al, bh, bl)
    q = ah / bh
    ph, pl = _ls_two_prod(q, bh)
    rh, rl = _ls_dd_add(ah, al, -ph, -(pl + q*bl))
    _ls_two_sum(q, (rh + rl) / bh)
end

function _ls_statistics_mixed(index, local_column, shared_design, response, scales)
    n, k = size(shared_design)
    groups = Int[]
    position = Dict{Int,Int}()
    rowgroup = Vector{Int}(undef, n)
    for (i, g) in enumerate(index)
        rowgroup[i] = get!(position, g) do
            push!(groups, g)
            length(groups)
        end
    end
    ng = length(groups)
    n >= _LS_MINIMUM_ROWS_PER_PARAMETER*(ng + k) && k > 0 || return _ls_mixed_unusable(n)
    first_group, last_group = extrema(groups)
    consecutive = widen(last_group) - first_group == ng - 1
    if consecutive
        groups .= first_group:last_group
        for i in eachindex(index)
            rowgroup[i] = index[i] - first_group + 1
        end
    end
    # Last two columns contain the response and the local column itself.
    columns = hcat(shared_design, response, local_column)
    hi = zeros(k + 2, ng)
    lo = zero(hi)
    for i in 1:n
        g = rowgroup[i]
        w = scales === nothing ? 1.0 : scales[i]
        t = local_column[i] / w
        for c in axes(columns, 2)
            ph, pl = _ls_two_prod(t, columns[i, c] / w)
            hi[c, g], lo[c, g] = _ls_dd_add(hi[c, g], lo[c, g], ph, pl)
        end
    end
    norm2 = hi[end, :] .+ lo[end, :]
    all(x -> isfinite(x) && x >= floatmin(Float64), norm2) || return _ls_mixed_unusable(n)
    for g in 1:ng, c in 1:k+1
        hi[c, g], lo[c, g] = _ls_ratio(hi[c, g], lo[c, g], hi[end, g], lo[end, g])
    end
    residual = Matrix{Float64}(undef, n, k + 1)
    max_column = 0.0
    for i in 1:n
        g = rowgroup[i]
        w = scales === nothing ? 1.0 : scales[i]
        for c in 1:k+1
            v = columns[i, c]
            c <= k && (max_column = max(max_column, abs(v / w)))
            ph, pl = _ls_two_prod(local_column[i], hi[c, g])
            rh, rl = _ls_dd_add(v, 0.0, -ph, -(pl + local_column[i]*lo[c, g]))
            residual[i, c] = rh + rl
        end
    end
    shared = _ls_statistics_dense(view(residual, :, 1:k), view(residual, :, k+1), scales)
    shared.usable || return _ls_mixed_unusable(n)
    # A shared column in the span of the grouped ones is a null direction
    # even if projecting it left a tiny, apparently full-rank residual.
    minimum(svdvals(view(shared.factor, :, :, 1))) > n*eps(Float64)*max_column ||
        return _ls_mixed_unusable(n)
    local_hi = copy(hi[k+1, :])
    local_lo = copy(lo[k+1, :])
    for g in 1:ng, c in 1:k
        ph, pl = _ls_two_prod(hi[c, g], shared.hi[c, 1])
        pl += hi[c, g]*shared.lo[c, 1] + lo[c, g]*shared.hi[c, 1]
        local_hi[g], local_lo[g] = _ls_dd_add(local_hi[g], local_lo[g], -ph, -pl)
    end
    all(isfinite, local_hi) && all(isfinite, local_lo) || return _ls_mixed_unusable(n)
    norms = sqrt.(norm2)
    cross = [norms[g] * hi[c, g] for g in 1:ng, c in 1:k]
    all(isfinite, cross) || return _ls_mixed_unusable(n)
    _LSMixedStatistics(shared, groups, first_group, last_group, consecutive,
        norms, cross, local_hi, local_lo, n, true)
end

@inline function _ls_mixed_delta(s, shared, c)
    (_ls_dense_entry(shared, c) - s.shared.hi[c, 1]) - s.shared.lo[c, 1]
end
function _ls_rss_mixed(s::_LSMixedStatistics, local_source, shared::Tuple)
    total = _ls_rss_dense(s.shared, shared)
    checkbounds(local_source, s.min_group)
    checkbounds(local_source, s.max_group)
    if s.consecutive
        return total + _ls_mixed_group_sum(s, local_source, s.min_group:s.max_group, shared)
    end
    total + _ls_mixed_group_sum(s, local_source, s.groups, shared)
end

function _ls_mixed_group_sum(s, local_source, groups, shared::Tuple{Float64})
    slope = _ls_mixed_delta(s, shared, 1)
    total = 0.0
    @inbounds @simd for g in eachindex(groups)
        d = s.norms[g] * ((local_source[groups[g]] - s.hi[g]) - s.lo[g])
        d = muladd(s.cross[g, 1], slope, d)
        total += d*d
    end
    total
end

@inline function _ls_group_update(d::NTuple{L,Float64}, cross, slope, g, c) where {L}
    ntuple(i -> muladd(@inbounds(cross[g + i - 1, c]), slope, d[i]), Val(L))
end

# Keep a cache line of local residuals in registers. Each shared coordinate
# updates adjacent levels. The last partial line uses the same sum of squares
# one level at a time. Both the group count and the column count remain data.
@inline function _ls_group_tile(s, local_source, groups, shared, start, ::Val{L}) where {L}
    d = ntuple(Val(L)) do i
        g = start + i - 1
        @inbounds s.norms[g] * ((local_source[groups[g]] - s.hi[g]) - s.lo[g])
    end
    for c in axes(s.cross, 2)
        d = _ls_group_update(d, s.cross, _ls_mixed_delta(s, shared, c), start, c)
    end
    sum(abs2, d)
end

function _ls_mixed_group_sum(s, local_source, groups, shared::Tuple)
    width = 64 ÷ sizeof(Float64)
    g = 1
    total = 0.0
    while g <= length(groups) - width + 1
        total += _ls_group_tile(s, local_source, groups, shared, g, Val(width))
        g += width
    end
    for j in g:length(groups)
        total += _ls_group_tile(s, local_source, groups, shared, j, Val(1))
    end
    total
end
