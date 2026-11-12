# Least-squares statistics of bound data for a summed squared residual (snag
# `bound-data-suffi-7c8369bd`; user decision
# `ReactiveKernels:snag.bound-data-suffi-7c8369bd/decisions/2026-11-12T12-04-45-460-1gq53h7`,
# option 1: on by default).
#
# A summed plate whose cells add `(b[i] - a[i]' * θ)^2`, with the response `b`
# and the coefficients `a` bound data and `θ` live values, adds a quadratic
# polynomial in `θ` whose coefficients are sums over the bound rows alone. Its
# sum is, for any least-squares solution `θ̂` of `A * θ ≈ b`,
#
#     sum_i (b[i] - a[i]' * θ)^2 = rss₀ + ‖R * (θ - θ̂)‖²,     R' * R = A' * A,
#
# with `rss₀` the minimum residual sum of squares. Binding computes `R`, `θ̂` and
# `rss₀` once; an evaluation then costs O(k²) per block instead of O(rows * k),
# in the value and in its ordinary reverse.
#
# Precision. Expanding the sum as `b'b - 2θ'A'b + θ'A'Aθ` subtracts large,
# nearly equal numbers. The form above does not, but `θ - θ̂` does, so `θ̂` is
# carried as an unevaluated sum of two doubles (`hi + lo`), refined against the
# residual in compensated arithmetic, and `θ - θ̂` is evaluated as
# `(θ - hi) - lo`. Measured against a 256-bit row-wise sum (decision `1gq53h7`,
# precision table), this keeps the value and gradient at least as accurate as
# the row-wise plate, and more accurate when the response is far from zero,
# where the row-wise sum itself loses digits.
#
# Blocks. A location read through a bound matrix (`X * beta`, plus live scalars
# with bound per-row coefficients) is one block over every row. A location
# whose live values are gathered through one bound index (`alpha[g] .+ f .*
# beta[g]`) is one block per index value present in the data: `θ` of a block
# is the gathered entries at that value. Rows need not be sorted or contiguous.
#
# Admission. The statistics are usable only when every bound entry is finite
# and they are cheaper than the rows (`_LS_MINIMUM_ROWS_PER_PARAMETER`). An
# unusable binding stores `usable = false` and `rss₀ = NaN`, with a zero factor,
# so an evaluation returns NaN with finite intermediate values: a caller that
# falls back to its own loop on a non-finite result differentiates a zero
# adjoint through finite values only. Whether to use them is decided by the
# data at binding; the compiled program is the same for every binding.

# --- compensated arithmetic --------------------------------------------------

@inline function _ls_two_sum(a::Float64, b::Float64)
    s = a + b
    v = s - a
    (s, (a - (s - v)) + (b - v))
end
@inline _ls_two_prod(a::Float64, b::Float64) = (p = a * b; (p, fma(a, b, -p)))
@inline function _ls_dd_add(ah::Float64, al::Float64, bh::Float64, bl::Float64)
    s, e = _ls_two_sum(ah, bh)
    e += al + bl
    _ls_two_sum(s, e)
end

# `b[i] - A[i, :]' * (hi + lo)` as a double-double.
@inline function _ls_residual(A::AbstractMatrix{Float64}, b::AbstractVector{Float64},
                              hi::AbstractVector{Float64}, lo::AbstractVector{Float64}, i::Int)
    rh, rl = b[i], 0.0
    @inbounds for j in eachindex(hi)
        ph, pl = _ls_two_prod(A[i, j], hi[j])
        pl += A[i, j] * lo[j]
        rh, rl = _ls_dd_add(rh, rl, -ph, -pl)
    end
    (rh, rl)
end

# --- statistics --------------------------------------------------------------

# Mean rows per block parameter below which a binding keeps its rows: the
# statistics cost about k²/2 multiply-adds per block, the rows about k per row.
const _LS_MINIMUM_ROWS_PER_PARAMETER = 2
# Refinement passes of θ̂ against its compensated residual.
const _LS_REFINEMENTS = 3

"""
    _LSStatistics

Bound-data least-squares statistics of `blocks` blocks of `k` parameters.
Block `j`'s factor is `factor[:, :, j]` (upper triangular `R_p` of a
column-pivoted QR, `A[:, perm] = Q * R_p`), its pivot `perm[:, j]`, its
solution `hi[:, j] + lo[:, j]`. `groups[j]` is the index value block `j`
gathers its parameters at (empty for one dense block). `rss0` is the minimum
residual sum of squares over every block, `rows` the bound row count.
"""
struct _LSStatistics
    k::Int
    blocks::Int
    factor::Array{Float64,3}
    perm::Matrix{Int}
    hi::Matrix{Float64}
    lo::Matrix{Float64}
    groups::Vector{Int}
    rss0::Float64
    rows::Int
    usable::Bool
end

function _ls_unusable(k::Int, blocks::Int, groups::Vector{Int}, rows::Int)
    perm = repeat(collect(1:k), 1, blocks)
    _LSStatistics(k, blocks, zeros(k, k, blocks), perm, zeros(k, blocks),
        zeros(k, blocks), groups, NaN, rows, false)
end

# One block: fill block `j` from the rows of `A` and `b`; return its rss₀ as a
# double-double (hi, lo) pair.
function _ls_block!(factor::Array{Float64,3}, perm::Matrix{Int}, hi::Matrix{Float64},
                    lo::Matrix{Float64}, j::Int, A::Matrix{Float64}, b::Vector{Float64})
    n, k = size(A)
    F = qr(A, ColumnNorm())
    R = F.R
    @inbounds for c in 1:k, r in 1:min(n, k, c)
        factor[r, c, j] = R[r, c]
    end
    perm[:, j] .= F.p
    θh = F \ b
    θl = zeros(k)
    r = Vector{Float64}(undef, n)
    for _ in 1:_LS_REFINEMENTS
        for i in 1:n
            rh, rl = _ls_residual(A, b, θh, θl, i)
            r[i] = rh + rl
        end
        δ = F \ r
        for c in 1:k
            θh[c], θl[c] = _ls_dd_add(θh[c], θl[c], δ[c], 0.0)
        end
    end
    hi[:, j] .= θh
    lo[:, j] .= θl
    # Σ (rh + rl)², compensated.
    sh, sl = 0.0, 0.0
    for i in 1:n
        rh, rl = _ls_residual(A, b, θh, θl, i)
        ph, pl = _ls_two_prod(rh, rh)
        pl += 2 * rh * rl
        sh, sl = _ls_dd_add(sh, sl, ph, pl)
    end
    (sh, sl)
end

_ls_finite(xs) = all(isfinite, xs)

"""
    _ls_statistics_dense(A, b)

Statistics of one block over every row: `A` is the bound design (one column
per live scalar or vector entry, in the order the evaluation reads them), `b`
the bound response.
"""
function _ls_statistics_dense(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real})
    n, k = size(A)
    length(b) == n || throw(DimensionMismatch(
        "least-squares statistics: $(n) design rows, $(length(b)) responses"))
    Af = Matrix{Float64}(A)
    bf = Vector{Float64}(b)
    usable = k >= 1 && n >= _LS_MINIMUM_ROWS_PER_PARAMETER * k &&
        _ls_finite(Af) && _ls_finite(bf)
    usable || return _ls_unusable(k, 1, Int[], n)
    factor = zeros(k, k, 1)
    perm = Matrix{Int}(undef, k, 1)
    hi = zeros(k, 1)
    lo = zeros(k, 1)
    sh, sl = _ls_block!(factor, perm, hi, lo, 1, Af, bf)
    _LSStatistics(k, 1, factor, perm, hi, lo, Int[], sh + sl, n, true)
end

"""
    _ls_statistics_grouped(index, columns, b)

Statistics of one block per value of the bound `index` present in the data.
`columns` holds one bound per-row coefficient vector per gathered live source
(`ones` for a source added as is), `b` the bound response.
"""
function _ls_statistics_grouped(index::AbstractVector{<:Integer},
                                columns::Tuple{Vararg{AbstractVector{<:Real}}},
                                b::AbstractVector{<:Real})
    n = length(b)
    k = length(columns)
    length(index) == n && all(c -> length(c) == n, columns) || throw(DimensionMismatch(
        "least-squares statistics: index, coefficient columns and responses differ in length"))
    groups = sort!(unique(index))
    blocks = length(groups)
    usable = k >= 1 && n >= _LS_MINIMUM_ROWS_PER_PARAMETER * k * blocks &&
        all(_ls_finite, columns) && _ls_finite(b)
    usable || return _ls_unusable(k, blocks, Vector{Int}(groups), n)
    # Rows of each group, by a counting pass over the sorted group values.
    position = Dict{Int,Int}(g => j for (j, g) in enumerate(groups))
    counts = zeros(Int, blocks)
    for g in index
        counts[position[g]] += 1
    end
    starts = cumsum(vcat(1, counts))
    order = Vector{Int}(undef, n)
    filled = copy(starts)
    for (i, g) in enumerate(index)
        j = position[g]
        order[filled[j]] = i
        filled[j] += 1
    end
    factor = zeros(k, k, blocks)
    perm = Matrix{Int}(undef, k, blocks)
    hi = zeros(k, blocks)
    lo = zeros(k, blocks)
    sh, sl = 0.0, 0.0
    for j in 1:blocks
        rows = view(order, starts[j]:starts[j+1]-1)
        A = Matrix{Float64}(undef, length(rows), k)
        for (c, column) in enumerate(columns), (r, i) in enumerate(rows)
            A[r, c] = column[i]
        end
        bj = Float64[b[i] for i in rows]
        bh, bl = _ls_block!(factor, perm, hi, lo, j, A, bj)
        sh, sl = _ls_dd_add(sh, sl, bh, bl)
    end
    _LSStatistics(k, blocks, factor, perm, hi, lo, Vector{Int}(groups), sh + sl, n, true)
end

# --- evaluation --------------------------------------------------------------

# Entry `j` of a tuple of scalars followed by a vector.
@inline _ls_dense_entry(::Tuple{}, v, j) = v[j]
@inline _ls_dense_entry(s::Tuple, v, j) =
    j == 1 ? first(s) : _ls_dense_entry(Base.tail(s), v, j - 1)
# Entry `g` of source `j` of a tuple of gathered sources.
@inline _ls_group_entry(s::Tuple{Any}, j, g) = first(s)[g]
@inline _ls_group_entry(s::Tuple, j, g) =
    j == 1 ? first(s)[g] : _ls_group_entry(Base.tail(s), j - 1, g)

# ‖R_p * (θ[perm] - θ̂[perm])‖² of block `b`, `θ` read by `entry(c)`. Only the
# statistics' own arrays are read in bounds: a live source keeps its bounds
# check, so an index value beyond it raises as the row-wise gather does.
@inline function _ls_block_norm(s::_LSStatistics, b::Int, entry::F) where {F}
    k = s.k
    total = 0.0
    for r in 1:k
        u = 0.0
        for c in r:k
            p = @inbounds s.perm[c, b]
            x = entry(p)
            d = (x - @inbounds(s.hi[p, b])) - @inbounds(s.lo[p, b])
            u += @inbounds(s.factor[r, c, b]) * d
        end
        total += u * u
    end
    total
end

"""
    _ls_rss_dense(s, scalars, v)

The residual sum of squares at `θ = [scalars..., v...]`.
"""
function _ls_rss_dense(s::_LSStatistics, scalars::Tuple, v::AbstractVector)
    length(scalars) + length(v) == s.k || throw(DimensionMismatch(
        "least-squares statistics of $(s.k) parameters read $(length(scalars) + length(v))"))
    s.rss0 + _ls_block_norm(s, 1, j -> _ls_dense_entry(scalars, v, j))
end

"""
    _ls_rss_grouped(s, sources)

The residual sum of squares when block `j` reads `source[s.groups[j]]` of each
gathered source, in the order the statistics' columns were given.
"""
function _ls_rss_grouped(s::_LSStatistics, sources::Tuple)
    length(sources) == s.k || throw(DimensionMismatch(
        "least-squares statistics of $(s.k) gathered sources read $(length(sources))"))
    total = s.rss0
    for b in 1:s.blocks
        g = @inbounds s.groups[b]
        total += _ls_block_norm(s, b, j -> _ls_group_entry(sources, j, g))
    end
    total
end
