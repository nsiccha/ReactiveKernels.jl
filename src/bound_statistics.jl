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

# The live values of one dense block, `θ` the concatenation of `sources`: a
# number is one entry, a vector its entries in order.
@inline _ls_source_length(x::Number) = 1
@inline _ls_source_length(x::AbstractVector) = length(x)
@inline _ls_source_entry(x::Number, j) = x
@inline _ls_source_entry(x::AbstractVector, j) = x[firstindex(x) + j - 1]
@inline _ls_dense_entry(s::Tuple{Any}, j) = _ls_source_entry(first(s), j)
@inline function _ls_dense_entry(s::Tuple, j)
    n = _ls_source_length(first(s))
    j <= n ? _ls_source_entry(first(s), j) : _ls_dense_entry(Base.tail(s), j - n)
end
@inline _ls_sources_length(::Tuple{}) = 0
@inline _ls_sources_length(s::Tuple) = _ls_source_length(first(s)) + _ls_sources_length(Base.tail(s))
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
    _ls_rss_dense(s, sources)

The residual sum of squares at `θ`, the concatenation of `sources` (numbers
and vectors, in the order of the design's columns).
"""
function _ls_rss_dense(s::_LSStatistics, sources::Tuple)
    _ls_sources_length(sources) == s.k || throw(DimensionMismatch(
        "least-squares statistics of $(s.k) parameters read $(_ls_sources_length(sources))"))
    s.rss0 + _ls_block_norm(s, 1, j -> _ls_dense_entry(sources, j))
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

# --- a location linear in live values ------------------------------------------
#
# The plan rewrite (below) recognises a plate's location as a sum of terms, each
# a live value times a coefficient of bound data, plus bound offsets. Its
# structure `S` is a type parameter of the operations, built from integers and
# numbers only:
#
#     S = (grouped::Bool, index::Int, terms, offsets)
#     term = (coefficient, live::Int, matrix::Int)
#     coefficient = (literal::Float64, data::NTuple{N,Int})
#
# `live` and `matrix` are positions among the operation's live and data
# arguments; a coefficient is `literal` times the elementwise product of the
# data arguments it names (a number, or one entry per row). A dense location
# (`grouped = false`) reads a live number as is (`matrix = 0`) or a live vector
# through a data matrix (`(X * v)[i]`); a grouped one reads each live vector at
# the data index `index` (`v[index[i]]`). Offsets are coefficients alone.

_ls_grouped(S) = S[1]
_ls_index(S) = S[2]
_ls_terms(S) = S[3]
_ls_offsets(S) = S[4]

# A coefficient's value at row `i`: a number is the same on every row.
@inline _ls_at(x::Number, i) = x
@inline _ls_at(x::AbstractVector, i) = x[i]

# Row `i` of `X * v`, summed in column order as `_native_mul`'s own loop does.
@inline function _ls_row_product(X::AbstractMatrix, v::AbstractVector, i)
    total = zero(Base.promote_op(LinearAlgebra.matprod, eltype(X), eltype(v)))
    for j in axes(X, 2)
        total += X[i, j] * v[j]
    end
    total
end

# The expressions of structure `S` at row `i` over `data` and `live`: generated,
# so every position is a literal and every read concretely typed.
function _ls_coefficient_expr((literal, positions), x)
    factors = Any[:(_ls_at(data[$p], i)) for p in positions]
    literal == 1 || pushfirst!(factors, literal)
    isempty(factors) ? (x === nothing ? Float64(literal) : x) :
        x === nothing ? Expr(:call, :*, factors...) : Expr(:call, :*, factors..., x)
end
function _ls_row_expr(S)
    grouped, index, terms, offsets = S
    parts = Any[]
    for (coefficient, l, m) in terms
        x = grouped ? :(live[$l][data[$index][i]]) :
            m == 0 ? :(live[$l]) : :(_ls_row_product(data[$m], live[$l], i))
        push!(parts, _ls_coefficient_expr(coefficient, x))
    end
    for coefficient in offsets
        push!(parts, _ls_coefficient_expr(coefficient, nothing))
    end
    foldl((a, b) -> :($a + $b), parts)
end

"""
    _LSLocation{S}(stats, data, live, rows)

The location a plate reads, `rows` entries computed where they are read from
the bound `data` and the `live` values (structure `S`), together with the
least-squares statistics of its plate's response (`_ls_rss`). It holds no
array of its own: the plate's own loop, which runs only where the collapsed
residual is not used, reads it row by row.
"""
struct _LSLocation{S,T,D<:Tuple,L<:Tuple} <: AbstractVector{T}
    stats::_LSStatistics
    data::D
    live::L
    rows::Int
end
Base.size(x::_LSLocation) = (x.rows,)
Base.IndexStyle(::Type{<:_LSLocation}) = IndexLinear()
@inline @generated function Base.getindex(x::_LSLocation{S}, i::Int) where {S}
    quote
        @boundscheck checkbounds(x, i)
        data, live = x.data, x.live
        $(_ls_row_expr(S))
    end
end

# The residual sum of squares of the plate's response against the location.
@generated function _ls_rss(x::_LSLocation{S}) where {S}
    grouped, _, terms, _ = S
    sources = Expr(:tuple, (:(live[$l]) for (_, l, _) in terms)...)
    # `X * v` requires `v` to be a vector with one entry per column of `X`.
    checks = Any[:(_ls_check_product(data[$m], live[$l])) for (_, l, m) in terms if m != 0]
    quote
        data, live = x.data, x.live
        $(checks...)
        $(grouped ? :(_ls_rss_grouped(x.stats, $sources)) : :(_ls_rss_dense(x.stats, $sources)))
    end
end
_ls_check_product(X::AbstractMatrix, v::AbstractVector) = size(X, 2) == length(v) ||
    throw(DimensionMismatch("matrix has $(size(X, 2)) columns, vector has length $(length(v))"))
_ls_check_product(X, v) = throw(ArgumentError(
    "a bound least-squares location reads a $(typeof(v)) through a $(typeof(X)); expected a vector through a matrix"))

"""
    _LSLocationCall{S,ND,NL,DROP}()

The operation `(stats, response, data..., live..., rest...) -> _LSLocation`:
the location of structure `S` over `ND` data and `NL` live arguments. Trailing
arguments are not read: the plan passes the authored location there while
its binding is undecided. `DROP` names the recipes that compute only that
authored location, which a usable binding drops (`_ls_class_residual`).
"""
struct _LSLocationCall{S,ND,NL,DROP} end
_LSLocationCall(S, ndata::Int, nlive::Int, drop::Tuple) = _LSLocationCall{S,ndata,nlive,drop}()
@inline function (::_LSLocationCall{S,ND,NL})(stats::_LSStatistics, response,
                                             args::Vararg{Any,N}) where {S,ND,NL,N}
    data = ntuple(k -> args[k], Val(ND))
    live = ntuple(k -> args[ND + k], Val(NL))
    T = _ls_location_eltype(data, live)
    _LSLocation{S,T,typeof(data),typeof(live)}(stats, data, live, length(response))
end
_ls_drop(::_LSLocationCall{S,ND,NL,DROP}) where {S,ND,NL,DROP} = DROP
_ls_eltype(x::Number) = typeof(x)
_ls_eltype(x::AbstractArray) = eltype(x)
_ls_location_eltype(data, live) =
    promote_type(Float64, map(_ls_eltype, data)..., map(_ls_eltype, live)...)

# --- binding -----------------------------------------------------------------

"""
    _LSStatisticsOp{S}()

The data-only operation `(response, data...) -> _LSStatistics` of a location
of structure `S`: the statistics of `response - offsets` against the design
the terms read. A binding whose data do not have the shapes the rows need
(one entry per row of the response, a real matrix, an integer index) is
`usable = false`, and the plate keeps its own location (`_ls_class_residual`).
"""
struct _LSStatisticsOp{S} end
(::_LSStatisticsOp{S})(response, data...) where {S} = _ls_bind(Val(S), response, data)

_ls_row_data(x::Real, n) = true
_ls_row_data(x::AbstractVector{<:Real}, n) = length(x) == n && firstindex(x) == 1
_ls_row_data(x, n) = false

# A coefficient over data `ds` as one Float64 per row, or `nothing` where its
# data do not fit.
function _ls_coefficient_rows(literal::Float64, ds::Tuple, n::Int)
    all(d -> _ls_row_data(d, n), ds) || return nothing
    rows = Vector{Float64}(undef, n)
    for i in 1:n
        rows[i] = literal * _ls_coefficient_at(ds, i)
    end
    rows
end
@inline _ls_coefficient_at(::Tuple{}, i) = 1.0
@inline _ls_coefficient_at(ds::Tuple, i) = _ls_at(first(ds), i) * _ls_coefficient_at(Base.tail(ds), i)

_ls_design_matrix(X::AbstractMatrix{<:Real}, n) =
    size(X, 1) == n && firstindex(X, 1) == 1 && firstindex(X, 2) == 1
_ls_design_matrix(X, n) = false
_ls_index_data(x::AbstractVector{<:Integer}, n) =
    !(eltype(x) <: Bool) && length(x) == n && firstindex(x) == 1
_ls_index_data(x, n) = false

# Generated from `S`, so each data read is concretely typed.
@generated function _ls_bind(::Val{S}, response, data::Tuple) where {S}
    grouped, index, terms, offsets = S
    coefficient((literal, positions)) =
        :(_ls_coefficient_rows($(Float64(literal)), ($((:(data[$p]) for p in positions)...),), n))
    body = Any[]
    for c in offsets
        push!(body, quote
            rows = $(coefficient(c))
            rows === nothing && return _ls_unusable(0, 0, Int[], n)
            b .-= rows
        end)
    end
    columns = Symbol[Symbol(:column_, k) for k in eachindex(terms)]
    for ((c, _, m), column) in zip(terms, columns)
        push!(body, :($column = $(coefficient(c))), :($column === nothing && return _ls_unusable(0, 0, Int[], n)))
        if !grouped && m != 0
            push!(body, :(_ls_design_matrix(data[$m], n) || return _ls_unusable(0, 0, Int[], n)),
                  :($column = $column .* data[$m]))
        end
    end
    result = grouped ? quote
        _ls_index_data(data[$index], n) || return _ls_unusable(0, 0, Int[], n)
        _ls_statistics_grouped(data[$index], ($(columns...),), b)
    end : :(_ls_statistics_dense(hcat($(columns...)), b))
    quote
        response isa AbstractVector{<:Real} && firstindex(response) == 1 ||
            return _ls_unusable(0, 0, Int[], 0)
        n = length(response)
        b = Vector{Float64}(response)
        $(body...)
        $result
    end
end

# --- the plan rewrite ----------------------------------------------------------
#
# Per bound port set (`_partial_bound_statistics`, before the index runs): a
# total-only summed plate whose factored loop (summed_plate_factors.jl) sums
# `(y - mu)^2`, `y` a bound argument, and whose location argument `mu` is
# linear in live values with bound coefficients, gets
#
#     stats = _LSStatisticsOp{S}()(y, data...)                 # data-only: frozen at binding
#     loc = _LSLocationCall{S}()(stats, y, data..., live..., mu)
#     plate(..., y, ..., loc, ...)                             # was mu
#
# Per binding (`_ls_class_residual`, keyed by `stats.usable` among the plate
# classes): a usable binding drops `mu` and its producers from the residual, so
# the plate reads `loc` row by row only where its own loop runs, and the
# lowering sums the squared residuals from `stats` (`_lower_bound_statistics_native`);
# an unusable one gives the plate `mu` back and drops `loc`: today's program.

# A recipe's parameter names and authored source, its called names resolved:
# a source operation's in its module, and a bare call's (`log(scale)`, whose
# operation is the function itself) as that function. `nothing` for anything
# else.
function _ls_recipe_source(r::Recipe)
    r.source isa Expr || return nothing
    names = _recipe_source_names(r)
    if r.op isa _KernelSourceOp
        return names, _ls_resolved_source(r.source, getfield(r.op, :mod), Set(names))
    elseif r.op isa Function && Meta.isexpr(r.source, :call) &&
            all(a -> a isa Symbol, r.source.args[2:end]) && r.source.args[2:end] == names
        return names, Expr(:call, r.op, names...)
    end
    nothing
end

# An authored source with each called name resolved where it was written, so
# the terms identify `+`, `*`, `^` there rather than in this module.
function _ls_resolved_source(ex, mod, params::Set{Symbol})
    mod isa Module || return ex
    ex isa Expr || return ex
    ex.head in (:quote, :inert) && return ex
    args = Any[_ls_resolved_source(a, mod, params) for a in ex.args]
    if ex.head === :call && ex.args[1] isa Symbol && !(ex.args[1] in params) &&
            isdefined(mod, ex.args[1])
        args[1] = GlobalRef(mod, ex.args[1])
    end
    Expr(ex.head, args...)
end

_ls_is_call(x, f, n) = x isa Expr && x.head === :call && length(x.args) == n + 1 &&
    _rw_resolve(x.args[1], @__MODULE__) === f

# --- the location ----------------------------------------------------------------

# A coefficient: `literal` times the product of data values.
const _LSCoef = Tuple{Float64,Vector{ValueId}}
# A location linear in live values: terms `(coefficient, live, matrix, index)`
# (`matrix`/`index` `nothing` where absent) and data offsets.
struct _LSAffine
    terms::Vector{Tuple{_LSCoef,ValueId,Union{Nothing,ValueId},Union{Nothing,ValueId}}}
    offsets::Vector{_LSCoef}
end
_ls_scale(c::_LSCoef, d::_LSCoef) = (c[1] * d[1], vcat(c[2], d[2]))
_ls_scale(x::_LSAffine, d::_LSCoef) = _LSAffine(
    [(_ls_scale(c, d), l, m, i) for (c, l, m, i) in x.terms], [_ls_scale(c, d) for c in x.offsets])
_ls_add(x::_LSAffine, y::_LSAffine) = _LSAffine(vcat(x.terms, y.terms), vcat(x.offsets, y.offsets))

# An operand of a location's arithmetic: a data coefficient, a live value the
# terms read whole (a number, or a vector a data matrix or index reads), or an
# affine location.
struct _LSLive
    id::ValueId
    value::Value
end

# The affine form of value `v` in plan `p`, or `nothing`. `dataonly` holds the
# data-only values; `seen` the recipes the form reads through.
function _ls_location(p::Plan, v::Value, dataonly::Set{ValueId}, seen::Vector{Recipe})
    x = _ls_operand(p, v, dataonly, seen, 0)
    x = x isa _LSAffine ? x : nothing
    x === nothing && return nothing
    isempty(x.terms) ? nothing : x
end

const _LS_MAX_DEPTH = 32

function _ls_operand(p::Plan, v::Value, dataonly, seen, depth)
    g = p.graph
    id = canon_id(g, v.id)
    id in dataonly && return (1.0, ValueId[id])::_LSCoef
    depth > _LS_MAX_DEPTH && return nothing
    r = get(p.producer, id, nothing)
    r === nothing && return _LSLive(id, v)
    # A gather of a live source by a data index.
    if length(r.inputs) == 2 && length(r.outputs) == 1 && !r.effectful &&
            (_index_run_gather(r.op) || r.op === _readonly_gather_runs)
        src, index = r.inputs
        canon_id(g, index.id) in dataonly && !(canon_id(g, src.id) in dataonly) || return _LSLive(id, v)
        push!(seen, r)
        return _LSAffine([((1.0, ValueId[]), canon_id(g, src.id), nothing, canon_id(g, index.id))],
                         _LSCoef[])
    end
    length(r.outputs) == 1 && !r.effectful || return _LSLive(id, v)
    r.op isa _AuthoredPlateOp && return _ls_plate_operand(p, r, id, v, dataonly, seen, depth)
    source = _ls_recipe_source(r)
    source === nothing && return _LSLive(id, v)
    names, source = source
    length(names) == length(r.inputs) || return _LSLive(id, v)
    env = Dict{Symbol,Any}(zip(names, r.inputs))
    x = _ls_expr(p, source, env, dataonly, seen, depth + 1, false)
    x === nothing && return _LSLive(id, v)
    push!(seen, r)
    x
end

# An array-valued plate read as the elementwise expression its cell computes
# (`plate(x, b) do x, v; x * v end` is `x .* b`): RKPPL builds a location from
# such plates. Each cell recipe is read from its source, over the operands of
# the plate's arguments.
function _ls_plate_operand(p::Plan, r::Recipe, id::ValueId, v::Value, dataonly, seen, depth)
    inner = getfield(r.op.kernel, :plan)
    length(inner.have) == length(r.inputs) && length(inner.want) == 1 || return _LSLive(id, v)
    ig = inner.graph
    env = Dict{ValueId,Any}()
    for (k, h) in enumerate(inner.have)
        x = _ls_operand(p, r.inputs[k], dataonly, seen, depth + 1)
        x === nothing && return _LSLive(id, v)
        env[canon_id(ig, h.id)] = x
    end
    for q in inner.recipes
        length(q.outputs) == 1 && !q.effectful || return _LSLive(id, v)
        source = _ls_recipe_source(q)
        source === nothing && return _LSLive(id, v)
        names, source = source
        length(names) == length(q.inputs) || return _LSLive(id, v)
        local_env = Dict{Symbol,Any}()
        for (name, input) in zip(names, q.inputs)
            haskey(env, canon_id(ig, input.id)) || return _LSLive(id, v)
            local_env[name] = env[canon_id(ig, input.id)]
        end
        x = _ls_expr(p, source, local_env, dataonly, seen, depth + 1, true)
        x === nothing && return _LSLive(id, v)
        env[canon_id(ig, only(q.outputs).id)] = x
    end
    x = get(env, canon_id(ig, only(inner.want).id), nothing)
    x === nothing && return _LSLive(id, v)
    push!(seen, r)
    x
end

_ls_affine_of(x::_LSAffine, p) = x
_ls_affine_of(x::_LSCoef, p) = _LSAffine([], [x])
# A live value read whole: a live number; anything else has no known reading.
_ls_affine_of(x::_LSLive, p) = valtype(x.value) <: Number ?
    _LSAffine([((1.0, ValueId[]), x.id, nothing, nothing)], _LSCoef[]) : nothing
_ls_affine_of(::Nothing, p) = nothing

# `elementwise`: the expression is a plate cell's, whose arithmetic is per
# element of its batched arguments.
function _ls_expr(p::Plan, ex, env::Dict{Symbol,Any}, dataonly, seen, depth, elementwise::Bool)
    if ex isa Symbol
        haskey(env, ex) || return nothing
        x = env[ex]
        return x isa Value ? _ls_operand(p, x, dataonly, seen, depth) : x
    end
    ex isa Real && !(ex isa Bool) && return (Float64(ex), ValueId[])::_LSCoef
    ex isa Expr || return nothing
    if ex.head === :block
        statements = Any[a for a in ex.args if !(a isa LineNumberNode)]
        length(statements) == 1 || return nothing
        return _ls_expr(p, only(statements), env, dataonly, seen, depth, elementwise)
    end
    # Dotted calls `a .+ b` and calls `+(a, b)` read alike here: the plate
    # reads the result row by row in either case.
    f, args = if ex.head === :call && ex.args[1] isa Union{Symbol,GlobalRef}
        name = ex.args[1] isa GlobalRef ? ex.args[1].name : ex.args[1]
        s = String(name)
        if startswith(s, ".") && length(s) > 1
            (_rw_resolve(Symbol(s[2:end]), Base), ex.args[2:end])
        else
            (_rw_resolve(ex.args[1], @__MODULE__), ex.args[2:end])
        end
    elseif ex.head === :call && ex.args[1] isa Function
        # A bare call's operation (`_ls_recipe_source`).
        (ex.args[1], ex.args[2:end])
    elseif ex.head === :. && length(ex.args) == 2 && Meta.isexpr(ex.args[2], :tuple)
        (_rw_resolve(ex.args[1], @__MODULE__), ex.args[2].args)
    elseif ex.head === :ref && length(ex.args) == 2
        # `src[index]`: a gather of a live source by a data index.
        src, index = ex.args
        src isa Symbol && index isa Symbol && haskey(env, src) && haskey(env, index) &&
            env[src] isa Value && env[index] isa Value || return nothing
        g = p.graph
        sid, iid = canon_id(g, env[src].id), canon_id(g, env[index].id)
        iid in dataonly && !(sid in dataonly) || return nothing
        return _LSAffine([((1.0, ValueId[]), sid, nothing, iid)], _LSCoef[])
    else
        return nothing
    end
    dotted = elementwise || ex.head === :. || (ex.head === :call && ex.args[1] isa Union{Symbol,GlobalRef} &&
        startswith(String(ex.args[1] isa GlobalRef ? ex.args[1].name : ex.args[1]), "."))
    operands = Any[_ls_expr(p, a, env, dataonly, seen, depth, elementwise) for a in args]
    any(o -> o === nothing, operands) && return nothing
    if f === Base.:+
        forms = Any[_ls_affine_of(o, p) for o in operands]
        any(isnothing, forms) && return nothing
        return foldl(_ls_add, forms)
    elseif f === Base.:- && length(operands) == 1
        x = _ls_affine_of(operands[1], p)
        return x === nothing ? nothing : _ls_scale(x, (-1.0, ValueId[]))
    elseif f === Base.:- && length(operands) == 2
        x, y = _ls_affine_of(operands[1], p), _ls_affine_of(operands[2], p)
        (x === nothing || y === nothing) && return nothing
        return _ls_add(x, _ls_scale(y, (-1.0, ValueId[])))
    elseif f === Base.:* && length(operands) == 2
        a, b = operands
        # `X * v`: a data matrix times a live vector, not elementwise.
        if !dotted && a isa _LSCoef && b isa _LSLive && a[1] == 1.0 && length(a[2]) == 1 &&
                !(valtype(b.value) <: Number)
            return _LSAffine([((1.0, ValueId[]), b.id, only(a[2]), nothing)], _LSCoef[])
        end
        # Otherwise a coefficient times a location, either side. A product
        # of data with a non-dotted `*` is elementwise only for numbers, which
        # the binding checks.
        a isa _LSCoef && b isa _LSCoef &&
            return dotted || isempty(a[2]) || isempty(b[2]) ? _ls_scale(a, b) : nothing
        a isa _LSCoef && (x = _ls_affine_of(b, p); return x === nothing ? nothing :
            (dotted || isempty(a[2]) ? _ls_scale(x, a) : nothing))
        b isa _LSCoef && (x = _ls_affine_of(a, p); return x === nothing ? nothing :
            (dotted || isempty(b[2]) ? _ls_scale(x, b) : nothing))
        return nothing
    elseif f === Base.:/ && length(operands) == 2 && operands[2] isa _LSCoef &&
            isempty(operands[2][2]) && operands[2][1] != 0
        x = operands[1] isa _LSCoef ? _LSAffine([], [operands[1]]) : _ls_affine_of(operands[1], p)
        return x === nothing ? nothing : _ls_scale(x, (1 / operands[2][1], ValueId[]))
    elseif f === identity && length(operands) == 1
        return operands[1]
    end
    nothing
end

# --- the plan pass -----------------------------------------------------------------

# The structure `S` of an affine location over `data` and `live` (positions).
function _ls_structure(x::_LSAffine, data::Vector{ValueId}, live::Vector{ValueId})
    coef((literal, ids)) = (literal, Tuple(findfirst(==(id), data) for id in ids))
    grouped = x.terms[1][4] !== nothing
    index = grouped ? findfirst(==(x.terms[1][4]), data) : 0
    terms = Tuple((coef(c), findfirst(==(l), live),
                   m === nothing ? 0 : findfirst(==(m), data)) for (c, l, m, _) in x.terms)
    (grouped, index, terms, Tuple(coef(c) for c in x.offsets))
end

# Whether the terms read one way: all through one index, or none through any;
# and each live value once.
function _ls_consistent(x::_LSAffine)
    indices = unique(i for (_, _, _, i) in x.terms)
    length(indices) == 1 || return false
    if only(indices) !== nothing
        all(m === nothing for (_, _, m, _) in x.terms) || return false
    end
    lives = [l for (_, l, _, _) in x.terms]
    allunique(lives)
end

# Whether this preparation rewrites: the setting, and the lowering that sums
# the factored residuals (summed_plate_factors.jl) runs.
_ls_enabled() = _bound_statistics() && _simplify() && !_plate_reverse_enabled() &&
    !_footprint_recording()

function _ls_vector_value(v::Value)
    T = typeintersect(valtype(v), AbstractVector)
    T === Union{} ? v : Value{T}(v.id, v.name)
end

# The positions `(a, b)` of a plate's arguments whose squared difference its
# factored loop sums, `(read(a) - read(b))^2`, or `nothing`. The plate and its
# total are lowered alone, over their arguments, and read as the lowering pass
# below reads them: the summed plate factors (summed_plate_factors.jl) decide
# what a cell adds, whatever the spelling of its law (a source per term, or a
# composed source with a guard the plate lowering takes outside its loop).
function _ls_squared_ports(p::Plan, r::Recipe, total::Recipe)
    g = p.graph
    # The batched arguments as the vectors a usable binding has (a bound
    # response narrowed to its class, `_partial_narrow_plate_inputs`; the
    # location an `_LSLocation`): an argument of unknown type would take the
    # plate's generic loop, which the summed plate factors do not read.
    atomic = typeof(r.op).parameters[2]
    inputs = Value[k in atomic ? v : _ls_vector_value(v) for (k, v) in enumerate(r.inputs)]
    plate = Recipe(r.id, Tuple(inputs), r.outputs, r.op, r.cost, r.cse_key, r.effectful, r.source)
    have = unique(v -> canon_id(g, v.id), inputs)
    alone = _partial_subplan(p, have, Value[only(total.outputs)], Recipe[plate, total])
    ast = first(_lower_with_ops(alone))
    signature = ast.args[1]
    params = Any[a isa Expr && a.head === :(::) ? a.args[1] : a for a in signature.args[3:end]]
    length(params) == length(have) || return nothing
    position = Dict{Symbol,Int}()
    for (k, name) in enumerate(params)
        name isa Symbol || return nothing
        h = canon_id(g, have[k].id)
        position[name] = findfirst(v -> canon_id(g, v.id) == h, inputs)
    end
    roots = _ls_roots(ast)
    root(x) = get(roots, x, x)
    found = nothing
    _ls_factored_blocks(ast) do partial, index, cellbody
        found === nothing || return
        pair = _ls_factored_pair(cellbody, index, partial, root)
        pair === nothing && return
        a, b = pair
        haskey(position, a) && haskey(position, b) && position[a] != position[b] &&
            (found = (position[a], position[b]))
    end
    found
end

function _partial_bound_statistics(p::Plan, bound_values::Vector{Value})
    isempty(bound_values) && return p
    _ls_enabled() || return p
    g = p.graph
    wanted = Set(canon_id(g, w.id) for w in p.want)
    bound = Set(canon_id(g, v.id) for v in bound_values)
    candidates = Recipe[]
    for r in p.recipes
        r.op isa _AuthoredPlateOp || continue
        total = _authored_plate_sum_recipe(p, r)
        total === nothing && continue
        id = canon_id(g, only(r.outputs).id)
        id in wanted && continue
        any(q -> q !== total && any(v -> canon_id(g, v.id) == id, q.inputs), p.recipes) && continue
        push!(candidates, r)
    end
    isempty(candidates) && return p
    _, _, dataonly = _partial_split(p, bound)
    rewrites = Dict{Int,Any}()
    for r in candidates
        rewrite = _ls_plate_rewrite(p, r, _authored_plate_sum_recipe(p, r), dataonly, wanted)
        rewrite === nothing || (rewrites[r.id] = rewrite)
    end
    isempty(rewrites) && return p
    copied = Graph(copy(g.values), copy(g.recipes), copy(g.producers), copy(g.aliases),
                   g.version, nothing, copy(g.checks))
    recipes = Recipe[]
    fresh = minimum((r.id for r in p.recipes); init = 0) - 1
    for r in p.recipes
        rewrite = get(rewrites, r.id, nothing)
        if rewrite === nothing
            push!(recipes, r)
            continue
        end
        (; S, response, location, position, data, live, drop) = rewrite
        stem = Symbol(location.name)
        # Named from the location, never a fresh gensym: a name that changes
        # between preparations changes the prepared program's identity.
        stats = value!(copied, Symbol(stem, :_least_squares), _LSStatistics)
        loc = value!(copied, Symbol(stem, :_least_squares_location), Any)
        datavalues = Value[copied.values[id] for id in data]
        livevalues = Value[copied.values[id] for id in live]
        push!(recipes, Recipe(fresh, (response, datavalues...), (stats,),
            _LSStatisticsOp{S}(), 0.0, nothing, false))
        push!(recipes, Recipe(fresh - 1, (stats, response, datavalues..., livevalues..., location),
            (loc,), _LSLocationCall(S, length(data), length(live), drop), 0.0, nothing, false))
        fresh -= 2
        inputs = collect(Value, r.inputs)
        inputs[position] = loc
        push!(recipes, Recipe(r.id, Tuple(inputs), r.outputs, r.op, r.cost, r.cse_key,
                              r.effectful, r.source))
    end
    template = Plan(copied, p.have, p.want, p.recipes, p.producer, p.cost, p.candidates)
    _partial_subplan(template, p.have, p.want, recipes)
end

# The rewrite of one total-only plate, or `nothing`.
function _ls_plate_rewrite(p::Plan, r::Recipe, total::Recipe, dataonly::Set{ValueId},
                           wanted::Set{ValueId})
    g = p.graph
    pair = _ls_squared_ports(p, r, total)
    pair === nothing && return nothing
    a, b = pair
    ida, idb = canon_id(g, r.inputs[a].id), canon_id(g, r.inputs[b].id)
    # The response is bound data, the location is not.
    position, response, location = ida in dataonly && !(idb in dataonly) ? (b, r.inputs[a], r.inputs[b]) :
        idb in dataonly && !(ida in dataonly) ? (a, r.inputs[b], r.inputs[a]) : (0, nothing, nothing)
    position == 0 && return nothing
    seen = Recipe[]
    x = _ls_location(p, location, dataonly, seen)
    x === nothing && return nothing
    _ls_consistent(x) || return nothing
    data = unique!(ValueId[id for (c, _, m, i) in x.terms for id in
        (c[2]..., (m === nothing ? () : (m,))..., (i === nothing ? () : (i,))...)])
    append!(data, unique!(ValueId[id for c in x.offsets for id in c[2] if !(id in data)]))
    unique!(data)
    live = ValueId[l for (_, l, _, _) in x.terms]
    S = _ls_structure(x, data, live)
    # The location operation reads the live values and the data itself.
    kept = union(wanted, Set(live), Set(data))
    drop = _ls_exclusive(p, canon_id(g, location.id), r, kept, dataonly)
    (; S, response, location, position, data, live, drop)
end

# The recipes that compute only the authored location for this plate: none
# of their outputs is `kept` (wanted, or read by the location operation) or
# read by anything else. A usable binding drops them (`_ls_class_residual`).
function _ls_exclusive(p::Plan, location::ValueId, plate::Recipe, kept, dataonly)
    g = p.graph
    readers = Dict{ValueId,Vector{Int}}()
    for q in p.recipes, v in q.inputs
        push!(get!(readers, canon_id(g, v.id), Int[]), q.id)
    end
    drop = Set{Int}()
    pending = [location]
    while !isempty(pending)
        id = pop!(pending)
        (id in kept || id in dataonly) && continue
        r = get(p.producer, id, nothing)
        r === nothing && continue
        r.effectful && continue
        r.id in drop && continue
        all(o -> !(canon_id(g, o.id) in kept) &&
            all(q -> q == plate.id || q in drop, get(readers, canon_id(g, o.id), Int[])),
            r.outputs) || continue
        push!(drop, r.id)
        append!(pending, (canon_id(g, v.id) for v in r.inputs))
    end
    Tuple(sort!(collect(drop)))
end

# --- per binding -----------------------------------------------------------------

# Whether each bound least-squares location's statistics are usable for this
# binding, in residual order: classes of the compiled residual
# (`_partial_plate_classes`), so a usable and an unusable binding never share
# one.
function _ls_classes(b::_PartialBoundary, @nospecialize(hoisted))
    usable = Bool[]
    for r in b.residual
        r.op isa _LSLocationCall || continue
        stats = first(r.inputs)
        slot = findfirst(c -> c.id == stats.id, b.constants)
        value = slot === nothing ? nothing : hoisted[slot]
        push!(usable, value isa _LSStatistics && value.usable)
    end
    usable
end

# The residual recipes for this binding's classes: a usable location stops
# reading the authored one, whose exclusive producers are dropped; an unusable
# one is dropped and its plate reads the authored location again.
function _ls_class_residual(g::Graph, b::_PartialBoundary, @nospecialize(classes::Tuple),
                            residual::Vector{Recipe})
    any(r -> r.op isa _LSLocationCall, residual) || return residual
    offset = length(b.plate_slots)
    drop = Set{Int}()
    restore = Dict{ValueId,Value}()   # an unusable location => the authored one
    replaced = Dict{Int,Recipe}()
    j = 0
    for r in residual
        r.op isa _LSLocationCall || continue
        j += 1
        if get(classes, offset + j, false) === true
            replaced[r.id] = Recipe(r.id, r.inputs[1:end-1], r.outputs, r.op, r.cost,
                                    r.cse_key, r.effectful, r.source)
            union!(drop, _ls_drop(r.op))
        else
            push!(drop, r.id)
            restore[canon_id(g, only(r.outputs).id)] = last(r.inputs)
        end
    end
    out = Recipe[]
    for r in residual
        r.id in drop && continue
        r = get(replaced, r.id, r)
        if !isempty(restore) && any(v -> haskey(restore, canon_id(g, v.id)), r.inputs)
            inputs = Tuple(get(restore, canon_id(g, v.id), v) for v in r.inputs)
            r = Recipe(r.id, inputs, r.outputs, r.op, r.cost, r.cse_key, r.effectful, r.source)
        end
        push!(out, r)
    end
    out
end

# --- lowering ----------------------------------------------------------------------
#
# After the summed plate factors (summed_plate_factors.jl), a plate that reads a
# usable `_LSLocation` sums `(y - loc)^2` in its factored loop. That loop
# becomes one call, `partial = _plate_total_add(partial, _ls_rss(loc))`: the
# statistics answer the sum without visiting a row. The rest of the factored
# block (its finiteness checks, `count * a + c * partial`) and the plate's own
# loop, which runs where the factored total is not kept and reads `loc` row by
# row, are unchanged.

_ls_location_callee(f, ops) = (slot = _callable_slot(f);
    slot !== nothing && slot <= length(ops) && ops[slot] isa _LSLocationCall)

# A read's array: `name = begin @inbounds local v = read(arg, index); v end`.
function _ls_read_arg(rhs, index::Symbol)
    parts = _rw_read_parts(rhs)
    parts === nothing && return nothing
    call = parts[2]
    _rw_resolve(call.args[1], @__MODULE__) in (Base.Broadcast._broadcast_getindex, _plate_read) &&
        length(call.args) == 3 && call.args[3] === index && call.args[2] isa Symbol || return nothing
    call.args[2]
end

# Every name that reads an array as another does (`Base.broadcastable`,
# `Broadcast.preprocess`), to the name it reads.
function _ls_roots(body::Expr)
    roots = Dict{Symbol,Symbol}()
    _index_run_visit(body) do ex
        ex.head === :(=) && length(ex.args) == 2 && ex.args[1] isa Symbol || return
        lhs, rhs = ex.args
        if _index_run_call(rhs, GlobalRef(Base, :broadcastable), 1) && rhs.args[2] isa Symbol
            roots[lhs] = get(roots, rhs.args[2], rhs.args[2])
        elseif _index_run_call(rhs, GlobalRef(Base.Broadcast, :preprocess), 2) &&
                rhs.args[3] isa Symbol
            roots[lhs] = get(roots, rhs.args[3], rhs.args[3])
        end
    end
    roots
end

# `(partial, index, cellbody, block, at)` of each factored block of `ex`: the
# summed plate factors' `if fast; partial = seed; <loop>; ... end`, whose loop
# is statement `at` of `block`; `nothing` for anything else.
function _ls_factored_block(ex)
    ex isa Expr && ex.head === :if && length(ex.args) == 2 && ex.args[1] isa Symbol &&
        endswith(String(ex.args[1]), _spf_mark()) && Meta.isexpr(ex.args[2], :block) ||
        return nothing
    block = ex.args[2]
    positions = [k for (k, a) in enumerate(block.args) if !(a isa LineNumberNode)]
    length(positions) >= 2 || return nothing
    seed, loop = block.args[positions[1]], block.args[positions[2]]
    Meta.isexpr(seed, :(=), 2) && seed.args[1] isa Symbol &&
        _index_run_call(seed.args[2], GlobalRef(@__MODULE__, :_plate_run_seed), 1) || return nothing
    plate = _index_run_simd_plate(loop)
    plate === nothing && return nothing
    _, index, cellbody = plate
    (seed.args[1], index, cellbody, block, positions[2])
end
_ls_factored_blocks(f, body::Expr) = _index_run_visit(body) do ex
    found = _ls_factored_block(ex)
    found === nothing || f(found[1], found[2], found[3])
end

function _lower_bound_statistics_native(body::Expr, ops)
    any(op -> op isa _LSLocationCall, ops) || return body
    # Each location's response.
    responses = Dict{Symbol,Any}()
    _index_run_visit(body) do ex
        ex.head === :(=) && length(ex.args) == 2 && ex.args[1] isa Symbol || return
        lhs, rhs = ex.args
        rhs isa Expr && rhs.head === :call && _ls_location_callee(rhs.args[1], ops) &&
            length(rhs.args) >= 3 && (responses[lhs] = rhs.args[3])
    end
    isempty(responses) && return body
    roots = _ls_roots(body)
    root(x) = get(roots, x, x)
    _index_run_replace(body) do ex
        found = _ls_factored_block(ex)
        found === nothing && return ex
        partial, index, cellbody, block, at = found
        pair = _ls_factored_pair(cellbody, index, partial, root)
        pair === nothing && return ex
        location = nothing
        for (loc, y) in (pair, reverse(pair))
            haskey(responses, loc) || continue
            response = responses[loc]
            response isa Symbol && root(response) === y && (location = loc; break)
        end
        location === nothing && return ex
        args = copy(block.args)
        args[at] = Expr(:(=), partial, Expr(:call, GlobalRef(@__MODULE__, :_plate_total_add), partial,
            Expr(:call, GlobalRef(@__MODULE__, :_ls_rss), location)))
        Expr(:if, ex.args[1], Expr(:block, args...))
    end
end

# The roots of the arrays a factored cell body sums the squared difference of,
# `(read(a) - read(b))^2`, or `nothing`.
function _ls_factored_pair(cellbody::Expr, index::Symbol, partial::Symbol, root)
    statements = Any[s for s in cellbody.args if !(s isa LineNumberNode)]
    length(statements) >= 2 || return nothing
    add, term = statements[end], statements[end-1]
    Meta.isexpr(add, :(=), 2) && add.args[1] === partial &&
        _index_run_call(add.args[2], GlobalRef(@__MODULE__, :_plate_total_add), 2) &&
        Meta.isexpr(term, :(=), 2) && add.args[2].args[3] === term.args[1] || return nothing
    reads = Dict{Symbol,Symbol}()     # a cell name => the root of the array it reads
    defs = Dict{Symbol,Any}()         # any other cell name => its value
    for s in statements[1:end-2]
        Meta.isexpr(s, :(=), 2) && s.args[1] isa Symbol || return nothing
        arg = _ls_read_arg(s.args[2], index)
        if arg !== nothing
            reads[s.args[1]] = root(arg)
        elseif s.args[2] isa Symbol && haskey(reads, s.args[2])
            reads[s.args[1]] = reads[s.args[2]]
        else
            defs[s.args[1]] = s.args[2]
        end
    end
    # A name the factored cell binds stands for its value.
    function value(x)
        for _ in 1:64
            x isa Symbol && haskey(defs, x) || return x
            x = defs[x]
        end
        x
    end
    t = value(term.args[2])
    _ls_is_call(t, Base.:^, 2) && t.args[3] == 2 || return nothing
    d = value(t.args[2])
    _ls_is_call(d, Base.:-, 2) || return nothing
    a, b = (x isa Symbol ? get(reads, x, nothing) : nothing for x in d.args[2:3])
    a === nothing || b === nothing || a === b ? nothing : (a, b)
end

# The residual recipes without the constants that only a rewritten location
# read: statistics an unusable binding's residual does not read, and the index
# runs (index_runs.jl) of a gather a usable binding dropped. An unusable
# binding's residual is then the plan without the rewrite.
_ls_dropped_constant(r::Recipe) =
    r.op isa Union{_BoundConstant{_LSStatistics},_BoundConstant{_IndexRunData}}
function _ls_unread_statistics(g::Graph, recipes::Vector{Recipe})
    any(_ls_dropped_constant, recipes) || return recipes
    read = Set{ValueId}(canon_id(g, v.id) for r in recipes for v in r.inputs)
    filter(r -> !_ls_dropped_constant(r) || canon_id(g, only(r.outputs).id) in read, recipes)
end
