module TestJSSource

using ReactiveKernels
using LogExpFunctions: log1pexp, logistic
using Test
using Base: OneTo
using SpecialFunctions: erf, erfc, erfcx, logerfc, gamma, loggamma, logabsgamma, logbeta, beta_inc,
    beta_inc_inv, gamma_inc, gamma_inc_inv
using LogExpFunctions: log1pexp, log1mexp
import Distributions

const NODE = ReactiveKernels._js_default_node() !== nothing

# Plain Julia helpers without annotations: branches, early returns and loops
# are translated from their typed code.
js_piecewise(x) = x < 0 ? -x : x <= 1 ? x^2 : 2x - 1
function js_early(x)
    x <= 1 && return log1p(x)
    return sqrt(x) + exp(-x) / x
end
function js_logistic_map(x0, r, n)
    x = x0
    for _ in 1:n
        x = r * x * (1 - x)
    end
    x
end
function js_cube_root(target, lo, hi)
    while hi - lo > 1e-12
        mid = (lo + hi) / 2
        if mid^3 > target
            hi = mid
        else
            lo = mid
        end
    end
    (lo + hi) / 2
end
function js_weighted_total(v)
    s = 0.0
    for i in eachindex(v)
        s += v[i] * i
    end
    s
end
js_squares(v) = [x^2 for x in v]
js_c_sin(x) = ccall(:sin, Float64, (Float64,), x)

# Keyword helpers (a required keyword, a default, a default computed from
# another keyword), named tuples built and read by field, and destructuring.
function js_kw_split(x, y; share, offset = 0.0, scale = 2 * share)
    lo, hi = zeros(length(x)), zeros(length(x))
    for i in eachindex(x)
        lo[i] = share * x[i] + offset
        hi[i] = scale * y[i]
    end
    return (; lo, hi = hi)
end
js_kw_total(r; weight) = weight * sum(r.lo) + sum(r.hi)
js_record_mean(rec) = (sum(rec.a) + sum(rec.b)) / (length(rec.a) + length(rec.b))
js_kw_need(x; a) = a * x
# a keyword `b` that `js_kw_need` does not take, and a missing `a`: Julia's
# MethodError and UndefKeywordError, when the branch runs
js_kw_unknown(x) = x > 0 ? js_kw_need(x; a = 2.0) : js_kw_need(x; a = 1.0, b = 3.0)
js_kw_missing(x) = x < 10 ? js_kw_need(x; a = 2.0) : js_kw_need(x)
# a keyword call reaches the keyword method, not a more specific one without
# keywords; a function without keyword methods is a MethodError
js_kw_pick(x; a) = a * x
js_kw_pick(x::Float64) = -x
js_kw_specific(x) = js_kw_pick(x; a = 3.0) + js_kw_pick(x)
js_kw_none(x) = x
js_kw_none_use(x) = x > -5 ? x : js_kw_none(x; a = 1.0)
# a generator over a range written in the source, destructured
function js_buffers(x)
    n = length(x)
    a, b, c = (Vector{Float64}(undef, n) for _ in 1:3)
    for i in 1:n
        a[i] = x[i]
        b[i] = 2 * x[i]
        c[i] = a[i] + b[i]
    end
    out = similar(x)
    for i in eachindex(out)
        out[i] = c[i] - a[i]
    end
    return out
end
js_window_means(x, w) =
    [sum(@view x[(w * (k - 1) + 1):(w * k)]) / length(@view x[(w * (k - 1) + 1):(w * k)]) for k in 1:(length(x) ÷ w)]
js_view_second(x) = (v = @view x[2:end]; v[1])
# `log1pexp` destructures thresholds that depend only on the argument's type
js_soft(x, s) = s * log1pexp(x / s)

# Loops over a range written in the helper (`for _ in 1:60`): the range is known
# while translating, needs no kernel input, and a loop over an empty one, like a
# branch on a constant, is dead code.
js_decay(x, k) = exp(-k * x)
function js_half_life(k)
    low, high = 0.0, 1.0
    while js_decay(high, k) > 0.5
        high *= 2
    end
    for _ in 1:60
        middle = (low + high) / 2
        js_decay(middle, k) > 0.5 ? (low = middle) : (high = middle)
    end
    (low + high) / 2
end
function js_fixed_series(x)
    s = 0.0
    for i in 1:12
        s += x^i / i
    end
    for i in OneTo(3)
        s = s * i + 1
    end
    for i in -3:2
        s += i * x
    end
    for _ in 1:0
        s = -s
    end
    s
end
const JS_CHECKED = false
function js_unchecked(x)
    if JS_CHECKED
        x < 0 && error("negative input")
    end
    sqrt(abs(x))
end
function js_range_sum(r, x)
    isempty(r) && return x
    s = 0.0
    for i in r
        s += i * x
    end
    s
end
function js_chosen_range(flag, n)
    r = flag ? (1:3) : (1:n)
    s = 0
    for i in r
        s += i
    end
    s
end
js_powers(x) = x .^ (0:4)
js_scaled(x) = [x / k for k in 1:5]
js_squares_scaled(x) = [k^2 for k in 1:5] .* x
js_multiples(x) = map(k -> k * x, 1:4)

# Tuples, polynomials and type parameters as the special functions' Julia code
# uses them: destructuring, `evalpoly` of a coefficient tuple, a tuple read at
# a run-time position, `promote`/`oftype` (of an irrational constant),
# `copysign`, `iseven` of a `Float64` and a method's static parameter.
js_poly(x) = evalpoly(x, (1.0, -0.5, 0.25, 2.0))
js_pick(x, k) = (x, 2x, 3x)[k]
function js_promoted(x, n)
    a, b = promote(x, n)
    a * b + copysign(1.0, -x) + (x < oftype(x, π) ? 1.0 : 0.0) + first((x, n)) + last((n, x))
end
js_static_parameter(x::T, n) where {T<:Real} = T(n) * x
js_parity(x) = iseven(x) ? 1.0 : isodd(x) ? -1.0 : 0.0
js_logabsgamma_value(a) = first(logabsgamma(a))
js_logabsgamma_sign(a) = last(logabsgamma(a))
js_tdist_cdf(x, nu) = Distributions.cdf(Distributions.TDist(nu), x)
# Stepped ranges, constant or computed at run time, and the tuple tools
# SpecialFunctions' incomplete gamma function uses: `ntuple`, `cumprod`,
# `findfirst` with `something`, `minmax`, `map` and an irrational constant.
function js_countdown(n, x)
    s = 0.0
    for k in n:-1:1
        s = s * x + k
    end
    for k in 10:-3:1
        s += k
    end
    s
end
js_stride(a, s, b) = (r = a:s:b; isempty(r) ? -length(r) - 1 : length(r) + 10first(r) + 100last(r) + 1000step(r))
function js_tuple_tools(x)
    ts = cumprod(ntuple(i -> x / i, Val(6)))
    k = something(findfirst(<(0.1), ts), 7)
    lo, hi = minmax(x, 2x)
    a, b = map(abs, (x, -2x))
    ts[min(k, 6)] + k + lo * hi + a + b + π * x + (iszero(x) ? 1.0 : 0.0)
end
js_beta_inc_inv_x(a, b, p) = first(beta_inc_inv(a, b, p))
js_beta_inc_inv_y(a, b, p) = last(beta_inc_inv(a, b, p))

# Irrational constants written in the source, through Julia's methods for them:
# arithmetic rounds the constant to `Float64`; a comparison rounds it up or down
# (`Float64(π) < π` holds) and `==` is false; `ℯ^x` is `exp(x)`, `sin(π)` is
# `0.0`; a constant of another package's `@irrational` rounds as Base's do; a
# helper given `π` is specialized on it.
Base.@irrational js_half_turn 1.57079632679489661923132169163975144 big(π) / 2
js_t_density(x, nu) =
    exp(loggamma((nu + 1) / 2) - loggamma(nu / 2) - log(nu * π) / 2 - (nu + 1) / 2 * log1p(x^2 / nu))
js_irr_arith(x, n) = x * π + π * x - 2π * n + (n + π) - x / π + inv(π) * x - π * x * x + 2 * π * x +
                     sqrt(2π) * x + max(x, π) - min(x, ℯ) + x^π + (x > 0) * π
js_irr_compare(x) = (x < π ? 1 : 0) + (x > π ? 2 : 0) + (x <= ℯ ? 4 : 0) + (x == π ? 8 : 0) +
                    (π >= x ? 16 : 0) + (x != ℯ ? 32 : 0)
js_irr_exact(x) = ℯ^x + log(ℯ) * x + sin(π) + cos(π) * x + ℯ^2
js_irr_custom(x) = x * js_half_turn + (x < js_half_turn ? 1.0 : 0.0) + (x > js_half_turn ? 2.0 : 0.0)
js_scaled_by(a, c) = c * a
js_circle(r) = js_scaled_by(r^2, π)
js_turns(v) = v / ℯ

@kernel js_scalar(a, b) = begin
    c = a * b + exp(-a) / (1 + b^2)
    return c
end
@kernel js_fused(x, a, b) = begin
    y = a .* x .+ b
    z = exp.(-y) .* sin.(x) ./ (1 .+ x .^ 2)
    return z
end
@kernel js_helpers(x) = begin
    p = js_piecewise.(x)
    e = js_early.(abs.(x) .+ 0.5)
    q = p .+ e
    return q
end
@kernel js_loops(x0, r, n, target) = begin
    y = js_logistic_map(x0, r, n)
    w = js_cube_root(target, 0.0, max(target, 1.0))
    return y
end
@kernel js_reductions(x) = begin
    s = sum(x)
    m = maximum(x)
    t = js_weighted_total(x)
    f = x[1] - x[end]
    return s
end
@kernel js_switch(x, flag, k) = begin
    y = ifelse(flag, x * k, x / k)
    return y
end
@kernel js_difference(x, y) = begin
    d = x .- y
    return d
end
@kernel js_comprehension(x) = begin
    y = js_squares(x)
    return y
end
@kernel js_branch(a, b) = begin
    y = if a > b
        a - b
    else
        log(b - a - 1)
    end
    return y
end
@kernel js_scan(x, decay, initial) = begin
    trajectory = scan(x; init = initial) do previous, observation
        current = decay * previous + observation
        (current, current)
    end
    return trajectory
end
@kernel js_plate(y, mu) = begin
    lp = plate(y, mu) do yi, mi
        -(yi - mi)^2 / 2
    end
    total = sum(lp)
    return total
end
@kernel js_fixed_ranges(x, k, flag, n) = begin
    h = js_half_life(k)
    s = js_fixed_series(x)
    u = js_unchecked(x)
    e = js_range_sum(1:0, x) + js_range_sum(1:3, x)
    c = js_chosen_range(flag, n)
    p = plate(1:4) do i
        i * x
    end
    return h
end
@kernel js_fixed_vectors(x) = begin
    p = js_powers(x)
    s = js_scaled(x)
    q = js_squares_scaled(x)
    m = js_multiples(x)
    return p
end
@kernel js_keywords(x, y, share, weight) = begin
    r = js_kw_split(x, y; share)
    lo = r.lo
    hi = r.hi
    t = js_kw_total(r; weight)
    return (; lo, hi, t)
end
@kernel js_record(a, b, s) = begin
    rec = (; a, b, s)
    m = js_record_mean(rec)
    return m
end
@kernel js_keyword_errors(x) = begin
    u = js_kw_unknown(x)
    m = js_kw_missing(x)
    p = js_kw_specific(x)
    q = js_kw_none_use(x)
    return u
end
@kernel js_destructure(x) = begin
    out = js_buffers(x)
    return out
end
@kernel js_views(x, w) = begin
    means = js_window_means(x, w)
    second = js_view_second(x)
    return (; means, second)
end
@kernel js_gather(x, index) = begin
    g = x[index]
    e = exp.(x[index])
    return (; g, e)
end
@kernel js_soft_logistic(x, s) = begin
    a = js_soft.(x, s)
    b = logistic.(x)
    return (; a, b)
end
@kernel js_c_call(x) = begin
    y = js_c_sin(x)
    return y
end
@kernel js_fma(a, b) = begin
    y = fma(a, b, 1.0)
    return y
end
@kernel js_core_ops(x, n, k) = begin
    p = js_poly(x)
    t = js_pick(x, k)
    q = js_promoted(x, n)
    s = js_static_parameter(x, n)
    r = js_parity(x)
    return p
end
# Special functions: `erf`, `erfc`, `erfcx`, `gamma` and `logabsgamma` (with
# `loggamma`) from the stdlib-js files in `src/js_vendor`; `logerfc`,
# `logbeta`, `log1pexp` and `log1mexp` from their own Julia code.
@kernel js_erfs(x) = begin
    e = erf(x)
    c = erfc(x)
    cx = erfcx(x)
    lc = logerfc(x)
    return c
end
@kernel js_gamma(a) = begin
    g = gamma(a)
    return g
end
@kernel js_loggamma(a) = begin
    lg = loggamma(a)
    return lg
end
@kernel js_logabsgamma(a) = begin
    v = js_logabsgamma_value(a)
    s = js_logabsgamma_sign(a)
    return v
end
@kernel js_logbeta(a, b) = begin
    y = logbeta(a, b)
    return y
end
@kernel js_logexp(x) = begin
    p = log1pexp(x)
    m = log1mexp(-abs(x))
    return p
end
@kernel js_steps(n, x, a, s, b) = begin
    c = js_countdown(n, x)
    r = js_stride(a, s, b)
    return c
end
@kernel js_tuple_tools_k(x) = begin
    y = js_tuple_tools(x)
    return y
end
# The incomplete beta function and its inverse come from stdlib-js; the
# incomplete gamma function and its inverse translate from their Julia code.
@kernel js_beta_incs(a, b, x) = begin
    t = beta_inc(a, b, x)
    p = first(t)
    q = last(t)
    return p
end
@kernel js_beta_inc_inv(a, b, p) = begin
    x = js_beta_inc_inv_x(a, b, p)
    y = js_beta_inc_inv_y(a, b, p)
    return x
end
@kernel js_gamma_incs(a, x) = begin
    t = gamma_inc(a, x)
    p = first(t)
    q = last(t)
    return p
end
@kernel js_gamma_inc_inv(a, p) = begin
    x = gamma_inc_inv(a, p, 1 - p)
    return x
end
@kernel js_tdist(x, nu) = begin
    p = js_tdist_cdf(x, nu)
    return p
end
@kernel js_irrationals(x, nu, n) = begin
    d = js_t_density.(x, nu)
    w = x .* π
    a = js_irr_arith.(x, n)
    c = js_irr_compare.(x)
    e = js_irr_exact.(x)
    h = js_irr_custom.(x)
    q = js_circle.(x)
    s = π * x
    u = x * π
    t = js_turns(x)
    return d
end
# A number times an array is a broadcast in Julia's methods, matrices included.
@kernel js_irr_matrix(A) = begin
    p = π * A
    s = A * ℯ
    r = A / π
    return p
end

# Matrices cross the boundary and are read as arrays of row arrays.
js_corner(m) = m[size(m, 1), size(m, 2)] + m[firstindex(m, 1), 1]
js_extent(m) = length(m) + 10 * last(axes(m, 2)) + 100 * lastindex(m, 1)
@kernel js_matrix_reads(m, j) = begin
    col = m[:, j]
    row = m[1, :]
    corner = js_corner(m)
    extent = js_extent(m)
    return col
end
@kernel js_matrix_identity(m) = begin
    same = copy(m)
    rows = size(m, 1)
    cols = size(m, 2)
    return same
end
@kernel js_int_matrix(m, flags) = begin
    total = sum(m[:, 1]) + m[1, size(m, 2)]
    flag = flags[1, 2]
    return total
end
# Vector literals: `[a, b]` and `T[a, b]`, with Julia's promotions.
@kernel js_literals(x, flag, k) = begin
    v = [x, 2.0, 3x]
    w = Float64[1, k, x]
    b = [flag, !flag]
    i = [k, 2k]
    p = [k, x]
    return v
end

# Matrix arithmetic. A Hilbert-space approximate Gaussian process surface: a
# basis matrix times its spectral weights at a length scale (one, or one per
# axis), on a synthetic sine basis.
const JS_SQRT2PI = sqrt(2pi)
js_gp_weights(lambda, sigma, rho::Number) =
    (sigma * sqrt(rho * JS_SQRT2PI)^size(lambda, 2)) .* exp.(-0.25 .* (rho * rho) .* vec(sum(lambda; dims = 2)))
js_gp_weights(lambda, sigma, rho::AbstractVector) =
    (sigma * prod(sqrt.(rho .* JS_SQRT2PI))) .* exp.(-0.25 .* (lambda * (rho .* rho)))
js_gp_field(phi, lambda, sigma, rho, z) = phi * (js_gp_weights(lambda, sigma, rho) .* z)
@kernel js_gp_surface(phi, lambda, sigma, rho, z) = begin
    field = js_gp_field(phi, lambda, sigma, rho, z)
    return field
end
const GP_HALF = 1.3
const GP_LAMBDA = [((k * pi) / (2GP_HALF))^2 + 0.1 * a for k in 1:6, a in 1:2]
const GP_PHI = [sin(sqrt(GP_LAMBDA[k, 1]) * (x + GP_HALF)) / sqrt(GP_HALF) for x in range(-1, 1; length = 9), k in 1:6]
const GP_Z = collect(range(-1.0, 1.0; length = 6))
@kernel js_matrix_products(A, B, v, w) = begin
    ab = A * B
    atv = A' * v
    vta = (v' * A)'
    vv = v' * v
    outer = v * w'
    btb = transpose(B) * B
    pb = permutedims(B)
    chain = A * B * w
    twice = 2.0 * v
    return ab
end
@kernel js_matrix_arithmetic(A, B, x) = begin
    s = 2.0 * A .- A / 4
    total = A + B
    gap = A - B
    flip = A' + B'
    neg = -A
    scaled = A * x
    return s
end
@kernel js_matrix_sums(A, v) = begin
    total = sum(A)
    product = prod(A)
    cols = sum(A; dims = 1)
    rows = sum(A; dims = 2)
    colprod = prod(A; dims = 1)
    rowprod = prod(A; dims = 2)
    flat = vec(A)
    vprod = prod(v)
    return total
end
@kernel js_matrix_extrema(A) = begin
    top = maximum(A)
    bottom = minimum(A)
    rowmax = maximum(A; dims = 2)
    colmin = minimum(A; dims = 1)
    return top
end
@kernel js_matrix_rowmax(A) = begin
    rowmax = maximum(A; dims = 2)
    return rowmax
end
function js_doubled(A)
    out = zeros(size(A, 1), size(A, 2))
    out .= 2 .* A
    out
end
@kernel js_matrix_broadcasts(A, v, w, k) = begin
    byrow = A .* v
    bycol = A .+ w'
    grid = v .* w'
    shifted = exp.(A) .- k
    mask = A .> 0.5
    kept = ifelse.(A .> 0, A, 0.0)
    ranged = (1:size(A, 1)) .+ A
    doubled = js_doubled(A)
    return byrow
end
@kernel js_matrix_fills(m, n, x) = begin
    z = zeros(m, n)
    o = ones(Int, m, n)
    f = fill(x, m, n)
    return z
end
@kernel js_int_algebra(M, u, flags) = begin
    mu = M * u
    rows = sum(M; dims = 2)
    count = sum(flags)
    colcount = sum(flags; dims = 1)
    return mu
end
# A range of length one broadcast against a longer vector.
@kernel js_range_broadcast(x, a, b) = begin
    y = (a:b) .+ x
    return y
end
@kernel js_solve(A, v) = begin
    y = A \ v
    return y
end
@kernel js_dims3(A) = begin
    y = sum(A; dims = 3)
    return y
end
# `const` global arrays the code reads are module-level arrays holding their
# values (snag js-source-stack-1e1692d1: a broadcast over one recursed until
# the stack overflowed, and a run-time use of one was refused).
const JS_LEVELS = [50.0, 100.0, 200.0, 300.0, 400.0]
const JS_COUNTS = [1, -2, 3]
const JS_FLAGS = [true, false, true]
const JS_GRID = [1.0 2.0; 3.0 4.0; 5.0 6.0]
const JS_NOROWS = zeros(0, 2)
const JS_BUFFER = zeros(3)
const JS_WORDS = ["a", "b"]
function js_share_above(line, v)
    n = 0
    for i in eachindex(v)
        v[i] > line && (n += 1)
    end
    n / length(v)
end
function js_fill_buffer(x)
    JS_BUFFER .= x
    sum(JS_BUFFER)
end
@kernel js_constants(scale, x, i, line) = begin
    dose = JS_LEVELS
    scaled = scale .* JS_LEVELS
    shifted = x .+ JS_LEVELS
    times = scale * JS_LEVELS
    picked = JS_LEVELS[i]
    total = scale * sum(JS_LEVELS)
    counts = 2 .* JS_COUNTS
    signs = ifelse.(JS_FLAGS, scale, -scale)
    grid = scale .* JS_GRID
    projected = JS_GRID * [scale, 1.0]
    norows = JS_NOROWS .+ scale
    share = js_share_above(line, JS_LEVELS)
    nested = scale .* (JS_LEVELS .+ 1.0)
    fused = x .+ (2 .* 3)
    return scaled
end
@kernel js_buffered(x) = begin
    t = js_fill_buffer(x)
    return t
end
@kernel js_words(scale) = begin
    y = scale .* length.(JS_WORDS)
    return y
end

const V = Vector{Float64}
const M = Matrix{Float64}
const GP_TYPES = (; phi = M, lambda = M, sigma = Float64, rho = Float64, z = V)
const MA = [0.5 -1.0 2.0 0.25; 1.5 0.0 -0.75 3.0; -2.0 0.6 1.0 -0.5]
const CASES = [
    (:js_scalar, js_scalar, (:a, :b), :c, (; a = Float64, b = Float64),
     [(0.5, 2.0), (-3.0, 0.0), (1e-300, 1e10)]),
    (:js_fused, js_fused, (:x, :a, :b), :z, (; x = V, a = Float64, b = Float64),
     [(collect(range(-3, 3; length = 41)), 0.7, -0.2), (Float64[], 1.0, 2.0), ([1.0], 0.0, 0.0)]),
    (:js_helpers, js_helpers, (:x,), :q, (; x = V), [(collect(range(-2, 3; length = 51)),), (Float64[],)]),
    (:js_loops, js_loops, (:x0, :r, :n, :target), (:y, :w), (; x0 = Float64, r = Float64, n = Int, target = Float64),
     [(0.3, 3.7, 0, 2.0), (0.3, 3.7, 1, 0.5), (0.3, 3.7, 50, 27.0), (0.2, 2.5, -4, 1.0)]),
    (:js_reductions, js_reductions, (:x,), (:s, :m, :t, :f), (; x = V),
     [(collect(range(-2, 3; length = 2001)),), ([4.0],), (Float64[],)]),
    (:js_switch, js_switch, (:x, :flag, :k), :y, (; x = Float64, flag = Bool, k = Int),
     [(3.0, true, 4), (3.0, false, 4), (3.0, false, 0)]),
    (:js_difference, js_difference, (:x, :y), :d, (; x = V, y = V),
     [([1.0, 2.0, 3.0], [0.5, 0.25, 0.125]), ([1.0, 2.0, 3.0], [1.0]), ([2.0], [1.0, 2.0, 3.0]),
      ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])]),
    (:js_comprehension, js_comprehension, (:x,), :y, (; x = V), [([1.0, -2.0, 3.5],), (Float64[],)]),
    (:js_keywords, js_keywords, (:x, :y, :share, :weight), (:lo, :hi, :t),
     (; x = V, y = V, share = Float64, weight = Float64),
     [([1.0, 2.0, 3.0], [0.5, 0.25, 0.0], 0.3, 2.0), (Float64[], Float64[], 0.1, 1.0), ([1.0, 2.0], [1.0], 0.5, 1.0)]),
    (:js_record, js_record, (:a, :b, :s), :m, (; a = V, b = V, s = Float64),
     [([1.0, 2.0], [3.0], 0.5), (Float64[], [2.0, 4.0], 1.0)]),
    (:js_keyword_errors, js_keyword_errors, (:x,), (:u, :m, :p, :q), (; x = Float64),
     [(1.5,), (-1.0,), (12.0,), (-6.0,)]),
    (:js_destructure, js_destructure, (:x,), :out, (; x = V), [([1.0, -2.0, 3.5],), (Float64[],)]),
    (:js_views, js_views, (:x, :w), (:means, :second), (; x = V, w = Int),
     [(collect(1.0:12.0), 3), (collect(1.0:13.0), 4), ([1.0], 1), (collect(range(0, 1; length = 5000)), 2500)]),
    (:js_gather, js_gather, (:x, :index), (:g, :e), (; x = V, index = Vector{Int}),
     [([1.0, 2.0, 3.0], [3, 1, 1, 2]), ([1.0, 2.0], Int[]), ([1.0, 2.0], [1, 3])]),
    (:js_soft_logistic, js_soft_logistic, (:x, :s), (:a, :b), (; x = V, s = Float64),
     [(collect(range(-800, 800; length = 33)), 1.0), ([-40.0, -20.0, 0.0, 20.0, 35.0], 0.5)]),
    # The arm not taken is not evaluated: `log` of a negative number is never
    # computed for a > b.
    (:js_branch, js_branch, (:a, :b), :y, (; a = Float64, b = Float64), [(5.0, 1.0), (1.0, 4.0), (1.0, 1.5)]),
    (:js_scan, js_scan, (:x, :decay, :initial), :trajectory, (; x = V, decay = Float64, initial = Float64),
     [([1.0, 0.0, 0.0, 2.0], 0.5, 0.0), (Float64[], 0.5, 1.0), (collect(range(0, 1; length = 1000)), 0.9, 2.0)]),
    (:js_plate, js_plate, (:y, :mu), :total, (; y = V, mu = V),
     [([1.0, 2.0, 3.0], [0.5, 2.5, 3.0]), ([1.0, 2.0], [0.0]), (Float64[], Float64[]), ([1.0, 2.0, 3.0], [1.0, 2.0])]),
    (:js_fixed_ranges, js_fixed_ranges, (:x, :k, :flag, :n), (:h, :s, :u, :e, :c, :p),
     (; x = Float64, k = Float64, flag = Bool, n = Int),
     [(0.5, 0.3, true, 10), (-0.9, 2.0, false, 0), (2.0, 1e-3, false, 7)]),
    (:js_fixed_vectors, js_fixed_vectors, (:x,), (:p, :s, :q, :m), (; x = Float64), [(1.5,), (-0.25,)]),
    # a position past the tuple raises `BoundsError`; above 2^53 every float is even
    (:js_core_ops, js_core_ops, (:x, :n, :k), (:p, :t, :q, :s, :r), (; x = Float64, n = Int, k = Int),
     [(0.5, 3, 2), (-1.5, 0, 1), (2.0, 7, 3), (3.0, 1, 3), (3.0, 1, 4), (1e300, 2, 1), (2.0^53 + 2, 1, 1),
      (2.0^53 - 1, 1, 1), (4.5, -2, 1)]),
    (:js_erfs, js_erfs, (:x,), (:e, :c, :cx, :lc), (; x = Float64),
     [(x,) for x in (-Inf, -30.0, -5.0, -1.0, -1e-20, -0.0, 0.0, 1e-300, 0.3, 0.5, 0.84375, 1.25, 2.857, 6.0,
                     26.5, 27.5, 1e5, 1e300, Inf, NaN)]),
    # the poles and negative arguments of `gamma` raise `DomainError`, as does
    # `loggamma` where the gamma function is negative
    (:js_gamma, js_gamma, (:a,), :g, (; a = Float64),
     [(a,) for a in (-Inf, -2.0, -1.5, -0.5, -0.0, 0.0, 1e-300, 0.5, 1.0, 5.0, 23.5, 171.0, 172.0, Inf, NaN)]),
    (:js_loggamma, js_loggamma, (:a,), :lg, (; a = Float64),
     [(a,) for a in (-Inf, -40.5, -3.0, -1.5, -0.5, -1e-25, -0.0, 0.0, 1e-25, 0.5, 1.0, 2.0, 7.9, 8.1,
                     171.6, 1e17, 1e300, Inf, NaN)]),
    (:js_logabsgamma, js_logabsgamma, (:a,), (:v, :s), (; a = Float64),
     [(a,) for a in (-Inf, -40.5, -3.0, -2.5, -1.5, -0.5, -1e-25, -0.0, 0.0, 1e-25, 0.5, 2.0, 1e17, Inf, NaN)]),
    (:js_logbeta, js_logbeta, (:a, :b), :y, (; a = Float64, b = Float64),
     [(a, b) for a in (0.25, 0.5, 2.5, 40.0, 1e6) for b in (0.5, 3.0, 7.5, 9.0, 1e3)] ∪
     [(-1.0, -2.0), (-0.5, 2.0), (0.0, 1.0), (-2.5, 0.75)]),
    # a step of zero raises `ArgumentError`; empty ranges count none
    (:js_steps, js_steps, (:n, :x, :a, :s, :b), (:c, :r), (; n = Int, x = Float64, a = Int, s = Int, b = Int),
     [(5, 0.5, 1, 2, 9), (0, 2.0, 9, -3, 1), (-2, 1.0, 1, 1, 0), (3, -1.0, 4, -1, 4), (4, 1.5, 9, 2, 1),
      (2, 1.0, 1, 0, 3)]),
    (:js_tuple_tools_k, js_tuple_tools_k, (:x,), :y, (; x = Float64), [(x,) for x in (0.05, 0.5, 2.0, 0.0, -3.0)]),
    (:js_beta_incs, js_beta_incs, (:a, :b, :x), (:p, :q), (; a = Float64, b = Float64, x = Float64),
     [(a, b, x) for a in (0.0, 0.25, 2.5, 30.0, 1e4, -1.0, NaN) for b in (0.0, 0.5, 3.0)
      for x in (0.0, 1e-300, 1e-10, 0.3, 0.9, 1 - 1e-12, 1.0, 1.5, NaN)]),
    (:js_beta_inc_inv, js_beta_inc_inv, (:a, :b, :p), (:x, :y), (; a = Float64, b = Float64, p = Float64),
     [(a, b, p) for a in (0.25, 1.0, 2.5, 30.0, 5e5) for b in (0.5, 3.0)
      for p in (0.0, 1e-12, 0.05, 0.5, 0.95, 1 - 1e-12, 1.0, -0.1, 1.1)]),
    # large shapes included: there stdlib-js's `gammainc` is inaccurate, and the
    # module runs SpecialFunctions' own algorithm
    (:js_gamma_incs, js_gamma_incs, (:a, :x), (:p, :q), (; a = Float64, x = Float64),
     [(a, x) for a in (0.0, 1e-3, 0.5, 1.0, 2.5, 30.0, 1e4, -1.0, NaN)
      for x in (0.0, 1e-8, 0.5, 2.0, 30.0, 1e4, Inf, -1.0, NaN)] ∪
     [(a, a + k * sqrt(a)) for a in (1e3, 2e3, 1e5) for k in (-3, -1, 0, 1, 3)]),
    (:js_gamma_inc_inv, js_gamma_inc_inv, (:a, :p), :x, (; a = Float64, p = Float64),
     [(a, p) for a in (1e-3, 0.5, 1.0, 2.5, 30.0, 2e3, 1e4) for p in (0.0, 1e-12, 0.05, 0.5, 0.95, 1 - 1e-12, 1.0)]),
    (:js_logexp, js_logexp, (:x,), (:p, :m), (; x = Float64),
     [(x,) for x in (-Inf, -800.0, -40.0, -1.0, -0.0, 0.0, 1e-10, 0.5, 3.0, 40.0, 800.0, Inf, NaN)]),
    (:js_matrix_reads, js_matrix_reads, (:m, :j), (:col, :row, :corner, :extent), (; m = M, j = Int),
     [([1.0 2.0 3.0; 4.0 5.0 6.0], 2), ([1.0 2.0 3.0; 4.0 5.0 6.0], 4), ([1.0 2.0 3.0; 4.0 5.0 6.0], 0),
      (reshape([7.5], 1, 1), 1), (zeros(0, 3), 1)]),
    (:js_matrix_identity, js_matrix_identity, (:m,), (:same, :rows, :cols), (; m = M),
     [([1.0 -2.0; 3.5 4.0; NaN Inf],), (zeros(0, 3),), (zeros(2, 0),), (zeros(0, 0),)]),
    (:js_int_matrix, js_int_matrix, (:m, :flags), (:total, :flag), (; m = Matrix{Int}, flags = Matrix{Bool}),
     [([1 2; 3 4], [true false; false true]), (reshape([5], 1, 1), reshape([true], 1, 1))]),
    (:js_literals, js_literals, (:x, :flag, :k), (:v, :w, :b, :i, :p), (; x = Float64, flag = Bool, k = Int),
     [(0.5, true, 3), (-2.0, false, 0)]),
    (:js_gp_surface, js_gp_surface, (:phi, :lambda, :sigma, :rho, :z), :field, GP_TYPES,
     [(GP_PHI, GP_LAMBDA, 0.7, 0.9, GP_Z), (GP_PHI, GP_LAMBDA, 2.0, 0.1, GP_Z),
      (GP_PHI, GP_LAMBDA, 0.7, 0.9, GP_Z[1:5]), (GP_PHI[:, 1:5], GP_LAMBDA, 0.7, 0.9, GP_Z),
      (zeros(0, 6), GP_LAMBDA, 0.7, 0.9, GP_Z)]),
    (:js_gp_axes, js_gp_surface, (:phi, :lambda, :sigma, :rho, :z), :field, merge(GP_TYPES, (; rho = V)),
     [(GP_PHI, GP_LAMBDA, 0.7, [0.9, 0.4], GP_Z), (GP_PHI, GP_LAMBDA, 0.7, [0.9], GP_Z)]),
    (:js_matrix_products, js_matrix_products, (:A, :B, :v, :w),
     (:ab, :atv, :vta, :vv, :outer, :btb, :pb, :chain, :twice),
     (; A = M, B = M, v = V, w = V),
     [(MA[:, 1:2], MA[1:2, :], [1.0, -2.0, 0.5], [0.25, 1.0, -1.0, 2.0]),
      (zeros(0, 2), MA[1:2, :], Float64[], [0.25, 1.0, -1.0, 2.0]),
      (MA[:, 1:2], MA, [1.0, -2.0, 0.5], [0.25, 1.0, -1.0, 2.0]),
      (MA[:, 1:2], MA[1:2, :], [1.0, -2.0], [0.25, 1.0, -1.0, 2.0])]),
    (:js_matrix_arithmetic, js_matrix_arithmetic, (:A, :B, :x), (:s, :total, :gap, :flip, :neg, :scaled),
     (; A = M, B = M, x = Float64),
     [(MA, MA .^ 2, -1.5), (MA, MA[:, 1:3], 2.0), (zeros(0, 3), zeros(0, 3), 1.0)]),
    (:js_matrix_sums, js_matrix_sums, (:A, :v),
     (:total, :product, :cols, :rows, :colprod, :rowprod, :flat, :vprod), (; A = M, v = V),
     [(MA, [1.5, -2.0, 0.5]), ([-0.0 -0.0; -0.0 NaN], Float64[]), (zeros(0, 3), [2.0]),
      (zeros(3, 0), [1.0, 2.0]), (zeros(0, 0), Float64[]), (collect(reshape(range(-1, 1; length = 2400), 40, 60)), [1.0])]),
    (:js_matrix_extrema, js_matrix_extrema, (:A,), (:top, :bottom, :rowmax, :colmin), (; A = M),
     [(MA,), ([1.0 NaN 3.0; -0.0 0.0 -1.0],), (reshape([7.0], 1, 1),), (zeros(0, 3),)]),
    (:js_matrix_rowmax, js_matrix_rowmax, (:A,), :rowmax, (; A = M),
     [(MA,), (zeros(0, 3),), (zeros(3, 0),), (zeros(0, 0),)]),
    (:js_matrix_broadcasts, js_matrix_broadcasts, (:A, :v, :w, :k),
     (:byrow, :bycol, :grid, :shifted, :mask, :kept, :ranged, :doubled), (; A = M, v = V, w = V, k = Float64),
     [(MA, [1.0, -2.0, 0.5], [0.25, 1.0, -1.0, 2.0], 1.0), (MA, [3.0], [2.0], 0.5),
      (MA[1:1, :], [1.0, -2.0, 0.5], [0.25, 1.0, -1.0, 2.0], 0.0), (MA, [1.0, 2.0], [0.25, 1.0, -1.0, 2.0], 1.0),
      (zeros(0, 4), Float64[], [0.25, 1.0, -1.0, 2.0], 1.0)]),
    (:js_matrix_fills, js_matrix_fills, (:m, :n, :x), (:z, :o, :f), (; m = Int, n = Int, x = Float64),
     [(2, 3, 1.5), (0, 3, 1.0), (2, 0, -1.0), (-1, 2, 1.0)]),
    (:js_int_algebra, js_int_algebra, (:M, :u, :flags), (:mu, :rows, :count, :colcount),
     (; M = Matrix{Int}, u = Vector{Int}, flags = Matrix{Bool}),
     [([1 2 3; 4 5 6], [1, -1, 2], [true false; true true]), ([1 2; 3 4], [1, 2, 3], reshape([true], 1, 1))]),
    (:js_range_broadcast, js_range_broadcast, (:x, :a, :b), :y, (; x = V, a = Int, b = Int),
     [([1.0, 2.0, 3.0], 1, 1), ([1.0, 2.0, 3.0], 2, 4), ([1.0, 2.0, 3.0], 1, 2), ([5.0], 1, 3)]),
    # the floats either side of `π`, `ℯ` and `π/2`; a negative base of `x^π`
    # raises `DomainError`
    (:js_irrationals, js_irrationals, (:x, :nu, :n), (:d, :w, :a, :c, :e, :h, :q, :s, :u, :t),
     (; x = V, nu = Float64, n = Int),
     [([0.0, 0.5, 1.0, float(ℯ), prevfloat(float(ℯ)), nextfloat(float(ℯ)), float(π), prevfloat(float(π)),
        nextfloat(float(π)), Float64(π / 2), nextfloat(Float64(π / 2)), 10.0, 1e3, Inf, NaN], 4.5, 3),
      (Float64[], 1.0, 0), ([2.0, 3.0], 0.25, -7), ([1.0, -2.0], 3.0, 1)]),
    (:js_irr_matrix, js_irr_matrix, (:A,), (:p, :s, :r), (; A = M),
     [([1.0 -2.0 0.5; 3.0 NaN -Inf],), (reshape([7.5], 1, 1),), (zeros(0, 3),), (zeros(2, 0),)]),
    # a length-two `x` against five levels is a DimensionMismatch, a sixth level a BoundsError
    (:js_constants, js_constants, (:scale, :x, :i, :line),
     (:dose, :scaled, :shifted, :times, :picked, :total, :counts, :signs, :grid, :projected, :norows, :share,
      :nested, :fused),
     (; scale = Float64, x = V, i = Int, line = Float64),
     [(100.0, [1.0, 2.0, 3.0, 4.0, 5.0], 2, 150.0), (-0.5, [1.0], 5, 1e3), (2.0, [1.0, 2.0], 1, 0.0),
      (2.0, [1.0], 6, 0.0)]),
    (:js_buffered, js_buffered, (:x,), :t, (; x = V), [([1.0, 2.0, 3.0],), ([4.0],), ([1.0, 2.0],)]),
]

agree(a::Real, b::Real) = (isnan(a) && isnan(b)) || isapprox(a, b; rtol = 1e-12, atol = 1e-15)
agree(a::AbstractVector, b::AbstractVector) = length(a) == length(b) && all(agree.(a, b))
agree(a::AbstractMatrix, b::AbstractMatrix) = size(a) == size(b) && all(agree.(a, b))
agree(a, b) = false

@testset "js_source" begin
    @testset "emits a module for $name" for (name, k, have, want, types, _) in CASES
        source = ReactiveKernels.js_source(k; have, want, types, name)
        @test occursin("export function " * string(name) * "(" * join(string.(have), ", ") * ")", source)
        @test occursin("export const INPUTS", source) && occursin("export class JuliaError", source)
    end

    @testset "Node agrees with native: $name ($vectors)" for (name, k, have, want, types, cases) in CASES,
                                                             vectors in (:array, :float64array)
        if !NODE
            @test_skip "no node on the PATH"
            continue
        end
        source = ReactiveKernels.js_source(k; have, want, types, name)
        wants = want isa Symbol ? (want,) : want
        native = prepare(k; have, want = wants)
        outcomes = ReactiveKernels.js_run(source, name, cases; vectors)
        for (c, r) in zip(cases, outcomes)
            expected = try
                native(c...)
            catch e
                e
            end
            @test !r.inputs_mutated
            if expected isa Exception
                # The same Julia exception type, thrown by the JavaScript.
                @test r.julia_error && r.error == string(nameof(typeof(expected)))
            else
                @test r.error === nothing
                r.error === nothing || continue
                # The native shape and types, and the native values.
                @test typeof(r.value) == typeof(expected)
                if want isa Symbol
                    @test agree(expected, r.value)
                else
                    @test all(agree(expected[j], r.value[j]) for j in eachindex(wants))
                end
            end
        end
    end

    @testset "branching helpers need no annotation" begin
        source = ReactiveKernels.js_source(js_helpers; have = (:x,), want = :q, types = (; x = V))
        @test occursin("function js_piecewise(x) {", source)
        @test occursin("function js_early(x) {", source)
        @test occursin("if (x <= 1) {", source)
    end

    @testset "data-dependent iteration stays a loop" begin
        loops = ReactiveKernels.js_source(js_loops; have = (:x0, :r, :n, :target), want = :y,
                                          types = (; x0 = Float64, r = Float64, n = Int, target = Float64))
        @test occursin("while (true)", loops)
        scan = ReactiveKernels.js_source(js_scan; have = (:x, :decay, :initial), want = :trajectory,
                                         types = (; x = V, decay = Float64, initial = Float64))
        body = scan[findfirst("export function kernel(", scan)[1]:end]
        body = body[1:findfirst("\n}", body)[1]]
        @test count("for (let", body) == 1
        @test occursin("x.length", body)
    end

    @testset "a range written in a helper is read as literals" begin
        source = ReactiveKernels.js_source(js_fixed_ranges; have = (:x, :k, :flag, :n), want = (:h, :s, :u, :e, :c, :p),
                                           types = (; x = Float64, k = Float64, flag = Bool, n = Int))
        body(name) = (i = findfirst("function " * name * "(", source);
                      source[i[1]:findnext("\n}\n", source, i[1])[end]])
        half = body("js_half_life")
        @test occursin("< 60 ?", half) && !occursin("_rk_range", half) && !occursin(".stop", half)
        # dead code is left out: the loop over `1:0`, the branch on `JS_CHECKED`
        # and, in the helper specialized on `1:0`, everything after `isempty`
        @test count("while (true)", body("js_fixed_series")) == 3
        @test !occursin("negative input", source) && !occursin("if (", body("js_unchecked"))
        @test occursin(r"function js_range_sum(_\d+)?\(x\) \{\n(  let \w+;\n)*  return x;\n\}", source)
        # a range assigned in a branch is a value at run time
        @test occursin("_rk_range(1, 3)", body("js_chosen_range"))
    end

    @testset "keyword calls, named tuples and destructuring" begin
        body(source, name) = (i = findfirst("function " * name * "(", source);
                              source[i[1]:findnext("\n}\n", source, i[1])[end]])
        kw = ReactiveKernels.js_source(js_keywords; have = (:x, :y, :share, :weight), want = (:lo, :hi, :t),
                                       types = (; x = V, y = V, share = Float64, weight = Float64))
        # the keyword sorter is inlined: the body function takes the keyword
        # values, the default `scale = 2 * share` computed in the module
        @test occursin(r"function js_kw_split_\d+\(share, offset, scale, x, y\) \{", kw)
        @test occursin("2 * share", kw) && !occursin("kwcall", kw) && !occursin("structdiff", kw)
        # a named tuple is an object, read by field
        @test occursin("return { lo: lo, hi: hi };", body(kw, "js_kw_split_" * match(r"js_kw_split_(\d+)", kw)[1]))
        # the loop over the data stays a loop; the destructured generator over
        # `1:3`, written in the source, is three allocations
        buffers = body(ReactiveKernels.js_source(js_destructure; have = (:x,), want = :out, types = (; x = V)),
                       "js_buffers")
        @test count("new Array(n).fill(0)", buffers) == 3 && count("while (true)", buffers) == 2
        # thresholds known from the argument's type are literals
        soft = ReactiveKernels.js_source(js_soft_logistic; have = (:x, :s), want = (:a, :b), types = (; x = V, s = Float64))
        @test occursin("x < -745.1332191019412", soft) && !occursin("indexed_iterate", soft)
    end

    @testset "irrational constants are translated through Julia's methods" begin
        source = ReactiveKernels.js_source(js_irrationals; have = (:x, :nu, :n), want = (:d, :a, :c, :e, :h, :q),
                                           types = (; x = V, nu = Float64, n = Int))
        body(name) = (i = findfirst("function " * name * "(", source);
                      source[i[1]:findnext("\n}\n", source, i[1])[end]])
        @test occursin("nu * 3.141592653589793", body("js_t_density")) && !occursin("Irrational", source)
        # `2 * π * x` folds from the left: `2 * π` is a constant
        @test occursin("6.283185307179586 * x", body("js_irr_arith")) && !occursin("afoldl", source)
        # `x < π` compares with π rounded down, `x > π` with π rounded up
        compare = body("js_irr_compare")
        @test occursin("x <= 3.141592653589793", compare) && occursin("3.1415926535897936 <= x", compare)
        @test occursin("Math.exp(x)", body("js_irr_exact")) && !occursin("_rk_pow", body("js_irr_exact"))
        @test occursin("1.5707963267948968 <= x", body("js_irr_custom"))
        # a helper given `π` takes only the other argument
        @test occursin(r"function js_scaled_by\(a\) \{", source)
    end

    @testset "unsupported constructs are named" begin
        e = try
            ReactiveKernels.js_source(js_c_call; have = (:x,), want = :y, types = (; x = Float64))
        catch err
            err
        end
        @test e isa ReactiveKernels.JSEmissionError
        @test occursin("C function `sin`", e.message) && occursin("js_c_sin", e.message)
        e = try
            ReactiveKernels.js_source(js_fma; have = (:a, :b), want = :y, types = (; a = Float64, b = Float64))
        catch err
            err
        end
        @test e isa ReactiveKernels.JSEmissionError
        @test occursin("`fma`", e.message)
        # A Distributions.jl law is named, with the RK distribution stack to use instead.
        e = try
            ReactiveKernels.js_source(js_tdist; have = (:x, :nu), want = :p, types = (; x = Float64, nu = Float64))
        catch err
            err
        end
        @test e isa ReactiveKernels.JSEmissionError
        @test occursin("Distributions.jl law", e.message) && occursin("standard_student_t(nu).cdf(z)", e.message)
        @test occursin("js_tdist_cdf", e.message)
        e = try
            ReactiveKernels.js_source(js_solve; have = (:A, :v), want = :y, types = (; A = M, v = V))
        catch err
            err
        end
        @test e isa ReactiveKernels.JSEmissionError
        @test occursin("`\\`", e.message)
        e = try
            ReactiveKernels.js_source(js_dims3; have = (:A,), want = :y, types = (; A = M))
        catch err
            err
        end
        @test e isa ReactiveKernels.JSEmissionError
        @test occursin("dims = 1", e.message) && occursin("(dims = 3,)", e.message)
    end

    @testset "a const global array is a module-level array" begin
        source = ReactiveKernels.js_source(js_constants; have = (:scale, :x, :i, :line),
                                           want = (:dose, :scaled, :grid, :norows, :share),
                                           types = (; scale = Float64, x = V, i = Int, line = Float64))
        # emitted once each, named after the global, its values exact
        @test count("const _rk_const_JS_LEVELS = [50.0, 100.0, 200.0, 300.0, 400.0];", source) == 1
        @test count("_rk_const_JS_LEVELS = ", source) == 1
        @test occursin("const _rk_const_JS_GRID = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];", source)
        @test occursin("const _rk_const_JS_NOROWS = Object.assign([], { cols: 2 });", source)
        # the broadcast is one loop over the constant
        @test occursin("_rk_const_JS_LEVELS.length", source)
        # a global the code writes stays writable, as natively
        buffered = ReactiveKernels.js_source(js_buffered; have = (:x,), want = :t, types = (; x = V))
        @test occursin("const _rk_const_JS_BUFFER = [", buffered) && !occursin("Object.freeze", buffered)
        # an array without a JavaScript value is refused by name
        e = try
            ReactiveKernels.js_source(js_words; have = (:scale,), want = :y, types = (; scale = Float64))
        catch err
            err
        end
        @test e isa ReactiveKernels.JSEmissionError
        @test occursin("`Vector{String}` is needed at run time", e.message) && occursin("JS_WORDS", e.message)
    end

    @testset "matrix arithmetic reads rows in place" begin
        source = ReactiveKernels.js_source(js_gp_surface; have = (:phi, :lambda, :sigma, :rho, :z), want = :field,
                                           types = GP_TYPES)
        @test occursin("_rk_matvec(phi, ", source) && occursin("_rk_reduce2(lambda, ", source)
        @test occursin("_rk_vec(", source) && !occursin("_rk_transpose", source)
    end

    @testset "a module copies only the special functions it uses" begin
        gamma_source = ReactiveKernels.js_source(js_gamma; have = (:a,), want = :g, types = (; a = Float64))
        @test occursin("var _rk_stdlib_gamma=", gamma_source)
        @test !occursin("_rk_stdlib_erf", gamma_source) && !occursin("_rk_stdlib_gammaln", gamma_source)
        @test occursin("// @stdlib/math-base-special-gamma ", gamma_source)
        @test occursin("Apache License, Version 2.0", gamma_source)
        erfs = ReactiveKernels.js_source(js_erfs; have = (:x,), want = (:e, :c, :cx, :lc), types = (; x = Float64))
        @test count("var _rk_stdlib_erfc=", erfs) == 1 && !occursin("_rk_stdlib_gamma", erfs)
        # the incomplete gamma function is translated, not vendored
        incgamma = ReactiveKernels.js_source(js_gamma_incs; have = (:a, :x), want = (:p, :q),
                                             types = (; a = Float64, x = Float64))
        @test !occursin("gammainc", incgamma) && occursin("while (true)", incgamma)
        @test occursin("var _rk_stdlib_betainc=", ReactiveKernels.js_source(js_beta_incs; have = (:a, :b, :x),
                       want = (:p, :q), types = (; a = Float64, b = Float64, x = Float64)))
        # each vendored file defines its one function and carries the notices
        # of the implementation it follows
        for (name, notice) in ((:erf, "Sun Microsystems"), (:erfc, "Sun Microsystems"),
                               (:erfcx, "Massachusetts Institute of Technology"),
                               (:gamma, "Cephes"), (:gammaln, "Sun Microsystems"),
                               (:betainc, "Boost Software License"), (:betaincinv, "Boost Software License"))
            text = read(joinpath(pkgdir(ReactiveKernels), "src", "js_vendor", string(name, ".js")), String)
            @test occursin("var _rk_stdlib_" * string(name) * "=", text)
            @test occursin(notice, text)
        end
    end

    @testset "inputs and names" begin
        @test_throws ArgumentError ReactiveKernels.js_source(js_switch; have = (:x, :flag, :k), want = :y,
            types = (; x = Float64, flag = Bool, k = Int), name = :switch)
        if NODE
            source = ReactiveKernels.js_source(js_scalar; have = (:a, :b), want = :c,
                                               types = (; a = Float64, b = Float64), name = :scalar)
            r = only(ReactiveKernels.js_run(source, :scalar, [([1.0], 2.0)]))
            @test r.julia_error && r.error == "ArgumentError" && occursin("`a` must be a number", r.message)
            # a matrix input is an array of rows of one length
            source = ReactiveKernels.js_source(js_matrix_identity; have = (:m,), want = :same,
                                               types = (; m = M), name = :identity)
            ragged, scalar = ReactiveKernels.js_run(source, :identity, [([[1.0, 2.0], [3.0]],), (1.0,)])
            @test ragged.julia_error && ragged.error == "ArgumentError" && occursin("rows of one length", ragged.message)
            @test scalar.julia_error && scalar.error == "ArgumentError" && occursin("array of rows", scalar.message)
            @test occursin("{ name: \"m\", type: \"Matrix{Float64}\" }", source)
        else
            @test_skip "no node on the PATH"
        end
    end
end

end
