using ReactiveKernels
using Test
using LinearAlgebra: dot
using LogExpFunctions: log1mexp
using SpecialFunctions: besseli, besselix

include(joinpath(@__DIR__, "jax_python.jl"))

# JAX source emission (todo 1l3nni7): the emitted module keeps the program's
# structure, plates, scans, lazy branches and loops, and computes the native
# values; `jax.grad` of it matches the native program's finite differences.

@kernel jax_gaussian(x::Vector{Float64}, mu::Float64, logsigma::Float64) = begin
    sigma = exp(logsigma)
    z = (x .- mu) ./ sigma
    lp = -0.5 * sum(abs2, z) - length(x) * (logsigma + 0.5 * log(2pi))
    return lp
end

@kernel jax_plate_scan(y::Vector{Float64}, x::Vector{Float64}, a::Float64, b::Float64,
                       logsigma::Float64) = begin
    sigma = exp(logsigma)
    lp = plate(y, x) do yi, xi
        if sigma > 0
            -0.5 * ((yi - a - b * xi) / sigma)^2 - log(sigma)
        else
            -Inf
        end
    end
    filtered = scan(y; init = 0.0) do acc, yi
        next = 0.9 * acc + b * yi
        (next, next)
    end
    total = sum(lp) + sum(filtered)
    return total
end

@traceable function jax_test_recurrence(R::AbstractVector, rho::Real, kappa::Real)
    T = promote_type(eltype(R), typeof(rho), typeof(kappa))
    W = Vector{T}(undef, length(R))
    C, D = one(T), zero(T)
    recover = 1 - exp(-rho)
    for t in eachindex(R)
        C = C + (1 - C) * recover
        dep = C * (1 - exp(-kappa * R[t]))
        C = C - dep
        W[t] = D + dep
        D = 0.5 * D + dep
    end
    return W
end

@kernel jax_loop(rho::Float64, R::Vector{Float64}, kappa::Float64) = begin
    W = jax_test_recurrence(R, rho, kappa)
    s = sum(abs2, W)
    return s
end

@kernel jax_generators(w::Float64, x::Vector{Float64}) = begin
    s = sum(w * x[i]^2 for i in 1:length(x); init = 0.0)
    v = [exp(w * x[i]) for i in eachindex(x)]
    t = s + sum(v) + sum(x[i] for i in eachindex(x) if x[i] > 0)
    return t
end

# A helper with a counted `while` loop (a `fori_loop`, which `jax.grad`
# reverses) and a branch whose arms assign an outer local.
@traceable function jax_test_traced(x::Real)
    i = 0
    h = one(x)
    while i < 20
        h = h * (1 + x / (i + 1))
        i = i + 1
    end
    result = zero(x)
    if x > 0
        result = h
    else
        result = -h
    end
    return result
end
@kernel jax_traced(x::Float64) = begin
    y = jax_test_traced(x)
    return y
end
# A `while` loop whose trip count is data: `jax.lax.while_loop`.
@traceable function jax_test_doubling(x::Real)
    h = one(x)
    n = 0
    while h < x
        h = 2 * h
        n = n + 1
    end
    return n * h
end
@kernel jax_doubling(x::Float64) = begin
    y = jax_test_doubling(x)
    return y
end

# A tuple or `NamedTuple` local that a loop or a branch carries stays a tuple,
# each element coerced alone (snag jax-source-tuple-6f1e9e4c): rebound by a
# data-dependent conditional, with an integer field, past a ragged row's
# length, assigned in an `if` arm, and carried by a `while` loop.
@traceable function jax_test_tuple_loop(s::Tuple, w::Real, n::Integer, m::Integer)
    s = (s[1] + w, s[2])
    for i in 2:m
        s = i <= n ? (0.9 * s[1] + w * s[2], 0.5 * s[2] + s[1]) : s
    end
    s
end
@kernel jax_tuple_loop(w::Float64, x::Float64, n::Int) = begin
    s = jax_test_tuple_loop((x, 1.0), w, n, 4)
    t = s[1] + 2 * s[2]
    return t
end
@traceable function jax_test_named_loop(w::Real, m::Integer)
    s = (level = w, count = 0)
    for i in 1:m
        s = (level = 0.5 * s.level + w, count = s.count + i)
    end
    s.level * s.count
end
@kernel jax_named_loop(w::Float64) = begin
    t = jax_test_named_loop(w, 3)
    return t
end
@traceable function jax_test_row_loop(r::AbstractVector, a::Real)
    s = (0.0, 1.0)
    for t in eachindex(r)
        s = (s[1] + a * r[t] * s[2], 0.5 * s[2])
    end
    s[1] + s[2]
end
@kernel jax_tuple_rows(a::Float64, R::Vector{Vector{Float64}}) = begin
    lp = plate(R) do r
        jax_test_row_loop(r, a)
    end
    total = sum(lp)
    return total
end
@traceable function jax_test_tuple_branch(x::Real)
    s = (x, 2.0)
    if x > 0
        s = (s[2], s[1])
    end
    k = 0
    while k < 3
        s = (s[1] * 0.5, s[2] + x)
        k = k + 1
    end
    s[1] - s[2]
end
@kernel jax_tuple_branch(x::Float64) = begin
    t = jax_test_tuple_branch(x)
    return t
end
@traceable function jax_test_tuple_while(x::Real)
    h = (one(x), 0)
    while h[1] < x
        h = (2 * h[1], h[2] + 1)
    end
    h[2] * h[1]
end
@kernel jax_tuple_while(x::Float64) = begin
    t = jax_test_tuple_while(x)
    return t
end

# Typed allocations, `hcat`, `permutedims`, a numeric-type broadcast, and a
# postfix on a compound operand (`size(x .- w, 1)`).
@kernel jax_alloc(x::Vector{Float64}, w::Float64) = begin
    n = length(x)
    D = Float64.(hcat(ones(Float64, n), x))
    P = permutedims(D)
    c = fill(w, 2)
    s = sum(D * c) + sum(P .* w) + size(x .- w, 1)
    return s
end

# A loop whose `if` arms end in values of different types (an assignment, a
# destructuring): the branch carries only the locals they assign.
@traceable function jax_test_triangle(z::AbstractVector, k::Integer)
    out = ones(Float64, k)
    i, j, d = 1, 2, 1.0
    for p in 1:length(z)
        d = d * sqrt(1 - z[p]^2)
        if i + 1 < j
            i = i + 1
        else
            out[j] = d
            i, j, d = 1, j + 1, 1.0
        end
    end
    return out
end
@kernel jax_triangle(z::Vector{Float64}) = begin
    v = jax_test_triangle(tanh.(z), 3)
    s = sum(log.(v))
    return s
end

# A scan whose step outputs named numbers, read field by field by a plate.
@kernel jax_named_scan(y::Vector{Float64}, b::Float64) = begin
    rows = scan(y; init = 0.0) do acc, yi
        next = b * acc + yi
        (next, (level = next, step = yi - next))
    end
    level = plate(rows) do row
        row.level
    end
    s = sum(level .^ 2)
    return s
end

# Runtime checks: one under a branch that a vmapped plate cell evaluates for
# every lane (it must hold for lanes that do not take the branch), and ones
# on the data.
@traceable function jax_test_positive_log(xi::Real)
    if xi > 0
        ReactiveKernels._runtime_check(xi > 0, ArgumentError("xi must be positive"))
        log(xi)
    else
        zero(xi)
    end
end
@traceable function jax_test_checked_sum(lp::AbstractVector)
    ReactiveKernels._runtime_check(length(lp) > 0, ArgumentError("lp is empty"))
    m = sum(lp)
    ReactiveKernels._runtime_check(m < 10, ArgumentError("m is too large"))
    return m
end
@kernel jax_checked(x::Vector{Float64}) = begin
    lp = plate(x) do xi
        jax_test_positive_log(xi)
    end
    m = jax_test_checked_sum(lp)
    return m
end

# A plain helper whose source is recorded for `jax_source` as written.
"The logarithm above one, the offset below it."
ReactiveKernels.@_record_source function jax_test_recorded(x)
    result = zero(x)
    if x > 1
        result = log(x)
    else
        result = x - 1
    end
    return result
end
@kernel jax_recorded(x::Float64) = begin
    y = jax_test_recorded(x)
    return y
end

# `clamp` is translated from Base's own straight-line code, whose
# `convert(promote_type(...), x)` passes a type as a value (snag
# `jax-source-clamp-af26e700`). The cell is an HSGP basis function of a
# clamped input; `clamp.(x, 0, 1)` converts integer bounds.
@kernel jax_clamp(x::Vector{Float64}, lo::Float64, hi::Float64, w::Float64) = begin
    v = plate(x) do xi
        sin(sqrt(w) * ((clamp(xi, lo, hi) - 0.5) + 2.0)) / sqrt(2.0)
    end
    c = clamp.(x, 0, 1)
    s = sum(v) + sum(c .^ 2)
    return s
end
@kernel jax_clamp_scalar(t::Float64) = begin
    y = 3.0 * clamp(t, -1.0, 2.0)
    return y
end
# Type arguments that a plain helper's lowered code passes as values: a global
# (`Int`, `Float64`) and a type-level call (`promote_type`).
jax_test_typed(x) = floor(Int, x) * sum(ones(promote_type(typeof(x), Int), 2)) +
                    convert(Float64, 3)
@kernel jax_typed(x::Float64) = begin
    y = jax_test_typed(x)
    return y
end

# A straight-line plain helper, translated from its lowered code: its literal
# powers and `eltype` are type values there (`Val{2}`, `Float64`), which bind
# no Python local.
jax_test_plain(w, x) = zero(eltype(x)) + sum(x) * w^2 + w^-1
@kernel jax_plain(w::Float64, x::Vector{Float64}) = begin
    s = jax_test_plain(w, x)
    return s
end

# A plain helper's lowered code reads module globals (`M.f` is
# `getproperty(M, :f)`), builds `Val` tags, and forwards a Vararg method's
# arguments with a splat (`Core._apply_iterate`). StanBlocks' `@deffun`
# methods do all of it: `StanBlocks.jcall(Val{:f}(), _JuliaCallScope{M}(),
# args...)` finds `f` with `getglobal(M, :f)` and splats the arguments into it
# (snag jax-source-stanb-06917e13).
module JaxTestDispatch
struct Scope{M} end
callee(::Scope{M}, ::Val{name}) where {M,name} = getglobal(M, name)
call(tag::Val, scope::Scope, args::Vararg{Any,N}) where {N} = callee(scope, tag)(args...)
scaled(x, w) = JaxTestDispatch.call(Val{:*}(), Scope{Base}(), x, w)
applied(f, x) = f(x) + x
pair_sum(x, w) = (t = (x, w); +(t...))
end
jax_test_dispatch(w, x) =
    JaxTestDispatch.call(Val{:scaled}(), JaxTestDispatch.Scope{JaxTestDispatch}(), w, sum(x)) +
    JaxTestDispatch.call(Val(:applied), JaxTestDispatch.Scope{JaxTestDispatch}(), sin, w) +
    JaxTestDispatch.pair_sum(w, x[1])
@kernel jax_dispatch(w::Float64, x::Vector{Float64}) = begin
    s = jax_test_dispatch(w, x)
    return s
end

@traceable jax_test_count(x::AbstractVector, n::Integer) = begin
    acc = zero(eltype(x))
    for i in 1:n
        acc += x[i]
    end
    acc
end
@kernel jax_data_count(x::Vector{Float64}, n::Int) = begin
    s = jax_test_count(x, n)
    return s
end

# Guards over values known while JAX traces are decided then: a literal a
# caller passes to a spliced child, a shape, a helper of a literal. Their
# calls are NumPy's, which JAX does not stage (snag jax-source-liter-a9e6219e).
@kernel jax_guarded_shift(x, jitter) = begin
    jit = begin
        isfinite(jitter) && jitter >= 0 ||
            throw(ArgumentError("jitter must be finite and nonnegative, got $jitter"))
        Float64(jitter)
    end
    total = sum(x) + jit
end
@kernel jax_literal_guard(x::Vector{Float64}) = begin
    t = jax_guarded_shift(x, 1.0e-9)
    return t
end
@kernel jax_literal_guard_fails(x::Vector{Float64}) = begin
    t = jax_guarded_shift(x, -1.0)
    return t
end
@kernel jax_static_guards(x::Vector{Float64}) = begin
    scale = begin
        log(length(x)) >= 0 || throw(ArgumentError("x is empty"))
        log(length(x))
    end
    offset = begin
        log1mexp(-1.0) < 0 || throw(ArgumentError("log1mexp(-1) is not negative"))
        log1mexp(-1.0)
    end
    total = scale * sum(x) + offset
    return total
end

# A `:` dimension is the remaining length, also inside a dimension tuple and
# in a plain helper's lowered code (snag jax-source-resha-959acdfb).
jax_test_row(v) = reshape(v, 1, :)
@kernel jax_reshapes(x::Vector{Float64}, a::Float64) = begin
    row = reshape(a .* x, 1, :)
    cols = reshape(a .* x, :, 2)
    rows = reshape(a .* x, (2, :))
    same = reshape(cols, size(rows))
    lowered = jax_test_row(a .* x)
    total = sum(row .* row) + sum(cols[:, 2] .^ 2) + sum(rows[1, :]) + a * sum(same[2, :]) +
            sum(lowered)
    return total
end
@kernel jax_ragged_reshape(R::Vector{Vector{Float64}}, a::Float64) = begin
    cells = plate(R) do r
        sum(reshape(a .* r, 1, :))
    end
    total = sum(cells)
    return total
end
@kernel jax_two_colons(x::Vector{Float64}) = begin
    r = reshape(x, :, :)
    total = sum(r)
    return total
end

# Data preparation whose shapes follow the data's values (decision 1ooufd4):
# with `data`, it is `prepare_data`, host NumPy with Python control flow.
# Helpers with `break`, `continue` or several ranges record their source with
# `@_record_source`, as RKPPL's own preparation helpers do.
ReactiveKernels.@_record_source function jax_prep_codes(x::AbstractVector, levels::AbstractVector)
    codes = Vector{Int}(undef, length(x))
    for (i, v) in enumerate(x)
        c = 0
        for (j, lv) in enumerate(levels)
            if isequal(v, lv)
                c = j
                break
            end
        end
        codes[i] = c
    end
    return codes
end
ReactiveKernels.@_record_source function jax_prep_lanes(y::AbstractVector{<:Integer}, K)
    out = Int[]
    for (i, count) in enumerate(min.(y, K .- 1)), j in 1:count
        push!(out, i)
    end
    return out
end
ReactiveKernels.@_record_source function jax_prep_first(keys::AbstractVector)
    seen = Dict{eltype(keys),Int}()
    codes = zeros(Int, length(keys))
    for i in eachindex(keys)
        haskey(seen, keys[i]) && (codes[i] = seen[keys[i]]; continue)
        codes[i] = get!(seen, keys[i], length(seen) + 1)
    end
    return codes
end
ReactiveKernels.@_record_source function jax_prep_search(x::AbstractVector, target)
    found = 0
    i = 1
    while i <= length(x)
        if x[i] >= target
            found = i
            break
        end
        i += 1
    end
    return found
end
@kernel jax_prep(g::Vector{String}, y::Vector{Int}, theta::Vector{Float64}) = begin
    codes = jax_prep_codes(g, ["a", "b", "c"])
    lanes = jax_prep_lanes(y, 3)
    first = jax_prep_first(g)
    at = jax_prep_search(y, 2)
    total = sum(theta[codes]) + 2 * sum(theta[first]) + sum(lanes) * theta[2] + at * theta[1]
    return total
end
# A ragged value read through a plate, then flattened.
@kernel jax_prep_flat(R::Vector{Vector{Float64}}, a::Float64) = begin
    scaled = plate(R) do r
        a .* r
    end
    flat = reduce(vcat, scaled)
    total = sum(abs2, flat)
    return total
end
# A plate over the group index whose cells read a shared ragged input by it
# and allocate at its rows' lengths, then flattened.
@kernel jax_prep_index(R::Vector{Vector{Float64}}, a::Float64) = begin
    groups = eachindex(R)
    cells = plate(groups, Ref(R)) do i, rows
        (a .* ones(length(rows[i]))) .+ rows[i]
    end
    flat = reduce(vcat, cells)
    total = sum(flat .* flat)
    return total
end
# Data-only comprehensions, do-blocks, lets, short circuits, scans and plates.
@kernel jax_prep_misc(x::Vector{Float64}, theta::Float64) = begin
    positive = [v^2 for v in x if v > 0]
    kept = map(x) do v
        let w = v + 1
            w > 1 && v < 3 ? w : 0.0
        end
    end
    running = scan(x; init = 0.0) do acc, v
        next = acc + v
        (next, next)
    end
    doubled = plate(x) do v
        2 * v
    end
    total = theta * (sum(positive) + sum(kept) + sum(running) + sum(doubled))
    return total
end
# The flattening BRM emits (reactivekernels-use §4c): type tests decided
# while emitting, then `reduce(vcat, cells)`.
@kernel jax_prep_cells(R::Vector{Vector{Float64}}, a::Float64) = begin
    cells = plate(R) do r
        a .* r .+ 1
    end
    flat = eltype(cells) <: Number ? cells : eltype(cells) <: AbstractVector ?
        (isempty(cells) ? eltype(eltype(cells))[] : reduce(vcat, cells)) : cells
    total = sum(flat .* flat) + length(flat)
    return total
end
# Order on the host (snag jax-host-sort-un-0edfe976): the sorted distinct
# values of a data column, built with `push!` and an `isequal` run test, then
# each entry's 1-based position among them.
@traceable function jax_order_levels(x, extra)
    levels = Float64[]
    for j in eachindex(x)
        v = convert(Float64, x[j])
        if isempty(levels) || !isequal(v, last(levels))
            push!(levels, v)
        end
    end
    for j in eachindex(extra)
        push!(levels, extra[j])
    end
    unique!(sort!(levels))
end
@traceable function jax_order_positions(levels, x)
    idx = zeros(Int, length(x))
    for j in eachindex(x)
        idx[j] = searchsortedfirst(levels, x[j])
    end
    idx
end
# Julia's `isless` and `isequal` on signed zeros and NaN, which NumPy's own
# `sort`, `unique` and `searchsorted` do not follow.
@traceable function jax_order_signs(z)
    s = sort(z)
    o = copy(z)
    sort!(o)
    zeros_equal = 0
    for v in z
        if isequal(v, 0.0)
            zeros_equal += 1
        end
    end
    length(unique(z)) + 10 * searchsortedfirst(s, 0.0) + 100 * searchsortedlast(o, 0.0) +
        1000 * zeros_equal + 10000 * length(unique!(sort(z)))
end
@kernel jax_prep_order(x::Vector{Float64}, extra::Vector{Float64}, z::Vector{Float64},
                       theta::Vector{Float64}) = begin
    levels = jax_order_levels(x, extra)
    idx = jax_order_positions(levels, x)
    upto = [searchsortedlast(levels, v) for v in x]
    signs = jax_order_signs(z)
    total = sum(theta[idx]) + sum(levels) + sum(upto) * theta[1] + signs * theta[2]
    return total
end
# `unique!` of a NumPy array, which keeps its length: no host translation, and
# the JAX functions cannot compute the recipe either.
@traceable function jax_order_dedupe(x)
    v = sort(x)
    unique!(v)
end
@kernel jax_prep_unique_array(x::Vector{Float64}, a::Float64) = begin
    d = jax_order_dedupe(x)
    total = a * sum(d)
    return total
end

# A gather through a ragged integer row is padded as the row is, so the
# cell's reduction counts its valid entries only (snag jax-ragged-index-b8420d16):
# by the group index, by the row itself, and through `view` (the call form).
@kernel jax_gather_indexed(x::Vector{Float64}, R::Vector{Vector{Int}}, a::Float64) = begin
    lp = plate(eachindex(R), Ref(R), Ref(x)) do i, rs, xs
        sum(a .* xs[rs[i]] .^ 2)
    end
    total = sum(lp)
    return total
end
@kernel jax_gather_rows(x::Vector{Float64}, R::Vector{Vector{Int}}, a::Float64) = begin
    lp = plate(R, Ref(x), Ref(a)) do r, xs, a
        v = xs[r]
        w = view(xs, r)
        sum(abs2, v) + dot(v, w) + v' * w + a * prod(v) + length(w)
    end
    total = sum(lp)
    return total
end
# Reductions of a ragged row itself that take a function or two operands.
@kernel jax_ragged_row_sums(a::Float64, Y::Vector{Vector{Float64}}) = begin
    lp = plate(Y, Ref(a)) do y, a
        w = a .* y
        sum(abs2, w) + dot(w, y) + w' * y
    end
    total = sum(lp)
    return total
end
# A gather through a ragged row beside a whole axis would pad an inner axis.
@kernel jax_gather_matrix_rows(M::Matrix{Float64}, R::Vector{Vector{Int}}, b::Vector{Float64}) = begin
    lp = plate(R, Ref(M), Ref(b)) do r, Ms, bs
        sum(Ms[r, :] * bs)
    end
    total = sum(lp)
    return total
end
# A broadcast `isequal` on the host: its generated scalar helper must not
# take the name of the fixed `_rk_isequal` helper's host twin.
@kernel jax_prep_isequal(g::Vector{String}, theta::Float64) = begin
    hits = isequal.(g, "b")
    total = theta * sum(hits)
    return total
end
# `get!` as a statement: its write runs although its value is discarded.
ReactiveKernels.@_record_source function jax_prep_declared(x::AbstractVector, levels::AbstractVector)
    positions = Dict{eltype(levels),Int}()
    for (j, lv) in enumerate(levels)
        get!(positions, lv, j)
    end
    codes = zeros(Int, length(x))
    for (i, v) in enumerate(x)
        codes[i] = get(positions, v, 0)
    end
    return codes
end
@kernel jax_prep_declared_sum(g::Vector{String}, theta::Vector{Float64}) = begin
    codes = jax_prep_declared(g, ["c", "a", "b"])
    total = sum(theta[codes] .^ 2)
    return total
end
# Data-only reshapes with a `:` dimension (BRM's GP pair locations).
@kernel jax_prep_reshapes(x::Vector{Float64}, a::Float64) = begin
    row = reshape(x, 1, :)
    rows = reshape(x, (2, :))
    total = a * sum(row .* row) + a^2 * sum(rows[1, :])
    return total
end
# A count-then-fill row partition (BRM's): the elements of a list of arrays
# the source allocates are written in place, with no copy.
@traceable function jax_prep_fill(groups::AbstractVector, cells::AbstractVector)
    lookup = Dict{eltype(cells),Int}()
    for k in eachindex(cells)
        lookup[cells[k]] = k
    end
    counts = zeros(Int, length(cells))
    for r in eachindex(groups)
        i = get(lookup, groups[r], 0)
        if i != 0
            counts[i] += 1
        end
    end
    rows = [Vector{Int}(undef, counts[k]) for k in eachindex(cells)]
    for k in eachindex(counts)
        counts[k] = 0
    end
    for r in eachindex(groups)
        i = get(lookup, groups[r], 0)
        if i != 0
            counts[i] += 1
            rows[i][counts[i]] = r
        end
    end
    rows
end
@kernel jax_prep_partition(groups::Vector{String}, cells::Vector{String}, a::Float64) = begin
    rows = jax_prep_fill(groups, cells)
    sizes = plate(rows) do r
        a * length(r) + sum(r)
    end
    total = sum(sizes)
    return total
end
# Elements written in a list whose arrays the source did not allocate itself
# (here arithmetic on an input's rows): the list and its elements are copied
# first, as an input array is before its first write.
@traceable function jax_prep_rescale(R::AbstractVector)
    rows = [2 .* R[k] for k in eachindex(R)]
    for k in eachindex(rows)
        for j in eachindex(rows[k])
            rows[k][j] = rows[k][j] * j
        end
    end
    rows
end
@kernel jax_prep_rescaled(R::Vector{Vector{Float64}}, a::Float64) = begin
    rows = jax_prep_rescale(R)
    cells = plate(rows) do r
        a * sum(r)
    end
    total = sum(cells)
    return total
end
# Conditions decided while emitting (type tests, a tuple's length) take their
# arm alone in an `if`, `elseif`, `? :`, `&&` and `||`: the untaken arms have
# no method for string labels (`Int128` of a string).
@traceable jax_prep_label_span(x::AbstractVector) = Int128(maximum(x)) - Int128(minimum(x))
@traceable function jax_prep_label_count(x::AbstractVector, axes::Tuple)
    if eltype(x) <: Integer
        n = Int(jax_prep_label_span(x)) + length(x)
    elseif length(axes) > 0
        n = Int(jax_prep_label_span(x))
    else
        n = length(x)
    end
    length(axes) > 0 && jax_prep_label_span(x) > 0 && (n = 0)
    wide = eltype(x) <: AbstractString || jax_prep_label_span(x) > 4
    m = eltype(x) <: Integer ? jax_prep_label_span(x) : 2
    n + m + 10 * wide
end
@kernel jax_prep_labels(g::Vector{String}, a::Float64) = begin
    n = jax_prep_label_count(g, ())
    total = a * n
    return total
end
@kernel jax_prep_int_labels(g::Vector{Int}, a::Float64) = begin
    n = jax_prep_label_count(g, ())
    total = a * n
    return total
end

# A derivative rule called from a plate cell: the JAX module computes its
# primal cut, whose operations a lowering's table holds as source records.
@kernel jax_rule_graph(a::Float64, b::Float64) = begin
    s::Float64 = a + b
    e::Float64 = exp(a * b)
    y::Float64 = e + log(s)
    dy_da::Float64 = b * e + 1 / s
    dy_db::Float64 = a * e + 1 / s
    return y, dy_da, dy_db
end
const jax_test_rule = scalar_derivative_rule(jax_rule_graph; primal = :y,
    partials = (a = :dy_da, b = :dy_db), name = :jax_test_rule)
@kernel jax_rule_plate(x::Vector{Float64}, a::Float64) = begin
    lp = plate(x) do xi
        jax_test_rule(a, xi)
    end
    total = sum(lp)
    return total
end
# `view(v, :)` in a plain helper's lowered code (a whole-axis slice).
jax_test_row(v::AbstractVector) = reshape(view(v, :), 1, length(v))
@kernel jax_view_row(x::Vector{Float64}, a::Float64) = begin
    r = jax_test_row(a .* x)
    total = sum(r * transpose(r))
    return total
end

# A kernel object whose method calls its own endpoint in a lazy arm (the
# shape of a location-scale family's CDF): the arm's source captures the
# prepared endpoint, and a cell that reaches it through another branch (a
# censoring bound) calls the generated body that unpacks those captures.
@kernel jax_obj_square() = begin
    logcdf(z::Float64)::Float64 = -z^2
end
@kernel jax_obj_scaled(standard, location::Float64, scale::Float64) = begin
    standardized(x::Float64)::Float64 = (x - location) / scale
    logcdf(x::Float64)::Float64 = begin
        z::Float64 = scale == 0 && x == location ? inv(scale) : standardized(x)
        standard.logcdf(z)
    end
end
@kernel jax_obj_law = jax_obj_scaled(jax_obj_square)
@kernel jax_obj_scores(x::Vector{Float64}, location::Float64, scale::Float64) = begin
    values = plate(x) do xi
        jax_obj_law(location, scale).logcdf(xi)
    end
    total = sum(values)
    return total
end
@kernel jax_obj_bounded(x::Vector{Float64}, location::Float64, scale::Float64, lo::Float64) = begin
    values = plate(x) do xi
        xi <= lo ? jax_obj_law(location, scale).logcdf(lo) : -(xi - location)^2
    end
    total = sum(values)
    return total
end

# A scan's per-step outputs that are tuples or named tuples (held as the
# tuple of their columns), read by a generator, by an element (`out[end]`)
# and by field; a scan seeded by `merge`; a reduction over `dims`.
@kernel jax_scan_tuple(xs::Vector{Float64}, a::Float64) = begin
    out = scan(xs; init = 0.0) do c, x
        n = a * c + x
        (n, (n, n * n))
    end
    total = sum(o[2] for o in out)
    return total
end
@kernel jax_scan_named(xs::Vector{Float64}, a::Float64) = begin
    out = scan(xs; init = 0.0) do c, x
        n = a * c + x
        (n, (level = n, square = n * n))
    end
    total = sum(o.square for o in out) + out[end].level
    return total
end
@kernel jax_scan_merge(xs::Vector{Float64}, a::Float64, u0::Float64) = begin
    init = (u = u0,)
    out = scan(xs; init = merge(init, (b = 0.0,))) do c, x
        n = (u = a * c.u + x, b = c.b + x)
        (n, n.u + n.b)
    end
    total = sum(out)
    return total
end
@kernel jax_dims(M::Matrix{Float64}, a::Float64) = begin
    f = vec(sum(a .* M; dims = 2)) .+ vec(maximum(M; dims = 1))[1]
    total = sum(f .^ 2)
    return total
end

# RK's tie-exact `logaddexp` (mixture densities) and integer `div` (a mode
# index `div(b - 1, k) % m + 1`).
@kernel jax_tie_div(x::Vector{Float64}, a::Float64, k::Int) = begin
    lse = plate(x) do xi
        ReactiveKernels._tie_exact_logaddexp(a * xi, -xi)
    end
    q = div(length(x) - 1, k) % 3 + 1
    total = sum(lse) + q * a
    return total
end

# Scans whose step reads the outputs so far (`history`): a vector of leading
# entries, and a scalar fill read up to the step's own index.
@kernel jax_history_leading(xs::Vector{Float64}, a::Float64) = begin
    ys = scan(xs; init = 0.0, history = [1.0, 2.0]) do carry, x, earlier
        (carry, a * x + sum(earlier))
    end
    total = sum(ys .^ 2)
    return total
end
@kernel jax_history_indexed(xs::Vector{Float64}, a::Float64) = begin
    ys = scan(xs, eachindex(xs); init = 0.0, history = 0.0) do carry, x, j, earlier
        s = sum(earlier[i] * a for i in 1:j-1; init = 0.0)
        (carry + x, x + s)
    end
    total = sum(ys)
    return total
end

# A scan step output with vector fields (a K-state filter's `(lpd, w, m)`,
# ODLProject:ap3): each vector field stacks as the (steps, K) array of rows.
@kernel jax_scan_vector_fields(ys::Vector{Float64}, w0::Vector{Float64}, a::Float64) = begin
    K = length(w0)
    steps = scan(ys; init = (m = a .* w0, w = w0)) do st, y
        nm = [0.5 * st.m[k] + a * y * k for k in 1:K]
        raw = [st.w[k] * exp(-0.5 * (y - nm[k])^2) for k in 1:K]
        lp = log(sum(raw))
        nw = [raw[k] / sum(raw) for k in 1:K]
        ((m = nm, w = nw), (lpd = lp, w = nw, m = nm))
    end
    total = sum(s.lpd for s in steps) + sum(steps[end].w .* steps[end].m)
    return total
end

# Recorded `@inline` helpers (`@_record_source`): a shape test with `in` and
# `&&`, decided while tracing, and a fold over a data length, run as a loop.
ReactiveKernels.@_record_source @inline jax_fits(x, n) = ndims(x) <= 2 && size(x, 1) in (1, n)
ReactiveKernels.@_record_source @inline function jax_fold(x, n)
    total = x[1]
    for i in 2:n
        total = ReactiveKernels._tie_exact_logaddexp(total, x[i])
    end
    total
end
@kernel jax_shape_fold(x::Vector{Float64}, a::Float64) = begin
    n = length(x)
    checked = jax_fits(x, n) ? a .* x : throw(DimensionMismatch("x must have n rows"))
    total = jax_fold(checked, n)
    return total
end
# Julia's unit indices: past a vector's rank (a literal and a computed one),
# of a number, and `size` past the rank; `dropdims` of a summed axis.
@kernel jax_unit_index(x::Vector{Float64}, a::Float64) = begin
    cells = plate(eachindex(x)) do i
        x[i, 1] * a[1, 1] + x[i, (i - 1) ÷ length(x) + 1] + size(x, 2)
    end
    M = reshape(cells, 2, length(cells) ÷ 2)
    total = sum(dropdims(sum(M; dims = 1); dims = 1) .^ 2)
    return total
end
# `stack` in `prepare_data`: on the host a plate of vectors is a list of
# arrays (BRM's HSGP basis, `raw_basis[:, b]`).
@kernel jax_prep_stack(x::Vector{Float64}, a::Float64) = begin
    rows = plate(x) do xi
        plate(1:3) do b
            xi * b
        end
    end
    M = stack(rows; dims = 1)
    col = M[:, 2]
    total = a * sum(col)
    return total
end

@testset "jax_source: structure" begin
    src = jax_source(jax_plate_scan; have = (:a, :y, :x, :b, :logsigma), want = :total)
    text = string(src)
    @test only(src.functions).name == "kernel"
    @test first.(only(src.functions).inputs) == ["a", "y", "x", "b", "logsigma"]
    @test occursin("jax.vmap(", text)
    @test occursin("jax.lax.scan(", text)
    @test occursin("jax.lax.cond(", text)
    # Nothing iterates in Python: every loop is a JAX loop primitive, so the
    # module is the same program for every data length.
    @test !occursin(r"^\s*for "m, text)
    loop = string(jax_source(jax_loop; have = (:rho, :R, :kappa), want = :s))
    @test occursin("jax.lax.fori_loop(", loop)
    @test occursin(".at[", loop)
    @test !occursin(r"^\s*for "m, loop)
    generators = string(jax_source(jax_generators; have = (:w, :x), want = :t))
    @test occursin("jax.vmap(element", generators)
    @test occursin("jnp.where(kept", generators)
    @test !occursin(r"^\s*for "m, generators)
    @test ReactiveKernels._traceable_source(typeof(jax_test_recorded), Float64) !== nothing
    @test occursin("logarithm above one", string(@doc jax_test_recorded))
    checked = string(jax_source(jax_checked; have = (:x,), want = :m))
    @test occursin("jax.debug.callback(", checked)
    @test occursin("_rk_check(jnp.logical_or(", checked)
    traced = string(jax_source(jax_traced; have = (:x,), want = :y))
    @test occursin("jax.lax.fori_loop(", traced)
    @test occursin("jax.lax.cond(", traced)
    @test occursin("jax.lax.while_loop(", string(jax_source(jax_doubling; have = (:x,), want = :y)))
    # A tuple carry enters the loop as a tuple of coerced elements, the
    # structure the body returns, never as one array of the tuple.
    tupled = string(jax_source(jax_tuple_loop; have = (:w, :x, :n), want = :t))
    @test occursin("((jnp.asarray(s[0], dtype=jnp.float64), jnp.asarray(s[1], dtype=jnp.float64)),)",
                   tupled)
    @test !occursin("jnp.asarray(s, ", tupled)
    named = string(jax_source(jax_named_loop; have = (:w,), want = :t))
    @test occursin("jnp.asarray(s[1], dtype=jnp.int64)", named)
    # Past a row's length each element keeps its value (`jnp.where` of a
    # tuple would stack it into an array).
    rows = string(jax_source(jax_tuple_rows; have = (:a, :R), want = :total))
    @test occursin(r"\(\(jnp\.where\(t <= r_n, s_3\[0\], s_2\[0\]\), jnp\.where\(t <= r_n, s_3\[1\], s_2\[1\]\)\),\)",
                   rows)
    @test occursin("jax.lax.while_loop(", string(jax_source(jax_tuple_while; have = (:x,), want = :t)))
    plain = string(jax_source(jax_plain; have = (:w, :x), want = :s))
    @test occursin("w ** 2", plain)
    @test !occursin(r"<[a-z]", plain)
    # Module reads, `Val` tags and Vararg splats resolve while emitting: only
    # the arithmetic they reach is emitted.
    dispatch = string(jax_source(jax_dispatch; have = (:w, :x), want = :s))
    @test occursin("jnp.sin(w)", dispatch)
    @test !occursin(r"<[a-z]", dispatch)
    @test !occursin("None", dispatch)
    # Static guards are Python conditionals over NumPy values, not `lax.cond`.
    guarded = string(jax_source(jax_literal_guard; have = (:x,), want = :t))
    @test occursin("np.isfinite(", guarded)
    @test occursin("import numpy as np", guarded)
    @test !occursin("jax.lax.cond(", guarded)
    static = string(jax_source(jax_static_guards; have = (:x,), want = :total))
    @test occursin("np.log(x.size)", static)
    @test occursin("_rk_log1mexp_np(-1.0)", static)
    @test !occursin("jax.lax.cond(", static)
    # A `:` dimension is NumPy's `-1`, and the reshape stays column-major.
    reshaped = string(jax_source(jax_reshapes; have = (:a, :x), want = :total))
    @test occursin("jnp.reshape(a * x, (1, -1), order=\"F\")", reshaped)
    @test occursin("jnp.reshape(a * x, (-1, 2), order=\"F\")", reshaped)
    @test occursin("tuple(-1 if d is None else d for d in (2, None))", reshaped)
    @test !occursin(r"<[a-z]", reshaped)
    # Not built yet: a reshape of a ragged value, whose padding has no shape
    # (docs/src/jax.md, "Limits"). It raises rather than reshaping the padding.
    @test_broken (jax_source(jax_ragged_reshape; have = (:a, :R), want = :total); true)
    # A module whose functions compute nothing with NumPy does not import it.
    @test !occursin("import numpy", text)
    # Several functions over one boundary, with module constants.
    multi = jax_source(jax_gaussian; have = (:mu, :x, :logsigma),
                       functions = (log_density = :lp, both = (:lp, :sigma),
                                    named = (density = :lp, scale = :sigma)),
                       constants = (DIMENSION = 2, NAMES = ["mu", "logsigma"]))
    @test [f.name for f in multi.functions] == ["log_density", "both", "named"]
    @test occursin("DIMENSION = 2", string(multi))
    @test occursin("return {\"density\": ", string(multi))
    # `clamp` keeps Julia's two `ifelse` selections; `jnp.clip` would give a
    # bound tie half the derivative.
    clamped = string(jax_source(jax_clamp; have = (:x, :lo, :hi, :w), want = :s))
    @test occursin("jnp.where(", clamped)
    @test !occursin("jnp.clip", clamped)
    @test !occursin(r"<[a-z]", clamped)
    typed = string(jax_source(jax_typed; have = (:x,), want = :y))
    @test occursin("jnp.floor(x)", typed)
    @test !occursin(r"<[a-z]", typed)
end

@testset "jax_source: refused constructs" begin
    # refused: a loop whose trip count is data has no static bound, and JAX
    # arrays and loops need static shapes (docs/src/jax.md, "Limits").
    @test_throws JAXEmissionError jax_source(jax_data_count; have = (:x, :n), want = :s)
    err = try
        jax_source(jax_data_count; have = (:x, :n), want = :s)
    catch e
        e
    end
    @test occursin("trip count depends on data", sprint(showerror, err))
    # refused: Julia itself rejects two `:` dimensions (`DimensionMismatch`).
    @test_throws DimensionMismatch prepare(jax_two_colons; have = (:x,), want = :total)([1.0, 2.0])
    @test_throws "more than one `Colon`" jax_source(jax_two_colons; have = (:x,), want = :total)
    # refused: a gather through a ragged row beside a whole axis pads an inner
    # axis that no reduction masks (its JAX value used to sum the padding).
    @test_throws "beside another kept dimension" jax_source(jax_gather_matrix_rows;
        have = (:b, :M, :R), want = :total)
end

@testset "jax_source: values and gradients" begin
    if !rk_jax_available()
        @info "RK_JAX_PYTHON is not set: JAX value and gradient parity checks skipped"
        @test_skip false
    else
        x = [0.3, -1.2, 0.8, 2.1, -0.4]
        native = prepare(jax_gaussian; have = (:mu, :x, :logsigma), want = :lp)
        src = jax_source(jax_gaussian; have = (:mu, :x, :logsigma), want = :lp)
        r = rk_jax_evaluate(src, "kernel", (0.2, x, -0.3); grad = true)
        @test r.value ≈ native(0.2, x, -0.3) rtol = 1e-12
        @test r.grad ≈ sum((x .- 0.2) ./ exp(-0.3)^2) rtol = 1e-10

        y = [0.5, -0.1, 1.3, 0.2]
        native = prepare(jax_plate_scan; have = (:a, :y, :x, :b, :logsigma), want = :total)
        src = jax_source(jax_plate_scan; have = (:a, :y, :x, :b, :logsigma), want = :total)
        r = rk_jax_evaluate(src, "kernel", (0.1, y, x[1:4], 0.7, -0.2); grad = true)
        f(a) = native(a, y, x[1:4], 0.7, -0.2)
        @test r.value ≈ f(0.1) rtol = 1e-12
        @test r.grad ≈ only(rk_fd_gradient(u -> f(only(u)), [0.1])) rtol = 1e-6

        native = prepare(jax_generators; have = (:w, :x), want = :t)
        src = jax_source(jax_generators; have = (:w, :x), want = :t)
        r = rk_jax_evaluate(src, "kernel", (0.4, x); grad = true)
        @test r.value ≈ native(0.4, x) rtol = 1e-12
        @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), x), [0.4])) rtol = 1e-6

        native = prepare(jax_traced; have = (:x,), want = :y)
        src = jax_source(jax_traced; have = (:x,), want = :y)
        @test rk_jax_evaluate(src, "kernel", (0.3,)).value ≈ native(0.3) rtol = 1e-12
        @test rk_jax_evaluate(src, "kernel", (-0.3,)).value ≈ native(-0.3) rtol = 1e-12
        @test rk_jax_evaluate(src, "kernel", (0.3,); grad = true).grad ≈
              only(rk_fd_gradient(u -> native(only(u)), [0.3])) rtol = 1e-6
        native = prepare(jax_doubling; have = (:x,), want = :y)
        src = jax_source(jax_doubling; have = (:x,), want = :y)
        @test rk_jax_evaluate(src, "kernel", (37.0,)).value ≈ native(37.0) rtol = 1e-12
        # Tuple carries (snag jax-source-tuple-6f1e9e4c): values and gradients
        # with the conditional taken for some steps, none, and all.
        native = prepare(jax_tuple_loop; have = (:w, :x, :n), want = :t)
        src = jax_source(jax_tuple_loop; have = (:w, :x, :n), want = :t)
        for (x0, n) in ((0.7, 3), (0.7, 1), (1.1, 4))
            r = rk_jax_evaluate(src, "kernel", (0.3, x0, n); grad = true)
            @test r.value ≈ native(0.3, x0, n) rtol = 1e-12
            @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), x0, n), [0.3])) rtol = 1e-6
        end
        R = [[1.0, 2.0, 3.0], [0.5], [-1.0, 0.25]]
        for (kernel, have, want, args) in ((jax_named_loop, (:w,), :t, (0.4,)),
                                           (jax_tuple_rows, (:a, :R), :total, (0.6, R)),
                                           (jax_tuple_branch, (:x,), :t, (0.8,)),
                                           (jax_tuple_branch, (:x,), :t, (-0.6,)))
            native = prepare(kernel; have, want)
            src = jax_source(kernel; have, want)
            r = rk_jax_evaluate(src, "kernel", args; grad = true)
            @test r.value ≈ native(args...) rtol = 1e-12
            @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), args[2:end]...), [args[1]])) rtol = 1e-6
        end
        native = prepare(jax_tuple_while; have = (:x,), want = :t)
        src = jax_source(jax_tuple_while; have = (:x,), want = :t)
        @test rk_jax_evaluate(src, "kernel", (37.0,)).value ≈ native(37.0) rtol = 1e-12
        native = prepare(jax_plain; have = (:w, :x), want = :s)
        src = jax_source(jax_plain; have = (:w, :x), want = :s)
        r = rk_jax_evaluate(src, "kernel", (0.7, x); grad = true)
        @test r.value ≈ native(0.7, x) rtol = 1e-12
        @test r.grad ≈ 2 * 0.7 * sum(x) - 0.7^-2 rtol = 1e-10
        native = prepare(jax_dispatch; have = (:w, :x), want = :s)
        src = jax_source(jax_dispatch; have = (:w, :x), want = :s)
        r = rk_jax_evaluate(src, "kernel", (0.7, x); grad = true)
        @test r.value ≈ native(0.7, x) rtol = 1e-12
        @test r.grad ≈ sum(x) + cos(0.7) + 2 rtol = 1e-10
        x6 = [x; 1.5]
        native = prepare(jax_reshapes; have = (:a, :x), want = :total)
        src = jax_source(jax_reshapes; have = (:a, :x), want = :total)
        r = rk_jax_evaluate(src, "kernel", (0.7, x6); grad = true)
        @test r.value ≈ native(0.7, x6) rtol = 1e-12
        @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), x6), [0.7])) rtol = 1e-6

        for (kernel, have, want, args) in (
                (jax_alloc, (:x, :w), :s, (x, 0.7)),
                (jax_triangle, (:z,), :s, ([0.3, -0.5, 0.2],)),
                (jax_named_scan, (:y, :b), :s, (y, 0.7)),
                (jax_checked, (:x,), :m, (x,)),
                (jax_recorded, (:x,), :y, (2.0,)), (jax_recorded, (:x,), :y, (0.5,)))
            native = prepare(kernel; have, want)
            src = jax_source(kernel; have, want)
            @test rk_jax_evaluate(src, "kernel", args).value ≈ native(args...) rtol = 1e-12
        end
        # A check that fails on the data raises when the module runs.
        src = jax_source(jax_checked; have = (:x,), want = :m)
        @test_throws "m is too large" rk_jax_evaluate(src, "kernel", ([5e4, 3e5],))

        for (kernel, want) in ((jax_literal_guard, :t), (jax_static_guards, :total))
            native = prepare(kernel; have = (:x,), want)
            src = jax_source(kernel; have = (:x,), want)
            r = rk_jax_evaluate(src, "kernel", (x,); grad = true)
            @test r.value ≈ native(x) rtol = 1e-12
            @test r.grad ≈ rk_fd_gradient(native, x) rtol = 1e-6
        end
        # A literal that fails its guard raises while JAX traces, as the
        # native kernel raises when called.
        @test_throws "jitter must be finite" prepare(jax_literal_guard_fails; have = (:x,), want = :t)(x)
        src = jax_source(jax_literal_guard_fails; have = (:x,), want = :t)
        @test_throws "jitter must be finite" rk_jax_evaluate(src, "kernel", (x,))

        R = [0.0, 1.2, 0.3, 0.0, 2.5, 0.1]
        native = prepare(jax_loop; have = (:rho, :R, :kappa), want = :s)
        src = jax_source(jax_loop; have = (:rho, :R, :kappa), want = :s)
        r = rk_jax_evaluate(src, "kernel", (0.7, R, 0.3); grad = true)
        g(rho) = native(rho, R, 0.3)
        @test r.value ≈ g(0.7) rtol = 1e-12
        @test r.grad ≈ only(rk_fd_gradient(u -> g(only(u)), [0.7])) rtol = 1e-6

        native = prepare(jax_clamp; have = (:x, :lo, :hi, :w), want = :s)
        src = jax_source(jax_clamp; have = (:x, :lo, :hi, :w), want = :s)
        xc = [0.1, -0.3, 0.9, 1.4, 0.25, 0.0, 1.0]
        @test rk_jax_evaluate(src, "kernel", (xc, 0.0, 1.0, 0.7)).value ≈
              native(xc, 0.0, 1.0, 0.7) rtol = 1e-12
        # No entry at a bound, where finite differences are one-sided.
        xg = xc[1:5]
        r = rk_jax_evaluate(src, "kernel", (xg, 0.0, 1.0, 0.7); grad = true)
        @test r.grad ≈ rk_fd_gradient(u -> native(u, 0.0, 1.0, 0.7), xg) rtol = 1e-6
        # At a bound Julia's `clamp` returns `x` (its `ifelse` selects it), so
        # the derivative there is the inner one, 3.
        src = jax_source(jax_clamp_scalar; have = (:t,), want = :y)
        for (t, d) in ((0.5, 3.0), (-1.5, 0.0), (2.5, 0.0), (-1.0, 3.0), (2.0, 3.0))
            r = rk_jax_evaluate(src, "kernel", (t,); grad = true)
            @test r.value ≈ 3.0 * clamp(t, -1.0, 2.0) rtol = 1e-12
            @test r.grad ≈ d rtol = 1e-12
        end
        native = prepare(jax_typed; have = (:x,), want = :y)
        src = jax_source(jax_typed; have = (:x,), want = :y)
        @test rk_jax_evaluate(src, "kernel", (2.7,)).value ≈ native(2.7) rtol = 1e-12
    end
end

@testset "jax_source: prepare_data" begin
    g = ["b", "a", "c", "a", "b", "b"]
    y = [1, 3, 2, 0, 3, 1]
    src = jax_source(jax_prep; have = (:theta, :g, :y), want = :total, data = (; g, y))
    text = string(src)
    @test [f.name for f in src.functions] == ["prepare_data", "kernel"]
    @test first.(src.functions[1].inputs) == ["g", "y"]
    @test first(first.(src.functions[2].inputs)) == "theta"
    # The preparation is host code with real loops; the function is JAX code.
    host_code = text[findfirst("def prepare_data", text).start:end]
    @test occursin(r"^\s*for .* in enumerate\("m, text)
    @test occursin("break", text) && occursin(".append(", text) && occursin(".setdefault(", text)
    @test !occursin("jnp.", split(host_code, "\ndef kernel")[1])
    @test occursin("PREPARED = {", text)
    # Without `data`, the same preparation has no JAX translation.
    @test_throws JAXEmissionError jax_source(jax_prep; have = (:theta, :g, :y), want = :total)
    R = [[0.5, 1.0], [2.0], [0.1, 0.2, 0.3]]
    flat = jax_source(jax_prep_flat; have = (:a, :R), want = :total, data = (; R))
    @test "R_flat" in first.(flat.functions[2].inputs)
    @test occursin("jnp.nonzero(", string(flat))
    xr = [0.5, -1.0, 2.0, 3.5, 0.2, 1.1]
    reshaped = jax_source(jax_prep_reshapes; have = (:a, :x), want = :total, data = (; x = xr))
    host_reshape = split(split(string(reshaped), "def prepare_data")[2], "\ndef kernel")[1]
    @test occursin("np.reshape(x, (1, -1), order=\"F\")", host_reshape)
    @test !occursin("jnp.", host_reshape)
    # Element writes into a list of arrays (snag jax-host-nested-6a304948):
    # in place where the source allocated the arrays, after a copy otherwise.
    labels_g = ["s2", "s1", "s2", "s3", "s1", "s9", "s2"]
    labels_c = ["s1", "s2", "s3", "s4"]
    partition = jax_source(jax_prep_partition; have = (:a, :groups, :cells), want = :total,
                           data = (; groups = labels_g, cells = labels_c))
    @test occursin(r"rows\w*\[i\w* - 1\]\[counts\w*\[i\w* - 1\] - 1\] = r", string(partition))
    @test !occursin("_rk_deepcopy", string(partition))
    rescaled = jax_source(jax_prep_rescaled; have = (:a, :R), want = :total, data = (; R))
    @test occursin(r"rows\w* = _rk_deepcopy_np\(rows\w*\)", string(rescaled))
    # Only the taken arm of a decided condition is translated.
    labels = jax_source(jax_prep_labels; have = (:a, :g), want = :total, data = (; g))
    @test !occursin("label_span", string(labels))
    int_labels = jax_source(jax_prep_int_labels; have = (:a, :g), want = :total,
                            data = (; g = [4, 9, 2]))
    @test occursin("label_span", string(int_labels))
    if !rk_jax_available()
        @test_skip false
    else
        theta = [0.3, -1.2, 2.0]
        native = sum(theta[jax_prep_codes(g, ["a", "b", "c"])]) +
                 2 * sum(theta[jax_prep_first(g)]) + sum(jax_prep_lanes(y, 3)) * theta[2] +
                 jax_prep_search(y, 2) * theta[1]
        r = rk_jax_evaluate(src, "kernel", (theta, g, y); grad = true, prepare = true)
        @test r.value ≈ native rtol = 1e-12
        # The prepared inputs' shapes are static: the function jits with them
        # as arguments.
        @test rk_jax_evaluate(src, "kernel", (theta, g, y); prepare = :arguments).value ≈ native rtol = 1e-12
        @test r.grad ≈ rk_fd_gradient(t -> sum(t[jax_prep_codes(g, ["a", "b", "c"])]) +
                 2 * sum(t[jax_prep_first(g)]) + sum(jax_prep_lanes(y, 3)) * t[2] +
                 jax_prep_search(y, 2) * t[1], theta) rtol = 1e-6
        r = rk_jax_evaluate(flat, "kernel", (0.7, R); grad = true, prepare = true)
        @test r.value ≈ sum(abs2, 0.7 .* reduce(vcat, R)) rtol = 1e-12
        @test r.grad ≈ 2 * 0.7 * sum(abs2, reduce(vcat, R)) rtol = 1e-10
        native = prepare(jax_prep_reshapes; have = (:a, :x), want = :total)
        r = rk_jax_evaluate(reshaped, "kernel", (0.7, xr); grad = true, prepare = true)
        @test r.value ≈ native(0.7, xr) rtol = 1e-12
        @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), xr), [0.7])) rtol = 1e-6
        index = jax_source(jax_prep_index; have = (:a, :R), want = :total, data = (; R))
        native = prepare(jax_prep_index; have = (:a, :R), want = :total)
        r = rk_jax_evaluate(index, "kernel", (0.7, R); grad = true, prepare = true)
        @test r.value ≈ native(0.7, R) rtol = 1e-12
        @test rk_jax_evaluate(index, "kernel", (0.7, R); prepare = :arguments).value ≈
              native(0.7, R) rtol = 1e-12
        @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), R), [0.7])) rtol = 1e-6
        x = [0.5, -1.0, 2.0, 3.5]
        misc = jax_source(jax_prep_misc; have = (:theta, :x), want = :total, data = (; x))
        host_part = split(split(string(misc), "def prepare_data")[2], "\ndef kernel")[1]
        @test !occursin("jax.", host_part) && occursin(" if ", host_part)
        native = prepare(jax_prep_misc; have = (:theta, :x), want = :total)
        @test rk_jax_evaluate(misc, "kernel", (0.7, x); prepare = true).value ≈ native(0.7, x) rtol = 1e-12
        cells = jax_source(jax_prep_cells; have = (:a, :R), want = :total, data = (; R))
        native = prepare(jax_prep_cells; have = (:a, :R), want = :total)
        r = rk_jax_evaluate(cells, "kernel", (0.7, R); grad = true, prepare = true)
        @test r.value ≈ native(0.7, R) rtol = 1e-12
        @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), R), [0.7])) rtol = 1e-6
        # A loop whose trip count is data runs on the host.
        counted = jax_source(jax_data_count; have = (:x, :n), want = :s, data = (x = [1.0, 2.0, 4.0], n = 2))
        @test occursin("for i in range(1, n + 1):", string(counted))
        native = prepare(jax_prep_partition; have = (:a, :groups, :cells), want = :total)
        @test rk_jax_evaluate(partition, "kernel", (0.7, labels_g, labels_c); prepare = true).value ≈
              native(0.7, labels_g, labels_c) rtol = 1e-12
        native = prepare(jax_prep_rescaled; have = (:a, :R), want = :total)
        r = rk_jax_evaluate(rescaled, "kernel", (0.7, R); grad = true, prepare = true)
        @test r.value ≈ native(0.7, R) rtol = 1e-12
        @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), R), [0.7])) rtol = 1e-6
        native = prepare(jax_prep_labels; have = (:a, :g), want = :total)
        @test rk_jax_evaluate(labels, "kernel", (0.5, g); prepare = true).value ≈ native(0.5, g) rtol = 1e-12
        native = prepare(jax_prep_int_labels; have = (:a, :g), want = :total)
        @test rk_jax_evaluate(int_labels, "kernel", (0.5, [4, 9, 2]); prepare = true).value ≈
              native(0.5, [4, 9, 2]) rtol = 1e-12
    end
end

@testset "jax_source: sort, unique and searchsorted in prepare_data" begin
    x = [0.5, 0.5, 2.0, 1.0, 1.0, 0.5]
    extra = [3.0, 1.0]
    z = [0.0, NaN, -0.0, 1.0, NaN, 0.0, -2.0]
    src = jax_source(jax_prep_order; have = (:theta, :x, :extra, :z), want = :total,
                     data = (; x, extra, z))
    host_part = split(split(string(src), "def prepare_data")[2], "\ndef kernel")[1]
    @test !occursin("Computed by the functions", host_part)
    # The host helpers `prepare_data` calls sort, deduplicate and search.
    @test all(h -> occursin(h, string(src)), ("= _rk_sort_np(", "= _rk_unique_np(",
                                              "_rk_searchsorted_np(levels"))
    # The list is rewritten in place, so it stays the list `push!` built; the
    # run test reads its last entry in place, not a copy per pass.
    @test occursin(r"levels\w*\[:\] = _rk_sort_np\(", string(src))
    @test occursin(r"levels\w*\[-1\]", string(src))
    @test !occursin("jnp.", host_part)
    # A NumPy array keeps its length: `unique!` of one is refused on the host,
    # and the functions' error names that reason as well as their own.
    err = try
        jax_source(jax_prep_unique_array; have = (:a, :x), want = :total, data = (; x))
        nothing
    catch e
        e
    end
    @test err isa JAXEmissionError
    @test occursin("`prepare_data` does not compute it on the host either", err.message)
    @test occursin("`unique!` of `v`", err.message)
    if !rk_jax_available()
        @test_skip false
    else
        theta = collect(1.0:6.0) ./ 7
        native = prepare(jax_prep_order; have = (:theta, :x, :extra, :z), want = :total)
        @test jax_order_signs(z) == 5 + 30 + 400 + 2000 + 50000
        r = rk_jax_evaluate(src, "kernel", (theta, x, extra, z); grad = true, prepare = true)
        @test r.value ≈ native(theta, x, extra, z) rtol = 1e-12
        @test r.grad ≈ rk_fd_gradient(t -> native(t, x, extra, z), theta) rtol = 1e-6
    end
end

@testset "jax_source: rule primals, captured endpoints, positional views and statement writes" begin
    x = [0.4, 1.3, 2.2, 0.7]
    rule_src = jax_source(jax_rule_plate; have = (:a, :x), want = :total)
    view_src = jax_source(jax_view_row; have = (:a, :x), want = :total)
    obj_src = jax_source(jax_obj_scores; have = (:location, :x, :scale), want = :total)
    bounded_src = jax_source(jax_obj_bounded; have = (:location, :scale, :x, :lo), want = :total)
    g = ["b", "c", "a", "b", "a"]
    declared = jax_source(jax_prep_declared_sum; have = (:theta, :g), want = :total, data = (; g))
    iseq = jax_source(jax_prep_isequal; have = (:theta, :g), want = :total, data = (; g))
    @test length(collect(eachmatch(r"^def (\w+)\("m, string(iseq)))) ==
          length(unique(m[1] for m in eachmatch(r"^def (\w+)\("m, string(iseq))))
    @test occursin(".setdefault(", string(declared))
    if !rk_jax_available()
        @test_skip false
    else
        native = prepare(jax_rule_plate; have = (:a, :x), want = :total)
        r = rk_jax_evaluate(rule_src, "kernel", (0.3, x); grad = true)
        @test r.value ≈ native(0.3, x) rtol = 1e-12
        @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), x), [0.3])) rtol = 1e-6
        native = prepare(jax_view_row; have = (:a, :x), want = :total)
        r = rk_jax_evaluate(view_src, "kernel", (0.3, x); grad = true)
        @test r.value ≈ native(0.3, x) rtol = 1e-12
        @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), x), [0.3])) rtol = 1e-6
        native = prepare(jax_obj_scores; have = (:location, :x, :scale), want = :total)
        r = rk_jax_evaluate(obj_src, "kernel", (0.3, x, 1.7); grad = true)
        @test r.value ≈ native(0.3, x, 1.7) rtol = 1e-12
        @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), x, 1.7), [0.3])) rtol = 1e-6
        native = prepare(jax_obj_bounded; have = (:location, :scale, :x, :lo), want = :total)
        for lo in (0.5, 3.0)
            r = rk_jax_evaluate(bounded_src, "kernel", (0.3, 1.7, x, lo); grad = true)
            @test r.value ≈ native(0.3, 1.7, x, lo) rtol = 1e-12
            @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), 1.7, x, lo), [0.3])) rtol = 1e-6
        end
        @test rk_jax_evaluate(iseq, "kernel", (0.5, g); prepare = true).value ≈ 1.0
        theta = [0.5, -1.5, 2.5]
        native = prepare(jax_prep_declared_sum; have = (:theta, :g), want = :total)
        r = rk_jax_evaluate(declared, "kernel", (theta, g); grad = true, prepare = true)
        @test r.value ≈ native(theta, g) rtol = 1e-12
        @test r.grad ≈ rk_fd_gradient(t -> native(t, g), theta) rtol = 1e-6
    end
end

@testset "jax_source: tuple scan outputs, history scans, merge and dims" begin
    xs = [1.0, 2.0, 3.0, -0.5]
    M = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    cases = ((jax_scan_tuple, (:a, :xs), (0.5, xs)), (jax_scan_named, (:a, :xs), (0.5, xs)),
             (jax_history_leading, (:a, :xs), (0.7, xs)), (jax_history_indexed, (:a, :xs), (0.7, xs)),
             (jax_tie_div, (:a, :x, :k), (0.7, xs, 2)),
             (jax_scan_merge, (:a, :xs, :u0), (0.5, xs, 1.0)), (jax_dims, (:a, :M), (0.5, M)),
             (jax_shape_fold, (:a, :x), (0.7, xs)), (jax_unit_index, (:a, :x), (0.7, xs)),
             (jax_scan_vector_fields, (:a, :ys, :w0), (0.7, xs, [0.2, 0.3, 0.5])))
    srcs = [jax_source(k; have, want = :total) for (k, have, _) in cases]
    @test !occursin(".shape[0] - 1][", string(srcs[2]))
    fold = string(srcs[end - 2])
    @test occursin(" in (1, ", fold) && occursin("jax.lax.fori_loop(", fold)
    @test occursin("jnp.squeeze(", string(srcs[end - 1]))
    # The whole output, its vector fields stacked.
    @test occursin("jax.lax.scan(", string(jax_source(jax_scan_vector_fields;
                                                      have = (:a, :ys, :w0), want = :steps)))
    stacked = jax_source(jax_prep_stack; have = (:a, :x), want = :total, data = (; x = xs))
    @test occursin("np.stack(", string(stacked))
    if !rk_jax_available()
        @test_skip false
    else
        for ((k, have, args), src) in zip(cases, srcs)
            native = prepare(k; have, want = :total)
            r = rk_jax_evaluate(src, "kernel", args; grad = true)
            @test r.value ≈ native(args...) rtol = 1e-12
            @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), args[2:end]...), [args[1]])) rtol = 1e-6
        end
        native = prepare(jax_prep_stack; have = (:a, :x), want = :total)
        r = rk_jax_evaluate(stacked, "kernel", (0.7, xs); grad = true, prepare = true)
        @test r.value ≈ native(0.7, xs) rtol = 1e-12
    end
end

@testset "jax_source: reductions of gathers through ragged rows" begin
    # The gathered entries carry the row's lengths: the emitted cell masks.
    direct = string(jax_source(jax_gather_rows; have = (:x, :R, :a), want = :total))
    @test occursin("jnp.where((jnp.arange(", direct)
    @test occursin("jnp.vdot(jnp.where(", direct)
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    Y = [[0.5, 1.0], [2.0], [0.1, 0.2, 0.3]]
    if !rk_jax_available()
        @test_skip false
    else
        # Rows of different lengths, and an empty row (padded with index 0,
        # which reads a valid entry that the mask then drops).
        for R in ([[2, 5], [1, 3, 7], [4]], [[2, 5], [1, 3, 7], [4], Int[]])
            for k in (jax_gather_indexed, jax_gather_rows)
                native = prepare(k; have = (:x, :R, :a), want = :total)
                expected = native(x, R, 1.3)
                fd = rk_fd_gradient(u -> native(u, R, 1.3), x)
                plain = jax_source(k; have = (:x, :R, :a), want = :total)
                r = rk_jax_evaluate(plain, "kernel", (x, R, 1.3); grad = true)
                @test r.value ≈ expected rtol = 1e-12
                @test r.grad ≈ fd rtol = 1e-6 atol = 1e-9
                prepared = jax_source(k; have = (:x, :R, :a), want = :total, data = (; R, a = 1.3))
                r = rk_jax_evaluate(prepared, "kernel", (x, R, 1.3); grad = true, prepare = true)
                @test r.value ≈ expected rtol = 1e-12
                @test r.grad ≈ fd rtol = 1e-6 atol = 1e-9
            end
        end
        native = prepare(jax_ragged_row_sums; have = (:a, :Y), want = :total)
        for src in (jax_source(jax_ragged_row_sums; have = (:a, :Y), want = :total),
                    jax_source(jax_ragged_row_sums; have = (:a, :Y), want = :total, data = (; Y)))
            prep = occursin("def prepare_data", string(src))
            r = rk_jax_evaluate(src, "kernel", (0.7, Y); grad = true, prepare = prep)
            @test r.value ≈ native(0.7, Y) rtol = 1e-12
            @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), Y), [0.7])) rtol = 1e-6
        end
    end
end

# Modified Bessel functions of any order (snag jax-source-perio-96f952a3): a
# periodic Hilbert-space GP basis weights harmonic `j` by
# `besselix(j, 1 / rho^2)`, with the harmonics a data value.
@kernel jax_bessel_values(nu::Vector{Float64}, x::Vector{Float64}) = begin
    scaled = besselix.(nu, x)
    plain = besseli.(nu, x)
    return scaled
end
@kernel jax_bessel_scale(rho::Float64, h::Vector{Float64}, sigma::Float64) = begin
    a = 1 / rho^2
    s = sum(log.(sigma .* besselix.(h, a))) + besselix(3, a) + besseli(0, rho) + besseli(2.5, rho)
    return s
end
# A data-only recipe: `prepare_data` computes it with SciPy.
@kernel jax_bessel_prepared(w::Float64, h::Vector{Float64}, x::Vector{Float64}) = begin
    basis = besselix.(h, x)
    s = w * sum(basis)
    return s
end

@testset "jax_source: modified Bessel functions of any order" begin
    values = string(jax_source(jax_bessel_values; have = (:nu, :x), want = :scaled))
    @test occursin("_rk_besselix(nu, x)", values)
    @test occursin("jax.custom_jvp", values)
    plain = string(jax_source(jax_bessel_values; have = (:nu, :x), want = :plain))
    @test occursin("_rk_besseli(nu, x)", plain) && occursin("def _rk_besselix(", plain)
    scale = string(jax_source(jax_bessel_scale; have = (:rho, :h, :sigma), want = :s))
    # A literal order 0 or 1 stays jax.scipy.special's own function.
    @test occursin("jsp.i0(", scale) && occursin("_rk_besselix(3, ", scale)
    prepared = jax_source(jax_bessel_prepared; have = (:w, :h, :x), want = :s,
                          data = (; h = [1.0, 2.0], x = [0.5, 3.0]))
    @test occursin("ssp.ive(", string(prepared))
    @test !occursin("_rk_besselix", string(prepared))
    if !rk_jax_available()
        @test_skip false
    else
        # Orders on both sides of the expansion's order 25 and x on both
        # sides of the series' x = 30; negative x and orders where Julia
        # defines them.
        nus = [0.0, 0.5, 1.0, 2.0, 3.5, 7.0, 12.0, 24.5, 25.0, 40.0, 120.0]
        xs = [1e-6, 0.3, 2.0, 9.0, 29.5, 31.0, 80.0, 400.0, 5e3, 1e6]
        nu = [repeat(nus; inner = length(xs)); 3.0; 4.0; -2.0; -3.0; 0.0]
        x = [repeat(xs; outer = length(nus)); -7.0; -0.4; 2.5; 50.0; 0.0]
        # Unscaled, past x ~ 700 Julia's `besseli` throws on overflow (JAX's is
        # Inf), so it is compared where it is finite.
        finite = abs.(x) .<= 400
        for (want, nu, x) in ((:scaled, nu, x), (:plain, nu[finite], x[finite]))
            native = prepare(jax_bessel_values; have = (:nu, :x), want)(nu, x)
            src = jax_source(jax_bessel_values; have = (:nu, :x), want)
            got = rk_jax_evaluate(src, "kernel", (nu, x)).value
            @test all(isapprox(g, n; rtol = 1e-12) || g == n for (g, n) in zip(got, native))
        end
        native = prepare(jax_bessel_scale; have = (:rho, :h, :sigma), want = :s)
        src = jax_source(jax_bessel_scale; have = (:rho, :h, :sigma), want = :s)
        h = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        for rho in (0.05, 0.3, 0.9, 2.0)
            r = rk_jax_evaluate(src, "kernel", (rho, h, 1.3); grad = true)
            @test r.value ≈ native(rho, h, 1.3) rtol = 1e-12
            @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), h, 1.3), [rho])) rtol = 1e-6
        end
        native = prepare(jax_bessel_prepared; have = (:w, :h, :x), want = :s)
        hp, xp = [0.0, 1.0, 2.5, 30.0], [0.2, 4.0, 45.0, 12.0]
        src = jax_source(jax_bessel_prepared; have = (:w, :h, :x), want = :s, data = (; h = hp, x = xp))
        r = rk_jax_evaluate(src, "kernel", (0.7, hp, xp); grad = true, prepare = true)
        @test r.value ≈ native(0.7, hp, xp) rtol = 1e-12
        @test r.grad ≈ native(1.0, hp, xp) rtol = 1e-12
    end
end

# Keyword calls in a plain helper's lowered code (snag
# jax-source-perio-96f952a3), `Core.kwcall(NamedTuple{names}(values), f,
# args...)`: the call with those keyword arguments, as source spells it. The
# shape of BRM's `hsgp_sqrt_spd`, a spectral scale over a basis' frequencies.
jax_kw_sqrt_spd(lambda, sigma, rho) =
    sigma .* exp.(-0.25 .* (rho * rho) .* vec(sum(lambda; dims = 2)))
@kernel jax_kw_plain(rho::Float64, lambda::Matrix{Float64}) = begin
    scale = jax_kw_sqrt_spd(lambda, 1.3, rho)
    s = sum(scale)
    return s
end
# A keyword call RK has no translation for names the keyword call (BRM's
# periodic basis read "a type applied to a computed parameter").
jax_kw_inner(x, k; anchor = nothing, mask = nothing) = x .* k
jax_kw_outer(x, k) = jax_kw_inner(x, k; anchor = nothing, mask = nothing)
@kernel jax_kw_refused(k::Float64, x::Vector{Float64}) = begin
    b = jax_kw_outer(x, k)
    s = sum(b)
    return s
end

@testset "jax_source: keyword calls in a plain helper" begin
    src = jax_source(jax_kw_plain; have = (:rho, :lambda), want = :s)
    @test occursin("jnp.sum(lambda_, axis=2 - 1, keepdims=True)", string(src))
    @test_throws "keyword arguments to `jax_kw_inner` have no JAX translation" jax_source(
        jax_kw_refused; have = (:k, :x), want = :s)
    if !rk_jax_available()
        @test_skip false
    else
        lambda = [0.1 0.4; 0.9 1.6; 2.5 0.3]
        native = prepare(jax_kw_plain; have = (:rho, :lambda), want = :s)
        r = rk_jax_evaluate(src, "kernel", (0.7, lambda); grad = true)
        @test r.value ≈ native(0.7, lambda) rtol = 1e-12
        @test r.grad ≈ only(rk_fd_gradient(u -> native(only(u), lambda), [0.7])) rtol = 1e-6
    end
end
