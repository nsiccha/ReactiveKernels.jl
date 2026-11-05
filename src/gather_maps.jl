# Elementwise maps over expanding data-indexed gathers.
#
# A dotted call `f.(x[k])` applies `f` once per element of the index `k`. When
# `k` reads the elements of a shorter `x` repeatedly, `f` runs several times on
# each of them: Bruno's generated brm7 simulator computes each subject's dose
# amounts as `op_amount[op_rows[i]] .* exp.(vcat(0.0, log_F[dose_rows[i]])[
# op_dose_pos[op_rows[i]] .+ 1])`, 488 exponentials of 9 values per subject and
# draw (snag `data-only-subexp-c79363b3`). On the user's decision `1j3rv6f`
# (option 1: on by default) RK applies `f` once per element of `x` and gathers
# afterwards, wherever the index is a value partial evaluation caches: under
# `bound=`, and in a position reader's per-cell prefix (`_replicated_shared_cells`).
#
# Splitting a recipe along its data-only nodes (`_partial_emit_split!`) leaves
# each live part reading its cached nodes by name. In such a part, every
# eligible `f.(x[k])` whose index `k` is one of those cached values becomes
#
#     c > 0 ? <the part with f.(_rk_gather_source(x, c))[k]> : <the part>
#
# with `c = _rk_gather_cover(k)` a further cached value: the `m` such that `k`
# reads each of the linear indices `1:m` and nothing else, provided `k` is
# longer than `m`, and 0 otherwise. The first arm applies `f` to `x`'s
# elements `1:c` (a view) and gathers the result by `k`. Every element `k`
# reads is among them and nothing else is, so `f` runs on exactly the elements
# the authored part reads, once each: it raises no error and produces no
# value the authored part would not, and the gathered values are the authored
# ones bit for bit. An `x` shorter than `c` throws the `BoundsError` the
# authored gather throws. The second arm is the authored part unchanged, and
# only one arm runs, so `x` is evaluated exactly as often as before.
#
# Reverse-mode derivatives of the first arm add the cotangents of an
# element's repeats before multiplying by `f'`, where the authored part
# multiplies each repeat first: equal up to floating-point rounding where an
# index repeats. The user accepted that (`1j3rv6f`); values are unchanged.
#
# Eligible: a named callee (not a mutating one, not one of RK's cheap calls,
# `_KERNEL_SPLIT_CHEAP`, whose work a cover check would not repay) dotted over
# a single gather with a single index, at a strict position of the part (as
# split nodes are, `_kernel_split_children`; a branch arm, closure or
# generator keeps its own evaluation), in a part whose source record is
# capture-free and compiled by RK (`_KernelSource` with a companion). JAX
# emission does not run partial evaluation and keeps the authored form.

"""
    _rk_gather_cover(index) -> Int

The `m` such that `index` reads each linear index `1:m` at least once and no
other index, when `index` is an array of (non-`Bool`) integers longer than
`m`; otherwise 0. A cached value of partial evaluation's gather-map
interchange (`src/gather_maps.jl`).
"""
_rk_gather_cover(index) = 0
function _rk_gather_cover(index::AbstractArray{T}) where {T<:Integer}
    T <: Bool && return 0
    n = length(index)
    n > 1 || return 0
    lo, hi = extrema(index)
    lo >= 1 && hi < n || return 0
    m = Int(hi)
    seen = falses(m)
    count = 0
    for j in index
        @inbounds if !seen[j]
            seen[j] = true
            count += 1
        end
    end
    count == m ? m : 0
end

# The elements `1:c` of a gather's source, read in place: exactly the elements
# a covering index (`_rk_gather_cover`) reads.
_rk_gather_source(x::AbstractArray, c::Int) = view(x, Base.OneTo(c))
_rk_gather_source(x, c::Int) = x[Base.OneTo(c)]

# Only the interchange below creates this marker. Its cover proves that the
# two arms read exactly the same source elements. Capture analysis can use the
# authored arm without treating an arbitrary user-written guard as equivalent.
# The executable state remains just the generated call, including its guard.
struct _KernelGatherMap{F}
    call::F
end
@inline (map::_KernelGatherMap)(args::Vararg{Any,N}) where {N} =
    @inline map.call(args...)
Base.@nospecializeinfer _source_record(@nospecialize(map::_KernelGatherMap)) =
    _source_record(getfield(map, :call))

# `(; f, x, k)` when `ex` is an eligible `f.(x[k])` with `k` among `cached`,
# else `nothing`.
function _gather_map_site(ex, cached::Set{Symbol})
    _cell_dotted_call(ex) || return nothing
    f = ex.args[1]
    name = _kernel_split_callee_name(f)
    name isa Symbol || return nothing
    name in _KERNEL_SPLIT_CHEAP && return nothing
    _kernel_mutating_callee(f) && return nothing
    args = ex.args[2].args
    length(args) == 1 || return nothing
    ref = only(args)
    Meta.isexpr(ref, :ref, 2) || return nothing
    k = ref.args[2]
    k isa Symbol && k in cached || return nothing
    (; f, x = ref.args[1], k)
end

# The eligible sites of `ex` at strict positions, outermost first, as
# `expression => site` pairs. A site's own source is not searched.
function _gather_map_sites!(sites, ex, cached::Set{Symbol})
    ex isa Expr || return sites
    site = _gather_map_site(ex, cached)
    if site !== nothing
        push!(sites, ex => site)
        return sites
    end
    for child in _kernel_split_children(ex)
        _gather_map_sites!(sites, child, cached)
    end
    sites
end

# A source operation over `params` whose source is `body`, compiled as RK
# compiles an authored capture-free source (`_kernel_source_function`), with
# its throw-stripped twin; `nothing` when the body needs a closure Julia owns.
function _partial_source_op(token::Symbol, suffix::Symbol, params::Vector{Symbol}, body,
                            mod::Module; form::Symbol = :fused)
    names = Set{Symbol}(params)
    native = _kernel_native_body(body, mod, names)
    stripped = _kernel_ignore_throw_source(body, mod, names)
    twin_native = _kernel_native_body(stripped, mod, names)
    (_kernel_source_native_scope(native) || _kernel_source_native_scope(twin_native)) &&
        return nothing
    record = _kernel_source_function(identity,
        Expr(:->, Expr(:tuple, params...), native), mod)
    twin = _kernel_source_function(identity,
        Expr(:->, Expr(:tuple, params...), twin_native), mod)
    name = Val(Symbol(token, suffix))
    ignored = _KernelSourceOp(name, Val(form), twin, nothing, nothing, mod)
    _KernelSourceOp(name, Val(form), record, ignored, nothing, mod)
end

"""
    _partial_gather_maps!(recipes, readable, g, part, params, cached, fresh, stem) -> Recipe

`part` (a live split part over `params`, its inputs' source names) with each
eligible `f.(x[k])` over a cached index `k` (`cached`, a subset of `params`)
guarded as `src/gather_maps.jl` describes, after appending one cover recipe
per such index to `recipes`; `part` itself when nothing applies. New values
are named from `stem` in graph `g`; `fresh` mints recipe ids.
"""
function _partial_gather_maps!(recipes::Vector{Recipe}, readable, g::Graph, part::Recipe,
                               params::Vector{Symbol}, cached::Set{Symbol}, fresh, stem::Symbol)
    isempty(cached) && return part
    length(params) == length(part.inputs) || return part
    op = part.op
    op isa _KernelSourceOp && kernel_sourceop_form(op) === :fused || return part
    record = getfield(op, :f)
    record isa _KernelSource && _source_native(record) === nothing &&
        isempty(_source_captures(record)) || return part
    mod = _kernel_endpoint_source_module(op)
    mod isa Module || return part
    body = part.source
    body isa Expr || return part
    sites = _gather_map_sites!(Pair{Any,Any}[], body, cached)
    isempty(sites) && return part
    indices = unique!(Symbol[site.k for (_, site) in sites])
    spelled = _kernel_collect_symbols!(Set{Symbol}(params), body)
    covers = Dict{Symbol,Symbol}(k => Symbol("__rk_cover_", k) for k in indices)
    any(c -> c in spelled, values(covers)) && return part
    token = kernel_sourceop_token(op)
    cover_ops = Any[]
    for k in indices
        cover = Expr(:call, GlobalRef(@__MODULE__, :_rk_gather_cover), k)
        cop = _partial_source_op(token, Symbol(:_cover_, k), Symbol[k], cover, mod)
        cop === nothing && return part
        push!(cover_ops, (k, cop, cover))
    end
    mapping = IdDict{Any,Any}(
        ex => Expr(:ref,
                   Expr(:., site.f, Expr(:tuple,
                        Expr(:call, GlobalRef(@__MODULE__, :_rk_gather_source),
                             site.x isa Expr ? copy(site.x) : site.x, covers[site.k]))),
                   site.k)
        for (ex, site) in sites)
    interchanged = _kernel_split_replace(body, mapping)
    tests = Any[Expr(:call, GlobalRef(Base, :>), covers[k], 0) for k in indices]
    condition = foldl((a, b) -> Expr(:&&, a, b), tests)
    source = Expr(:if, condition, interchanged, copy(body))
    operation = _partial_source_op(token, :_gather, vcat(params, Symbol[covers[k] for k in indices]),
                               source, mod)
    operation === nothing && return part
    operation = _KernelSourceOp(Val(kernel_sourceop_token(operation)),
        Val(kernel_sourceop_form(operation)), _KernelGatherMap(operation.f),
        operation.ignored_throws, operation.call_identity, operation.mod)
    cover_values = Value[]
    for (k, cop, cover) in cover_ops
        input = part.inputs[findfirst(==(k), params)]
        value = value!(g, Symbol(stem, :_cover_, k), Any)
        q = Recipe(fresh(), (input,), (value,), cop, part.cost, nothing, false, cover)
        push!(recipes, q)
        readable[q.id] = q
        push!(cover_values, value)
    end
    Recipe(part.id, (part.inputs..., cover_values...), part.outputs, operation, part.cost,
           nothing, false, source)
end
