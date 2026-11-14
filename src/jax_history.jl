# A history scan whose reads have bounded lag needs only a rolling buffer in
# its primal carry. lax.scan stacks the outputs separately. This does not
# bound every reverse-mode carry: gradients of shared arrays can still be
# full-sized. See benchmark/jax_hmm_forward.md for measured time and memory.
# Unknown reads and whole-array uses keep the original full-history layout.

struct _JaxScanRange
    first::String
end

# An integer between floor(position / divisor) * multiplier + lo and + hi.
# position is the scan's zero-based step; all other terms are static Python
# integer expressions. This includes lagged reads and block recurrences
# (previous row, with a data-sized number of cells in each row).
struct _JaxHistoryIndex
    scan::String
    divisor::String
    multiplier::String
    lo::String
    hi::String
    conditions::Vector{String}
    value::_PyVal
end

_jax_map_origin(o::_JaxScanRange, code) = _JaxScanRange(code(o.first))
_jax_map_origin(o::_JaxHistoryIndex, code) = _JaxHistoryIndex(o.scan,
    code(o.divisor), code(o.multiplier), code(o.lo), code(o.hi), code.(o.conditions),
    _jax_map_value(o.value, code))

# Retain ordinary integer provenance alongside the lag proof. In particular,
# a scanned prepared range can index history and bound a nested loop.
_jax_history_value(v::_PyVal) = v.origin isa _JaxHistoryIndex ? v.origin.value : v
_jax_lower_end(v::_PyVal, h::_JaxHistoryIndex) = _jax_lower_end(h.value)
_jax_upper_end(v::_PyVal, h::_JaxHistoryIndex) = _jax_upper_end(h.value)

mutable struct _JaxHistoryLayout
    token::String
    earlier::String
    position::String
    offset::String
    fill::String
    reads::Vector{Tuple{String,String,_JaxHistoryIndex}}
end

# Small integer-expression identities, independent of data values. In
# particular (position + 1) - 1 is position, before division into blocks.
function _jax_history_arithmetic(op, a, b)
    ai, bi = tryparse(Int, a), tryparse(Int, b)
    if ai !== nothing && bi !== nothing
        return string(op == "+" ? ai + bi : op == "-" ? ai - bi : ai * bi)
    end
    op == "+" && a == "0" && return b
    op in ("+", "-") && b == "0" && return a
    op == "-" && a == b && return "0"
    op == "*" && (a == "0" || b == "0") && return "0"
    op == "*" && a == "1" && return b
    op == "*" && b == "1" && return a
    _py_infix(op, [a, b])
end

function _jax_history_static_bounds(v::_PyVal)
    v.T isa Type && v.T <: Integer || return nothing
    v.static && return (v.code, v.code)
    if v.origin isa _JaxBetween
        lo, hi = _jax_lower_end(v), _jax_upper_end(v)
        lo.static && hi.static && return (lo.code, hi.code)
    end
    nothing
end

function _jax_history_index_result(f, args, result)
    result.T isa Type && result.T <: Integer || return result
    length(args) == 2 || return result
    a, b = args
    if f === (+) && !(a.origin isa _JaxHistoryIndex) && b.origin isa _JaxHistoryIndex
        a, b = b, a
    end
    h = a.origin
    h isa _JaxHistoryIndex || return result
    bounds = _jax_history_static_bounds(b)
    bounds === nothing && return result
    lo, hi = bounds
    op = _jax_history_arithmetic
    conditions = copy(h.conditions)
    d, m = h.divisor, h.multiplier
    if f === (+)
        lower, upper = op("+", h.lo, lo), op("+", h.hi, hi)
    elseif f === (-)
        lower, upper = op("-", h.lo, hi), op("-", h.hi, lo)
    elseif f === (*) && lo == hi
        # Multiplication preserves the interval only for a nonnegative
        # factor. A static guard selects the original layout otherwise.
        push!(conditions, "(" * lo * ") >= 0")
        m = op("*", m, lo)
        lower, upper = op("*", h.lo, lo), op("*", h.hi, lo)
    elseif f === div && lo == hi && d == m == "1"
        # A nonnegative position plus offset uses floor division. A static
        # offset may cross a block boundary, so bound both possible blocks.
        push!(conditions, "(" * lo * ") > 0")
        push!(conditions, "(" * h.lo * ") >= 0")
        d = lo
        lower = h.lo == "0" ? "0" : "((" * h.lo * ") // (" * d * "))"
        upper = h.hi == "0" ? "0" :
            "((" * h.hi * " + " * d * " - 1) // (" * d * "))"
    else
        return result
    end
    _jax_with_origin(result, _JaxHistoryIndex(h.scan, d, m, lower, upper, conditions,
                                             _jax_history_value(result)))
end

function _jax_history_element(ctx, name, seq, layout, position)
    v = _jax_scan_element(name, seq, position)
    start = seq.origin isa _JaxScanRange ? seq.origin.first :
        seq.T isa Type && seq.T <: Base.OneTo ? "1" :
        seq.T isa Type && seq.T <: AbstractUnitRange && seq.origin isa _JaxOrigin &&
        seq.origin.position === nothing ? _jax_prepared_bound!(ctx, :min, seq.origin.key) : nothing
    start === nothing && return v
    _jax_with_origin(v, _JaxHistoryIndex(layout.token, "1", "1", start, start, String[], v))
end

function _jax_history_read!(ctx, x, indices)
    length(indices) == 1 || return nothing
    index = only(indices)
    if index === :(:) || index == :(Colon()) ||
       index isa Expr && index.head === :call && index.args[1] === :(:)
        return _jax_index(ctx, _PyVal(x.code, x.T), indices)
    end
    v = _jax_bound_expr(ctx, x, 1, index)
    h = v.origin
    layout = x.origin
    h isa _JaxHistoryIndex && h.scan == layout.token && h.divisor == h.multiplier ||
        return _jax_index(ctx, _PyVal(x.code, x.T), [v])
    # Branches have independent local name sets. An access placeholder is
    # unique across those scopes, including nested history scans.
    token = _jax_private_identifier(ctx.mod.used, :history_read)
    push!(layout.reads, (token, v.code, h))
    _PyVal(token, eltype(x.T))
end

# Resolve the access placeholders only after every use is known. The full
# vector's placeholder surviving in the step means some use is not an
# understood scalar read. It keeps the original layout, including length,
# slicing, reductions, and arbitrary data-indexed gathers.
# String literals (including error messages) are not identifier uses. The
# emitter quotes strings with backslash escapes; keep those tokens intact.
const _JAX_HISTORY_TOKENS = r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|\#[^\n]*|\b[A-Za-z_][A-Za-z_0-9]*\b"

function _jax_history_layout!(ctx, layout, inner, wants, steps, history, dtype)
    body = join([inner.lines; [v.code for v in wants]], "\n")
    compact = !isempty(layout.reads) &&
        !any(m -> m.match == layout.token, eachmatch(_JAX_HISTORY_TOKENS, body))
    bounds = ["(" * layout.offset * ") + (" * h.divisor * ") - (" * h.lo * ")"
              for (_, _, h) in layout.reads]
    conditions = unique(String[c for (_, _, h) in layout.reads for c in h.conditions])
    # A shape expression must be available outside the step. Locals from a
    # nested helper/loop are not eligible, even when static in that scope.
    available = union(ctx.scope.used, ctx.mod.used, _JAX_RESERVED, Set(["shape"]))
    compact &= all(s -> all(m -> m.match in available,
                            eachmatch(r"\b[A-Za-z_][A-Za-z_0-9]*\b", s)), [bounds; conditions])
    width = _jax_fresh(ctx, :history_width)
    use_compact = _jax_fresh(ctx, :compact_history)
    _jax_emit!(ctx, use_compact * " = " * (compact ?
        (isempty(conditions) ? "True" : join(conditions, " and ")) : "False"))
    _jax_emit!(ctx, width * " = max(1, " * join(isempty(bounds) ? ["1"] : bounds, ", ") *
                   ") if " * use_compact * " else " * _py_infix("+", [layout.offset, steps]))
    replacements = Pair{String,String}[layout.token => layout.earlier]
    for (token, index, _) in layout.reads
        at = _py_infix("-", [index, "1"])
        original = layout.earlier * "[" * at * "]"
        current = _py_infix("+", [layout.offset, layout.position])
        rolling = "jnp.where(" * at * " >= " * current * ", " * layout.fill * ", " *
            layout.earlier * "[(" * at * ") % " * width * "])"
        push!(replacements, token => "(" * rolling * " if " * use_compact * " else " * original * ")")
    end
    # Replace whole identifier tokens, never substrings of another name.
    lookup = Dict(replacements)
    replace_accesses(s) = replace(s, _JAX_HISTORY_TOKENS =>
                                  word -> get(lookup, word, word))
    inner.lines .= replace_accesses.(inner.lines)
    result_wants = [_PyVal(replace_accesses(v.code), v.T) for v in wants]
    buffer = _jax_fresh(ctx, :history)
    if history.T <: Number
        _jax_emit!(ctx, buffer * " = jnp.full((" * width * ",), " * history.code * ", dtype=" * dtype * ")")
    else
        # Tail seeds, rotated into their absolute positions modulo width.
        seed = "jnp.asarray(" * history.code * ", dtype=" * dtype * ")"
        rolling = "jnp.roll(jnp.concatenate([jnp.zeros((max(" * width * " - " * layout.offset *
            ", 0),), dtype=" * dtype * "), " * seed * "[-" * width * ":]]), " * layout.offset * " % " * width * ")"
        original = "jnp.concatenate([" * seed * ", jnp.zeros((" * steps * ",), dtype=" * dtype * ")])"
        _jax_emit!(ctx, buffer * " = " * rolling * " if " * use_compact * " else " * original)
    end
    (; width, use_compact, buffer, wants = result_wants)
end
