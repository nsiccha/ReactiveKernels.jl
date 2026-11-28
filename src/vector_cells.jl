# What a summed plate's cell loop runs as vector code beyond the calls
# shared_softplus.jl lists: any call the compiler turns into vector code
# (`_vector_compiles`), and every spelling of a softplus once the stable
# rewrites have named it (`_vectorize_cell`).

# ------------------------------------------------------ asking the compiler
#
# Whether a call keeps a loop vector code is a fact about the code the compiler
# makes of it on this CPU and this Julia. `z^2` is a product and `z^4` a call
# of `pow`. `sqrt(z)` checks its domain and can throw, while in
# `sqrt(abs2(z) + 1.0)` the compiler proves the argument is not negative and
# drops the check. So the compiler is asked: the call, with the expressions it
# is given, becomes a type (`_vector_part`); a sum over it, with a `Float64`
# for every name it reads and its literals in place, is compiled once per
# process (`_vector_call_probe`); and the call is vector code when that loop
# holds vectors as wide as a softplus loop's (`_vector_doubles`) and calls
# nothing. The answer only chooses between two forms of a cell whose values
# agree to one unit in the last place, so a wrong answer costs time and never
# a value.

struct _VectorOperand{K} end          # the K-th name the call reads: a `Float64`
struct _VectorLiteral{V} end          # a literal number, or a function or type given as an argument
struct _VectorCall{F,A<:Tuple} end    # the function `F.instance` of the parts `A`
struct _VectorPower{P,A} end          # `A ^ P` with the literal integer `P`

_vector_part_expr(::Type{_VectorOperand{K}}) where {K} = :(xs[$K][i])
_vector_part_expr(::Type{_VectorLiteral{V}}) where {V} = V
_vector_part_expr(::Type{_VectorCall{F,A}}) where {F,A} =
    Expr(:call, F.instance, Any[_vector_part_expr(part) for part in A.parameters]...)
# Written as the prepared program writes it, so Julia lowers it the same way:
# `Base.literal_pow`, a product for the small powers.
_vector_part_expr(::Type{_VectorPower{P,A}}) where {P,A} = Expr(:call, :^, _vector_part_expr(A), P)

# A sum over the call `C` of `N` vectors, as a summed plate's cell loop runs it.
@generated function _vector_call_probe(::Type{C}, xs::NTuple{N,Vector{Float64}}) where {C,N}
    quote
        total = 0.0
        @inbounds @simd for i in eachindex(xs[1])
            total += $(_vector_part_expr(C))
        end
        total
    end
end

# The part type of expression `ex`, numbering the names it reads in `names`, or
# `nothing` where the probe cannot stand for it: a callee it cannot name, a
# callable that is not a singleton, a block, a table read. `callable(callee)`
# answers the function a callee that is not a global names, or `nothing`.
function _vector_part(ex, callable, names::Vector{Symbol})
    if ex isa Symbol
        k = findfirst(==(ex), names)
        k === nothing && (push!(names, ex); k = length(names))
        return _VectorOperand{k}
    end
    ex isa GlobalRef && (ex = _rw_binding(ex.mod, ex.name))
    ex isa Type && return _VectorLiteral{ex}
    ex isa Union{Number,Function} && return isbits(ex) ? _VectorLiteral{ex} : nothing
    Meta.isexpr(ex, :call) && !isempty(ex.args) || return nothing
    value = _vector_callee_value(ex.args[1])
    value === nothing && (value = callable(ex.args[1]))
    value === nothing && return nothing
    f = something(value)
    parts = Any[]
    for argument in @view ex.args[2:end]
        part = _vector_part(argument, callable, names)
        part === nothing && return nothing
        push!(parts, part)
    end
    if f === (^) && ex.args[1] isa Union{Symbol,GlobalRef} && length(parts) == 2 && ex.args[3] isa Int
        return _VectorPower{ex.args[3],parts[1]}
    end
    isdefined(typeof(f), :instance) || return nothing
    _VectorCall{typeof(f),Tuple{parts...}}
end

# This process's answers, by part type (`__init__` empties it: an image must
# not carry the answers of the process that built it).
const _VECTOR_COMPILED = Dict{Any,Bool}()
const _VECTOR_COMPILED_LOCK = ReentrantLock()

# Whether the call `ex` stays vector code in a summed loop.
function _vector_compiles(ex, callable = Returns(nothing))
    names = Symbol[]
    part = _vector_part(ex, callable, names)
    part === nothing && return false
    @lock _VECTOR_COMPILED_LOCK get!(() -> _vector_probe_compiles(part, length(names)), _VECTOR_COMPILED, part)
end

function _vector_probe_compiles(part, operands::Int)
    doubles = _vector_doubles()
    doubles >= 2 || return false
    io = IOBuffer()
    InteractiveUtils.code_llvm(io, _vector_call_probe, Tuple{Type{part},NTuple{max(operands, 1),Vector{Float64}}};
                               debuginfo = :none)
    code = String(take!(io))
    # A call left in the loop keeps it scalar; LLVM's own intrinsics are not calls.
    any(line -> occursin(r"\b(?:call|invoke)\b", line) && !occursin("@llvm.", line), eachline(IOBuffer(code))) &&
        return false
    # A call of literals alone is a constant of the loop.
    operands == 0 && return true
    widths = (parse(Int, m.captures[1]) for m in eachmatch(r"<(\d+) x double>", code))
    maximum(widths; init = 1) >= doubles
end

# ------------------------------------------------ every spelling of a softplus
#
# The stable rewrites give a softplus that is written out one of their helpers:
# `log1p(exp(x))` is `_rk_log1p_of(exp, x)`, `log(logistic(x))` is
# `_rk_log_of(logistic, x)`. Each such helper has a twin, as `log1pexp` has
# `_vector_log1pexp`: the vector softplus for a `Float64` where the helper's
# own method computes a softplus, the helper itself for any other operands.
const _VectorOne = Union{Int,Float64,Float32,Float16}      # a literal one (`_rw_one`)
const _VectorLogistic = typeof(LogExpFunctions.logistic)

@inline _vector_log1p_of(f::F, x) where {F} = _rk_log1p_of(f, x)
@inline _vector_log1p_of(::_RwExp, x::Float64) = _vector_softplus(x)
@inline _vector_log_one_plus_of(c, f::F, x) where {F} = _rk_log_one_plus_of(c, f, x)
@inline _vector_log_one_plus_of(::_VectorOne, ::_RwExp, x::Float64) = _vector_softplus(x)
@inline _vector_log_of(f::F, x) where {F} = _rk_log_of(f, x)
@inline _vector_log_of(::_VectorLogistic, x::Float64) = -_vector_softplus(-x)
@inline _vector_log1p_neg_of(f::F, x) where {F} = _rk_log1p_neg_of(f, x)
@inline _vector_log1p_neg_of(::_VectorLogistic, x::Float64) = -_vector_softplus(x)
@inline _vector_log_one_minus_of(c, f::F, x) where {F} = _rk_log_one_minus_of(c, f, x)
@inline _vector_log_one_minus_of(::_VectorOne, ::_VectorLogistic, x::Float64) = -_vector_softplus(x)
@inline _vector_log_sigmoid(c1, c2, f::F, u) where {F} = _rk_log_sigmoid(c1, c2, f, u)
@inline _vector_log_sigmoid(::_VectorOne, ::_VectorOne, ::_RwExp, u::Float64) = -_vector_softplus(-u)
@inline _vector_log1m_sigmoid(c1, c2, f::F, u) where {F} = _rk_log1m_sigmoid(c1, c2, f, u)
@inline _vector_log1m_sigmoid(::_VectorOne, ::_VectorOne, ::_RwExp, u::Float64) = -_vector_softplus(u)

# Each such helper with its twin, the position of the function it is given
# (literal ones stand before it) and the functions for which its method is a
# softplus. LOCKSTEP: the twins' `Float64` methods above.
const _VECTOR_HELPER_TWINS = (
    (_rk_log1p_of, _vector_log1p_of, 1, _rw_is_exp),
    (_rk_log_one_plus_of, _vector_log_one_plus_of, 2, _rw_is_exp),
    (_rk_log_of, _vector_log_of, 1, f -> f === LogExpFunctions.logistic),
    (_rk_log1p_neg_of, _vector_log1p_neg_of, 1, f -> f === LogExpFunctions.logistic),
    (_rk_log_one_minus_of, _vector_log_one_minus_of, 2, f -> f === LogExpFunctions.logistic),
    (_rk_log_sigmoid, _vector_log_sigmoid, 3, _rw_is_exp),
    (_rk_log1m_sigmoid, _vector_log1m_sigmoid, 3, _rw_is_exp))

# Every function a vector loop calls in place of a softplus beyond the two
# first twins, for code that reads a prepared program by function identity:
# the generated plate reverse expands each into the method its operands select
# (`_pr_expand_stable_helper`, `src/plate_reverse.jl`).
const _RK_VECTOR_HELPERS = (map(row -> row[2], _VECTOR_HELPER_TWINS)...,
    _vector_loglogistic, _vector_log1mlogistic)

# The twin of the helper call `f(arguments...)` whose method is a softplus, or
# `nothing`.
function _vector_helper_twin(f, arguments)
    for (helper, twin, position, admits) in _VECTOR_HELPER_TWINS
        f === helper || continue
        length(arguments) == position + 1 || return nothing
        all(_rw_one, @view arguments[1:position-1]) || return nothing
        return admits(_vector_log_callee(arguments[position])) ? twin : nothing
    end
    nothing
end

# What is safe beside the vector forms of a rewritten cell. A softplus still
# written as its source's function or helper is not: the CPU's vectors are too
# narrow for its twin, and it keeps the loop scalar.
_vector_cell_safe(f) = f === _log_exp_sum || f === _add_with_sign_proof ||
    f === _exp_with_sign_proof || f === _scale_with_sign_proof || f === _scaled_with_sign_proof ||
    any(twin -> f === twin, _RK_VECTOR_HELPERS) ||
    (f !== rk_log1pexp && f !== LogExpFunctions.log1pexp && _vector_safe_function(f))

# `ex` with the vector form of each call that has one: the vector logarithm of
# a sum of exponentials (vector_log.jl) and, where a vector holds enough cells
# for it, the vector softplus for the functions and the helpers that compute
# one. An expression that holds none is returned as it is.
function _vector_cell_forms(ex, doubles)
    ex isa Expr || return ex
    ex.head in _RW_OPAQUE && return ex
    args = ex.args
    replaced = nothing
    for (i, a) in enumerate(args)
        b = _vector_cell_forms(a, doubles)
        b === a && continue
        replaced === nothing && (replaced = copy(args))
        replaced[i] = b
    end
    replaced === nothing || (ex = Expr(ex.head, replaced...))
    ex.head === :call && !isempty(ex.args) || return ex
    f = _vector_log_callee(ex.args[1])
    if f === log && length(ex.args) == 2
        sum = ex.args[2]
        # A lone exponential is left to the rewrites that cancel it.
        Meta.isexpr(sum, :call) && length(sum.args) >= 3 || return ex
        proven = _exp_sum_with_sign_proof(sum)
        return proven === nothing ? ex : Expr(:call, _rw_helper(:_log_exp_sum), proven)
    end
    (f === nothing || doubles < _VECTOR_SOFTPLUS_DOUBLES) && return ex
    twin = _vector_softplus_twin(f, doubles)
    twin === nothing || return Expr(:call, _rw_helper(twin), ex.args[2:end]...)
    helper = _vector_helper_twin(f, @view ex.args[2:end])
    helper === nothing ? ex : Expr(:call, _rw_helper(nameof(helper)), ex.args[2:end]...)
end

# A summed plate's cell after the stable rewrites, as vector code where every
# call of it can be: the cell with its vector forms in place when nothing else
# in it keeps the loop scalar, and the cell as it stands otherwise, because
# one call that stays scalar makes each vector form the slower one.
function _vectorize_cell(body)
    doubles = _vector_doubles()
    doubles >= 2 || return body
    candidate = _vector_cell_forms(body, doubles)
    candidate === body && return body
    _vector_safe_cell(candidate, Returns(nothing), _vector_cell_safe) ? candidate : body
end
