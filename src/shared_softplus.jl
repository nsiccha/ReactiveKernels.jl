# The value-only recipe skips work in both tails and returns the joint
# recipe's Float64 bits there. For `x >= 34`, `exp(-x) < 2^-48 <= eps(x) / 2`,
# so `x + log1p(exp(-x))` rounds to `x`: no `exp`. For `x < -37`,
# `exp(x) < 2^-53`, where Julia's `log1p` returns its argument: no `log1p`.
@_record_source @inline _rk_log1pexp_value(x) =
    x >= 34.0 ? x : x < -37.0 ? exp(x) : _rk_log1pexp_middle(x)

@_record_source @inline _rk_log1pexp_middle(x) =
    max(x, 0.0) + log1p(exp(-abs(x)))

# Near zero, select the positive part without branching on the sign. Farther
# out, the sign-specific expressions are faster on wider logit distributions.
# The split at 1 changes only the evaluation cost: all three arms perform the
# joint recipe's arithmetic, omitting an addition of zero for negative x.
@_record_source @inline function _rk_log1pexp_middle(x::Float64)
    if abs(x) <= 1.0
        ifelse(x > 0.0, x, 0.0) + log1p(exp(-abs(x)))
    elseif x > 0.0
        x + log1p(exp(-x))
    else
        log1p(exp(x))
    end
end

# A joint numerical graph keeps the expensive exponential available to both
# the stable value and its partial. Planning selects the value-only recipe for
# a primal call and the joint recipe for a derivative cut.
@kernel rk_log1pexp_graph(x::Float64) = begin
    value::Float64 = _rk_log1pexp_value(x)
    q::Float64 = exp(-abs(x))
    (value, partial::Float64) = (max(x, 0.0) + log1p(q), ifelse(x >= 0, 1.0, q) / (1 + q))
    return value, partial
end

"""
    rk_log1pexp(x::Float64)

Stable `log(1 + exp(x))` with a generated scalar derivative rule. The joint
value/partial graph shares `exp(-abs(x))`; neither tail evaluates `exp(abs(x))`.
A primal-only call selects a value recipe that returns `x` for `x >= 34`
without an `exp` and `exp(x)` for `x < -37` without a `log1p`, the same
Float64 values. Ordinary supported AD backends select the joint
value/partial recipe.
"""
const rk_log1pexp = scalar_derivative_rule(rk_log1pexp_graph;
    primal=:value, partials=(x=:partial,), name=:rk_log1pexp)

# ------------------------------------------------ the softplus of a vector loop
#
# Base's `log1p` chooses among range branches, so a loop that calls it stays
# scalar. SLEEFPirates' `log1p` is straight-line arithmetic: a loop over it runs
# as vector code, at about 1.7 ns a cell where the recipes above take about
# 8 ns. Called one number at a time it takes about twice as long as Base's,
# and one other call that keeps a loop scalar (Base's `log`, say) makes a cell
# with it twice as slow as before. So only a loop whose every call the
# compiler can vectorize calls it: a summed plate's cells
# (`_vector_safe_cell`) and a broadcast materialized by itself
# (`_vector_safe_broadcast`). Everywhere else `rk_log1pexp` and
# `LogExpFunctions.log1pexp` run as written.
@_record_source @inline _vector_log1p(q::Float64) = SLEEFPirates.log1p(q)

# `exp(-abs(x))` cannot overflow, and both tails follow from it without a
# branch. For `x >= 34`, `exp(-x) < 2^-48 <= eps(x) / 2`, so the sum rounds to
# `x`. For very negative `x` the first term is zero and `log1p` of a number
# below `2^-53` is that number, `exp(x)`. `max(x, 0)` is one half of this
# identity for `log(1 + exp(x))`, which is smooth at zero with derivative
# `logistic(x)`. The exponential is Base's, inlined where it is called
# (`_native_scalar_exp`).
@_record_source @inline _vector_softplus_value(x::Float64) =
    max(x, 0.0) + _vector_log1p(_native_scalar_exp(-abs(x)))

# The value alone comes from the vector recipe. A derivative cut keeps Base's
# `exp` and `log1p` and the partial `rk_log1pexp_graph` authors: the loops that
# run a cut are not vector loops.
@kernel _vector_softplus_graph(x::Float64) = begin
    value::Float64 = _vector_softplus_value(x)
    q::Float64 = exp(-abs(x))
    (value, partial::Float64) = (max(x, 0.0) + log1p(q), ifelse(x >= 0, 1.0, q) / (1 + q))
    return value, partial
end

# `log(1 + exp(x))` of a `Float64` inside a vector loop. Its value agrees with
# `rk_log1pexp(x)` to one unit in the last place, and a derivative cut returns
# `rk_log1pexp`'s value and partial bit for bit.
const _vector_softplus = scalar_derivative_rule(_vector_softplus_graph;
    primal=:value, partials=(x=:partial,), name=:_vector_softplus)

# What a vector loop calls where its source calls `rk_log1pexp` or
# `LogExpFunctions.log1pexp`: the vector softplus for a `Float64`, the source's
# own function for any other argument.
@inline _vector_rk_log1pexp(x::Float64) = _vector_softplus(x)
@inline _vector_rk_log1pexp(x) = rk_log1pexp(x)
@inline _vector_log1pexp(x::Float64) = _vector_softplus(x)
@inline _vector_log1pexp(x) = LogExpFunctions.log1pexp(x)

# The logarithms of a logistic are a softplus by definition
# (`loglogistic(x) = -log1pexp(-x)`, `log1mlogistic(x) = -log1pexp(x)`), so a
# vector loop computes them from the vector softplus too.
@inline _vector_loglogistic(x::Float64) = -_vector_softplus(-x)
@inline _vector_loglogistic(x) = LogExpFunctions.loglogistic(x)
@inline _vector_log1mlogistic(x::Float64) = -_vector_softplus(x)
@inline _vector_log1mlogistic(x) = LogExpFunctions.log1mlogistic(x)

# The loop whose compiled form `_vector_doubles` reads: a sum over the vector
# softplus, as a summed plate's cell loop runs it.
function _vector_softplus_probe(x::Vector{Float64})
    total = 0.0
    @inbounds @simd for i in eachindex(x)
        total += _vector_softplus_value(x[i])
    end
    total
end

# How many `Float64` one vector holds in the code this process compiles for a
# sum over the vector softplus: 8 with AVX-512, 4 with AVX2, 2 with SSE2 or
# NEON, 1 where the loop stays scalar. It is read once per process from the
# compiled loop itself, so it follows the CPU target Julia was started with
# (`-C`), which a query of the host's CPU does not. 0 means not read yet
# (`__init__` resets it: an image must not carry another process's answer).
const _VECTOR_DOUBLES = Ref(0)
function _vector_doubles()
    doubles = _VECTOR_DOUBLES[]
    doubles == 0 || return doubles
    io = IOBuffer()
    InteractiveUtils.code_llvm(io, _vector_softplus_probe, Tuple{Vector{Float64}}; debuginfo = :none)
    widths = (parse(Int, m.captures[1]) for m in eachmatch(r"<(\d+) x double>", String(take!(io))))
    _VECTOR_DOUBLES[] = maximum(widths; init = 1)
end

# The vector softplus costs about twice the scalar one a cell, so it pays off
# only where a vector holds at least four cells: per cell of a summed loop,
# 1.8 ns against 11.7 ns with eight (Zen 5) and 5.3 ns against 9.6 ns with
# four (Zen 3). With two it would be level at best, so those loops keep the
# source's softplus.
const _VECTOR_SOFTPLUS_DOUBLES = 4

# The functions that compute a softplus, each with the name of its vector twin.
_vector_softplus_twins() = ((rk_log1pexp, :_vector_rk_log1pexp),
    (LogExpFunctions.log1pexp, :_vector_log1pexp),
    (LogExpFunctions.loglogistic, :_vector_loglogistic),
    (LogExpFunctions.log1mlogistic, :_vector_log1mlogistic))

# The vector twin of such a function where a vector holds `doubles` cells (this
# process's CPU unless given), or `nothing`.
function _vector_softplus_twin(f, doubles = _vector_doubles())
    doubles < _VECTOR_SOFTPLUS_DOUBLES && return nothing
    for (source, twin) in _vector_softplus_twins()
        f === source && return twin
    end
    nothing
end

# The function a vector-loop call stands for, for the passes that read a cell
# by the function it computes: `LogExpFunctions.log1pexp` for either softplus
# twin, the logistic logarithm for its twin.
_vector_softplus_source(f) =
    f === _vector_rk_log1pexp || f === _vector_log1pexp ? LogExpFunctions.log1pexp :
    f === _vector_loglogistic ? LogExpFunctions.loglogistic :
    f === _vector_log1mlogistic ? LogExpFunctions.log1mlogistic : f

# Calls known to leave a loop straight-line arithmetic: the compiler turns a
# loop of these into vector code. They are the answers that need no asking;
# for a call of any other function the compiler is asked (`_vector_compiles`,
# vector_cells.jl).
_vector_safe_function(f) = f === (+) || f === (-) || f === (*) || f === (/) ||
    f === _native_mul || f === _native_scalar_exp ||
    f === abs || f === abs2 || f === max || f === min || f === ifelse || f === muladd ||
    f === float || f === identity || f === Float64 ||
    f === (<) || f === (<=) || f === (>) || f === (>=) || f === (==) || f === (!=) ||
    f === (!) || f === (&) || f === (|) || f === xor ||
    f === Base.CartesianIndex || f === Base.Broadcast._broadcast_getindex ||
    f === _plate_read || f === _plate_total_add ||
    f === rk_log1pexp || f === LogExpFunctions.log1pexp ||
    f === _vector_rk_log1pexp || f === _vector_log1pexp ||
    f === _vector_loglogistic || f === _vector_log1mlogistic

# Whether a summed plate's cell `ex` is straight-line arithmetic: assignments
# and calls that leave a loop vector code, with no lazy branch, loop or indexed
# read of its own (a constant's table read aside). A call is one of
# `safe_function`'s, or one the compiler turns into vector code together with
# the expressions it is given (`_vector_compiles`). `callable(callee)` answers
# the function a callee that is not a global names, or `nothing`.
function _vector_safe_cell(ex, callable = Returns(nothing), safe_function = _vector_safe_function)
    ex isa Expr || return true
    head = ex.head
    head in (:quote, :inert, :meta, :inbounds, :boundscheck, :loopinfo, :line) && return true
    if head === :call
        isempty(ex.args) && return false
        value = _vector_callee_value(ex.args[1])
        value === nothing && (value = callable(ex.args[1]))
        value === nothing && return false
        safe_function(something(value)) || return _vector_compiles(ex, callable)
        return all(arg -> _vector_safe_cell(arg, callable, safe_function), @view ex.args[2:end])
    elseif head === :ref
        return length(ex.args) == 2 && ex.args[1] === _CONSTANTS_ARG
    elseif head in (:block, :(=), :local, :let, :tuple, :(::))
        return all(arg -> _vector_safe_cell(arg, callable, safe_function), ex.args)
    end
    false
end

# Whether a broadcast of a softplus over operands of these types is vector
# code once materialized: each operand is an array or a number, or a lazy
# broadcast of `_vector_safe_function`s over such operands.
_vector_safe_broadcast(types) = all(_vector_safe_operand, types)
_vector_safe_operand(T) = T isa Type && (T <: Union{Number,AbstractArray{<:Number}} ||
    T isa DataType && T <: Base.Broadcast.Broadcasted && length(T.parameters) == 4 &&
    _vector_safe_broadcast_function(T.parameters[3]) &&
    T.parameters[4] isa DataType && T.parameters[4] <: Tuple &&
    all(_vector_safe_operand, T.parameters[4].parameters))
_vector_safe_broadcast_function(F) = F isa DataType && isdefined(F, :instance) &&
    _vector_safe_function(F.instance)

# The function a callee expression names, when it is a global that is bound.
function _vector_callee_value(callee)
    callee isa GlobalRef && isdefined(callee.mod, callee.name) &&
        return Some(getglobal(callee.mod, callee.name))
    callee isa Function && return Some(callee)
    nothing
end

# `ex` with each softplus call replaced by its vector twin, or `ex` itself
# where it holds none. `ex` is code a vector loop runs: a summed plate's cell.
# `callable(callee)` answers the function a callee that is not a global names,
# or `nothing`.
function _vector_softplus_calls(ex, callable = Returns(nothing))
    ex isa Expr || return ex
    ex.head in (:quote, :inert, :meta, :macrocall) && return ex
    args = Any[_vector_softplus_calls(arg, callable) for arg in ex.args]
    changed = any(i -> args[i] !== ex.args[i], eachindex(args))
    if ex.head === :call && !isempty(args)
        value = _vector_callee_value(ex.args[1])
        value === nothing && (value = callable(ex.args[1]))
        twin = value === nothing ? nothing : _vector_softplus_twin(something(value))
        twin === nothing || (args[1] = GlobalRef(@__MODULE__, twin); changed = true)
    end
    changed ? Expr(ex.head, args...) : ex
end
