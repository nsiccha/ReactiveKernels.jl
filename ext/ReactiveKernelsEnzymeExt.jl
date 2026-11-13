# Enzyme adapter for rules generated from one pure-math graph
# (`src/derivative_rules.jl`). The rule methods are generic: the call's
# activity pattern selects the graph cut, and its partials combine with
# Enzyme's directions or covectors by the scalar chain rule. The native AD
# preparation below chooses a bounded forward batch over the unchanged RK
# callable. No derivative mathematics is authored here, and nothing attaches
# to a function ReactiveKernels does not own. This extension also declares
# which Enzyme modes retain WANT values during a prepared gradient's primal
# sweep, and which shadow updates a prepared operator's reverse mode compiles.
module ReactiveKernelsEnzymeExt

using ReactiveKernels: ScalarDerivativeRule, derivative_cut, DerivativeRule, forward_cut,
    has_forward_branch, has_reverse_branch, stage_primal, stage_reverse
using ReactiveKernels: _stage_residual_sources, _rule_activity, _derivative_inputs,
    rule_inputs
import ReactiveKernels
import Enzyme
using Enzyme: Const, Active, Duplicated, DuplicatedNoNeed, BatchDuplicated,
    BatchDuplicatedNoNeed, MixedDuplicated
using Enzyme.EnzymeCore: Annotation
import Enzyme.EnzymeRules
using Enzyme.EnzymeRules: FwdConfig, RevConfig, AugmentedReturn, needs_primal,
    needs_shadow, width, overwritten

@inline _values(args::Tuple) = map(arg -> arg.val, args)

# ---------------------------------------------------------------- reverse ----

# Activity patterns are read off the argument TYPES, so they are literal
# tuples at code-generation time and every cut selection folds statically.
@generated _reverse_active(args::Tuple) = Tuple(T <: Active for T in args.parameters)

function EnzymeRules.augmented_primal(
        config::RevConfig, rule::Const{<:ScalarDerivativeRule}, ::Type{RT},
        args::Vararg{Annotation,N}) where {RT,N}
    RT <: Union{Const,Active} || throw(ArgumentError(
        "ScalarDerivativeRule reverse rule: unsupported return activity $(RT); " *
        "a real scalar result is Const or Active"))
    active = _reverse_active(args)
    values = _values(args)
    if any(active) && RT <: Active
        cut = derivative_cut(rule.val, active, values...)
        primal = first(cut)
        tape = Base.tail(cut)
    else
        primal = rule.val(values...)
        tape = nothing
    end
    AugmentedReturn(needs_primal(config) ? primal : nothing, nothing, tape)
end

function EnzymeRules.reverse(
        ::RevConfig, ::Const{<:ScalarDerivativeRule}, dret::Active, tape,
        args::Vararg{Annotation,N}) where {N}
    _reverse_shadows(dret.val, tape, args)
end

function EnzymeRules.reverse(
        config::RevConfig, ::Const{<:ScalarDerivativeRule}, ::Type{<:Const}, tape,
        args::Vararg{Annotation,N}) where {N}
    # A constant result contributes zero even when an input annotation is
    # Active. Enzyme requires a numeric cotangent for that input slot.
    map(arg -> arg isa Active ? _zero_tangent(Val(width(config)), arg.val) : nothing, args)
end

@inline _reverse_shadows(ȳ, ::Nothing, args::Tuple) = map(_ -> nothing, args)
@inline _reverse_shadows(ȳ, tape::Tuple, ::Tuple{}) = ()
@inline function _reverse_shadows(ȳ, tape::Tuple, args::Tuple)
    arg = first(args)
    if arg isa Active
        (_reverse_shadow(ȳ, first(tape)),
         _reverse_shadows(ȳ, Base.tail(tape), Base.tail(args))...)
    else
        (nothing, _reverse_shadows(ȳ, tape, Base.tail(args))...)
    end
end
@inline _reverse_shadow(ȳ::Number, partial) = ȳ * partial
# Batch width > 1: one covector per lane.
@inline _reverse_shadow(ȳ::Tuple, partial) = map(lane -> lane * partial, ȳ)

# ---------------------------------------------------------------- forward ----

const _Tangent = Union{Duplicated,DuplicatedNoNeed,BatchDuplicated,BatchDuplicatedNoNeed}
@generated _forward_active(args::Tuple) = Tuple(T <: _Tangent for T in args.parameters)

function EnzymeRules.forward(
        config::FwdConfig, rule::Const{<:ScalarDerivativeRule}, ::Type{RT},
        args::Vararg{Annotation,N}) where {RT,N}
    values = _values(args)
    if RT <: Const || !needs_shadow(config)
        primal = rule.val(values...)
        return needs_primal(config) ? primal : nothing
    end
    active = _forward_active(args)
    if any(active)
        cut = derivative_cut(rule.val, active, values...)
        primal = first(cut)
        tangent = _forward_accumulate(
            _zero_tangent(Val(width(config)), primal), Base.tail(cut), args)
    else
        primal = rule.val(values...)
        tangent = _zero_tangent(Val(width(config)), primal)
    end
    needs_primal(config) || return tangent
    width(config) == 1 ? Duplicated(primal, tangent) :
        BatchDuplicated(primal, tangent)
end

@inline _zero_tangent(::Val{1}, primal) = zero(primal)
@inline _zero_tangent(::Val{W}, primal) where {W} = ntuple(_ -> zero(primal), Val(W))

@inline _forward_accumulate(acc, ::Tuple{}, ::Tuple{}) = acc
@inline function _forward_accumulate(acc, partials::Tuple, args::Tuple)
    arg = first(args)
    if arg isa _Tangent
        _forward_accumulate(_forward_add(acc, first(partials), arg.dval),
                            Base.tail(partials), Base.tail(args))
    else
        _forward_accumulate(acc, partials, Base.tail(args))
    end
end
@inline _forward_add(acc::Number, partial, direction::Number) =
    acc + partial * direction
@inline _forward_add(acc::Tuple, partial, directions::Tuple) =
    map((lane, direction) -> lane + partial * direction, acc, directions)

# ================================================================ vector ====
# Rules with authored forward/reverse branches (`DerivativeRule`). Reverse
# mode stages the graph: the augmented primal runs the primal stage of the
# selected activity pattern (`stage_primal`: the primal plus the reverse
# cut's covector-independent residuals, computed once), hands Enzyme a zero
# shadow for an array result (Enzyme accumulates the covector into it), and
# retains those residuals; the reverse pass runs `stage_reverse` with the
# covector and accumulates the cotangents into the argument shadows. Forward mode runs the forward cut with
# Enzyme's directions (zero for inactive inputs), once per batch lane.

const _ReverseActive = Union{Active,Duplicated,BatchDuplicated}
# The activity mask as a TYPE (bit `i` set when argument `i` is active), read
# off the annotation types at code generation. Cut selection and the
# no-derivative branch then never depend on constant propagation, which
# inference skips for large rule types: a value-level mask there left
# `augmented_primal` returning a Union of tapes, and Enzyme compiled `reverse`
# against the `nothing` one.
@generated function _reverse_mask(args::Tuple)
    mask = 0
    for (i, T) in enumerate(args.parameters)
        T <: _ReverseActive && (mask |= 1 << (i - 1))
    end
    :(Val{$mask}())
end

# A data declaration is a promise about this primitive's derivative domain,
# not a numerical zero. Reject active data instead of manufacturing a shadow
# for it (a data record may itself contain floating-point coordinates).
@generated function _check_data_activity(rule, ::Val{D}, args::Tuple) where {D}
    for (i, T) in enumerate(args.parameters)
        if D & (1 << (i - 1)) == 0 && !(T <: Const)
            return :(throw(ArgumentError("Enzyme through " * string(rule) *
                ": data-only input :" * string(rule_inputs(rule)[$i]) * " requires Const")))
        end
    end
    :(nothing)
end

@inline _zero_shadow(::Val{1}, primal) = _zero_like(primal)
@inline _zero_shadow(::Val{W}, primal) where {W} = ntuple(_ -> _zero_like(primal), Val(W))
@inline _zero_like(x::AbstractArray) = zero(x)
@inline _zero_like(x::Number) = zero(x)

# Graph provenance does not prove storage ownership. Preserve derivative
# snapshots across caller mutation, borrowing only backend-proven stable input
# arrays in flat environments when no mutable primal alias escapes.
# Use Julia's documented deepcopy specialization hook on an owned wrapper. The
# memo is shared across the flat environment, preserving repeated array
# references. Nested containers retain the existing whole-tuple snapshot path.
struct _DerivativeResidualEnvironment{R,S,F}
    residuals::R
    sources::S
    overwritten::F
    borrow_inputs::Bool
end

_flat_residual_value(value) = isbitstype(typeof(value)) ||
    (value isa Array && isbitstype(eltype(value)))
_flat_residual_environment(residuals::Tuple) = all(_flat_residual_value, residuals)

_seed_borrowed_residuals!(memo, ::Tuple{}, ::Tuple{}, flags) = nothing
function _seed_borrowed_residuals!(memo, residuals::Tuple, sources::Tuple, flags)
    value = first(residuals)
    source = first(sources)
    # A direct dense bits array has no reference-valued element. The backend
    # flag must establish that its memory is unchanged through the reverse.
    if value isa Array && isbitstype(eltype(value)) && source > 0 &&
            !flags[source + 1]
        # Keep Base's heterogeneous memo implementation at its Base boundary;
        # the caller and the retained tuple positions stay concretely typed.
        Base.@noinline setindex!(memo, value, value)
    end
    _seed_borrowed_residuals!(memo, Base.tail(residuals), Base.tail(sources), flags)
end

function Base.deepcopy_internal(env::_DerivativeResidualEnvironment{R}, memo::IdDict) where {R}
    env.borrow_inputs && _seed_borrowed_residuals!(memo, env.residuals, env.sources,
        env.overwritten)
    # Preserve each structural tuple position's type. Base's tuple copier uses
    # integer indexing and infers a tuple of unions for heterogeneous residuals.
    retained = map(env.residuals) do value
        isbitstype(typeof(value)) && return value
        (Base.@noinline Base.deepcopy_internal(value, memo))::typeof(value)
    end
    _DerivativeResidualEnvironment(retained, env.sources, env.overwritten, env.borrow_inputs)
end

# Dense bits arrays have contiguous payloads. Compare their occupied byte
# ranges, including distinct array headers that wrap overlapping storage.
# Subtract the smaller address so an end-address addition cannot overflow.
@inline function _residual_storage_overlaps(a::Array, b::Array)
    isempty(a) && return false
    isempty(b) && return false
    GC.@preserve a b begin
        pa, pb = UInt(pointer(a)), UInt(pointer(b))
        pa <= pb ? pb - pa < sizeof(a) : pa - pb < sizeof(b)
    end
end

_protect_primal_aliases(::Tuple{}, ::Tuple{}, flags, primal) = flags
function _protect_primal_aliases(residuals::Tuple, sources::Tuple, flags, primal)
    value, source = first(residuals), first(sources)
    if value isa Array && isbitstype(eltype(value)) && source > 0 &&
            !flags[source + 1] && _residual_storage_overlaps(value, primal)
        flags = Base.setindex(flags, true, source + 1)
    end
    _protect_primal_aliases(Base.tail(residuals), Base.tail(sources), flags, primal)
end

_borrow_all_residuals(::Tuple{}, ::Tuple{}, flags) = true
function _borrow_all_residuals(residuals::Tuple, sources::Tuple, flags)
    value, source = first(residuals), first(sources)
    (isbitstype(typeof(value)) || (source > 0 && !flags[source + 1])) &&
        _borrow_all_residuals(Base.tail(residuals), Base.tail(sources), flags)
end

@inline function _retain_residuals(sources::NTuple{R,Int}, residuals::NTuple{R,Any},
        primal, flags) where {R}
    isbitstype(typeof(residuals)) && return residuals
    if !_flat_residual_environment(residuals)
        return Base.@noinline deepcopy(residuals)
    end
    # A mutable result can expose input storage independently of its write
    # flag. For a dense bits-array result, protect only inputs whose storage
    # overlaps it; a fresh result need not force copies of readonly inputs.
    # Other mutable result structures retain the complete snapshot path.
    if !isbitstype(typeof(primal))
        if primal isa Array && isbitstype(eltype(primal))
            flags = _protect_primal_aliases(residuals, sources, flags, primal)
        else
            return Base.@noinline deepcopy(residuals)
        end
    end
    # Flat readonly inputs need neither a copy nor a deepcopy identity memo.
    # Check after protecting storage exposed by the mutable primal result.
    _borrow_all_residuals(residuals, sources, flags) && return residuals
    env = _DerivativeResidualEnvironment(residuals, sources, flags, true)
    saved = Base.@noinline deepcopy(env)
    saved.residuals
end

function EnzymeRules.augmented_primal(
        config::RevConfig, rule::Const{<:DerivativeRule}, ::Type{RT},
        args::Vararg{Annotation,N}) where {RT,N}
    _check_data_activity(rule.val, _rule_activity(rule.val), args)
    mask = _reverse_mask(args)
    values = _values(args)
    if mask === Val(0) || RT <: Const
        primal = rule.val(values...)
        shadow = needs_shadow(config) ? _zero_shadow(Val(width(config)), primal) : nothing
        return AugmentedReturn(needs_primal(config) ? primal : nothing, shadow, nothing)
    end
    has_reverse_branch(rule.val) || throw(ArgumentError(
        "reverse-mode Enzyme through $(rule.val): the rule has no reverse branch " *
        "(author a covector + cotangents branch in its graph)"))
    primal, staged = stage_primal(rule.val, mask, values...)
    shadow = needs_shadow(config) ? _zero_shadow(Val(width(config)), primal) : nothing
    residuals = _retain_residuals(_stage_residual_sources(rule.val, mask), staged, primal,
                                  overwritten(config))
    AugmentedReturn(needs_primal(config) ? primal : nothing, shadow, (shadow, residuals))
end

function EnzymeRules.reverse(
        config::RevConfig, rule::Const{<:DerivativeRule}, dret, tape,
        args::Vararg{Annotation,N}) where {N}
    # No derivative flowed (constant result or no active input): an Active
    # input still needs a numeric zero cotangent.
    tape === nothing && return map(arg -> arg isa Active ?
        _zero_tangent(Val(width(config)), arg.val) : nothing, args)
    shadow, residuals = tape
    _reverse_lanes(Val(width(config)), rule.val, _reverse_mask(args), dret, shadow,
                   residuals, args)
end

# Width 1: one covector, one cotangent per active input.
@inline function _reverse_lanes(::Val{1}, rule, ::Val{M}, dret, shadow, residuals, args::Tuple) where {M}
    ȳ = dret isa Active ? dret.val : shadow
    cotangents = stage_reverse(rule, Val(M), residuals, ȳ)
    _apply_cotangents(args, cotangents)
end
# Width W: one covector per lane; Active arguments return a tuple of lanes,
# batched arguments accumulate lane by lane.
@inline function _reverse_lanes(::Val{W}, rule, ::Val{M}, dret, shadow, residuals, args::Tuple) where {W,M}
    lanes = ntuple(Val(W)) do w
        ȳ = dret isa Active ? dret.val[w] : shadow[w]
        stage_reverse(rule, Val(M), residuals, ȳ)
    end
    _apply_cotangent_lanes(args, lanes, Val(W))
end

@inline _apply_cotangents(::Tuple{}, ::Tuple) = ()
@inline function _apply_cotangents(args::Tuple, cotangents::Tuple)
    arg = first(args)
    if arg isa Active
        (first(cotangents), _apply_cotangents(Base.tail(args), Base.tail(cotangents))...)
    elseif arg isa Duplicated
        _accumulate!(arg.dval, first(cotangents))
        (nothing, _apply_cotangents(Base.tail(args), Base.tail(cotangents))...)
    else
        (nothing, _apply_cotangents(Base.tail(args), cotangents)...)
    end
end
@inline _apply_cotangent_lanes(::Tuple{}, lanes::Tuple, ::Val) = ()
@inline function _apply_cotangent_lanes(args::Tuple, lanes::Tuple, ::Val{W}) where {W}
    arg = first(args)
    if arg isa Active
        value = ntuple(w -> first(lanes[w]), Val(W))
        (value, _apply_cotangent_lanes(Base.tail(args), map(Base.tail, lanes), Val(W))...)
    elseif arg isa BatchDuplicated
        for w in 1:W
            _accumulate!(arg.dval[w], first(lanes[w]))
        end
        (nothing, _apply_cotangent_lanes(Base.tail(args), map(Base.tail, lanes), Val(W))...)
    else
        (nothing, _apply_cotangent_lanes(Base.tail(args), lanes, Val(W))...)
    end
end
@inline function _accumulate!(shadow::AbstractArray, cotangent)
    shadow .+= cotangent
    nothing
end

@inline _direction(arg::Union{Duplicated,DuplicatedNoNeed}) = arg.dval
@inline _direction(arg::Annotation) = _zero_like(arg.val)
@inline _direction(arg::Union{BatchDuplicated,BatchDuplicatedNoNeed}, w::Int) = arg.dval[w]
@inline _direction(arg::Annotation, w::Int) = _zero_like(arg.val)

function EnzymeRules.forward(
        config::FwdConfig, rule::Const{<:DerivativeRule}, ::Type{RT},
        args::Vararg{Annotation,N}) where {RT,N}
    activity = _rule_activity(rule.val)
    _check_data_activity(rule.val, activity, args)
    values = _values(args)
    if RT <: Const || !needs_shadow(config)
        primal = rule.val(values...)
        return needs_primal(config) ? primal : nothing
    end
    has_forward_branch(rule.val) || throw(ArgumentError(
        "forward-mode Enzyme through $(rule.val): the rule has no forward branch " *
        "(author a directions + tangent branch in its graph)"))
    differentiated = _derivative_inputs(activity, args)
    if width(config) == 1
        primal, tangent = forward_cut(rule.val, values..., map(_direction, differentiated)...)
        return needs_primal(config) ? Duplicated(primal, tangent) : tangent
    end
    lanes = ntuple(Val(width(config))) do w
        forward_cut(rule.val, values..., map(arg -> _direction(arg, w), differentiated)...)
    end
    tangents = map(last, lanes)
    needs_primal(config) ? BatchDuplicated(first(lanes)[1], tangents) : tangents
end

# ------------------------------------------------- retained WANT outputs ----

# A reverse-mode Enzyme gradient runs the differentiated target once, at the
# requested point, and keeps the target's writes, so WANT values retained in
# that sweep (`prepare_ad(...; retain)`) are its primal values.
ReactiveKernels.ad_retains_primal_sweep(
    ::ReactiveKernels.DifferentiationInterface.AutoEnzyme{
        <:Union{Nothing,Enzyme.ReverseMode}}) = true

# RKPPL's sampler prepares the non-allocating kernel by default for the same
# reverse-mode backends (`ReactiveKernels._nonallocating_ad_default`).
ReactiveKernels._nonallocating_ad_default(
    ::ReactiveKernels.DifferentiationInterface.AutoEnzyme{
        <:Union{Nothing,Enzyme.ReverseMode}}) = true

# --------------------------------------------- structured gradient buffers ----

# DI's reverse interface can write a structured gradient into caller-owned
# shadows. Keep the existing copy path when the destination representation is
# different, or when zeroing a shadow could modify a primal/context operand.
_dense_shadow_pair(x, dx) = false
_dense_shadow_pair(x::Array{T,N}, dx::Array{T,N}) where {T,N} =
    isbitstype(T) && size(x) == size(dx)
_dense_shadow_pair(x::Base.RefValue{T}, dx::Base.RefValue{T}) where {T} =
    isbitstype(T)

@inline _shadow_disjoint(dx, x) = isbitstype(typeof(x))
@inline _shadow_disjoint(dx::Array, x::Array) =
    isbitstype(eltype(x)) && !_residual_storage_overlaps(dx, x)
@inline function _ref_array_disjoint(ref::Base.RefValue{T}, array::Array) where {T}
    isbitstype(T) && isbitstype(eltype(array)) || return false
    isempty(array) && return true
    GC.@preserve ref array begin
        pr = UInt(Base.unsafe_convert(Ptr{T}, ref))
        pa = UInt(pointer(array))
        pr <= pa ? pa - pr >= sizeof(T) : pr - pa >= sizeof(array)
    end
end
@inline _shadow_disjoint(dx::Base.RefValue, x::Array) = _ref_array_disjoint(dx, x)
@inline function _shadow_disjoint(dx::Array, x::Base.RefValue{T}) where {T}
    isbitstype(T) && return _ref_array_disjoint(x, dx)
    T !== Any && (!isassigned(x) || _shadow_disjoint(dx, x[]))
end
@inline _shadow_disjoint(dx, x::Base.RefValue{T}) where {T} =
    T !== Any && dx !== x && (!isassigned(x) || _shadow_disjoint(dx, x[]))
@inline _shadow_disjoint(dx, x::NamedTuple) = _shadow_disjoint(dx, values(x))
@inline _shadow_disjoint(dx, x::Tuple{}) = true
function _shadow_disjoint(dx, x::NTuple{N,T}) where {N,T}
    for value in x
        _shadow_disjoint(dx, value) || return false
    end
    true
end
@generated function _shadow_disjoint(dx, x::T) where {T<:Tuple}
    checks = [:(_shadow_disjoint(dx, getfield(x, $i)) || return false)
              for i in 1:fieldcount(T)]
    Expr(:block, Expr(:meta, :inline), checks..., true)
end

# This direct check is used only for up to four active ports. Context tuple
# traversal stays linear; larger selectors use the sorted interval sweep.
@inline _dense_shadow_pairs(point::Tuple{}, gradient::Tuple{}) = true
@inline _dense_shadow_pairs(point::Tuple, gradient::Tuple) =
    _dense_shadow_pair(first(point), first(gradient)) &&
    _shadow_disjoint(first(point), Base.tail(point)) &&
    _shadow_disjoint(first(gradient), Base.tail(gradient)) &&
    _dense_shadow_pairs(Base.tail(point), Base.tail(gradient))
@inline _shadows_disjoint(gradient::Tuple{}, sources) = true
@inline _shadows_disjoint(gradient::Tuple, sources) =
    _shadow_disjoint(first(gradient), sources) &&
    _shadows_disjoint(Base.tail(gradient), sources)

@inline function _structured_destinations(point::Tuple, gradient::Tuple, contexts)
    length(point) == length(gradient) &&
        _dense_shadow_pairs(point, gradient) &&
        _shadows_disjoint(gradient, point) &&
        _shadows_disjoint(gradient,
            map(ReactiveKernels.DifferentiationInterface.unwrap, contexts))
end
_structured_destinations(point, gradient, contexts) = false

struct _ShadowInterval
    lo::UInt
    hi::UInt
    kind::UInt8  # 1: active input, 2: destination, 3: inactive context
end
struct _ShadowIntervals
    ranges::Vector{_ShadowInterval}
end
_ShadowIntervals() = _ShadowIntervals(_ShadowInterval[])

@inline function _push_shadow_interval!(work, x::Array, kind)
    isbitstype(eltype(x)) || return false
    isempty(x) && return true
    lo = UInt(pointer(x))
    bytes = sizeof(x)
    bytes == 0 || push!(work.ranges, _ShadowInterval(lo, lo + bytes, kind))
    true
end
@inline function _push_shadow_interval!(work, x::Base.RefValue{T}, kind) where {T}
    if isbitstype(T)
        lo = UInt(Base.unsafe_convert(Ptr{T}, x))
        bytes = sizeof(T)
        bytes == 0 || push!(work.ranges, _ShadowInterval(lo, lo + bytes, kind))
        return true
    end
    T !== Any && (!isassigned(x) || _push_shadow_interval!(work, x[], kind))
end
@inline _push_shadow_interval!(work, x::Tuple{}, kind) = true
function _push_shadow_interval!(work, x::NTuple{N,T}, kind) where {N,T}
    for value in x
        _push_shadow_interval!(work, value, kind) || return false
    end
    true
end
# Heterogeneous fields follow the prepared boundary's structural types;
# homogeneous tuples above retain a runtime loop without copying tuple tails.
@generated function _push_shadow_interval!(work, x::T, kind) where {T<:Tuple}
    checks = [:(_push_shadow_interval!(work, getfield(x, $i), kind) || return false)
              for i in 1:fieldcount(T)]
    Expr(:block, Expr(:meta, :inline), checks..., true)
end
@inline _push_shadow_interval!(work, x::NamedTuple, kind) =
    _push_shadow_interval!(work, values(x), kind)
@inline _push_shadow_interval!(work, x, kind) = isbitstype(typeof(x))

# In-place heapsort keeps worst-case O(m log m) work without a sorting buffer.
function _sift_shadow_intervals!(a, root, last)
    saved = a[root]
    while root <= last ÷ 2
        child = 2root
        child < last && a[child].lo < a[child+1].lo && (child += 1)
        saved.lo >= a[child].lo && break
        a[root] = a[child]
        root = child
    end
    a[root] = saved
end
function _sort_shadow_intervals!(a)
    for root in length(a)÷2:-1:1
        _sift_shadow_intervals!(a, root, length(a))
    end
    for last in length(a):-1:2
        a[1], a[last] = a[last], a[1]
        _sift_shadow_intervals!(a, 1, last-1)
    end
    a
end
function _shadow_intervals_disjoint!(work)
    _sort_shadow_intervals!(work.ranges)
    input_end = UInt(0)
    destination_end = UInt(0)
    context_end = UInt(0)
    for r in work.ranges
        if r.kind == 0x01
            r.lo < max(input_end, destination_end) && return false
            input_end = max(input_end, r.hi)
        elseif r.kind == 0x02
            r.lo < max(input_end, destination_end, context_end) && return false
            destination_end = max(destination_end, r.hi)
        else
            r.lo < destination_end && return false
            context_end = max(context_end, r.hi)
        end
    end
    true
end
@inline _compatible_shadow_pairs(::Tuple{}, ::Tuple{}) = true
function _compatible_shadow_pairs(point::NTuple{N,P}, gradient::NTuple{N,G}) where {N,P,G}
    for i in eachindex(point)
        _dense_shadow_pair(point[i], gradient[i]) || return false
    end
    true
end
@generated function _compatible_shadow_pairs(point::P, gradient::G) where {P<:Tuple,G<:Tuple}
    fieldcount(P) == fieldcount(G) || return false
    checks = [:(_dense_shadow_pair(getfield(point, $i), getfield(gradient, $i)) || return false)
              for i in 1:fieldcount(P)]
    Expr(:block, Expr(:meta, :inline), checks..., true)
end
_interval_destinations!(::Nothing, point, gradient, contexts) = false
function _interval_destinations!(work, point, gradient, contexts)
    length(point) == length(gradient) &&
        _compatible_shadow_pairs(point, gradient) || return false
    empty!(work.ranges)
    sources = map(ReactiveKernels.DifferentiationInterface.unwrap, contexts)
    GC.@preserve point gradient sources begin
        _push_shadow_interval!(work, point, 0x01) &&
        _push_shadow_interval!(work, gradient, 0x02) &&
        _push_shadow_interval!(work, sources, 0x03) &&
        _shadow_intervals_disjoint!(work)
    end
end

function ReactiveKernels._ad_gradient_buffer_storage(
        ::ReactiveKernels.DifferentiationInterface.AutoEnzyme{
            <:Union{Nothing,Enzyme.ReverseMode}}, point::Tuple, contexts)
    length(point) <= 4 && return nothing
    work = _ShadowIntervals()
    sources = map(ReactiveKernels.DifferentiationInterface.unwrap, contexts)
    GC.@preserve point sources begin
        _push_shadow_interval!(work, point, 0x01)
        _push_shadow_interval!(work, point, 0x02)
        _push_shadow_interval!(work, sources, 0x03)
    end
    empty!(work.ranges)
    work
end
@inline _prepared_structured_destinations(prepared, point::Tuple, gradient::Tuple, contexts) =
    length(point) <= 4 ? _structured_destinations(point, gradient, contexts) :
    _interval_destinations!(prepared.storage.buffers, point, gradient, contexts)
_prepared_structured_destinations(prepared, point, gradient, contexts) = false

@inline function ReactiveKernels._ad_prepared_tuple_value_and_gradient!(
        backend::ReactiveKernels.DifferentiationInterface.AutoEnzyme{
            <:Union{Nothing,Enzyme.ReverseMode}},
        prepared, gradient, point, contexts)
    # DI's no-context Enzyme path currently calls gradient!, whose active
    # argument must be an AbstractArray. Preserve the copy path for tuples.
    if !isempty(contexts) &&
       _prepared_structured_destinations(prepared, point, gradient, contexts)
        return ReactiveKernels.DifferentiationInterface.value_and_gradient!(
            prepared.call, gradient, prepared.preparation, backend,
            point, contexts...)
    end
    ReactiveKernels._ad_copy_value_and_gradient!(prepared, gradient, point, contexts)
end

# ------------------------------------------------- atomic shadow updates ----

# Enzyme compiles every shadow accumulation as an atomic update whenever the
# process has more than one default-pool thread, whether or not another thread
# can reach the shadow. A prepared operator's shadows are private to its running
# call (`_ad_prepared_backend`), so a reverse mode that leaves the choice to
# Enzyme is prepared with plain updates. Enzyme still keeps atomic updates in
# code that starts or waits for a task or runs a threaded loop, and a mode
# whose caller chose either way is kept. Only an Enzyme that has the switch
# (`set_atomic_add`, EnzymeCore 0.8.27 on the local dogfood line) defines these
# methods; with any other Enzyme the backend is stored as given.
@static if isdefined(Enzyme, :set_atomic_add)
    _plain_shadow_updates(mode::Enzyme.ReverseMode) =
        Enzyme.atomic_add(mode) === nothing ? Enzyme.set_atomic_add(mode, false) : mode

    ReactiveKernels._ad_prepared_backend(::Val,
            backend::ReactiveKernels.DifferentiationInterface.AutoEnzyme{
                <:Enzyme.ReverseMode,A}) where {A} =
        ReactiveKernels.DifferentiationInterface.AutoEnzyme(;
            mode = _plain_shadow_updates(backend.mode), function_annotation = A)

    # Without a mode, DifferentiationInterface takes gradients and pullbacks in
    # `Enzyme.Reverse`; any other operator keeps the backend as given.
    ReactiveKernels._ad_prepared_backend(::Union{Val{:gradient},Val{:pullback}},
            ::ReactiveKernels.DifferentiationInterface.AutoEnzyme{Nothing,A}) where {A} =
        ReactiveKernels.DifferentiationInterface.AutoEnzyme(;
            mode = _plain_shadow_updates(Enzyme.Reverse), function_annotation = A)
end

# A small active vector can carry one forward tangent lane per coordinate.
# Enzyme differentiates the *same* native RK callable, so every plate cell,
# prior, transform, and bound-data read follows its ordinary primal path. This
# changes the AD algorithm and therefore requires an explicit opt-in without
# an explicit backend mode. It never overrides a requested reverse mode.
# `ReactiveKernels._require_enzyme_repairs`. The repairs its callers name concern
# Julia 1.10's LLVM (typed pointers); later Julia versions differentiate those
# programs with registered Enzyme. Dogfood from `a68104d` reports the repairs it
# carries (`Enzyme.require_dogfood_repairs` throws `Enzyme.DogfoodRepairMissing`
# naming each missing one). Registered Enzyme and older dogfood cannot report
# them, so they get one warning naming the floor, and the derivative is prepared
# anyway: such a pin may carry the repair (`enzyme-use`, "Stating a dogfood floor").
# The warning is given once per process for each requester and set of repairs.
_enzyme_backend(::ReactiveKernels.DifferentiationInterface.AutoEnzyme) = true
_enzyme_backend(backend) = nameof(typeof(backend)) in (:SecondOrder, :MixedMode) &&
    any(f -> _enzyme_backend(getfield(backend, f)), fieldnames(typeof(backend)))

const _UNREPORTED_REPAIRS_WARNED = Set{Tuple{String,Tuple{Vararg{Symbol}}}}()
const _UNREPORTED_REPAIRS_LOCK = ReentrantLock()

function ReactiveKernels._require_enzyme_repairs(
        backend::ReactiveKernels.DifferentiationInterface.AbstractADType,
        repairs::Tuple{Vararg{Symbol}}, requester::AbstractString, floor::AbstractString)
    VERSION < v"1.11-" && _enzyme_backend(backend) || return nothing
    if isdefined(Enzyme, :require_dogfood_repairs)
        Enzyme.require_dogfood_repairs(repairs...; requester)
        return nothing
    end
    key = (String(requester), repairs)
    first = lock(_UNREPORTED_REPAIRS_LOCK) do
        key in _UNREPORTED_REPAIRS_WARNED && return false
        push!(_UNREPORTED_REPAIRS_WARNED, key)
        true
    end
    first && @warn string(uppercasefirst(requester), " can need Enzyme ", floor,
        " or later for native differentiation on Julia 1.10 (dogfood repair ",
        join(repairs, ", "), "). This Enzyme cannot report its dogfood repairs: it is a ",
        "registered release or a dogfood commit before a68104d. If the first gradient ",
        "fails (an EnzymeRuntimeActivityError, for example), move Enzyme to the dogfood ",
        "tip; runtime activity is not the fix (enzyme-use, \"Stating a dogfood floor\").")
    nothing
end

const _FORWARD_BATCH_MODE = Enzyme.set_abi(Enzyme.ForwardWithPrimal, Enzyme.InlineABI)
const _AutomaticModeEnzyme = ReactiveKernels.DifferentiationInterface.AutoEnzyme{Nothing}

struct _NativeForwardBatchAD{K,R,B}
    reverse::R
    basis::B
end

function ReactiveKernels._native_forward_batch_eligible(
        ::_AutomaticModeEnzyme, kernel::ReactiveKernels.PreparedKernel,
        point::Vector{Float64})
    1 <= length(point) <= 4 || return false
    plan = kernel.plan
    recipes = plan.recipes
    all(r -> !r.effectful && !(r.op isa ReactiveKernels._AuthoredScanOp),
        recipes) || return false
    any(recipes) do recipe
        recipe.op isa ReactiveKernels._AuthoredPlateOp || return false
        all(r -> !r.effectful && !(r.op isa Union{
                ReactiveKernels._AuthoredPlateOp,
                ReactiveKernels._AuthoredScanOp}),
            recipe.op.kernel.plan.recipes) || return false
        consumer = ReactiveKernels._authored_plate_sum_recipe(plan, recipe)
        consumer === nothing && return false
        output_id = ReactiveKernels.canon_id(
            plan.graph, only(recipe.outputs).id)
        any(w -> ReactiveKernels.canon_id(plan.graph, w.id) == output_id,
            plan.want) && return false
        !any(candidate -> candidate !== consumer && any(input ->
            ReactiveKernels.canon_id(plan.graph, input.id) == output_id,
            candidate.inputs), recipes)
    end
end

function ReactiveKernels._prepare_native_forward_batch_ad(
        kernel::ReactiveKernels.PreparedKernel, resolver,
        backend::_AutomaticModeEnzyme, args::Tuple, kwargs::NamedTuple, active;
        retain::Tuple = (), warm::Bool = false)
    isempty(retain) || throw(ArgumentError(
        "native forward batching does not retain further WANT values"))
    reverse = ReactiveKernels._prepare_ad(kernel, resolver, backend,
        args, kwargs, active; warm = false)
    point, contexts = ReactiveKernels._ad_prepared_arguments(
        reverse, args, kwargs)
    point isa Vector{Float64} || throw(ArgumentError(
        "native forward batching needs one active Vector{Float64} port"))
    1 <= length(point) <= 4 || throw(ArgumentError(
        "native forward batching supports one to four active coordinates"))
    all(c -> c isa ReactiveKernels.DifferentiationInterface.Constant,
        contexts) || throw(ArgumentError(
        "native forward batching needs read-only Constant contexts"))
    K = length(point)
    basis = ntuple(K) do direction
        [coordinate == direction ? 1.0 : 0.0 for coordinate in 1:K]
    end
    prepared = _NativeForwardBatchAD{K,typeof(reverse),typeof(basis)}(
        reverse, basis)
    warm && Base.invokelatest(ReactiveKernels.ad_value_and_gradient!,
        prepared, similar(point), args...; kwargs...)
    prepared
end

# An optimization hint cannot override an explicitly requested AD mode.
function ReactiveKernels._prepare_native_forward_batch_ad(
        kernel::ReactiveKernels.PreparedKernel, resolver,
        backend::ReactiveKernels.DifferentiationInterface.AutoEnzyme{<:Enzyme.ReverseMode},
        args::Tuple, kwargs::NamedTuple, active;
        retain::Tuple = (), warm::Bool = false)
    ReactiveKernels._prepare_ad(kernel, resolver, backend, args, kwargs, active;
        retain, warm)
end

function _forward_batch_value_gradient!(prepared::_NativeForwardBatchAD{K},
        gradient::AbstractVector, point::Vector{Float64}, contexts) where {K}
    axes(gradient) == axes(point) || throw(DimensionMismatch(
        "gradient axes $(axes(gradient)) differ from active point axes $(axes(point))"))
    annotations = map(c -> Enzyme.Const(c.data), contexts)
    tangents, value = Enzyme.autodiff(_FORWARD_BATCH_MODE,
        prepared.reverse.call, Enzyme.BatchDuplicated(point, prepared.basis),
        annotations...)
    @inbounds for coordinate in 1:K
        gradient[coordinate] = getfield(tangents, coordinate)
    end
    value, gradient
end

function _forward_batch_value_gradient!(prepared::_NativeForwardBatchAD{1},
        gradient::AbstractVector, point::Vector{Float64}, contexts)
    axes(gradient) == axes(point) || throw(DimensionMismatch(
        "gradient axes $(axes(gradient)) differ from active point axes $(axes(point))"))
    annotations = map(c -> Enzyme.Const(c.data), contexts)
    tangent, value = Enzyme.autodiff(_FORWARD_BATCH_MODE,
        prepared.reverse.call, Enzyme.Duplicated(point, prepared.basis[1]),
        annotations...)
    gradient[1] = tangent
    value, gradient
end

function ReactiveKernels.ad_value_and_gradient!(prepared::_NativeForwardBatchAD{K},
        gradient, args...; kwargs...) where {K}
    point, contexts = ReactiveKernels._ad_prepared_arguments(
        prepared.reverse, args, NamedTuple(kwargs))
    if point isa Vector{Float64} && length(point) == K &&
            all(c -> c isa ReactiveKernels.DifferentiationInterface.Constant,
                contexts)
        return _forward_batch_value_gradient!(prepared, gradient, point, contexts)
    end
    ReactiveKernels.ad_value_and_gradient!(prepared.reverse, gradient,
                                           args...; kwargs...)
end

function ReactiveKernels.ad_value_and_gradient(prepared::_NativeForwardBatchAD,
        args...; kwargs...)
    point, _ = ReactiveKernels._ad_prepared_arguments(
        prepared.reverse, args, NamedTuple(kwargs))
    point isa AbstractVector || return ReactiveKernels.ad_value_and_gradient(
        prepared.reverse, args...; kwargs...)
    ReactiveKernels.ad_value_and_gradient!(prepared, similar(point), args...;
                                            kwargs...)
end

ReactiveKernels.ad_gradient(prepared::_NativeForwardBatchAD, args...; kwargs...) =
    last(ReactiveKernels.ad_value_and_gradient(prepared, args...; kwargs...))

ReactiveKernels.ad_value_gradient_and_retained!(prepared::_NativeForwardBatchAD,
        gradient, args...; kwargs...) =
    ReactiveKernels.ad_value_gradient_and_retained!(prepared.reverse,
        gradient, args...; kwargs...)

ReactiveKernels.ad_value_gradient_and_retained(prepared::_NativeForwardBatchAD,
        args...; kwargs...) =
    ReactiveKernels.ad_value_gradient_and_retained(prepared.reverse,
        args...; kwargs...)

# This is an adapter to Enzyme's supplied-shadow API, not a derivative rule.
# Unlike DI.gradient!, autodiff adds into Duplicated storage without first
# clearing it. A cell sweep can therefore share one large cotangent buffer.
const _AccumulatingEnzyme = ReactiveKernels.DifferentiationInterface.AutoEnzyme{
    <:Union{Nothing,Enzyme.ReverseMode},<:Union{Nothing,Enzyme.Const}}
ReactiveKernels._ad_accumulation_supported(::_AccumulatingEnzyme) = true

function ReactiveKernels._ad_accumulate_backend!(backend::_AccumulatingEnzyme,
        call, destination, point, direction, weight, contexts)
    all(c -> c isa ReactiveKernels.DifferentiationInterface.Constant, contexts) ||
        throw(ArgumentError("additive AD requires read-only Constant contexts"))
    mode = backend.mode === nothing ? Enzyme.ReverseWithPrimal : Enzyme.WithPrimal(backend.mode)
    annotations = map(c -> Enzyme.Const(c.data), contexts)
    last(Enzyme.autodiff(mode, Enzyme.Const(call), Enzyme.Active,
        Enzyme.Duplicated(point, destination), Enzyme.Const(direction),
        Enzyme.Const(weight), annotations...))
end


# ------------------------------------------------ generated plate reverse ----
# `_PlateSumRuleBody` (src/plate_reverse.jl): the storage-compatible call of a
# summed plate program with a reverse generated from its own code. The
# augmented primal runs the program, so the
# primal is unchanged; the reverse pass runs the generated reverse once per
# lane with the covector, which adds each batched operand's cotangent into its
# shadow and returns the scalar operands' cotangents. A call whose activity
# the generated reverse does not take differentiates the program with
# Enzyme's ordinary reverse instead.

using ReactiveKernels: _PlateSumRuleBody, _pr_scalar_positions, _pr_batched_positions, _CellArray

function EnzymeRules.augmented_primal(
        config::RevConfig, rule::Const{<:_PlateSumRuleBody}, ::Type{RT},
        args::Vararg{Annotation,N}) where {RT,N}
    values = _pr_values(args)
    primal = rule.val(values...)
    if RT <: Const || all(a -> a isa Const, args)
        return AugmentedReturn(needs_primal(config) ? primal : nothing, nothing, nothing)
    end
    tape = _pr_tape(Val(overwritten(config)), values)
    AugmentedReturn(needs_primal(config) ? primal : nothing, nothing, tape)
end

# The operands' values, unrolled. A plate program can take more operands than
# `map` over a tuple unrolls (32): past that, `map` loses the element types, the
# primal and tape types are no longer inferred, and Enzyme refuses the rule's
# `AugmentedReturn` (a downstream application’s SH program, about 40 operands).
@generated _pr_values(args::Tuple) =
    Expr(:tuple, (:(getfield(args, $i).val) for i in 1:length(args.parameters))...)

# The tape holds a copy of each array operand Enzyme may overwrite before the
# reverse pass (`F` counts the function object at position 1) and nothing
# else: the reverse pass reads every other operand from its annotation, so a
# call whose operands stay intact keeps an empty tape and allocates none.
@generated function _pr_tape(::Val{F}, values::Tuple) where {F}
    Expr(:tuple, (F[i+1] && T <: AbstractArray ? :(copy(getfield(values, $i))) : :nothing
                  for (i, T) in enumerate(values.parameters))...)
end
# The operand values for the reverse pass: the tape's copy where it holds one.
@generated function _pr_operands(tape::Tuple, args::Tuple)
    Expr(:tuple, (tape.parameters[i] === Nothing ? :(getfield(args, $i).val) :
                  :(getfield(tape, $i)) for i in 1:length(args.parameters))...)
end

# The shadow of each batched operand, for lane `w` (`nothing` for a constant one).
@generated function _pr_shadows(::Val{B}, args::Tuple, ::Val{W}, w) where {B,W}
    Expr(:tuple, (:(_pr_shadow(getfield(args, $b), Val($W), w)) for b in B)...)
end

@inline _pr_shadow(arg::Union{Duplicated,DuplicatedNoNeed}, ::Val{1}, w) = arg.dval
@inline _pr_shadow(arg::Union{BatchDuplicated,BatchDuplicatedNoNeed}, ::Val, w) = arg.dval[w]
@inline _pr_shadow(arg::Active{<:Tuple}, ::Val{1}, w) = zeros(eltype(arg.val), length(arg.val))
@inline _pr_shadow(arg, ::Val, w) = nothing

# A batched Active tuple of floats accumulates its cotangent in a vector and
# returns it as a tuple. Define this predicate before the generators that call
# it so it is available in their world when loading the extension from source.
_pr_tuple_operand(::Type{<:Active{T}}) where {T} = T <: Tuple{Vararg{AbstractFloat}}
_pr_tuple_operand(::Type) = false

# The reverse's return: the cotangent of each Active operand (zero for one the
# rule does not differentiate), `nothing` for every other annotation.
@generated function _pr_reverse_return(::Val{S}, args::Tuple, cot::Tuple,
        ::Val{B} = Val(()), shadows::Tuple = ()) where {S,B}
    exprs = Any[]
    for (i, T) in enumerate(args.parameters)
        j = findfirst(==(i), B)
        if T <: Active && j !== nothing && _pr_tuple_operand(T)
            # A tuple operand: its cotangent is the vector the reverse filled.
            V = T.parameters[1]
            n = length(V.parameters)
            # Unrolled: a generated function's body holds no closure.
            elements = Any[:(@inbounds getfield(shadows, $j)[$k]) for k in 1:n]
            push!(exprs, :(convert($V, $(Expr(:tuple, elements...)))::$V))
        elseif T <: Active
            k = findfirst(==(i), S)
            push!(exprs, k === nothing ? :(zero(getfield(args, $i).val)) :
                  :(convert($(T.parameters[1]), getfield(cot, $k))::$(T.parameters[1])))
        else
            push!(exprs, :nothing)
        end
    end
    Expr(:tuple, exprs...)
end

# Whether the generated reverse returns or accumulates the cotangent of every
# operand Enzyme treats as active. The rule was generated for the operands its
# own activity analysis marked active (scalar positions `S`, batched `B`); an
# operand Enzyme also differentiates outside them would otherwise get a zero
# cotangent, which is a wrong gradient. Such a call differentiates the program
# with Enzyme's ordinary reverse instead.
# A batched operand is covered only as a plain array shadow: a `MixedDuplicated`
# operand (a struct of active scalars and arrays, such as a cell broadcast
# `a .+ v`) has a shadow the generated reverse does not write, and neither does
# a lazy cell array of arrays only (`_CellArray`, such as `u .+ v`), which
# Enzyme passes as `Duplicated` with a shadow of the same lazy structure.
@generated function _pr_covers(::Val{S}, ::Val{B}, args::Tuple) where {S,B}
    for (i, T) in enumerate(args.parameters)
        T <: Const && continue
        if T <: Active
            i in S || (i in B && _pr_tuple_operand(T)) || return :(false)
        elseif T <: Union{Duplicated,DuplicatedNoNeed,BatchDuplicated,BatchDuplicatedNoNeed}
            i in B && !(T.parameters[1] <: _CellArray) || return :(false)
        else
            return :(false)
        end
    end
    :(true)
end

struct _PRSeeded{F,T}
    program::F
    seed::T
end
@inline (s::_PRSeeded)(args::Vararg{Any,N}) where {N} = s.seed * s.program(args...)

@inline _pr_rewrap(a::Active, v) = Active(v)
@inline _pr_rewrap(a::Union{Duplicated,DuplicatedNoNeed}, v) = Duplicated(v, a.dval)
@inline _pr_rewrap(a::Const, v) = Const(v)
# Its shadow is a reference the ordinary reverse accumulates into in place.
@inline _pr_rewrap(a::MixedDuplicated, v) = MixedDuplicated(v, a.dval)

# The reverse's return type for these annotations: the value type of each
# Active operand, `nothing` for every other.
@generated _pr_return_type(args::Tuple) = Tuple{(T <: Active ? T.parameters[1] : Nothing
                                                for T in args.parameters)...}

# Each annotation rewrapped around its operand value, unrolled (see `_pr_values`).
@generated _pr_rewrapped(args::Tuple, values::Tuple) =
    Expr(:tuple, (:(_pr_rewrap(getfield(args, $i), getfield(values, $i)))
                  for i in 1:length(args.parameters))...)

function _pr_ordinary_reverse(program::F, ȳ, tape, args::Tuple) where {F}
    annotations = _pr_rewrapped(args, tape)
    first(Enzyme.autodiff(Enzyme.Reverse, Const(_PRSeeded(program, ȳ)), Active,
                          annotations...))::_pr_return_type(args)
end

function EnzymeRules.reverse(
        config::RevConfig, rule::Const{<:_PlateSumRuleBody}, dret, tape,
        args::Vararg{Annotation,N}) where {N}
    r = rule.val.rule
    S = _pr_scalar_positions(r)
    B = _pr_batched_positions(r)
    if tape === nothing
        return _pr_reverse_return(Val(()), args, ())
    end
    operands = _pr_operands(tape, args)
    covered = _pr_covers(Val(S), Val(B), args)
    if width(config) == 1 && !covered
        ȳ = dret isa Active ? dret.val : dret
        return _pr_ordinary_reverse(r.program, ȳ, operands, args)
    end
    W = width(config)
    covered || throw(ArgumentError(
        "plate reverse: a batched (width $W) reverse of a plate program with an active " *
        "operand the generated reverse does not differentiate; prepare without `with_plate_reverse`"))
    if W == 1
        ȳ = dret isa Active ? dret.val : dret
        shadows = _pr_shadows(Val(B), args, Val(1), 1)
        cot = r.reverse(operands..., ȳ, shadows...)
        return _pr_reverse_return(Val(S), args, cot, Val(B), shadows)
    end
    # The lanes' own names: a closure assigning the `ȳ` and `shadows` of the
    # width-1 branches above would share them, boxed, and the width-1 call
    # would dispatch dynamically and allocate every argument.
    lanes = ntuple(Val(W)) do w
        lane_ȳ = dret.val[w]
        lane_shadows = _pr_shadows(Val(B), args, Val(W), w)
        r.reverse(operands..., lane_ȳ, lane_shadows...)
    end
    # Active scalars of a batched call take one cotangent per lane.
    ntuple(Val(N)) do i
        a = args[i]
        a isa Active || return nothing
        k = findfirst(==(i), S)
        ntuple(w -> k === nothing ? zero(a.val[w]) : lanes[w][k], Val(W))
    end
end

end # module
