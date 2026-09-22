# Self-contained recurrence reproducer using the public RK/PPL PK math.
# No dependency on the experimental rectangular helpers in the RK branch.
using ReactiveKernels, ReactiveKernelsPPL, Reactant, Enzyme
using ReactiveKernelsPPL: SubjectSlice, SubjectScalar, LINEAR_EVENT_READ,
    LINEAR_EVENT_DOSE, LINEAR_EVENT_DOSE_SEGMENT, _traced_op_read,
    _pk_tmul4, _pk_tmatvec3, _pk_expm3
const PPL = ReactiveKernelsPPL
@inline function _rectangular_fold(step, init, columns::Tuple, shared::Tuple, marker)
    isempty(columns) && throw(ArgumentError("a rectangular fold needs columns"))
    n = length(first(columns))
    all(c -> c isa AbstractVector && length(c) == n, columns) ||
        throw(DimensionMismatch("rectangular fold columns must be equal-length vectors"))
    Base.require_one_based_indexing(columns...)
    _rectangular_fold_impl(marker, step, init, columns, shared, n)
end

@inline function _rectangular_fold_impl(marker, step, init, columns, shared, n)
    carry = init
    for i in 1:n
        carry = step(carry, map(c -> c[i], columns), shared...)
    end
    carry
end

# Lazy scalar control: inactive singular/overflowing transitions must not run.
@inline _recurrence_branch(pred, yes, no, args) = pred ? yes(args...) : no(args...)

_recurrence_trace(x) = x
_recurrence_trace(x::Tuple) = map(_recurrence_trace, x)
_recurrence_trace(x::NamedTuple) = map(_recurrence_trace, x)
_recurrence_trace(x::AbstractArray) = Reactant.promote_to(Reactant.TracedRArray, x)
_recurrence_trace(x::T) where {T<:Number} =
    Reactant.promote_to(Reactant.TracedRNumber{T}, x)
_recurrence_trace(x::Reactant.TracedRNumber) = copy(x)

function _rectangular_fold_impl(
        marker::Reactant.TracedType, step, init, columns, shared, n)
    n == 0 && return init
    carry = _recurrence_trace(init)
    data = _recurrence_trace(columns)
    args = _recurrence_trace(shared)
    Reactant.@trace for i in 1:n
        row = Reactant.@allowscalar map(c -> c[i], data)
        carry = _recurrence_trace(step(carry, row, args...))
    end
    carry
end

function _recurrence_branch(
        pred::Reactant.TracedRNumber{Bool}, yes, no, args)
    Reactant.@trace if pred
        result = yes(args...)
    else
        result = no(args...)
    end
    result
end

@inline function _pk_propagate_positive(A, state, dt)
    M = (A[1] * dt, A[2] * dt, A[3] * dt, A[4] * dt, A[5] * dt,
        A[6] * dt, A[7] * dt, A[8] * dt, A[9] * dt)
    _pk_tmatvec3(_pk_expm3(M), state)
end

@inline function _pk_dose_affine(A, amount, interval)
    M = (A[1] * interval, A[2] * interval, A[3] * interval,
        A[4] * interval, A[5] * interval, A[6] * interval,
        A[7] * interval, A[8] * interval, A[9] * interval)
    P = _pk_expm3(M)
    (P[1], P[2], P[3], 0.0, P[4], P[5], P[6], 0.0,
        P[7], P[8], P[9], 0.0, amount, 0.0, 0.0, 1.0)
end

@inline function _pk_apply_affine(Q, after_first)
    return (Q[1] * after_first[1] + Q[5] * after_first[2] +
            Q[9] * after_first[3] + Q[13],
        Q[2] * after_first[1] + Q[6] * after_first[2] +
            Q[10] * after_first[3] + Q[14],
        Q[3] * after_first[1] + Q[7] * after_first[2] +
            Q[11] * after_first[3] + Q[15])
end# Rectangular backend adapter for grouped PK cells. The host schedule becomes
# fixed-length columns; subject boundaries and ragged read ranges are data.
@inline _subject_value(a) = a
@inline _subject_value(a::Union{SubjectSlice,SubjectScalar}) = a.v
@inline _subject_mode(a) = Val(:shared)
@inline _subject_mode(a::SubjectSlice) = Val(:slice)
@inline _subject_mode(a::SubjectScalar) = Val(:subject)
@inline _pk_row_arg(::Val{:shared}, a, s, j) = a
@inline _pk_row_arg(::Val{:slice}, a, s, j) = _traced_op_read(a, j)
@inline _pk_row_arg(::Val{:subject}, a, s, j) = _traced_op_read(a, s)
_pk_has_auc(::typeof(linear_pk_read_locs)) = false
_pk_has_auc(::typeof(linear_pk_read_locs_auc)) = true

function _pk_rectangular(cell, ends, opcols, args, marker)
    n = length(first(opcols))
    isempty(ends) && throw(ArgumentError("subject-batched cell needs a subject"))
    all(c -> length(c) == n, opcols) || throw(DimensionMismatch("PK op columns"))
    ends[end] == n || throw(DimensionMismatch("PK subject ends and op columns"))
    # Normalize the v1 no-bioavailability spelling before building the step.
    fullargs = length(args) == 5 ? (SubjectSlice(zeros(n)), args...) : args
    length(fullargs) == 6 || throw(ArgumentError("PK recurrence expects six parameters"))
    auc = _pk_has_auc(cell)
    subject = zeros(Int, n)
    reset = zeros(Bool, n)
    conc_index = zeros(Int, n)
    auc_index = zeros(Int, n)
    prev = 0
    offset = 0
    for s in eachindex(ends)
        hi = ends[s]
        hi > prev || throw(ArgumentError("each PK subject needs at least one op"))
        reads = count(==(LINEAR_EVENT_READ), view(opcols[1], prev+1:hi))
        reset[prev+1] = true
        r = 0
        for j in prev+1:hi
            kind = opcols[1][j]
            kind in (LINEAR_EVENT_READ, LINEAR_EVENT_DOSE, LINEAR_EVENT_DOSE_SEGMENT) ||
                throw(ArgumentError("unknown linear PK operation type $kind"))
            subject[j] = s
            if kind == LINEAR_EVENT_READ
                r += 1
                conc_index[j] = offset + r
                auc_index[j] = offset + reads + r
            end
        end
        offset += (auc ? 2 : 1) * reads
        prev = hi
    end
    # The power recurrence has a static capacity, with inactive iterations
    # frozen after each individual exponent is exhausted. This preserves the
    # original binary-power arithmetic for unequal segment dose counts.
    maxcount = maximum(opcols[5])
    bits = ndigits(max(maxcount - 1, 0); base=2)
    modes = map(_subject_mode, fullargs)
    step = _PKRectangularStep{auc,typeof(modes)}(modes)
    columns = (collect(1:n), subject, reset, conc_index, auc_index, opcols[1:5]...)
    init = (state=(0.0, 0.0, 0.0), given=0.0, out=zeros(offset))
    shared = (map(_subject_value, fullargs), zeros(Int, bits))
    _rectangular_fold(step, init, columns, shared, marker).out
end

struct _PKRectangularStep{AUC,M}
    modes::M
end

function (step::_PKRectangularStep{AUC})(carry, row, args, powcols) where {AUC}
    j, s, reset, ci, ai, kind, dt, amount, interval, count = row
    log_F, log_Vc, log_k10, log_k12, log_k21, log_ka =
        map((mode, a) -> _pk_row_arg(mode, a, s, j), step.modes, args)
    log_CL = log_Vc + log_k10
    A = linear_pk_system_3(log_CL, log_Vc, log_Vc + log_k12,
        log_Vc + log_k12 - log_k21, log_ka)
    state = map(x -> ifelse(reset, zero(x), x), carry.state)
    given = ifelse(reset, zero(carry.given), carry.given)
    state = _recurrence_branch(dt > 0,
        _pk_propagate_positive, (A, state, dt) -> state, (A, state, dt))
    context = (state, given, carry.out, ci, ai, exp(log_Vc), exp(log_CL),
        A, amount, log_F, interval, ifelse(kind == LINEAR_EVENT_DOSE, 1, count), powcols)
    _recurrence_branch(kind == LINEAR_EVENT_READ,
        _PKRectangularRead{AUC}(), _PKRectangularDose(), context)
end

struct _PKRectangularRead{AUC} end
function (::_PKRectangularRead{AUC})(state, given, out, ci, ai, Vc, CL,
        A, amount, log_F, interval, count, powcols) where {AUC}
    out = ReactiveKernels._tensorized_setindex(out, state[2] / Vc, ci)
    if AUC
        out = ReactiveKernels._tensorized_setindex(out,
            (given - (state[1] + state[2] + state[3])) / CL, ai)
    end
    (; state, given, out)
end

struct _PKRectangularDose end
function (::_PKRectangularDose)(state, given, out, ci, ai, Vc, CL,
        A, amount, log_F, interval, count, powcols)
    effective = amount * exp(log_F)
    first = linear_pk_add_dose_3(state, effective)
    state = _recurrence_branch(count > 1,
        _pk_rectangular_regular, (A, first, effective, interval, count, cols) -> first,
        (A, first, effective, interval, count, powcols))
    (; state, given=given + count * effective, out)
end

function _pk_rectangular_regular(A, first, amount, interval, count, cols)
    B = _pk_dose_affine(A, amount, interval)
    R = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    init = (; R, B, e=count - 1)
    step = (c, row) -> _recurrence_branch(c.e > 0,
        _pk_power_step, identity, (c,))
    power = _rectangular_fold(step, init, (cols,), (), amount)
    _pk_apply_affine(power.R, first)
end

function _pk_power_step(c)
    R = _recurrence_branch((c.e & 1) == 1,
        _pk_tmul4, (R, B) -> R, (c.R, c.B))
    (; R, B=_pk_tmul4(c.B, c.B), e=div(c.e, 2))
end

const schedule = build_linear_pk_schedule([1, 1, 2, 2], [96., 120., 0., 5.],
    [1, 1, 1, 1, 2], [0., 24., 48., 72., 0.], [100., 100., 100., 100., 50.])
const columns = (schedule.op_type, schedule.op_dt, schedule.op_amount,
    schedule.op_interval, schedule.op_count, schedule.op_read_idx)
function pk(q)
    args = (SubjectSlice(zeros(length(schedule.op_type))),
        ntuple(i -> SubjectScalar(ReactiveKernels._tensorized_getindex(q, [2i-1, 2i])), 5)...)
    sum(_pk_rectangular(linear_pk_read_locs_auc, schedule.op_ends, columns, args, q))
end
function native_pk(q)
    args = (SubjectSlice(zeros(length(schedule.op_type))),
        ntuple(i -> SubjectScalar(q[[2i-1, 2i]]), 5)...)
    sum(PPL.linear_pk_read_locs_auc_over_subjects(schedule.op_ends, columns..., args...))
end
q = repeat(log.([10., .1, .2, .3, .5]); inner=2)
@assert pk(q) ≈ native_pk(q)
expected = only(Enzyme.gradient(Enzyme.Reverse, native_pk, q))
println("native value: ", native_pk(q), "; native gradient: ", expected)
flush(stdout)
gradient(q) = only(Enzyme.gradient(Enzyme.Reverse, pk, q))
rq = Reactant.to_rarray(q)
compiled = Reactant.@compile sync=true gradient(rq)
@assert Array(compiled(rq)) ≈ expected rtol=1e-8
