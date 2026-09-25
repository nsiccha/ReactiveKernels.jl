# Rectangular backend adapter for grouped PK cells. The host schedule becomes
# fixed-length columns; subject boundaries and ragged read ranges are data.
@inline _subject_value(a) = a
@inline _subject_value(a::Union{SubjectSlice,SubjectScalar}) = a.v
@inline _subject_mode(a) = Val(:shared)
@inline _subject_mode(a::SubjectSlice) = Val(:slice)
@inline _subject_mode(a::SubjectScalar) = Val(:subject)
@inline _pk_row_arg(::Val{:shared}, a, s, j) = a
@inline _pk_row_arg(::Val{:slice}, a, s, j) = _traced_op_read(a, j)
@inline _pk_row_arg(::Val{:subject}, a, s, j) = _traced_op_read(a, s)

# exp(A*t) for the given rows, as nine column vectors (entry k of every row's
# column-major 3x3 result); `nothing` when no row needs one. Rows come from
# bound schedule data, so only rows whose lazy branch is taken are evaluated.
# The table builds outside the sequential recurrence (the exponential does not
# depend on the state) with the StaticArrays built-in, one row at a time.
function _pk_expm_table(modes, values, subject, rows, t)
    isempty(rows) && return nothing
    outs = [Any[] for _ in 1:9]
    for j in rows
        _, log_Vc, log_k10, log_k12, log_k21, log_ka =
            map((mode, a) -> _pk_row_arg(mode, a, subject[j], j), modes,
                values)
        A = linear_pk_system_3(log_Vc + log_k10, log_Vc, log_Vc + log_k12,
            log_Vc + log_k12 - log_k21, log_ka)
        P = exp(A * t[j])
        for e in 1:9
            push!(outs[e], P[e])
        end
    end
    return outs
end
@inline _pk_table_smat(P, i) = SMatrix{3,3}(
    _traced_op_read(P[1], i), _traced_op_read(P[2], i),
    _traced_op_read(P[3], i), _traced_op_read(P[4], i),
    _traced_op_read(P[5], i), _traced_op_read(P[6], i),
    _traced_op_read(P[7], i), _traced_op_read(P[8], i),
    _traced_op_read(P[9], i))
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
    values = map(_subject_value, fullargs)
    kinds, dts, intervals, counts = opcols[1], opcols[2], opcols[4], opcols[5]
    prop_rows = [j for j in 1:n if dts[j] > 0]
    segment_rows = [j for j in 1:n if kinds[j] == LINEAR_EVENT_DOSE_SEGMENT && counts[j] > 1]
    prop_index = zeros(Int, n)
    prop_index[prop_rows] = eachindex(prop_rows)
    segment_index = zeros(Int, n)
    segment_index[segment_rows] = eachindex(segment_rows)
    prop = _pk_expm_table(modes, values, subject, prop_rows, dts)
    segment = _pk_expm_table(modes, values, subject, segment_rows, intervals)
    columns = (collect(1:n), subject, reset, conc_index, auc_index, opcols[1:5]...,
        prop_index, segment_index)
    init = (state=SVector(0.0, 0.0, 0.0), given=0.0, out=zeros(offset))
    shared = (values, zeros(Int, bits), prop, segment)
    ReactiveKernels._rectangular_fold(step, init, columns, shared, marker).out
end

struct _PKRectangularStep{AUC,M}
    modes::M
end

function (step::_PKRectangularStep{AUC})(carry, row, args, powcols, prop,
        segment) where {AUC}
    j, s, reset, ci, ai, kind, dt, amount, interval, count, pidx, sidx = row
    log_F, log_Vc, log_k10 = map((mode, a) -> _pk_row_arg(mode, a, s, j),
        step.modes[1:3], args[1:3])
    log_CL = log_Vc + log_k10
    state = map(x -> ifelse(reset, zero(x), x), carry.state)
    given = ifelse(reset, zero(carry.given), carry.given)
    state = ReactiveKernels._recurrence_branch(dt > 0,
        _pk_propagate_row, (P, i, state) -> state, (prop, pidx, state))
    context = (state, given, carry.out, ci, ai, exp(log_Vc), exp(log_CL),
        segment, sidx, amount, log_F, ifelse(kind == LINEAR_EVENT_DOSE, 1, count), powcols)
    ReactiveKernels._recurrence_branch(kind == LINEAR_EVENT_READ,
        _PKRectangularRead{AUC}(), _PKRectangularDose(), context)
end

# The hoisted exp(A*dt) of this row applied to the state. Without a table no
# row has dt > 0, so the branch calling this is never taken.
_pk_propagate_row(P, i, state) = _pk_table_smat(P, i) * state
_pk_propagate_row(::Nothing, i, state) = state

struct _PKRectangularRead{AUC} end
function (::_PKRectangularRead{AUC})(state, given, out, ci, ai, Vc, CL,
        segment, si, amount, log_F, count, powcols) where {AUC}
    out = ReactiveKernels._tensorized_setindex(out, state[2] / Vc, ci)
    if AUC
        out = ReactiveKernels._tensorized_setindex(out,
            (given - (state[1] + state[2] + state[3])) / CL, ai)
    end
    (; state, given, out)
end

struct _PKRectangularDose end
function (::_PKRectangularDose)(state, given, out, ci, ai, Vc, CL,
        segment, si, amount, log_F, count, powcols)
    effective = amount * exp(log_F)
    first = linear_pk_add_dose_3(state, effective)
    state = ReactiveKernels._recurrence_branch(count > 1,
        _pk_rectangular_regular, (P, i, first, effective, count, cols) -> first,
        (segment, si, first, effective, count, powcols))
    (; state, given=given + count * effective, out)
end

# `count` doses at a fixed interval from the hoisted exp(A*interval) of this
# segment row. Without a table no segment repeats, so this is never taken.
function _pk_rectangular_regular(P, i, first, amount, count, cols)
    S = _pk_table_smat(P, i)
    B = SMatrix{4,4}(S[1], S[2], S[3], 0.0, S[4], S[5], S[6], 0.0,
        S[7], S[8], S[9], 0.0, amount, 0.0, 0.0, 1.0)
    Q = _pk_retained_power(B, count - 1, cols, amount)
    q = Q * SVector(first[1], first[2], first[3], 1.0)
    return SVector(q[1], q[2], q[3])
end
_pk_rectangular_regular(::Nothing, i, first, amount, count, cols) = first

# Shared by standalone matrix powers and the grouped recurrence. The exponent
# and matrix are loop-carried data; the bit capacity only sizes the row table.
function _pk_retained_power(B, exponent, cols, marker)
    R = SMatrix{4,4}(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    init = (; R, B, e=exponent)
    step = (c, row) -> ReactiveKernels._recurrence_branch(c.e > 0,
        _pk_power_step, identity, (c,))
    ReactiveKernels._rectangular_fold(step, init, (cols,), (), marker).R
end

function _pk_power_step(c)
    R = ReactiveKernels._recurrence_branch((c.e & 1) == 1,
        (R, B) -> R * B, (R, B) -> R, (c.R, c.B))
    (; R, B=c.B * c.B, e=div(c.e, 2))
end
