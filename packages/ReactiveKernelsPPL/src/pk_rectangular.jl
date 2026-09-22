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
    ReactiveKernels._rectangular_fold(step, init, columns, shared, marker).out
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
    state = ReactiveKernels._recurrence_branch(dt > 0,
        _pk_propagate_positive, (A, state, dt) -> state, (A, state, dt))
    context = (state, given, carry.out, ci, ai, exp(log_Vc), exp(log_CL),
        A, amount, log_F, interval, ifelse(kind == LINEAR_EVENT_DOSE, 1, count), powcols)
    ReactiveKernels._recurrence_branch(kind == LINEAR_EVENT_READ,
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
    state = ReactiveKernels._recurrence_branch(count > 1,
        _pk_rectangular_regular, (A, first, effective, interval, count, cols) -> first,
        (A, first, effective, interval, count, powcols))
    (; state, given=given + count * effective, out)
end

function _pk_rectangular_regular(A, first, amount, interval, count, cols)
    B = _pk_dose_affine(A, amount, interval)
    Q = _pk_retained_power(B, count - 1, cols, amount)
    _pk_apply_affine(Q, first)
end

# Shared by standalone matrix powers and the grouped recurrence. The exponent
# and matrix are loop-carried data; the bit capacity only sizes the row table.
function _pk_retained_power(B, exponent, cols, marker)
    R = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    init = (; R, B, e=exponent)
    step = (c, row) -> ReactiveKernels._recurrence_branch(c.e > 0,
        _pk_power_step, identity, (c,))
    ReactiveKernels._rectangular_fold(step, init, (cols,), (), marker).R
end

function _pk_power_step(c)
    R = ReactiveKernels._recurrence_branch((c.e & 1) == 1,
        _pk_tmul4, (R, B) -> R, (c.R, c.B))
    (; R, B=_pk_tmul4(c.B, c.B), e=div(c.e, 2))
end
