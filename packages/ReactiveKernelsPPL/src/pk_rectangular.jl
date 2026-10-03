# Schedule-only rectangular storage. No parameter-dependent work runs here.
# Zero indices are inactive slots, guarded lazily by the event recurrence.
function _pk_subject_plan(ends, kinds; auc=false)
    previous = 0
    lengths = Int[]
    reads = Int[]
    for hi in ends
        previous <= hi <= length(kinds) || throw(ArgumentError("invalid PK subject ends"))
        push!(lengths, hi - previous)
        push!(reads, count(==(LINEAR_EVENT_READ), view(kinds, previous+1:hi)))
        previous = hi
    end
    previous == length(kinds) || throw(DimensionMismatch("PK subject ends and operations"))
    width = maximum(lengths; init=0)
    capacity = maximum(reads; init=0)
    indices = zeros(Int, width, length(ends))
    slots = ones(Int, capacity, length(ends))
    packing = Int[]
    previous = 0
    for s in eachindex(ends)
        hi = ends[s]
        r = 0
        for j in previous+1:hi
            kind = kinds[j]
            kind in (LINEAR_EVENT_READ, LINEAR_EVENT_DOSE, LINEAR_EVENT_DOSE_SEGMENT) ||
                throw(ArgumentError("unknown linear PK operation type $kind"))
            indices[j - previous, s] = j
            if kind == LINEAR_EVENT_READ
                r += 1
                slots[r, s] = j - previous
            end
        end
        offset = (s - 1) * capacity * (auc ? 2 : 1)
        append!(packing, offset .+ (1:r))
        auc && append!(packing, offset .+ capacity .+ (1:r))
        previous = hi
    end
    (; indices, slots, packing, capacity)
end

# The scalar-output scan owns its compartment and accumulated-dose carry.
# Non-read steps emit zero; schedule-only read positions gather its buffer.
@kernel _pk_event_values(indices, op_type, op_dt, op_amount,
        op_interval, op_count, log_F, log_Vc, log_k10, log_k12, log_k21,
        log_ka, exposure) = begin
    A = linear_pk_system_3(log_Vc + log_k10, log_Vc, log_Vc + log_k12,
        log_Vc + log_k12 - log_k21, log_ka)
    Vc = exp(log_Vc)
    CL = exp(log_Vc + log_k10)
    seed = (state = SVector(0.0, 0.0, 0.0), given = zero(Vc))
    values = scan(indices, Ref(A), Ref(Vc), Ref(CL), Ref(op_type), Ref(op_dt),
            Ref(op_amount), Ref(op_interval), Ref(op_count), Ref(log_F),
            Ref(exposure); init=seed) do carry, j, system, volume, clearance,
                kinds, dts, amounts, intervals, counts, bioavailability, auc
        advanced = if j > 0
            state = linear_pk_propagate_3(system, carry.state, dts[j])
            given = carry.given
            result = if kinds[j] == LINEAR_EVENT_READ
                value = if auc
                    (given - sum(state)) / clearance
                else
                    state[2] / volume
                end
                ((state=state, given=given), value)
            else
                effective = amounts[j] * exp(bioavailability[j])
                updated = if kinds[j] == LINEAR_EVENT_DOSE
                    (state=linear_pk_add_dose_3(state, effective),
                        given=given + effective)
                else
                    (state=linear_pk_add_regular_doses_3(system, state,
                        effective, intervals[j], counts[j]),
                        given=given + counts[j] * effective)
                end
                (updated, zero(volume))
            end
            result
        else
            (carry, zero(volume))
        end
        (advanced[1], advanced[2])
    end
    return values
end

# Both outputs use the same step graph. Two scalar scans fit the current
# compiled scan contract; their fixed buffers are gathered before packing.
@kernel _pk_event_scan(indices, slots, capacity, op_type, op_dt, op_amount,
        op_interval, op_count, log_F, log_Vc, log_k10, log_k12, log_k21,
        log_ka, auc) = begin
    conc_mode = false
    conc = _pk_event_values(indices, op_type, op_dt, op_amount, op_interval,
        op_count, log_F, log_Vc, log_k10, log_k12, log_k21, log_ka, conc_mode)
    reads = conc[slots]
    return reads
end
@kernel _pk_event_scan_auc(indices, slots, capacity, op_type, op_dt, op_amount,
        op_interval, op_count, log_F, log_Vc, log_k10, log_k12, log_k21,
        log_ka, auc) = begin
    conc_mode = false
    auc_mode = true
    conc = _pk_event_values(indices, op_type, op_dt, op_amount, op_interval,
        op_count, log_F, log_Vc, log_k10, log_k12, log_k21, log_ka, conc_mode)
    exposure = _pk_event_values(indices, op_type, op_dt, op_amount,
        op_interval, op_count, log_F, log_Vc, log_k10, log_k12, log_k21,
        log_ka, auc_mode)
    reads = vcat(conc[slots], exposure[slots])
    return reads
end

# Emit the same graph for generated models and the direct grouped-cell API.
# Argument axes are positional facts: the event vector stays whole and each
# event gathers its entry; subject values are plate operands; shared values
# are Ref operands.
function _pk_subject_statements(nm, auc, ends, cols, values, modes)
    table = gensym(:pk_schedule)
    indices, slots, packing, capacity, lanes =
        (gensym(n) for n in (:pk_indices, :pk_slots, :pk_packing, :pk_capacity, :pk_lanes))
    formals = [gensym(:pk_parameter) for _ in values]
    parameters = Any[m === :subject ? a : :(Ref($a)) for (a, m) in zip(values, modes)]
    colformals = [gensym(:pk_column) for _ in cols]
    ix, rs, cap = gensym(:pk_events), gensym(:pk_read_slots), gensym(:pk_capacity)
    af = gensym(:pk_auc)
    callargs = Any[ix, rs, cap, colformals[1:5]..., formals..., af]
    subject = gensym(:pk_subject)
    cell = Expr(:block)
    for i in eachindex(modes)
        if modes[i] === :marked
            localparam = gensym(:pk_local_parameter)
            push!(cell.args, :($localparam = _pk_subject_parameter($(formals[i]), $subject)))
            callargs[8 + i] = localparam
        end
    end
    callee = auc ? :_pk_event_scan_auc : :_pk_event_scan
    push!(cell.args, :($callee($(callargs...))))
    bodyformals = Any[ix, rs, cap, colformals..., formals..., af, subject]
    operands = Any[:(eachcol($indices)), :(eachcol($slots)), :(Ref($capacity)),
        [:(Ref($c)) for c in cols]..., parameters..., :(Ref($auc)), :(eachindex($ends))]
    mapped = Expr(:do, Expr(:call, :plate, operands...),
        Expr(:->, Expr(:tuple, bodyformals...), cell))
    Expr[
        :($table = _pk_subject_plan($ends, $(first(cols)); auc=$auc)),
        :($indices = $table.indices), :($slots = $table.slots),
        :($packing = $table.packing), :($capacity = $table.capacity),
        :($lanes = $mapped),
        :($nm = _pk_pack_subjects($lanes, $packing)),
    ]
end

_pk_pack_subjects(lanes, packing) = isempty(packing) ? Float64[] : vec(stack(lanes))[packing]
_pk_pack_subjects(lanes::ReactiveKernels._TensorizedPlateBatch, packing) =
    vec(permutedims(ReactiveKernels._tensorized_plate_materialize(lanes)))[packing]
@inline _pk_subject_parameter(a, s) = a
@inline _pk_subject_parameter(a::SubjectScalar, s) = _traced_op_read(a.v, s)

@inline _subject_value(a) = a
@inline _subject_value(a::Union{SubjectSlice,SubjectScalar}) = a.v
@inline _subject_mode(a) = :shared
@inline _subject_mode(a::SubjectSlice) = :event
@inline _subject_mode(a::SubjectScalar) = :subject

function _pk_direct_spec(auc)
    columns = [:op_type, :op_dt, :op_amount, :op_interval, :op_count, :op_read_idx]
    parameters = [:log_F, :log_Vc, :log_k10, :log_k12, :log_k21, :log_ka]
    modes = (:event, :marked, :marked, :marked, :marked, :marked)
    body = Expr(:block,
        _pk_subject_statements(:reads, auc, :ends, columns, parameters, modes)...,
        :(return reads))
    signature = [(n, Any) for n in [:ends; columns; parameters]]
    Core.eval(@__MODULE__, ReactiveKernels._kernel_expand(
        body, signature, nothing, @__MODULE__))
end
const _pk_conc_spec = _pk_direct_spec(false)
const _pk_auc_spec = _pk_direct_spec(true)
const _pk_direct_conc = prepare(_pk_conc_spec)
const _pk_direct_auc = prepare(_pk_auc_spec)
_pk_direct_kernel(::typeof(linear_pk_read_locs)) = _pk_direct_conc
_pk_direct_kernel(::typeof(linear_pk_read_locs_auc)) = _pk_direct_auc

function _pk_subject_call(cell, ends, cols, args)
    n = length(first(cols))
    all(c -> length(c) == n, cols) || throw(DimensionMismatch("PK op columns"))
    fullargs = length(args) == 5 ? (SubjectSlice(zeros(n)), args...) : args
    length(fullargs) == 6 || throw(ArgumentError("PK recurrence expects six parameters"))
    modes = map(_subject_mode, fullargs)
    first(modes) === :event || throw(ArgumentError("PK log_F requires an explicit event axis"))
    values = map(_subject_value, fullargs)
    length(first(values)) == n || throw(DimensionMismatch("PK log_F and operations"))
    kernel = _pk_direct_kernel(cell)
    kernel(ends, cols..., first(values), fullargs[2:end]...)
end

# Compatibility for the old experimental diagnostic entry point. It uses the
# ordinary plate graph, with no parameter-dependent host propagator table.
_pk_rectangular(cell, ends, cols, args, marker) = _pk_subject_call(cell, ends, cols, args)
