# Linear-PK recurrence cell vocabulary (grouped kernels).
#
# The SB joint PK/QT/TGI `linear_pk_read_locs_cell` mirror (bruno mirror ref
# `kb-impl/Bruno-arv393-tgi` @ `3af22846`: `web-pkpd/src/brm_integration.jl`
# `linear_pk_read_locs_cell`, `src/pkpd_models.jl` `linear_pk_system` /
# `propagate` / `add_dose` / `add_regular_doses`, `src/dosing_schedule.jl`
# `build_linear_event_schedule`) plus the event-schedule recipe (D5a).
#
# Two deliberate deltas from SB, both documented at the use site:
# 1. Read numbering is SUBJECT-LOCAL (1..R_s per subject; SB numbers reads
#    globally across subjects). Op columns are bind-materialized and never
#    cross to SB, so the numbering is thin-layer-internal; local numbering
#    avoids the O(global reads) per-subject buffer SB's `max(op_read_idx)`
#    sizing implies.
# 2. The v1 `linear_pk_read_locs` spelling (no `log_F` argument) keeps
#    bioavailability identically 1 (the dose branch is `op_amount[j]`
#    unscaled — the v1 parity fixture models F = 1). The V2 event-LP
#    seam is the 7-op-column spelling: per-op `log_F` values thread the
#    dose branch (`op_amount[j] * exp(log_F[j])`, SB's
#    `linear_pk_read_locs_cell` dose branch verbatim), evaluated by
#    `linear_pk_event_log_f` over the `linear_pk_op_log_dose` event
#    axis. One recurrence loop serves both (the 6-column method is a
#    thin zeros wrapper — dev §4, bit-identical `exp(0) == 1`).
# Everything else (op codes, token order, read-before-dose, segment
# compression, recurrence op order) mirrors SB exactly.
#
# The recurrence runs as PLAIN JULIA called from generated code (the
# `gp_chol_latent` native+Enzyme precedent): the generator emits one
# `linear_pk_read_locs` call per subject over bound op-column slices plus
# traced LP scalars, then a vectorized `flat[obs_map]` gather
# (`reactivekernels-use` §7d shape) and the Gaussian plate. All
# intermediate 3- and 4-dimensional storage is FUNCTIONAL (length-typed
# tuples, never mutated): a concrete `Matrix{Float64}`/`Vector{Float64}`
# rejects traced stores (`convert(Float64, ::TracedRNumber)` has no
# method — the pkcell slice's measured Reactant failure). The matrix
# exponential is a faithful port of Stan's `matrix_exp_pade` (Pade
# fraction + scaling-and-squaring, predicated selection, fixed trip
# counts) because Enzyme cannot reverse `LinearAlgebra.exp`
# (`EnzymeNoDerivativeError` in its internals, measured 2026-09-20;
# plain `Matrix * Matrix` reverses fine — an earlier note here claimed
# otherwise). The 3x3/4x4 kernels are scalar tuple code, which stays in
# registers natively. Only the returned reads vector is an array,
# allocated eltype-generic (see `linear_pk_read_locs`).
# The accuracy envelope (`l1norm < 5499`, exact-Stan inside) is
# documented on `_pk_expm3`.

"""Operation codes of the linear-PK event stream (SB `dosing_schedule.jl`)."""
const LINEAR_EVENT_READ = 1
const LINEAR_EVENT_DOSE = 2
const LINEAR_EVENT_DOSE_SEGMENT = 3

# Per-op reads of the `log_F` slice inside the recurrence loop.  `log_F` is
# parameter-dependent (`linear_pk_event_log_f`, an HSGP over the event axis),
# so under Reactant it arrives as a `TracedRArray` — or a `view` of one — and
# a plain `log_F[j]` is the scalar read the tracer refuses (`Scalar indexing
# is disallowed`, measured on Reactant 0.2.285 compiling the joint fixture;
# `test_reactant_joint.jl` pins the fix).  The cell is opaque Julia called
# from generated code, so RK's `@kernel` tensorized rewrite cannot reach this
# read and no Reactant-extension method can intercept `getindex` on a Base
# `SubArray` without piracy; the read itself routes through RK's
# tensorized-gather hook instead.  Natively that hook IS `Base.getindex`
# (bit-identical, zero overhead); the RK Reactant extension lowers the
# concrete-index read on a traced vector/view to a 1-element slice.  The op
# columns (`op_type`, `op_dt`, `op_amount`, …) are bound data and keep plain
# `getindex`.
const _traced_op_read = ReactiveKernels._tensorized_getindex

# --- subject-batched cell runner --------------------------------------------
#
# The generator emits ONE statement per grouped cell assignment, whatever
# the subject count: `<cell>_over_subjects(op_ends, opcols..., args...)`
# loops over subjects at RUNTIME with `view(col, lo:hi)` slices from the
# bound `op_ends` (the statement count of the generated program is O(1)
# in the data; only layout, bound data and loop trip counts scale).  Each
# extra argument carries its per-subject access mode explicitly:
#
#   `SubjectSlice(v)`  — `view(v, lo:hi)` over the subject's op range (the
#                        flat event-frame vectors such as the W2 `log_F`
#                        provider output);
#   `SubjectScalar(v)` — the subject's entry `v[s]` (per-subject LP
#                        vectors), read through `_traced_op_read` so a
#                        traced LP vector lowers as a 1-element slice;
#   anything else      — passed verbatim to every subject (model scalars,
#                        literals).
#
# Natively this is exactly the former per-subject unroll (same slices,
# same scalars, `reduce(vcat, parts)` == the former `vcat(s1, s2, …)`);
# the experimental Reactant path in pk_rectangular.jl promotes bound columns
# to traced constants and combines subject and operation traversal in one
# fixed-trip loop. It requires an explicit benchmark opt-in below.
"""
    SubjectSlice(v)

Marks a flat op-ordered vector argument of a subject-batched cell call:
subject `s` receives `view(v, lo:hi)` over its op range (see
[`linear_pk_read_locs_auc_over_subjects`](@ref)).
"""
struct SubjectSlice{V}
    v::V
end

"""
    SubjectScalar(v)

Marks a per-subject vector argument of a subject-batched cell call:
subject `s` receives the scalar `v[s]` (see
[`linear_pk_read_locs_auc_over_subjects`](@ref)).
"""
struct SubjectScalar{V}
    v::V
end

@inline _subject_arg(a::SubjectSlice, rng, s) = view(a.v, rng)
@inline _subject_arg(a::SubjectScalar, rng, s) = _traced_op_read(a.v, s)
@inline _subject_arg(a, rng, s) = a
@inline _subject_args(args::Tuple, rng, s) =
    map(a -> _subject_arg(a, rng, s), args)
@inline _subject_views(cols::Tuple, rng) = map(c -> view(c, rng), cols)

# Experimental until Enzyme/MLIR can reverse the PK recurrence. The benchmark
# opts in explicitly; compiled callers fail explicitly while it is disabled.
# Native callers retain ordinary subject and operation iteration.
const _rectangular_pk_enabled = Ref(false)

function _pk_compiled_cell(cell, ends, opcols, args, marker)
    _rectangular_pk_enabled[] || throw(ArgumentError(
        "compiled PK recurrences are disabled pending Reactant reverse control-flow " *
        "support (https://github.com/nsiccha/ReactiveKernels.jl/issues/13); " *
        "data-derived loop unrolling is not a supported fallback"))
    ReactiveKernels._dynamic_tensorized_marker(opcols) === nothing ||
        throw(ArgumentError("rectangular PK requires bound operation columns"))
    _pk_rectangular(cell, ends, opcols, args, marker)
end

function _cell_over_subjects(cell::F, op_ends::AbstractVector{<:Integer},
        opcols::Tuple, args::Tuple) where {F}
    marker = ReactiveKernels._dynamic_tensorized_marker((opcols..., map(_subject_value, args)...))
    marker === nothing || return _pk_compiled_cell(cell, op_ends, opcols, args, marker)
    n_sub = length(op_ends)
    n_sub >= 1 || throw(ArgumentError(
        "subject-batched cell call needs at least one subject (empty op_ends)"))
    hi = Int(op_ends[1])
    rng = 1:hi
    first = cell(_subject_views(opcols, rng)..., _subject_args(args, rng, 1)...)
    parts = [first]
    prev = hi
    for s in 2:n_sub
        hi = Int(op_ends[s])
        rng = (prev + 1):hi
        push!(parts, cell(_subject_views(opcols, rng)...,
            _subject_args(args, rng, s)...))
        prev = hi
    end
    return reduce(vcat, parts)
end

"""
    linear_pk_read_locs_over_subjects(op_ends, op_type, op_dt, op_amount,
        op_interval, op_count, op_read_idx, args...)
    linear_pk_read_locs_auc_over_subjects(op_ends, op_type, op_dt, op_amount,
        op_interval, op_count, op_read_idx, args...)

Run [`linear_pk_read_locs`](@ref) / [`linear_pk_read_locs_auc`](@ref) once
per subject over the bound op columns and concatenate the per-subject
results in subject order: subject `s` sees the op range
`op_ends[s-1]+1:op_ends[s]` (`view`s of the six op columns), and each
extra argument by its marker — [`SubjectSlice`](@ref) (a `view` over the
same range), [`SubjectScalar`](@ref) (entry `s`), or verbatim.  This is
the generated-code spelling of a grouped cell assignment (one statement
per assignment, independent of the subject count); the result equals
`vcat` of the per-subject cell calls.
"""
linear_pk_read_locs_over_subjects(op_ends::AbstractVector{<:Integer},
        op_type, op_dt, op_amount, op_interval, op_count, op_read_idx,
        args...) =
    _cell_over_subjects(linear_pk_read_locs, op_ends,
        (op_type, op_dt, op_amount, op_interval, op_count, op_read_idx), args)
linear_pk_read_locs_auc_over_subjects(op_ends::AbstractVector{<:Integer},
        op_type, op_dt, op_amount, op_interval, op_count, op_read_idx,
        args...) =
    _cell_over_subjects(linear_pk_read_locs_auc, op_ends,
        (op_type, op_dt, op_amount, op_interval, op_count, op_read_idx), args)

_pk_sched_fail(msg) = throw(ContractValidationError("[schedule] " * msg))

"""
    build_linear_pk_schedule(obs_subj, obs_time, dose_subj, dose_time, dose_amt;
                             min_segment_count = 3, combine_simultaneous = true,
                             ecg = nothing, tgi = nothing)

Parameter-independent per-subject operation stream for the linear-PK
recurrence, from raw observation + dose columns (D5a bind-time recipe —
the spline fit/apply precedent: sorting, grouping, and segment
compression are inexpressible in the kernel graph, so the schedule
builds on the host at bind and materializes op COLUMNS as bound
vectors).

Verbatim transliteration of Bruno `build_linear_event_schedule`
(`src/dosing_schedule.jl`), minus the dose-pattern axis (every dose is
pattern 1 — the V2 joint has no formulation/meal fields, and SB rejects
them in the log-F formula) and with SUBJECT-LOCAL read numbering (see
the file header). Same validations, same token order (read-before-dose
at equal time), same exact-match segment compression. SB's
`combine_simultaneous` flag is honored verbatim: `true` (default)
pre-sums same-time doses (exact only at state-independent F ≡ 1 — the
v1 spelling); `false` keeps separate same-time dose rows as separate
zero-delta jumps, which the nonlinear V2 event-LP requires (SB builds
the joint schedule that way). The event-LP bind path selects `false`
automatically; the flag is explicit here for direct builder users.

`ecg`/`tgi` are optional `(subject, time)` column pairs naming EXTRA
READ AXES (the joint model's ECG + tumor assessment rows): their times
join the stream as read-only points, so the recurrence evaluates
concentration/AUC at every endpoint's times (SB's joint prep
`vcat(pk, qt, tgi)` subject/time concatenation,
`brm_integration.jl`, sliced back per axis below — the same
concatenation order: obs, ecg, tgi). A subject with no PK or dose rows
still gets its read-only stream (SB's dose-free subject precedent);
subject coverage and the 1:n contiguity rule apply to the UNION axis.
Reads stay subject-local; `obs_read`/`obs_map` keep their v1 meaning
over the primary (PK) rows only, so builds without extra axes are
byte-identical to v1.

Returns a NamedTuple with flat subject-blocked op columns (`op_type`,
`op_dt`, `op_amount`, `op_interval`, `op_count`, `op_read_idx`), per-subject
`op_ends`, per-axis `(obs_read, obs_map, ecg_read, ecg_map, tgi_read,
tgi_map)` row products (`*_read` the subject-local read index,
`*_map` the flat index into the concatenated per-subject read
vectors — empty vectors for undeclared axes), the `[conc; auc]`-space
`conc_map` (flat positions of the concentration half SB's
`tgi_conc_idx` recovers) and `tgi_auc_map` (flat positions of the AUC
at the tumor rows, SB's `tgi_auc_idx` globalized via the schedule's
read offsets), cumulative per-subject `tgi_seg_ends` (empty without
the tgi axis), `n_subjects`, per-subject `n_reads`, and build receipts
(`n_dose_rows`, `n_grouped_dose_events`, `n_segments`).
"""
function build_linear_pk_schedule(
        obs_subj::AbstractVector, obs_time::AbstractVector,
        dose_subj::AbstractVector, dose_time::AbstractVector,
        dose_amt::AbstractVector; min_segment_count::Integer = 3,
        combine_simultaneous::Bool = true,
        ecg::Union{Nothing,Tuple{AbstractVector,AbstractVector}} = nothing,
        tgi::Union{Nothing,Tuple{AbstractVector,AbstractVector}} = nothing)
    length(obs_subj) == length(obs_time) ||
        _pk_sched_fail("observation subject/time lengths differ " *
                       "($(length(obs_subj)) vs $(length(obs_time)))")
    length(dose_subj) == length(dose_time) == length(dose_amt) ||
        _pk_sched_fail("dose subject/time/amount lengths differ")
    for (axname, axis) in (("ecg", ecg), ("tgi", tgi))
        axis === nothing && continue
        length(axis[1]) == length(axis[2]) ||
            _pk_sched_fail("$axname axis subject/time lengths differ " *
                           "($(length(axis[1])) vs $(length(axis[2])))")
    end
    min_segment_count >= 3 ||
        _pk_sched_fail("min_segment_count must be at least 3 " *
                       "(got $min_segment_count)")

    subjects = Int.(obs_subj)
    times = Float64.(obs_time)
    dsubjects = Int.(dose_subj)
    dtimes = Float64.(dose_time)
    amounts = Float64.(dose_amt)
    # Extra read axes (SB's `vcat(pk, qt, tgi)` concatenation, same
    # order): validated like the primary axis, then joined into the
    # stream below and sliced back per axis at the end.
    n_obs_rows = length(subjects)
    extra_subj = Int[]
    extra_time = Float64[]
    n_ecg_rows = 0
    n_tgi_rows = 0
    for (axname, axis) in (("ecg", ecg), ("tgi", tgi))
        axis === nothing && continue
        asubj = Int.(axis[1])
        atime = Float64.(axis[2])
        all(>(0), asubj) ||
            _pk_sched_fail("$axname axis subject IDs must be positive")
        all(isfinite, atime) ||
            _pk_sched_fail("$axname axis times must be finite")
        append!(extra_subj, asubj)
        append!(extra_time, atime)
        if axname == "ecg"
            n_ecg_rows = length(asubj)
        else
            n_tgi_rows = length(asubj)
        end
    end
    append!(subjects, extra_subj)
    append!(times, extra_time)
    isempty(subjects) &&
        _pk_sched_fail("at least one observation is required")

    all(>(0), subjects) ||
        _pk_sched_fail("observation subject IDs must be positive")
    all(>(0), dsubjects) ||
        _pk_sched_fail("dose subject IDs must be positive")
    all(isfinite, times) ||
        _pk_sched_fail("observation times must be finite")
    all(isfinite, dtimes) ||
        _pk_sched_fail("dose times must be finite")
    all(isfinite, amounts) ||
        _pk_sched_fail("dose amounts must be finite")
    all(>=(0), amounts) ||
        _pk_sched_fail("dose amounts must be non-negative")

    n_subjects = maximum(subjects)
    unique_subjects = sort!(unique(subjects))
    unique_subjects == collect(1:n_subjects) ||
        _pk_sched_fail("observation subject IDs must be contiguous " *
                       "1:$n_subjects")
    all(<=(n_subjects), dsubjects) ||
        _pk_sched_fail("dose subject IDs must refer to observed subjects " *
                       "1:$n_subjects")

    obs_read = Vector{Int}(undef, length(subjects))
    op_ends = Int[]
    op_type = Int[]
    op_dt = Float64[]
    op_amount = Float64[]
    op_interval = Float64[]
    op_count = Int[]
    op_read_idx = Int[]
    n_reads = Int[]

    # Token layout: (time, kind, amount, read_idx). READ sorts before
    # DOSE at equal time because its kind code is smaller — a read at
    # the same timestamp as a dose sees the pre-dose state.
    Token = Tuple{Float64,Int,Float64,Int}
    n_segments = 0
    n_grouped_dose_events = 0

    push_op!(kind, dt, amount, interval, count, read_idx) = begin
        dt >= 0 ||
            _pk_sched_fail("internal schedule error: negative operation " *
                           "delta $dt")
        push!(op_type, kind)
        push!(op_dt, dt)
        push!(op_amount, amount)
        push!(op_interval, interval)
        push!(op_count, count)
        push!(op_read_idx, read_idx)
    end

    for subject in 1:n_subjects
        obs_idxs = findall(==(subject), subjects)
        read_times = sort!(unique(times[obs_idxs]))
        read_lookup = Dict{Float64,Int}()
        tokens = Token[]

        # Subject-local read numbering (file-header delta 1): the
        # counter restarts at 1 for every subject.
        next_read_idx = 0
        for time in read_times
            next_read_idx += 1
            read_lookup[time] = next_read_idx
            push!(tokens, (time, LINEAR_EVENT_READ, 0.0, next_read_idx))
        end
        push!(n_reads, next_read_idx)
        for idx in obs_idxs
            obs_read[idx] = read_lookup[times[idx]]
        end

        dose_idxs = findall(==(subject), dsubjects)
        if combine_simultaneous
            # Simultaneous doses are exactly additive (state-independent
            # — valid for the linear recurrence at F ≡ 1, the v1
            # spelling; the V2 event-LP binds with `false` below).
            summed_doses = Dict{Float64,Float64}()
            for idx in dose_idxs
                key = dtimes[idx]
                summed_doses[key] = get(summed_doses, key, 0.0) + amounts[idx]
            end
            for time in sort!(collect(keys(summed_doses)))
                amount = summed_doses[time]
                iszero(amount) && continue
                push!(tokens, (time, LINEAR_EVENT_DOSE, amount, 0))
                n_grouped_dose_events += 1
            end
        else
            # SB's `combine_simultaneous=false` path verbatim: separate
            # same-time dose rows stay separate zero-delta jumps (dose-row
            # order within the subject — a nonlinear dose-only effect
            # makes pre-summing raw amounts invalid).
            for idx in dose_idxs
                iszero(amounts[idx]) && continue
                push!(tokens, (dtimes[idx], LINEAR_EVENT_DOSE, amounts[idx], 0))
                n_grouped_dose_events += 1
            end
        end
        # SB sorts `(time, kind, pattern, amount)`; pattern is uniformly 1
        # here so its key drops out and this matches SB exactly.
        sort!(tokens; by = t -> (t[1], t[2], t[3]))

        isempty(tokens) &&
            _pk_sched_fail("internal schedule error: observed subject " *
                           "$subject has no tokens")
        last_time = first(tokens)[1]
        i = 1
        while i <= length(tokens)
            time, kind, amount, read_idx = tokens[i]
            if kind == LINEAR_EVENT_READ
                push_op!(kind, time - last_time, amount, 0.0, 0, read_idx)
                last_time = time
                i += 1
                continue
            end

            # The maximal exact equal-amount/equal-interval run among
            # consecutive DOSE tokens. Any intervening READ token ends
            # the run.
            run_end = i
            interval = 0.0
            if i + 2 <= length(tokens) &&
               tokens[i + 1][2] == LINEAR_EVENT_DOSE &&
               tokens[i + 2][2] == LINEAR_EVENT_DOSE &&
               tokens[i + 1][3] == amount && tokens[i + 2][3] == amount
                candidate = tokens[i + 1][1] - time
                if candidate > 0 &&
                   tokens[i + 2][1] - tokens[i + 1][1] == candidate
                    interval = candidate
                    run_end = i + 2
                    while run_end + 1 <= length(tokens) &&
                          tokens[run_end + 1][2] == LINEAR_EVENT_DOSE &&
                          tokens[run_end + 1][3] == amount &&
                          tokens[run_end + 1][1] - tokens[run_end][1] == interval
                        run_end += 1
                    end
                end
            end

            count = run_end - i + 1
            if count >= min_segment_count
                push_op!(LINEAR_EVENT_DOSE_SEGMENT, time - last_time,
                    amount, interval, count, 0)
                last_time = tokens[run_end][1]
                n_segments += 1
                i = run_end + 1
            else
                push_op!(LINEAR_EVENT_DOSE, time - last_time, amount,
                    0.0, 1, 0)
                last_time = time
                i += 1
            end
        end
        push!(op_ends, length(op_type))
    end

    # Flat read offsets: subject s owns `offsets[s]+1 : offsets[s]+R_s`
    # of the concatenated per-subject read vectors. The generator's
    # obs gather is `flat[obs_map]` (one static int per obs row).
    offsets = Vector{Int}(undef, n_subjects)
    total = 0
    for s in 1:n_subjects
        offsets[s] = total
        total += n_reads[s]
    end
    full_map = Vector{Int}(undef, length(subjects))
    for i in eachindex(subjects, obs_read)
        full_map[i] = offsets[subjects[i]] + obs_read[i]
    end
    # SB's per-axis slicing: row ranges over the concatenated axis
    # recover each endpoint's products in caller row order (empty
    # ranges stay empty vectors for undeclared axes).
    ecg_range = (n_obs_rows + 1):(n_obs_rows + n_ecg_rows)
    tgi_range =
        (n_obs_rows + n_ecg_rows + 1):(n_obs_rows + n_ecg_rows + n_tgi_rows)
    full_read = obs_read
    obs_read = full_read[1:n_obs_rows]
    obs_map = full_map[1:n_obs_rows]
    ecg_read = full_read[ecg_range]
    ecg_map = full_map[ecg_range]
    tgi_read = full_read[tgi_range]
    tgi_map = full_map[tgi_range]
    # [conc; auc]-space positions (SB's per-subject `tgi_conc_idx` /
    # `tgi_auc_idx` globalized via the read offsets): subject s owns
    # the flat block `2*offsets[s]+1 : 2*offsets[s]+2R_s`
    # (concentrations, then cumulative AUCs).
    conc_map = Int[]
    for s in 1:n_subjects
        append!(conc_map,
            (2 * offsets[s] + 1):(2 * offsets[s] + n_reads[s]))
    end
    tgi_auc_map = Vector{Int}(undef, n_tgi_rows)
    for (k, i) in enumerate(tgi_range)
        s = subjects[i]
        tgi_auc_map[k] = 2 * offsets[s] + n_reads[s] + full_read[i]
    end
    # Cumulative per-subject TGI row ends (the `op_ends` precedent):
    # segment s is tgi rows `prev+1:ends[s]`, empty when ends repeat
    # (subjects without assessments). Empty for undeclared axes; the
    # bind validator requires grouped-by-subject rows (counts alone
    # cannot see interleaving).
    tgi_seg_ends = Int[]
    if tgi !== nothing
        seg_counts = zeros(Int, n_subjects)
        for i in tgi_range
            seg_counts[subjects[i]] += 1
        end
        tgi_seg_ends = cumsum(seg_counts)
    end

    return (; obs_read, obs_map, ecg_read, ecg_map, tgi_read, tgi_map,
        conc_map, tgi_auc_map, tgi_seg_ends, op_ends, op_type, op_dt,
        op_amount, op_interval, op_count, op_read_idx, n_subjects, n_reads,
        n_reads_total = total, n_dose_rows = length(amounts),
        n_grouped_dose_events, n_segments)
end

"""Zero-effect reference dose for the event-axis log-dose column (SB
`_LINEAR_PK_V6_LOG_REF_DOSE`, verbatim): 10,000 is the zero-effect
reference and 200,000 maps to twice the raw slope."""
const _PK_LOG_REF_DOSE = log(10_000.0)

"""
    linear_pk_op_log_dose(op_type, op_amount) -> Vector{Float64}

The normalized log-dose event-axis column (SB
`_linear_pk_dose_event_axis`'s `op_log_dose`, same validations, same
construction): `log(op_amount) - log(10_000)` on dosing ops over the
flat subject-blocked op stream. READ ops carry no dose, and `log(0)`
is `-Inf`, so they take the mean over dosing operations — clamped
into the observed dosing support (SB's one-ulp guard), hence interior
by construction, so an `hsgp(op_log_dose)` basis built from the
column's range never widens. The cell never reads `log_F` on the READ
branch, so the fill value never enters the likelihood.

Fails closed exactly where SB errors: no dosing operation, a
non-positive dosing amount, a non-finite column, or a READ fill
outside the dosing support.
"""
function linear_pk_op_log_dose(op_type::AbstractVector,
        op_amount::AbstractVector)
    n = length(op_type)
    length(op_amount) == n ||
        _pk_sched_fail("op_log_dose columns disagree in length " *
                       "($n vs $(length(op_amount)))")
    dosing = op_type .!= LINEAR_EVENT_READ
    any(dosing) ||
        _pk_sched_fail("the event schedule has no dosing operation")
    all(>(0.0), op_amount[dosing]) ||
        _pk_sched_fail("a dosing operation has a non-positive amount, so " *
                       "log dose is undefined")
    dose_log_dose = log.(op_amount[dosing]) .- _PK_LOG_REF_DOSE
    dose_lo, dose_hi = extrema(dose_log_dose)
    # `sum/length` is `Statistics.mean` verbatim for real vectors (the
    # `_fit_hsgp_bases` precedent); the clamp is SB's one-ulp guard for
    # a mean rounding outside a degenerate/narrow interval.
    read_fill =
        clamp(sum(dose_log_dose) / length(dose_log_dose), dose_lo, dose_hi)
    op_log_dose = similar(dose_log_dose, n)
    op_log_dose[dosing] .= dose_log_dose
    op_log_dose[.!dosing] .= read_fill
    all(isfinite, op_log_dose) ||
        _pk_sched_fail("op_log_dose is not finite")
    all(x -> dose_lo <= x <= dose_hi, op_log_dose) ||
        _pk_sched_fail("READ-row fill widened the log-dose range, which " *
                       "would distort an hsgp basis built from it")
    return op_log_dose
end

# --- Hand-rolled matrix kernels (no BLAS/LAPACK — see file header) ---
#
# Matrices and states are column-major tuples, updated FUNCTIONALLY
# (never mutated, never heap-allocated): a concretely-typed
# `Matrix{Float64}`/`Vector{Float64}` rejects traced stores
# (`setindex!` hits `convert(Float64, ::TracedRNumber)` — measured),
# while tuples promote elementwise and trace cleanly. Signatures use
# `NTuple{N,Any}` (length-checked, eltype-open: traced mixes with
# constant zeros). Natively the tuples live in registers (LLVM SROA);
# under Enzyme they are plain SSA values; under Reactant the static
# loops unroll. The one heap array (`read_locs`, in
# `linear_pk_read_locs`) is allocated eltype-generic.

"""Stan `matrix_exp_pade` degree-selection thresholds (Higham thetas, verbatim)."""
const _PK_PADE_THETA3 = 1.495585217958292e-002
const _PK_PADE_THETA5 = 2.539398330063230e-001
const _PK_PADE_THETA7 = 9.504178996162932e-001
const _PK_PADE_THETA9 = 2.097847961257068e+000
"""Stan `matrix_exp_pade` scaling target norm (theta-13, verbatim)."""
const _PK_PADE_MAXNORM = 5.371920351148152
"""Column-major 3x3 identity tuple (Stan's `MatrixType::Identity`)."""
const _PK_I3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

# C = A*B for column-major 3x3 tuples (C[i,j] = Σ_k A[i,k]*B[k,j],
# linear index (k-1)*3+i).
_pk_tmul3(a::NTuple{9,Any}, b::NTuple{9,Any}) = (
    a[1] * b[1] + a[4] * b[2] + a[7] * b[3],
    a[2] * b[1] + a[5] * b[2] + a[8] * b[3],
    a[3] * b[1] + a[6] * b[2] + a[9] * b[3],
    a[1] * b[4] + a[4] * b[5] + a[7] * b[6],
    a[2] * b[4] + a[5] * b[5] + a[8] * b[6],
    a[3] * b[4] + a[6] * b[5] + a[9] * b[6],
    a[1] * b[7] + a[4] * b[8] + a[7] * b[9],
    a[2] * b[7] + a[5] * b[8] + a[8] * b[9],
    a[3] * b[7] + a[6] * b[8] + a[9] * b[9])

# Elementwise tuple arithmetic. Stan's `s*A + t*B` chains evaluate
# left-associatively per entry; these helpers compose in that order so
# the port matches Eigen's rounding term for term.
_pk_tadd3(a::NTuple{9,Any}, b::NTuple{9,Any}) = (
    a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4], a[5] + b[5],
    a[6] + b[6], a[7] + b[7], a[8] + b[8], a[9] + b[9])
_pk_tsub3(a::NTuple{9,Any}, b::NTuple{9,Any}) = (
    a[1] - b[1], a[2] - b[2], a[3] - b[3], a[4] - b[4], a[5] - b[5],
    a[6] - b[6], a[7] - b[7], a[8] - b[8], a[9] - b[9])
_pk_tscale3(s, a::NTuple{9,Any}) = (
    s * a[1], s * a[2], s * a[3], s * a[4], s * a[5], s * a[6],
    s * a[7], s * a[8], s * a[9])

# 1-norm (max column absolute sum), Eigen's maxCoeff scan: strict `>`,
# first maximum wins.
function _pk_l1norm3(M::NTuple{9,Any})
    c1 = abs(M[1]) + abs(M[2]) + abs(M[3])
    c2 = abs(M[4]) + abs(M[5]) + abs(M[6])
    c3 = abs(M[7]) + abs(M[8]) + abs(M[9])
    m12 = ifelse(c2 > c1, c2, c1)
    return ifelse(c3 > m12, c3, m12)
end

# (3,3)-Pade pair (Stan `matrix_exp_pade3`, same op order):
# U = A*(b3*A2 + b1*I), V = b2*A2 + b0*I.
function _pk_pade3(A::NTuple{9,Any})
    A2 = _pk_tmul3(A, A)
    tmp = _pk_tadd3(_pk_tscale3(1.0, A2), _pk_tscale3(60.0, _PK_I3))
    U = _pk_tmul3(A, tmp)
    V = _pk_tadd3(_pk_tscale3(12.0, A2), _pk_tscale3(120.0, _PK_I3))
    return U, V
end

# (5,5)-Pade pair (Stan `matrix_exp_pade5`, same op order).
function _pk_pade5(A::NTuple{9,Any})
    A2 = _pk_tmul3(A, A)
    A4 = _pk_tmul3(A2, A2)
    tmp = _pk_tadd3(_pk_tadd3(_pk_tscale3(1.0, A4),
            _pk_tscale3(420.0, A2)), _pk_tscale3(15120.0, _PK_I3))
    U = _pk_tmul3(A, tmp)
    V = _pk_tadd3(_pk_tadd3(_pk_tscale3(30.0, A4),
            _pk_tscale3(3360.0, A2)), _pk_tscale3(30240.0, _PK_I3))
    return U, V
end

# (7,7)-Pade pair (Stan `matrix_exp_pade7`, same op order).
function _pk_pade7(A::NTuple{9,Any})
    A2 = _pk_tmul3(A, A)
    A4 = _pk_tmul3(A2, A2)
    A6 = _pk_tmul3(A4, A2)
    tmp = _pk_tadd3(_pk_tadd3(_pk_tadd3(_pk_tscale3(1.0, A6),
                    _pk_tscale3(1512.0, A4)), _pk_tscale3(277200.0, A2)),
        _pk_tscale3(8648640.0, _PK_I3))
    U = _pk_tmul3(A, tmp)
    V = _pk_tadd3(_pk_tadd3(_pk_tadd3(_pk_tscale3(56.0, A6),
                    _pk_tscale3(25200.0, A4)), _pk_tscale3(1995840.0, A2)),
        _pk_tscale3(17297280.0, _PK_I3))
    return U, V
end

# (9,9)-Pade pair (Stan `matrix_exp_pade9`, same op order).
function _pk_pade9(A::NTuple{9,Any})
    A2 = _pk_tmul3(A, A)
    A4 = _pk_tmul3(A2, A2)
    A6 = _pk_tmul3(A4, A2)
    A8 = _pk_tmul3(A6, A2)
    tmp = _pk_tadd3(_pk_tadd3(_pk_tadd3(_pk_tadd3(
                        _pk_tscale3(1.0, A8), _pk_tscale3(3960.0, A6)),
                    _pk_tscale3(2162160.0, A4)),
                _pk_tscale3(302702400.0, A2)),
        _pk_tscale3(8821612800.0, _PK_I3))
    U = _pk_tmul3(A, tmp)
    V = _pk_tadd3(_pk_tadd3(_pk_tadd3(_pk_tadd3(
                        _pk_tscale3(90.0, A8), _pk_tscale3(110880.0, A6)),
                    _pk_tscale3(30270240.0, A4)),
                _pk_tscale3(2075673600.0, A2)),
        _pk_tscale3(17643225600.0, _PK_I3))
    return U, V
end

# (13,13)-Pade pair (Stan `matrix_exp_pade13`, same op order — the C++
# reuses `V` for scratch storage; the functional port names each
# intermediate instead, identical arithmetic).
function _pk_pade13(A::NTuple{9,Any})
    A2 = _pk_tmul3(A, A)
    A4 = _pk_tmul3(A2, A2)
    A6 = _pk_tmul3(A4, A2)
    V6 = _pk_tadd3(_pk_tadd3(_pk_tscale3(1.0, A6),
            _pk_tscale3(16380.0, A4)), _pk_tscale3(40840800.0, A2))
    tmp = _pk_tadd3(_pk_tmul3(A6, V6), _pk_tadd3(_pk_tadd3(
                _pk_tadd3(_pk_tscale3(33522128640.0, A6),
                    _pk_tscale3(10559470521600.0, A4)),
                _pk_tscale3(1187353796428800.0, A2)),
            _pk_tscale3(32382376266240000.0, _PK_I3)))
    U = _pk_tmul3(A, tmp)
    tmp2 = _pk_tadd3(_pk_tadd3(_pk_tscale3(182.0, A6),
            _pk_tscale3(960960.0, A4)), _pk_tscale3(1323241920.0, A2))
    V = _pk_tadd3(_pk_tmul3(A6, tmp2), _pk_tadd3(_pk_tadd3(
                _pk_tadd3(_pk_tscale3(670442572800.0, A6),
                    _pk_tscale3(129060195264000.0, A4)),
                _pk_tscale3(7771770303897600.0, A2)),
            _pk_tscale3(64764752532480000.0, _PK_I3)))
    return U, V
end

# Predicated row-entry swap (branchless pivot): the pair with entries
# exchanged iff `c`.
_pk_pivot_swap(c, x, y) = (ifelse(c, y, x), ifelse(c, x, y))

# Predicated 3x3 pick (branchless stage select): `A` iff `c`, else `B`.
_pk_tpick3(c, A::NTuple{9,Any}, B::NTuple{9,Any}) = (
    ifelse(c, A[1], B[1]), ifelse(c, A[2], B[2]), ifelse(c, A[3], B[3]),
    ifelse(c, A[4], B[4]), ifelse(c, A[5], B[5]), ifelse(c, A[6], B[6]),
    ifelse(c, A[7], B[7]), ifelse(c, A[8], B[8]), ifelse(c, A[9], B[9]))

# Unit-lower forward + upper back substitution for one RHS column
# (Eigen's small-matrix triangular solves, same per-entry order).
function _pk_tri_solve3(l10, l20, l21, d11, d12, d13, d22, d23, d33,
        n0, n1, n2)
    y0 = n0
    y1 = n1 - l10 * y0
    y2 = (n2 - l20 * y0) - l21 * y1
    x2 = y2 / d33
    x1 = (y1 - d23 * x2) / d22
    x0 = ((y0 - d12 * x1) - d13 * x2) / d11
    return (x0, x1, x2)
end

"""
    _pk_lu_solve3(D, N) -> 9-tuple

Solve `D*X = N` for column-major 3x3 tuples via partial-pivot LU
(Stan's `denom.partialPivLu().solve(number)`): the row swaps are
predicated on the same strict-magnitude comparisons (Eigen's
first-maximum-wins pivot rule — a two-swap sorting network
reproduces it exactly, ties included), then straight-line
elimination and triangular solves. No guards: a singular `D`
propagates Inf/NaN exactly as Eigen's unguarded factorization does.
"""
function _pk_lu_solve3(D::NTuple{9,Any}, N::NTuple{9,Any})
    d11, d21, d31 = D[1], D[2], D[3]
    d12, d22, d32 = D[4], D[5], D[6]
    d13, d23, d33 = D[7], D[8], D[9]
    n11, n21, n31 = N[1], N[2], N[3]
    n12, n22, n32 = N[4], N[5], N[6]
    n13, n23, n33 = N[7], N[8], N[9]
    # Pivot step 0: the first-maximum-magnitude row leads.
    s01 = abs(d11) < abs(d21)
    (d11, d21) = _pk_pivot_swap(s01, d11, d21)
    (d12, d22) = _pk_pivot_swap(s01, d12, d22)
    (d13, d23) = _pk_pivot_swap(s01, d13, d23)
    (n11, n21) = _pk_pivot_swap(s01, n11, n21)
    (n12, n22) = _pk_pivot_swap(s01, n12, n22)
    (n13, n23) = _pk_pivot_swap(s01, n13, n23)
    s02 = abs(d11) < abs(d31)
    (d11, d31) = _pk_pivot_swap(s02, d11, d31)
    (d12, d32) = _pk_pivot_swap(s02, d12, d32)
    (d13, d33) = _pk_pivot_swap(s02, d13, d33)
    (n11, n31) = _pk_pivot_swap(s02, n11, n31)
    (n12, n32) = _pk_pivot_swap(s02, n12, n32)
    (n13, n33) = _pk_pivot_swap(s02, n13, n33)
    # Eliminate column 0.
    l10 = d21 / d11
    d22 = d22 - l10 * d12
    d23 = d23 - l10 * d13
    l20 = d31 / d11
    d32 = d32 - l20 * d12
    d33 = d33 - l20 * d13
    # Pivot step 1 (the computed multipliers travel with their rows)
    # + eliminate column 1.
    s12 = abs(d22) < abs(d32)
    (d22, d32) = _pk_pivot_swap(s12, d22, d32)
    (d23, d33) = _pk_pivot_swap(s12, d23, d33)
    (l10, l20) = _pk_pivot_swap(s12, l10, l20)
    (n21, n31) = _pk_pivot_swap(s12, n21, n31)
    (n22, n32) = _pk_pivot_swap(s12, n22, n32)
    (n23, n33) = _pk_pivot_swap(s12, n23, n33)
    l21 = d32 / d22
    d33 = d33 - l21 * d23
    X0 = _pk_tri_solve3(l10, l20, l21, d11, d12, d13, d22, d23, d33,
        n11, n21, n31)
    X1 = _pk_tri_solve3(l10, l20, l21, d11, d12, d13, d22, d23, d33,
        n12, n22, n32)
    X2 = _pk_tri_solve3(l10, l20, l21, d11, d12, d13, d22, d23, d33,
        n13, n23, n33)
    return (X0[1], X0[2], X0[3], X1[1], X1[2], X1[3], X2[1], X2[2], X2[3])
end

# One predicated degree select (Stan's theta cascade, branchless).
_pk_pick1(c3, a3, c5, a5, c7, a7, c9, a9, a13) =
    ifelse(c3, a3, ifelse(c5, a5, ifelse(c7, a7, ifelse(c9, a9, a13))))

# Degree selection over (U, V) pairs: the same strict-`<` comparisons
# Stan makes choose the same pair, entry by entry.
function _pk_pick_uv(c3, T3::NTuple{9,Any}, c5, T5::NTuple{9,Any},
        c7, T7::NTuple{9,Any}, c9, T9::NTuple{9,Any}, T13::NTuple{9,Any})
    return (_pk_pick1(c3, T3[1], c5, T5[1], c7, T7[1], c9, T9[1], T13[1]),
        _pk_pick1(c3, T3[2], c5, T5[2], c7, T7[2], c9, T9[2], T13[2]),
        _pk_pick1(c3, T3[3], c5, T5[3], c7, T7[3], c9, T9[3], T13[3]),
        _pk_pick1(c3, T3[4], c5, T5[4], c7, T7[4], c9, T9[4], T13[4]),
        _pk_pick1(c3, T3[5], c5, T5[5], c7, T7[5], c9, T9[5], T13[5]),
        _pk_pick1(c3, T3[6], c5, T5[6], c7, T7[6], c9, T9[6], T13[6]),
        _pk_pick1(c3, T3[7], c5, T5[7], c7, T7[7], c9, T9[7], T13[7]),
        _pk_pick1(c3, T3[8], c5, T5[8], c7, T7[8], c9, T9[8], T13[8]),
        _pk_pick1(c3, T3[9], c5, T5[9], c7, T7[9], c9, T9[9], T13[9]))
end

"""
    _pk_expm3(M) -> 9-tuple

`exp(M)` for column-major 3x3 `M`: faithful julianic port of Stan's
`matrix_exp_pade` (Eigen `matrix_exp_computeUV` + the `(U+V)/(-U+V)`
Pade fraction solved by partial-pivot LU + repeated squaring) — same
approximants, same theta selection, same squaring schedule;
traceable expression only, no re-mathematizing.

The norm-based selection is predicated, not branched: all five
`(U, V)` pairs evaluate and the theta cascade selects via `ifelse`
(the same strict-`<` comparisons Stan makes); squaring step `i`
applies iff `l1norm >= maxnorm * 2^(i-1)`, which is exactly Stan's
`frexp(l1norm/maxnorm)` schedule (`s >= i` iff `x >= 2^(i-1)`,
including the `s < 0 → 0` clamp and exact-power-of-two
boundaries — the clamped case is `l1norm < maxnorm/2`, where every
predicate is false). Scaling and squaring run as ten straight-line
predicated stages (candidate chains + prefix picks — no
loop-carried predicated values), and the LU pivot is predicated row
swaps (Eigen's first-maximum-wins rule, ties included). Scaling is
exact powers of two (`*0.5` halving, Stan's `ldexp` bit-identically).

Accuracy envelope: exact-Stan while `l1norm(M) < maxnorm * 2^10 ≈
5499` (the ten predicated stages saturate the schedule beyond that —
typical PK arguments are `||A*dt|| < 50`, the committed edges reach
`l1 ≈ 355`). Beyond the envelope the scaling saturates and accuracy
degrades gradually (no guard: a norm check cannot fail loudly
in-graph under Reactant). Validated against `LinearAlgebra.exp` in
`test_pkcells.jl`.
"""
function _pk_expm3(M::NTuple{9,Any})
    l1 = _pk_l1norm3(M)
    # Squaring predicates (see docstring): the bound doubles exactly.
    bound = _PK_PADE_MAXNORM
    ap1 = l1 >= bound
    bound = bound * 2.0
    ap2 = l1 >= bound
    bound = bound * 2.0
    ap3 = l1 >= bound
    bound = bound * 2.0
    ap4 = l1 >= bound
    bound = bound * 2.0
    ap5 = l1 >= bound
    bound = bound * 2.0
    ap6 = l1 >= bound
    bound = bound * 2.0
    ap7 = l1 >= bound
    bound = bound * 2.0
    ap8 = l1 >= bound
    bound = bound * 2.0
    ap9 = l1 >= bound
    bound = bound * 2.0
    ap10 = l1 >= bound
    # Scale candidates 2^-k (exact halving) + prefix picks: the bounds
    # grow monotonically, so the predicates are a prefix pattern
    # (ap_i ⟹ ap_j for j < i) and stage i keeps candidate i iff
    # s >= i. Straight-line — no loop-carried predicated values
    # (Enzyme's reverse zeroes those; forward + values agree).
    sc0 = 1.0
    sc1 = sc0 * 0.5
    sc2 = sc1 * 0.5
    sc3 = sc2 * 0.5
    sc4 = sc3 * 0.5
    sc5 = sc4 * 0.5
    sc6 = sc5 * 0.5
    sc7 = sc6 * 0.5
    sc8 = sc7 * 0.5
    sc9 = sc8 * 0.5
    sc10 = sc9 * 0.5
    sc = sc0
    sc = ifelse(ap1, sc1, sc)
    sc = ifelse(ap2, sc2, sc)
    sc = ifelse(ap3, sc3, sc)
    sc = ifelse(ap4, sc4, sc)
    sc = ifelse(ap5, sc5, sc)
    sc = ifelse(ap6, sc6, sc)
    sc = ifelse(ap7, sc7, sc)
    sc = ifelse(ap8, sc8, sc)
    sc = ifelse(ap9, sc9, sc)
    sc = ifelse(ap10, sc10, sc)
    As = (M[1] * sc, M[2] * sc, M[3] * sc, M[4] * sc, M[5] * sc,
        M[6] * sc, M[7] * sc, M[8] * sc, M[9] * sc)
    # All five pairs; the cascade selects Stan's degree (degrees
    # 3-9 see the unscaled arg, 13 the scaled one, exactly as Stan).
    U3, V3 = _pk_pade3(M)
    U5, V5 = _pk_pade5(M)
    U7, V7 = _pk_pade7(M)
    U9, V9 = _pk_pade9(M)
    U13, V13 = _pk_pade13(As)
    c3 = l1 < _PK_PADE_THETA3
    c5 = l1 < _PK_PADE_THETA5
    c7 = l1 < _PK_PADE_THETA7
    c9 = l1 < _PK_PADE_THETA9
    U = _pk_pick_uv(c3, U3, c5, U5, c7, U7, c9, U9, U13)
    V = _pk_pick_uv(c3, V3, c5, V5, c7, V7, c9, V9, V13)
    number = _pk_tadd3(U, V)
    denom = _pk_tsub3(V, U)
    X = _pk_lu_solve3(denom, number)
    # Square back up: the ten repeated squares as candidates + prefix
    # picks (same prefix argument as the scale above). The kept value
    # is exactly the loop form's (X_s); the discarded higher powers
    # are picked away, never observed.
    X0 = X
    X1 = _pk_tmul3(X0, X0)
    X2 = _pk_tmul3(X1, X1)
    X3 = _pk_tmul3(X2, X2)
    X4 = _pk_tmul3(X3, X3)
    X5 = _pk_tmul3(X4, X4)
    X6 = _pk_tmul3(X5, X5)
    X7 = _pk_tmul3(X6, X6)
    X8 = _pk_tmul3(X7, X7)
    X9 = _pk_tmul3(X8, X8)
    X10 = _pk_tmul3(X9, X9)
    Xf = X0
    Xf = _pk_tpick3(ap1, X1, Xf)
    Xf = _pk_tpick3(ap2, X2, Xf)
    Xf = _pk_tpick3(ap3, X3, Xf)
    Xf = _pk_tpick3(ap4, X4, Xf)
    Xf = _pk_tpick3(ap5, X5, Xf)
    Xf = _pk_tpick3(ap6, X6, Xf)
    Xf = _pk_tpick3(ap7, X7, Xf)
    Xf = _pk_tpick3(ap8, X8, Xf)
    Xf = _pk_tpick3(ap9, X9, Xf)
    Xf = _pk_tpick3(ap10, X10, Xf)
    return Xf
end

# C = A*B for column-major 4x4 tuples (linear index (k-1)*4+i).
_pk_tmul4(a::NTuple{16,Any}, b::NTuple{16,Any}) = (
    a[1] * b[1] + a[5] * b[2] + a[9] * b[3] + a[13] * b[4],
    a[2] * b[1] + a[6] * b[2] + a[10] * b[3] + a[14] * b[4],
    a[3] * b[1] + a[7] * b[2] + a[11] * b[3] + a[15] * b[4],
    a[4] * b[1] + a[8] * b[2] + a[12] * b[3] + a[16] * b[4],
    a[1] * b[5] + a[5] * b[6] + a[9] * b[7] + a[13] * b[8],
    a[2] * b[5] + a[6] * b[6] + a[10] * b[7] + a[14] * b[8],
    a[3] * b[5] + a[7] * b[6] + a[11] * b[7] + a[15] * b[8],
    a[4] * b[5] + a[8] * b[6] + a[12] * b[7] + a[16] * b[8],
    a[1] * b[9] + a[5] * b[10] + a[9] * b[11] + a[13] * b[12],
    a[2] * b[9] + a[6] * b[10] + a[10] * b[11] + a[14] * b[12],
    a[3] * b[9] + a[7] * b[10] + a[11] * b[11] + a[15] * b[12],
    a[4] * b[9] + a[8] * b[10] + a[12] * b[11] + a[16] * b[12],
    a[1] * b[13] + a[5] * b[14] + a[9] * b[15] + a[13] * b[16],
    a[2] * b[13] + a[6] * b[14] + a[10] * b[15] + a[14] * b[16],
    a[3] * b[13] + a[7] * b[14] + a[11] * b[15] + a[15] * b[16],
    a[4] * b[13] + a[8] * b[14] + a[12] * b[15] + a[16] * b[16])

# Binary exponentiation for the 4x4 augmented dose transition.
# Native execution uses ordinary binary-power iteration. Tracing retains the
# recurrence even when the exponent is bound data known during preparation.
function _pk_matpow4(A::NTuple{16,Any}, n::Integer)
    marker = ReactiveKernels._dynamic_tensorized_marker(A)
    marker === nothing || return _pk_retained_power(A, n,
        zeros(Int, ndigits(max(n, 0); base=2)), marker)
    R = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    B = A
    e = Int(n)
    while e > 0
        if e & 1 == 1
            R = _pk_tmul4(R, B)
        end
        B = _pk_tmul4(B, B)
        e >>= 1
    end
    return R
end

# --- SB primitive mirrors (`src/pkpd_models.jl`, same op order) ---

"""
    linear_pk_system_3(log_CL, log_Vc, log_Q, log_Vp, log_ka) -> 9-tuple

Three-state first-order-absorption/two-compartment amount-system matrix
(SB `linear_pk_system`, same op order: `exp` the logs, derive the
micro-constants, column-stack) as a column-major tuple (see the
functional-tuple note above).
"""
# Scalar arguments are deliberately untyped: Reactant's traced scalars
# are not `Real`, so an annotation would reject the traced call (the
# `gp_chol_latent` precedent leaves scalars untyped the same way).
function linear_pk_system_3(log_CL, log_Vc, log_Q, log_Vp, log_ka)
    CL = exp(log_CL)
    Vc = exp(log_Vc)
    Q = exp(log_Q)
    Vp = exp(log_Vp)
    ka = exp(log_ka)
    k10 = CL / Vc
    k12 = Q / Vc
    k21 = Q / Vp
    return (-ka, ka, 0.0, 0.0, -(k10 + k12), k12, 0.0, k21, -k21)
end

# y = E*x for a column-major 3x3 tuple and a 3-tuple.
_pk_tmatvec3(E::NTuple{9,Any}, x::NTuple{3,Any}) = (
    E[1] * x[1] + E[4] * x[2] + E[7] * x[3],
    E[2] * x[1] + E[5] * x[2] + E[8] * x[3],
    E[3] * x[1] + E[6] * x[2] + E[9] * x[3])

"""
    linear_pk_propagate_3(A, state, dt) -> 3-tuple

Propagate a three-state linear PK system across one interval (SB
`linear_pk_propagate`: identity at `dt <= 0`, else `exp(A*dt)*state`
— the exponential is [`_pk_expm3`](@ref), a faithful port of Stan's
`matrix_exp_pade`, so there is no approximant delta from SB's
`matrix_exp`).
"""
function linear_pk_propagate_3(A::NTuple{9,Any}, state::NTuple{3,Any}, dt)
    if dt > 0
        return _pk_propagate_positive(A, state, dt)
    end
    return state
end

@inline function _pk_propagate_positive(A, state, dt)
    M = (A[1] * dt, A[2] * dt, A[3] * dt, A[4] * dt, A[5] * dt,
        A[6] * dt, A[7] * dt, A[8] * dt, A[9] * dt)
    _pk_tmatvec3(_pk_expm3(M), state)
end

"""
    linear_pk_add_dose_3(state, amount) -> 3-tuple

One oral dose jump to the gut amount state (SB `linear_pk_add_dose`).
"""
function linear_pk_add_dose_3(state::NTuple{3,Any}, amount)
    return (state[1] + amount, state[2], state[3])
end

"""
    linear_pk_add_regular_doses_3(A, state, amount, interval, count) -> 3-tuple

`count` equal oral doses at a fixed interval, beginning now (SB
`linear_pk_add_regular_doses`: immediate first jump, then the augmented
affine transition `[P b; 0 1]^(count-1)` with `P = exp(A*interval)`
and `b = [amount, 0, 0]` — the power is [`_pk_matpow4`](@ref)).
"""
function linear_pk_add_regular_doses_3(A::NTuple{9,Any}, state::NTuple{3,Any},
        amount, interval, count::Integer)
    after_first = linear_pk_add_dose_3(state, amount)
    count > 1 || return after_first
    affine = _pk_dose_affine(A, amount, interval)
    Q = _pk_matpow4(affine, count - 1)
    _pk_apply_affine(Q, after_first)
end

@inline function _pk_dose_affine(A, amount, interval)
    M = (A[1] * interval, A[2] * interval, A[3] * interval,
        A[4] * interval, A[5] * interval, A[6] * interval,
        A[7] * interval, A[8] * interval, A[9] * interval)
    _pk_affine_from_exp(_pk_expm3(M), amount)
end

# The dose-interval affine map from exp(A*interval): propagate, then add a dose.
@inline _pk_affine_from_exp(P, amount) =
    (P[1], P[2], P[3], 0.0, P[4], P[5], P[6], 0.0,
        P[7], P[8], P[9], 0.0, amount, 0.0, 0.0, 1.0)

@inline function _pk_apply_affine(Q, after_first)
    return (Q[1] * after_first[1] + Q[5] * after_first[2] +
            Q[9] * after_first[3] + Q[13],
        Q[2] * after_first[1] + Q[6] * after_first[2] +
            Q[10] * after_first[3] + Q[14],
        Q[3] * after_first[1] + Q[7] * after_first[2] +
            Q[11] * after_first[3] + Q[15])
end

"""`sqrt(2π)` verbatim from SB `brm_hsgp_sqrt_spd` (the event-LP
spectral scale — the `_HSGP_SQRT2PI` Stage-B twin)."""
const _PK_EVENT_SQRT2PI = 2.5066282746310002

"""
    linear_pk_event_log_f(op_log_dose, slope, rho, sigma, beta_raw, mu, L, k)
        -> AbstractVector

The default-V2 event-axis bioavailability LP (SB `log_F ~ 0 +
op_log_dose + hsgp(op_log_dose; k = 5)`, same op order as the landed
Stage-B `_hsgp_basis_stmts` evaluation): the data linear part
`slope .* op_log_dose` plus the 1-D HSGP smooth `PHI * (sqrt_spd .*
beta_raw)` over `k` modes, one value per schedule op (reads
included — never read on the READ branch, exactly as SB).

The HSGP axis is DATA (`op_log_dose` bound at bind, `(mu, L)` frozen
at bind) while the hyperparameters (`slope`, `rho`, `sigma`,
`beta_raw`) are TRACED — the landed basis-at-bind/coefficients-traced
split. Called from generated grouped-kernel code (one call over the
flat event axis — the provider emits the flat `log_F` local the
per-subject expansion slices) and from the host-side oracle path;
the same function on both paths keeps spec and graph bit-identical
by construction (the `linear_pk_read_locs` precedent).

Scalar arguments are deliberately untyped (the `linear_pk_system_3`
precedent — Reactant's traced scalars are not `Real`); `k` is a
static integer (surface-admitted, default 5). No mutation, no
scalar indexing into traced stores: the data basis fills by
explicit loops (no `hcat` splat — see the body), then one matvec.
"""
function linear_pk_event_log_f(op_log_dose::AbstractVector, slope, rho,
        sigma, beta_raw::AbstractVector, mu, L, k::Integer)
    length(beta_raw) == k ||
        throw(ArgumentError("linear_pk_event_log_f needs $k hsgp " *
                            "coefficients (got $(length(beta_raw)))"))
    L > 0 ||
        throw(ArgumentError("linear_pk_event_log_f needs a positive " *
                            "domain half-width L (got $L)"))
    inv_sqrt_L = 1.0 / sqrt(L)
    # Per-mode trig columns (SB `_brm_apply_hsgp`, element order
    # verbatim — the `sqrt((k*pi/(2L))^2)` shape included). Explicit
    # fill loops, NOT a comprehension + `hcat` splat: the splat boxes
    # constant column vectors into a container that feeds active
    # math, which Enzyme's static activity analysis rejects
    # (`EnzymeRuntimeActivityError` — measured); the graph's Stage-B
    # emission unrolls the same columns as separate statements for
    # the same reason. Element values are identical either way.
    n = length(op_log_dose)
    PHI = Matrix{Float64}(undef, n, k)
    for b in 1:k
        lam_sqrt = sqrt((b * pi / (2.0 * L))^2)
        for i in 1:n
            PHI[i, b] =
                inv_sqrt_L * sin(lam_sqrt * (op_log_dose[i] - mu + L))
        end
    end
    # Spectral weights (SB `brm_hsgp_sqrt_spd`, 1-D iso: `scale =
    # sigma * sqrt(rho * sqrt(2π))`, `s[b] = scale * exp(-0.25 *
    # rho^2 * lam[b])` — left-assoc, SB order).
    sscale = sigma * sqrt(rho * _PK_EVENT_SQRT2PI)
    S = [sscale * exp(-0.25 * rho * rho * (b * pi / (2.0 * L))^2)
        for b in 1:k]
    w = S .* beta_raw
    smooth = PHI * w
    return slope .* op_log_dose .+ smooth
end

"""
    linear_pk_read_locs(op_type, op_dt, op_amount, op_interval, op_count, op_read_idx,
                        log_F, log_Vc, log_k10, log_k12, log_k21, log_ka)
        -> AbstractVector

One subject's linear-PK read locations (SB `linear_pk_read_locs_cell`,
same op order): derive `log_CL`/`log_Q`/`log_Vp` from the varyingsource4
prior coordinates, build the system, walk the event stream threading
the 3-state, record `state[2]/Vc` at READ ops, return every READ
location.

`log_F` is the event-axis bioavailability LP in SB position
(immediately after `op_read_idx`), one value per op of this subject's
slice: the dose branch is `op_amount[j] * exp(log_F[j])` (SB's dose
branch verbatim), fed by [`linear_pk_event_log_f`](@ref)'s flat
vector. The v1 6-op-column method (no `log_F`) is preserved exactly
(F ≡ 1) as a thin zeros wrapper — bit-identical `exp(0) == 1`.

Called from generated grouped-kernel code (one call per subject over
bound op-column slices plus traced LP scalars — the `gp_chol_latent`
native+Enzyme precedent) and from the host-side oracle path; the
same function on both paths keeps spec and graph bit-identical by
construction.

Native reads accumulate with `push!` in encounter order (`op_read_idx` is
the per-op cumsum of READs), matching SB's counter-write. Traced calls never
execute this host loop: compiled PK recurrences are explicitly unsupported
pending [Reactant reverse control-flow support](https://github.com/nsiccha/ReactiveKernels.jl/issues/13).
The experimental rectangular adapter uses a retained loop and a fixed output
buffer; its private diagnostic opt-in does not enable a supported sampler path.
"""
function linear_pk_read_locs(op_type::AbstractVector,
        op_dt::AbstractVector, op_amount::AbstractVector,
        op_interval::AbstractVector, op_count::AbstractVector,
        op_read_idx::AbstractVector, log_F::AbstractVector, log_Vc, log_k10,
        log_k12, log_k21, log_ka)
    cols = (op_type, op_dt, op_amount, op_interval, op_count, op_read_idx)
    args = (SubjectSlice(log_F), log_Vc, log_k10, log_k12, log_k21, log_ka)
    marker = ReactiveKernels._dynamic_tensorized_marker((cols..., map(_subject_value, args)...))
    marker === nothing || return _pk_compiled_cell(linear_pk_read_locs,
        [length(op_type)], cols, args, marker)
    n_ops = length(op_type)
    (length(op_dt) == n_ops && length(op_amount) == n_ops &&
     length(op_interval) == n_ops && length(op_count) == n_ops &&
     length(op_read_idx) == n_ops && length(log_F) == n_ops) ||
        throw(ArgumentError("linear_pk_read_locs op columns disagree " *
                            "in length (all seven must match)"))
    n_ops >= 1 ||
        throw(ArgumentError("linear_pk_read_locs needs at least one op"))
    log_CL = log_Vc + log_k10
    log_Q = log_Vc + log_k12
    log_Vp = log_Vc + log_k12 - log_k21
    A = linear_pk_system_3(log_CL, log_Vc, log_Q, log_Vp, log_ka)
    Vc = exp(log_Vc)
    state = (0.0, 0.0, 0.0)
    read_locs = zeros(typeof(Vc / Vc), 0)
    for j in 1:n_ops
        state = linear_pk_propagate_3(A, state, op_dt[j])
        if op_type[j] == LINEAR_EVENT_READ
            # Encounter order is read order (see docstring): append.
            push!(read_locs, state[2] / Vc)
        else
            effective_amount = op_amount[j] * exp(_traced_op_read(log_F, j))
            if op_type[j] == LINEAR_EVENT_DOSE
                state = linear_pk_add_dose_3(state, effective_amount)
            else
                op_type[j] == LINEAR_EVENT_DOSE_SEGMENT ||
                    throw(ArgumentError("linear PK kernel event stream " *
                                        "has an unknown operation type " *
                                        "$(op_type[j]) (SB @stan_assert mirror)"))
                state = linear_pk_add_regular_doses_3(A, state,
                    effective_amount, op_interval[j], op_count[j])
            end
        end
    end
    return read_locs
end

"""
    linear_pk_read_locs(op_type, op_dt, op_amount, op_interval, op_count, op_read_idx,
                        log_Vc, log_k10, log_k12, log_k21, log_ka)

The v1 spelling (no `log_F`): bioavailability identically 1. Thin
wrapper over the 7-op-column method with explicit zeros —
bit-identical `exp(0) == 1`, so v1 values never drift from the V2
loop.
"""
function linear_pk_read_locs(op_type::AbstractVector,
        op_dt::AbstractVector, op_amount::AbstractVector,
        op_interval::AbstractVector, op_count::AbstractVector,
        op_read_idx::AbstractVector, log_Vc, log_k10, log_k12, log_k21,
        log_ka)
    return linear_pk_read_locs(op_type, op_dt, op_amount, op_interval,
        op_count, op_read_idx, zeros(length(op_type)), log_Vc, log_k10,
        log_k12, log_k21, log_ka)
end

# SB `linear_pk_read_locs_auc_cell` mirror (bruno mirror ref
# `kb-impl/Bruno-arv393-tgi` @ `3af22846`:
# `web-pkpd/src/brm_joint_tgi.jl`): the base recurrence plus the exact
# mass-balance identity `AUC(0, t) = (given - sum(state(t))) / CL` — no
# quadrature, no extra state. At each read op the running bioavailable
# dose `given` (prior doses only) minus the current body burden, over
# CL. Returns the flat SB `[conc; auc]` reads vector (subject-local
# read numbering, per the file header). Unlike the v1 base, this
# variant takes the SB `log_F` per-op vector (SB position, after
# `op_read_idx`): the event-LP provider (W2 lane) emits it flat and the
# grouped expansion slices per-subject views.
function linear_pk_read_locs_auc(op_type::AbstractVector,
        op_dt::AbstractVector, op_amount::AbstractVector,
        op_interval::AbstractVector, op_count::AbstractVector,
        op_read_idx::AbstractVector, log_F::AbstractVector,
        log_Vc, log_k10, log_k12, log_k21, log_ka)
    cols = (op_type, op_dt, op_amount, op_interval, op_count, op_read_idx)
    args = (SubjectSlice(log_F), log_Vc, log_k10, log_k12, log_k21, log_ka)
    marker = ReactiveKernels._dynamic_tensorized_marker((cols..., map(_subject_value, args)...))
    marker === nothing || return _pk_compiled_cell(linear_pk_read_locs_auc,
        [length(op_type)], cols, args, marker)
    n_ops = length(op_type)
    (length(op_dt) == n_ops && length(op_amount) == n_ops &&
     length(op_interval) == n_ops && length(op_count) == n_ops &&
     length(op_read_idx) == n_ops && length(log_F) == n_ops) ||
        throw(ArgumentError("linear_pk_read_locs_auc op columns disagree " *
                            "in length (all seven must match)"))
    n_ops >= 1 ||
        throw(ArgumentError("linear_pk_read_locs_auc needs at least one op"))
    log_CL = log_Vc + log_k10
    log_Q = log_Vc + log_k12
    log_Vp = log_Vc + log_k12 - log_k21
    A = linear_pk_system_3(log_CL, log_Vc, log_Q, log_Vp, log_ka)
    Vc = exp(log_Vc)
    CL = exp(log_CL)
    state = (0.0, 0.0, 0.0)
    given = 0.0
    conc = zeros(typeof(Vc / Vc), 0)
    auc = zeros(typeof(Vc / Vc), 0)
    for j in 1:n_ops
        state = linear_pk_propagate_3(A, state, op_dt[j])
        if op_type[j] == LINEAR_EVENT_READ
            push!(conc, state[2] / Vc)
            push!(auc, (given - (state[1] + state[2] + state[3])) / CL)
        else
            effective_amount = op_amount[j] * exp(_traced_op_read(log_F, j))
            if op_type[j] == LINEAR_EVENT_DOSE
                state = linear_pk_add_dose_3(state, effective_amount)
                given += effective_amount
            else
                op_type[j] == LINEAR_EVENT_DOSE_SEGMENT ||
                    throw(ArgumentError("linear PK kernel event stream " *
                                        "has an unknown operation type " *
                                        "$(op_type[j]) (SB @stan_assert mirror)"))
                state = linear_pk_add_regular_doses_3(A, state,
                    effective_amount, op_interval[j], op_count[j])
                given += op_count[j] * effective_amount
            end
        end
    end
    return vcat(conc, auc)
end
