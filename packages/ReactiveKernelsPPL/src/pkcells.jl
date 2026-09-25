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
# intermediate 3- and 4-dimensional storage is static
# (`SMatrix`/`SVector`: stack-allocated, immutable): a concrete
# `Matrix{Float64}`/`Vector{Float64}` rejects traced stores
# (`convert(Float64, ::TracedRNumber)` has no method — the pkcell
# slice's measured Reactant failure). The matrix exponential is the
# StaticArrays built-in `exp` on `SMatrix{3,3}` (Higham-2008 Padé,
# no LAPACK balancing — faster and at least as accurate as Stan's
# `matrix_exp_pade` on the PK range, measured 2026-09-25; the former
# hand port was deleted with the user's no-Stan-exactness direction).
# Native Enzyme reverses through it (pure Julia, no ccalls). Only the
# returned reads vector is an array, allocated eltype-generic (see
# `linear_pk_read_locs`).

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

# --- SB primitive mirrors (`src/pkpd_models.jl`, same op order) ---

"""
    linear_pk_system_3(log_CL, log_Vc, log_Q, log_Vp, log_ka) -> SMatrix{3,3}

Three-state first-order-absorption/two-compartment amount-system matrix
(SB `linear_pk_system`, same op order: `exp` the logs, derive the
micro-constants, column-stack) as a static matrix.
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
    return SMatrix{3,3}(-ka, ka, 0.0, 0.0, -(k10 + k12), k12, 0.0, k21,
        -k21)
end

"""
    linear_pk_propagate_3(A, state, dt) -> SVector{3}

Propagate a three-state linear PK system across one interval (SB
`linear_pk_propagate`: identity at `dt <= 0`, else `exp(A*dt)*state`
— the exponential is the StaticArrays built-in).
"""
function linear_pk_propagate_3(A::SMatrix{3,3}, state::SVector{3}, dt)
    if dt > 0
        return _pk_propagate_positive(A, state, dt)
    end
    return state
end

@inline function _pk_propagate_positive(A, state, dt)
    exp(A * dt) * state
end

"""
    linear_pk_add_dose_3(state, amount) -> SVector{3}

One oral dose jump to the gut amount state (SB `linear_pk_add_dose`).
"""
function linear_pk_add_dose_3(state::SVector{3}, amount)
    return SVector(state[1] + amount, state[2], state[3])
end

"""
    linear_pk_add_regular_doses_3(A, state, amount, interval, count) -> SVector{3}

`count` equal oral doses at a fixed interval, beginning now (SB
`linear_pk_add_regular_doses`: immediate first jump, then the augmented
affine transition `[P b; 0 1]^(count-1)` with `P = exp(A*interval)`
and `b = [amount, 0, 0]`).
"""
function linear_pk_add_regular_doses_3(A::SMatrix{3,3}, state::SVector{3},
        amount, interval, count::Integer)
    after_first = linear_pk_add_dose_3(state, amount)
    count > 1 || return after_first
    Q = _pk_smat_pow4(_pk_dose_affine(A, amount, interval), count - 1)
    q = Q * SVector(after_first[1], after_first[2], after_first[3], 1.0)
    return SVector(q[1], q[2], q[3])
end

# The dose-interval affine map from exp(A*interval): propagate, then add a dose.
@inline function _pk_dose_affine(A, amount, interval)
    P = exp(A * interval)
    return SMatrix{4,4}(P[1], P[2], P[3], 0.0, P[4], P[5], P[6], 0.0,
        P[7], P[8], P[9], 0.0, amount, 0.0, 0.0, 1.0)
end

# Binary exponentiation for the 4x4 augmented dose transition (Stan's
# `matrix_power` with a data-only integer exponent — same math).
function _pk_smat_pow4(B::SMatrix{4,4}, n::Integer)
    R = SMatrix{4,4}(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    e = Int(n)
    while e > 0
        if e & 1 == 1
            R = R * B
        end
        B = B * B
        e >>= 1
    end
    return R
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
    state = SVector(0.0, 0.0, 0.0)
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
    state = SVector(0.0, 0.0, 0.0)
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

# --- destination-passing cell variants (non-allocating execution) -----------
#
# `linear_pk_read_locs!`, `linear_pk_read_locs_auc!` and `linear_pk_event_log_f!`
# run the same recurrences as their allocating twins, writing into a caller-owned
# `out` vector instead of `push!`ing into fresh vectors. The MutatingFunctions
# extension calls them from its `apply!!` methods, so a warmed non-allocating
# kernel reuses one result buffer per grouped assignment instead of reallocating
# it on every call.
#
# Native-only: unlike the twins, these skip the tensorized-marker check — the
# extension's `apply!!` performs it once per call and falls back to the allocating
# twin (which routes to the compiled path or throws the documented error) when
# traced values are present. Do not call these with traced operands directly.
#
# Bit-exactness: the recurrence bodies mirror the allocating twins
# statement-for-statement (same op order, same encounter order, same expressions);
# only the sink changes (`out[off + k]` for the k-th READ op instead of `push!`).
# The v1 `linear_pk_read_locs!` spelling (no `log_F`) uses `op_amount[j]` directly:
# the twin computes `op_amount[j] * exp(0)`, and `x * 1.0 === x` in IEEE, so the
# two agree bit-for-bit. `linear_pk_event_log_f!` fuses the PHI/S/w temporaries
# into one accumulation with the same per-read summation order as the allocating
# form's `PHI * w` plus `slope .* d .+ smooth` association, so the two agree to
# floating-point summation association (a few ulp; the joint parity test holds the
# non-allocating path to ≤ 1e-14 relative against the allocating path).

# Native-execution gate for the MutatingFunctions extension: true when no
# argument carries a traced (tensorized) marker, i.e. the destination-passing
# `!` variants below may run. Traced calls fall back to the allocating twin
# (which routes to the compiled path or throws the documented error).
function _pk_no_traced_marker(fixed::Tuple, extra::Tuple)
    ReactiveKernels._dynamic_tensorized_marker((fixed..., extra...)) === nothing
end

function _pk_count_reads(op_type::AbstractVector, lo::Integer, hi::Integer)
    n = 0
    for j in lo:hi
        op_type[j] == LINEAR_EVENT_READ && (n += 1)
    end
    n
end

"""
    linear_pk_read_locs!(out, off, op_type, op_dt, op_amount, op_interval,
                         op_count, op_read_idx, log_F, log_Vc, log_k10, log_k12,
                         log_k21, log_ka)
    linear_pk_read_locs!(out, off, op_type, op_dt, op_amount, op_interval,
                         op_count, op_read_idx, log_Vc, log_k10, log_k12, log_k21,
                         log_ka)

Destination-passing [`linear_pk_read_locs`](@ref): write the subject's READ
locations into `out` at `off + 1:off + R` (`R` = number of READ ops) and return
`out`. The second method is the v1 spelling (bioavailability identically 1).
Validations mirror the allocating twin; `out` must have room for the `R` reads.
"""
function linear_pk_read_locs!(out::AbstractVector, off::Integer,
        op_type::AbstractVector,
        op_dt::AbstractVector, op_amount::AbstractVector,
        op_interval::AbstractVector, op_count::AbstractVector,
        op_read_idx::AbstractVector, log_F::AbstractVector, log_Vc, log_k10,
        log_k12, log_k21, log_ka)
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
    state = SVector(0.0, 0.0, 0.0)
    k = 0
    for j in 1:n_ops
        state = linear_pk_propagate_3(A, state, op_dt[j])
        if op_type[j] == LINEAR_EVENT_READ
            k += 1
            out[off + k] = state[2] / Vc
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
    return out
end

function linear_pk_read_locs!(out::AbstractVector, off::Integer,
        op_type::AbstractVector,
        op_dt::AbstractVector, op_amount::AbstractVector,
        op_interval::AbstractVector, op_count::AbstractVector,
        op_read_idx::AbstractVector, log_Vc, log_k10, log_k12, log_k21,
        log_ka)
    n_ops = length(op_type)
    (length(op_dt) == n_ops && length(op_amount) == n_ops &&
     length(op_interval) == n_ops && length(op_count) == n_ops &&
     length(op_read_idx) == n_ops) ||
        throw(ArgumentError("linear_pk_read_locs op columns disagree " *
                            "in length (all six must match)"))
    n_ops >= 1 ||
        throw(ArgumentError("linear_pk_read_locs needs at least one op"))
    log_CL = log_Vc + log_k10
    log_Q = log_Vc + log_k12
    log_Vp = log_Vc + log_k12 - log_k21
    A = linear_pk_system_3(log_CL, log_Vc, log_Q, log_Vp, log_ka)
    Vc = exp(log_Vc)
    state = SVector(0.0, 0.0, 0.0)
    k = 0
    for j in 1:n_ops
        state = linear_pk_propagate_3(A, state, op_dt[j])
        if op_type[j] == LINEAR_EVENT_READ
            k += 1
            out[off + k] = state[2] / Vc
        else
            effective_amount = op_amount[j]
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
    return out
end

"""
    linear_pk_read_locs_auc!(out, off, op_type, op_dt, op_amount, op_interval,
                             op_count, op_read_idx, log_F, log_Vc, log_k10,
                             log_k12, log_k21, log_ka)

Destination-passing [`linear_pk_read_locs_auc`](@ref): write the subject's
`[conc; auc]` reads vector into `out` at `off + 1:off + 2R` (`R` = number of
READ ops) and return `out`. Validations mirror the allocating twin; `out` must
have room for the `2R` reads.
"""
function linear_pk_read_locs_auc!(out::AbstractVector, off::Integer,
        op_type::AbstractVector,
        op_dt::AbstractVector, op_amount::AbstractVector,
        op_interval::AbstractVector, op_count::AbstractVector,
        op_read_idx::AbstractVector, log_F::AbstractVector,
        log_Vc, log_k10, log_k12, log_k21, log_ka)
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
    state = SVector(0.0, 0.0, 0.0)
    given = 0.0
    n_reads = _pk_count_reads(op_type, 1, n_ops)
    k = 0
    for j in 1:n_ops
        state = linear_pk_propagate_3(A, state, op_dt[j])
        if op_type[j] == LINEAR_EVENT_READ
            k += 1
            out[off + k] = state[2] / Vc
            out[off + n_reads + k] = (given - (state[1] + state[2] + state[3])) / CL
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
    return out
end

"""
    linear_pk_event_log_f!(out, op_log_dose, slope, rho, sigma, beta_raw, mu, L, k)

Destination-passing [`linear_pk_event_log_f`](@ref): write the event-axis
bioavailability LP into `out` (which must have `length(op_log_dose)` entries)
and return `out`. Validations mirror the allocating twin. The PHI/S/w
temporaries are fused into one accumulation with the same per-read summation
order, so values agree with the allocating form to floating-point summation
association.
"""
function linear_pk_event_log_f!(out::AbstractVector,
        op_log_dose::AbstractVector, slope, rho,
        sigma, beta_raw::AbstractVector, mu, L, k::Integer)
    length(beta_raw) == k ||
        throw(ArgumentError("linear_pk_event_log_f needs $k hsgp " *
                            "coefficients (got $(length(beta_raw)))"))
    L > 0 ||
        throw(ArgumentError("linear_pk_event_log_f needs a positive " *
                            "domain half-width L (got $L)"))
    n = length(op_log_dose)
    length(out) == n || throw(DimensionMismatch(
        "linear_pk_event_log_f! destination has length $(length(out)), " *
        "needs $n"))
    inv_sqrt_L = 1.0 / sqrt(L)
    sscale = sigma * sqrt(rho * _PK_EVENT_SQRT2PI)
    for i in 1:n
        out[i] = 0.0
    end
    for b in 1:k
        lam_sq = (b * pi / (2.0 * L))^2
        lam_sqrt = sqrt(lam_sq)
        wb = sscale * exp(-0.25 * rho * rho * lam_sq) * beta_raw[b]
        for i in 1:n
            out[i] += inv_sqrt_L * sin(lam_sqrt * (op_log_dose[i] - mu + L)) * wb
        end
    end
    for i in 1:n
        out[i] = slope * op_log_dose[i] + out[i]
    end
    return out
end
