using DifferentiationInterface
using Distributions
using Enzyme
using LinearAlgebra: exp
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using ReactiveKernelsPPL
using Reactant
using Test

# Grouped emitter per-family dispatch + multi-axis prep bundling (W3c):
# the joint kernel's QT/TGI observation plates and the foreign-axis
# gather machinery they need. R1 union-reads schedule first (it gates
# the gathers), then D1-D9 per the parent brief. Every section is a
# reusable suite, not a probe: hand oracles stay independent
# (Distributions.jl loops, never the emitted code).

const _JE_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

# --- R1 fixture: oracle rows, SB-sorted --------------------------------------
#
# Replicated constants from
# joint-oracle/provenance/joint_pk_qt_fixture.jl (SB prep order: pk by
# (subject, time), qt by (subject, time, ecg_source, i),
# tumor by (subject, time, lesion, i)). Expected op columns + maps are
# the joint-oracle continuous/stan_data.json ragged mem vectors
# (transcribed, grouped by ends in the cross-check test below).

_je_pk_subj() = [1, 1, 1, 1, 2, 2, 2]
_je_pk_time() = [73.0, 150.0, 150.0, 180.0, 0.0, 24.0, 60.0]
_je_dose_subj() = [1, 1, 1, 1, 1, 1, 2, 2, 2]
_je_dose_time() = [0.0, 24.0, 48.0, 96.0, 108.0, 120.0, 0.0, 24.0, 48.0]
_je_dose_amt() =
    [100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 80.0, 80.0, 80.0]
_je_ecg_subj() = [1, 1, 1, 2, 2, 3, 3]
_je_ecg_time() = [-1.0, 25.0, 72.0, 0.0, 24.0, 1.0, 25.0]
_je_tgi_subj() = [1, 1, 1, 2, 2, 2]
_je_tgi_time() = [0.0, 96.0, 170.0, -24.0, 48.0, 100.0]

_je_sched() = build_linear_pk_schedule(_je_pk_subj(), _je_pk_time(),
    _je_dose_subj(), _je_dose_time(), _je_dose_amt();
    combine_simultaneous = false, ecg = (_je_ecg_subj(), _je_ecg_time()),
    tgi = (_je_tgi_subj(), _je_tgi_time()))

@testset "R1 union-reads schedule byte-equals SB op stream" begin
    sched = _je_sched()
    @test sched.n_subjects == 3
    @test sched.n_reads == [9, 6, 2]
    @test sched.n_segments == 1
    @test sched.n_grouped_dose_events == 9
    @test sched.op_ends == [13, 22, 24]
    @test sched.op_type ==
        [1, 1, 2, 2, 1, 2, 1, 1, 1, 3, 1, 1, 1,
            1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1]
    @test sched.op_dt ==
        [0.0, 1.0, 0.0, 24.0, 1.0, 23.0, 24.0, 1.0, 23.0, 0.0, 30.0,
            20.0, 10.0, 0.0, 24.0, 0.0, 24.0, 0.0, 24.0, 0.0, 12.0,
            40.0, 0.0, 24.0]
    @test sched.op_amount ==
        [0.0, 0.0, 100.0, 100.0, 0.0, 100.0, 0.0, 0.0, 0.0, 50.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 80.0, 0.0, 80.0, 0.0, 80.0, 0.0, 0.0,
            0.0, 0.0]
    want_interval = zeros(24)
    want_interval[10] = 12.0
    @test sched.op_interval == want_interval
    @test sched.op_count ==
        [0, 0, 1, 1, 0, 1, 0, 0, 0, 3, 0, 0, 0,
            0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0]
    @test sched.op_read_idx ==
        [1, 2, 0, 0, 3, 0, 4, 5, 6, 0, 7, 8, 9,
            1, 2, 0, 3, 0, 4, 0, 5, 6, 1, 2]
    # The DOSE_SEGMENT time jump: S1's segment @96 covers doses
    # 96/108/120, so post-segment delays measure from the segment END
    # (read @150 lands 30h after 120, not 54h after 96).
    @test sched.op_type[10] == LINEAR_EVENT_DOSE_SEGMENT
    @test sched.op_amount[10] == 50.0
    @test sched.op_interval[10] == 12.0
    @test sched.op_count[10] == 3
    @test sched.op_dt[11] == 30.0
    # Read-before-dose at shared times (the read sees the pre-dose state).
    @test sched.op_type[2:3] == [1, 2]
    @test sched.op_dt[3] == 0.0
    @test sched.op_type[15:16] == [1, 2]
    @test sched.op_dt[16] == 0.0
    # Negative event times survive (ecg -1, tgi -24 open their streams).
    @test sched.op_dt[1] == 0.0
    @test sched.op_dt[14] == 0.0
    # Per-axis row products (SB's per-axis slicing, caller row order).
    @test sched.obs_read == [5, 7, 7, 9, 2, 3, 5]
    @test sched.obs_map == [5, 7, 7, 9, 11, 12, 14]
    @test sched.ecg_read == [1, 3, 4, 2, 3, 1, 2]
    @test sched.ecg_map == [1, 3, 4, 11, 12, 16, 17]
    @test sched.tgi_read == [2, 6, 8, 1, 4, 6]
    @test sched.tgi_map == [2, 6, 8, 10, 13, 15]
    # [conc; auc]-space maps (SB's tgi_conc_idx/tgi_auc_idx globalized
    # via the read offsets: S1 +0, S2 +18, S3 +30 in [conc; auc]
    # space).
    @test sched.conc_map ==
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 20, 21, 22, 23, 24, 31, 32]
    @test sched.tgi_auc_map == [11, 15, 17, 25, 28, 30]
    @test sched.n_reads_total == 17
end

@testset "R1 derived maps cross-check SB read indices exactly" begin
    # SB per-subject index vectors (oracle ragged mem, grouped by the
    # ends) globalize with SB's own read-offset rule; the derived flat
    # maps must match exactly.
    sched = _je_sched()
    sb_pk = [[5, 7, 7, 9], [2, 3, 5], Int[]]
    sb_ecg = [[1, 3, 4], [2, 3], [1, 2]]
    sb_tgi_local = [[2, 6, 8], [1, 4, 6], Int[]]
    sb_tgi_auc = [[11, 15, 17], [7, 10, 12], Int[]]
    sb_tgi_conc = [collect(1:9), collect(1:6), [1, 2]]
    off1 = [0, 9, 15]
    off2 = [0, 18, 30]
    want_pk = vcat([sb_pk[s] .+ off1[s] for s in 1:3]...)
    want_ecg = vcat([sb_ecg[s] .+ off1[s] for s in 1:3]...)
    want_tgi = vcat([sb_tgi_local[s] .+ off1[s] for s in 1:2]...)
    want_auc = vcat([sb_tgi_auc[s] .+ off2[s] for s in 1:2]...)
    want_conc = vcat([sb_tgi_conc[s] .+ off2[s] for s in 1:3]...)
    @test sched.obs_map == want_pk
    @test sched.ecg_map == want_ecg
    @test sched.tgi_map == want_tgi
    @test sched.tgi_auc_map == want_auc
    @test sched.conc_map == want_conc
end

@testset "R1 dose-free subject keeps a read-only stream" begin
    # Subject 2 has no PK rows and no dose rows — only ECG rows — and
    # still gets its stream (SB's dose-free subject precedent).
    sched = build_linear_pk_schedule([1, 1], [10.0, 20.0], [1],
        [0.0], [100.0]; ecg = ([1, 2, 2], [15.0, 1.0, 25.0]))
    @test sched.n_subjects == 2
    @test sched.n_reads == [3, 2]
    @test sched.op_ends == [4, 6]
    @test sched.op_type[5:6] == [1, 1]
    @test sched.op_amount[5:6] == [0.0, 0.0]
    @test sched.op_read_idx[5:6] == [1, 2]
    @test sched.obs_read == [1, 3]
    @test sched.obs_map == [1, 3]
    @test sched.ecg_read == [2, 1, 2]
    @test sched.ecg_map == [2, 4, 5]
    @test isempty(sched.tgi_read)
    @test isempty(sched.tgi_map)
    @test isempty(sched.tgi_auc_map)
    # Builds without extra axes carry empty axis products.
    plain = build_linear_pk_schedule([1, 1], [10.0, 20.0], [1],
        [0.0], [100.0])
    @test isempty(plain.ecg_read)
    @test isempty(plain.ecg_map)
    @test isempty(plain.tgi_read)
    @test isempty(plain.tgi_map)
    @test isempty(plain.tgi_auc_map)
    @test plain.conc_map == [1, 2]
end

@testset "R1 builder fail-closed battery" begin
    kw = (; combine_simultaneous = false)
    @test_throws "ecg axis subject/time lengths differ" build_linear_pk_schedule(
        [1], [1.0], [1], [0.0], [10.0]; kw...,
        ecg = ([1, 2], [1.0]))
    @test_throws "tgi axis subject/time lengths differ" build_linear_pk_schedule(
        [1], [1.0], [1], [0.0], [10.0]; kw...,
        tgi = ([1], [1.0, 2.0]))
    @test_throws "ecg axis subject IDs must be positive" build_linear_pk_schedule(
        [1], [1.0], [1], [0.0], [10.0]; kw...,
        ecg = ([0], [1.0]))
    @test_throws "tgi axis times must be finite" build_linear_pk_schedule(
        [1], [1.0], [1], [0.0], [10.0]; kw...,
        tgi = ([1], [Inf]))
    # Union contiguity: subject 2 missing from every read axis.
    @test_throws "contiguous" build_linear_pk_schedule(
        [1], [1.0], [1], [0.0], [10.0]; kw...,
        tgi = ([3], [5.0]))
    # Doses must reference union subjects.
    @test_throws "observed subjects" build_linear_pk_schedule(
        [1], [1.0], [2], [0.0], [10.0]; kw...,
        ecg = ([1], [2.0]))
    # Empty primary axis is legal with a non-empty extra axis; an
    # empty union still fails.
    empty_pk = build_linear_pk_schedule(Int[], Float64[], Int[],
        Float64[], Float64[]; kw..., ecg = ([1], [2.0]))
    @test empty_pk.n_subjects == 1
    @test empty_pk.n_reads == [1]
    @test_throws "at least one observation" build_linear_pk_schedule(
        Int[], Float64[], Int[], Float64[], Float64[]; kw...)
end

const _JE_DATA = Set([:subj, :time, :dsubj, :dtime, :damt, :dv, :age_s,
    :esubj, :etime, :tsubj, :ttime])

_je_lp_defs(decl = "pk_sched = linear_pk_schedule(obs = (:subj, :time), " *
    "dose = (:dsubj, :dtime, :damt), ecg = (:esubj, :etime), " *
    "tgi = (:tsubj, :ttime))") = quote
    sigma ~ Exponential(1.0)
    b0_vc ~ Normal(0.0, 1.0)
    b1_vc ~ Normal(0.0, 1.0)
    log_Vc = b0_vc .+ b1_vc .* age_s
    $(Meta.parse(decl))
end

_je_kernel_cell() = quote
    read_locs =
        linear_pk_read_locs(pk_sched, log_Vc, log_Vc, log_Vc, log_Vc, log_Vc)
    mu = read_locs[pk_sched.obs_map]
    e = read_locs[pk_sched.ecg_map]
    t = read_locs[pk_sched.tgi_map]
    dv .~ Normal.(mu, sigma)
    mu
end

function _je_ast(cell = _je_kernel_cell(), decl = nothing)
    pre = decl === nothing ? _je_lp_defs() : _je_lp_defs(decl)
    foro = Expr(:for, Expr(:(=), :s, Expr(:call, :(:), 1, 2)),
        Expr(:block, cell.args...))
    ker = Expr(:macrocall, Symbol("@plate"), LineNumberNode(0), :conc, foro)
    return Expr(:block, pre.args..., ker)
end

function _je_columns()
    Dict{Symbol,AbstractVector}(
        :subj => [1, 1, 2], :time => [10.0, 20.0, 5.0],
        :dsubj => [1, 2], :dtime => [0.0, 0.0],
        :damt => [100.0, 50.0], :dv => [1.0, 2.0, 3.0],
        :age_s => [30.0, 40.0],
        :esubj => [1, 2, 2], :etime => [15.0, 0.0, 8.0],
        :tsubj => [2], :ttime => [12.0])
end

@testset "R1 surface ecg/tgi keywords" begin
    schedcase = (decl) -> lower_rkppl(
        Expr(:block, Meta.parse("begin\n$decl\nend").args...,
            Meta.parse("begin\nsigma ~ Exponential(1.0)\nend").args...),
        _JE_DATA)
    @test_throws "keywords only" schedcase(
        "pk_sched = linear_pk_schedule(obs = (:subj, :time), " *
        "dose = (:dsubj, :dtime, :damt), extra = 1)")
    @test_throws "repeats `ecg=`" schedcase(
        "pk_sched = linear_pk_schedule(obs = (:subj, :time), " *
        "dose = (:dsubj, :dtime, :damt), ecg = (:esubj, :etime), " *
        "ecg = (:esubj, :etime))")
    @test_throws "repeats `tgi=`" schedcase(
        "pk_sched = linear_pk_schedule(obs = (:subj, :time), " *
        "dose = (:dsubj, :dtime, :damt), tgi = (:tsubj, :ttime), " *
        "tgi = (:tsubj, :ttime))")
    @test_throws "is not bound data" schedcase(
        "pk_sched = linear_pk_schedule(obs = (:subj, :time), " *
        "dose = (:dsubj, :dtime, :damt), ecg = (:esubj, :nope))")
    @test_throws "2-tuple" schedcase(
        "pk_sched = linear_pk_schedule(obs = (:subj, :time), " *
        "dose = (:dsubj, :dtime, :damt), tgi = (:tsubj,))")
    # Success shape: the declared axes land on the spec.
    plan = lower_rkppl(_je_ast(), _JE_DATA)
    spec = only(only(plan.kernel_plates).schedules)
    @test spec.ecg == (:esubj, :etime)
    @test spec.tgi == (:tsubj, :ttime)
end

@testset "R1 bind materializes axis products" begin
    unbound = lower_rkppl(_je_ast(), _JE_DATA)
    bound = bind_data(unbound, _je_columns())
    @test bound.n_obs == 3
    @test only(bound.kernel_plates).subjects == 2
    sched_cols = filter(c -> startswith(String(c), "pk_sched_"),
        keys(bound.columns))
    # v1 nine + (ecg_read, ecg_map, tgi_read, tgi_map); no
    # [conc; auc]-space maps without an AUC cell call.
    @test length(sched_cols) == 13
    @test !haskey(bound.columns, :pk_sched_conc_map)
    @test !haskey(bound.columns, :pk_sched_tgi_auc_map)
    @test bound.columns[:pk_sched_ecg_read] == [2, 1, 3]
    @test bound.columns[:pk_sched_ecg_map] == [2, 4, 6]
    @test bound.columns[:pk_sched_tgi_read] == [4]
    @test bound.columns[:pk_sched_tgi_map] == [7]
    built = build_kernel(bound)
    names = coordinate_names(assign_layout(bound))
    u = zeros(length(names))
    @test isfinite(prepare_query(built, bound, :sampler)(u))
    # Extra products verify by exact rebuild, never trusted.
    tampered = bind_data(unbound, _je_columns())
    tampered.columns[:pk_sched_ecg_map] .= 1
    @test_throws "is not the schedule build" validate_data(tampered)
    # A caller collision on an extra product fails closed.
    collision = _je_columns()
    collision[:pk_sched_tgi_map] = [1]
    @test_throws "is reserved for schedule" bind_data(unbound, collision)
    # A missing extra-axis column fails closed.
    missing_col = _je_columns()
    delete!(missing_col, :etime)
    @test_throws "is not bound" bind_data(unbound, missing_col)
end

@testset "R1 undeclared-axis gathers fail closed" begin
    noecg = "pk_sched = linear_pk_schedule(obs = (:subj, :time), " *
        "dose = (:dsubj, :dtime, :damt), tgi = (:tsubj, :ttime))"
    @test_throws "is not available" lower_rkppl(_je_ast(_je_kernel_cell(),
        noecg), _JE_DATA)
    notgi = "pk_sched = linear_pk_schedule(obs = (:subj, :time), " *
        "dose = (:dsubj, :dtime, :damt), ecg = (:esubj, :etime))"
    @test_throws "is not available" lower_rkppl(_je_ast(_je_kernel_cell(),
        notgi), _JE_DATA)
    conc_cell = quote
        read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_Vc, log_Vc,
            log_Vc, log_Vc)
        c = read_locs[pk_sched.conc_map]
        mu = read_locs[pk_sched.obs_map]
        dv .~ Normal.(mu, sigma)
        mu
    end
    @test_throws "needs an AUC cell call" lower_rkppl(
        _je_ast(conc_cell), _JE_DATA)
end

@testset "R1 schedule-map availability units" begin
    has_auc = Pair{Symbol,Any}[:pk_reads =>
        :(linear_pk_read_locs_auc(s, l, a, b, c, d, e))]
    no_auc = Pair{Symbol,Any}[:reads =>
        :(linear_pk_read_locs(s, a, b, c, d, e))]
    @test ReactiveKernelsPPL._cell_has_auc_call(has_auc)
    @test !ReactiveKernelsPPL._cell_has_auc_call(no_auc)
    @test !ReactiveKernelsPPL._cell_has_auc_call(Pair{Symbol,Any}[])
    v1 = LinearPKScheduleSpec(:s, :a, :b, :c, :d, :e)
    @test v1.ecg === nothing
    @test v1.tgi === nothing
    @test ReactiveKernelsPPL._sched_available_maps(v1, no_auc) ==
        Set([:obs_map])
    @test isempty(ReactiveKernelsPPL._sched_extra_fields(v1, no_auc))
    full = LinearPKScheduleSpec(:s, :a, :b, :c, :d, :e, (:f, :g),
        (:h, :i))
    @test ReactiveKernelsPPL._sched_available_maps(full, no_auc) ==
        Set([:obs_map, :ecg_map, :tgi_map])
    @test ReactiveKernelsPPL._sched_extra_fields(full, no_auc) ==
        [:ecg_read, :ecg_map, :tgi_read, :tgi_map]
    @test ReactiveKernelsPPL._sched_available_maps(full, has_auc) ==
        Set([:obs_map, :ecg_map, :tgi_map, :conc_map, :tgi_auc_map])
    @test ReactiveKernelsPPL._sched_extra_fields(full, has_auc) ==
        [:ecg_read, :ecg_map, :tgi_read, :tgi_map, :conc_map, :tgi_auc_map]
    ecg_only = LinearPKScheduleSpec(:s, :a, :b, :c, :d, :e, (:f, :g),
        nothing)
    @test ReactiveKernelsPPL._sched_extra_fields(ecg_only, has_auc) ==
        [:ecg_read, :ecg_map, :conc_map]
end

# --- D1: AUC cell registration -----------------------------------------------
#
# Independent `[conc; auc]` recurrence oracle (fresh from the SB math:
# LinearAlgebra matrix exponential, own event loop, the mass-balance
# AUC identity — never the emitted code). Reused by the D9 joint e2e.

function _je_oracle_system(lVc, lk10, lk12, lk21, lka)
    CL, Vc, Q, Vp, ka =
        exp.((lVc + lk10, lVc, lVc + lk12, lVc + lk12 - lk21, lka))
    k10, k12, k21 = CL / Vc, Q / Vc, Q / Vp
    return [-ka 0.0 0.0; ka -(k10 + k12) k21; 0.0 k12 -k21]
end

function _je_oracle_auc_reads(sched, s, log_F, lVc, lk10, lk12, lk21, lka)
    lo = s == 1 ? 1 : sched.op_ends[s-1] + 1
    hi = sched.op_ends[s]
    A = _je_oracle_system(lVc, lk10, lk12, lk21, lka)
    Vc = exp(lVc)
    CL = exp(lVc + lk10)
    R = maximum(sched.op_read_idx[lo:hi])
    out = zeros(2R)
    state = zeros(3)
    given = 0.0
    for j in lo:hi
        dt = sched.op_dt[j]
        dt > 0 && (state = exp(A * dt) * state)
        t = sched.op_type[j]
        if t == LINEAR_EVENT_READ
            r = sched.op_read_idx[j]
            out[r] = state[2] / Vc
            out[R + r] = (given - sum(state)) / CL
        elseif t == LINEAR_EVENT_DOSE
            eff = sched.op_amount[j] * exp(log_F[j - lo + 1])
            state = copy(state)
            state[1] += eff
            given += eff
        else
            @assert t == LINEAR_EVENT_DOSE_SEGMENT
            eff = sched.op_amount[j] * exp(log_F[j - lo + 1])
            st = copy(state)
            st[1] += eff
            P = exp(A * sched.op_interval[j])
            aff = [P [eff, 0.0, 0.0]; 0.0 0.0 0.0 1.0]
            aug = aff^(sched.op_count[j] - 1) * [st; 1.0]
            state = aug[1:3]
            given += sched.op_count[j] * eff
        end
    end
    return out
end

_je_auc_data() = Set([:subj, :time, :dsubj, :dtime, :damt, :dv])

function _je_auc_ast()
    Meta.parse("""begin
        sigma ~ Exponential(1.0)
        b0_vc ~ Normal(0.0, 1.0)
        log_Vc = b0_vc
        pk_sched = linear_pk_schedule(obs = (:subj, :time),
            dose = (:dsubj, :dtime, :damt))
        log_F = linear_pk_log_f(pk_sched; k = 5)
        @plate conc for s in 1:1
            pk_reads = linear_pk_read_locs_auc(pk_sched, log_F, log_Vc,
                log_Vc, log_Vc, log_Vc, log_Vc)
            mu = pk_reads[pk_sched.obs_map]
            dv .~ Normal.(mu, sigma)
            mu
        end
    end""")
end

function _je_auc_columns()
    # Two distinct dose amounts: a constant op_log_dose column has no
    # usable event-LP domain (L == 0 fails closed at bind).
    Dict{Symbol,AbstractVector}(:subj => [1, 1], :time => [10.0, 20.0],
        :dsubj => [1, 1], :dtime => [0.0, 12.0], :damt => [100.0, 50.0],
        :dv => [1.0, 2.0])
end

@testset "D1 AUC cell binds + expands end to end" begin
    unbound = lower_rkppl(_je_auc_ast(), _je_auc_data())
    kp = only(unbound.kernel_plates)
    @test ReactiveKernelsPPL._cell_has_auc_call(kp.assignments)
    bound = bind_data(unbound, _je_auc_columns())
    @test haskey(bound.columns, :pk_sched_op_log_dose)
    @test haskey(bound.columns, :pk_sched_conc_map)
    @test !haskey(bound.columns, :pk_sched_tgi_auc_map)
    built = build_kernel(bound)
    layout = assign_layout(bound)
    r = repr(kernel_expr(bound, layout))
    @test occursin("linear_pk_read_locs_auc", r)
    # Expanded over op-column slices — never verbatim (a verbatim
    # `pk_sched` schedule handle has no runtime binding).
    @test occursin("pk_sched_op_type", r)
    @test occursin("view", r)
    names = coordinate_names(layout)
    u = zeros(length(names))
    @test isfinite(prepare_query(built, bound, :sampler)(u))
    # Value vs the independent oracle (u = 0 ⇒ all LPs 0, sigma 1,
    # event-LP identically 0; single subject so obs_map ⊂ 1..R).
    cols = _je_auc_columns()
    sched = build_linear_pk_schedule(cols[:subj], cols[:time],
        cols[:dsubj], cols[:dtime], cols[:damt];
        combine_simultaneous = false)
    R = sched.n_reads[1]
    oracle = _je_oracle_auc_reads(sched, 1, zeros(sched.op_ends[1]),
        0.0, 0.0, 0.0, 0.0, 0.0)
    want_like = sum(logpdf.(Normal.(oracle[1:R][sched.obs_map], 1.0),
        cols[:dv]))
    @test prepare_query(built, bound, :likelihood)(u) ≈ want_like atol = 1e-9
end

@testset "D1 AUC call shape fails closed" begin
    # LP-less 6-arg AUC call (no event-LP decl — the contract arity pin).
    no_elp = Meta.parse("""begin
        sigma ~ Exponential(1.0)
        b0_vc ~ Normal(0.0, 1.0)
        log_Vc = b0_vc
        pk_sched = linear_pk_schedule(obs = (:subj, :time),
            dose = (:dsubj, :dtime, :damt))
        @plate conc for s in 1:1
            pk_reads = linear_pk_read_locs_auc(pk_sched, log_Vc, log_Vc,
                log_Vc, log_Vc, log_Vc)
            mu = pk_reads[pk_sched.obs_map]
            dv .~ Normal.(mu, sigma)
            mu
        end
    end""")
    @test_throws "takes 7 arguments" lower_rkppl(no_elp, _je_auc_data())
    # Non-provider second arg, hand-spliced (surface rejects this shape
    # first, so the contract layer is proved directly).
    good = lower_rkppl(_je_auc_ast(), _je_auc_data())
    kp = only(good.kernel_plates)
    badassigns = deepcopy(kp.assignments)
    for (_, ex) in badassigns
        ex isa Expr && ex.head === :call &&
            ex.args[1] === :linear_pk_read_locs_auc && (ex.args[3] = :log_Vc)
    end
    kpbad = KernelPlate(kp.result, kp.subjects, kp.timepoints, kp.slices,
        badassigns, kp.obs, kp.collected, kp.label, kp.lp_args, kp.schedules)
    badcall = StructuralPlan(good.responses, good.predictors,
        good.population_priors, good.parameters, good.assignments,
        good.columns, good.n_obs; roles = good.roles, derived = good.derived,
        levelmaps = good.levelmaps,
        plate_parameters = good.plate_parameters, scans = good.scans,
        varying_draws = good.varying_draws,
        varying_slices = good.varying_slices,
        vector_parameters = good.vector_parameters,
        spline_bases = good.spline_bases,
        spline_vectors = good.spline_vectors, hsgp_bases = good.hsgp_bases,
        kernel_plates = KernelPlate[kpbad],
        r2d2_priors = good.r2d2_priors, matrices = good.matrices,
        event_lps = good.event_lps)
    @test_throws "second argument must be the event-LP" validate_structure(badcall)
end

# --- D2/D3: prep-map gathers + axis-length shapes ------------------------------

_je_gather_cellcase(cell) = lower_rkppl(_je_ast(Meta.parse(
    "begin\n$cell\nend")), _JE_DATA)

const _JE_FULLCALL = "read_locs = linear_pk_read_locs(pk_sched, log_Vc, " *
    "log_Vc, log_Vc, log_Vc, log_Vc)"

@testset "D2 bare LP uses fail closed" begin
    good_tail = "mu = read_locs[pk_sched.obs_map]\n" *
        "dv .~ Normal.(mu, sigma)\nmu"
    @test_throws "gather explicitly" _je_gather_cellcase(
        "$_JE_FULLCALL\nx = log_Vc + 1.0\n$good_tail")
    # Dotted + nested verbatim positions fail the same way.
    @test_throws "gather explicitly" _je_gather_cellcase(
        "$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "x = log_Vc .* mu\ndv .~ Normal.(mu, sigma)\nmu")
    @test_throws "gather explicitly" _je_gather_cellcase(
        "$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "x = ifelse.(log_Vc .> 0, mu, mu)\n" *
        "dv .~ Normal.(mu, sigma)\nmu")
    @test_throws "gather explicitly" _je_gather_cellcase(
        "read_locs = linear_pk_read_locs(pk_sched, log_Vc + 1, log_Vc, " *
        "log_Vc, log_Vc, log_Vc)\n$good_tail")
    # LP cell params as obs args fail (gather to rows first).
    @test_throws "do not lower as obs args" _je_gather_cellcase(
        "$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "dv .~ Normal.(log_Vc, sigma)\nmu")
    @test_throws "do not lower as obs args" _je_gather_cellcase(
        "$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "dv .~ CensoredAddpropnormal.(mu, sigma, log_Vc, sigma)\nmu")
end

@testset "D2 gather index shapes fail closed" begin
    good_obs = "dv .~ Normal.(mu, sigma)\nmu"
    @test_throws "names a cell/model value" _je_gather_cellcase(
        "$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "x = read_locs[mu]\n$good_obs")
    @test_throws "names a cell/model value" _je_gather_cellcase(
        "$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "x = read_locs[sigma]\n$good_obs")
    @test_throws "names a cell/model value" _je_gather_cellcase(
        "$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "x = read_locs[log_Vc]\n$good_obs")
    @test_throws "compile-time handle" _je_gather_cellcase(
        "$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "x = read_locs[pk_sched]\n$good_obs")
    @test_throws "takes one index" _je_gather_cellcase(
        "$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "x = read_locs[1, 2]\n$good_obs")
    # A putative bind column passes structure; bind proves bound-ness.
    unbound = _je_gather_cellcase(
        "$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "e = read_locs[nope]\n$good_obs")
    @test_throws "is not bound" bind_data(unbound, _je_columns())
end

@testset "D2 gather rewrite units" begin
    spec = LinearPKScheduleSpec(:pk_sched, :subj, :time, :dsubj, :dtime,
        :damt)
    rw = ReactiveKernelsPPL._rewrite_grouped_gather
    lps = Dict{Symbol,Symbol}(:s_vc => :lpvec)
    # LP gather sources rewrite to their LP vectors.
    @test rw(:(s_vc[tsubj]), spec, lps) == :(lpvec[tsubj])
    @test rw(:(s_vc[pk_sched.obs_map]), spec, lps) ==
        :(lpvec[pk_sched_obs_map])
    # Bare LPs elsewhere pass through (structure rejects them; the
    # generator never silently vectors them).
    @test rw(:s_vc, spec, lps) == :s_vc
    @test rw(:(s_vc + 1), spec, lps) == :(s_vc + 1)
    # Non-LP traffic is unchanged with an lps dict present.
    @test rw(:(reads[pk_sched.obs_map]), spec, lps) ==
        :(reads[pk_sched_obs_map])
    @test rw(:(pk_reads[tgi_read_map]), spec, lps) ==
        :(pk_reads[tgi_read_map])
    # Slice do-params rewrite to their bound columns (kernel ports are
    # column names); the LP rule still leads in gather position.
    fm = Dict{Symbol,Symbol}(:ww => :qt_w)
    @test rw(:ww, spec, lps, fm) == :qt_w
    @test rw(:(qt_scale .* ww), spec, lps, fm) == :(qt_scale .* qt_w)
    @test rw(:(s_vc[m]), spec, lps, fm) == :(lpvec[m])
end

@testset "D3 axis-length shapes fail closed" begin
    # pk axis (3 rows) vs ecg axis (2 rows): mixing fails naming both.
    cols2 = _je_columns()
    cols2[:esubj] = [1, 2]
    cols2[:etime] = [15.0, 0.0]
    cellcase = (cell) -> lower_rkppl(_je_ast(Meta.parse(
        "begin\n$cell\nend")), _JE_DATA)
    bindshapes = (cell) -> bind_data(cellcase(cell), cols2)
    @test_throws "mixes obs axes of length 2 and 3" bindshapes(
        "$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "e = read_locs[pk_sched.ecg_map]\n" *
        "x = mu .+ e\ndv .~ Normal.(mu, sigma)\nmu")
    @test_throws "has length 2 ≠ response `dv` length 3" bindshapes(
        "$_JE_FULLCALL\ne = read_locs[pk_sched.ecg_map]\n" *
        "dv .~ Normal.(e, sigma)\ne")
    # Read-space and scalar gather sources fail at shape.
    @test_throws "is read-space" bindshapes(
        "$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "dv .~ Normal.(read_locs, sigma)\nmu")
    @test_throws "is scalar" bindshapes(
        "$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "x = sigma[pk_sched.obs_map]\ndv .~ Normal.(mu, sigma)\nmu")
    # Same-axis dotted arithmetic + scalar broadcast still bind.
    ok = bindshapes("$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "e = read_locs[pk_sched.ecg_map]\n" *
        "x = e .* 2.0 .+ sigma\ndv .~ Normal.(mu, sigma)\nmu")
    @test ok.n_obs == 3
end

function _je_gather_columns()
    Dict{Symbol,AbstractVector}(
        :subj => [1, 1, 2, 2], :time => [10.0, 20.0, 5.0, 8.0],
        :dsubj => [1, 2], :dtime => [0.0, 0.0],
        :damt => [100.0, 50.0],
        :dv => [1.0, 2.0, 3.0, 4.0], :dv2 => [1.5, 2.5, 3.5, 4.5],
        :age_s => [30.0, 40.0], :revmap => [4, 3, 2, 1])
end

function _je_gather_ast()
    Meta.parse("""begin
        sigma ~ Exponential(1.0)
        sigma2 ~ Exponential(1.0)
        b0_vc ~ Normal(0.0, 1.0)
        b1_vc ~ Normal(0.0, 1.0)
        log_Vc = b0_vc .+ b1_vc .* age_s
        pk_sched = linear_pk_schedule(obs = (:subj, :time),
            dose = (:dsubj, :dtime, :damt))
        @plate conc for s in 1:2
            read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_Vc,
                log_Vc, log_Vc, log_Vc)
            mu = read_locs[pk_sched.obs_map]
            bump = log_Vc[subj]
            mu2 = mu .+ bump
            rev = read_locs[revmap]
            dv .~ Normal.(mu2, sigma)
            dv2 .~ Normal.(rev, sigma2)
            mu2
        end
    end""")
end

_je_gather_data() =
    Set([:subj, :time, :dsubj, :dtime, :damt, :dv, :dv2, :age_s, :revmap])

function _je_gather_u(names, sig, sig2, b0, b1)
    map(names) do n
        n === :sigma && return log(sig)
        n === :sigma2 && return log(sig2)
        startswith(String(n), "log_Vc.") ||
            error("unexpected coordinate $n")
        return endswith(String(n), ".Intercept") ? b0 : b1
    end
end

@testset "D2/D3 LP + prep-map gathers end to end" begin
    unbound = lower_rkppl(_je_gather_ast(), _je_gather_data())
    cols = _je_gather_columns()
    bound = bind_data(unbound, cols)
    @test bound.n_obs == 4
    built = build_kernel(bound)
    names = coordinate_names(assign_layout(bound))
    sig, sig2, b0, b1 = 0.5, 0.7, 2.0, 0.01
    u = _je_gather_u(names, sig, sig2, b0, b1)
    lps = b0 .+ b1 .* cols[:age_s]
    sched = build_linear_pk_schedule(cols[:subj], cols[:time],
        cols[:dsubj], cols[:dtime], cols[:damt])
    # v1 conc is the AUC oracle's conc half at F ≡ 1 (bit-identical
    # `exp(0) == 1` — the zeros-wrapper precedent).
    conc = vcat([_je_oracle_auc_reads(sched, s,
        zeros(s == 1 ? sched.op_ends[1] :
            sched.op_ends[s] - sched.op_ends[s-1]),
        lps[s], lps[s], lps[s], lps[s], lps[s])[1:sched.n_reads[s]]
        for s in 1:2]...)
    muo = conc[sched.obs_map]
    mu2 = muo .+ lps[cols[:subj]]
    rev = conc[cols[:revmap]]
    want_like = sum(logpdf.(Normal.(mu2, sig), cols[:dv])) +
        sum(logpdf.(Normal.(rev, sig2), cols[:dv2]))
    want = want_like + logpdf(Exponential(1.0), sig) +
        logpdf(Exponential(1.0), sig2) + log(sig) + log(sig2) +
        logpdf(Normal(0.0, 1.0), b0) + logpdf(Normal(0.0, 1.0), b1)
    sampler = prepare_query(built, bound, :sampler)
    @test sampler(u) ≈ want atol = 1e-9
    @test prepare_query(built, bound, :likelihood)(u) ≈ want_like atol = 1e-9
end

# --- D4: foreign-axis response slices ------------------------------------------

function _je_foreign_columns()
    Dict{Symbol,AbstractVector}(
        :subj => [1, 1, 2, 2], :time => [10.0, 20.0, 5.0, 8.0],
        :dsubj => [1, 2], :dtime => [0.0, 0.0],
        :damt => [100.0, 50.0],
        :dv => [1.0, 2.0, 3.0, 4.0], :qy => [0.1, 0.2, 0.3],
        :age_s => [30.0, 40.0],
        :esubj => [1, 1, 2], :etime => [15.0, 25.0, 3.0])
end

function _je_foreign_ast()
    Meta.parse("""begin
        sigma ~ Exponential(1.0)
        sigma2 ~ Exponential(1.0)
        b0_vc ~ Normal(0.0, 1.0)
        b1_vc ~ Normal(0.0, 1.0)
        log_Vc = b0_vc .+ b1_vc .* age_s
        pk_sched = linear_pk_schedule(obs = (:subj, :time),
            dose = (:dsubj, :dtime, :damt), ecg = (:esubj, :etime))
        @plate conc for s in 1:2
            read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_Vc,
                log_Vc, log_Vc, log_Vc)
            mu = read_locs[pk_sched.obs_map]
            e = read_locs[pk_sched.ecg_map]
            dv .~ Normal.(mu, sigma)
            qy .~ Normal.(e, sigma2)
            mu
        end
    end""")
end

_je_foreign_data() = Set([:subj, :time, :dsubj, :dtime, :damt, :dv, :qy,
    :age_s, :esubj, :etime])

@testset "D4 foreign-axis response binds + values" begin
    unbound = lower_rkppl(_je_foreign_ast(), _je_foreign_data())
    cols = _je_foreign_columns()
    bound = bind_data(unbound, cols)
    # n_obs stays the primary (first-obs) response length.
    @test bound.n_obs == 4
    built = build_kernel(bound)
    names = coordinate_names(assign_layout(bound))
    sig, sig2, b0, b1 = 0.5, 0.7, 2.0, 0.01
    u = _je_gather_u(names, sig, sig2, b0, b1)
    lps = b0 .+ b1 .* cols[:age_s]
    sched = build_linear_pk_schedule(cols[:subj], cols[:time],
        cols[:dsubj], cols[:dtime], cols[:damt];
        ecg = (cols[:esubj], cols[:etime]))
    conc = vcat([_je_oracle_auc_reads(sched, s,
        zeros(s == 1 ? sched.op_ends[1] :
            sched.op_ends[s] - sched.op_ends[s-1]),
        lps[s], lps[s], lps[s], lps[s], lps[s])[1:sched.n_reads[s]]
        for s in 1:2]...)
    want_like = sum(logpdf.(Normal.(conc[sched.obs_map], sig),
        cols[:dv])) + sum(logpdf.(Normal.(conc[sched.ecg_map], sig2),
        cols[:qy]))
    want = want_like + logpdf(Exponential(1.0), sig) +
        logpdf(Exponential(1.0), sig2) + log(sig) + log(sig2) +
        logpdf(Normal(0.0, 1.0), b0) + logpdf(Normal(0.0, 1.0), b1)
    sampler = prepare_query(built, bound, :sampler)
    @test sampler(u) ≈ want atol = 1e-9
    @test prepare_query(built, bound, :likelihood)(u) ≈ want_like atol = 1e-9
    # A foreign-axis FIRST response fails at the n_obs gate.
    swapped = Meta.parse("""begin
        sigma ~ Exponential(1.0)
        sigma2 ~ Exponential(1.0)
        b0_vc ~ Normal(0.0, 1.0)
        b1_vc ~ Normal(0.0, 1.0)
        log_Vc = b0_vc .+ b1_vc .* age_s
        pk_sched = linear_pk_schedule(obs = (:subj, :time),
            dose = (:dsubj, :dtime, :damt), ecg = (:esubj, :etime))
        @plate conc for s in 1:2
            read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_Vc,
                log_Vc, log_Vc, log_Vc)
            mu = read_locs[pk_sched.obs_map]
            e = read_locs[pk_sched.ecg_map]
            qy .~ Normal.(e, sigma2)
            dv .~ Normal.(mu, sigma)
            mu
        end
    end""")
    @test_throws "≠ schedule obs axis" bind_data(
        lower_rkppl(swapped, _je_foreign_data()), cols)
end

# --- D5: segmented nadir -------------------------------------------------------

@testset "D5 host segmented nadir" begin
    change = [-0.3, 0.5, 0.1, -0.4]
    ends = [2, 2, 4]
    want = vcat(tgi_running_nadir(change[1:2]), Float64[],
        tgi_running_nadir(change[3:4]))
    @test tgi_segmented_nadir(change, ends) == want
    @test want == [0.0, -0.3, 0.0, 0.0]
    @test_throws "must be nondecreasing" tgi_segmented_nadir(change, [2, 1])
    @test_throws "must be non-negative" tgi_segmented_nadir(change, [-1, 4])
    @test_throws "≠ row count" tgi_segmented_nadir(change, [1, 2])
    @test tgi_segmented_nadir(Float64[], Int[]) == Float64[]
end

@testset "D5 builder derives tgi_seg_ends" begin
    # Oracle fixture: S1 3 rows, S2 3 rows, S3 none (empty segment).
    @test _je_sched().tgi_seg_ends == [3, 6, 6]
    plain = build_linear_pk_schedule([1, 1], [10.0, 20.0], [1],
        [0.0], [100.0])
    @test isempty(plain.tgi_seg_ends)
    declared_empty = build_linear_pk_schedule([1, 1], [10.0, 20.0],
        [1], [0.0], [100.0]; tgi = (Int[], Float64[]))
    @test declared_empty.tgi_seg_ends == [0]
end

function _je_nadir_columns(; ends = [2, 3])
    Dict{Symbol,AbstractVector}(
        :subj => [1, 1, 2], :time => [10.0, 20.0, 5.0],
        :dsubj => [1, 2], :dtime => [0.0, 0.0],
        :damt => [100.0, 50.0],
        :dv => [1.0, 2.0, 3.0], :qy => [0.1, 0.2, 0.3],
        :age_s => [30.0, 40.0], :my_ends => ends)
end

function _je_nadir_ast(cell)
    Meta.parse("""begin
        sigma ~ Exponential(1.0)
        sigma2 ~ Exponential(1.0)
        b0_vc ~ Normal(0.0, 1.0)
        b1_vc ~ Normal(0.0, 1.0)
        log_Vc = b0_vc .+ b1_vc .* age_s
        pk_sched = linear_pk_schedule(obs = (:subj, :time),
            dose = (:dsubj, :dtime, :damt))
        @plate conc for s in 1:2
            $(cell)
        end
    end""")
end

_je_nadir_data() = Set([:subj, :time, :dsubj, :dtime, :damt, :dv, :qy,
    :age_s, :my_ends])

const _JE_NADIR_GOOD = "$_JE_FULLCALL\n" *
    "chg = read_locs[pk_sched.obs_map]\n" *
    "ref = tgi_segmented_nadir(chg, my_ends)\n" *
    "mu = read_locs[pk_sched.obs_map]\n" *
    "dv .~ Normal.(mu, sigma)\n" *
    "qy .~ Normal.(ref, sigma2)\nmu"

@testset "D5 nadir cell form fails closed" begin
    cellcase = (cell) -> lower_rkppl(_je_nadir_ast(cell), _je_nadir_data())
    # Walker: arity, ends shape, change refs.
    @test_throws "takes 2 arguments" cellcase(
        "$_JE_FULLCALL\nchg = read_locs[pk_sched.obs_map]\n" *
        "ref = tgi_segmented_nadir(chg)\n" *
        "dv .~ Normal.(chg, sigma)\nchg")
    @test_throws "must be a bound integer column name" cellcase(
        "$_JE_FULLCALL\nchg = read_locs[pk_sched.obs_map]\n" *
        "ref = tgi_segmented_nadir(chg, 1)\n" *
        "dv .~ Normal.(chg, sigma)\nchg")
    @test_throws "gather explicitly" cellcase(
        "$_JE_FULLCALL\nchg = read_locs[pk_sched.obs_map]\n" *
        "ref = tgi_segmented_nadir(log_Vc, my_ends)\n" *
        "dv .~ Normal.(chg, sigma)\nchg")
    @test_throws "unknown name" cellcase(
        "$_JE_FULLCALL\nref = tgi_segmented_nadir(nope, my_ends)\n" *
        "dv .~ Normal.(ref, sigma)\nref")
    # Only the nadir wires in: other TGI cell functions stay out.
    @test_throws "not in the grouped-v1 cell vocabulary" cellcase(
        "$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "x = tgi_ratio_loglinear(mu, log_Vc, log_Vc)\n" *
        "dv .~ Normal.(mu, sigma)\nmu")
    # Shapes: change space + ends bound-ness.
    bindshapes = (cell, cols = _je_nadir_columns()) ->
        bind_data(cellcase(cell), cols)
    @test_throws "is read-space" bindshapes(
        "$_JE_FULLCALL\nref = tgi_segmented_nadir(read_locs, my_ends)\n" *
        "mu = read_locs[pk_sched.obs_map]\n" *
        "dv .~ Normal.(mu, sigma)\nmu")
    @test_throws "is scalar" bindshapes(
        "$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "ref = tgi_segmented_nadir(sigma, my_ends)\n" *
        "dv .~ Normal.(mu, sigma)\nmu")
    @test_throws "is not bound" bindshapes(
        "$_JE_FULLCALL\nchg = read_locs[pk_sched.obs_map]\n" *
        "ref = tgi_segmented_nadir(chg, nope_ends)\n" *
        "mu = read_locs[pk_sched.obs_map]\n" *
        "dv .~ Normal.(mu, sigma)\nmu")
    # Ends validator: integer, one per subject, nondecreasing from a
    # non-negative start, last == change length.
    @test_throws "must be an integer column" bindshapes(_JE_NADIR_GOOD,
        _je_nadir_columns(; ends = [1.0, 3.0]))
    @test_throws "≠ subjects 2" bindshapes(_JE_NADIR_GOOD,
        _je_nadir_columns(; ends = [3]))
    @test_throws "must be nondecreasing" bindshapes(_JE_NADIR_GOOD,
        _je_nadir_columns(; ends = [3, 2]))
    @test_throws "must be non-negative" bindshapes(_JE_NADIR_GOOD,
        _je_nadir_columns(; ends = [-1, 3]))
    @test_throws "≠ change length 3" bindshapes(_JE_NADIR_GOOD,
        _je_nadir_columns(; ends = [2, 2]))
    # Valid caller-supplied ends bind.
    @test bind_data(cellcase(_JE_NADIR_GOOD),
        _je_nadir_columns()).n_obs == 3
end

function _je_seg_columns()
    Dict{Symbol,AbstractVector}(
        :subj => [1, 1, 2], :time => [10.0, 20.0, 5.0],
        :dsubj => [1, 2], :dtime => [0.0, 0.0],
        :damt => [100.0, 50.0],
        :dv => [1.0, 2.0, 3.0], :qy => [0.1, 0.2, 0.3],
        :age_s => [30.0, 40.0],
        :tsubj => [1, 1, 1], :ttime => [12.0, 18.0, 25.0])
end

function _je_seg_ast()
    Meta.parse("""begin
        sigma ~ Exponential(1.0)
        sigma2 ~ Exponential(1.0)
        b0_vc ~ Normal(0.0, 1.0)
        b1_vc ~ Normal(0.0, 1.0)
        log_Vc = b0_vc .+ b1_vc .* age_s
        pk_sched = linear_pk_schedule(obs = (:subj, :time),
            dose = (:dsubj, :dtime, :damt), tgi = (:tsubj, :ttime))
        @plate conc for s in 1:2
            read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_Vc,
                log_Vc, log_Vc, log_Vc)
            mu = read_locs[pk_sched.obs_map]
            chg = read_locs[pk_sched.tgi_map]
            ref = tgi_segmented_nadir(chg, pk_sched_tgi_seg_ends)
            dv .~ Normal.(mu, sigma)
            qy .~ Normal.(ref, sigma2)
            mu
        end
    end""")
end

_je_seg_data() = Set([:subj, :time, :dsubj, :dtime, :damt, :dv, :qy,
    :age_s, :tsubj, :ttime])

@testset "D5 nadir unroll end to end (empty segment)" begin
    # S2 has no tgi rows: ends [3, 3], empty second segment.
    unbound = lower_rkppl(_je_seg_ast(), _je_seg_data())
    cols = _je_seg_columns()
    bound = bind_data(unbound, cols)
    @test bound.n_obs == 3
    @test bound.columns[:pk_sched_tgi_seg_ends] == [3, 3]
    built = build_kernel(bound)
    r = repr(kernel_expr(bound, assign_layout(bound)))
    # One scan per non-empty segment; the empty segment emits an
    # empty range-copy (never a scan over nothing; never a view —
    # heterogeneous vcats break Enzyme AD).
    @test occursin("scan", r)
    @test occursin("1:0", r)
    names = coordinate_names(assign_layout(bound))
    sig, sig2, b0, b1 = 0.5, 0.7, 2.0, 0.01
    u = _je_gather_u(names, sig, sig2, b0, b1)
    lps = b0 .+ b1 .* cols[:age_s]
    sched = build_linear_pk_schedule(cols[:subj], cols[:time],
        cols[:dsubj], cols[:dtime], cols[:damt];
        tgi = (cols[:tsubj], cols[:ttime]))
    conc = vcat([_je_oracle_auc_reads(sched, s,
        zeros(s == 1 ? sched.op_ends[1] :
            sched.op_ends[s] - sched.op_ends[s-1]),
        lps[s], lps[s], lps[s], lps[s], lps[s])[1:sched.n_reads[s]]
        for s in 1:2]...)
    chg = conc[sched.tgi_map]
    ref = tgi_segmented_nadir(chg, sched.tgi_seg_ends)
    want_like = sum(logpdf.(Normal.(conc[sched.obs_map], sig),
        cols[:dv])) + sum(logpdf.(Normal.(ref, sig2), cols[:qy]))
    @test prepare_query(built, bound, :likelihood)(u) ≈ want_like atol = 1e-9
    # No nadir call, no ends product (use-gated like conc_map).
    bound_r1 = bind_data(lower_rkppl(_je_ast(), _JE_DATA), _je_columns())
    @test !haskey(bound_r1.columns, :pk_sched_tgi_seg_ends)
end

@testset "D5 nadir unroll traces under Reactant" begin
    # The emitted unroll shape (scan-over-range-copy + empty
    # range-copy + vcat) with static ranges, inside a @kernel body
    # (`scan` is kernel authoring sugar — a bare Reactant.@compile
    # wrapper cannot see it): native parity first, then the compiled
    # AD path at two points (away from the min kinks).
    @kernel _je_nadir_seg_kernel(change::Vector{Float64}) = begin
        r1 = scan(change[1:3]; init = 0.0) do carry, x
            (min(carry, x), carry)
        end
        r2 = change[1:0]
        ref = vcat(r1, r2)
        total::Float64 = sum(ref)
        return total
    end
    kern = prepare(_je_nadir_seg_kernel; have = (:change,),
        want = :total, bound = (;))
    for c in ([-0.3, 0.5, 0.1], [0.2, -0.6, -0.1])
        want = sum(tgi_segmented_nadir(c, [3, 3]))
        @test kern(c) ≈ want rtol = 1e-14
        prep = prepare_ad(_je_nadir_seg_kernel, _JE_BACKEND, c;
            active = :change, want = :total, bound = (;))
        g = Vector{Float64}(undef, 3)
        val, _ = ReactiveKernels.ad_value_and_gradient!(prep, g, c)
        @test val ≈ want rtol = 1e-14
        compiled = compile_ad_value_and_gradient(prep, Reactant.to_rarray(c))
        rval, rgrad = compiled(Reactant.to_rarray(c))
        @test Float64(rval) ≈ want rtol = 1e-11
        @test Array(rgrad) ≈ g rtol = 1e-5 atol = 1e-7
    end
end

# --- D6: TGI per-family lowering builders --------------------------------------

# Dissect a builder's `(stmts, term)`: pw/node names, plate inputs,
# cell head. (`pw = plate(inputs...) do dovars...; cell; end` +
# `node::Float64 = sum(pw)` — the `_plate_sum_stmts` shape.)
function _je_plate_parts(stmts)
    @assert length(stmts) == 2
    doex = stmts[1].args[2]
    platecall = doex.args[1]
    cell = doex.args[2].args[2].args[2]
    return (pw = stmts[1].args[1],
        node = stmts[2].args[1].args[1],
        inputs = platecall.args[2:end], cell = cell)
end

@testset "D6 builder shapes thread-or-inline" begin
    dovar(i) = Symbol(:_ppl_c, i)
    # All-symbol: every value threads as a plate input.
    stmts, node = tgi_category_stmts(; response = :cc, r = :r, ref = :ref,
        c_cr = :ccr, c_pr = :cpr, c_pd = :cpd, sigma = :sd, eps = :eps,
        label = :k)
    p = _je_plate_parts(stmts)
    @test node == :_ppl_lik_tgi_category_k
    @test p.node == node
    @test p.pw == :_ppl_pw_tgi_category_k
    @test p.inputs == [:cc, :r, :ref, :ccr, :cpr, :cpd, :sd, :eps]
    @test p.cell.args[1] === :tgi_category_lpmf
    @test p.cell.args[2:end] == [dovar(i) for i in 1:8]
    # Literals inline; symbols thread.
    stmts, _ = tgi_category_stmts(; response = :cc, r = :r, ref = :ref,
        c_cr = -0.5, c_pr = -0.3, c_pd = 0.2, sigma = :sd, eps = 1e-6,
        label = :k)
    p = _je_plate_parts(stmts)
    @test p.inputs == [:cc, :r, :ref, :sd]
    @test p.cell.args[2:end] ==
        [dovar(1), dovar(2), dovar(3), -0.5, -0.3, 0.2, dovar(4), 1e-6]
    stmts, node = tgi_response_stmts(; response = :bb, r = :r, ref = :ref,
        c_pr = :cpr, c_pd = -0.2, sigma = :sd, eps = 1e-6, label = :k)
    p = _je_plate_parts(stmts)
    @test node == :_ppl_lik_tgi_response_k
    @test p.inputs == [:bb, :r, :ref, :cpr, :sd]
    @test p.cell.args[1] === :tgi_response_lpmf
    @test p.cell.args[2:end] == [dovar(1), dovar(2), dovar(3), dovar(4),
        -0.2, dovar(5), 1e-6]
    stmts, node = tgi_censored_stmts(; response = :yl, mu = :mu,
        sigma = :sd, lloq = :llq, label = :k)
    p = _je_plate_parts(stmts)
    @test node == :_ppl_lik_tgi_censored_k
    @test p.inputs == [:yl, :mu, :sd, :llq]
    @test p.cell.args[1] === :tgi_censored_lpdf
    @test p.cell.args[2:end] == [dovar(i) for i in 1:4]
end

@testset "D6 category plate values" begin
    cc = [1, 2, 3, 4]
    r = [-0.6, -0.4, 0.0, 0.3]
    ref = [0.0, -0.2, -0.1, 0.0]
    ccr, cpr, cpd = fill(-0.5, 4), fill(-0.3, 4), fill(0.2, 4)
    eps = 1e-6
    @kernel _je_cat_plate_kernel(u::Vector{Float64}, cc, r, ref, ccr,
            cpr, cpd, eps) = begin
        sd::Float64 = u[1]
        pw = plate(cc, r, ref, ccr, cpr, cpd, sd, eps) do yv, rv, refv,
                ccrv, cprv, cpdv, sv, epsv
            tgi_category_lpmf(yv, rv, refv, ccrv, cprv, cpdv, sv, epsv)
        end
        total::Float64 = sum(pw)
        return total
    end
    bound = (; cc, r, ref, ccr, cpr, cpd, eps)
    kern = prepare(_je_cat_plate_kernel;
        have = (:u, :cc, :r, :ref, :ccr, :cpr, :cpd, :eps), want = :total,
        bound = bound)
    @test kern([0.25]) ≈
        sum(tgi_category_lpmfs(cc, r, ref, -0.5, -0.3, 0.2, 0.25, eps)) rtol = 1e-12
end

@testset "D6 response plate values" begin
    bb = [0, 1, 1, 0]
    r = [0.1, -0.4, -0.5, 0.2]
    ref = [0.0, 0.0, -0.3, -0.1]
    cpr, cpd = fill(-0.3, 4), fill(0.2, 4)
    eps = 1e-6
    @kernel _je_resp_plate_kernel(u::Vector{Float64}, bb, r, ref, cpr,
            cpd, eps) = begin
        sd::Float64 = u[1]
        pw = plate(bb, r, ref, cpr, cpd, sd, eps) do yv, rv, refv, cprv,
                cpdv, sv, epsv
            tgi_response_lpmf(yv, rv, refv, cprv, cpdv, sv, epsv)
        end
        total::Float64 = sum(pw)
        return total
    end
    bound = (; bb, r, ref, cpr, cpd, eps)
    kern = prepare(_je_resp_plate_kernel;
        have = (:u, :bb, :r, :ref, :cpr, :cpd, :eps), want = :total,
        bound = bound)
    @test kern([0.25]) ≈
        sum(tgi_response_lpmfs(bb, r, ref, -0.3, 0.2, 0.25, eps)) rtol = 1e-12
end

@testset "D6 censored plate values" begin
    ylog = [2.1, log(5.0), 0.5]
    mu = [2.0, 1.7, 1.0]
    llq = log(5.0)
    @kernel _je_cens_plate_kernel(u::Vector{Float64}, ylog, mu,
            llq) = begin
        sd::Float64 = u[1]
        pw = plate(ylog, mu, sd, llq) do yv, muv, sv, llqv
            tgi_censored_lpdf(yv, muv, sv, llqv)
        end
        total::Float64 = sum(pw)
        return total
    end
    bound = (; ylog, mu, llq)
    kern = prepare(_je_cens_plate_kernel;
        have = (:u, :ylog, :mu, :llq), want = :total, bound = bound)
    @test kern([0.13]) ≈ sum(tgi_censored_lpdfs(ylog, mu, 0.13, llq)) rtol = 1e-12
end

# --- D7: grouped emitter per-family dispatch -----------------------------------

_je_router_kp() =
    only(lower_rkppl(_je_gather_ast(), _je_gather_data()).kernel_plates)

@testset "D7 emitter routes each joint family" begin
    kp = _je_router_kp()
    route = ReactiveKernelsPPL._grouped_obs_likelihood_stmts
    # PK censored → QT pk builder, exact.
    pk = (response = :dv, family = CensoredAddpropnormalFam,
        location = :mu, scale = :add, params = (:prop, :llq))
    @test route(kp, pk, :dv, :o1) ==
        ReactiveKernelsPPL._qt_joint_pk_likelihood_stmts(; response = :dv,
            location = :mu, add = :add, prop = :prop, lloq = :llq,
            label = :o1)
    # TGI families → D6 builders, exact (pins the positional arg map).
    cat = (response = :cc, family = TgiCategoryFam, location = :r,
        scale = :ref, params = (:ccr, :cpr, :cpd, :sd, :eps))
    @test route(kp, cat, :cc, :o2) ==
        tgi_category_stmts(; response = :cc, r = :r, ref = :ref,
            c_cr = :ccr, c_pr = :cpr, c_pd = :cpd, sigma = :sd, eps = :eps,
            label = :o2)
    resp = (response = :bb, family = TgiResponseFam, location = :r,
        scale = :ref, params = (:cpr, :cpd, :sd, :eps))
    @test route(kp, resp, :bb, :o3) ==
        tgi_response_stmts(; response = :bb, r = :r, ref = :ref,
            c_pr = :cpr, c_pd = :cpd, sigma = :sd, eps = :eps, label = :o3)
    cens = (response = :yl, family = TgiCensoredFam, location = :mu,
        scale = :sd, params = (:llq,))
    @test route(kp, cens, :yl, :o4) ==
        tgi_censored_stmts(; response = :yl, mu = :mu, sigma = :sd,
            lloq = :llq, label = :o4)
    # TGI literals inline through the router.
    catlit = (response = :cc, family = TgiCategoryFam, location = :r,
        scale = :ref, params = (-0.5, -0.3, 0.2, :sd, 1e-6))
    stmts, _ = route(kp, catlit, :cc, :o5)
    @test _je_plate_parts(stmts).inputs == [:cc, :r, :ref, :sd]
    # Censored literals fail closed (the QT builder threads names).
    badlit = (response = :dv, family = CensoredAddpropnormalFam,
        location = :mu, scale = 0.1, params = (:prop, :llq))
    @test_throws "literals do not lower" route(kp, badlit, :dv, :o1)
    # Unrouted families fail naming the admitted set.
    badfam = (response = :yy, family = BernoulliLogitFam,
        location = :mu, scale = 1.0, params = ())
    @test_throws "has no in-cell emitter" route(kp, badfam, :yy, :o1)
end

@testset "D7 QT Gaussian matches the builder shape" begin
    # QT-shaped Gaussian obs: the weight folds via a surface
    # pre-assignment (the KernelObs node cannot carry it), so the
    # emitter routes the generic path — pinned to the QT builder's
    # plate pair modulo the scale provenance.
    ast = Meta.parse("""begin
        sigma ~ Exponential(1.0)
        qt_scale ~ Exponential(1.0)
        b0_vc ~ Normal(0.0, 1.0)
        b1_vc ~ Normal(0.0, 1.0)
        log_Vc = b0_vc .+ b1_vc .* age_s
        pk_sched = linear_pk_schedule(obs = (:subj, :time),
            dose = (:dsubj, :dtime, :damt), ecg = (:esubj, :etime))
        @plate conc for s in 1:2
            read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_Vc,
                log_Vc, log_Vc, log_Vc)
            mu = read_locs[pk_sched.obs_map]
            e = read_locs[pk_sched.ecg_map]
            qt_sd = qt_scale .* qt_w
            dv .~ Normal.(mu, sigma)
            qy .~ Normal.(e, qt_sd)
            mu
        end
    end""")
    cols = _je_foreign_columns()
    cols[:qt_w] = [1.0, 1.1, 0.9]
    data = union(_je_foreign_data(), Set([:qt_w]))
    kp = only(lower_rkppl(ast, data).kernel_plates)
    obs = kp.obs[2]
    @test obs.family === GaussianFam
    @test obs.location === :e
    @test obs.scale === :qt_sd
    route = ReactiveKernelsPPL._grouped_obs_likelihood_stmts
    rstmts, rnode = route(kp, obs, :qy, :qt_joint_qt_qe)
    bstmts, bnode =
        ReactiveKernelsPPL._qt_joint_qt_likelihood_stmts(; response = :qy,
            location = :e, scale = :qt_scale, weight = :ww, label = :qe)
    @test rnode == bnode
    @test length(bstmts) == 3
    # The surface pre-assignment RHS == the builder pre-local RHS.
    @test bstmts[1].args[2] == :(qt_scale .* ww)
    # The emitter plate pair == the builder plate pair modulo the
    # scale input name (surface `qt_sd` vs builder pre-local).
    sc = Symbol(:_ppl_qt_scale_, :qe)
    subsc(ex) = ex isa Symbol ? (ex === sc ? :qt_sd : ex) :
        ex isa Expr ? Expr(ex.head, map(subsc, ex.args)...) : ex
    @test rstmts == subsc.(bstmts[2:3])
end

# --- D8: prep validators -------------------------------------------------------

function _je_map_columns(; subjmap = [1, 1, 2, 2], readmap = [1, 2, 3, 4],
        readmap2 = [4, 3, 2, 1])
    cols = _je_gather_columns()
    cols[:subjmap] = subjmap
    cols[:readmap] = readmap
    cols[:readmap2] = readmap2
    return cols
end

function _je_map_ast()
    Meta.parse("""begin
        sigma ~ Exponential(1.0)
        b0_vc ~ Normal(0.0, 1.0)
        b1_vc ~ Normal(0.0, 1.0)
        log_Vc = b0_vc .+ b1_vc .* age_s
        pk_sched = linear_pk_schedule(obs = (:subj, :time),
            dose = (:dsubj, :dtime, :damt))
        @plate conc for s in 1:2
            read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_Vc,
                log_Vc, log_Vc, log_Vc)
            mu = read_locs[pk_sched.obs_map]
            bump = log_Vc[subjmap]
            mu2 = mu .+ bump
            rev = read_locs[readmap]
            x = mu[readmap2]
            dv .~ Normal.(mu2, sigma)
            mu2
        end
    end""")
end

_je_map_data() = union(_je_gather_data(), Set([:subjmap, :readmap, :readmap2]))

@testset "D8 gather-map prep contract" begin
    unbound = lower_rkppl(_je_map_ast(), _je_map_data())
    # Valid maps bind (LP-subject, read-space, and obs-axis sources).
    @test bind_data(unbound, _je_map_columns()).n_obs == 4
    # Subject maps stay within 1..n_sub.
    @test_throws "outside 1..2" bind_data(unbound,
        _je_map_columns(; subjmap = [1, 9, 1, 1]))
    # Read-space maps stay within the flat reads (R = 4 here).
    @test_throws "outside 1..4" bind_data(unbound,
        _je_map_columns(; readmap = [1, 2, 3, 99]))
    # Obs-axis maps stay within the source axis length.
    @test_throws "outside 1..4" bind_data(unbound,
        _je_map_columns(; readmap2 = [1, 2, 3, 99]))
    # Maps are integer columns.
    @test_throws "must be an integer column" bind_data(unbound,
        _je_map_columns(; subjmap = [1.0, 1.0, 2.0, 2.0]))
    @test_throws "must be an integer column" bind_data(unbound,
        _je_map_columns(; readmap = [1.0, 2.0, 3.0, 4.0]))
    # Responses are numeric and finite (the slice contract, pinned).
    badnum = _je_map_columns()
    badnum[:dv] = ["a", "b", "c", "d"]
    @test_throws "must be numeric" bind_data(unbound, badnum)
    badfin = _je_map_columns()
    badfin[:dv] = [1.0, 2.0, Inf, 4.0]
    @test_throws "must be finite" bind_data(unbound, badfin)
end

@testset "D8 AUC read-space bound is 2R" begin
    # Single subject, R = 2 reads: plain maps over the AUC cell's
    # [conc; auc] flat admit 1..4.
    ast = Meta.parse("""begin
        sigma ~ Exponential(1.0)
        b0_vc ~ Normal(0.0, 1.0)
        log_Vc = b0_vc
        pk_sched = linear_pk_schedule(obs = (:subj, :time),
            dose = (:dsubj, :dtime, :damt))
        log_F = linear_pk_log_f(pk_sched; k = 5)
        @plate conc for s in 1:1
            pk_reads = linear_pk_read_locs_auc(pk_sched, log_F, log_Vc,
                log_Vc, log_Vc, log_Vc, log_Vc)
            mu = pk_reads[pk_sched.obs_map]
            x = pk_reads[aucmap]
            dv .~ Normal.(mu, sigma)
            mu
        end
    end""")
    data = union(_je_auc_data(), Set([:aucmap]))
    unbound = lower_rkppl(ast, data)
    cols = _je_auc_columns()
    cols[:aucmap] = [3, 4]
    @test bind_data(unbound, cols).n_obs == 2
    cols[:aucmap] = [5, 5]
    @test_throws "outside 1..4" bind_data(unbound, cols)
end

@testset "D8 TGI axis order fails closed" begin
    unbound = lower_rkppl(_je_seg_ast(), _je_seg_data())
    @test bind_data(unbound, _je_seg_columns()).n_obs == 3
    interleave = _je_seg_columns()
    interleave[:tsubj] = [1, 2, 1]
    interleave[:ttime] = [12.0, 5.0, 25.0]
    @test_throws "grouped by subject" bind_data(unbound, interleave)
    untimed = _je_seg_columns()
    untimed[:tsubj] = [1, 1, 2]
    untimed[:ttime] = [25.0, 12.0, 6.0]
    @test_throws "nondecreasing within subject 1" bind_data(unbound, untimed)
end

@testset "D8 per-subject t0 gathers like the LPs" begin
    # `t0base = t0subj` is a bare-column offset predictor (pure data
    # LP); the cell gathers it per tgi row (NOT a slice — slices are
    # response-length and the axis check would reject an n_sub
    # series in obs arithmetic).
    ast = Meta.parse("""begin
        sigma ~ Exponential(1.0)
        sigma2 ~ Exponential(1.0)
        b0_vc ~ Normal(0.0, 1.0)
        b1_vc ~ Normal(0.0, 1.0)
        log_Vc = b0_vc .+ b1_vc .* age_s
        t0base = t0subj
        pk_sched = linear_pk_schedule(obs = (:subj, :time),
            dose = (:dsubj, :dtime, :damt), tgi = (:tsubj, :ttime))
        @plate conc for s in 1:2
            read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_Vc,
                log_Vc, log_Vc, log_Vc)
            mu = read_locs[pk_sched.obs_map]
            chg = read_locs[pk_sched.tgi_map]
            t0rows = t0base[tsubj]
            adj = chg .+ t0rows
            dv .~ Normal.(mu, sigma)
            qy .~ Normal.(adj, sigma2)
            mu
        end
    end""")
    cols = _je_seg_columns()
    cols[:t0subj] = [-0.1, 0.0]
    data = union(_je_seg_data(), Set([:t0subj]))
    bound = bind_data(lower_rkppl(ast, data), cols)
    @test bound.n_obs == 3
    built = build_kernel(bound)
    layout = assign_layout(bound)
    r = repr(kernel_expr(bound, layout))
    # The offset predictor materializes an LP vector (pure data).
    @test occursin("_ppl_lp_t0base", r)
    names = coordinate_names(layout)
    # No coordinates for the coefficient-free LP.
    @test !any(n -> startswith(String(n), "t0base."), names)
    sig, sig2, b0, b1 = 0.5, 0.7, 2.0, 0.01
    u = _je_gather_u(names, sig, sig2, b0, b1)
    lps = b0 .+ b1 .* cols[:age_s]
    sched = build_linear_pk_schedule(cols[:subj], cols[:time],
        cols[:dsubj], cols[:dtime], cols[:damt];
        tgi = (cols[:tsubj], cols[:ttime]))
    conc = vcat([_je_oracle_auc_reads(sched, s,
        zeros(s == 1 ? sched.op_ends[1] :
            sched.op_ends[s] - sched.op_ends[s-1]),
        lps[s], lps[s], lps[s], lps[s], lps[s])[1:sched.n_reads[s]]
        for s in 1:2]...)
    adj = conc[sched.tgi_map] .+ cols[:t0subj][cols[:tsubj]]
    want_like = sum(logpdf.(Normal.(conc[sched.obs_map], sig),
        cols[:dv])) + sum(logpdf.(Normal.(adj, sig2), cols[:qy]))
    @test prepare_query(built, bound, :likelihood)(u) ≈ want_like atol = 1e-9
end

# --- D9: two-subject joint e2e (all five families, one program) ----------------
#
# The W3c deliverable proof: PK censored + QT Gaussian + TGI
# category/response/censored in ONE generated program through
# lower_rkppl → bind_data → build_kernel → prepare_query (the eventlp
# e2e pattern), against an independent Distributions.jl oracle. Stub
# LPs only (one fixed-effect LP, corpus-48 style — the true joint
# program is the parent's once decl lands). Subject 2 has no TGI rows
# (ends [3, 3]): the empty segment rides the value, Enzyme, AND
# Reactant paths. Cutoffs are SB's exact (lugano_ct c_cr/c_pr/c_pd +
# eps 0.01); the all-equal-LP stub puts conc ≈ 0, so every observed
# TGI cell sits at a moderate z (|z| ≤ 2) where the cdf-difference
# oracle agrees with the erfc port to 1e-12 (the test_tgi `_hand_*`
# precedent); the PK BLQ arm (z ≈ 3) rides the landed censored-Gaussian
# `log(cdf)` form (the test_qt_joint SB-bridge precedent, 1e-12).

function _je_joint_columns()
    Dict{Symbol,AbstractVector}(
        :subj => [1, 1, 2, 2], :time => [10.0, 20.0, 5.0, 8.0],
        :dsubj => [1, 2], :dtime => [0.0, 0.0],
        :damt => [100.0, 50.0],
        # PK row 3 sits exactly at LLOQ (the prep clamp — cdf arm).
        :dv => [1.0, 2.0, 0.3, 4.0], :llq => fill(0.3, 4),
        :qy => [0.1, 0.2, 0.3], :qt_w => [1.0, 1.1, 0.9],
        :age_s => [30.0, 40.0],
        :esubj => [1, 1, 2], :etime => [15.0, 25.0, 3.0],
        :tsubj => [1, 1, 1], :ttime => [12.0, 18.0, 25.0],
        :cc => [1, 2, 4], :bb => [0, 1, 0],
        # Tumor row 2 sits exactly at LLOQ (censored cdf arm).
        :yl => [0.5, 0.3, 0.9])
end

function _je_joint_ast()
    Meta.parse("""begin
        s_add ~ Exponential(1.0)
        s_prop ~ Exponential(1.0)
        qt_scale ~ Exponential(1.0)
        sd ~ Exponential(1.0)
        b0_vc ~ Normal(0.0, 1.0)
        b1_vc ~ Normal(0.0, 1.0)
        log_Vc = b0_vc .+ b1_vc .* age_s
        pk_sched = linear_pk_schedule(obs = (:subj, :time),
            dose = (:dsubj, :dtime, :damt), ecg = (:esubj, :etime),
            tgi = (:tsubj, :ttime))
        @plate conc for s in 1:2
            read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_Vc,
                log_Vc, log_Vc, log_Vc)
            mu = read_locs[pk_sched.obs_map]
            e = read_locs[pk_sched.ecg_map]
            r = read_locs[pk_sched.tgi_map]
            ref = tgi_segmented_nadir(r, pk_sched_tgi_seg_ends)
            qt_sd = qt_scale .* qt_w
            dv .~ CensoredAddpropnormal.(mu, s_add, s_prop, llq)
            qy .~ Normal.(e, qt_sd)
            cc .~ TgiCategory.(r, ref, -0.5, -0.3, 0.2, sd, 0.01)
            bb .~ TgiResponse.(r, ref, -0.3, 0.2, sd, 0.01)
            yl .~ TgiCensored.(r, sd, 0.3)
            mu
        end
    end""")
end

_je_joint_data() = Set([:subj, :time, :dsubj, :dtime, :damt, :dv, :llq,
    :qy, :qt_w, :age_s, :esubj, :etime, :tsubj, :ttime, :cc, :bb, :yl])

# Unconstrained pack: log scales + identity LP coefs.
function _je_joint_u(names, sa, sp, qs, sd, b0, b1)
    map(names) do n
        n === :s_add && return log(sa)
        n === :s_prop && return log(sp)
        n === :qt_scale && return log(qs)
        n === :sd && return log(sd)
        startswith(String(n), "log_Vc.") ||
            error("unexpected coordinate $n")
        return endswith(String(n), ".Intercept") ? b0 : b1
    end
end

_je_joint_theta() = (sa = 0.1, sp = 0.05, qs = 0.5, sd = 0.25,
    b0 = 2.0, b1 = 0.01)

# Independent hand oracles (own running-min transcription of the nadir
# doc + Distributions.jl loops — never the emitted code, never the
# library cells): cdf/logpdf primitives vs the port's erfc-log (the
# test_tgi `_hand_*` precedent, agreement 1e-12 per row).
function _je_hand_nadir(change, ends)
    out = Float64[]
    prev = 0
    for hi in ends
        m = 0.0
        for x in change[(prev+1):hi]
            push!(out, m)
            m = min(m, x)
        end
        prev = hi
    end
    return out
end

function _je_hand_category(y, r, ref, c_cr, c_pr, c_pd, sigma, eps)
    pd = (c_pd + ref - r) / sigma
    pr = min((c_pr - r) / sigma, pd)
    cr = min((c_cr - r) / sigma, pr)
    Φ = Normal()
    p = y == 1 ? cdf(Φ, cr) :
        y == 2 ? cdf(Φ, pr) - cdf(Φ, cr) :
        y == 3 ? cdf(Φ, pd) - cdf(Φ, pr) : 1 - cdf(Φ, pd)
    return log((1 - eps) * p + eps / 4)
end

function _je_hand_response(y, r, ref, c_pr, c_pd, sigma, eps)
    pr = min(c_pr - r, c_pd + ref - r) / sigma
    Φ = Normal()
    p = y == 1 ? cdf(Φ, pr) : 1 - cdf(Φ, pr)
    return log((1 - eps) * p + eps / 2)
end

function _je_hand_censored(y, mu, sigma, lloq)
    y <= lloq && return logcdf(Normal(mu, sigma), lloq)
    return logpdf(Normal(mu, sigma), y)
end

# Full-joint sampler oracle at unconstrained `w` (value + PK location
# vector, the `_pkl_oracle_u` shape): conc via the D1
# matrix-exponential oracle, plates as Distributions.jl loops, priors +
# jacobians.
function _je_joint_oracle_u(w, names, cols, sched)
    d = Dict(n => v for (n, v) in zip(names, w))
    sa, sp = exp(d[:s_add]), exp(d[:s_prop])
    qs, sd = exp(d[:qt_scale]), exp(d[:sd])
    b0 = d[only(n for n in names if endswith(String(n), ".Intercept"))]
    b1 = d[only(n for n in names if startswith(String(n), "log_Vc.") &&
        !endswith(String(n), ".Intercept"))]
    lps = b0 .+ b1 .* cols[:age_s]
    conc = vcat([_je_oracle_auc_reads(sched, s,
        zeros(s == 1 ? sched.op_ends[1] :
            sched.op_ends[s] - sched.op_ends[s-1]),
        lps[s], lps[s], lps[s], lps[s], lps[s])[1:sched.n_reads[s]]
        for s in 1:2]...)
    muo = conc[sched.obs_map]
    dv, llq = cols[:dv], cols[:llq]
    ll = 0.0
    for i in eachindex(muo)
        sc = sqrt(sa^2 + (muo[i] * sp)^2)
        dd = Normal(muo[i], sc)
        ll += dv[i] == llq[i] ? logcdf(dd, llq[i]) : logpdf(dd, dv[i])
    end
    e = conc[sched.ecg_map]
    for i in eachindex(e)
        ll += logpdf(Normal(e[i], qs * cols[:qt_w][i]), cols[:qy][i])
    end
    r = conc[sched.tgi_map]
    ref = _je_hand_nadir(r, sched.tgi_seg_ends)
    for i in eachindex(r)
        ll += _je_hand_category(cols[:cc][i], r[i], ref[i], -0.5, -0.3,
            0.2, sd, 0.01)
        ll += _je_hand_response(cols[:bb][i], r[i], ref[i], -0.3, 0.2,
            sd, 0.01)
        ll += _je_hand_censored(cols[:yl][i], r[i], sd, 0.3)
    end
    for s in (sa, sp, qs, sd)
        ll += logpdf(Exponential(1.0), s) + log(s)
    end
    ll += logpdf(Normal(0.0, 1.0), b0) + logpdf(Normal(0.0, 1.0), b1)
    return ll, muo
end

function _je_findiff(f, u; h = cbrt(eps(Float64)))
    g = Vector{Float64}(undef, length(u))
    for i in eachindex(u)
        up = Vector{Float64}(u)
        up[i] += h
        dn = Vector{Float64}(u)
        dn[i] -= h
        g[i] = (f(up) - f(dn)) / (2h)
    end
    return g
end

@testset "D9 nadir over a bare slice resolves" begin
    # A bare response column as nadir change (shapes admit slices —
    # `(:obs, len)`): generation resolves the response to its column
    # (the grouped flatmap — no undeclared port).
    cell = "$_JE_FULLCALL\nmu = read_locs[pk_sched.obs_map]\n" *
        "ref = tgi_segmented_nadir(qy, my_ends)\n" *
        "dv .~ Normal.(mu, sigma)\nqy .~ Normal.(ref, sigma2)\nmu"
    cols = _je_nadir_columns()
    bound = bind_data(
        lower_rkppl(_je_nadir_ast(cell), _je_nadir_data()), cols)
    @test bound.n_obs == 3
    built = build_kernel(bound)
    names = coordinate_names(assign_layout(bound))
    sig, sig2, b0, b1 = 0.5, 0.7, 2.0, 0.01
    u = _je_gather_u(names, sig, sig2, b0, b1)
    lps = b0 .+ b1 .* cols[:age_s]
    sched = build_linear_pk_schedule(cols[:subj], cols[:time],
        cols[:dsubj], cols[:dtime], cols[:damt])
    conc = vcat([_je_oracle_auc_reads(sched, s,
        zeros(s == 1 ? sched.op_ends[1] :
            sched.op_ends[s] - sched.op_ends[s-1]),
        lps[s], lps[s], lps[s], lps[s], lps[s])[1:sched.n_reads[s]]
        for s in 1:2]...)
    ref = _je_hand_nadir(cols[:qy], cols[:my_ends])
    want_like = sum(logpdf.(Normal.(conc[sched.obs_map], sig),
        cols[:dv])) + sum(logpdf.(Normal.(ref, sig2), cols[:qy]))
    @test prepare_query(built, bound, :likelihood)(u) ≈ want_like atol = 1e-9
end

@testset "D9 two-subject joint e2e value parity" begin
    unbound = lower_rkppl(_je_joint_ast(), _je_joint_data())
    @test length(only(unbound.kernel_plates).obs) == 5
    cols = _je_joint_columns()
    bound = bind_data(unbound, cols)
    @test bound.n_obs == 4
    # Subject 2 contributes no TGI rows: the empty segment rides every
    # D9 path below (value, Enzyme, Reactant).
    @test bound.columns[:pk_sched_tgi_seg_ends] == [3, 3]
    built = build_kernel(bound)
    names = coordinate_names(assign_layout(bound))
    θ = _je_joint_theta()
    u = _je_joint_u(names, θ.sa, θ.sp, θ.qs, θ.sd, θ.b0, θ.b1)
    sched = build_linear_pk_schedule(cols[:subj], cols[:time],
        cols[:dsubj], cols[:dtime], cols[:damt];
        ecg = (cols[:esubj], cols[:etime]),
        tgi = (cols[:tsubj], cols[:ttime]))
    want, muo = _je_joint_oracle_u(u, names, cols, sched)
    sampler = prepare_query(built, bound, :sampler)
    @test sampler(u) ≈ want atol = 1e-9
    like = prepare_query(built, bound, :likelihood)(u)
    prior = prepare_query(built, bound, :prior)(u)
    jac = prepare_query(built, bound, :log_jacobian)(u)
    @test like + prior + jac ≈ want atol = 1e-9
    # All five families lowered into the one program (generic Gaussian
    # `logpdf`, censored-PK `cdf` arm, three TGI cells).
    r = repr(kernel_expr(bound, assign_layout(bound)))
    for head in ("tgi_category_lpmf", "tgi_response_lpmf",
            "tgi_censored_lpdf", "cdf", "logpdf")
        @test occursin(head, r)
    end
end

@testset "D9 joint Enzyme gradient vs findiff" begin
    cols = _je_joint_columns()
    bound = bind_data(lower_rkppl(_je_joint_ast(), _je_joint_data()),
        cols)
    built = build_kernel(bound)
    names = coordinate_names(assign_layout(bound))
    θ = _je_joint_theta()
    u = _je_joint_u(names, θ.sa, θ.sp, θ.qs, θ.sd, θ.b0, θ.b1)
    sched = build_linear_pk_schedule(cols[:subj], cols[:time],
        cols[:dsubj], cols[:dtime], cols[:damt];
        ecg = (cols[:esubj], cols[:etime]),
        tgi = (cols[:tsubj], cols[:ttime]))
    want, _ = _je_joint_oracle_u(u, names, cols, sched)
    sampler = prepare_query(built, bound, :sampler)
    q = prepare_sampler(built, bound, u; backend = _JE_BACKEND)
    g = Vector{Float64}(undef, length(u))
    val, _ = sampler_value_and_gradient!(q, g, u)
    @test val ≈ want atol = 1e-9
    go = _je_findiff(w -> _je_joint_oracle_u(w, names, cols, sched)[1],
        u)
    gg = _je_findiff(sampler, u)
    @test g ≈ go rtol = 1e-6 atol = 1e-6
    @test gg ≈ go rtol = 1e-6 atol = 1e-6
end

# The joint assembly (emission mirror over bound reads): the exact
# per-line emission shapes — sched-map gathers, nadir unroll with the
# empty second segment (static ranges, the codegen freeze), QT-scale
# pre-local, censored-PK pre-local + `==` arm, three TGI plates with
# inlined literal cutoffs — with the recurrence output bound. The
# full-program reverse (recurrence under Enzyme-through-Reactant) aborts
# in XLA (`stablehlo.add` 9-vs-8) and grinds 20+ min on even the small
# program — snagged to RK root (no ecosystem precedent compiles a full
# built program; every Reactant proof here is shape-level per the W2
# precedent). Recurrence params ride native-Enzyme only (the D9 Enzyme
# test covers all 6 coords); Reactant covers the joint assembly.
@kernel _je_joint_asm_kernel(u::Vector{Float64}, reads, obs_map, ecg_map,
        tgi_map, dv, llq, qy, qt_w, cc, bb, yl) = begin
    sa::Float64 = exp(u[1])
    sp::Float64 = exp(u[2])
    qs::Float64 = exp(u[3])
    sd::Float64 = exp(u[4])
    mu = reads[obs_map]
    e = reads[ecg_map]
    r = reads[tgi_map]
    s1 = scan(r[1:3]; init = 0.0) do carry, x
        (min(carry, x), carry)
    end
    s2 = r[1:0]
    ref = vcat(s1, s2)
    qt_sd = qs .* qt_w
    pk_sc = sqrt.(sa^2 .+ (mu .* sp) .^ 2)
    pk_pw = plate(dv, mu, pk_sc, llq) do yv, lpv, sv, lov
        ifelse(yv == lov, log(normal(lpv, sv).cdf(lov)),
            normal(lpv, sv).logpdf(yv))
    end
    pk_ll::Float64 = sum(pk_pw)
    qt_pw = plate(qy, e, qt_sd) do yv, lpv, sv
        normal(lpv, sv).logpdf(yv)
    end
    qt_ll::Float64 = sum(qt_pw)
    cat_pw = plate(cc, r, ref, sd) do yv, rv, refv, sv
        tgi_category_lpmf(yv, rv, refv, -0.5, -0.3, 0.2, sv, 0.01)
    end
    cat_ll::Float64 = sum(cat_pw)
    resp_pw = plate(bb, r, ref, sd) do yv, rv, refv, sv
        tgi_response_lpmf(yv, rv, refv, -0.3, 0.2, sv, 0.01)
    end
    resp_ll::Float64 = sum(resp_pw)
    cens_pw = plate(yl, r, sd) do yv, muv, sv
        tgi_censored_lpdf(yv, muv, sv, 0.3)
    end
    cens_ll::Float64 = sum(cens_pw)
    total::Float64 = pk_ll + qt_ll + cat_ll + resp_ll + cens_ll
    return total
end

@testset "D9 joint Reactant compiled vs native (empty segment)" begin
    cols = _je_joint_columns()
    bound = bind_data(lower_rkppl(_je_joint_ast(), _je_joint_data()),
        cols)
    built = build_kernel(bound)
    names = coordinate_names(assign_layout(bound))
    θ = _je_joint_theta()
    u = _je_joint_u(names, θ.sa, θ.sp, θ.qs, θ.sd, θ.b0, θ.b1)
    sched = build_linear_pk_schedule(cols[:subj], cols[:time],
        cols[:dsubj], cols[:dtime], cols[:damt];
        ecg = (cols[:esubj], cols[:etime]),
        tgi = (cols[:tsubj], cols[:ttime]))
    lps = θ.b0 .+ θ.b1 .* cols[:age_s]
    conc = vcat([_je_oracle_auc_reads(sched, s,
        zeros(s == 1 ? sched.op_ends[1] :
            sched.op_ends[s] - sched.op_ends[s-1]),
        lps[s], lps[s], lps[s], lps[s], lps[s])[1:sched.n_reads[s]]
        for s in 1:2]...)
    u4 = [log(θ.sa), log(θ.sp), log(θ.qs), log(θ.sd)]
    asm_bound = (; reads = conc, obs_map = sched.obs_map,
        ecg_map = sched.ecg_map, tgi_map = sched.tgi_map, dv = cols[:dv],
        llq = cols[:llq], qy = cols[:qy], qt_w = cols[:qt_w],
        cc = cols[:cc], bb = cols[:bb], yl = cols[:yl])
    asm_have = (:u, :reads, :obs_map, :ecg_map, :tgi_map, :dv, :llq,
        :qy, :qt_w, :cc, :bb, :yl)
    kern = prepare(_je_joint_asm_kernel; have = asm_have, want = :total,
        bound = asm_bound)
    gen_like = prepare_query(built, bound, :likelihood)(u)
    # The assembly mirrors the emission faithfully (same math, same
    # data — the cross-check that the shape proof proves the emission).
    @test kern(u4) ≈ gen_like rtol = 1e-12
    pad = prepare_ad(_je_joint_asm_kernel, _JE_BACKEND, u4; active = :u,
        want = :total, bound = asm_bound)
    g = Vector{Float64}(undef, 4)
    val, _ = ReactiveKernels.ad_value_and_gradient!(pad, g, u4)
    @test val ≈ gen_like rtol = 1e-12
    traced = Reactant.to_rarray(u4)
    compiled = compile_ad_value_and_gradient(pad, traced)
    rval, rgrad = compiled(traced)
    @test Float64(rval) ≈ val atol = 1e-9
    @test Array(rgrad) ≈ g atol = 1e-6
end
