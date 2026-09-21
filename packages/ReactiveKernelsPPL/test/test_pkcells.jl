using DifferentiationInterface
using Distributions
using Enzyme
using LinearAlgebra: exp
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Grouped-kernel + linear-PK recurrence cell (SB-mirror of the joint
# brm2 V2 kernel). Vocabulary tests standalone (no compiler): the
# schedule transliteration, the hand-rolled matrix kernels vs
# LinearAlgebra, and the full per-subject recurrence vs the reference
# oracle. Enzyme parity + Reactant compiled-vs-native coverage follow
# below.

const _PK_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

@testset "linear-PK schedule transliteration" begin
    # Subject 1 takes a 4-dose q24 run (segment: equal amount, equal
    # interval, no intervening read) with late obs; subject 2 has a
    # same-time read+dose at t=0 (the read sees the pre-dose state)
    # plus a later obs.
    sched = build_linear_pk_schedule(
        [1, 1, 2, 2], [96.0, 120.0, 0.0, 5.0],
        [1, 1, 1, 1, 2], [0.0, 24.0, 48.0, 72.0, 0.0],
        [100.0, 100.0, 100.0, 100.0, 50.0])
    @test sched.n_subjects == 2
    @test sched.n_reads == [2, 2]
    @test sched.n_segments == 1
    @test sched.n_grouped_dose_events == 5
    # Subject 1: one segment op (dt 0: first token at t=0) then reads.
    @test sched.op_type[1:3] == [3, 1, 1]
    @test sched.op_dt[1] == 0.0
    @test sched.op_amount[1] == 100.0
    @test sched.op_interval[1] == 24.0
    @test sched.op_count[1] == 4
    @test sched.op_dt[2] == 96.0 - 72.0
    @test sched.op_dt[3] == 24.0
    @test sched.op_read_idx[1:3] == [0, 1, 2]
    @test sched.op_ends == [3, 6]
    # Subject 2: read@0 (dt 0), dose@0 (dt 0), read@5 (dt 5).
    @test sched.op_type[4:6] == [1, 2, 1]
    @test sched.op_dt[4:6] == [0.0, 0.0, 5.0]
    @test sched.op_amount[5] == 50.0
    @test sched.op_count[5] == 1
    @test sched.op_read_idx[4:6] == [1, 0, 2]
    # Per-obs-row subject-local reads + flat map (subject 2 offset by
    # R_1 = 2).
    @test sched.obs_read == [1, 2, 1, 2]
    @test sched.obs_map == [1, 2, 3, 4]
    @test sched.n_reads_total == 4
    # Simultaneous same-time doses sum exactly; zero amounts drop.
    summed = build_linear_pk_schedule(
        [1], [10.0], [1, 1, 1], [0.0, 0.0, 5.0], [30.0, 70.0, 0.0])
    @test summed.op_type == [2, 1]
    @test summed.op_amount == [100.0, 0.0]
    @test summed.n_grouped_dose_events == 1
    # Fail-closed battery (SB's validations, thin-layer error type).
    @test_throws "lengths differ" build_linear_pk_schedule(
        [1], [1.0, 2.0], [1], [0.0], [10.0])
    @test_throws "at least one observation" build_linear_pk_schedule(
        Int[], Float64[], [1], [0.0], [10.0])
    @test_throws "contiguous" build_linear_pk_schedule(
        [1, 3], [1.0, 2.0], [1], [0.0], [10.0])
    @test_throws "non-negative" build_linear_pk_schedule(
        [1], [1.0], [1], [0.0], [-10.0])
    @test_throws "observed subjects" build_linear_pk_schedule(
        [1], [1.0], [2], [0.0], [10.0])
end

# Tuple/matrix bridges (test-only): the kernels take column-major
# tuples; the oracle and LinearAlgebra references take matrices.
_pk_tup2mat(t) = reshape(collect(t), 3, 3)
_pk_mat2tup(M) = Tuple(vec(M))

_pk_test_system(lVc, lk10, lk12, lk21, lka) =
    _pk_tup2mat(ReactiveKernelsPPL.linear_pk_system_3(
        lVc + lk10, lVc, lVc + lk12, lVc + lk12 - lk21, lka))

# Reactant probes (test-only): vector-in/vector-out wrappers — the
# kernels take tuples/scalars, XLA takes arrays. The splat uses
# `@allowscalar`, scoped to the tuple construction ONLY: the traced
# region below it (expm + recurrence) stays scalar-guard-clean, and
# `@compile` example/call args are `to_rarray`-marked (plain args
# bake as constants). The cell factory closes over bound op columns
# (constants, the generated-code shape) and traces only the LP
# scalars.
function _pkc_expm_wrap(x::AbstractVector)
    t = Reactant.@allowscalar (x[1], x[2], x[3], x[4], x[5], x[6],
        x[7], x[8], x[9])
    return collect(ReactiveKernelsPPL._pk_expm3(t))
end
_pkc_read_wrap_factory(opcols) = function (lp::AbstractVector)
    s_vc, s_k10, s_k12, s_k21, s_ka =
        Reactant.@allowscalar (lp[1], lp[2], lp[3], lp[4], lp[5])
    return linear_pk_read_locs(opcols[1], opcols[2], opcols[3], opcols[4],
        opcols[5], opcols[6], s_vc, s_k10, s_k12, s_k21, s_ka)
end

@testset "hand-rolled matrix kernels vs LinearAlgebra" begin
    # Typical PK arguments plus a fast-rates edge and a weekly-dt edge.
    # Small-dt cases cover Pade degrees 3/5/7 (l1 0.0017/0.083/0.50);
    # dt=1 selects 9 and the rest 13.
    cases = ((2.3, -1.4, -0.35, -2.65, -2.08, 1.0),
        (2.3, -1.4, -0.35, -2.65, -2.08, 24.0),
        (4.0, 1.0, 1.0, 1.0, 2.0, 24.0),
        (2.3, -1.4, -0.35, -2.65, -2.08, 168.0),
        (2.3, -1.4, -0.35, -2.65, -2.08, 0.001),
        (2.3, -1.4, -0.35, -2.65, -2.08, 0.05),
        (2.3, -1.4, -0.35, -2.65, -2.08, 0.3))
    for (lVc, lk10, lk12, lk21, lka, dt) in cases
        M = _pk_test_system(lVc, lk10, lk12, lk21, lka) .* dt
        d = maximum(abs,
            collect(ReactiveKernelsPPL._pk_expm3(_pk_mat2tup(M))) - vec(exp(M)))
        @test d < 5e-13
    end
    # Static 4x4 power vs `^` (segment counts incl. non-powers of two).
    A = [0.9 0.1 0.0 1.0; 0.0 0.8 0.2 0.0; 0.1 0.0 0.7 0.0; 0.0 0.0 0.0 1.0]
    for n in (1, 2, 3, 7, 64, 100)
        @test collect(ReactiveKernelsPPL._pk_matpow4(_pk_mat2tup(A), n)) ≈
            vec(A^n) atol = 1e-12
    end
end

# Reference oracle: the SB recurrence transliterated with
# LinearAlgebra.exp/^ (independent approximant from the hand-rolled
# kernels above — agreement bounds the approximant delta).
function _pk_oracle_read_locs(sched, s, lVc, lk10, lk12, lk21, lka)
    lo = s == 1 ? 1 : sched.op_ends[s-1] + 1
    hi = sched.op_ends[s]
    A = _pk_test_system(lVc, lk10, lk12, lk21, lka)
    Vc = exp(lVc)
    R = maximum(sched.op_read_idx[lo:hi])
    read_locs = zeros(R)
    state = zeros(3)
    for j in lo:hi
        dt = sched.op_dt[j]
        dt > 0 && (state = exp(A * dt) * state)
        t = sched.op_type[j]
        if t == LINEAR_EVENT_READ
            read_locs[sched.op_read_idx[j]] = state[2] / Vc
        elseif t == LINEAR_EVENT_DOSE
            state = copy(state)
            state[1] += sched.op_amount[j]
        else
            @assert t == LINEAR_EVENT_DOSE_SEGMENT
            st = copy(state)
            st[1] += sched.op_amount[j]
            P = exp(A * sched.op_interval[j])
            aff = [P [sched.op_amount[j], 0.0, 0.0]; 0.0 0.0 0.0 1.0]
            aug = aff^(sched.op_count[j] - 1) * [st; 1.0]
            state = aug[1:3]
        end
    end
    return read_locs
end

@testset "linear_pk_read_locs vs reference oracle" begin
    sched = build_linear_pk_schedule(
        [1, 1, 2, 2], [96.0, 120.0, 0.0, 5.0],
        [1, 1, 1, 1, 2], [0.0, 24.0, 48.0, 72.0, 0.0],
        [100.0, 100.0, 100.0, 100.0, 50.0])
    for (s, lp) in ((1, (2.3, -1.4, -0.35, -2.65, -2.08)),
            (2, (2.0, -1.0, -0.5, -2.0, -1.5)))
        lo = s == 1 ? 1 : sched.op_ends[s-1] + 1
        hi = sched.op_ends[s]
        got = linear_pk_read_locs(sched.op_type[lo:hi], sched.op_dt[lo:hi],
            sched.op_amount[lo:hi], sched.op_interval[lo:hi],
            sched.op_count[lo:hi], sched.op_read_idx[lo:hi], lp...)
        @test got ≈ _pk_oracle_read_locs(sched, s, lp...) atol = 1e-12
    end
    # A read before any dose sees the zero state (subject 2, read 1).
    got2 = linear_pk_read_locs(sched.op_type[4:6], sched.op_dt[4:6],
        sched.op_amount[4:6], sched.op_interval[4:6], sched.op_count[4:6],
        sched.op_read_idx[4:6], 2.0, -1.0, -0.5, -2.0, -1.5)
    @test got2[1] == 0.0
    @test_throws "disagree in length" linear_pk_read_locs(
        [1], [0.0], [0.0], [0.0], [0], [1, 2], 1.0, 1.0, 1.0, 1.0, 1.0)
    @test_throws "unknown operation type" linear_pk_read_locs(
        [9], [0.0], [0.0], [0.0], [0], [0], 1.0, 1.0, 1.0, 1.0, 1.0)
end

# AUC oracle: SB `linear_pk_read_locs_auc_cell` transliterated with
# LinearAlgebra.exp (mass-balance identity `AUC = (given - sum(state))/CL`,
# flat [conc; auc] reads, per-subject `log_F` slice).
function _pk_oracle_read_locs_auc(sched, s, log_F, lVc, lk10, lk12, lk21, lka)
    lo = s == 1 ? 1 : sched.op_ends[s-1] + 1
    hi = sched.op_ends[s]
    A = _pk_test_system(lVc, lk10, lk12, lk21, lka)
    Vc = exp(lVc)
    CL = exp(lVc + lk10)
    R = maximum(sched.op_read_idx[lo:hi])
    reads = zeros(2R)
    state = zeros(3)
    given = 0.0
    for j in lo:hi
        dt = sched.op_dt[j]
        dt > 0 && (state = exp(A * dt) * state)
        t = sched.op_type[j]
        if t == LINEAR_EVENT_READ
            r = sched.op_read_idx[j]
            reads[r] = state[2] / Vc
            reads[R + r] = (given - sum(state)) / CL
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
    return reads
end

@testset "linear_pk_read_locs_auc vs reference oracle" begin
    sched = build_linear_pk_schedule(
        [1, 1, 2, 2], [96.0, 120.0, 0.0, 5.0],
        [1, 1, 1, 1, 2], [0.0, 24.0, 48.0, 72.0, 0.0],
        [100.0, 100.0, 100.0, 100.0, 50.0])
    lps = ((2.3, -1.4, -0.35, -2.65, -2.08), (2.0, -1.0, -0.5, -2.0, -1.5))
    for s in 1:2
        lo = s == 1 ? 1 : sched.op_ends[s-1] + 1
        hi = sched.op_ends[s]
        n = hi - lo + 1
        # Nontrivial bioavailability: dose-varying log_F on dose ops only
        # (reads carry 0.0, matching the provider's event-frame output).
        log_F = zeros(n)
        for (k, j) in enumerate(lo:hi)
            sched.op_type[j] == LINEAR_EVENT_READ ||
                (log_F[k] = 0.05 * k - 0.1)
        end
        got = linear_pk_read_locs_auc(sched.op_type[lo:hi],
            sched.op_dt[lo:hi], sched.op_amount[lo:hi],
            sched.op_interval[lo:hi], sched.op_count[lo:hi],
            sched.op_read_idx[lo:hi], log_F, lps[s]...)
        @test got ≈ _pk_oracle_read_locs_auc(sched, s, log_F, lps[s]...) atol = 1e-12
    end
    # F = 1: the conc block equals the base recurrence reads exactly.
    lo, hi = 1, sched.op_ends[1]
    base = linear_pk_read_locs(sched.op_type[lo:hi], sched.op_dt[lo:hi],
        sched.op_amount[lo:hi], sched.op_interval[lo:hi],
        sched.op_count[lo:hi], sched.op_read_idx[lo:hi], lps[1]...)
    withauc = linear_pk_read_locs_auc(sched.op_type[lo:hi], sched.op_dt[lo:hi],
        sched.op_amount[lo:hi], sched.op_interval[lo:hi],
        sched.op_count[lo:hi], sched.op_read_idx[lo:hi],
        zeros(hi - lo + 1), lps[1]...)
    R = length(base)
    @test withauc[1:R] == base
    @test_throws "disagree in length" linear_pk_read_locs_auc(
        [1], [0.0], [0.0], [0.0], [0], [1], [0.0, 0.1],
        1.0, 1.0, 1.0, 1.0, 1.0)
    @test_throws "unknown operation type" linear_pk_read_locs_auc(
        [9], [0.0], [0.0], [0.0], [0], [0], [0.0],
        1.0, 1.0, 1.0, 1.0, 1.0)
end

# --- Grouped-kernel surface + compiler tests ---

const _PKC_DATA =
    Set([:subj, :time, :dsubj, :dtime, :damt, :dv, :dv2, :age_s, :sex_s])

_pkc_lp_defs() = quote
    sigma ~ Exponential(1.0)
    b0_vc ~ Normal(0.0, 1.0)
    b1_vc ~ Normal(0.0, 1.0)
    b0_k10 ~ Normal(0.0, 1.0)
    b1_k10 ~ Normal(0.0, 1.0)
    b0_k12 ~ Normal(0.0, 1.0)
    b1_k12 ~ Normal(0.0, 1.0)
    b0_k21 ~ Normal(0.0, 1.0)
    b1_k21 ~ Normal(0.0, 1.0)
    b0_ka ~ Normal(0.0, 1.0)
    b1_ka ~ Normal(0.0, 1.0)
    log_Vc = b0_vc .+ b1_vc .* age_s
    log_k10 = b0_k10 .+ b1_k10 .* age_s
    log_k12 = b0_k12 .+ b1_k12 .* age_s
    log_k21 = b0_k21 .+ b1_k21 .* age_s
    log_ka = b0_ka .+ b1_ka .* age_s
    pk_sched = linear_pk_schedule(obs = (:subj, :time),
        dose = (:dsubj, :dtime, :damt))
end

_pkc_kernel_cell() = quote
    read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_k10, log_k12,
        log_k21, log_ka)
    mu = read_locs[pk_sched.obs_map]
    dv .~ Normal.(mu, sigma)
    mu
end

function _pkc_ast(cell = _pkc_kernel_cell())
    pre = _pkc_lp_defs()
    foro = Expr(:for, Expr(:(=), :s, Expr(:call, :(:), 1, 2)),
        Expr(:block, cell.args...))
    ker = Expr(:macrocall, Symbol("@plate"), LineNumberNode(0), :conc, foro)
    return Expr(:block, pre.args..., ker)
end

function _pkc_columns(; n_sub::Int = 2)
    Dict{Symbol,AbstractVector}(
        :subj => [1, 1, 2, 2], :time => [96.0, 120.0, 0.0, 5.0],
        :dsubj => [1, 1, 1, 1, 2], :dtime => [0.0, 24.0, 48.0, 72.0, 0.0],
        :damt => [100.0, 100.0, 100.0, 100.0, 50.0],
        :dv => [10.0, 8.0, 0.5, 7.0],
        :dv2 => [9.5, 8.2, 0.4, 7.1],
        :age_s => [35.0, 52.0])
end

function _pkc_findiff(f, u; h = cbrt(eps(Float64)))
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

@testset "grouped surface fail-closed battery" begin
    base = _pkc_lp_defs().args
    kern = (nucleus) -> Expr(:block, base...,
        Meta.parse("begin\n$nucleus\nend").args...)
    # Header shape.
    @test_throws "range is `1:<N>`" lower_rkppl(kern(
        "@plate conc for s in 2\n" *
        " dv .~ Normal.(dv, sigma)\n dv\nend"), _PKC_DATA)
    @test_throws "binds one axis variable" lower_rkppl(kern(
        "@plate conc for (s, t) in 1:2\n" *
        " dv .~ Normal.(dv, sigma)\n dv\nend"), _PKC_DATA)
    # Removed kernel-do spells the plate pointer.
    @test_throws "was removed" lower_rkppl(kern(
        "conc ~ kernel(dv, log_Vc; subjects = 2) do yy, s_vc\n" *
        " yy .~ Normal.(yy, sigma)\n yy\nend"), _PKC_DATA)
    # Responses name data columns; one `.~` per response axis.
    @test_throws "must name a response data column" lower_rkppl(kern(
        "@plate conc for s in 1:2\n" *
        " mu = dv\n nope .~ Normal.(mu, sigma)\n mu\nend"), _PKC_DATA)
    @test_throws "observed twice" lower_rkppl(kern(
        "@plate conc for s in 1:2\n" *
        " mu = dv\n dv .~ Normal.(mu, sigma)\n" *
        " dv .~ Normal.(mu, sigma)\n mu\nend"), _PKC_DATA)
    # At least one LP definition is referenced in-cell.
    @test_throws "references no LP definition" lower_rkppl(kern(
        "@plate conc for s in 1:2\n" *
        " read_locs = linear_pk_read_locs(pk_sched, sigma, sigma, sigma, " *
        "sigma, sigma)\n" *
        " mu = read_locs[pk_sched.obs_map]\n" *
        " dv .~ Normal.(mu, sigma)\n mu\nend"), _PKC_DATA)
    # Cell locals must not shadow outer names (SB write rule): data
    # shadows fail in the plate check (data is unclaimed); definition
    # shadows fail at claim time.
    @test_throws "shadows a bound data column" lower_rkppl(kern(
        "@plate conc for s in 1:2\n" *
        " dv = 1.0\n dv .~ Normal.(dv, sigma)\n dv\nend"), _PKC_DATA)
    @test_throws "defined twice" lower_rkppl(kern(
        "@plate conc for s in 1:2\n" *
        " log_Vc = 1.0\n dv .~ Normal.(dv, sigma)\n dv\nend"), _PKC_DATA)
    # Cell shape.
    @test_throws "no `.~` observation" lower_rkppl(kern(
        "@plate conc for s in 1:2\n" *
        " mu = dv\n mu\nend"), _PKC_DATA)
    @test_throws "broadcasts" lower_rkppl(kern(
        "@plate conc for s in 1:2\n" *
        " mu = dv\n dv ~ Normal(mu, sigma)\n mu\nend"), _PKC_DATA)
    @test_throws "not a cell" lower_rkppl(kern(
        "@plate conc for s in 1:2\n" *
        " mu = log_Vc[subj]\n dv .~ Normal.(mu, sigma)\n nope\nend"),
        _PKC_DATA)
    # Unknown families name the admitted five (message changed when the
    # joint families joined — the fail-closed behavior is unchanged).
    @test_throws "admits in-cell observations" lower_rkppl(kern(
        "@plate conc for s in 1:2\n" *
        " mu = log_Vc[subj]\n" *
        " dv .~ StudentT.(3.0, dv, sigma)\n mu\nend"), _PKC_DATA)
    # Joint-family arities are exact (TGI cores take sigma too).
    @test_throws "exactly 7 arguments" lower_rkppl(kern(
        "@plate conc for s in 1:2\n" *
        " mu = log_Vc[subj]\n" *
        " dv .~ TgiCategory.(dv, dv)\n mu\nend"), _PKC_DATA)
    @test_throws "exactly 4 arguments" lower_rkppl(kern(
        "@plate conc for s in 1:2\n" *
        " mu = log_Vc[subj]\n" *
        " dv .~ CensoredAddpropnormal.(dv, sigma)\n mu\nend"), _PKC_DATA)
    # No inline scale/location expressions (assignment route only):
    # complex locations spell via a pre-assignment (julianic delta).
    @test_throws "cell/model name or a numeric literal" lower_rkppl(kern(
        "@plate conc for s in 1:2\n" *
        " mu = log_Vc[subj]\n" *
        " dv .~ Normal.(dv, sigma .* w)\n mu\nend"), _PKC_DATA)
    # Schedules: declared, referenced, used.
    nosched = Expr(:block,
        filter(a -> !(a isa Expr && a.head === :(=) &&
            a.args[1] === :pk_sched), base)...,
        Meta.parse("begin\n@plate conc for s in 1:2\n" *
            " read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_Vc, " *
            "log_Vc, log_Vc, log_Vc)\n" *
            " mu = read_locs[pk_sched.obs_map]\n" *
            " dv .~ Normal.(mu, sigma)\n mu\nend\nend").args...)
    @test_throws "not declared" lower_rkppl(nosched, _PKC_DATA)
    unused = Expr(:block, base...,
        Meta.parse("begin\n@plate conc for s in 1:2\n" *
            " mu = log_Vc[subj]\n" *
            " dv .~ Normal.(mu, sigma)\n mu\nend\nend").args...)
    @test_throws "leaves schedule" lower_rkppl(unused, _PKC_DATA)
    # Schedule declaration shape.
    schedcase = (decl) -> lower_rkppl(
        Expr(:block, Meta.parse("begin\n$decl\nend").args...,
            Meta.parse("begin\nsigma ~ Exponential(1.0)\nend").args...),
        _PKC_DATA)
    @test_throws "needs `obs=" schedcase(
        "pk_sched = linear_pk_schedule(dose = (:dsubj, :dtime, :damt))")
    @test_throws "needs `dose=" schedcase(
        "pk_sched = linear_pk_schedule(obs = (:subj, :time))")
    @test_throws "keywords only" schedcase(
        "pk_sched = linear_pk_schedule(obs = (:subj, :time), " *
        "dose = (:dsubj, :dtime, :damt), extra = 1)")
    @test_throws "is not bound data" schedcase(
        "pk_sched = linear_pk_schedule(obs = (:subj, :nope), " *
        "dose = (:dsubj, :dtime, :damt))")
    @test_throws "2-tuple" schedcase(
        "pk_sched = linear_pk_schedule(obs = (:subj,), " *
        "dose = (:dsubj, :dtime, :damt))")
    @test_throws "3-tuple" schedcase(
        "pk_sched = linear_pk_schedule(obs = (:subj, :time), " *
        "dose = (:dsubj, :dtime))")
end

@testset "grouped multi-param obs nodes" begin
    low = ReactiveKernelsPPL._lower_kernel_obs
    # Unit shapes: location-first, scale second positional, params the rest.
    n = low(Meta.parse("yy .~ Normal.(mu, sigma)"), [:yy], "t", "grouped v1")
    @test (n.family, n.location, n.scale, n.params) ===
        (GaussianFam, :mu, :sigma, ())
    n = low(Meta.parse("yy .~ CensoredAddpropnormal.(mu, a, p, l)"), [:yy],
        "t", "grouped v1")
    @test n.family === CensoredAddpropnormalFam
    @test (n.location, n.scale, n.params) === (:mu, :a, (:p, :l))
    n = low(Meta.parse("yy .~ TgiCategory.(r, ref, c1, c2, c3, sg, e)"),
        [:yy], "t", "grouped v1")
    @test n.family === TgiCategoryFam
    @test (n.location, n.scale, n.params) === (:r, :ref, (:c1, :c2, :c3, :sg, :e))
    n = low(Meta.parse("yy .~ TgiResponse.(r, ref, c1, c2, sg, e)"), [:yy],
        "t", "grouped v1")
    @test n.family === TgiResponseFam
    @test (n.location, n.scale, n.params) === (:r, :ref, (:c1, :c2, :sg, :e))
    n = low(Meta.parse("yy .~ TgiCensored.(mu, sg, lq)"), [:yy],
        "t", "grouped v1")
    @test n.family === TgiCensoredFam
    @test (n.location, n.scale, n.params) === (:mu, :sg, (:lq,))
    # Panel form still admits Gaussian only.
    @test_throws "Gaussian in-cell observation only" low(
        Meta.parse("yy .~ TgiCategory.(r, ref, c1, c2, c3, sg, e)"),
        [:yy], "t")
    # End to end through surface + contract with defined names.
    prog = Expr(:block, _pkc_lp_defs().args...,
        Meta.parse("begin\nc_cr ~ Normal(0.0, 1.0)\nc_pr ~ Normal(0.0, 1.0)\n" *
            "c_pd ~ Normal(0.0, 1.0)\ntgi_sg ~ Normal(0.0, 1.0)\n" *
            "eps ~ Normal(0.0, 1.0)\n" *
            "@plate cc for s in 1:2\n" *
            " read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_Vc, " *
            "log_Vc, log_Vc, log_Vc)\n" *
            " r = read_locs[pk_sched.obs_map]\n" *
            " ref = read_locs[pk_sched.obs_map]\n" *
            " dv .~ TgiCategory.(r, ref, c_cr, c_pr, c_pd, tgi_sg, eps)\n" *
            " r\nend\nend").args...)
    plan = lower_rkppl(prog, _PKC_DATA)
    obs = only(only(plan.kernel_plates).obs)
    @test obs.family === TgiCategoryFam
    @test obs.params == (:c_cr, :c_pr, :c_pd, :tgi_sg, :eps)
end

@testset "grouped cell + structure fail-closed battery" begin
    base = _pkc_lp_defs().args
    cellcase = (cell) -> lower_rkppl(Expr(:block, base...,
            Meta.parse("begin\n@plate conc for s in 1:2\n$cell\nend\nend").args...),
        _PKC_DATA)
    fullcall = "read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_k10, " *
        "log_k12, log_k21, log_ka)"
    # Cell-call shape (contract walker).
    @test_throws "takes 6 arguments" cellcase(
        "read_locs = linear_pk_read_locs(pk_sched, log_Vc)\n" *
        "mu = read_locs[pk_sched.obs_map]\n" *
        "dv .~ Normal.(mu, sigma)\nmu")
    @test_throws "not declared" cellcase(
        "read_locs = linear_pk_read_locs(log_Vc, log_Vc, log_Vc, log_Vc, log_Vc, log_Vc)\n" *
        "mu = read_locs[pk_sched.obs_map]\n" *
        "dv .~ Normal.(mu, sigma)\nmu")
    @test_throws "not in the grouped-v1 cell vocabulary" cellcase(
        "read_locs = pk_conc(pk_sched, log_Vc, log_k10, log_k12, log_k21, log_ka)\n" *
        "mu = read_locs[pk_sched.obs_map]\n" *
        "dv .~ Normal.(mu, sigma)\nmu")
    @test_throws "compile-time handle" cellcase(
        "x = pk_sched\n$fullcall\n" *
        "mu = read_locs[pk_sched.obs_map]\n" *
        "dv .~ Normal.(mu, sigma)\nmu")
    # Gathers.
    @test_throws "must be a schedule map" cellcase(
        "$fullcall\nmu = read_locs[1]\n" *
        "dv .~ Normal.(mu, sigma)\nmu")
    @test_throws "not declared" cellcase(
        "$fullcall\nmu = read_locs[other.obs_map]\n" *
        "dv .~ Normal.(mu, sigma)\nmu")
    # Names + reductions.
    @test_throws "unknown name" cellcase(
        "$fullcall\nmu = read_locs[pk_sched.obs_map]\n" *
        "x = nope .+ 1.0\ndv .~ Normal.(mu, sigma)\nmu")
    @test_throws "does not lower in a cell" cellcase(
        "$fullcall\nmu = read_locs[pk_sched.obs_map]\n" *
        "x = sum(mu)\ndv .~ Normal.(mu, sigma)\nmu")
    # Bind-time shapes (surface + structure pass; bind proves spaces).
    bindshapes = (cell) -> bind_data(cellcase(cell), _pkc_columns())
    @test_throws "is not read-space" bindshapes(
        "$fullcall\nmu = dv[pk_sched.obs_map]\n" *
        "dv .~ Normal.(mu, sigma)\nmu")
    @test_throws "gather to the obs axis first" bindshapes(
        "$fullcall\nx = read_locs .+ 1.0\n" *
        "mu = read_locs[pk_sched.obs_map]\n" *
        "dv .~ Normal.(mu, sigma)\nmu")
    @test_throws "write the dotted form" bindshapes(
        "$fullcall\nmu = read_locs[pk_sched.obs_map]\n" *
        "x = mu + 1.0\ndv .~ Normal.(x, sigma)\nx")
    @test_throws "per-subject LP cell params or model scalars" bindshapes(
        "read_locs = linear_pk_read_locs(pk_sched, dv, log_k10, log_k12, log_k21, log_ka)\n" *
        "mu = read_locs[pk_sched.obs_map]\n" *
        "dv .~ Normal.(mu, sigma)\nmu")

    # Hand-mutated IR (panel/grouped form rules).
    panel0 = lower_rkppl(Meta.parse("begin\nsigma ~ Exponential(1.0)\n" *
            "pred ~ plate(t, dv; subjects = 2) do ts, yy\n" *
            " mu = ts\nyy .~ Normal.(mu, sigma)\nmu\nend\nend"),
        Set([:t, :dv]))
    kp_panel = only(panel0.kernel_plates)
    replate = (kp) -> validate_structure(StructuralPlan(panel0.responses,
        panel0.predictors, panel0.population_priors, panel0.parameters,
        panel0.assignments, Dict{Symbol,AbstractVector}(), 0;
        kernel_plates = [kp]))
    two_obs = KernelPlate(kp_panel.result, kp_panel.subjects,
        kp_panel.timepoints, kp_panel.slices, kp_panel.assignments,
        vcat(kp_panel.obs, kp_panel.obs), kp_panel.collected, kp_panel.label,
        kp_panel.lp_args, kp_panel.schedules)
    @test_throws "exactly one in-cell observation" replate(two_obs)
    with_lp = KernelPlate(kp_panel.result, kp_panel.subjects,
        kp_panel.timepoints, kp_panel.slices, kp_panel.assignments,
        kp_panel.obs, kp_panel.collected, kp_panel.label,
        [(:mu, :s_mu)], kp_panel.schedules)
    @test_throws "take no LP args" replate(with_lp)

    grouped0 = lower_rkppl(_pkc_ast(), _PKC_DATA)
    kp_grouped = only(grouped0.kernel_plates)
    regroup = (kp) -> validate_structure(StructuralPlan(grouped0.responses,
        grouped0.predictors, grouped0.population_priors, grouped0.parameters,
        grouped0.assignments, Dict{Symbol,AbstractVector}(), 0;
        kernel_plates = [kp]))
    sched2 = LinearPKScheduleSpec(:pk_sched2, :subj, :time, :dsubj,
        :dtime, :damt)
    two_sched = KernelPlate(kp_grouped.result, kp_grouped.subjects,
        kp_grouped.timepoints, kp_grouped.slices, kp_grouped.assignments,
        kp_grouped.obs, kp_grouped.collected, kp_grouped.label,
        kp_grouped.lp_args, vcat(kp_grouped.schedules, [sched2]))
    @test_throws "exactly one schedule" regroup(two_sched)
    with_T = KernelPlate(kp_grouped.result, kp_grouped.subjects, 2,
        kp_grouped.slices, kp_grouped.assignments, kp_grouped.obs,
        kp_grouped.collected, kp_grouped.label, kp_grouped.lp_args,
        kp_grouped.schedules)
    @test_throws "take no timepoints" regroup(with_T)
    bad_lp = KernelPlate(kp_grouped.result, kp_grouped.subjects,
        kp_grouped.timepoints, kp_grouped.slices, kp_grouped.assignments,
        kp_grouped.obs, kp_grouped.collected, kp_grouped.label,
        vcat(kp_grouped.lp_args, [(:sigma, :s_x)]), kp_grouped.schedules)
    @test_throws "is not a predictor" regroup(bad_lp)
    relinked = [PredictorSpec(:log_Vc, LogitLink,
        only(filter(p -> p.name === :log_Vc, grouped0.predictors)).terms,
        :log_Vc);
        filter(p -> p.name !== :log_Vc, grouped0.predictors)...]
    @test_throws "needs IdentityLink" validate_structure(StructuralPlan(
        grouped0.responses, relinked, grouped0.population_priors,
        grouped0.parameters, grouped0.assignments,
        Dict{Symbol,AbstractVector}(), 0;
        kernel_plates = grouped0.kernel_plates))

    # Mixed-level predictor (response + kernel share one definition).
    @test_throws "mixed-level" lower_rkppl(Expr(:block, base...,
            Meta.parse("begin\ny .~ Normal.(log_Vc, sigma)\nend").args...,
            Meta.parse("begin\n@plate conc for s in 1:2\n" *
                " read_locs = linear_pk_read_locs(pk_sched, log_Vc, " *
                "log_Vc, log_Vc, log_Vc, log_Vc)\n" *
                " mu = read_locs[pk_sched.obs_map]\n" *
                " dv .~ Normal.(mu, sigma)\n mu\nend\nend").args...),
        union(_PKC_DATA, Set([:y])))

    # Varying-backed subject predictor (varying seam): admitted since
    # VaryingEffectTerm joined the subject-term kinds (the joint model
    # feeds varying-backed LPs to its kernel). This used to fail closed
    # ("carries a") before the seam; the stale rejection was already
    # dead on main (verified: same nucleus lowers on clean 0dcf84d).
    varyok = lower_rkppl(Expr(:block, base...,
            Meta.parse("begin\nr ~ varying_effect(g, [1])\nend").args...,
            Meta.parse("begin\nb0_v2 ~ Normal(0.0, 1.0)\nend").args...,
            Meta.parse("begin\nlog_Vc2 = b0_v2 .+ r\nend").args...,
            Meta.parse("begin\n@plate conc for s in 1:2\n" *
                " read_locs = linear_pk_read_locs(pk_sched, log_Vc2, " *
                "log_Vc2, log_Vc2, log_Vc2, log_Vc2)\n" *
                " mu = read_locs[pk_sched.obs_map]\n" *
                " dv .~ Normal.(mu, sigma)\n mu\nend\nend").args...),
        union(_PKC_DATA, Set([:g])))
    @test any(t -> t.kind === VaryingEffectTerm,
        only(filter(p -> p.name === :log_Vc2, varyok.predictors)).terms)
    @test (:log_Vc2, :log_Vc2) in
        only(varyok.kernel_plates).lp_args
    # Hand-built call with a non-schedule first arg (the contract side of
    # the schedule-first rule — surface catches the undeclared case).
    kp_call = only(grouped0.kernel_plates)
    mut_assign = [nm === :read_locs ?
                  (nm => Expr(:call, :linear_pk_read_locs, :s_vc, :s_vc,
                      :s_vc, :s_vc, :s_vc, :s_vc)) : (nm => ex)
                  for (nm, ex) in kp_call.assignments]
    bad_call = KernelPlate(kp_call.result, kp_call.subjects,
        kp_call.timepoints, kp_call.slices, mut_assign, kp_call.obs,
        kp_call.collected, kp_call.label, kp_call.lp_args, kp_call.schedules)
    @test_throws "takes a declared schedule first" regroup(bad_call)
end

@testset "grouped bind fail-closed battery" begin
    unbound = lower_rkppl(_pkc_ast(), _PKC_DATA)
    bound = bind_data(unbound, _pkc_columns())
    @test bound.n_obs == 4
    @test only(bound.kernel_plates).subjects == 2
    @test bound.roles[:dv] === :response
    sched_cols = filter(c -> startswith(String(c), "pk_sched_"),
        keys(bound.columns))
    @test length(sched_cols) == 9
    # Subjects / axis / subject-column lengths.
    cols3 = _pkc_columns()
    cols3[:subj] = [1, 1, 2, 2, 3]
    cols3[:time] = [96.0, 120.0, 0.0, 5.0, 10.0]
    cols3[:dsubj] = [1, 1, 1, 1, 2, 3]
    cols3[:dtime] = [0.0, 24.0, 48.0, 72.0, 0.0, 0.0]
    cols3[:damt] = [100.0, 100.0, 100.0, 100.0, 50.0, 25.0]
    cols3[:dv] = [10.0, 8.0, 0.5, 7.0, 6.0]
    @test_throws "≠ subjects" bind_data(unbound, cols3)
    # D4 (W3c): foreign-axis response slices admit — a short FIRST
    # response fails instead: per-obs axis agreement fires in resolve
    # (the n_obs gate in bound-plan validation is the backstop).
    bad_resp = _pkc_columns()
    bad_resp[:dv] = [10.0, 8.0, 0.5]
    @test_throws "≠ response `dv` length 3" bind_data(unbound, bad_resp)
    bad_sub = _pkc_columns()
    bad_sub[:age_s] = [35.0, 52.0, 40.0]
    @test_throws "want n_sub" bind_data(unbound, bad_sub)
    missing_col = _pkc_columns()
    delete!(missing_col, :damt)
    @test_throws "is not bound" bind_data(unbound, missing_col)
    collision = _pkc_columns()
    collision[:pk_sched_op_type] = [1, 1]
    @test_throws "is reserved for schedule" bind_data(unbound, collision)
    # Hand-bound products verify by exact rebuild, never trusted.
    tampered = bind_data(unbound, _pkc_columns())
    tampered.columns[:pk_sched_op_count] .= 1
    @test_throws "is not the schedule build" validate_data(tampered)
    # Dims-key subjects resolve; leftover keys fail closed.
    keyed = lower_rkppl(Meta.parse("begin\nsigma ~ Exponential(1.0)\n" *
            "b0 ~ Normal(0.0, 1.0)\nlog_Vc = b0\n" *
            "pk_sched = linear_pk_schedule(obs = (:subj, :time), " *
            "dose = (:dsubj, :dtime, :damt))\n" *
            "@plate conc for s in 1:kernel_nsub_conc\n" *
            " read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_Vc, " *
            "log_Vc, log_Vc, log_Vc)\n" *
            " mu = read_locs[pk_sched.obs_map]\n" *
            " dv .~ Normal.(mu, sigma)\n mu\nend\nend"), _PKC_DATA)
    keyed_cols = _pkc_columns()
    delete!(keyed_cols, :age_s)
    keyed_bound = bind_data(keyed, keyed_cols;
        dims = Dict{Symbol,Int}(:kernel_nsub_conc => 2))
    @test only(keyed_bound.kernel_plates).subjects == 2
    @test_throws "not consumed" bind_data(keyed, keyed_cols;
        dims = Dict{Symbol,Int}(:kernel_nsub_conc => 2, :kernel_T_conc => 4))
end

@testset "subject-level factor LP (LevelMap over subject columns)" begin
    # A factor term over a subject column designs at n_sub rows (the
    # joint model's indication-style LP): full smoke through build.
    no_k10 = filter(_pkc_lp_defs().args) do a
        a isa LineNumberNode && return true
        a isa Expr && a.head === :(=) && a.args[1] === :log_k10 &&
            return false
        a isa Expr && a.head === :call && a.args[1] === :~ &&
            a.args[2] in (:b0_k10, :b1_k10) && return false
        return true
    end
    ast = Expr(:block, no_k10...,
        Meta.parse("begin\nc_sex[levels(sex_s)] .~ Normal.(0, 2)\nend").args...,
        Meta.parse("begin\nlog_k10 = c_sex[sex_s]\nend").args...,
        Meta.parse("begin\n@plate conc for s in 1:2\n" *
            " read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_k10, " *
            "log_k10, log_k10, log_k10)\n" *
            " mu = read_locs[pk_sched.obs_map]\n" *
            " dv .~ Normal.(mu, sigma)\n mu\nend\nend").args...)
    unbound = lower_rkppl(ast, _PKC_DATA)
    @test any(t -> t.kind === FactorTerm,
        only(filter(p -> p.name === :log_k10, unbound.predictors)).terms)
    fcols = _pkc_columns()
    fcols[:sex_s] = [1, 2]
    bound = bind_data(unbound, fcols)
    built = build_kernel(bound)
    names = coordinate_names(assign_layout(bound))
    u = zeros(length(names))
    @test isfinite(prepare_query(built, bound, :sampler)(u))
end

# Constrained point for the parity fixture (LP coefs + sigma), keyed by
# layout coordinate.
function _pkc_theta()
    Dict(:sigma => 0.5, :b0_vc => 2.0, :b1_vc => 0.01, :b0_k10 => -1.2,
        :b1_k10 => -0.005, :b0_k12 => -0.3, :b1_k12 => 0.002,
        :b0_k21 => -2.5, :b1_k21 => -0.003, :b0_ka => -2.0, :b1_ka => 0.004)
end

function _pkc_cval(θ, n)
    n === :sigma && return θ[n]
    pred, term = split(String(n), ".")
    suffix = Symbol(lowercase(pred[5:end]))
    return term == "Intercept" ? θ[Symbol(:b0_, suffix)] :
        θ[Symbol(:b1_, suffix)]
end

# Oracle density at unconstrained u (reference recurrence + independent
# Gaussian math + priors + Jacobian).
function _pkc_oracle_u(u, names, cols, sched, age)
    θu = Dict{Symbol,Float64}()
    for (n, v) in zip(names, u)
        if n === :sigma
            θu[n] = exp(v)
            continue
        end
        pred, term = split(String(n), ".")
        suffix = lowercase(pred[5:end])
        θu[Symbol(term == "Intercept" ? :b0_ : :b1_, suffix)] = v
    end
    lps = [θu[Symbol(:b0_, s)] .+ θu[Symbol(:b1_, s)] .* age
           for s in ("vc", "k10", "k12", "k21", "ka")]
    r1 = _pk_oracle_read_locs(sched, 1, lps[1][1], lps[2][1], lps[3][1],
        lps[4][1], lps[5][1])
    r2 = _pk_oracle_read_locs(sched, 2, lps[1][2], lps[2][2], lps[3][2],
        lps[4][2], lps[5][2])
    muo = vcat(r1, r2)[sched.obs_map]
    sig = θu[:sigma]
    tot = sum(logpdf.(Normal.(muo, sig), cols[:dv])) +
        logpdf(Exponential(1.0), sig) + log(sig)
    for (kk, vv) in θu
        kk === :sigma && continue
        tot += logpdf(Normal(0.0, 1.0), vv)
    end
    return tot, muo
end

@testset "grouped PK parity vs SB oracle (value + Enzyme + findiff)" begin
    unbound = lower_rkppl(_pkc_ast(), _PKC_DATA)
    @test length(unbound.predictors) == 5
    @test length(only(unbound.kernel_plates).lp_args) == 5
    cols = _pkc_columns()
    bound = bind_data(unbound, cols)
    @test bound.n_obs == 4
    built = build_kernel(bound)
    names = coordinate_names(assign_layout(bound))
    θ = _pkc_theta()
    u = [n === :sigma ? log(θ[n]) : _pkc_cval(θ, n) for n in names]
    sched = build_linear_pk_schedule(cols[:subj], cols[:time], cols[:dsubj],
        cols[:dtime], cols[:damt])
    want, muo = _pkc_oracle_u(u, names, cols, sched, cols[:age_s])
    sampler = prepare_query(built, bound, :sampler)
    @test sampler(u) ≈ want atol = 1e-9
    like = prepare_query(built, bound, :likelihood)(u)
    prior = prepare_query(built, bound, :prior)(u)
    jac = prepare_query(built, bound, :log_jacobian)(u)
    @test like + prior + jac ≈ want atol = 1e-9
    @test like ≈ sum(logpdf.(Normal.(muo, θ[:sigma]), cols[:dv])) atol = 1e-9
    # Enzyme gradient vs findiff'd oracle + findiff'd graph (each entry).
    q = prepare_sampler(built, bound, u; backend = _PK_BACKEND)
    g = Vector{Float64}(undef, length(u))
    val, _ = sampler_value_and_gradient!(q, g, u)
    @test val ≈ want atol = 1e-9
    go = _pkc_findiff(w -> _pkc_oracle_u(w, names, cols, sched,
        cols[:age_s])[1], u)
    gg = _pkc_findiff(sampler, u)
    @test g ≈ go rtol = 1e-6 atol = 1e-6
    @test gg ≈ go rtol = 1e-6 atol = 1e-6
end

@testset "multi-obs grouped cell (two response axes sum)" begin
    # Two assays over the same reads: two response slices, two in-cell
    # plates, one collected name. The joint is the plate sum.
    ast = Expr(:block, _pkc_lp_defs().args...,
        Meta.parse("begin\nsigma2 ~ Exponential(1.0)\nend").args...,
        Meta.parse("begin\n@plate conc for s in 1:2\n" *
            " read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_k10, " *
            "log_k12, log_k21, log_ka)\n" *
            " mu = read_locs[pk_sched.obs_map]\n" *
            " dv .~ Normal.(mu, sigma)\n" *
            " dv2 .~ Normal.(mu, sigma2)\n mu\nend\nend").args...)
    unbound = lower_rkppl(ast, _PKC_DATA)
    @test length(only(unbound.kernel_plates).obs) == 2
    cols = _pkc_columns()
    bound = bind_data(unbound, cols)
    @test bound.roles[:dv] === :response
    @test bound.roles[:dv2] === :response
    built = build_kernel(bound)
    names = coordinate_names(assign_layout(bound))
    θ = _pkc_theta()
    θ[:sigma2] = 0.7
    u = [n === :sigma ? log(θ[n]) :
        n === :sigma2 ? log(θ[n]) : _pkc_cval(θ, n) for n in names]
    sched = build_linear_pk_schedule(cols[:subj], cols[:time], cols[:dsubj],
        cols[:dtime], cols[:damt])
    names1 = filter(n -> n !== :sigma2, names)
    u1 = [v for (n, v) in zip(names, u) if n !== :sigma2]
    want1, muo = _pkc_oracle_u(u1, names1, cols, sched, cols[:age_s])
    want = want1 + sum(logpdf.(Normal.(muo, θ[:sigma2]), cols[:dv2])) +
        logpdf(Exponential(1.0), θ[:sigma2]) + log(θ[:sigma2])
    @test prepare_query(built, bound, :sampler)(u) ≈ want atol = 1e-9
end

@testset "grouped gather rewrite" begin
    spec = LinearPKScheduleSpec(:pk_sched, :subj, :time, :dsubj, :dtime, :damt)
    rw = ReactiveKernelsPPL._rewrite_grouped_gather
    # Schedule-map gathers rewrite to bind columns (existing behavior).
    @test rw(:(reads[pk_sched.obs_map]), spec) == :(reads[pk_sched_obs_map])
    # Plain-symbol flat indices (TGI/QT prep maps) pass through untouched.
    @test rw(:(pk_reads[tgi_read_map]), spec) == :(pk_reads[tgi_read_map])
    # Nested arithmetic recurses; bare symbols pass through.
    @test rw(:(a + b .* c), spec) == :(a + b .* c)
    @test rw(:x, spec) == :x
end

@testset "PK expm + cell Reactant parity vs native" begin
    # One static trace serves every Pade degree (the selection is
    # fully predicated — no value-specialized recompiles), then the
    # full one-subject recurrence (segment + reads) at two LP points.
    v0 = Vector{Float64}(vec(_pk_test_system(2.3, -1.4, -0.35, -2.65, -2.08) .* 24.0))
    c = Reactant.@compile _pkc_expm_wrap(Reactant.to_rarray(v0))
    for (lVc, lk10, lk12, lk21, lka, dt) in ((2.3, -1.4, -0.35, -2.65, -2.08, 0.001),
            (2.3, -1.4, -0.35, -2.65, -2.08, 0.05),
            (2.3, -1.4, -0.35, -2.65, -2.08, 0.3),
            (2.3, -1.4, -0.35, -2.65, -2.08, 1.0),
            (2.3, -1.4, -0.35, -2.65, -2.08, 24.0),
            (4.0, 1.0, 1.0, 1.0, 2.0, 24.0),
            (2.3, -1.4, -0.35, -2.65, -2.08, 168.0))
        M = _pk_test_system(lVc, lk10, lk12, lk21, lka) .* dt
        v = Vector{Float64}(vec(M))
        @test Array(c(Reactant.to_rarray(v))) ≈
            collect(ReactiveKernelsPPL._pk_expm3(_pk_mat2tup(M))) atol = 1e-12
    end
    sched = build_linear_pk_schedule(
        [1, 1, 2, 2], [96.0, 120.0, 0.0, 5.0],
        [1, 1, 1, 1, 2], [0.0, 24.0, 48.0, 72.0, 0.0],
        [100.0, 100.0, 100.0, 100.0, 50.0])
    opcols = (sched.op_type[1:3], sched.op_dt[1:3], sched.op_amount[1:3],
        sched.op_interval[1:3], sched.op_count[1:3], sched.op_read_idx[1:3])
    cell = _pkc_read_wrap_factory(opcols)
    lp0 = [2.3, -1.4, -0.35, -2.65, -2.08]
    cc = Reactant.@compile cell(Reactant.to_rarray(lp0))
    for lp in (lp0, [2.0, -1.0, -0.5, -2.0, -1.5])
        @test Array(cc(Reactant.to_rarray(lp))) ≈ Vector{Float64}(cell(lp)) atol = 1e-12
    end
end

# ── W2 event-LP seam (default-V2 log_F) ─────────────────────────────
# The joint V2 bioavailability LP (`log_F ~ 0 + op_log_dose +
# hsgp(op_log_dose; k = 5)`) over the schedule event axis, threading
# the recurrence dose branch (`op_amount * exp(log_F)`, SB's
# `linear_pk_read_locs_cell` dose branch verbatim). Generated-code
# end-to-end runs single-subject (the flat provider vector needs no
# slicing there — the per-subject expansion is the parent's parallel
# piece); multi-subject recurrence parity runs host-side through the
# same functions the generated code calls.

# V2 single-subject fixture: simultaneous 30+70 @ t=0 (the no-pre-sum
# crux — nonlinear log_F makes pre-summing invalid), a q24 100-run
# segment, two late reads.
function _pkl_columns1()
    Dict{Symbol,AbstractVector}(
        :age_s => [0.1],
        :subj => [1, 1], :time => [96.0, 120.0],
        :dsubj => [1, 1, 1, 1, 1], :dtime => [0.0, 0.0, 24.0, 48.0, 72.0],
        :damt => [30.0, 70.0, 100.0, 100.0, 100.0],
        :dv => [0.9, 0.45])
end

# V2 two-subject fixture (host-side multi-subject parity): subject 1
# as above; subject 2 a same-time read+dose at t=0 plus a later obs.
function _pkl_columns2()
    Dict{Symbol,AbstractVector}(
        :age_s => [0.1, -0.2],
        :subj => [1, 1, 2, 2], :time => [96.0, 120.0, 0.0, 5.0],
        :dsubj => [1, 1, 1, 1, 1, 2],
        :dtime => [0.0, 0.0, 24.0, 48.0, 72.0, 0.0],
        :damt => [30.0, 70.0, 100.0, 100.0, 100.0, 50.0],
        :dv => [0.9, 0.45, 0.05, 0.5])
end

const _PKL_DATA = Set([:subj, :time, :dsubj, :dtime, :damt, :dv, :age_s])

_pkl_lp_defs() = Expr(:block, _pkc_lp_defs().args...,
    Meta.parse("begin\nlog_F = linear_pk_log_f(pk_sched; k = 5)\nend").args...)

function _pkl_ast1()
    pre = _pkl_lp_defs()
    inner = quote
        read_locs = linear_pk_read_locs(pk_sched, log_F, log_Vc, log_k10,
            log_k12, log_k21, log_ka)
        mu = read_locs[pk_sched.obs_map]
        dv .~ Normal.(mu, sigma)
        mu
    end
    foro = Expr(:for, Expr(:(=), :s, Expr(:call, :(:), 1, 1)), inner)
    ker = Expr(:macrocall, Symbol("@plate"), LineNumberNode(0), :conc, foro)
    return Expr(:block, pre.args..., ker)
end

# Independent event-LP evaluator (hand-rolled from SB
# `_brm_apply_hsgp` + `brm_hsgp_sqrt_spd` 1-D, loop-based — shares no
# code with `linear_pk_event_log_f`): `slope .* x` plus the k-mode
# smooth over the (mu, L) fit.
function _pkl_ref_log_f(x::AbstractVector, slope::Real, rho::Real,
        sigma::Real, beta::AbstractVector, mu::Real, L::Real, k::Integer)
    n = length(x)
    inv_sqrt_L = 1.0 / sqrt(L)
    smooth = zeros(n)
    sscale = sigma * sqrt(rho * 2.5066282746310002)
    for b in 1:k
        lam = (b * pi / (2.0 * L))^2
        w = sscale * exp(-0.25 * rho * rho * lam) * beta[b]
        lam_sqrt = sqrt(lam)
        for i in 1:n
            smooth[i] += inv_sqrt_L * sin(lam_sqrt * (x[i] - mu + L)) * w
        end
    end
    return slope .* x .+ smooth
end

# V2 recurrence oracle: `_pk_oracle_read_locs` with SB's dose branch
# verbatim — the ONLY delta is `effective_amount = op_amount *
# exp(log_F[j])` per dosing op over this subject's slice of the flat
# event-axis vector (the `ragged(log_F, op_subject)` regroup, by
# static op ranges). The segment keeps the affine form (first dose +
# matrix power — a per-count loop would skip the state propagation
# through the segment span).
function _pkl_oracle_read_locs(sched, s::Int, logf::AbstractVector,
        log_Vc::Real, log_k10::Real, log_k12::Real, log_k21::Real,
        log_ka::Real)
    lo = s == 1 ? 1 : sched.op_ends[s - 1] + 1
    hi = sched.op_ends[s]
    A = _pk_test_system(log_Vc, log_k10, log_k12, log_k21, log_ka)
    Vc = exp(log_Vc)
    R = maximum(sched.op_read_idx[lo:hi])
    read_locs = zeros(R)
    state = zeros(3)
    for j in lo:hi
        dt = sched.op_dt[j]
        dt > 0 && (state = exp(A * dt) * state)
        t = sched.op_type[j]
        if t == LINEAR_EVENT_READ
            read_locs[sched.op_read_idx[j]] = state[2] / Vc
        else
            eff = sched.op_amount[j] * exp(logf[j])
            if t == LINEAR_EVENT_DOSE
                state = copy(state)
                state[1] += eff
            else
                @assert t == LINEAR_EVENT_DOSE_SEGMENT
                st = copy(state)
                st[1] += eff
                P = exp(A * sched.op_interval[j])
                aff = [P [eff, 0.0, 0.0]; 0.0 0.0 0.0 1.0]
                aug = aff^(sched.op_count[j] - 1) * [st; 1.0]
                state = aug[1:3]
            end
        end
    end
    return read_locs
end

@testset "event-LP op_log_dose recipe (SB dose_event_axis)" begin
    cols = _pkl_columns2()
    sched = build_linear_pk_schedule(cols[:subj], cols[:time], cols[:dsubj],
        cols[:dtime], cols[:damt]; combine_simultaneous = false)
    op_ld = linear_pk_op_log_dose(sched.op_type, sched.op_amount)
    dosing = sched.op_type .!= 1
    # Dosing rows: log(amount) - log(10000), in flat op order.
    @test op_ld[dosing] ≈
        log.(sched.op_amount[dosing]) .- log(10_000.0) atol = 1e-15
    # READ rows: the clamped dosing mean — interior by construction.
    m = sum(op_ld[dosing]) / sum(dosing)
    lo, hi = extrema(op_ld[dosing])
    @test all(op_ld[.!dosing] .== clamp(m, lo, hi))
    @test all(lo .<= op_ld .<= hi)
    @test eltype(op_ld) === Float64
    # SB edge semantics, verbatim.
    @test_throws "no dosing operation" linear_pk_op_log_dose([1, 1],
        [0.0, 0.0])
    @test_throws "non-positive" linear_pk_op_log_dose([2], [0.0])
    @test_throws "non-positive" linear_pk_op_log_dose([2], [-3.0])
    @test_throws "disagree in length" linear_pk_op_log_dose([2, 1], [1.0])
    # A degenerate single-dose stream still builds (one dosing value,
    # fill identical) — the BIND fit rejects the constant axis, not
    # the recipe.
    one = linear_pk_op_log_dose([2, 1], [50.0, 0.0])
    @test one[2] == one[1] == log(50.0) - log(10_000.0)
end

@testset "event-LP combine=false schedule (no pre-sum)" begin
    cols = _pkl_columns2()
    args = (cols[:subj], cols[:time], cols[:dsubj], cols[:dtime], cols[:damt])
    v1 = build_linear_pk_schedule(args...)
    v2 = build_linear_pk_schedule(args...; combine_simultaneous = false)
    # v1 default: simultaneous 30+70 pre-sum (then join the q24 run as
    # a 4-count segment); V2: separate zero-delta jumps (the 100-run
    # still segments — equal amount, equal interval).
    @test v1.op_type[1:3] == [3, 1, 1]
    @test v1.op_amount[1] == 100.0
    @test v1.op_count[1] == 4
    @test v2.op_type == [2, 2, 3, 1, 1, 1, 2, 1]
    @test v2.op_amount == [30.0, 70.0, 100.0, 0.0, 0.0, 0.0, 50.0, 0.0]
    @test v2.op_dt[1:2] == [0.0, 0.0]
    @test v2.op_ends == [5, 8]
    @test v2.n_grouped_dose_events == 6
    @test v2.n_segments == 1
    # Without simultaneous rows both builds agree exactly.
    plain = build_linear_pk_schedule([1], [10.0], [1], [0.0], [100.0])
    @test plain.op_type ==
        build_linear_pk_schedule([1], [10.0], [1], [0.0], [100.0];
            combine_simultaneous = false).op_type
    # Zero-amount rows drop in both paths.
    z = build_linear_pk_schedule([1], [10.0], [1, 1], [0.0, 0.0],
        [30.0, 0.0]; combine_simultaneous = false)
    @test z.op_type == [2, 1]
    @test z.n_grouped_dose_events == 1
end

@testset "linear_pk_event_log_f vs independent oracle" begin
    cols = _pkl_columns1()
    sched = build_linear_pk_schedule(cols[:subj], cols[:time], cols[:dsubj],
        cols[:dtime], cols[:damt]; combine_simultaneous = false)
    x = linear_pk_op_log_dose(sched.op_type, sched.op_amount)
    mu = sum(x) / length(x)
    L = 1.5 * maximum(abs.(x .- mu))
    # Slope-only (beta = 0 kills the smooth exactly).
    @test linear_pk_event_log_f(x, 0.5, 1.0, 0.3, zeros(5), mu, L, 5) ≈
        0.5 .* x atol = 1e-15
    # Smooth-only (slope 0) vs the loop oracle at two hyperpoints.
    for (slope, rho, sig, beta) in ((0.2, 1.0, 0.4, [0.1, -0.2, 0.3, 0.0, 0.15]),
            (-0.35, 1.7, 0.9, [1.2, 0.7, -0.5, 0.3, -1.1]))
        got = linear_pk_event_log_f(x, slope, rho, sig, beta, mu, L, 5)
        want = _pkl_ref_log_f(x, slope, rho, sig, beta, mu, L, 5)
        @test got ≈ want atol = 1e-12
        @test eltype(got) === Float64
    end
    # Lockstep with the landed Stage-B emission: the same axis as a
    # data column through `_hsgp_basis_statements`, evaluated with
    # the same hyperparameters — agreement proves the plain function
    # honors the basis/coefficient split it reuses.
    hast = quote
        hsgp_basis(:h, x; k = 5, c = 1.5)
        a ~ Normal(0, 5)
        sigma ~ Exponential(1)
        mu = a .+ hsgp(:h)
        y .~ Normal.(mu, sigma)
    end
    hplan = bind_data(lower_rkppl(hast, Set([:x, :y])),
        Dict{Symbol,AbstractVector}(:x => Vector{Float64}(x), :y => ones(5)))
    stmts = ReactiveKernelsPPL._hsgp_basis_statements(hplan)
    beta = [0.1, -0.2, 0.3, 0.0, 0.15]
    probe = Expr(:let,
        Expr(:block, :(x = $x), :(rho_h = 1.0), :(sigma_h = 0.4),
            :(beta_raw_h = $beta)),
        Expr(:block, stmts..., :_ppl_hsgp_h))
    smooth_stageb = @eval $probe
    smooth_mine = linear_pk_event_log_f(x, 0.0, 1.0, 0.4, beta, mu, L, 5)
    @test smooth_mine ≈ smooth_stageb atol = 1e-12
    # Fail-closed shapes.
    @test_throws "needs 5 hsgp" linear_pk_event_log_f(x, 0.0, 1.0, 0.4,
        zeros(4), mu, L, 5)
    @test_throws "positive" linear_pk_event_log_f(x, 0.0, 1.0, 0.4,
        zeros(5), mu, 0.0, 5)
end

@testset "linear_pk_read_locs 7-arg vs SB-branch oracle" begin
    # Multi-subject host parity (the generated path needs the
    # parent's per-subject slicing; the same function serves both).
    cols = _pkl_columns2()
    sched = build_linear_pk_schedule(cols[:subj], cols[:time], cols[:dsubj],
        cols[:dtime], cols[:damt]; combine_simultaneous = false)
    x = linear_pk_op_log_dose(sched.op_type, sched.op_amount)
    mu = sum(x) / length(x)
    L = 1.5 * maximum(abs.(x .- mu))
    logf = linear_pk_event_log_f(x, 0.2, 1.0, 0.4,
        [0.1, -0.2, 0.3, 0.0, 0.15], mu, L, 5)
    @test logf ≈ _pkl_ref_log_f(x, 0.2, 1.0, 0.4,
        [0.1, -0.2, 0.3, 0.0, 0.15], mu, L, 5) atol = 1e-12
    lps = ((2.3, -1.4, -0.35, -2.65, -2.08), (2.0, -1.0, -0.5, -2.0, -1.5))
    for s in (1, 2)
        lo = s == 1 ? 1 : sched.op_ends[s - 1] + 1
        hi = sched.op_ends[s]
        got = linear_pk_read_locs(sched.op_type[lo:hi], sched.op_dt[lo:hi],
            sched.op_amount[lo:hi], sched.op_interval[lo:hi],
            sched.op_count[lo:hi], sched.op_read_idx[lo:hi], logf[lo:hi],
            lps[s]...)
        want = _pkl_oracle_read_locs(sched, s, logf, lps[s]...)
        @test got ≈ want atol = 1e-9
    end
    # Full-recurrence V2 check at a second hyperpoint (nonzero slope
    # + live smooth move every read).
    logf2 = linear_pk_event_log_f(x, -0.35, 1.7, 0.9,
        [1.2, 0.7, -0.5, 0.3, -1.1], mu, L, 5)
    for s in (1, 2)
        lo = s == 1 ? 1 : sched.op_ends[s - 1] + 1
        hi = sched.op_ends[s]
        got = linear_pk_read_locs(sched.op_type[lo:hi], sched.op_dt[lo:hi],
            sched.op_amount[lo:hi], sched.op_interval[lo:hi],
            sched.op_count[lo:hi], sched.op_read_idx[lo:hi], logf2[lo:hi],
            lps[s]...)
        @test got ≈ _pkl_oracle_read_locs(sched, s, logf2, lps[s]...) atol = 1e-9
    end
    # v1 6-arg spelling preserved bit-identically (thin zeros wrapper).
    r6 = linear_pk_read_locs(sched.op_type[1:5], sched.op_dt[1:5],
        sched.op_amount[1:5], sched.op_interval[1:5], sched.op_count[1:5],
        sched.op_read_idx[1:5], lps[1]...)
    r7 = linear_pk_read_locs(sched.op_type[1:5], sched.op_dt[1:5],
        sched.op_amount[1:5], sched.op_interval[1:5], sched.op_count[1:5],
        sched.op_read_idx[1:5], zeros(5), lps[1]...)
    @test r6 == r7
    # Seven-column length gate names all seven.
    bad = zeros(4)
    @test_throws "all seven must match" linear_pk_read_locs(
        sched.op_type[1:5], sched.op_dt[1:5], sched.op_amount[1:5],
        sched.op_interval[1:5], sched.op_count[1:5], sched.op_read_idx[1:5],
        bad, lps[1]...)
end

@testset "event-LP surface fail-closed battery" begin
    base = _pkc_lp_defs().args
    declcase = (decl) -> lower_rkppl(Expr(:block, base...,
            Meta.parse("begin\n$decl\nend").args...,
            Meta.parse("begin\n@plate conc for s in 1:2\n" *
                "read_locs = linear_pk_read_locs(pk_sched, log_F, log_Vc, " *
                "log_k10, log_k12, log_k21, log_ka)\n" *
                "mu = read_locs[pk_sched.obs_map]\n" *
                "dv .~ Normal.(mu, sigma)\nmu\nend\nend").args...), _PKC_DATA)
    # Declaration shape.
    @test_throws "fixed to `log_F`" declcase(
        "bioav = linear_pk_log_f(pk_sched; k = 5)")
    @test_throws "one schedule handle" declcase(
        "log_F = linear_pk_log_f(pk_sched, pk_sched)")
    @test_throws "one schedule handle" declcase(
        "log_F = linear_pk_log_f(42)")
    @test_throws "declared schedule handle" declcase(
        "log_F = linear_pk_log_f(subj)")
    @test_throws "k=`/`c=` keywords only" declcase(
        "log_F = linear_pk_log_f(pk_sched; k = 5, iso = true)")
    @test_throws "positive-integer literal" declcase(
        "log_F = linear_pk_log_f(pk_sched; k = 1.5)")
    @test_throws "at least 2" declcase(
        "log_F = linear_pk_log_f(pk_sched; k = 1)")
    @test_throws "exceed 1" declcase(
        "log_F = linear_pk_log_f(pk_sched; c = 1.0)")
    # `Inf` parses as a global ref, not a literal — rejected as such
    # (a non-finite `c` literal is inexpressible; the contract pins
    # the non-finite gate directly).
    @test_throws "numeric literal" declcase(
        "log_F = linear_pk_log_f(pk_sched; c = Inf)")
    @test_throws "positive-integer literal" declcase(
        "log_F = linear_pk_log_f(pk_sched; k = true)")
    # k/c admit explicit V2 values (non-default k lowers too).
    ok = declcase("log_F = linear_pk_log_f(pk_sched; k = 3, c = 2.0)")
    @test only(ok.event_lps).k == 3
    @test only(ok.event_lps).c == 2.0
    # Call shape (7-arg form; the 6-arg v1 battery above is untouched).
    fullbase = _pkl_lp_defs().args
    cellcase = (cell) -> lower_rkppl(Expr(:block, fullbase...,
            Meta.parse("begin\n@plate conc for s in 1:2\n$cell\nend\nend").args...),
        _PKC_DATA)
    full7 = "read_locs = linear_pk_read_locs(pk_sched, log_F, log_Vc, log_k10, " *
        "log_k12, log_k21, log_ka)"
    tail = "\nmu = read_locs[pk_sched.obs_map]\n" *
        "dv .~ Normal.(mu, sigma)\nmu"
    # A malformed call never counts as a use (surface reports the
    # unused LP; the walker's count pin is proved at contract level).
    @test_throws "leaves event-LP" cellcase(
        "read_locs = linear_pk_read_locs(pk_sched, log_F, log_Vc, log_k10)" *
        tail)
    @test_throws "not declared" cellcase(
        "read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_Vc, log_k10, " *
        "log_k12, log_k21, log_ka)" * tail)
    # Linkage both ways (surface owns declared/used).
    @test_throws "not declared" lower_rkppl(Expr(:block, base...,
            Meta.parse("begin\n@plate conc for s in 1:2\n$full7$tail\nend\nend").args...),
        _PKC_DATA)
    @test_throws "leaves event-LP" lower_rkppl(Expr(:block, fullbase...,
            Meta.parse("begin\n@plate conc for s in 1:2\n" *
                "read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_k10, " *
                "log_k12, log_k21, log_ka)$tail\nend\nend").args...), _PKC_DATA)
    # The 7-arg valid call lowers (structure proves it below).
    good = cellcase(full7 * tail)
    @test length(good.event_lps) == 1
end

@testset "event-LP contract fail-closed battery" begin
    good = lower_rkppl(_pkl_ast1(), _PKL_DATA)
    strip = (p) -> StructuralPlan(p.responses, p.predictors,
        p.population_priors, p.parameters, p.assignments, p.columns, p.n_obs;
        roles = p.roles, derived = p.derived, levelmaps = p.levelmaps,
        plate_parameters = p.plate_parameters, scans = p.scans,
        varying_draws = p.varying_draws, varying_slices = p.varying_slices,
        vector_parameters = p.vector_parameters,
        spline_bases = p.spline_bases, spline_vectors = p.spline_vectors,
        hsgp_bases = p.hsgp_bases, kernel_plates = p.kernel_plates,
        r2d2_priors = p.r2d2_priors, matrices = p.matrices,
        event_lps = LinearPKEventLPSpec[])
    # 7-arg call without a declared LP: unbound `log_F` never reaches
    # codegen.
    @test_throws "not declared" validate_structure(strip(good))
    # Dangling LP (declared, never called) fails closed too.
    six = Meta.parse("begin\n@plate conc for s in 1:1\n" *
        "read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_k10, " *
        "log_k12, log_k21, log_ka)\nmu = read_locs[pk_sched.obs_map]\n" *
        "dv .~ Normal.(mu, sigma)\nmu\nend\nend")
    # (surface rejects this first — prove the contract layer directly
    # on a hand-spliced plan instead.)
    kp6 = only(lower_rkppl(Expr(:block, _pkc_lp_defs().args...,
        six.args...), _PKL_DATA).kernel_plates)
    dangling = StructuralPlan(good.responses, good.predictors,
        good.population_priors, good.parameters, good.assignments,
        good.columns, good.n_obs; roles = good.roles, derived = good.derived,
        levelmaps = good.levelmaps,
        plate_parameters = good.plate_parameters, scans = good.scans,
        varying_draws = good.varying_draws,
        varying_slices = good.varying_slices,
        vector_parameters = good.vector_parameters,
        spline_bases = good.spline_bases,
        spline_vectors = good.spline_vectors, hsgp_bases = good.hsgp_bases,
        kernel_plates = KernelPlate[kp6], r2d2_priors = good.r2d2_priors,
        matrices = good.matrices, event_lps = good.event_lps)
    @test_throws "never called" validate_structure(dangling)
    # Spec-level gates (fixed name, declared schedule, k/c, one per
    # schedule, sane fits).
    el = only(good.event_lps)
    badname = LinearPKEventLPSpec(:bioav, el.schedule, el.k, el.c, nothing,
        el.label)
    with = (els) -> StructuralPlan(good.responses, good.predictors,
        good.population_priors, good.parameters, good.assignments,
        good.columns, good.n_obs; roles = good.roles, derived = good.derived,
        levelmaps = good.levelmaps,
        plate_parameters = good.plate_parameters, scans = good.scans,
        varying_draws = good.varying_draws,
        varying_slices = good.varying_slices,
        vector_parameters = good.vector_parameters,
        spline_bases = good.spline_bases,
        spline_vectors = good.spline_vectors, hsgp_bases = good.hsgp_bases,
        kernel_plates = good.kernel_plates,
        r2d2_priors = good.r2d2_priors, matrices = good.matrices,
        event_lps = els)
    @test_throws "must be `log_F`" validate_structure(with([badname]))
    badsched = LinearPKEventLPSpec(el.name, :nope, el.k, el.c, nothing,
        el.label)
    @test_throws "is not declared" validate_structure(with([badsched]))
    badk = LinearPKEventLPSpec(el.name, el.schedule, 1, el.c, nothing,
        el.label)
    @test_throws "≥ 2" validate_structure(with([badk]))
    badc = LinearPKEventLPSpec(el.name, el.schedule, el.k, 1.0, nothing,
        el.label)
    @test_throws "exceed 1" validate_structure(with([badc]))
    badcinf = LinearPKEventLPSpec(el.name, el.schedule, el.k, Inf, nothing,
        el.label)
    @test_throws "exceed 1" validate_structure(with([badcinf]))
    # A second LP anywhere duplicates the fixed name (W2: one LP per
    # model).
    @test_throws "duplicate event-LP names" validate_structure(with([el,
        LinearPKEventLPSpec(el.name, el.schedule, 3, 1.5, nothing,
            :event_lp_other)]))
    # The walker's positional pin (hand-spliced 7-arg call with a
    # non-provider arg2 — surface rejects this shape first, so the
    # contract layer is proved directly).
    kp = only(good.kernel_plates)
    badassigns = deepcopy(kp.assignments)
    for (_, ex) in badassigns
        ex isa Expr && ex.head === :call && length(ex.args) == 8 &&
            ex.args[1] === :linear_pk_read_locs && (ex.args[3] = :s_vc)
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
    # The walker's dual-arity count pin names both shapes.
    shortassigns = deepcopy(kp.assignments)
    for (_, ex) in shortassigns
        ex isa Expr && ex.head === :call && length(ex.args) == 8 &&
            ex.args[1] === :linear_pk_read_locs &&
            deleteat!(ex.args, 6:8)
    end
    kpshort = KernelPlate(kp.result, kp.subjects, kp.timepoints, kp.slices,
        shortassigns, kp.obs, kp.collected, kp.label, kp.lp_args,
        kp.schedules)
    badcount = StructuralPlan(good.responses, good.predictors,
        good.population_priors, good.parameters, good.assignments,
        good.columns, good.n_obs; roles = good.roles, derived = good.derived,
        levelmaps = good.levelmaps,
        plate_parameters = good.plate_parameters, scans = good.scans,
        varying_draws = good.varying_draws,
        varying_slices = good.varying_slices,
        vector_parameters = good.vector_parameters,
        spline_bases = good.spline_bases,
        spline_vectors = good.spline_vectors, hsgp_bases = good.hsgp_bases,
        kernel_plates = KernelPlate[kpshort],
        r2d2_priors = good.r2d2_priors, matrices = good.matrices,
        event_lps = good.event_lps)
    @test_throws "or 7 with the event-LP" validate_structure(badcount)
    badfit = LinearPKEventLPSpec(el.name, el.schedule, el.k, el.c,
        (0.0, 0.0), el.label)
    @test_throws "L > 0" validate_structure(with([badfit]))
    # Name tables: provider + sampled names collide loudly.
    collide = StructuralPlan(good.responses, good.predictors,
        good.population_priors,
        [SampledParameter(:slope_log_F, :normal, (arg1 = 0, arg2 = 1),
            nothing, :slope_log_F); good.parameters],
        good.assignments, good.columns, good.n_obs; roles = good.roles,
        derived = good.derived, levelmaps = good.levelmaps,
        plate_parameters = good.plate_parameters, scans = good.scans,
        varying_draws = good.varying_draws,
        varying_slices = good.varying_slices,
        vector_parameters = good.vector_parameters,
        spline_bases = good.spline_bases,
        spline_vectors = good.spline_vectors, hsgp_bases = good.hsgp_bases,
        kernel_plates = good.kernel_plates,
        r2d2_priors = good.r2d2_priors, matrices = good.matrices,
        event_lps = good.event_lps)
    @test_throws "event-LP names and parameters" validate_structure(collide)
end

@testset "event-LP bind fit + data validation" begin
    unbound = lower_rkppl(_pkl_ast1(), _PKL_DATA)
    cols = _pkl_columns1()
    bound = bind_data(unbound, cols)
    # Event-LP ⟹ no-pre-sum build (separate 30/70 jumps + segment).
    @test bound.columns[:pk_sched_op_type] == [2, 2, 3, 1, 1]
    # The event-axis product: present, recipe-identical, predictor role.
    x = linear_pk_op_log_dose(bound.columns[:pk_sched_op_type],
        bound.columns[:pk_sched_op_amount])
    @test bound.columns[:pk_sched_op_log_dose] == x
    @test bound.roles[:pk_sched_op_log_dose] === :predictor
    # The fit: hand-identical (mu, L), floor below the V2 upper bound.
    el = only(bound.event_lps)
    mu = sum(x) / length(x)
    L = 1.5 * maximum(abs.(x .- mu))
    @test el.fit == (mu, L)
    floor = (4.0 * L / pi) * sqrt(log(100.0) / (5^2 - 1))
    @test floor < 2.0
    # v1 models bind byte-identical products (no event axis, pre-sum).
    v1 = bind_data(lower_rkppl(_pkc_ast(), _PKC_DATA), _pkc_columns())
    @test !haskey(v1.columns, :pk_sched_op_log_dose)
    @test v1.columns[:pk_sched_op_amount][1] == 100.0
    @test v1.columns[:pk_sched_op_count][1] == 4
    # A constant event axis (single dose) fails closed at bind fit.
    single = Dict{Symbol,AbstractVector}(
        :age_s => [0.1], :subj => [1], :time => [24.0],
        :dsubj => [1], :dtime => [0.0], :damt => [100.0], :dv => [1.0])
    one = Expr(:block, _pkl_lp_defs().args...,
        Meta.parse("begin\n@plate conc for s in 1:1\n" *
            "read_locs = linear_pk_read_locs(pk_sched, log_F, log_Vc, " *
            "log_k10, log_k12, log_k21, log_ka)\n" *
            "mu = read_locs[pk_sched.obs_map]\n" *
            "dv .~ Normal.(mu, sigma)\nmu\nend\nend").args...)
    @test_throws "degenerate" bind_data(lower_rkppl(one, _PKL_DATA), single)
    # Rebuild gates: tampered products never validate.
    tam = deepcopy(bound.columns)
    tam[:pk_sched_op_log_dose] = x .+ 1e-3
    tamp = StructuralPlan(bound.responses, bound.predictors,
        bound.population_priors, bound.parameters, bound.assignments, tam,
        bound.n_obs; roles = bound.roles, derived = bound.derived,
        levelmaps = bound.levelmaps,
        plate_parameters = bound.plate_parameters, scans = bound.scans,
        varying_draws = bound.varying_draws,
        varying_slices = bound.varying_slices,
        vector_parameters = bound.vector_parameters,
        spline_bases = bound.spline_bases,
        spline_vectors = bound.spline_vectors, hsgp_bases = bound.hsgp_bases,
        kernel_plates = bound.kernel_plates,
        r2d2_priors = bound.r2d2_priors, matrices = bound.matrices,
        event_lps = bound.event_lps)
    @test_throws "not the event-axis build" validate_data(tamp)
    badfit = LinearPKEventLPSpec(el.name, el.schedule, el.k, el.c,
        (mu + 0.5, L), el.label)
    tamf = StructuralPlan(bound.responses, bound.predictors,
        bound.population_priors, bound.parameters, bound.assignments,
        bound.columns, bound.n_obs; roles = bound.roles,
        derived = bound.derived, levelmaps = bound.levelmaps,
        plate_parameters = bound.plate_parameters, scans = bound.scans,
        varying_draws = bound.varying_draws,
        varying_slices = bound.varying_slices,
        vector_parameters = bound.vector_parameters,
        spline_bases = bound.spline_bases,
        spline_vectors = bound.spline_vectors, hsgp_bases = bound.hsgp_bases,
        kernel_plates = bound.kernel_plates,
        r2d2_priors = bound.r2d2_priors, matrices = bound.matrices,
        event_lps = [badfit])
    @test_throws "not the bind fit" validate_data(tamf)
end

@testset "event-LP layout (entries, interval rho, names)" begin
    bound = bind_data(lower_rkppl(_pkl_ast1(), _PKL_DATA), _pkl_columns1())
    lo = assign_layout(bound)
    # Formula order per LP (slope first), the HSGP triple in SB
    # `_sb_hsgp` declaration order, appended after the slice-1
    # entries so peer offsets never move.
    kinds = [(e.kind, e.name, e.size, e.transform) for e in lo.entries]
    @test kinds[end-3:end] == [(:sampled, :slope_log_F, 1, :identity),
        (:sampled, :rho_log_F, 1, :interval),
        (:sampled, :sigma_log_F, 1, :exp),
        (:hsgp, :beta_raw_log_F, 5, :identity)]
    rho_e = only(e for e in lo.entries if e.name === :rho_log_F)
    mu, L = only(bound.event_lps).fit
    floor = (4.0 * L / pi) * sqrt(log(100.0) / (5^2 - 1))
    @test rho_e.lo ≈ floor
    @test rho_e.hi == 2.0
    @test lo.total == 19  # 10 coefs + sigma + slope/rho/sigma_h + 5 beta
    names = coordinate_names(lo)
    @test names[end-7:end] == [:slope_log_F, :rho_log_F, :sigma_log_F,
        Symbol("beta_raw_log_F.1"), Symbol("beta_raw_log_F.2"),
        Symbol("beta_raw_log_F.3"), Symbol("beta_raw_log_F.4"),
        Symbol("beta_raw_log_F.5")]
    # Jacobian: sigma + interval rho + sigma_h (slope/beta identity) —
    # hand-summed from coordinates, never `logjac`.
    u = collect(range(0.05, step = 0.05, length = length(names)))
    iu = Dict(n => v for (n, v) in zip(names, u))
    rho = floor + (2.0 - floor) / (1 + exp(-iu[:rho_log_F]))
    want_jac = iu[:sigma] +
        (log(rho - floor) + log(2.0 - rho) - log(2.0 - floor)) +
        iu[:sigma_log_F]
    @test logjac(lo, u) ≈ want_jac atol = 1e-12
    nt = constrain(lo, u)
    @test Float64(nt.slope_log_F) ≈ iu[:slope_log_F]
    @test Float64(nt.rho_log_F) ≈ rho
    @test Float64(nt.sigma_log_F) ≈ exp(iu[:sigma_log_F])
    @test Vector(nt.beta_raw_log_F) ≈
        [iu[Symbol("beta_raw_log_F.$b")] for b in 1:5]
    @test unconstrain(lo, nt) ≈ u atol = 1e-12
    # Guards: stripped fits fail loud at layout and at the emitter
    # (the public path fails earlier, at validation).
    el = only(bound.event_lps)
    nofit = LinearPKEventLPSpec(el.name, el.schedule, el.k, el.c, nothing,
        el.label)
    bad = StructuralPlan(bound.responses, bound.predictors,
        bound.population_priors, bound.parameters, bound.assignments,
        bound.columns, bound.n_obs; roles = bound.roles,
        derived = bound.derived, levelmaps = bound.levelmaps,
        plate_parameters = bound.plate_parameters, scans = bound.scans,
        varying_draws = bound.varying_draws,
        varying_slices = bound.varying_slices,
        vector_parameters = bound.vector_parameters,
        spline_bases = bound.spline_bases,
        spline_vectors = bound.spline_vectors, hsgp_bases = bound.hsgp_bases,
        kernel_plates = bound.kernel_plates,
        r2d2_priors = bound.r2d2_priors, matrices = bound.matrices,
        event_lps = [nofit])
    @test_throws ContractValidationError assign_layout(bad)
    @test_throws ContractValidationError ReactiveKernelsPPL._event_lp_statements(bad)
end

function _pkl_ast2()
    pre = _pkl_lp_defs()
    inner = quote
        read_locs = linear_pk_read_locs(pk_sched, log_F, log_Vc, log_k10,
            log_k12, log_k21, log_ka)
        mu = read_locs[pk_sched.obs_map]
        dv .~ Normal.(mu, sigma)
        mu
    end
    foro = Expr(:for, Expr(:(=), :s, Expr(:call, :(:), 1, 2)), inner)
    ker = Expr(:macrocall, Symbol("@plate"), LineNumberNode(0), :conc, foro)
    return Expr(:block, pre.args..., ker)
end

# Two-subject V2 oracle: per-subject LP scalars, the flat event-axis
# vector regrouped per subject (SB `ragged(log_F, op_subject)`), the
# V2 term priors + hand-summed Jacobian (the single-subject shape —
# only the subject loop is new).
function _pkl_oracle_u2(u, names, cols, sched, age, x, fit, floor)
    θu = Dict{Symbol,Float64}()
    beta = zeros(5)
    for (n, v) in zip(names, u)
        if n === :sigma
            θu[n] = exp(v)
            continue
        end
        if n === :slope_log_F
            θu[n] = v
            continue
        end
        if n === :rho_log_F
            θu[n] = floor + (2.0 - floor) / (1 + exp(-v))
            continue
        end
        if n === :sigma_log_F
            θu[n] = exp(v)
            continue
        end
        s = String(n)
        if startswith(s, "beta_raw_log_F.")
            beta[parse(Int, split(s, ".")[2])] = v
            continue
        end
        pred, term = split(s, ".")
        suffix = lowercase(pred[5:end])
        θu[Symbol(term == "Intercept" ? :b0_ : :b1_, suffix)] = v
    end
    lps = [[θu[Symbol(:b0_, s)] + θu[Symbol(:b1_, s)] * age[sub]
            for s in ("vc", "k10", "k12", "k21", "ka")] for sub in (1, 2)]
    logf = _pkl_ref_log_f(x, θu[:slope_log_F], θu[:rho_log_F],
        θu[:sigma_log_F], beta, fit[1], fit[2], 5)
    r1 = _pkl_oracle_read_locs(sched, 1, logf, lps[1]...)
    r2 = _pkl_oracle_read_locs(sched, 2, logf, lps[2]...)
    muo = vcat(r1, r2)[sched.obs_map]
    sig = θu[:sigma]
    tot = sum(logpdf.(Normal.(muo, sig), cols[:dv])) +
        logpdf(Exponential(1.0), sig) + log(sig)
    for (kk, vv) in θu
        if kk === :sigma
            continue
        elseif kk === :slope_log_F
            tot += logpdf(Normal(0.0, 0.6676), vv)
        elseif kk === :rho_log_F
            tot += -log(2.0 - floor)
        elseif kk === :sigma_log_F
            tot += logpdf(Normal(0.0, 1.0), vv) + log(vv)
        else
            tot += logpdf(Normal(0.0, 1.0), vv)
        end
    end
    rho = θu[:rho_log_F]
    tot += log(rho - floor) + log(2.0 - rho) - log(2.0 - floor)
    tot += sum(logpdf.(Normal(0.0, 1.0), beta))
    return tot, muo
end

function _pkl_theta()
    θ = Dict{Symbol,Any}(k => v for (k, v) in _pkc_theta())
    θ[:slope_log_F] = 0.2
    θ[:rho_log_F] = 1.0
    θ[:sigma_log_F] = 0.4
    θ[:beta_raw_log_F] = [0.1, -0.2, 0.3, 0.0, 0.15]
    return θ
end

function _pkl_u(names, θ, floor)
    return map(names) do n
        n === :sigma && return log(θ[n])
        n === :slope_log_F && return θ[n]
        n === :rho_log_F && return log(θ[n] - floor) - log(2.0 - θ[n])
        n === :sigma_log_F && return log(θ[n])
        s = String(n)
        if startswith(s, "beta_raw_log_F.")
            return θ[:beta_raw_log_F][parse(Int, split(s, ".")[2])]
        end
        return _pkc_cval(θ, n)
    end
end

# V2 oracle density at unconstrained u: the independent event-LP
# evaluator + the SB-branch recurrence oracle + V2 term priors
# (slope Normal(0, 0.6676), rho Uniform constant, sigma_h Normal(0,1)
# kernel, beta Normal(0,1)) + the hand-summed Jacobian (interval edge
# in the constrained value — the bijector contract).
function _pkl_oracle_u(u, names, cols, sched, age, x, fit, floor)
    θu = Dict{Symbol,Float64}()
    beta = zeros(5)
    for (n, v) in zip(names, u)
        if n === :sigma
            θu[n] = exp(v)
            continue
        end
        if n === :slope_log_F
            θu[n] = v
            continue
        end
        if n === :rho_log_F
            θu[n] = floor + (2.0 - floor) / (1 + exp(-v))
            continue
        end
        if n === :sigma_log_F
            θu[n] = exp(v)
            continue
        end
        s = String(n)
        if startswith(s, "beta_raw_log_F.")
            beta[parse(Int, split(s, ".")[2])] = v
            continue
        end
        pred, term = split(s, ".")
        suffix = lowercase(pred[5:end])
        θu[Symbol(term == "Intercept" ? :b0_ : :b1_, suffix)] = v
    end
    lps = [θu[Symbol(:b0_, s)] + θu[Symbol(:b1_, s)] * age[1]
           for s in ("vc", "k10", "k12", "k21", "ka")]
    logf = _pkl_ref_log_f(x, θu[:slope_log_F], θu[:rho_log_F],
        θu[:sigma_log_F], beta, fit[1], fit[2], 5)
    muo = _pkl_oracle_read_locs(sched, 1, logf, lps...)[sched.obs_map]
    sig = θu[:sigma]
    tot = sum(logpdf.(Normal.(muo, sig), cols[:dv])) +
        logpdf(Exponential(1.0), sig) + log(sig)
    for (kk, vv) in θu
        if kk === :sigma
            continue
        elseif kk === :slope_log_F
            tot += logpdf(Normal(0.0, 0.6676), vv)
        elseif kk === :rho_log_F
            tot += -log(2.0 - floor)
        elseif kk === :sigma_log_F
            tot += logpdf(Normal(0.0, 1.0), vv) + log(vv)
        else
            tot += logpdf(Normal(0.0, 1.0), vv)
        end
    end
    rho = θu[:rho_log_F]
    tot += log(rho - floor) + log(2.0 - rho) - log(2.0 - floor)
    tot += sum(logpdf.(Normal(0.0, 1.0), beta))
    return tot, muo
end

@testset "single-subject event-LP end to end (value + Enzyme + findiff)" begin
    unbound = lower_rkppl(_pkl_ast1(), _PKL_DATA)
    cols = _pkl_columns1()
    bound = bind_data(unbound, cols)
    built = build_kernel(bound)
    lo = assign_layout(bound)
    names = coordinate_names(lo)
    el = only(bound.event_lps)
    mu, L = el.fit
    floor = (4.0 * L / pi) * sqrt(log(100.0) / (5^2 - 1))
    θ = _pkl_theta()
    u = _pkl_u(names, θ, floor)
    x = Vector{Float64}(bound.columns[:pk_sched_op_log_dose])
    sched = build_linear_pk_schedule(cols[:subj], cols[:time], cols[:dsubj],
        cols[:dtime], cols[:damt]; combine_simultaneous = false)
    want, muo = _pkl_oracle_u(u, names, cols, sched, cols[:age_s], x,
        el.fit, floor)
    sampler = prepare_query(built, bound, :sampler)
    @test sampler(u) ≈ want atol = 1e-9
    like = prepare_query(built, bound, :likelihood)(u)
    prior = prepare_query(built, bound, :prior)(u)
    jac = prepare_query(built, bound, :log_jacobian)(u)
    @test like + prior + jac ≈ want atol = 1e-9
    @test like ≈ sum(logpdf.(Normal.(muo, θ[:sigma]), cols[:dv])) atol = 1e-9
    # Enzyme gradient vs findiff'd oracle + findiff'd graph (each entry).
    q = prepare_sampler(built, bound, u; backend = _PK_BACKEND)
    g = Vector{Float64}(undef, length(u))
    val, _ = sampler_value_and_gradient!(q, g, u)
    @test val ≈ want atol = 1e-9
    go = _pkc_findiff(w -> _pkl_oracle_u(w, names, cols, sched,
        cols[:age_s], x, el.fit, floor)[1], u)
    gg = _pkc_findiff(sampler, u)
    @test g ≈ go rtol = 1e-6 atol = 1e-6
    @test gg ≈ go rtol = 1e-6 atol = 1e-6
end

@testset "two-subject event-LP end to end (sliced provider + Enzyme)" begin
    # The full seam through generated code: the flat provider vector
    # sliced per subject by the grouped expansion (`view(log_F,
    # lo:hi)`), each subject's recurrence on its own slice. A wrong
    # slice cannot match at 1e-9 (subject 2's dose-50 log_F differs
    # from subject 1's) — the value proof IS the slicing proof.
    unbound = lower_rkppl(_pkl_ast2(), _PKL_DATA)
    cols = _pkl_columns2()
    bound = bind_data(unbound, cols)
    built = build_kernel(bound)
    lo = assign_layout(bound)
    names = coordinate_names(lo)
    el = only(bound.event_lps)
    mu, L = el.fit
    floor = (4.0 * L / pi) * sqrt(log(100.0) / (5^2 - 1))
    θ = _pkl_theta()
    u = _pkl_u(names, θ, floor)
    x = Vector{Float64}(bound.columns[:pk_sched_op_log_dose])
    sched = build_linear_pk_schedule(cols[:subj], cols[:time], cols[:dsubj],
        cols[:dtime], cols[:damt]; combine_simultaneous = false)
    want, muo = _pkl_oracle_u2(u, names, cols, sched, cols[:age_s], x,
        el.fit, floor)
    sampler = prepare_query(built, bound, :sampler)
    @test sampler(u) ≈ want atol = 1e-9
    like = prepare_query(built, bound, :likelihood)(u)
    prior = prepare_query(built, bound, :prior)(u)
    jac = prepare_query(built, bound, :log_jacobian)(u)
    @test like + prior + jac ≈ want atol = 1e-9
    @test like ≈ sum(logpdf.(Normal.(muo, θ[:sigma]), cols[:dv])) atol = 1e-9
    q = prepare_sampler(built, bound, u; backend = _PK_BACKEND)
    g = Vector{Float64}(undef, length(u))
    val, _ = sampler_value_and_gradient!(q, g, u)
    @test val ≈ want atol = 1e-9
    go = _pkc_findiff(w -> _pkl_oracle_u2(w, names, cols, sched,
        cols[:age_s], x, el.fit, floor)[1], u)
    gg = _pkc_findiff(sampler, u)
    @test g ≈ go rtol = 1e-6 atol = 1e-6
    @test gg ≈ go rtol = 1e-6 atol = 1e-6
end

_pkl_logf_wrap_factory(x, mu, L, k) = function (s::AbstractVector,
        beta::AbstractVector)
    slope, rho, sig = Reactant.@allowscalar (s[1], s[2], s[3])
    return linear_pk_event_log_f(x, slope, rho, sig, beta, mu, L, k)
end

_pkl_read7_wrap_factory(opcols) = function (lp::AbstractVector,
        lf::AbstractVector)
    s_vc, s_k10, s_k12, s_k21, s_ka =
        Reactant.@allowscalar (lp[1], lp[2], lp[3], lp[4], lp[5])
    # Per-op `log_F[j]` reads inside the recurrence are scalar
    # iteration over the traced provider vector — inherent to the
    # seam (the flat event-axis vector feeds per-op dose scaling),
    # so scalar fallback is explicit here; the v1 factory's
    # `@allowscalar` scalar reads are the same pattern.
    return Reactant.@allowscalar linear_pk_read_locs(opcols[1], opcols[2],
        opcols[3], opcols[4], opcols[5], opcols[6], lf, s_vc, s_k10, s_k12,
        s_k21, s_ka)
end

@testset "event-LP seam Reactant parity vs native" begin
    cols = _pkl_columns1()
    sched = build_linear_pk_schedule(cols[:subj], cols[:time], cols[:dsubj],
        cols[:dtime], cols[:damt]; combine_simultaneous = false)
    x = Vector{Float64}(linear_pk_op_log_dose(sched.op_type, sched.op_amount))
    mu = sum(x) / length(x)
    L = 1.5 * maximum(abs.(x .- mu))
    # Provider: one static trace over (slope, rho, sigma) + beta.
    pv = _pkl_logf_wrap_factory(x, mu, L, 5)
    s0 = [0.2, 1.0, 0.4]
    beta0 = [0.1, -0.2, 0.3, 0.0, 0.15]
    cp = Reactant.@compile pv(Reactant.to_rarray(s0),
        Reactant.to_rarray(beta0))
    for (s, beta) in ((s0, beta0), ([-0.35, 1.7, 0.9], [1.2, 0.7, -0.5, 0.3, -1.1]))
        @test Array(cp(Reactant.to_rarray(s), Reactant.to_rarray(beta))) ≈
            Vector{Float64}(pv(s, beta)) atol = 1e-12
    end
    # 7-arg cell at a traced log_F slice (+ zeros == v1 native).
    opcols = (sched.op_type, sched.op_dt, sched.op_amount, sched.op_interval,
        sched.op_count, sched.op_read_idx)
    cell = _pkl_read7_wrap_factory(opcols)
    lp0 = [2.3, -1.4, -0.35, -2.65, -2.08]
    lf0 = Vector{Float64}(pv(s0, beta0))
    cc = Reactant.@compile cell(Reactant.to_rarray(lp0),
        Reactant.to_rarray(lf0))
    for (lp, lf) in ((lp0, lf0),
            ([2.0, -1.0, -0.5, -2.0, -1.5],
                Vector{Float64}(pv([-0.35, 1.7, 0.9],
                    [1.2, 0.7, -0.5, 0.3, -1.1]))))
        @test Array(cc(Reactant.to_rarray(lp), Reactant.to_rarray(lf))) ≈
            Vector{Float64}(cell(lp, lf)) atol = 1e-12
    end
    @test Array(cc(Reactant.to_rarray(lp0),
        Reactant.to_rarray(zeros(length(lf0))))) ≈
        Vector{Float64}(linear_pk_read_locs(opcols..., lp0...)) atol = 1e-12
end
