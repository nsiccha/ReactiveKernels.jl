using DifferentiationInterface
using Distributions
using Enzyme
using ReactiveKernels
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using ReactiveKernelsPPL
using Reactant
using SpecialFunctions: erfc
using Test

# QT coupling + PK/QT in-cell observations (SB-mirror of the brm2 V2 kernel
# cell at default config): builder spellings + goldens, fail-closed
# admission, prep contract, lowering shape, and standalone value/gradient
# parity of the joint QT+PK cell vs hand SB oracles (native + Enzyme +
# Reactant + central-findiff cross-check). SB provenance: bruno
# `kb-impl/Bruno-arv393-tgi` @ `3af22846`, triple
# `web-pkpd/test/stan_audit_triples/joint_brm2_default.md`.
#
# The standalone cell assumes per-subject concentration columns and the
# `qt_base`/`qt_slope` LP values as HAVE ports (grouped-kernel foundation,
# PK recurrence, and TGI block are sibling slices); the joint parity
# harness is the parent's integration step.

const _QT_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)
const _QT_GOLDEN_DIR = joinpath(@__DIR__, "qt_joint_goldens")
const _QT_HAVE =
    (:conc_ecg, :qt_y, :qt_weight, :pk_loc, :pk_y, :pk_lloq, :qb_idx, :qs_idx, :u, :reference_exposure)

function _qt_findiff(f, u; h = cbrt(eps(Float64)))
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

# Golden compare for builder/lowering output (`repr` is deterministic for
# the clean Exprs the builders emit). Re-bless deliberately with
# RKPPL_REBLESS=1 after review, never to make red green.
function _qt_check_golden(name::String, ex)
    got = repr(ex) * "\n"
    path = joinpath(_QT_GOLDEN_DIR, name * ".repr")
    if get(ENV, "RKPPL_REBLESS", "") == "1"
        mkpath(_QT_GOLDEN_DIR)
        write(path, got)
        @test true
        return nothing
    end
    @test isfile(path)
    if isfile(path)
        @test got == read(path, String)
    end
    return nothing
end

@testset "qt_loc builder SB-mirror" begin
    @test admit_qt_spine(:direct_linear) === :direct_linear
    got = qt_loc_assignment(reference_exposure = 0.8)
    # Triple §1 cell line verbatim (scalar-context `*`, baked literal).
    @test got == (:qt_loc => :(qbase + qslope * (conc[ecg_idx] ./ 0.8)))
    custom = qt_loc_assignment(:direct_linear; base = :qb, slope = :qs,
        conc = :cc, ecg_idx = :ei, reference_exposure = 2)
    @test custom == (:qt_loc => :(qb + qs * (cc[ei] ./ 2.0)))
    _qt_check_golden("qt_loc_assignment", got)
end

@testset "qt/pk obs statement spellings" begin
    @test admit_qt_obs_family(:gaussian) === :gaussian
    qt = qt_obs_statement()
    # `_joint_pk_qt_obs_statement("gaussian", "qt_scale")`, dotted per the
    # explicit-dots ruling.
    @test qt == :(qt_y .~ Normal.(qt_loc, qt_scale .* qt_weight))
    pk = pk_obs_statement()
    # V2 `censored(Normal(pk_loc, addprop(...)); lower=pk_lloq)` as the
    # dotted `censored_addpropnormal` in-cell family.
    @test pk == :(pk_y .~ CensoredAddpropnormal.(pk_loc, sigma_add, sigma_prop, pk_lloq))
    custom = pk_obs_statement(; response = :y, location = :mu,
        add = :a, prop = :p, lloq = :lo)
    @test custom == :(y .~ CensoredAddpropnormal.(mu, a, p, lo))
    _qt_check_golden("qt_obs_statement", qt)
    _qt_check_golden("pk_obs_statement", pk)
end

@testset "qt admission fail-closed battery" begin
    @test_throws "not admitted (admitted: :direct_linear" qt_loc_assignment(
        :direct_emax; reference_exposure = 0.8)
    @test_throws "not admitted (admitted: :direct_linear" qt_loc_assignment(
        :direct_logistic; reference_exposure = 0.8)
    @test_throws "not admitted (admitted: :direct_linear" admit_qt_spine(:linear)
    @test_throws "not admitted (admitted: :gaussian" qt_obs_statement(:student_t)
    @test_throws "not admitted (admitted: :gaussian" admit_qt_obs_family(:normal)
    @test_throws "reference_exposure must be" qt_loc_assignment(
        reference_exposure = 0.0)
    @test_throws "reference_exposure must be" qt_loc_assignment(
        reference_exposure = -0.8)
    @test_throws "reference_exposure must be" qt_loc_assignment(
        reference_exposure = Inf)
end

@testset "qt prep contract" begin
    good_conc = [0.58, 0.1, 0.70, 0.38]
    good_lloq = [0.1, 0.1, 0.1, 0.1]
    good_w = [1.0, 0.7071067811865476]
    @test isnothing(validate_qt_joint_prep(; reference_exposure = 0.8,
        pk_conc = good_conc, pk_lloq = good_lloq, inv_sqrt_k = good_w))
    # BLQ rows sit exactly at LLOQ (prep clamp) — admitted.
    @test isnothing(validate_qt_joint_prep(; reference_exposure = 0.8,
        pk_conc = [0.1, 0.1], pk_lloq = [0.1, 0.1], inv_sqrt_k = [1.0]))
    @test_throws "reference_exposure must be" validate_qt_joint_prep(;
        reference_exposure = 0.0, pk_conc = good_conc, pk_lloq = good_lloq,
        inv_sqrt_k = good_w)
    @test_throws "length mismatch" validate_qt_joint_prep(;
        reference_exposure = 0.8, pk_conc = [0.58], pk_lloq = good_lloq,
        inv_sqrt_k = good_w)
    @test_throws "clamp BLQ rows" validate_qt_joint_prep(;
        reference_exposure = 0.8, pk_conc = [0.58, 0.05, 0.70, 0.38],
        pk_lloq = good_lloq, inv_sqrt_k = good_w)
    @test_throws "inv_sqrt_k must be" validate_qt_joint_prep(;
        reference_exposure = 0.8, pk_conc = good_conc, pk_lloq = good_lloq,
        inv_sqrt_k = [1.0, 0.0])
    @test_throws "inv_sqrt_k must be" validate_qt_joint_prep(;
        reference_exposure = 0.8, pk_conc = good_conc, pk_lloq = good_lloq,
        inv_sqrt_k = [1.0, NaN])
end

@testset "qt/pk lowering shape" begin
    qstmts, qterm = ReactiveKernelsPPL._qt_joint_qt_likelihood_stmts(;
        response = :qt_y, location = :qt_loc, scale = :qt_scale,
        weight = :qt_weight, label = :pk_loc)
    @test qterm === :_ppl_lik_qt_joint_qt_pk_loc
    @test length(qstmts) == 3
    @test qstmts[1] == :(_ppl_qt_scale_pk_loc = qt_scale .* qt_weight)
    @test qstmts[2].head === :(=) && qstmts[2].args[1] === :_ppl_pw_qt_joint_qt_pk_loc
    @test qstmts[3] == :(_ppl_lik_qt_joint_qt_pk_loc::Float64 =
        sum(_ppl_pw_qt_joint_qt_pk_loc))
    @test occursin("logpdf", repr(qstmts[2]))
    _qt_check_golden("qt_likelihood_stmts", qstmts)

    pstmts, pterm = ReactiveKernelsPPL._qt_joint_pk_likelihood_stmts(;
        response = :pk_y, location = :pk_loc, add = :sigma_add,
        prop = :sigma_prop, lloq = :pk_lloq, label = :pk_loc)
    @test pterm === :_ppl_lik_qt_joint_pk_pk_loc
    @test length(pstmts) == 3
    @test pstmts[1] == :(_ppl_pk_scale_pk_loc =
        sqrt.(sigma_add^2 .+ (pk_loc .* sigma_prop) .^ 2))
    @test pstmts[2].head === :(=) && pstmts[2].args[1] === :_ppl_pw_qt_joint_pk_pk_loc
    @test pstmts[3] == :(_ppl_lik_qt_joint_pk_pk_loc::Float64 =
        sum(_ppl_pw_qt_joint_pk_pk_loc))
    cell = repr(pstmts[2])
    @test occursin("ifelse", cell) && occursin("cdf", cell) &&
        occursin("logpdf", cell)
    _qt_check_golden("pk_likelihood_stmts", pstmts)
end

# Standalone joint QT+PK cell: per-subject concentration columns and LP
# values arrive as HAVE ports (flat pre-gathered, the panel-v1 flatmap
# precedent); `u` packs the active model params
# [log_qt_scale, log_sigma_add, log_sigma_prop, qbase..., qslope...] with
# bind-time subject gathers.
@kernel _qt_joint_cell(conc_ecg, qt_y, qt_weight, pk_loc, pk_y, pk_lloq,
        qb_idx, qs_idx, u, reference_exposure) = begin
    qt_scale = exp(u[1])
    sigma_add = exp(u[2])
    sigma_prop = exp(u[3])
    qbase_ecg = u[qb_idx]
    qslope_ecg = u[qs_idx]
    qt_loc = qbase_ecg .+ qslope_ecg .* (conc_ecg ./ reference_exposure)
    qt_pw = plate(qt_y, qt_loc, qt_weight, qt_scale) do yv, lpv, wv, sv
        normal(lpv, sv * wv).logpdf(yv)
    end
    qt_ll::Float64 = sum(qt_pw)
    pk_pw = plate(pk_y, pk_loc, pk_lloq, sigma_add, sigma_prop) do yv, lpv, lov, sa, sp
        sv = sqrt(sa^2 + (lpv * sp)^2)
        ifelse(yv == lov, log(normal(lpv, sv).cdf(lov)),
            normal(lpv, sv).logpdf(yv))
    end
    pk_ll::Float64 = sum(pk_pw)
    joint_ll::Float64 = qt_ll + pk_ll
end

function _qt_fixture()
    # S = 2 subjects; ECG rows [1,1,2,2,2], PK rows [1,1,2,2] with PK row
    # 2 BLQ (y == LLOQ exactly, the prep clamp).
    conc_ecg = [0.5, 0.3, 0.7, 0.4, 0.6]
    qt_y = [412.0, 408.5, 395.0, 401.2, 399.8]
    qt_weight = [1.0, 0.7071067811865476, 1.0, 0.5773502691896258, 1.0]
    pk_loc = [0.55, 0.05, 0.72, 0.35]
    pk_y = [0.58, 0.1, 0.70, 0.38]
    pk_lloq = [0.1, 0.1, 0.1, 0.1]
    qb_idx = [4, 4, 5, 5, 5]
    qs_idx = [6, 6, 7, 7, 7]
    u = [log(8.0), log(0.25), log(0.15), 400.0, 395.0, 5.0, -3.0]
    reference_exposure = 0.8
    subj_ecg = [1, 1, 2, 2, 2]
    return (; conc_ecg, qt_y, qt_weight, pk_loc, pk_y, pk_lloq, qb_idx,
        qs_idx, u, reference_exposure, subj_ecg)
end

_qt_call_args(fx) = (fx.conc_ecg, fx.qt_y, fx.qt_weight, fx.pk_loc, fx.pk_y,
    fx.pk_lloq, fx.qb_idx, fx.qs_idx, fx.u, fx.reference_exposure)

# Stan `normal_lcdf_stable` (triple §3 `functions`), the exact SB BLQ-arm
# formula the hand oracle bridges to.
_qt_sb_lcdf_stable(x, loc, scale) =
    log(erfc(-(x - loc) / (scale * sqrt(2.0)))) - log(2.0)

# Hand SB oracle: Distributions.jl loops, never fused forms. QT is plain
# normal (SB model block); PK is `lower_clamping` (BLQ `==` → log-cdf).
function _qt_oracle(u, fx)
    qt_scale = exp(u[1])
    sa = exp(u[2])
    sp = exp(u[3])
    ll = 0.0
    for i in eachindex(fx.qt_y)
        s = fx.subj_ecg[i]
        loc = u[3 + s] + u[5 + s] * (fx.conc_ecg[i] / fx.reference_exposure)
        ll += logpdf(Normal(loc, qt_scale * fx.qt_weight[i]), fx.qt_y[i])
    end
    for j in eachindex(fx.pk_y)
        sv = sqrt(sa^2 + (fx.pk_loc[j] * sp)^2)
        d = Normal(fx.pk_loc[j], sv)
        ll += fx.pk_y[j] == fx.pk_lloq[j] ? logcdf(d, fx.pk_lloq[j]) :
            logpdf(d, fx.pk_y[j])
    end
    return ll
end

@testset "SB oracle bridge (lcdf_stable vs Distributions)" begin
    fx = _qt_fixture()
    sa, sp = exp(fx.u[2]), exp(fx.u[3])
    # The BLQ row (PK row 2): SB's exact formula vs the oracle's logcdf.
    sv = sqrt(sa^2 + (fx.pk_loc[2] * sp)^2)
    sb = _qt_sb_lcdf_stable(fx.pk_lloq[2], fx.pk_loc[2], sv)
    @test sb ≈ logcdf(Normal(fx.pk_loc[2], sv), fx.pk_lloq[2]) atol = 1e-12
end

@testset "joint cell value parity vs SB oracle" begin
    fx = _qt_fixture()
    k = prepare(_qt_joint_cell; have = _QT_HAVE, want = :joint_ll)
    @test k(_qt_call_args(fx)...) ≈ _qt_oracle(fx.u, fx) atol = 1e-10
end

@testset "joint cell Enzyme gradient vs findiff" begin
    fx = _qt_fixture()
    # Positional in have-order (the kernel authorizes no kwargs, so the AD
    # resolver is plain `tuple`).
    pad = prepare_ad(_qt_joint_cell, _QT_BACKEND, _qt_call_args(fx)...;
        active = :u, want = :joint_ll)
    g = Vector{Float64}(undef, length(fx.u))
    val, _ = ad_value_and_gradient!(pad, g, _qt_call_args(fx)...)
    @test val ≈ _qt_oracle(fx.u, fx) atol = 1e-10
    k = prepare(_qt_joint_cell; have = _QT_HAVE, want = :joint_ll)
    base = _qt_call_args(fx)
    f = (uu) -> k(base[1:8]..., uu, base[10])
    @test g ≈ _qt_findiff(f, fx.u) atol = 1e-6
end

@testset "joint cell Reactant parity vs native" begin
    fx = _qt_fixture()
    bound = (; conc_ecg = fx.conc_ecg, qt_y = fx.qt_y,
        qt_weight = fx.qt_weight, pk_loc = fx.pk_loc, pk_y = fx.pk_y,
        pk_lloq = fx.pk_lloq, qb_idx = fx.qb_idx, qs_idx = fx.qs_idx,
        reference_exposure = fx.reference_exposure)
    pad = prepare_ad(_qt_joint_cell, _QT_BACKEND, fx.u;
        active = :u, want = :joint_ll, bound = bound)
    @test Tuple(input.name for input in inputs(pad.kernel)) == (:u,)
    gref = Vector{Float64}(undef, length(fx.u))
    vref, _ = ad_value_and_gradient!(pad, gref, fx.u)
    @test vref ≈ _qt_oracle(fx.u, fx) atol = 1e-10
    traced_u = Reactant.to_rarray(fx.u)
    compiled = compile_ad_value_and_gradient(pad, traced_u)
    value, gradient = compiled(traced_u)
    @test Float64(value) ≈ vref atol = 1e-9
    @test Array(gradient) ≈ gref atol = 1e-6
end

# In-graph `conc[ecg_idx]` gather (the builder's exposure gather): value vs
# the Julia gather, gradient wrt the full concentration vector vs findiff.
@kernel _qt_gather_cell(conc_full, ecg_idx) = begin
    gathered = conc_full[ecg_idx]
    total::Float64 = sum(gathered)
end

@testset "exposure gather parity" begin
    conc_full = [0.0, 0.5, 0.3, 0.7, 0.0, 0.4, 0.6]
    ecg_idx = [2, 3, 4, 6, 7]
    kv = prepare(_qt_gather_cell; have = (:conc_full, :ecg_idx),
        want = (:gathered, :total))
    gathered, total = kv(conc_full, ecg_idx)
    @test gathered == conc_full[ecg_idx]
    @test total == sum(conc_full[ecg_idx])
    gad = prepare_ad(_qt_gather_cell, _QT_BACKEND, conc_full, ecg_idx;
        active = :conc_full, want = :total)
    g = Vector{Float64}(undef, length(conc_full))
    val, _ = ad_value_and_gradient!(gad, g, conc_full, ecg_idx)
    @test val == sum(conc_full[ecg_idx])
    want = zeros(length(conc_full))
    for i in ecg_idx
        want[i] += 1.0
    end
    @test g == want
    kt = prepare(_qt_gather_cell; have = (:conc_full, :ecg_idx), want = :total)
    @test g ≈ _qt_findiff((cc) -> kt(cc, ecg_idx), conc_full) atol = 1e-8
end
