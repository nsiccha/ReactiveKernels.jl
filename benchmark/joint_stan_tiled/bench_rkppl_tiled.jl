# RKPPL side of the Stan-vs-RKPPL speed comparison (throwaway probe).
# Tiles the joint fixture K times (subject replicas, same order as tile.py:
# replica-major: S1,S2,S3,S1,S2,S3,...).
using ReactiveKernels
using ReactiveKernelsPPL
using DifferentiationInterface
using Enzyme
using Printf
using Random

const PPLT = "/home/n/.local/state/kb-agents-worktrees/ReactiveKernels-brm-tgi/packages/ReactiveKernelsPPL/test"
include(joinpath(PPLT, "parity", "joint_parity_fixture.jl"))

K = parse(Int, get(ENV, "TILE_K", "3"))

"""Remap subject-id vector for K replicas (replica-major global ids)."""
function _tile_ids(v::AbstractVector{<:Integer}, K)
    return vcat([v .+ 3 * (r - 1) for r in 1:K]...)
end
_tile_vals(v::AbstractVector, K) = vcat([v for _ in 1:K]...)

function tiled_columns(K)
    subj3 = 1:3
    names = ["S$(s)r$(r)" for r in 1:K for s in subj3]
    cols = Dict{Symbol,AbstractVector}(
        :subject => names,
        :male => _tile_vals(_FX_MALE, K),
        :age_yr => _tile_vals(_FX_AGE_YR, K),
        :weight_kg => _tile_vals(_FX_WEIGHT_KG, K),
        :indication => _tile_vals(_FX_INDICATION, K),
        :qt_prolonging_drug_ongoing => _tile_vals(_FX_COMED, K),
        :subj => _tile_ids(_FX_OBS_SUBJ, K),
        :time => _tile_vals(_FX_OBS_TIME, K),
        :dsubj => _tile_ids(_FX_DOSE_SUBJ, K),
        :dtime => _tile_vals(_FX_DOSE_TIME, K),
        :damt => _tile_vals(_FX_DOSE_AMT, K),
        :pk_conc => _tile_vals(
            max.(_FX_OBS_VALUE, [0.1, 0.1, 0.05, 0.1, 0.05, 0.05, 0.05]), K),
        :pk_lloq => _tile_vals([0.1, 0.1, 0.05, 0.1, 0.05, 0.05, 0.05], K),
        :T0_data => _tile_vals([0.0, -48.0 / 1344.0, 0.0], K),
    )
    # QT / TGI frames: tile then sort by (subj, time) per the D8 rule.
    esubj = _tile_ids(_FX_QT_SUBJ, K)
    etime = _tile_vals(_FX_QT_TIME, K)
    qperm = sortperm(collect(zip(esubj, etime)))
    cols[:esubj] = esubj[qperm]
    cols[:etime] = etime[qperm]
    cols[:ecg_y] = _tile_vals(_FX_QT_Y, K)[qperm]
    cols[:qt_w] = _tile_vals(_FX_QT_INV_SQRT_K, K)[qperm]
    tsubj = _tile_ids(_FX_TGI_SUBJ, K)
    ttime = _tile_vals(_FX_TGI_TIME, K)
    tperm = sortperm(collect(zip(tsubj, ttime)))
    cols[:tsubj] = tsubj[tperm]
    cols[:ttime_h] = ttime[tperm]
    cols[:tgi_t] = ttime[tperm] ./ 1344.0
    cols[:tgi_y] = log.(_tile_vals(_FX_TGI_VALUE_CONT, K)[tperm])
    return cols
end

cols = tiled_columns(K)
plan = final_plan("continuous")
t_bind = @elapsed bound =
    bind_data(plan, cols; dims = Dict(:kernel_nsub_pk_loc => 3K))
lay = assign_layout(bound)
@printf("K=%d n_obs=%d n_subj=%d u_dim=%d\n", K, bound.n_obs, 3K, lay.total)
t_build = @elapsed k = build_kernel(bound)
post_q = prepare_query(k, bound, :sampler)
u = Vector{Float64}(0.1 .* randn(Xoshiro(20260917), lay.total))
Base.invokelatest(post_q, u) # warmup (compile)
lp = Base.invokelatest(post_q, u)
@printf("lp=%.6f finite=%s\n", lp, isfinite(lp))
t_eval = @elapsed for _ in 1:20
    Base.invokelatest(post_q, u)
end
t_eval /= 20
be = AutoEnzyme(; mode = Enzyme.Reverse)
t_prep = @elapsed q = prepare_sampler(k, bound, u; backend = be)
g = similar(u)
sampler_value_and_gradient!(q, g, u) # warmup
@printf("grad_norm=%.4f all_finite=%s\n", sqrt(sum(abs2, g)), all(isfinite, g))
t_grad = @elapsed for _ in 1:10
    sampler_value_and_gradient!(q, g, u)
end
t_grad /= 10
@printf("RKPPL K=%d bind=%.2fs build=%.2fs eval=%.5fs gradprep=%.2fs grad=%.5fs\n",
    K, t_bind, t_build, t_eval, t_prep, t_grad)
