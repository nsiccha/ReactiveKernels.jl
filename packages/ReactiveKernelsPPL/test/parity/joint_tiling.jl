# Subject-tiling of the joint parity fixture: K replicas of the 3-subject
# fixture, replica-major (S1,S2,S3,S1,S2,S3,...), the same order
# `benchmark/joint_stan_tiled/tile.py` produces for the Stan side.  Shared by
# the tiled Stan-vs-RKPPL benchmark (`bench_rkppl_tiled.jl`) and the Reactant
# ladder tests (`test_reactant_joint.jl`), so both sides bind the SAME columns.
# Requires `joint_parity_fixture.jl` (the `_FX_*` constants) to be loaded.

"""Remap subject-id vector for K replicas (replica-major global ids)."""
function _tile_ids(v::AbstractVector{<:Integer}, K)
    return vcat([v .+ 3 * (r - 1) for r in 1:K]...)
end
_tile_vals(v::AbstractVector, K) = vcat([v for _ in 1:K]...)

"""
    tiled_columns(K) -> Dict{Symbol,AbstractVector}

Raw data columns of the joint fixture tiled `K` times (`3K` subjects,
`7K` PK observations).  QT / TGI frames are tiled then sorted by
`(subj, time)` (the D8 rule); every other frame keeps replica-major order.
Bind with `dims = Dict(:kernel_nsub_pk_loc => 3K)`.
"""
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
