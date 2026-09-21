# data: subject male age_yr weight_kg indication qt_prolonging_drug_ongoing subj time dsubj dtime damt pk_conc pk_lloq esubj etime ecg_y qt_w tsubj ttime_h tgi_t T0_data tgi_cc
begin
standardize_age_yr = (age_yr .- mean(age_yr)) ./ std(age_yr)
standardize_weight_kg = (weight_kg .- mean(weight_kg)) ./ std(weight_kg)
a_Vc ~ Normal(2.302585092994046, 0.8)
b_Vc_male ~ Normal(0.0, 0.1)
b_Vc_age ~ Normal(0.0, 0.1)
b_Vc_wt ~ Normal(0.0, 0.1)
c_Vc[levels(indication)[2:end]] .~ Normal.(0.0, 1.0)
a_k10 ~ Normal(-1.405170185988091, 0.8)
b_k10_male ~ Normal(0.0, 0.1)
b_k10_age ~ Normal(0.0, 0.1)
b_k10_wt ~ Normal(0.0, 0.1)
c_k10[levels(indication)[2:end]] .~ Normal.(0.0, 1.0)
a_k12 ~ Normal(-0.34657359027997265, 2.0)
c_k12[levels(indication)[2:end]] .~ Normal.(0.0, 1.0)
a_k21 ~ Normal(-2.649158683274018, 2.0)
c_k21[levels(indication)[2:end]] .~ Normal.(0.0, 1.0)
a_ka ~ Normal(-2.0794415416798357, 0.8)
c_ka[levels(indication)[2:end]] .~ Normal.(0.0, 1.0)
a_qb ~ Normal(0.0, 0.8333333333333334)
b_qb_male ~ Normal(0.0, 0.8333333333333334)
b_qb_age ~ Normal(0.0, 0.8333333333333334)
b_qb_ongoing ~ Normal(0.0, 0.8333333333333334)
c_qb[levels(indication)[2:end]] .~ Normal.(0.0, 0.8333333333333334)
a_qs ~ Normal(0.0, 0.8333333333333334)
c_qs[levels(indication)[2:end]] .~ Normal.(0.0, 0.8333333333333334)
a_tg ~ Normal(0.0, 1.0)
a_tkd ~ Normal(0.0, 1.5)
sigma_add ~ Exponential(0.25)
sigma_prop ~ Exponential(0.25)
qt_scale ~ LogNormal(0.0, 1.0)
tgi_sigma ~ LogNormal(-2.0402208285265546, 0.5)
tgi_c_cr ~ truncated(Normal(-2.3, 1.0), -Inf, -0.6931471805599453)
d_p ~ varying_draws(subject, [1, 1, 1, 1, 1, 1, 1]; eta = 2.0)
r_Vc ~ varying_slice(d_p, 1)
r_k10 ~ varying_slice(d_p, 2)
r_k12 ~ varying_slice(d_p, 3)
r_k21 ~ varying_slice(d_p, 4)
r_ka ~ varying_slice(d_p, 5)
r_qb ~ varying_slice(d_p, 6)
r_qs ~ varying_slice(d_p, 7)
d_tg ~ varying_draws(subject, [1, 1]; eta = 2.0)
r_tg ~ varying_slice(d_tg, 1)
r_tkd ~ varying_slice(d_tg, 2)
log_Vc = a_Vc .+ b_Vc_male .* male .+ b_Vc_age .* standardize_age_yr .+ b_Vc_wt .* standardize_weight_kg .+ c_Vc[indication] .+ r_Vc
log_k10 = a_k10 .+ b_k10_male .* male .+ b_k10_age .* standardize_age_yr .+ b_k10_wt .* standardize_weight_kg .+ c_k10[indication] .+ r_k10
log_k12 = a_k12 .+ c_k12[indication] .+ r_k12
log_k21 = a_k21 .+ c_k21[indication] .+ r_k21
log_ka = a_ka .+ c_ka[indication] .+ r_ka
qt_base = a_qb .+ b_qb_male .* male .+ b_qb_age .* standardize_age_yr .+ b_qb_ongoing .* qt_prolonging_drug_ongoing .+ c_qb[indication] .+ r_qb
qt_slope = a_qs .+ c_qs[indication] .+ r_qs
log_tgi_kg = a_tg .+ r_tg
log_tgi_kd = a_tkd .+ r_tkd
tgi_baseline_time = T0_data
pk_sched = linear_pk_schedule(obs = (:subj, :time),
    dose = (:dsubj, :dtime, :damt), ecg = (:esubj, :etime),
    tgi = (:tsubj, :ttime_h))
log_F = linear_pk_log_f(pk_sched; k = 5)
        @plate pk_loc for s in 1:kernel_nsub_pk_loc
pk_reads = linear_pk_read_locs_auc(pk_sched, log_F, log_Vc,
    log_k10, log_k12, log_k21, log_ka)
conc = pk_reads[pk_sched.conc_map]
mu = conc[pk_sched.obs_map]
conc_ecg = conc[pk_sched.ecg_map]
qbase_rows = qt_base[esubj]
qslope_rows = qt_slope[esubj]
qt_loc = qbase_rows .+ qslope_rows .* (conc_ecg ./ 0.8)
qt_sd = qt_scale .* qt_w
tgi_exposure = pk_reads[pk_sched.tgi_auc_map] ./ 140.62960372536338
tgi_kg_rows = exp.(log_tgi_kg[tsubj])
tgi_kd_rows = exp.(log_tgi_kd[tsubj])
tgi_r = tgi_kg_rows .* tgi_t .- tgi_kd_rows .* tgi_exposure
tgi_t0_rows = tgi_baseline_time[tsubj]
tgi_change = tgi_r .- tgi_kg_rows .* tgi_t0_rows
tgi_ref = tgi_segmented_nadir(tgi_change, pk_sched_tgi_seg_ends)
            tgi_sd = 1.4142135623730951 * tgi_sigma
        pk_conc .~ CensoredAddpropnormal.(mu, sigma_add, sigma_prop, pk_lloq)
        ecg_y .~ Normal.(qt_loc, qt_sd)
        tgi_cc .~ TgiCategory.(tgi_change, tgi_ref, tgi_c_cr, -0.6931471805599453, 0.4054651081081644, tgi_sd, 0.01)
        mu
    end
end
