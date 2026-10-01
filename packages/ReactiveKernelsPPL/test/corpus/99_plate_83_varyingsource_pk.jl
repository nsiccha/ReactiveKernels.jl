# data: subj time dsubj dtime damt treatment dose_x gp_weights age_s dv
begin
    sigma ~ Exponential(1.0)
    b_vc ~ Normal(0.0, 1.0)
    b_age ~ Normal(0.0, 1.0)
    b_k10 ~ Normal(0.0, 1.0)
    b_k12 ~ Normal(0.0, 1.0)
    b_k21 ~ Normal(0.0, 1.0)
    b_rate ~ Normal(0.0, 1.0)
    b_mode ~ Normal(0.0, 1.0)
    d_rate ~ Normal(0.0, 1.0)
    d_mode ~ Normal(0.0, 1.0)
    d_f ~ Normal(0.0, 1.0)
    dose_slope ~ Normal(0.0, 1.0)
    conc_slope ~ Normal(0.0, 1.0)
    log_Vc = b_vc .+ b_age .* age_s
    log_k10 = b_k10
    log_k12 = b_k12
    log_k21 = b_k21
    log_rate = b_rate
    log_mode = b_mode
    vs = varyingsource_pk_schedule(obs = (:subj, :time),
        dose = (:dsubj, :dtime, :damt, :treatment))
    rate_mod = d_rate .* dose_x
    mode_mod = d_mode .* dose_x
    f_mod = d_f .* dose_x
    reads = varyingsource_pk_read_locs(vs, rate_mod, mode_mod, f_mod,
        gp_weights, dose_slope, conc_slope, log_Vc, log_k10, log_k12,
        log_k21, log_rate, log_mode)
    conc = reads[vs.obs_map]
    @plate for i in eachindex(dv)
        dv[i] ~ Normal(conc[i], sigma)
    end
end
