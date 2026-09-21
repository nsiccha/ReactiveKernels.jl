# data: subj time dsubj dtime damt dv age_s
begin
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
    pk_sched = linear_pk_schedule(obs = (:subj, :time), dose = (:dsubj, :dtime, :damt))
    log_F = linear_pk_log_f(pk_sched; k = 5)
    conc ~ kernel(dv, log_Vc, log_k10, log_k12, log_k21, log_ka; subjects = 2) do yy, s_vc, s_k10, s_k12, s_k21, s_ka
        read_locs = linear_pk_read_locs(pk_sched, log_F, s_vc, s_k10, s_k12, s_k21, s_ka)
        mu = read_locs[pk_sched.obs_map]
        yy .~ Normal.(mu, sigma)
        mu
    end
end
