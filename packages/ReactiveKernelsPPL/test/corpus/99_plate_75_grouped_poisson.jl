# data: subj time dsubj dtime damt cc age_s
begin
    b0_vc ~ Normal(0.0, 1.0)
    b1_vc ~ Normal(0.0, 1.0)
    log_Vc = b0_vc .+ b1_vc .* age_s
    pk_sched = linear_pk_schedule(obs = (:subj, :time), dose = (:dsubj, :dtime, :damt))
    read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_Vc, log_Vc, log_Vc, log_Vc)
    conc = read_locs[pk_sched.obs_map]
    @plate for i in eachindex(cc)
        lam = exp(conc[i])
        cc[i] ~ Poisson(lam)
    end
end
