# data: county_id county_idx time dsubj dtime damt ff yy
begin
    r ~ varying_effect(county_id, [1])
    mu_alpha ~ Normal(0.0, 10.0)
    beta ~ Normal(0.0, 10.0)
    sigma_y ~ HalfNormal(1.0)
    alpha = mu_alpha .+ r
    cy = linear_pk_schedule(obs = (:county_idx, :time),
        dose = (:dsubj, :dtime, :damt))
    @plate radon for s in 1:8
        aa = alpha[county_idx]
        mu = aa .+ beta .* ff
        yy .~ Normal.(mu, sigma_y)
        mu
    end
end
