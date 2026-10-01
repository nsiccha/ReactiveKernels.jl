# data: county_idx ff yy
begin
    r ~ varying_effect(county_idx, [1])
    mu_alpha ~ Normal(0.0, 10.0)
    beta ~ Normal(0.0, 10.0)
    sigma_y ~ HalfNormal(1.0)
    @plate for i in eachindex(yy)
        yy[i] ~ Normal(mu_alpha + r[i] + beta * ff[i], sigma_y)
    end
end
