# data: t dose obs
begin
    b0 ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    nu ~ Gamma(2.0, 0.1)
    @plate for i in eachindex(obs)
        obs[i] ~ StudentT(nu, (b0 * dose[i]) * t[i], sigma)
    end
end
