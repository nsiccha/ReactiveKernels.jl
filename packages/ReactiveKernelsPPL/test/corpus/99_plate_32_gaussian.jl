# data: t dose obs
begin
    sigma ~ Exponential(1.0)
    b0 ~ Normal(0.0, 1.0)
    @plate for i in eachindex(obs)
        mu = (b0 * dose[i]) * t[i]
        obs[i] ~ Normal(mu, sigma)
    end
end
