# data: t dose obs
begin
    b0 ~ Normal(0.0, 1.0)
    alpha ~ Exponential(1.0)
    @plate for i in eachindex(obs)
        eta = b0 * dose[i] * t[i]
        obs[i] ~ Gamma(alpha, exp(eta) / alpha)
    end
end
