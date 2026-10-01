# data: t dose obs
begin
    b0 ~ Normal(0.0, 1.0)
    kappa ~ Gamma(2.0, 1000.0)
    @plate for i in eachindex(obs)
        eta = b0 * dose[i] * t[i]
        obs[i] ~ Beta(logistic(eta) * kappa, (1 - logistic(eta)) * kappa)
    end
end
