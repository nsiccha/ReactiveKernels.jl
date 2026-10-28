# data: t dose obs
begin
    b0 ~ Normal(0.0, 1.0)
    phi ~ Exponential(1.0)
    @plate for i in eachindex(obs)
        obs[i] ~ NegativeBinomial2(exp(b0 * dose[i] * t[i]), phi)
    end
end
