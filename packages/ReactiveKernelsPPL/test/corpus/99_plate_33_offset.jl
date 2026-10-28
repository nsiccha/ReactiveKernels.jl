# data: dose dv ls
begin
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(dv)
        mu = (dose[i] / 10.0) * exp(ls[i])
        dv[i] ~ Normal(mu, sigma)
    end
end
