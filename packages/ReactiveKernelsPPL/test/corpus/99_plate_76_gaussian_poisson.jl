# data: x1 y1 x2 y2
begin
    b0 ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    @plate for i in eachindex(y1)
        y1[i] ~ Normal(b0 * x1[i], sigma)
    end
    @plate for j in eachindex(y2)
        y2[j] ~ Poisson(exp(b0 * x2[j]))
    end
end
