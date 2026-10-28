# data: y
begin
    a ~ Normal(0, 1)
    phi1 ~ Normal(0, 0.5)
    phi2 ~ Normal(0, 0.5)
    s ~ Exponential(1)
    sigma ~ Exponential(1)
    @scan begin
        h[1] ~ Normal(0, 1)
        h[2] ~ Normal(0, 1)
        level[1] = 0.0
        level[2] = h[2]
        for t in 3:T
            eps ~ Normal(0, 1)
            h[t] = phi1 * h[t - 1] + phi2 * h[t - 2] + s * eps
            level[t] = level[t - 1] + h[t]
        end
    end
    mu = a .+ level
    y .~ Normal.(mu, sigma)
end
