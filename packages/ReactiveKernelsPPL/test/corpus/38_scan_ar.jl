# data: y
begin
    a ~ Normal(0, 1)
    beta_ar ~ Normal(0, 2)
    phi_raw ~ Normal(0, 1)
    sigma ~ Exponential(1)
    @scan begin
        u[1] ~ Normal(0, 1)
        for t in 2:T
            eps ~ Normal(0, 1)
            u[t] = phi * u[t - 1] + eps
        end
    end
    phi = tanh(phi_raw)
    mu = a .+ beta_ar .* u
    y .~ Normal.(mu, sigma)
end
