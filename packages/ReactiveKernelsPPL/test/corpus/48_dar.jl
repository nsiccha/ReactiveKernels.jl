# data: y
begin
    a ~ Normal(0, 1)
    beta ~ truncated(Normal(0.5, 0.2), 0, 1)
    sigmad ~ HalfNormal(0.2)
    sigma ~ Exponential(1)
    mu = a .+ dar(beta, sigmad)
    y .~ Normal.(mu, sigma)
end
