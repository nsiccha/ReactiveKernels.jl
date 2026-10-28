# data: y age
begin
    Linf ~ Normal(2.0, 1.0)
    kk ~ Normal(0.0, 1.0)
    t0 ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    mu = Linf .* (1 .- exp.(-kk .* (age .- t0)))
    y .~ Normal.(mu, sigma)
end
