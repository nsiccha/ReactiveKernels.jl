# data: y g
begin
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    r ~ varying_effect(g, [1]; levels=["c", "a", "b", "d"])
    mu = a .+ r
    y .~ Normal.(mu, sigma)
end
