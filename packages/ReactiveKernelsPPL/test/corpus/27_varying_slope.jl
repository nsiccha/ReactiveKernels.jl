# data: y x g
begin
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    r ~ varying_effect(g, [x])
    mu = a .+ r
    y .~ Normal.(mu, sigma)
end
