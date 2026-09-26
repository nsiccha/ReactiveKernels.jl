# data: y x g1 g2
begin
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    r ~ varying_effect(mm(g1, g2), [x])
    mu = a .+ r
    y .~ Normal.(mu, sigma)
end
