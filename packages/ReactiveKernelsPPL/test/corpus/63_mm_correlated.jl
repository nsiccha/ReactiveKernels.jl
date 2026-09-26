# data: y x g1 g2 w1 w2
begin
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    d ~ varying_draws(mm(g1, g2; weights = (w1, w2)), [1, x])
    r ~ varying_slice(d, 1:2)
    mu = a .+ r
    y .~ Normal.(mu, sigma)
end
