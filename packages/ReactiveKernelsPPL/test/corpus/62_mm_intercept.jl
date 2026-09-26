# data: y g1 g2
begin
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    d ~ varying_draws(mm(g1, g2), [1])
    r ~ varying_slice(d, 1)
    mu = a .+ r
    y .~ Normal.(mu, sigma)
end
