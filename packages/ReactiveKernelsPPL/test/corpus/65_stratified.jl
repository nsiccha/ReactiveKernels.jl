# data: y x g b
begin
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    d ~ varying_draws(gr(g; by = b), [1, x])
    r ~ varying_slice(d, 1:2)
    mu = a .+ r
    y .~ Normal.(mu, sigma)
end
