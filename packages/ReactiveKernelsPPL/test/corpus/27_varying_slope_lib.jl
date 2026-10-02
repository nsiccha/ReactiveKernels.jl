# data: y x g
begin
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    r ~ varying_coefs(g)
    mu = a .+ x .* r[g]
    y .~ Normal.(mu, sigma)
end
