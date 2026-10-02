# data: y g
begin
    mu_alpha ~ Normal(0, 10)
    s ~ Exponential(1)
    c ~ varying_coefs_centered(g)
    mu = mu_alpha .+ c[g]
    y .~ Normal.(mu, s)
end
