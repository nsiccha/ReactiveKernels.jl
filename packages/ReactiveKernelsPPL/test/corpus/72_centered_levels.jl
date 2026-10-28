# data: y g
begin
    mu_alpha ~ Normal(0, 10)
    sigma_alpha ~ Exponential(1)
    s ~ Exponential(1)
    c[levels(g)] .~ Normal.(mu_alpha, sigma_alpha)
    mu = c[g]
    y .~ Normal.(mu, s)
end
