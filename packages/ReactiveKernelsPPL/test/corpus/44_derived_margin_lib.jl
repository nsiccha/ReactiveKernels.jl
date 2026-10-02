# data: y x z g
begin
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    w = x .* z
    r ~ varying_coefs_correlated(g, 2)
    mu = a .+ r[g, 1] .+ w .* r[g, 2]
    y .~ Normal.(mu, sigma)
end
