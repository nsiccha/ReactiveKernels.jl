# data: y x g
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    sigma ~ Exponential(1)
    r ~ varying_coefs_correlated(g, 2)
    mu = a .+ b .* x .+ r[g, 1] .+ x .* r[g, 2]
    y .~ Normal.(mu, sigma)
end
