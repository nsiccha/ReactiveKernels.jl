# data: y x g b
begin
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    r ~ varying_stratified_correlated(g, b, 2)
    mu = a .+ r[:, 1] .+ x .* r[:, 2]
    y .~ Normal.(mu, sigma)
end
