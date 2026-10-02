# data: y g b
begin
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    r ~ varying_stratified(g, b)
    mu = a .+ r
    y .~ Normal.(mu, sigma)
end
