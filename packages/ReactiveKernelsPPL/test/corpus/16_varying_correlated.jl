# data: y x g
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    sigma ~ Exponential(1)
    r ~ varying_effect(g, [1, x]; eta = 1.0)
    mu = a .+ b .* x .+ r
    y .~ Normal.(mu, sigma)
end
