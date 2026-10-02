# data: y x1 x2
begin
    a ~ Normal(0, 1)
    X = hcat(x1, x2)
    b ~ horseshoe_coefs(X)
    mu = a .+ X * b
    sigma ~ Exponential(1.0)
    y .~ Normal.(mu, sigma)
end
