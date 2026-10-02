# data: y x1 x2
begin
    a ~ Normal(0, 1)
    X = hcat(x1, x2)
    b ~ r2d2_coefs(X, [1.0, 1.0])
    mu = a .+ X * b
    sigma ~ Exponential(1.0)
    y .~ Normal.(mu, sigma)
end
