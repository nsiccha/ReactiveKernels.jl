# data: y x
begin
    spline_basis(:s_x, x; k = 4)
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    mu = a .+ spline(:s_x)
    y .~ Normal.(mu, sigma)
end
