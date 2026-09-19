# data: y x z
begin
    spline_basis(:t2_xz, x, z)
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    mu = a .+ spline(:t2_xz)
    y .~ Normal.(mu, sigma)
end
