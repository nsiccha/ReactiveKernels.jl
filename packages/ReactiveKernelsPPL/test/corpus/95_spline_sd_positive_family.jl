# data: y x
# Smoothing-sd hyper prior from a positive-support family: the sd vector
# keeps the family's own support (no Stan-kernel half override).
begin
    spline_basis(:s_x, x; k = 4, sd = LogNormal(0, 1))
    a ~ Normal(0, 5)
    sigma ~ Exponential(1)
    mu = a .+ spline(:s_x)
    y .~ Normal.(mu, sigma)
end
