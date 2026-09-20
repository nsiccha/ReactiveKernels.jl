# data: y x1 x2
begin
    R2 ~ Beta(1.0, 1.0)
    phi ~ Dirichlet([1.0, 1.0])
    mu = a .+ b1 .* x1 .+ b2 .* x2
    r2d2(mu, R2, phi)
    sigma ~ Exponential(1.0)
    y .~ Normal.(mu, sigma)
end
