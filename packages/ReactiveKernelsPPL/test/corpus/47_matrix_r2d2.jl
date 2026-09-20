# data: y x1 x2
begin
    R2 ~ Beta(1.0, 1.0)
    phi ~ Dirichlet([1.0, 1.0])
    b[axes(X, 2)] .~ Normal.(0, 2)
    X = hcat(1, x1, x2)
    mu = X * b
    r2d2(mu, R2, phi)
    sigma ~ Exponential(1.0)
    y .~ Normal.(mu, sigma)
end
