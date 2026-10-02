# data: y x1 x2
begin
    b[axes(X, 2)] .~ Normal.([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    sigma ~ Exponential(1)
    X = hcat(ones(length(x1)), x1, x2)
    mu = X * b
    y .~ Normal.(mu, sigma)
end
