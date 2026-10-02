# data: y x x1 x2 g
begin
    a ~ Normal(0, 5)
    b ~ Uniform(0, 2)
    s ~ HalfNormal(1)
    c[levels(g)] .~ Normal.(0.5, s)
    X = hcat(x1, x2)
    w[axes(X, 2)] .~ Normal.([1.0, -1.0], [1.0, 2.0])
    sigma ~ Exponential(1)
    mu = a .- b .* x .- c[g] .- X * w
    y .~ Normal.(mu, sigma)
end
