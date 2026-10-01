# data: y x x1
begin
    s ~ HalfNormal(1)
    a ~ Normal(0, 5)
    z ~ Normal(0, 1)
    lam ~ HalfCauchy(1)
    b = s * z
    sigma ~ Exponential(1)
    mu = a .+ b .* x .+ x1 .* (z * lam)
    y .~ Normal.(mu, sigma)
end
