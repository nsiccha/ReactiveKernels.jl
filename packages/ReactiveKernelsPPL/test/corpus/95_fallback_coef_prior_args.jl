# data: y x x1 x2 g
begin
    s ~ HalfNormal(1)
    m ~ Normal(0, 1)
    a ~ Normal(m, 5)
    b ~ Normal(0, s)
    b1 ~ Normal(0, 1 / sqrt(var(x1)))
    t ~ Uniform(0, 2)
    c[levels(g)] .~ Normal.(0, 2 * s)
    sigma ~ Exponential(1)
    mu = a .+ b .* x .+ b1 .* x1 .- t .* x2 .+ c[g]
    y .~ Normal.(mu, sigma)
end
