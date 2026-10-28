# data: y x x1
begin
    a ~ Normal(0, 5)
    b ~ HalfNormal(2)
    c ~ Normal(0, 1)
    d ~ Normal(c, 1)
    sigma ~ Exponential(1)
    mu = a .+ b .* x .+ c .* x1 .+ d .* x .+ c .* x
    y .~ Normal.(mu, sigma)
end
