# data: y x
begin
    a ~ Normal(0, 1)
    k ~ Ordered(Normal(0, 2), 3)
    s = exp(k[3] - k[1])
    mu = a .+ k[2] .* x
    y .~ Normal.(mu, s)
end
