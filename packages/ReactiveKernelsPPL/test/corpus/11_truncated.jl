# data: y x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    mu = a .+ b .* x
    y .~ truncated.(Normal.(mu, s), 0, 10)
    s ~ Exponential(1)
end
