# data: y x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    s ~ Exponential(1)
    mu = a .+ b .* x
    y .~ truncated.(StudentT.(3.0, mu, s), 0, 10)
end
