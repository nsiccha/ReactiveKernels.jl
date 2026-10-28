# data: y x lo hi
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    s ~ Exponential(1)
    mu = a .+ b .* x
    y .~ censored.(StudentT.(3.0, mu, s), lo, hi)
end
