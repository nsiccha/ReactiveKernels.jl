# data: y x lo hi
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    mu = a .+ b .* x
    y .~ censored.(Normal.(mu, s), lo, hi)
    s ~ Exponential(1)
end
