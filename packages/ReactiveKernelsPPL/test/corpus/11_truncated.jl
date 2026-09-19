# data: y x
begin
    mu = a .+ b .* x
    y .~ truncated.(Normal.(mu, s), 0, 10)
    s ~ Exponential(1)
end
