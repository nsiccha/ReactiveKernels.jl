# data: y x
begin
    mu = a .+ b .* x
    y .~ Normal.(mu, s)
    a ~ Cauchy(0, 1)
    b ~ Flat()
    s ~ Exponential(1)
end
