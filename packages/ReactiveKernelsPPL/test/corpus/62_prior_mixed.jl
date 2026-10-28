# data: y x
begin
    mu = a .+ b .* x
    y .~ Normal.(mu, s)
    a ~ StudentT(4, 0, 2)
    b ~ Laplace(0, 1)
    s ~ Exponential(1)
end
