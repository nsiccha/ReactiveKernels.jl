# data: y x
begin
    a ~ Normal(0, 1)
    mu = a .+ b .* x
    y .~ Normal.(mu, s)
    b ~ Normal(0, 1)
    s ~ Flat(; lower=0)
    u ~ Flat(; lower=0, upper=100)
end
