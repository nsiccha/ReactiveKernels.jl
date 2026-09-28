# data: y x
begin
    a ~ Normal(0, 1)
    mu = a .+ b .* x
    y .~ Normal.(mu, s)
    b ~ Normal(0, 1)
    s ~ Normal(0, 2; lower=0)
    t ~ Cauchy(0, 5; lower=0)
end
