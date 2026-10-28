# data: y x
begin
    a ~ Normal(0, 1)
    mu = a .+ b .* x
    y .~ Normal.(mu, s)
    b ~ Normal(0, 1)
    s ~ HalfNormal(2)
    t ~ HalfCauchy(5)
end
