# data: y x z n
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    c ~ Normal(0, 1)
    d ~ Normal(0, 1)
    mu = a .+ b .* x
    hup = c .+ d .* z
    y .~ BetaBinomial2.(n, logistic.(mu), exp.(hup))
end
