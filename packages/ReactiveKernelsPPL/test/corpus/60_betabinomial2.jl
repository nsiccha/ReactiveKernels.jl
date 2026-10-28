# data: c x n
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    phi ~ Gamma(2.0, 0.1)
    mu = a .+ b .* x
    c .~ BetaBinomial2.(n, logistic.(mu), phi)
end
