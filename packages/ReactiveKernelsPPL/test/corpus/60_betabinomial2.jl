# data: c x n
begin
    phi ~ Gamma(2.0, 0.1)
    mu = a .+ b .* x
    c .~ BetaBinomial2.(n, logistic.(mu), phi)
end
