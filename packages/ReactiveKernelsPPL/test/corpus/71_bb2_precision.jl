# data: y x z n
begin
    mu = a .+ b .* x
    hup = c .+ d .* z
    y .~ BetaBinomial2.(n, logistic.(mu), exp.(hup))
end
