# data: y x n
begin
    mu = a .+ b .* x
    y .~ Binomial.(n, logistic.(mu))
end
