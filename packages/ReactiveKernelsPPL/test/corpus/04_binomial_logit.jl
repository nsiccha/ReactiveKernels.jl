# data: y x n
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    mu = a .+ b .* x
    y .~ Binomial.(n, logistic.(mu))
end
