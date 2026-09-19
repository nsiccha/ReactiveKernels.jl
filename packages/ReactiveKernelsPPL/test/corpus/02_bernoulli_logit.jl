# data: y x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 2)
    eta = a .+ b .* x
    y .~ Bernoulli.(logistic.(eta))
end
