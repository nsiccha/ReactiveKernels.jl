# data: y x det
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    mu = a .+ b .* x
    pick = ifelse.(det .== 1, mu, -30.0)
    y .~ Bernoulli.(logistic.(pick))
end
