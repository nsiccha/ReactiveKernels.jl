# data: y x det
begin
    mu = a .+ b .* x
    pick = ifelse.(det .== 1, mu, -30.0)
    y .~ Bernoulli.(logistic.(pick))
end
