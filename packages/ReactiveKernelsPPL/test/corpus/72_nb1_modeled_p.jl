# data: y x z
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    c ~ Normal(0, 1)
    d ~ Normal(0, 1)
    eta = a .+ b .* x
    hu = c .+ d .* z
    y .~ NegativeBinomial.(exp.(eta), logistic.(hu))
end
