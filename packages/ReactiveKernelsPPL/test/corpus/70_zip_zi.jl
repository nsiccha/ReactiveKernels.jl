# data: y x z
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    d ~ Normal(0, 1)
    e ~ Normal(0, 1)
    eta = a .+ b .* x
    zeta = d .+ e .* z
    y .~ ZeroInflatedPoisson.(exp.(eta), logistic.(zeta))
end
