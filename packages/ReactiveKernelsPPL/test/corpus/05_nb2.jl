# data: y x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    phi ~ Exponential(1.0)
    eta = a .+ b .* x
    y .~ NegativeBinomial2.(exp.(eta), phi)
end
