# data: y x
begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    p ~ Beta(2.0, 2.0)
    eta = a .+ b .* x
    y .~ NegativeBinomial.(exp.(eta), p)
end
