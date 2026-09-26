# data: y x
begin
    p ~ Beta(2.0, 2.0)
    eta = a .+ b .* x
    y .~ NegativeBinomial.(exp.(eta), p)
end
