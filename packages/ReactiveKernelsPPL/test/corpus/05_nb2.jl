# data: y x
begin
    phi ~ Exponential(1.0)
    eta = a .+ b .* x
    y .~ NegativeBinomial2.(exp.(eta), phi)
end
