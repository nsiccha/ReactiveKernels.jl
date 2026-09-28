# data: y x z
begin
    eta = a .+ b .* x
    hu = c .+ d .* z
    y .~ NegativeBinomial.(exp.(eta), logistic.(hu))
end
