# data: y x z
begin
    eta = a .+ b .* x
    phi = c .+ d .* z
    y .~ NegativeBinomial2.(exp.(eta), exp.(phi))
end
