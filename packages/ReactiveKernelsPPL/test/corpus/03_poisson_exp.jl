# data: y x
begin
    eta = a .+ b .* x
    y .~ Poisson.(exp.(eta))
end
