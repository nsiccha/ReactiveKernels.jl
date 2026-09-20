# data: y x z
begin
    eta = a .+ b .* x
    s = c .+ d .* z
    y .~ Gamma.(exp.(s), exp.(eta) ./ exp.(s))
end
