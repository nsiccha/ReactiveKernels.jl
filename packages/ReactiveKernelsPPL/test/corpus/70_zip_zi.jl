# data: y x z
begin
    eta = a .+ b .* x
    zeta = d .+ e .* z
    y .~ ZeroInflatedPoisson.(exp.(eta), logistic.(zeta))
end
