# data: y x
begin
    eta2 = a2 .+ b2 .* x
    eta3 = a3 .+ b3 .* x
    y .~ CategoricalLogit.(eta2, eta3)
end
