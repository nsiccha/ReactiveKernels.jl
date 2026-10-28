# data: y x
begin
    a2 ~ Normal(0, 1)
    b2 ~ Normal(0, 1)
    a3 ~ Normal(0, 1)
    b3 ~ Normal(0, 1)
    eta2 = a2 .+ b2 .* x
    eta3 = a3 .+ b3 .* x
    y .~ CategoricalLogit.(eta2, eta3)
end
